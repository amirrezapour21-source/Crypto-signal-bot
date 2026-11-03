"""
1-HOUR FUTURES-STYLE INTRADAY BACKTEST - two hypotheses fixed IN ADVANCE, nothing tuned (read-only research).
Prices: KuCoin spot 1h candles used as a proxy for the perpetual (price difference is tiny); costs are futures-style:
  taker 0.06% per side + 0.03% slippage per side = 0.15% round trip, plus funding 0.01% per 8h held (charged both ways).
  stress case: 0.30% round trip.
H1  "DONCHIAN-24 BREAKOUT" (both directions): signal at the close of a 1h bar that breaks the previous 24-bar high (long) / low (short);
    entry = next bar open; stop = entry -/+ 2 x ATR14; exit at the first of: stop (intrabar, gap-aware), close back through the midpoint of the
    prior-24-bar channel (exit at the next open), or 72 bars.
H2  "DAILY-TREND PULLBACK RESUMPTION" (both directions): daily trend from the LAST COMPLETED UTC day (EMA20 > EMA50 and close > EMA50 = up; mirrored = down).
    long when the 1h close crosses back above the 1h EMA20 in an up-trend (short mirrored); entry = next bar open; stop = 2 x ATR14; target = 3 x ATR14; 48 bars.
Common: one position per coin and hypothesis; stop distance must be 0.4%-8% of price; SL checked before TP inside a bar; same-bar gaps fill at the open.
Pass rule (written before running): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND BOTH halves of the period net-positive.
Biases: today's liquid coin list (survivorship), spot proxy for futures, no order-book impact. Output file names are kept so the existing workflow commits them.
"""
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import crypto_signal_bot as M

DAYS = max(int(os.getenv("BT_DAYS", "0") or 0), 1095)       # at least 3 years whatever the workflow input says
COST_RT = 0.0015
COST_STRESS = 0.0030
FUNDING_8H = 0.0001
ATR_N = 14
CH_N = 24
MIN_RISK, MAX_RISK = 0.004, 0.08
SKIP = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "USDD", "BUSD", "EURT", "XAUT", "PAXG"}


def fetch_1h(pair: str, start: int, end: int) -> pd.DataFrame:
    rows, cur_end = [], end
    while cur_end > start:
        cur_start = max(start, cur_end - 1400 * 3600)
        payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/candles",
                                  {"symbol": pair, "type": "1hour", "startAt": cur_start, "endAt": cur_end})
        if isinstance(payload, dict) and payload.get("code") == "200000" and isinstance(payload.get("data"), list):
            for it in payload["data"]:
                try:
                    rows.append({"time": int(it[0]), "open": float(it[1]), "close": float(it[2]),
                                 "high": float(it[3]), "low": float(it[4]), "volume": float(it[5])})
                except (TypeError, ValueError, IndexError):
                    continue
        cur_end = cur_start
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df = df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.volume >= 0)]
    df = df[df.time + 3600 <= int(time.time()) - 60]
    return df.reset_index(drop=True)


def atr(df: pd.DataFrame, n: int = ATR_N) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def simulate(o, h, l, c, i, side, stop, target, mid, max_hold):
    """Position opened at open[i+1]. Returns (r_gross, exit_index, reason, entry) or None if the data ends first."""
    n = len(c)
    e = i + 1
    if e >= n:
        return None
    entry = o[e]
    risk = abs(entry - stop)
    if risk <= 0 or (side == 1 and stop >= entry) or (side == -1 and stop <= entry):
        return None
    last = min(n - 1, i + max_hold)
    for k in range(e, last + 1):
        if side == 1:
            if o[k] <= stop:
                return (o[k] - entry) / risk, k, "SL", entry
            if l[k] <= stop:
                return -1.0, k, "SL", entry
            if target is not None and h[k] >= target:
                return (target - entry) / risk, k, "TP", entry
        else:
            if o[k] >= stop:
                return (entry - o[k]) / risk, k, "SL", entry
            if h[k] >= stop:
                return -1.0, k, "SL", entry
            if target is not None and l[k] <= target:
                return (entry - target) / risk, k, "TP", entry
        if mid is not None and k + 1 < n and np.isfinite(mid[k]) and ((side == 1 and c[k] < mid[k]) or (side == -1 and c[k] > mid[k])):
            return side * (o[k + 1] - entry) / risk, k + 1, "MID", entry
        if k == i + max_hold:
            return side * (c[k] - entry) / risk, k, "TIME", entry
    return None


def finish(trade, risk_pct, hold):
    r, k, reason, entry = trade
    costs = COST_RT + FUNDING_8H * hold / 8.0
    return r, r - costs / risk_pct, r - (COST_STRESS + FUNDING_8H * hold / 8.0) / risk_pct


def daily_trend(df: pd.DataFrame) -> np.ndarray:
    """+1 / -1 / 0 per 1h bar from the last COMPLETED UTC day."""
    day = df["time"] // 86400
    dc = df.groupby(day)["close"].last()
    ema20, ema50 = dc.ewm(span=20, adjust=False, min_periods=20).mean(), dc.ewm(span=50, adjust=False, min_periods=50).mean()
    st = pd.Series(0, index=dc.index)
    st[(ema20 > ema50) & (dc > ema50)] = 1
    st[(ema20 < ema50) & (dc < ema50)] = -1
    prev = st.shift(1).fillna(0)
    return day.map(prev).to_numpy()


def run_coin(pair: str, rank: int, df: pd.DataFrame) -> list:
    if len(df) < 1500:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy()
    a = atr(df)
    hh = df["high"].shift(1).rolling(CH_N).max().to_numpy()
    ll = df["low"].shift(1).rolling(CH_N).min().to_numpy()
    mid = (hh + ll) / 2.0
    e20 = df["close"].ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    tr = daily_trend(df)
    out = []
    busy = {"H1": -1, "H2": -1}
    for i in range(max(CH_N, 60), len(df) - 2):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        cands = []
        if i > busy["H1"] and np.isfinite(hh[i]):
            if c[i] > hh[i]:
                cands.append(("H1", 1, o[i + 1] - 2 * a[i], None, mid, 72))
            elif c[i] < ll[i]:
                cands.append(("H1", -1, o[i + 1] + 2 * a[i], None, mid, 72))
        if i > busy["H2"] and np.isfinite(e20[i]) and np.isfinite(e20[i - 1]):
            if tr[i] == 1 and c[i - 1] < e20[i - 1] and c[i] > e20[i]:
                cands.append(("H2", 1, o[i + 1] - 2 * a[i], o[i + 1] + 3 * a[i], None, 48))
            elif tr[i] == -1 and c[i - 1] > e20[i - 1] and c[i] < e20[i]:
                cands.append(("H2", -1, o[i + 1] + 2 * a[i], o[i + 1] - 3 * a[i], None, 48))
        for name, side, stop, target, midarr, hold in cands:
            risk_pct = abs(o[i + 1] - stop) / o[i + 1]
            if not (MIN_RISK <= risk_pct <= MAX_RISK):
                continue
            res = simulate(o, h, l, c, i, side, stop, target, midarr, hold)
            if res is None:
                continue
            rg, rn, rs = finish(res, risk_pct, res[1] - (i + 1) + 1)
            busy[name] = res[1]
            out.append({"pair": pair, "hyp": name, "side": "LONG" if side == 1 else "SHORT", "time": int(t[i]) + 3600,
                        "reason": res[2], "bars": res[1] - i, "risk_pct": risk_pct, "r_gross": rg, "r_net": rn, "r_stress": rs})
    return out


def stats(rs):
    a = np.asarray(rs, float)
    if len(a) == 0:
        return None
    wins, losses = a[a > 0].sum(), -a[a < 0].sum()
    cum = np.cumsum(a)
    return {"n": len(a), "win%": 100 * float((a > 0).mean()), "avgR": float(a.mean()), "totalR": float(a.sum()),
            "PF": float(wins / losses) if losses > 0 else float("inf"), "maxDD": float((np.maximum.accumulate(cum) - cum).max())}


def fmt(label, rs):
    s = stats(rs)
    if s is None:
        return f"{label:38s} n=0"
    return f"{label:38s} n={s['n']:5d} win%={s['win%']:5.1f} avgR={s['avgR']:+.3f} totalR={s['totalR']:+8.1f} PF={s['PF']:.2f} maxDD={s['maxDD']:.1f}"


def report(trades, meta):
    L = ["1H INTRADAY BACKTEST - two fixed hypotheses (nothing tuned)",
         f"period: {meta['start']} -> {meta['end']}  coins: {meta['coins']}  (spot 1h as perpetual proxy; costs 0.15% round trip + funding; stress 0.30%)",
         "Pass rule fixed before the run: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.", ""]
    ts = sorted(trades, key=lambda x: x["time"])
    mid_t = ts[len(ts) // 2]["time"] if ts else 0
    for hyp, title in (("H1", "H1 DONCHIAN-24 BREAKOUT"), ("H2", "H2 DAILY-TREND PULLBACK RESUMPTION")):
        rows = [x for x in ts if x["hyp"] == hyp]
        L.append("=" * 60); L.append(title)
        if not rows:
            L.append("NO TRADES"); continue
        for k in ("r_gross", "r_net", "r_stress"):
            L.append(fmt(f"  {k}", [x[k] for x in rows]))
        a = [x["r_net"] for x in rows if x["time"] < mid_t]; b = [x["r_net"] for x in rows if x["time"] >= mid_t]
        L.append(fmt("  first half (net)", a)); L.append(fmt("  second half (net)", b))
        for side in ("LONG", "SHORT"):
            L.append(fmt(f"  {side} (net)", [x["r_net"] for x in rows if x["side"] == side]))
        years = sorted({datetime.fromtimestamp(x["time"], timezone.utc).year for x in rows})
        for y in years:
            L.append(fmt(f"  year {y} (net)", [x["r_net"] for x in rows if datetime.fromtimestamp(x["time"], timezone.utc).year == y]))
        byc = {}
        for x in rows:
            byc[x["pair"]] = byc.get(x["pair"], 0.0) + x["r_net"]
        top = sorted(byc.items(), key=lambda kv: -kv[1])[:3]
        L.append("  top 3 coins by net R: " + ", ".join(f"{k} {v:+.0f}" for k, v in top) + f" | total {sum(byc.values()):+.0f}")
        res = {}
        for x in rows:
            res[x["reason"]] = res.get(x["reason"], 0) + 1
        L.append("  exits: " + ", ".join(f"{k}={v}" for k, v in sorted(res.items())))
        L.append(f"  median risk {100 * float(np.median([x['risk_pct'] for x in rows])):.2f}% | median bars held {int(np.median([x['bars'] for x in rows]))}")
        sn, sa, sb = stats([x["r_net"] for x in rows]), stats(a), stats(b)
        ok = bool(sn and sa and sb and sn["avgR"] >= 0.10 and sn["PF"] >= 1.2 and sn["n"] >= 300 and sa["avgR"] > 0 and sb["avgR"] > 0)
        L.append(f"  VERDICT (pre-registered rule): {'PASS - candidate for forward paper test' if ok else 'FAIL - rejected'}")
        L.append("")
    return "\n".join(L)


def main():
    end = int(time.time()) // 3600 * 3600
    start = end - DAYS * 86400
    coins = [c for c in M.get_top_coins(M.SCAN_TOP_N_COINS) if c["symbol"] not in SKIP]
    print("universe:", len(coins), [c["symbol"] for c in coins])
    trades, have = [], 0
    for rank, c in enumerate(coins):
        df = fetch_1h(c["pair"], start, end)
        got = run_coin(c["pair"], rank, df)
        print(f"{c['pair']}: 1h bars {len(df)} trades {len(got)}", flush=True)
        if len(df) >= 1500:
            have += 1
        trades += got
    meta = {"start": datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
            "end": datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d"), "coins": have}
    text = report(trades, meta) if trades else "NO TRADES"
    print(text)
    with open("lvar_backtest_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open("lvar_backtest_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "hyp", "side", "time", "reason", "bars", "risk_pct", "r_gross", "r_net"]
        w.writerow(cols)
        for x in trades:
            w.writerow([x[k] for k in cols])


if __name__ == "__main__":
    main()
