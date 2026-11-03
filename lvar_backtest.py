"""
5-MINUTE ENTRY ON PREVIOUS-DAY LEVELS - hypothesis fixed IN ADVANCE, nothing tuned (read-only research).
Idea (the owner's): mark support/resistance from the PREVIOUS UTC day, then enter on a 5-minute rejection after price touches a level.
  Levels (previous completed UTC day, 5m bars, same value-area algorithm as the bot): PDH, PDL, VAH, VAL, POC.
  SHORT: a 5m bar pierces PDH or VAH (high >= level), closes back below it, is bearish, closes in the lower 40% of its range,
         volume >= 2.0 x mean of the previous 20 bars. LONG: mirrored with PDL / VAL.
  Session 07:00-20:00 UTC. Entry = next 5m open. Stop = rejection-bar extreme -/+ 0.4 x ATR14(5m); stop distance 0.4%-3% of price.
  Target = 2R; the trade is SKIPPED if any previous-day level (PDH, PDL, VAH, VAL, POC) lies between entry and the target (S/R clearance).
  Time stop 48 bars (4h). One position per coin. SL checked before TP inside a bar; gaps fill at the open.
  Variant A = all signals. Variant B = only signals aligned with the 4H trend (EMA20/EMA50 of the last COMPLETED 4H bar).
Costs (futures-style): 0.15% round trip + funding 0.01% per 8h; stress 0.30%.
Pass rule (fixed before the run): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
Biases: today's liquid coin list, spot proxy for the perpetual, 1 year of data (5m history is heavy), no order-book impact.
Output file names are kept so the existing workflow commits them.
"""
import csv
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import crypto_signal_bot as M

DAYS = 365
COST_RT = 0.0015
COST_STRESS = 0.0030
FUNDING_8H = 0.0001
ATR_N = 14
VOL_MULT = 2.0
CLOSE_FRAC = 0.40
SL_ATR = 0.4
RR = 2.0
MIN_RISK, MAX_RISK = 0.004, 0.03
MAX_HOLD = 48
SESSION = (7, 20)
BARS_PER_DAY = 288
MIN_DAY_BARS = 250
SKIP = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "USDD", "BUSD", "EURT", "XAUT", "PAXG"}


def fetch_5m(pair: str, start: int, end: int) -> pd.DataFrame:
    rows, cur_end = [], end
    while cur_end > start:
        cur_start = max(start, cur_end - 1400 * 300)
        payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/candles",
                                  {"symbol": pair, "type": "5min", "startAt": cur_start, "endAt": cur_end})
        if isinstance(payload, dict) and payload.get("code") == "200000" and isinstance(payload.get("data"), list):
            for it in payload["data"]:
                try:
                    rows.append({"time": int(it[0]), "open": float(it[1]), "close": float(it[2]),
                                 "high": float(it[3]), "low": float(it[4]), "volume": float(it[5])})
                except (TypeError, ValueError, IndexError):
                    continue
        cur_end = cur_start
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df = df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.volume >= 0)]
    df = df[df.time + 300 <= int(time.time()) - 30]
    return df.reset_index(drop=True)


def atr(df: pd.DataFrame, n: int = ATR_N) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def day_levels(df: pd.DataFrame) -> dict:
    """UTC day index -> (pdh, pdl, vah, val, poc) from the PREVIOUS completed day (needs >= 250 of 288 bars, both ends of the day present)."""
    out = {}
    day = (df["time"] // 86400).to_numpy()
    for d in np.unique(day):
        prev = df[day == d - 1]
        if len(prev) < MIN_DAY_BARS:
            continue
        if int(prev.time.min()) > (d - 1) * 86400 + 3600 or int(prev.time.max()) < d * 86400 - 3600 - 300:
            continue
        vp = M.calculate_volume_profile(prev)
        if vp["vah"] > vp["poc"] > vp["val"] > 0:
            out[int(d)] = (float(prev.high.max()), float(prev.low.min()), vp["vah"], vp["val"], vp["poc"])
    return out


def h4_trend(df: pd.DataFrame) -> np.ndarray:
    """+1 / -1 / 0 per 5m bar from the last COMPLETED 4H bar."""
    b = df["time"] // 14400
    cl = df.groupby(b)["close"].last()
    e20, e50 = cl.ewm(span=20, adjust=False, min_periods=20).mean(), cl.ewm(span=50, adjust=False, min_periods=50).mean()
    st = pd.Series(0, index=cl.index)
    st[(e20 > e50) & (cl > e50)] = 1
    st[(e20 < e50) & (cl < e50)] = -1
    return b.map(st.shift(1).fillna(0)).to_numpy()


def simulate(o, h, l, c, i, side, stop, target, max_hold):
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
            if h[k] >= target:
                return (target - entry) / risk, k, "TP", entry
        else:
            if o[k] >= stop:
                return (entry - o[k]) / risk, k, "SL", entry
            if h[k] >= stop:
                return -1.0, k, "SL", entry
            if l[k] <= target:
                return (entry - target) / risk, k, "TP", entry
        if k == i + max_hold:
            return side * (c[k] - entry) / risk, k, "TIME", entry
    return None


def finish(res, risk_pct, hold_bars):
    r = res[0]
    fund = FUNDING_8H * hold_bars * 5.0 / 60.0 / 8.0
    return r, r - (COST_RT + fund) / risk_pct, r - (COST_STRESS + fund) / risk_pct


def run_coin(pair: str, rank: int, df: pd.DataFrame) -> list:
    if len(df) < 20000:
        return []
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = df["time"].to_numpy()
    a = atr(df)
    lv = day_levels(df)
    day = t // 86400
    lev = np.full((len(df), 5), np.nan)
    for d, vals in lv.items():
        lev[day == d] = vals
    avg20 = pd.Series(v).shift(1).rolling(20).mean().to_numpy()
    hour = (t % 86400) // 3600
    rng_ = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = (c - l) / rng_
    trend = h4_trend(df)
    in_session = (hour >= SESSION[0]) & (hour < SESSION[1])
    pdh, pdl, vah, val, poc = (lev[:, k] for k in range(5))
    ok = in_session & np.isfinite(pdh) & np.isfinite(a) & (rng_ > 0) & np.isfinite(avg20) & (avg20 > 0) & (v >= VOL_MULT * avg20)
    out, busy = [], -1
    for i in np.flatnonzero(ok):
        if i <= busy or i + 2 >= len(df):
            continue
        side, level = 0, None
        res_levels = [x for x in (pdh[i], vah[i]) if h[i] >= x and c[i] < x]
        sup_levels = [x for x in (pdl[i], val[i]) if l[i] <= x and c[i] > x]
        if res_levels and c[i] < o[i] and pos[i] <= CLOSE_FRAC:
            side, level = -1, max(res_levels)
        elif sup_levels and c[i] > o[i] and pos[i] >= 1 - CLOSE_FRAC:
            side, level = 1, min(sup_levels)
        if side == 0:
            continue
        entry = o[i + 1]
        stop = (h[i] + SL_ATR * a[i]) if side == -1 else (l[i] - SL_ATR * a[i])
        risk = abs(entry - stop)
        risk_pct = risk / entry
        if not (MIN_RISK <= risk_pct <= MAX_RISK) or (side == 1 and stop >= entry) or (side == -1 and stop <= entry):
            continue
        target = entry + side * RR * risk
        if target <= 0:
            continue
        lo_, hi_ = min(entry, target), max(entry, target)
        if any(lo_ < x < hi_ for x in (pdh[i], pdl[i], vah[i], val[i], poc[i])):
            continue                      # S/R clearance: a previous-day level sits between entry and target
        res = simulate(o, h, l, c, i, side, stop, target, MAX_HOLD)
        if res is None:
            continue
        rg, rn, rs = finish(res, risk_pct, res[1] - i)
        busy = res[1]
        out.append({"pair": pair, "side": "LONG" if side == 1 else "SHORT", "time": int(t[i]) + 300, "reason": res[2],
                    "bars": res[1] - i, "risk_pct": risk_pct, "r_gross": rg, "r_net": rn, "r_stress": rs,
                    "with_trend": bool(trend[i] == side)})
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


def block(title, rows, mid_t):
    L = ["=" * 60, title]
    if not rows:
        return L + ["NO TRADES", ""]
    for k in ("r_gross", "r_net", "r_stress"):
        L.append(fmt(f"  {k}", [x[k] for x in rows]))
    a = [x["r_net"] for x in rows if x["time"] < mid_t]
    b = [x["r_net"] for x in rows if x["time"] >= mid_t]
    L += [fmt("  first half (net)", a), fmt("  second half (net)", b)]
    for side in ("LONG", "SHORT"):
        L.append(fmt(f"  {side} (net)", [x["r_net"] for x in rows if x["side"] == side]))
    months = sorted({datetime.fromtimestamp(x["time"], timezone.utc).strftime("%Y-%m") for x in rows})
    pos_m = sum(1 for m in months if sum(x["r_net"] for x in rows if datetime.fromtimestamp(x["time"], timezone.utc).strftime("%Y-%m") == m) > 0)
    L.append(f"  profitable months (net): {pos_m} of {len(months)}")
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
    return L


def report(trades, meta):
    L = ["5-MINUTE ENTRY ON PREVIOUS-DAY LEVELS (hypothesis fixed in advance; nothing tuned)",
         f"period: {meta['start']} -> {meta['end']}  coins: {meta['coins']}  (spot 5m as perpetual proxy; costs 0.15% round trip + funding; stress 0.30%)",
         "Pass rule fixed before the run: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.", ""]
    ts = sorted(trades, key=lambda x: x["time"])
    mid_t = ts[len(ts) // 2]["time"] if ts else 0
    L += block("VARIANT A: all signals", ts, mid_t)
    L += block("VARIANT B: only with the 4H trend", [x for x in ts if x["with_trend"]], mid_t)
    return "\n".join(L)


def main():
    end = int(time.time()) // 300 * 300
    start = end - DAYS * 86400
    coins = [c for c in M.get_top_coins(M.SCAN_TOP_N_COINS) if c["symbol"] not in SKIP]
    print("universe:", len(coins), [c["symbol"] for c in coins])
    trades, have = [], 0
    for rank, c in enumerate(coins):
        df = fetch_5m(c["pair"], start, end)
        got = run_coin(c["pair"], rank, df)
        print(f"{c['pair']}: 5m bars {len(df)} trades {len(got)}", flush=True)
        if len(df) >= 20000:
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
        cols = ["pair", "side", "time", "reason", "bars", "risk_pct", "r_gross", "r_net", "with_trend"]
        w.writerow(cols)
        for x in trades:
            w.writerow([x[k] for k in cols])


if __name__ == "__main__":
    main()
