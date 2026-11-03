"""
ORDER BLOCK (15m) - hypothesis fixed IN ADVANCE, nothing tuned (read-only research).
  Bullish OB: a bearish candle i, followed within 3 bars by a bar j that (1) closes above the most recent CONFIRMED swing high
    (pivot of 3 bars each side, known only 3 bars after it forms) and (2) displacement: close[j] - open[i+1] >= 1.5 x ATR14.
    Zone = high/low of candle i. Bearish OB is the mirror. Zone is active from bar j+1 for 96 bars (24h).
  Entry = first retest: price trades back to the zone edge (long: low <= zone top). Fill at the zone edge (or the open if it gaps inside).
    Zone is void if a bar closes beyond the far edge before the retest, or the retest bar opens beyond the far edge.
  Stop = far edge -/+ 0.2 x ATR14; risk 0.3%-3%. Target 2R. If the retest bar also hits the stop it is a loss (conservative).
  Session for the retest: 07:00-20:00 UTC. Time stop 48 bars (12h). One position per coin. SL before TP; gaps fill at the open.
  Variant A = all. Variant B = only with the 4H trend (EMA20/EMA50 of the last COMPLETED 4H bar).
Costs: 0.15% round trip + funding 0.01% per 8h; stress 0.30%.
Pass rule (fixed before the run): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
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
PIVOT = 3
DISP_ATR = 1.5
SL_ATR = 0.2
RR = 2.0
MIN_RISK, MAX_RISK = 0.003, 0.03
MAX_HOLD = 48
ZONE_LIFE = 96
SESSION = (7, 20)
BAR_MIN = 15
SKIP = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "USDD", "BUSD", "EURT", "XAUT", "PAXG"}


def fetch_15m(pair, start, end):
    rows, cur_end = [], end
    while cur_end > start:
        cur_start = max(start, cur_end - 1400 * 900)
        payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/candles",
                                  {"symbol": pair, "type": "15min", "startAt": cur_start, "endAt": cur_end})
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
    df = df[df.time + 900 <= int(time.time()) - 30]
    return df.reset_index(drop=True)


def atr(df, n=ATR_N):
    pc = df["close"].shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def h4_trend(df):
    b = df["time"] // 14400
    cl = df.groupby(b)["close"].last()
    e20, e50 = cl.ewm(span=20, adjust=False, min_periods=20).mean(), cl.ewm(span=50, adjust=False, min_periods=50).mean()
    st = pd.Series(0, index=cl.index)
    st[(e20 > e50) & (cl > e50)] = 1
    st[(e20 < e50) & (cl < e50)] = -1
    return b.map(st.shift(1).fillna(0)).to_numpy()


def last_pivots(h, l):
    """most recent CONFIRMED pivot high / low as known at each bar (pivot at p is known from p+PIVOT)."""
    n = len(h)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    ch, cl_ = np.nan, np.nan
    for t in range(n):
        p = t - PIVOT
        if p >= PIVOT:
            if h[p] > h[p - PIVOT:p].max() and h[p] >= h[p + 1:p + PIVOT + 1].max():
                ch = h[p]
            if l[p] < l[p - PIVOT:p].min() and l[p] <= l[p + 1:p + PIVOT + 1].min():
                cl_ = l[p]
        sh[t], sl[t] = ch, cl_
    return sh, sl


def sim(o, h, l, c, k, side, entry, stop, target):
    n = len(c)
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if (side == 1 and l[k] <= stop) or (side == -1 and h[k] >= stop):
        return -1.0, k, "SL"
    last = min(n - 1, k + MAX_HOLD)
    for m in range(k + 1, last + 1):
        if side == 1:
            if o[m] <= stop:
                return (o[m] - entry) / risk, m, "SL"
            if l[m] <= stop:
                return -1.0, m, "SL"
            if h[m] >= target:
                return (target - entry) / risk, m, "TP"
        else:
            if o[m] >= stop:
                return (entry - o[m]) / risk, m, "SL"
            if h[m] >= stop:
                return -1.0, m, "SL"
            if l[m] <= target:
                return (entry - target) / risk, m, "TP"
        if m == k + MAX_HOLD:
            return side * (c[m] - entry) / risk, m, "TIME"
    return None


def finish(r, risk_pct, hold_bars):
    fund = FUNDING_8H * hold_bars * BAR_MIN / 60.0 / 8.0
    return r, r - (COST_RT + fund) / risk_pct, r - (COST_STRESS + fund) / risk_pct


def find_zone(o, h, l, c, a, sh, sl, i):
    """returns (side, top, bottom, j) for an order block candidate at candle i, or None."""
    n = len(c)
    if i + 4 >= n or not np.isfinite(a[i]):
        return None
    for j in range(i + 1, i + 4):
        if c[i] < o[i] and np.isfinite(sh[i]) and c[j] > sh[i] and c[j] - o[i + 1] >= DISP_ATR * a[i]:
            return 1, h[i], l[i], j
        if c[i] > o[i] and np.isfinite(sl[i]) and c[j] < sl[i] and o[i + 1] - c[j] >= DISP_ATR * a[i]:
            return -1, h[i], l[i], j
    return None


def run_coin(pair, df):
    if len(df) < 8000:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy()
    a = atr(df)
    sh, sl = last_pivots(h, l)
    hour = (t % 86400) // 3600
    trend = h4_trend(df)
    out, busy = [], -1
    n = len(df)
    for i in range(60, n - 5):
        z = find_zone(o, h, l, c, a, sh, sl, i)
        if z is None:
            continue
        side, top, bot, j = z
        k_hit = None
        for k in range(j + 1, min(n - 1, j + ZONE_LIFE) + 1):
            if side == 1 and c[k] < bot or side == -1 and c[k] > top:
                break
            if (side == 1 and l[k] <= top) or (side == -1 and h[k] >= bot):
                k_hit = k
                break
        if k_hit is None or k_hit <= busy or not (SESSION[0] <= hour[k_hit] < SESSION[1]):
            continue
        k = k_hit
        if side == 1:
            if o[k] < bot:
                continue
            entry = min(top, o[k]); stop = bot - SL_ATR * a[i]
        else:
            if o[k] > top:
                continue
            entry = max(bot, o[k]); stop = top + SL_ATR * a[i]
        risk = abs(entry - stop)
        risk_pct = risk / entry
        if not (MIN_RISK <= risk_pct <= MAX_RISK) or (side == 1 and stop >= entry) or (side == -1 and stop <= entry):
            continue
        target = entry + side * RR * risk
        if target <= 0:
            continue
        res = sim(o, h, l, c, k, side, entry, stop, target)
        if res is None:
            continue
        rg, rn, rs = finish(res[0], risk_pct, res[1] - k)
        busy = res[1]
        out.append({"pair": pair, "side": "LONG" if side == 1 else "SHORT", "time": int(t[k]), "reason": res[2],
                    "bars": res[1] - k, "risk_pct": risk_pct, "r_gross": rg, "r_net": rn, "r_stress": rs,
                    "with_trend": bool(trend[k] == side)})
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
    L = ["ORDER BLOCK 15m (hypothesis fixed in advance; nothing tuned)",
         f"period: {meta['start']} -> {meta['end']}  coins: {meta['coins']}  (spot 15m as perpetual proxy; costs 0.15% round trip + funding; stress 0.30%)",
         "Pass rule fixed before the run: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.", ""]
    ts = sorted(trades, key=lambda x: x["time"])
    mid_t = ts[len(ts) // 2]["time"] if ts else 0
    L += block("VARIANT A: all signals", ts, mid_t)
    L += block("VARIANT B: only with the 4H trend", [x for x in ts if x["with_trend"]], mid_t)
    return "\n".join(L)


def main():
    end = int(time.time()) // 900 * 900
    start = end - DAYS * 86400
    coins = [c for c in M.get_top_coins(M.SCAN_TOP_N_COINS) if c["symbol"] not in SKIP]
    print("universe:", len(coins), [c["symbol"] for c in coins])
    trades, have = [], 0
    for rank, c in enumerate(coins):
        df = fetch_15m(c["pair"], start, end)
        got = run_coin(c["pair"], df)
        print(f"{c['pair']}: 15m bars {len(df)} trades {len(got)}", flush=True)
        if len(df) >= 8000:
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
