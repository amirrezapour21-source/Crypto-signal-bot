"""
DAILY MULTI-POSITION BREAKOUT PORTFOLIO - hypothesis fixed IN ADVANCE, nothing tuned (read-only research).
Universe: up to 25 liquid KuCoin USDT coins (top by 24h turnover today, with >= 1100 days of daily history). Spot 1D as perpetual proxy.
Regime (BTC, completed daily bar): UP if EMA20 > EMA50 and close > EMA50 -> longs only; DOWN if the mirror -> shorts only; else no trades.
Signal (coin, at the close of day t): close breaks the prior 20-day high (long) / low (short), close on the right side of its own EMA50,
  volume >= 1.0 x its prior 20-day average, and (Variant A) the signal day is not a shock day: true range <= 3 x ATR14 of the day before.
Entry = open of day t+1 (no lookahead). Stop = 2 x ATR14(t). Target = 3R. Time stop = close of the 15th day (entry day = day 1).
SL before TP inside one bar; SL gaps fill at the open; TP fills at the target. Risk must be 2%-15% of price. One position per coin.
Portfolio: at most 8 open positions; if more signals arrive the same day, the strongest breakout (distance beyond the level in ATR) wins.
Costs: 0.15% round trip + funding 0.01% per 8h (3 per day) charged to every trade as an assumption; stress 0.30%.
Pass rule (fixed before the run, Variant A only): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
Variant B (no shock filter) is a sensitivity check only and is not eligible for selection.
"""
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import crypto_signal_bot as M

DAYS = int(os.environ.get("BT_DAYS") or 1460)
MIN_HISTORY = 1100
N_COINS = 25
DON_N = 20
EMA_COIN = 50
ATR_N = 14
SL_ATR = 2.0
RR = 3.0
MAX_HOLD = 15
VOL_N = 20
VOL_MIN = 1.0
SHOCK_ATR = 3.0
MIN_RISK, MAX_RISK = 0.02, 0.15
MAX_POS = 8
COST_RT = 0.0015
COST_STRESS = 0.0030
FUNDING_8H = 0.0001
RISK_PER_TRADE = 0.005
BOOT = 2000
DAY = 86400
SKIP = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "USDD", "BUSD", "EURT", "XAUT", "PAXG"}


def fetch_daily(pair, start, end):
    rows, cur_end = [], end
    while cur_end > start:
        cur_start = max(start, cur_end - 1400 * DAY)
        payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/candles",
                                  {"symbol": pair, "type": "1day", "startAt": cur_start, "endAt": cur_end})
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
    df = df[df.time + DAY <= int(time.time()) - 30]
    return df.reset_index(drop=True)


def missing_days(df):
    if len(df) < 2:
        return 10 ** 6
    d = np.diff(df["time"].to_numpy())
    return int(((d // DAY) - 1).clip(min=0).sum())


def prep(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()
    ema = c.ewm(span=EMA_COIN, adjust=False, min_periods=EMA_COIN).mean()
    hh = h.rolling(DON_N).max().shift(1)
    ll = l.rolling(DON_N).min().shift(1)
    vr = v / v.rolling(VOL_N).mean().shift(1)
    return {k: x.to_numpy(float) for k, x in (("tr", tr), ("atr", atr), ("ema", ema), ("hh", hh), ("ll", ll), ("vr", vr))}


def btc_regime(df):
    c = df["close"]
    e20, e50 = c.ewm(span=20, adjust=False, min_periods=20).mean(), c.ewm(span=50, adjust=False, min_periods=50).mean()
    st = pd.Series(0, index=df.index)
    st[(e20 > e50) & (c > e50)] = 1
    st[(e20 < e50) & (c < e50)] = -1
    return dict(zip(df["time"].to_numpy(), st.to_numpy()))


def sim_daily(o, h, l, c, k, side, entry, stop, target):
    n = len(c)
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    end_bar = k + MAX_HOLD - 1
    for m in range(k, min(n - 1, end_bar) + 1):
        if side == 1:
            if m > k and o[m] <= stop:
                return (o[m] - entry) / risk, m, "SL"
            if l[m] <= stop:
                return -1.0, m, "SL"
            if h[m] >= target:
                return (target - entry) / risk, m, "TP"
        else:
            if m > k and o[m] >= stop:
                return (entry - o[m]) / risk, m, "SL"
            if h[m] >= stop:
                return -1.0, m, "SL"
            if l[m] <= target:
                return (entry - target) / risk, m, "TP"
        if m == end_bar:
            return side * (c[m] - entry) / risk, m, "TIME"
    return None


def finish(r, risk_pct, hold_days):
    fund = FUNDING_8H * 3.0 * hold_days
    return r, r - (COST_RT + fund) / risk_pct, r - (COST_STRESS + fund) / risk_pct


def candidates(pair, df, reg, use_shock):
    n = len(df)
    if n < 120:
        return []
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = df["time"].to_numpy()
    p = prep(df)
    out = []
    for i in range(max(DON_N, EMA_COIN, ATR_N, VOL_N) + 2, n - 1):
        a_prev, a = p["atr"][i - 1], p["atr"][i]
        if not (np.isfinite(a) and np.isfinite(a_prev) and np.isfinite(p["vr"][i]) and np.isfinite(p["ema"][i])):
            continue
        r = reg.get(t[i], 0)
        if r == 0 or p["vr"][i] < VOL_MIN:
            continue
        if use_shock and p["tr"][i] > SHOCK_ATR * a_prev:
            continue
        side = 0
        if r == 1 and np.isfinite(p["hh"][i]) and c[i] > p["hh"][i] and c[i] > p["ema"][i]:
            side = 1
        elif r == -1 and np.isfinite(p["ll"][i]) and c[i] < p["ll"][i] and c[i] < p["ema"][i]:
            side = -1
        if side == 0:
            continue
        k = i + 1
        if t[k] - t[i] != DAY:
            continue
        entry = o[k]
        stop = entry - side * SL_ATR * a
        risk_pct = abs(entry - stop) / entry
        if stop <= 0 or not (MIN_RISK <= risk_pct <= MAX_RISK):
            continue
        target = entry + side * RR * abs(entry - stop)
        if target <= 0:
            continue
        res = sim_daily(o, h, l, c, k, side, entry, stop, target)
        if res is None:
            continue
        days = res[1] - k + 1
        rg, rn, rs = finish(res[0], risk_pct, days)
        level = p["hh"][i] if side == 1 else p["ll"][i]
        out.append({"pair": pair, "side": "LONG" if side == 1 else "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[1]]),
                    "reason": res[2], "days": days, "risk_pct": risk_pct, "r_gross": rg, "r_net": rn, "r_stress": rs,
                    "strength": float(side * (c[i] - level) / a)})
    return out


def portfolio(cands):
    ordered = sorted(cands, key=lambda x: (x["entry_t"], -x["strength"], x["pair"]))
    open_pos, taken = [], []
    for x in ordered:
        open_pos = [q for q in open_pos if q[0] >= x["entry_t"]]
        if len(open_pos) >= MAX_POS or any(q[1] == x["pair"] for q in open_pos):
            continue
        taken.append(x)
        open_pos.append((x["exit_t"], x["pair"]))
    return taken


def timeline(rows):
    t0 = min(x["entry_t"] for x in rows)
    t1 = max(x["exit_t"] for x in rows)
    cnt = np.zeros((t1 - t0) // DAY + 1, int)
    for x in rows:
        cnt[(x["entry_t"] - t0) // DAY:(x["exit_t"] - t0) // DAY + 1] += 1
    entry_days = len({x["entry_t"] for x in rows})
    return cnt, entry_days


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


def month_of(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def year_of(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y")


def week_bootstrap(rows):
    wk = {}
    for x in rows:
        wk.setdefault(x["entry_t"] // (7 * DAY), []).append(x["r_net"])
    sums = np.array([sum(v) for v in wk.values()])
    cnts = np.array([len(v) for v in wk.values()])
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(sums), size=(BOOT, len(sums)))
    avg = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(np.percentile(avg, 2.5)), float(np.percentile(avg, 97.5)), float((avg > 0).mean()), len(sums)


def block(title, rows, eligible):
    L = ["=" * 60, title]
    if not rows:
        return L + ["NO TRADES", ""]
    rows = sorted(rows, key=lambda x: x["entry_t"])
    mid_t = rows[len(rows) // 2]["entry_t"]
    for k in ("r_gross", "r_net", "r_stress"):
        L.append(fmt(f"  {k}", [x[k] for x in rows]))
    a = [x["r_net"] for x in rows if x["entry_t"] < mid_t]
    b = [x["r_net"] for x in rows if x["entry_t"] >= mid_t]
    L += [fmt("  first half (net)", a), fmt("  second half (net)", b)]
    for side in ("LONG", "SHORT"):
        L.append(fmt(f"  {side} (net)", [x["r_net"] for x in rows if x["side"] == side]))
    for y in sorted({year_of(x["entry_t"]) for x in rows}):
        L.append(fmt(f"  year {y} (net)", [x["r_net"] for x in rows if year_of(x["entry_t"]) == y]))
    cnt, entry_days = timeline(rows)
    span = len(cnt)
    d = np.array([x["days"] for x in rows])
    L.append(f"  days open per trade: mean {d.mean():.1f} | median {int(np.median(d))} | 90th pct {int(np.percentile(d, 90))} | max {int(d.max())}")
    L.append(f"  trades per week: {len(rows) / (span / 7):.2f} | days with a new entry: {100 * entry_days / span:.0f}%")
    L.append(f"  open positions: mean {cnt.mean():.1f} | max {int(cnt.max())} | days with none open {100 * float((cnt == 0).mean()):.0f}% | days at cap {100 * float((cnt >= MAX_POS).mean()):.0f}%")
    exits = {}
    for x in rows:
        exits[x["reason"]] = exits.get(x["reason"], 0) + 1
    L.append("  exits: " + ", ".join(f"{k}={v}" for k, v in sorted(exits.items())))
    L.append(f"  median risk {100 * float(np.median([x['risk_pct'] for x in rows])):.1f}% of price")
    byx = sorted(rows, key=lambda x: x["exit_t"])
    cum, peak, peak_t, under = 0.0, 0.0, byx[0]["exit_t"], 0
    for x in byx:
        cum += x["r_net"]
        if cum >= peak:
            peak, peak_t = cum, x["exit_t"]
        under = max(under, (x["exit_t"] - peak_t) // DAY)
    L.append(f"  longest time below equity peak (net): {int(under)} days")
    mon = {}
    for x in byx:
        mon[month_of(x["exit_t"])] = mon.get(month_of(x["exit_t"]), 0.0) + x["r_net"]
    pos_m = sum(1 for v in mon.values() if v > 0)
    L.append(f"  profitable months (net, by exit): {pos_m} of {len(mon)} | worst month {min(mon.values()):+.1f}R | best month {max(mon.values()):+.1f}R")
    sn = stats([x["r_net"] for x in rows])
    L.append(f"  at {100 * RISK_PER_TRADE:.1f}% risk per trade (not compounded): total {sn['totalR'] * 100 * RISK_PER_TRADE:+.0f}% | max drawdown {sn['maxDD'] * 100 * RISK_PER_TRADE:.0f}%")
    byc = {}
    for x in rows:
        byc[x["pair"]] = byc.get(x["pair"], 0.0) + x["r_net"]
    top = sorted(byc.items(), key=lambda kv: -kv[1])
    top3 = {k for k, _ in top[:3]}
    rest = [x["r_net"] for x in rows if x["pair"] not in top3]
    L.append("  top 3 coins by net R: " + ", ".join(f"{k} {v:+.0f}" for k, v in top[:3]) + f" | total {sum(byc.values()):+.0f}")
    L.append(f"  avgR net without those 3 coins: {np.mean(rest):+.3f} (n={len(rest)})" if rest else "  without top 3 coins: no trades")
    wk = {}
    for x in rows:
        wk[x["entry_t"] // (7 * DAY)] = wk.get(x["entry_t"] // (7 * DAY), 0.0) + x["r_net"]
    wins = sorted((v for v in wk.values() if v > 0), reverse=True)
    if wins:
        L.append(f"  share of positive-week profit from the best 5 weeks: {100 * sum(wins[:5]) / sum(wins):.0f}%")
    lo, hi, pp, nw = week_bootstrap(rows)
    L.append(f"  weekly-block bootstrap of net avgR: 95% CI [{lo:+.3f}, {hi:+.3f}], P(avgR>0)={pp:.2f} ({nw} weeks)")
    sa, sb = stats(a), stats(b)
    ok = bool(sn and sa and sb and sn["avgR"] >= 0.10 and sn["PF"] >= 1.2 and sn["n"] >= 300 and sa["avgR"] > 0 and sb["avgR"] > 0)
    if eligible:
        L.append(f"  VERDICT (pre-registered rule): {'PASS - candidate for forward paper test' if ok else 'FAIL - rejected'}")
        if sn["n"] < 300:
            L.append("  note: fewer than 300 trades = insufficient sample")
    else:
        L.append("  (sensitivity only - not eligible for selection)")
    L.append("")
    return L


def report(a_all, a_cap, b_cap, meta):
    L = ["DAILY MULTI-POSITION BREAKOUT (hypothesis fixed in advance; nothing tuned)",
         f"period: {meta['start']} -> {meta['end']} | coins: {meta['coins']} | spot 1D as perpetual proxy",
         f"costs: 0.15% round trip + funding 0.01%/8h assumed on every trade (shorts too); stress 0.30%; max {MAX_POS} open positions",
         "Pass rule fixed before the run (Variant A): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Known biases (all favour the strategy): today's coin list, spot instead of perpetual data, funding is an assumption not measured data.",
         "News is not in the data; the shock-day filter (true range > 3 ATR) is only a price-based proxy.",
         f"signals before the position cap (Variant A): {len(a_all)} | taken: {len(a_cap)}", ""]
    L += block("VARIANT A: BTC regime + breakout + shock filter (the tested rule)", a_cap, True)
    L += block("VARIANT B: same without the shock-day filter", b_cap, False)
    return "\n".join(L)


def main():
    end = int(time.time()) // DAY * DAY
    start = end - DAYS * DAY
    btc = fetch_daily("BTC-USDT", start, end)
    if len(btc) < MIN_HISTORY:
        print("BTC history too short:", len(btc))
        raise SystemExit(1)
    reg = btc_regime(btc)
    coins = [c for c in M.get_top_coins(90) if c["symbol"] not in SKIP]
    data, skipped = {}, []
    for c in coins:
        if len(data) >= N_COINS:
            break
        df = btc if c["pair"] == "BTC-USDT" else fetch_daily(c["pair"], start, end)
        if len(df) < MIN_HISTORY or missing_days(df) > 3:
            skipped.append(f"{c['pair']}({len(df)} bars, {missing_days(df)} missing)")
            continue
        data[c["pair"]] = df
        print(f"{c['pair']}: daily bars {len(df)}", flush=True)
    print("skipped:", skipped)
    a_all, b_all = [], []
    for pair, df in data.items():
        a_all += candidates(pair, df, reg, True)
        b_all += candidates(pair, df, reg, False)
    a_cap, b_cap = portfolio(a_all), portfolio(b_all)
    meta = {"start": datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
            "end": datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d"), "coins": len(data)}
    text = report(a_all, a_cap, b_cap, meta) if a_cap else "NO TRADES"
    text += "\nuniverse: " + ", ".join(data.keys()) + "\nskipped: " + ", ".join(skipped) + "\n"
    print(text)
    with open("lvar_daily_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open("lvar_daily_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "strength"]
        w.writerow(cols)
        for x in a_cap:
            w.writerow([x[k] for k in cols])


if __name__ == "__main__":
    main()
