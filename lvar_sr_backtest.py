"""
SUPPORT / RESISTANCE BACKTEST - Binance USDT-M perpetuals, 4-hour bars, real funding. Rules fixed IN ADVANCE, nothing tuned.
Reuses the download, cost and report code of lvar_box_backtest.py, lvar_fut_backtest.py and lvar_daily_backtest.py (all three must stay in the repository).

LEVEL (only closed bars are used):
  Pivot: bar p is a pivot HIGH if high[p] > the 5 highs before it and >= the 5 highs after it (pivot LOW is the mirror). It is known only
  after bar p+5 has closed. A level = a cluster of >= 3 pivots (highs and lows together) of the last 180 bars whose prices lie within
  0.25 x ATR14 of each other; the level price is the median of the cluster (clusters closer than 0.25 ATR are one level: the one with more pivots). A level is support when price is above it, resistance when below.
SIGNAL BAR i (LONG; SHORT is the exact mirror):
  Touch: bullish candle (close > open), close > level, low <= level + 0.25 ATR and low >= level - 0.5 ATR (a wick through the zone is allowed).
  A  BOUNCE: no close at or below the level in the 10 bars before i (support was holding).
  B  BREAK AND RETEST: within the 10 bars before i price closed at or below the level (last such bar k, k <= i-2), then closed back above it
     and reached at least level + 0.5 ATR before the retest bar i (a clean breakout, not hugging the level).
  If several levels qualify the one closest to the bar's low (high for SHORT) is used.
Entry = open of the next bar. Stop = min(low of bar i, level) - 0.25 ATR (SHORT: max(high, level) + 0.25 ATR). Target = 2R. Time stop = close of the
30th bar. Risk must be 1%-12% of price. Costs: fee + slippage 0.20% round trip (stress 0.40%) plus the REAL funding while open.
Universe, liquidity filter (30-day average dollar volume >= 20M), one position per coin, at most 8 open positions: identical to lvar_box_backtest.py.
Pass rule (fixed before the run, applied separately to A and B): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import lvar_box_backtest as X
import lvar_daily_backtest as B
import lvar_fut_backtest as F

TF = X.TF
DAY = X.DAY
PIV = 5
LOOKBACK = 180
TOL = 0.25
MIN_PIV = 3
ZONE_BELOW = 0.5
FRESH = 10
BREAK_CLEAN = 0.5
STOP_BUF = 0.25
RR = 2.0
HOLD = 30
RISK_LO, RISK_HI = 0.01, 0.12
MIN_VOL30 = X.MIN_VOL30
COST_RT, COST_STRESS = X.COST_RT, X.COST_STRESS


# ---------------------------------------------------------------- levels
def find_pivots(h, l):
    """(index, price) of confirmed-structure pivots. Pivot p is only usable from bar p+PIV on."""
    n = len(h)
    idx, px = [], []
    for p in range(PIV, n - PIV):
        left_h, right_h = h[p - PIV:p], h[p + 1:p + PIV + 1]
        left_l, right_l = l[p - PIV:p], l[p + 1:p + PIV + 1]
        if h[p] > left_h.max() and h[p] >= right_h.max():
            idx.append(p)
            px.append(h[p])
        if l[p] < left_l.min() and l[p] <= right_l.min():
            idx.append(p)
            px.append(l[p])
    return np.array(idx, int), np.array(px, float)


def levels_at(i, piv_idx, piv_px, a):
    """[(level_price, n_pivots)] using only pivots confirmed by bar i (p + PIV <= i) within the last LOOKBACK bars."""
    lo = int(np.searchsorted(piv_idx, i - LOOKBACK, side="left"))
    hi = int(np.searchsorted(piv_idx, i - PIV, side="right"))
    if hi - lo < MIN_PIV:
        return []
    pr = piv_px[lo:hi]
    tol = TOL * a
    near = np.abs(pr[:, None] - pr[None, :]) <= tol
    cnt = near.sum(1)
    out = []
    for q in np.nonzero(cnt >= MIN_PIV)[0]:
        lvl, npv = float(np.median(pr[near[q]])), int(cnt[q])
        dup = [j for j, (x, _) in enumerate(out) if abs(x - lvl) <= tol]
        if not dup:
            out.append((lvl, npv))
        elif npv > out[dup[0]][1]:
            out[dup[0]] = (lvl, npv)
    return out


def make_candidates(sym, df, ft, fr, mode):
    n = len(df)
    if n < LOOKBACK + 10:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy(np.int64)
    ind = X.indicators4(df)
    atr, vol = ind["atr"], ind["vol"]
    piv_idx, piv_px = find_pivots(h, l)
    out = []
    for i in range(LOOKBACK, n - 2):
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(vol[i])) or vol[i] < MIN_VOL30:
            continue
        bull = c[i] > o[i]
        bear = c[i] < o[i]
        if not (bull or bear):
            continue
        if t[i] - t[i - FRESH - 1] != (FRESH + 1) * TF or t[i + 1] - t[i] != TF:
            continue
        side = 1 if bull else -1
        best = None
        for lvl, npv in levels_at(i, piv_idx, piv_px, a):
            tol = TOL * a
            if side == 1:
                if not (c[i] > lvl and lvl - ZONE_BELOW * a <= l[i] <= lvl + tol):
                    continue
                dist = abs(l[i] - lvl)
                prev = c[i - FRESH:i]
                bad = np.nonzero(prev <= lvl)[0]
                if mode == "A":
                    ok = len(bad) == 0
                else:
                    if len(bad) == 0:
                        continue
                    kk = i - FRESH + int(bad[-1])
                    ok = kk <= i - 2 and c[kk + 1:i].max() >= lvl + BREAK_CLEAN * a
            else:
                if not (c[i] < lvl and lvl - tol <= h[i] <= lvl + ZONE_BELOW * a):
                    continue
                dist = abs(h[i] - lvl)
                prev = c[i - FRESH:i]
                bad = np.nonzero(prev >= lvl)[0]
                if mode == "A":
                    ok = len(bad) == 0
                else:
                    if len(bad) == 0:
                        continue
                    kk = i - FRESH + int(bad[-1])
                    ok = kk <= i - 2 and c[kk + 1:i].min() <= lvl - BREAK_CLEAN * a
            if ok and (best is None or dist < best[0]):
                best = (dist, lvl, npv)
        if best is None:
            continue
        _, lvl, npv = best
        k = i + 1
        entry = o[k]
        stop = (min(l[i], lvl) - STOP_BUF * a) if side == 1 else (max(h[i], lvl) + STOP_BUF * a)
        risk = side * (entry - stop)
        risk_pct = risk / entry if entry > 0 else 0.0
        if risk <= 0 or not (RISK_LO <= risk_pct <= RISK_HI):
            continue
        target = entry + side * RR * risk
        if target <= 0:
            continue
        res = F.sim(o, h, l, c, k, side, entry, stop, target, HOLD)
        if res is None:
            continue
        fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
        out.append({"pair": sym, "side": "LONG" if side == 1 else "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[1]]),
                    "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY), "risk_pct": float(risk_pct),
                    "r_gross": float(res[0]), "r_net": float(res[0] - (COST_RT + side * fsum) / risk_pct),
                    "r_stress": float(res[0] - (COST_STRESS + side * fsum) / risk_pct), "strength": float(npv),
                    "funding_sum": fsum, "touches": int(npv)})
    return out


# ---------------------------------------------------------------- main run
def main():
    t_run = time.time()
    months = F.month_list()
    dl_months = ["2022-10"] + months
    now = datetime.now(timezone.utc)
    end_t = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp())
    start_t = int(datetime(F.START_Y, F.START_M, 1, tzinfo=timezone.utc).timestamp())
    listed = F.list_symbols()
    source = "download-site listing" if listed else "built-in fallback list (listing not reachable)"
    symbols = [s for s in (listed or F.FALLBACK) if F.usable_symbol(s) and not X.non_crypto(s)]
    selected, _ = F.rank_symbols(symbols)
    print(f"selected {len(selected)} symbols", flush=True)
    L = ["SUPPORT / RESISTANCE BACKTEST - pivot-cluster levels, 4h bars, Binance USDT-M perpetuals, real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source} (TradFi names removed)",
         "selected: " + ", ".join(selected),
         "costs: fee+slippage 0.20% round trip (stress 0.40%) plus REAL funding while open; max 8 open positions",
         "Pass rule fixed before the run, applied to A and B separately: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Two hypotheses are tested (A bounce, B break+retest): a pass of exactly one of them is weak evidence.",
         "Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (SL assumed first), news not in data,",
         "the level definition is one mechanical reading of discretionary support/resistance (a fail rejects this reading, not the trader)."]
    if len(selected) < 20:
        L.append("ABORTED: too few symbols could be downloaded.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_sr_result.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return
    tasks = []
    for s in selected:
        for m in dl_months:
            tasks.append(("k4", s, f"{F.BASE}/monthly/klines/{s}/4h/{s}-4h-{m}.zip"))
            tasks.append(("fu", s, f"{F.BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"))
    kl, fu = {s: [] for s in selected}, {s: [] for s in selected}
    cov = {"k4": [0, 0], "fu": [0, 0]}
    done = 0
    with ThreadPoolExecutor(F.WORKERS) as ex:
        for kind, s, data in ex.map(X.job4, tasks):
            cov[kind][1] += 1
            if data:
                cov[kind][0] += 1
                (kl if kind == "k4" else fu)[s].extend(data)
            done += 1
            if done % 1000 == 0:
                print(f"  downloaded {done}/{len(tasks)} files ({time.time() - t_run:.0f}s)", flush=True)
    L.append(f"files found/requested: 4h klines {cov['k4'][0]}/{cov['k4'][1]} | funding {cov['fu'][0]}/{cov['fu'][1]}")
    cands = {"A": [], "B": []}
    used = 0
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = X.build_frame4(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) < 600:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        for mode in ("A", "B"):
            cands[mode] += [x for x in make_candidates(s, df, ft, fr, mode) if x["entry_t"] >= start_t]
        if used % 10 == 0:
            print(f"  signals computed for {used} symbols ({time.time() - t_run:.0f}s)", flush=True)
    L.append(f"symbols with usable data: {used}")
    L.append("")
    taken_all = []
    for title, mode in (("A - BOUNCE off a support/resistance level (eligible)", "A"), ("B - BREAK AND RETEST of a level (eligible)", "B")):
        cap = B.portfolio(cands[mode])
        L.append(f"signals before the position cap: {len(cands[mode])} | taken: {len(cap)}")
        L += B.block(title, cap, True)
        for x in cap:
            taken_all.append(dict(x, mode=mode))
        if cap:
            fsm = np.array([x["funding_sum"] * (1 if x["side"] == "LONG" else -1) for x in cap])
            L.append(f"  average real funding cost per trade: {100 * fsm.mean():+.3f}% of position (negative = funding was income)")
            L.append("")
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_sr_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_sr_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["mode", "pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "funding_sum", "touches"]
        w.writerow(cols)
        for x in taken_all:
            w.writerow([x[k] for k in cols])


# ---------------------------------------------------------------- self-tests (offline)
PASSED = 0
T0 = 1_650_000_000 // DAY * DAY


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def zigzag(n_cycles=14, tail=None, qv=5e7):
    """closes zig-zag between 100.3 and 112 with a 20-bar period; every minimum bar has low 100.0 and every maximum bar has high 112.3,
    so the lows form a clean support at 100.0 and the highs a clean resistance at 112.3."""
    c = []
    for cyc in range(n_cycles):
        c += list(np.linspace(112.0, 100.3, 11)[:-1]) + list(np.linspace(100.3, 112.0, 11)[:-1])
    c = np.array(c + [112.0])
    n = len(c)
    o = np.concatenate([[c[0] + 0.5], c[:-1]])
    h = np.maximum(o, c) + 0.3
    l = np.minimum(o, c) - 0.3
    return pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": o, "high": h, "low": l, "close": c, "qv": qv})


def with_bars(df, bars):
    """replace bars (list of (idx, o, h, l, c)) and return a copy long enough for the exit logic."""
    d = df.copy()
    for idx, o_, h_, l_, c_ in bars:
        d.loc[idx, ["open", "high", "low", "close"]] = [o_, h_, l_, c_]
    return d


def fund_for(df, rate=0.0001):
    return np.array([float(x) for x in df["time"] + 3600]), np.full(len(df), rate)


def cands(df, mode):
    ft, fr = fund_for(df)
    return make_candidates("R", df, ft, fr, mode)


def cands_at(i, df, mode):
    """candidates whose signal bar is bar i (the zig-zag also produces other natural signals elsewhere)."""
    te = int(df["time"].iloc[i + 1]) if len(df) > i + 1 else -1
    return [x for x in cands(df, mode) if x["entry_t"] == te]


def extend_flat(df, price, bars):
    last_t = int(df["time"].iloc[-1])
    add = pd.DataFrame({"time": last_t + TF * np.arange(1, bars + 1), "open": price, "high": price + 0.2, "low": price - 0.2, "close": price, "qv": 5e7})
    return pd.concat([df, add], ignore_index=True)


def brute_reference(df, mode):
    """independent, deliberately naive reading of the written rules (plain loops, no shared helper) used to cross-check make_candidates."""
    import statistics
    o, h, l, c = (df[k].tolist() for k in ("open", "high", "low", "close"))
    t = df["time"].tolist()
    n = len(c)
    ind = X.indicators4(df)
    atr, vol = ind["atr"], ind["vol"]
    piv = []
    for p in range(5, n - 5):
        if h[p] > max(h[p - 5:p]) and h[p] >= max(h[p + 1:p + 6]):
            piv.append((p, h[p]))
        if l[p] < min(l[p - 5:p]) and l[p] <= min(l[p + 1:p + 6]):
            piv.append((p, l[p]))
    out = set()
    o_a, h_a, l_a, c_a = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    for i in range(180, n - 2):
        a = atr[i]
        if not (a == a and a > 0 and vol[i] == vol[i]) or vol[i] < 2e7 or c[i] == o[i]:
            continue
        if t[i] - t[i - 11] != 11 * 14400 or t[i + 1] - t[i] != 14400:
            continue
        side = 1 if c[i] > o[i] else -1
        pr = [px for (p, px) in piv if p + 5 <= i and p >= i - 180]
        tol = 0.25 * a
        levels = []
        for x in pr:
            mem = [y for y in pr if abs(y - x) <= tol]
            if len(mem) >= 3:
                lv = statistics.median(mem)
                dup = [j for j, q in enumerate(levels) if abs(lv - q[0]) <= tol]
                if not dup:
                    levels.append((lv, len(mem)))
                elif len(mem) > levels[dup[0]][1]:
                    levels[dup[0]] = (lv, len(mem))
        best = None
        for lv, _ in levels:
            if side == 1:
                if not (c[i] > lv and lv - 0.5 * a <= l[i] <= lv + tol):
                    continue
                bad = [j for j in range(i - 10, i) if c[j] <= lv]
                dist = abs(l[i] - lv)
            else:
                if not (c[i] < lv and lv - tol <= h[i] <= lv + 0.5 * a):
                    continue
                bad = [j for j in range(i - 10, i) if c[j] >= lv]
                dist = abs(h[i] - lv)
            if mode == "A":
                ok = not bad
            else:
                if not bad or bad[-1] > i - 2:
                    continue
                seg = c[bad[-1] + 1:i]
                ok = (max(seg) >= lv + 0.5 * a) if side == 1 else (min(seg) <= lv - 0.5 * a)
            if ok and (best is None or dist < best[0]):
                best = (dist, lv)
        if not best:
            continue
        lv, k = best[1], i + 1
        entry = o[k]
        stop = (min(l[i], lv) - 0.25 * a) if side == 1 else (max(h[i], lv) + 0.25 * a)
        risk = side * (entry - stop)
        if risk <= 0 or not (0.01 <= risk / entry <= 0.12) or entry + side * 2 * risk <= 0:
            continue
        if F.sim(o_a, h_a, l_a, c_a, k, side, entry, stop, entry + side * 2 * risk, 30) is None:
            continue
        out.add((t[k], side, round(risk / entry, 10)))
    return out


def run_selftest():
    # pivots and levels
    z = zigzag()
    h, l = z["high"].to_numpy(float), z["low"].to_numpy(float)
    pi, pp = find_pivots(h, l)
    check(len(pi) > 20 and all(abs(p - 100.0) < 1e-9 or abs(p - 112.3) < 1e-9 for p in pp), "pivots found only at the real turning bars")
    check(all(h[p] > h[p - PIV:p].max() or l[p] < l[p - PIV:p].min() for p in pi), "every pivot beats the 5 bars before it")
    lv = levels_at(150, pi, pp, 1.5)
    check(len(lv) == 2 and abs(min(x[0] for x in lv) - 100.0) < 1e-9 and abs(max(x[0] for x in lv) - 112.3) < 1e-9, f"support 100.0 and resistance 112.3 found: {lv}")
    check(levels_at(30, pi, pp, 1.5) == [], "too few pivots early on gives no level")
    check(all(p + PIV <= 150 for p in pi[(pi >= 150 - LOOKBACK) & (pi + PIV <= 150)]), "only confirmed pivots are used")
    lv_future = levels_at(150, pi[pi + PIV <= 150], pp[pi + PIV <= 150], 1.5)
    check(lv == lv_future, "levels at bar 150 do not depend on later bars")
    # bounce long: after 7 cycles (bar 140 = minimum) craft a bullish touch bar at a later minimum
    n_touch = 14 * 20 - 10                     # a minimum bar inside the series (bar index of the last full minimum)
    i = 190
    base = z.iloc[:i + 1].copy()
    base = with_bars(base, [(i - 1, 103.0, 103.3, 100.6, 100.9), (i, 100.9, 102.4, 100.2, 102.2)])
    base = extend_flat(base, 106.0, 6)
    base.loc[i + 1:, "open"] = [102.2, 104.5, 107.0, 109.5, 111.0, 111.5, 111.5][:len(base) - i - 1]
    base.loc[i + 1:, "close"] = [104.5, 107.0, 109.5, 111.0, 111.5, 111.5, 111.5][:len(base) - i - 1]
    base.loc[i + 1:, "high"] = base.loc[i + 1:, ["open", "close"]].max(axis=1) + 0.3
    base.loc[i + 1:, "low"] = base.loc[i + 1:, ["open", "close"]].min(axis=1) - 0.3
    got = cands_at(i, base, "A")
    check(len(got) == 1 and got[0]["side"] == "LONG" and got[0]["entry_t"] == int(base["time"].iloc[i + 1]), f"A: one bounce LONG with entry at the next bar open: {got}")
    x = got[0]
    a = X.indicators4(base)["atr"][i]
    entry = base["open"].iloc[i + 1]
    stop = min(base["low"].iloc[i], 100.0) - STOP_BUF * a
    check(abs(x["risk_pct"] - (entry - stop) / entry) < 1e-9, "stop = min(bar low, level) - 0.25 ATR")
    check(x["reason"] == "TP" and abs(x["r_gross"] - 2.0) < 1e-9, f"target reached: {x['reason']} {x['r_gross']}")
    check(cands_at(i, base, "B") == [], "bounce is not a break-and-retest")
    expect = 2.0 - (COST_RT + x["funding_sum"]) / x["risk_pct"]
    check(abs(x["r_net"] - expect) < 1e-9 and x["r_stress"] < x["r_net"] < x["r_gross"], "net R = gross - (cost + funding) / risk")
    # mirror: resistance bounce SHORT with the same R
    m = base.copy()
    m["open"], m["close"] = 212.3 - base["open"], 212.3 - base["close"]
    m["high"], m["low"] = 212.3 - base["low"], 212.3 - base["high"]
    gm = cands_at(i, m, "A")
    check(len(gm) == 1 and gm[0]["side"] == "SHORT" and abs(gm[0]["r_gross"] - 2.0) < 1e-9 and gm[0]["entry_t"] == x["entry_t"], f"mirror gives the SHORT with the same R: {gm}")
    # rules that must cancel the signal
    check(cands_at(i, with_bars(base, [(i, 100.9, 101.5, 100.2, 100.8)]), "A") == [], "bearish bar is not a long signal")
    check(cands_at(i, with_bars(base, [(i, 100.9, 102.4, 98.3, 102.2)]), "A") == [], "wick deeper than 0.5 ATR below the level cancels")
    check(cands_at(i, with_bars(base, [(i, 101.5, 103.0, 101.4, 102.9)]), "A") == [], "no touch of the zone, no signal")
    check(cands_at(i, with_bars(base, [(i - 3, 101.0, 101.3, 99.0, 99.8)]), "A") == [], "a close below the level in the last 10 bars cancels the bounce")
    d7 = base.copy()
    d7["qv"] = 1e6
    check(cands_at(i, d7, "A") == [], "30-day volume filter")
    d8 = base.copy()
    d8.loc[i - 4:, "time"] += TF
    check(cands_at(i, d8, "A") == [], "a missing bar before the signal blocks it")
    # break and retest: price closes below the support at bars i-6..i-5, recovers above 100.5 then retests at bar i
    bt = z.iloc[:i + 1].copy()
    bt = with_bars(bt, [(i - 7, 104.0, 104.2, 99.0, 99.5), (i - 6, 99.5, 99.9, 98.5, 99.2), (i - 5, 99.2, 103.0, 99.0, 102.8),
                        (i - 4, 102.8, 104.5, 102.5, 104.2), (i - 3, 104.2, 105.0, 103.2, 103.6), (i - 2, 103.6, 103.9, 101.9, 102.1),
                        (i - 1, 102.1, 102.3, 100.7, 100.9), (i, 100.9, 102.4, 100.3, 102.2)])
    bt = extend_flat(bt, 106.0, 6)
    bt.loc[i + 1:, "open"] = [102.2, 104.5, 107.0, 109.5, 111.0, 111.5, 111.5][:len(bt) - i - 1]
    bt.loc[i + 1:, "close"] = [104.5, 107.0, 109.5, 111.0, 111.5, 111.5, 111.5][:len(bt) - i - 1]
    bt.loc[i + 1:, "high"] = bt.loc[i + 1:, ["open", "close"]].max(axis=1) + 0.3
    bt.loc[i + 1:, "low"] = bt.loc[i + 1:, ["open", "close"]].min(axis=1) - 0.3
    gb = cands_at(i, bt, "B")
    check(len(gb) == 1 and gb[0]["side"] == "LONG" and gb[0]["entry_t"] == int(bt["time"].iloc[i + 1]), f"B: one retest LONG: {gb}")
    check(cands_at(i, bt, "A") == [], "a level that was just broken is not a bounce")
    weak = with_bars(bt, [(i - 5, 99.2, 100.9, 99.0, 100.6), (i - 4, 100.6, 100.9, 100.4, 100.7), (i - 3, 100.7, 100.9, 100.4, 100.7), (i - 2, 100.7, 100.9, 100.4, 100.8)])
    check(cands_at(i, weak, "B") == [], "a breakout that never reached level + 0.5 ATR is not clean")
    # no look-ahead: results that finish before the cut do not change when the future is changed
    r1 = np.random.default_rng(100)
    nb = 12000
    cc = 100 * np.exp(np.cumsum(r1.normal(0, 0.006, nb)))
    oo = np.concatenate([[100.0], cc[:-1]])
    big = pd.DataFrame({"time": T0 + np.arange(nb) * TF, "open": oo, "high": np.maximum(oo, cc) * (1 + np.abs(r1.normal(0, 0.003, nb))),
                        "low": np.minimum(oo, cc) * (1 - np.abs(r1.normal(0, 0.003, nb))), "close": cc, "qv": 5e7})
    for mode in ("A", "B"):
        full = cands(big, mode)
        check(len(full) >= 5, f"random series produce {mode} setups ({len(full)})")
        cut = 8000
        fut = big.copy()
        fut.loc[cut:, ["open", "high", "low", "close"]] *= 3.0
        alt = cands(fut, mode)
        tcut = int(big["time"].iloc[cut])
        a1 = {(q["entry_t"], q["side"]): q for q in full if q["exit_t"] < tcut}
        a2 = {(q["entry_t"], q["side"]): q for q in alt if q["exit_t"] < tcut}
        check(len(a1) >= 3 and a1.keys() == a2.keys(), f"{mode}: future prices do not change earlier trades ({len(a1)})")
        for k_, y in a1.items():
            z_ = a2[k_]
            check(abs(y["risk_pct"] - z_["risk_pct"]) < 1e-12 and abs(y["r_gross"] - z_["r_gross"]) < 1e-12 and y["exit_t"] == z_["exit_t"], "identical trade data")
    # independent cross-check: a naive plain-loop reading of the written rules gives exactly the same signals
    for sd in (500, 501):
        r3 = np.random.default_rng(sd)
        n3 = 4000
        c3 = 100 * np.exp(np.cumsum(r3.normal(0, 0.006, n3)))
        o3 = np.concatenate([[100.0], c3[:-1]])
        d3 = pd.DataFrame({"time": T0 + np.arange(n3) * TF, "open": o3, "high": np.maximum(o3, c3) * (1 + np.abs(r3.normal(0, 0.003, n3))),
                           "low": np.minimum(o3, c3) * (1 - np.abs(r3.normal(0, 0.003, n3))), "close": c3, "qv": 5e7})
        for mode in ("A", "B"):
            got = {(q["entry_t"], 1 if q["side"] == "LONG" else -1, round(q["risk_pct"], 10)) for q in cands(d3, mode)}
            check(len(got) >= 20 and got == brute_reference(d3, mode), f"{mode}: matches the independent naive implementation ({len(got)} signals)")
    # null test: driftless random walk -> gross about zero, net clearly negative
    allg, alln = [], []
    for sd in range(6):
        r2 = np.random.default_rng(100 + sd)
        n2 = 12000
        c2 = 100 * np.exp(np.cumsum(r2.normal(0, 0.006, n2)))
        o2 = np.concatenate([[100.0], c2[:-1]])
        d2 = pd.DataFrame({"time": T0 + np.arange(n2) * TF, "open": o2, "high": np.maximum(o2, c2) * (1 + np.abs(r2.normal(0, 0.003, n2))),
                           "low": np.minimum(o2, c2) * (1 - np.abs(r2.normal(0, 0.003, n2))), "close": c2, "qv": 5e7})
        for mode in ("A", "B"):
            for q in cands(d2, mode):
                allg.append(q["r_gross"])
                alln.append(q["r_net"])
    gm_, nm_ = float(np.mean(allg)), float(np.mean(alln))
    print(f"null (random walk): n={len(allg)} gross={gm_:+.3f} net={nm_:+.3f}")
    check(len(allg) >= 300 and abs(gm_) < 0.12 and nm_ < gm_ - 0.05, "random walk: gross about 0, net below gross (no hidden edge or bias)")
    cap = B.portfolio(cands(big, "A"))
    check(0 < len(cap), "portfolio keeps trades")
    text = "\n".join(B.block("test", cap, True))
    check("VERDICT" in text and "bootstrap" in text, "report block renders")
    print(f"ALL SR SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
