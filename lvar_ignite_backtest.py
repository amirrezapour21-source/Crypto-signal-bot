"""
IGNITION BACKTEST ("catch the start of a pump") - Binance USDT-M perpetuals, 4-hour bars, real funding.
Rules are fixed IN ADVANCE and written as numbers; nothing is tuned after seeing results (read-only research).
Reuses download / cost / report code of lvar_box_backtest.py, lvar_fut_backtest.py and lvar_daily_backtest.py (all three must stay in the repository).

IDEA IN ONE LINE: a coin sits quietly in a tight range for days (the compressed spring), then one big high-volume candle breaks out of the range
(the spring is released). We buy at the next candle open. We do NOT try to predict a pump; we only join one in its first candle.

LONG ONLY. Signal bar i (a closed 4h bar). ATR means ATR14 of the bar BEFORE i (bar i-1), so the breakout bar cannot inflate it.
  1. BASE   : the 40 bars before i (bars i-40 .. i-1) have (highest high - lowest low) <= 6 x ATR.            -> quiet, compressed
  2. BREAK  : close of bar i is above the highest high of those 40 bars.                                      -> left the range
  3. VOLUME : quote volume of bar i >= 3 x the average quote volume of those 40 bars.                          -> real participation
  4. BODY   : close - open of bar i >= 1.5 x ATR.                                                              -> a strong candle, not a drift
  5. PUMP FILTERS (set by the owner): 24h price change at the close of bar i (close[i] / close[i-6] - 1) between +10% and +50%,
     and 24h quote volume (sum of bars i-5 .. i) >= 40,000,000 USDT. These replace the old 30-day liquidity filter.
     (The "boosted token", chain and age filters of the screener do not exist for futures contracts and are not used.)
Entry  = open of bar i+1.  Stop = low of bar i - 0.25 x ATR.  Risk (entry - stop) must be 1% to 15% of the entry price, else the signal is skipped.
Target = 3R (pumps are asymmetric: a low win rate is expected).  Time stop = close of the 30th bar after entry (5 days).  SL is checked before TP inside a bar;
a bar that opens below the stop exits at its open.  One position per coin, at most 8 open positions at once (the strongest volume surge wins a tie).
Costs: fee + slippage 0.20% round trip (stress 0.40%) plus the REAL funding paid while the position was open (long pays positive funding).
Universe: every Binance USDT-M perpetual listed on the download site (delisted ones included), TradFi names removed. Period: 2022-11-01 to the end of the last complete month.
Pass rule (fixed before the run): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
Information only (never eligible): (a) the same signals with a 2R target, (b) the same setup WITHOUT the 3x volume condition,
(c) the earlier specification WITHOUT the 24h filters but with a 30-day average daily volume >= 5M USDT (shows what the owner's filters change).
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
BPD = X.BPD
BASE_N = 40
BASE_MAX_ATR = 6.0
VOL_MULT = 3.0
BODY_ATR = 1.5
STOP_BUF = 0.25
RR = 3.0
HOLD = 30
RISK_LO, RISK_HI = 0.01, 0.15
MIN_VOL30 = 5_000_000          # only used by the information-only 'legacy' run
CHG_LO, CHG_HI = 0.10, 0.50
VOL24_MIN = 40_000_000
VOL_BARS = X.VOL_BARS
COST_RT, COST_STRESS = X.COST_RT, X.COST_STRESS


# ---------------------------------------------------------------- signals
def make_candidates(sym, df, ft, fr, rr=RR, use_volume=True, mode="user"):
    n = len(df)
    if n < VOL_BARS + BASE_N + 10:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    qv = df["qv"].to_numpy(float)
    t = df["time"].to_numpy(np.int64)
    atr = X.indicators4(df)["atr"]
    s_h, s_l, s_q = pd.Series(h), pd.Series(l), pd.Series(qv)
    base_hi = s_h.rolling(BASE_N).max().shift(1).to_numpy(float)
    base_lo = s_l.rolling(BASE_N).min().shift(1).to_numpy(float)
    base_qv = s_q.rolling(BASE_N).mean().shift(1).to_numpy(float)
    vol30 = (s_q.rolling(VOL_BARS, min_periods=VOL_BARS).mean().shift(1) * BPD).to_numpy(float)
    a_prev = np.concatenate([[np.nan], atr[:-1]])
    c_6 = np.concatenate([np.full(6, np.nan), c[:-6]])
    vol24 = s_q.rolling(6, min_periods=6).sum().to_numpy(float)
    with np.errstate(invalid="ignore"):
        chg24 = c / c_6 - 1.0
        ok = (np.isfinite(a_prev) & (a_prev > 0) & np.isfinite(base_hi) & np.isfinite(base_qv)
              & ((base_hi - base_lo) <= BASE_MAX_ATR * a_prev) & (c > base_hi) & ((c - o) >= BODY_ATR * a_prev))
        if mode == "user":
            ok &= np.isfinite(chg24) & (chg24 >= CHG_LO) & (chg24 <= CHG_HI) & np.isfinite(vol24) & (vol24 >= VOL24_MIN)
        else:
            ok &= np.isfinite(vol30) & (vol30 >= MIN_VOL30)
        if use_volume:
            ok &= qv >= VOL_MULT * base_qv
    out = []
    for i in np.nonzero(ok)[0]:
        k = i + 1
        if k >= n - 1 or t[k] - t[i] != TF or t[i] - t[i - BASE_N] != BASE_N * TF:
            continue
        entry = o[k]
        stop = l[i] - STOP_BUF * a_prev[i]
        risk = entry - stop
        risk_pct = risk / entry if entry > 0 else 0.0
        if risk <= 0 or not (RISK_LO <= risk_pct <= RISK_HI):
            continue
        res = F.sim(o, h, l, c, k, 1, entry, stop, entry + rr * risk, HOLD)
        if res is None:
            continue
        fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
        out.append({"pair": sym, "side": "LONG", "entry_t": int(t[k]), "exit_t": int(t[res[1]]), "reason": res[2],
                    "days": float((res[1] - k + 1) * TF / DAY), "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                    "r_net": float(res[0] - (COST_RT + fsum) / risk_pct), "r_stress": float(res[0] - (COST_STRESS + fsum) / risk_pct),
                    "strength": float(qv[i] / base_qv[i]) if base_qv[i] > 0 else 0.0, "funding_sum": fsum, "touches": 0})
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
    symbols = sorted(s for s in (listed or F.FALLBACK) if F.usable_symbol(s) and not X.non_crypto(s))
    print(f"symbols from {source}: {len(symbols)}", flush=True)
    L = ["IGNITION BACKTEST - compressed base + high-volume breakout candle, long only, 4h bars, Binance USDT-M perpetuals, real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(symbols)} from the {source} (TradFi names removed); pump filters: 24h change +10%..+50% and 24h volume >= 40M USDT at the signal bar",
         "costs: fee+slippage 0.20% round trip (stress 0.40%) plus REAL funding while open; max 8 open positions; target 3R",
         "Pass rule fixed before the run: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Known problems: entry at the next bar open (the pump may already be under way), small coins can be hard to fill at size, 4h bars hide intrabar order",
         "(SL assumed first), news not in data, listing-day pumps are in the sample, delisted contracts are included."]
    if len(symbols) < 50:
        L.append("ABORTED: too few symbols could be listed; the download site may be blocked or changed.")
        text = "\n".join(L) + "\n"
        print(text)
        open("lvar_ignite_result.txt", "w", encoding="utf-8").write(text)
        return
    tasks = []
    for s in symbols:
        for m in dl_months:
            tasks.append(("k4", s, f"{F.BASE}/monthly/klines/{s}/4h/{s}-4h-{m}.zip"))
            tasks.append(("fu", s, f"{F.BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"))
    kl, fu = {s: [] for s in symbols}, {s: [] for s in symbols}
    cov = {"k4": [0, 0], "fu": [0, 0]}
    done = 0
    with ThreadPoolExecutor(F.WORKERS) as ex:
        for kind, s, data in ex.map(X.job4, tasks):
            cov[kind][1] += 1
            if data:
                cov[kind][0] += 1
                (kl if kind == "k4" else fu)[s].extend(data)
            done += 1
            if done % 2000 == 0:
                print(f"  downloaded {done}/{len(tasks)} files ({time.time() - t_run:.0f}s)", flush=True)
    L.append(f"files found/requested: 4h klines {cov['k4'][0]}/{cov['k4'][1]} | funding {cov['fu'][0]}/{cov['fu'][1]}")
    sets = {"main": [], "rr2": [], "novol": [], "legacy": []}
    used = 0
    for s in symbols:
        if not kl[s] or not fu[s]:
            continue
        df = X.build_frame4(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) < 600:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        sets["main"] += [x for x in make_candidates(s, df, ft, fr) if x["entry_t"] >= start_t]
        sets["rr2"] += [x for x in make_candidates(s, df, ft, fr, rr=2.0) if x["entry_t"] >= start_t]
        sets["novol"] += [x for x in make_candidates(s, df, ft, fr, use_volume=False) if x["entry_t"] >= start_t]
        sets["legacy"] += [x for x in make_candidates(s, df, ft, fr, mode="legacy") if x["entry_t"] >= start_t]
    L.append(f"symbols with usable data: {used}")
    L.append("")
    taken_all = []
    plan = [("IGNITION - compressed base + breakout candle, 3x volume, 24h change +10..+50%, 24h volume >= 40M, target 3R (eligible)", "main", True),
            ("INFO ONLY - same signals with a 2R target", "rr2", False),
            ("INFO ONLY - same setup WITHOUT the 3x volume condition (does the volume surge matter?)", "novol", False),
            ("INFO ONLY - earlier specification: NO 24h change/volume filters, 30-day average volume >= 5M (what do the owner's filters change?)", "legacy", False)]
    for title, key, eligible in plan:
        cap = B.portfolio(sets[key])
        L.append(f"signals before the position cap: {len(sets[key])} | taken: {len(cap)}")
        L += B.block(title, cap, eligible)
        if eligible:
            taken_all += cap
            if cap:
                fsm = np.array([x["funding_sum"] for x in cap])
                L.append(f"  average real funding cost per trade: {100 * fsm.mean():+.3f}% of position (negative = funding was income)")
        L.append("")
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_ignite_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_ignite_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "funding_sum", "strength"]
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


def crafted(n_flat=200, qv=5e7):
    """200 quiet bars around 100 (range 0.6), then ignition bar 200 (+2.0, high volume), then a rally of +1.0 per bar."""
    rows = []
    rng = np.random.default_rng(3)
    for j in range(n_flat):
        c = 100.0 + rng.uniform(-0.15, 0.15)
        o = 100.0 + rng.uniform(-0.15, 0.15)
        rows.append([o, max(o, c) + 0.2, min(o, c) - 0.2, c, qv])
    rows.append([100.0, 102.3, 99.9, 102.0, 4 * qv])
    price = 102.0
    for j in range(40):
        o = price + 0.1
        c = o + 1.0
        rows.append([o, c + 0.3, o - 0.2, c, qv])
        price = c
    d = pd.DataFrame(rows, columns=["open", "high", "low", "close", "qv"])
    d.insert(0, "time", T0 + np.arange(len(d)) * TF)
    return d


def fund_for(df, rate=0.0001):
    return np.array([float(x) for x in df["time"] + 3600]), np.full(len(df), rate)


def cands(df, **kw):
    ft, fr = fund_for(df)
    kw.setdefault("mode", "legacy")
    return make_candidates("R", df, ft, fr, **kw)


def edit(df, idx, **cols):
    d = df.copy()
    for k, v in cols.items():
        d.loc[idx, k] = v
    return d


def naive_reference(df, rr=RR, use_volume=True, mode="legacy"):
    """independent plain-loop reading of the written rules (no pandas rolling, no vector masks)"""
    o, h, l, c, q = (df[k].tolist() for k in ("open", "high", "low", "close", "qv"))
    t = df["time"].tolist()
    n = len(c)
    atr = X.indicators4(df)["atr"]
    o_a, h_a, l_a, c_a = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    out = set()
    for i in range(VOL_BARS + 1, n - 2):
        a = atr[i - 1]
        if not (a == a and a > 0):
            continue
        hi = max(h[i - BASE_N:i])
        lo = min(l[i - BASE_N:i])
        mq = sum(q[i - BASE_N:i]) / BASE_N
        v30 = sum(q[i - VOL_BARS:i]) / VOL_BARS * BPD
        if not (hi - lo <= BASE_MAX_ATR * a and c[i] > hi and c[i] - o[i] >= BODY_ATR * a):
            continue
        if mode == "user":
            if not (CHG_LO <= c[i] / c[i - 6] - 1.0 <= CHG_HI and sum(q[i - 5:i + 1]) >= VOL24_MIN):
                continue
        elif not v30 >= MIN_VOL30:
            continue
        if use_volume and not q[i] >= VOL_MULT * mq:
            continue
        k = i + 1
        if t[k] - t[i] != TF or t[i] - t[i - BASE_N] != BASE_N * TF:
            continue
        entry = o[k]
        stop = l[i] - STOP_BUF * a
        risk = entry - stop
        if risk <= 0 or not (RISK_LO <= risk / entry <= RISK_HI):
            continue
        if F.sim(o_a, h_a, l_a, c_a, k, 1, entry, stop, entry + rr * risk, HOLD) is None:
            continue
        out.add((t[k], round(risk / entry, 10)))
    return out


def fat_series(seed, n=20000, with_spikes=True, scale=1.0):
    r = np.random.default_rng(seed)
    ret = r.standard_t(3, n) * 0.004 * scale
    quiet = (np.sin(np.arange(n) / 150.0 + seed) > 0.3)
    ret = np.where(quiet, ret * 0.35, ret)                       # alternating quiet and active regimes so that compressed bases exist
    c = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100.0], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.0015, n)))
    lo = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.0015, n)))
    qv = r.lognormal(17.5, 0.9, n) * (np.where(quiet, 0.5, 1.0) if with_spikes else 1.0)
    return pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": o, "high": hi, "low": lo, "close": c, "qv": qv})


def run_selftest():
    d = crafted()
    got = cands(d)
    check(len(got) == 1 and got[0]["entry_t"] == int(d["time"].iloc[201]), f"one ignition signal, entry at the next bar open: {got}")
    x = got[0]
    a = X.indicators4(d)["atr"][199]
    entry = d["open"].iloc[201]
    stop = d["low"].iloc[200] - STOP_BUF * a
    check(abs(x["risk_pct"] - (entry - stop) / entry) < 1e-12, "stop = low of the ignition bar - 0.25 ATR(bar before)")
    check(x["reason"] == "TP" and abs(x["r_gross"] - RR) < 1e-9, f"target = 3R reached: {x['reason']} {x['r_gross']}")
    expect = RR - (COST_RT + x["funding_sum"]) / x["risk_pct"]
    check(abs(x["r_net"] - expect) < 1e-9 and x["r_stress"] < x["r_net"] < x["r_gross"], "net R = gross - (cost + funding) / risk")
    check(len(cands(d, rr=2.0)) == 1 and abs(cands(d, rr=2.0)[0]["r_gross"] - 2.0) < 1e-9, "2R sensitivity run")
    # each condition on its own cancels the signal
    te = int(d["time"].iloc[201])
    at = lambda lst: [q for q in lst if q["entry_t"] == te]
    check(at(cands(edit(d, 200, qv=2.0 * 5e7))) == [] and len(at(cands(edit(d, 200, qv=2.0 * 5e7), use_volume=False))) == 1, "volume below 3x cancels (and the no-volume control still sees it)")
    check(cands(edit(d, 200, close=100.5, high=100.8)) == [], "body below 1.5 ATR cancels")
    check(cands(edit(d, 180, low=95.0)) == [], "a base wider than 6 ATR cancels")
    check(cands(edit(d, 190, high=102.6)) == [], "close not above the highest high of the base cancels")
    check(cands(d.assign(qv=d["qv"] * 0.01)) == [], "liquidity filter (average daily volume >= 5M) cancels")
    check(cands(edit(d, 200, low=60.0)) == [], "risk above 15% of price cancels")
    d2 = d.copy()
    d2.loc[161:, "time"] += TF
    check(cands(d2) == [], "a missing bar inside the base blocks the signal")
    # stop hit: price falls after entry
    ds = d.copy()
    ds.loc[201, ["open", "high", "low", "close"]] = [102.1, 102.2, 97.0, 97.5]
    gs = cands(ds)
    check(len(gs) == 1 and gs[0]["reason"] == "SL" and abs(gs[0]["r_gross"] + 1.0) < 1e-9, "stop inside the entry bar = -1R")
    dg = d.copy()
    dg.loc[202, ["open", "high", "low", "close"]] = [96.0, 96.5, 95.5, 96.0]
    gg = cands(dg)
    check(len(gg) == 1 and gg[0]["r_gross"] < -1.0 and gg[0]["reason"] == "SL", "a gap below the stop exits at the open (worse than -1R)")
    # the owner's pump filters (24h change +10..+50%, 24h volume >= 40M) in "user" mode
    def pump(close=112.0, high=112.3, qv=5e7):
        e = d.copy()
        e.loc[200, ["open", "high", "low", "close"]] = [100.0, high, 99.9, close]
        e["qv"] = qv
        e.loc[200, "qv"] = 4 * qv
        return e
    u = lambda e, **kw: [q for q in cands(e, mode="user", **kw) if q["entry_t"] == te]
    check(len(u(pump())) == 1, "a +12% ignition with large volume passes the owner's filters")
    check(u(pump(close=109.0, high=109.3)) == [], "a 24h change of +9% is below the 10% filter")
    check(u(pump(close=156.0, high=156.3)) == [], "a 24h change of +56% is above the 50% filter")
    check(u(pump(qv=3e6)) == [] and len([q for q in cands(pump(qv=3e6), mode="legacy") if q["entry_t"] == te]) == 1, "24h volume below 40M cancels (the legacy filter alone would accept it)")
    check(u(d) == [] and len(at(cands(d, mode="legacy"))) == 1, "the earlier quiet +2% ignition no longer qualifies; the legacy run still sees it")
    # independent naive implementation gives exactly the same signals on fat-tailed random data
    total = 0
    for sd in range(1, 7):
        df = fat_series(sd)
        for use_vol in (True, False):
            g = {(q["entry_t"], round(q["risk_pct"], 10)) for q in cands(df, use_volume=use_vol)}
            r_ = naive_reference(df, use_volume=use_vol)
            check(g == r_, f"seed {sd} volume={use_vol}: vectorised signals equal the naive implementation ({len(g)})")
            total += len(g) if use_vol else 0
    check(total >= 10, f"random data produce ignition signals ({total})")
    tot_u = 0
    for sd in range(1, 9):
        df = fat_series(sd, scale=5.0)
        for use_vol in (True, False):
            g = {(q["entry_t"], round(q["risk_pct"], 10)) for q in cands(df, use_volume=use_vol, mode="user")}
            r_ = naive_reference(df, use_volume=use_vol, mode="user")
            check(g == r_, f"seed {sd} volume={use_vol}: user-mode signals equal the naive implementation ({len(g)})")
            tot_u += len(g)
    check(tot_u >= 10, f"volatile random data produce user-mode signals ({tot_u})")
    # no look-ahead
    df = fat_series(2)
    full = cands(df, use_volume=False)
    cut = 14000
    fut = df.copy()
    fut.loc[cut:, ["open", "high", "low", "close"]] *= 3.0
    fut.loc[cut:, "qv"] *= 10.0
    alt = cands(fut, use_volume=False)
    tc = int(df["time"].iloc[cut])
    a1 = {q["entry_t"]: q for q in full if q["exit_t"] < tc}
    a2 = {q["entry_t"]: q for q in alt if q["exit_t"] < tc}
    check(len(a1) >= 5 and a1.keys() == a2.keys() and all(abs(a1[k]["r_gross"] - a2[k]["r_gross"]) < 1e-12 for k in a1), f"future data do not change earlier trades ({len(a1)})")
    # null test: pure random walk with random volume -> no edge before costs, negative after
    gl, nl = [], []
    for sd in range(40):
        r = np.random.default_rng(1000 + sd)
        n = 20000
        quiet = (np.sin(np.arange(n) / 150.0 + sd) > 0.3)
        ret = r.normal(0, 0.004, n) * np.where(quiet, 0.35, 1.0)
        c = 100 * np.exp(np.cumsum(ret))
        o = np.concatenate([[100.0], c[:-1]])
        dn = pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": o, "high": np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.0015, n))),
                           "low": np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.0015, n))), "close": c, "qv": r.lognormal(17.5, 0.9, n)})
        for q in cands(dn, use_volume=False):
            gl.append(q["r_gross"])
            nl.append(q["r_net"])
    gm, nm = float(np.mean(gl)), float(np.mean(nl))
    print(f"null (random walk, volume ignored): n={len(gl)} gross={gm:+.3f} net={nm:+.3f}")
    check(len(gl) >= 150 and abs(gm) < 0.25 and nm < gm - 0.05, "random walk: gross about 0, net below gross (no hidden edge or bias)")
    cap = B.portfolio(cands(fat_series(3), use_volume=False))
    check(len(cap) > 0, "portfolio keeps trades")
    check("VERDICT" in "\n".join(B.block("test", cap, True)), "report block renders")
    print(f"ALL IGNITION SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
