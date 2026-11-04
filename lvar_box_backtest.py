"""
BOX SETUP BACKTEST ("consolidation box + liquidity sweep + reclaim") - Binance USDT-M perpetuals, 4-hour bars.
Rules fixed IN ADVANCE and written as numbers, nothing tuned (read-only research). Data: data.binance.vision (4h klines + funding history).
Reuses the download and report code of lvar_fut_backtest.py and lvar_daily_backtest.py (both must stay in the repository).

Universe: Binance USDT-M perpetuals from the download site (delisted contracts included). TradFi contracts (gold, oil, stock and index
  perpetuals) are removed by name. The 70 symbols with the highest average daily dollar volume in six sample months are kept; a coin is
  tradable only if its 30-day average dollar volume is >= 20M USDT. Period: 2022-11-01 up to the end of the last complete month.

LONG setup (SHORT is the exact mirror: box high, equal highs, sweep above, close back below, 20-day high):
  1. BOX: the last 16 bars ending at bar b. Box height between 1.5 and 6 x ATR14(b). At least 3 distinct touches of the box low
     (a touch = bar low <= box low + 0.25 x ATR; consecutive touching bars count as one touch). These are the "equal lows" (liquidity).
  2. SWEEP: within the next 8 bars after b, price trades below the box low by more than 0.10 x ATR (stop run). The deepest low is the
     sweep low. If the sweep goes deeper than 1.5 x ATR the setup is cancelled. If a bar closes above the box high before a reclaim, cancelled.
  3. HTF POI (higher time frame): the sweep low must be at or below the lowest low of the previous 20 completed daily candles
     (the sweep took a 20-day low). The control run drops this condition.
  4. RECLAIM: the first bar whose close is back above the box low after the sweep. Signal bar = that bar.
  Entry = open of the next bar. Stop = sweep low - 0.25 x ATR. Target = 2R. Time stop = close of the 30th bar. Risk must be 1%-12% of price.
Common: one position per coin, at most 8 open positions (more touches wins a tie), SL before TP inside a bar, SL gaps fill at the open,
  one signal per reclaim bar (the earliest box end wins). Costs: fee + slippage 0.20% round trip (stress 0.40%) plus the REAL funding paid or
  received while the position was open (long pays positive funding, short receives it).
Pass rule (fixed before the run): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
Control (information only, not eligible): the same setup without the HTF condition.
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import lvar_daily_backtest as B
import lvar_fut_backtest as F

DAY = B.DAY
TF = 4 * 3600
BPD = 6
BOX_N = 16
MIN_H = 1.5
MAX_H = 6.0
TOUCH_TOL = 0.25
MIN_TOUCH = 3
SWEEP_BARS = 8
SWEEP_MIN = 0.10
SWEEP_MAX = 1.5
STOP_BUF = 0.25
RR = 2.0
HOLD = 30
RISK_LO = 0.01
RISK_HI = 0.12
HTF_DAYS = 20
ATR_N = 14
VOL_BARS = 180
MIN_VOL30 = 20_000_000
COST_RT = 0.0020
COST_STRESS = 0.0040
NON_CRYPTO = {"XAU", "XAG", "PAXG", "XAUT", "SOXL", "SOXS", "SPCX", "SKHYNIX", "SKHY", "SNDK", "MU", "KORU", "CL", "BZ", "NATGAS", "COPPER",
              "TSLA", "NVDA", "AAPL", "MSTR", "COIN", "HOOD", "CRCL", "QQQ", "SPY", "TSM", "INTC", "AMD", "SNXX", "GOOGL", "AMZN", "META",
              "MSFT", "PLTR", "EWY", "EWJ", "USO", "GLD", "SLV"}


# ---------------------------------------------------------------- data
def parse_klines4(rows):
    """(bar_open_time, open, high, low, close, quote_volume) from a 4h kline CSV (header optional)."""
    out = []
    for r in rows:
        try:
            out.append((int(F.norm_t(r[0])), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])))
        except (ValueError, IndexError):
            continue
    return out


def job4(args):
    kind, sym, url = args
    c = F.fetch(url)
    if c is None:
        return kind, sym, None
    try:
        rows = F.read_zip_rows(c)
        return kind, sym, (parse_klines4(rows) if kind == "k4" else F.parse_funding(rows))
    except Exception:
        return kind, sym, None


def build_frame4(kl):
    df = pd.DataFrame(kl, columns=["time", "open", "high", "low", "close", "qv"])
    df = df.drop_duplicates("time").sort_values("time")
    return df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.qv >= 0)].reset_index(drop=True)


def non_crypto(sym):
    return sym[:-4] in NON_CRYPTO


# ---------------------------------------------------------------- signals
def indicators4(df):
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return {"atr": tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean().to_numpy(float),
            "vol": (df["qv"].rolling(VOL_BARS, min_periods=VOL_BARS).mean() * BPD).to_numpy(float)}


def htf_levels(t, h, l):
    """lowest low / highest high of the previous 20 COMPLETED days, aligned to every bar (NaN if a day is missing)."""
    day = t // DAY
    d0, d1 = int(day.min()), int(day.max())
    s = pd.DataFrame({"day": day, "h": h, "l": l}).groupby("day").agg(hi=("h", "max"), lo=("l", "min")).reindex(np.arange(d0, d1 + 1))
    lo20 = s["lo"].rolling(HTF_DAYS, min_periods=HTF_DAYS).min().shift(1).to_numpy(float)
    hi20 = s["hi"].rolling(HTF_DAYS, min_periods=HTF_DAYS).max().shift(1).to_numpy(float)
    pos = (day - d0).astype(int)
    return lo20[pos], hi20[pos]


def count_touches(x, thr, side):
    """distinct touches of the threshold (low <= thr for side 1, high >= thr for side -1): a touch starts on a bar that hits the
    threshold after a bar that did not, so a flat bottom made of consecutive touching bars counts once."""
    cnt, prev = 0, False
    for v in x:
        hit = bool(v <= thr) if side == 1 else bool(v >= thr)
        if hit and not prev:
            cnt += 1
        prev = hit
    return cnt


def funding_sum4(ft, fr, t_entry, t_exit_bar):
    """sum of funding settled after the entry open up to the end of the exit bar."""
    lo = int(np.searchsorted(ft, t_entry, side="right"))
    hi = int(np.searchsorted(ft, t_exit_bar + TF, side="right"))
    return float(fr[lo:hi].sum())


def make_candidates(sym, df, ft, fr, use_htf):
    n = len(df)
    if n < 400:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy(np.int64)
    ind = indicators4(df)
    atr, vol = ind["atr"], ind["vol"]
    lo20, hi20 = htf_levels(t, h, l)
    boxlo = pd.Series(l).rolling(BOX_N).min().to_numpy(float)
    boxhi = pd.Series(h).rolling(BOX_N).max().to_numpy(float)
    seen, out = set(), []
    for b in range(BOX_N - 1, n - 2):
        a = atr[b]
        if not np.isfinite(a) or a <= 0:
            continue
        hgt = boxhi[b] - boxlo[b]
        if not (np.isfinite(hgt) and MIN_H * a <= hgt <= MAX_H * a):
            continue
        if t[b] - t[b - BOX_N + 1] != (BOX_N - 1) * TF:
            continue
        for side in (1, -1):
            if side == 1:
                edge = boxlo[b]
                touches = count_touches(l[b - BOX_N + 1:b + 1], edge + TOUCH_TOL * a, 1)
            else:
                edge = boxhi[b]
                touches = count_touches(h[b - BOX_N + 1:b + 1], edge - TOUCH_TOL * a, -1)
            if touches < MIN_TOUCH:
                continue
            ext, swept, sig = None, False, None
            for j in range(b + 1, min(b + SWEEP_BARS, n - 2) + 1):
                if t[j] - t[b] != (j - b) * TF:
                    break
                if side == 1:
                    if l[j] < edge:
                        ext = l[j] if ext is None else min(ext, l[j])
                        swept = swept or l[j] < edge - SWEEP_MIN * a
                    if ext is not None and edge - ext > SWEEP_MAX * a:
                        break
                    if swept and c[j] > edge:
                        sig = j
                        break
                    if c[j] > boxhi[b]:
                        break
                else:
                    if h[j] > edge:
                        ext = h[j] if ext is None else max(ext, h[j])
                        swept = swept or h[j] > edge + SWEEP_MIN * a
                    if ext is not None and ext - edge > SWEEP_MAX * a:
                        break
                    if swept and c[j] < edge:
                        sig = j
                        break
                    if c[j] < boxlo[b]:
                        break
            if sig is None or (side, sig) in seen:
                continue
            i, k = sig, sig + 1
            if not (np.isfinite(atr[i]) and np.isfinite(vol[i])) or vol[i] < MIN_VOL30 or t[k] - t[i] != TF:
                continue
            if use_htf:
                lvl = lo20[i] if side == 1 else hi20[i]
                if not np.isfinite(lvl) or (ext > lvl if side == 1 else ext < lvl):
                    continue
            seen.add((side, i))
            entry = o[k]
            stop = ext - side * STOP_BUF * atr[i]
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
            fsum = funding_sum4(ft, fr, t[k], t[res[1]])
            out.append({"pair": sym, "side": "LONG" if side == 1 else "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[1]]),
                        "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY), "risk_pct": float(risk_pct),
                        "r_gross": float(res[0]), "r_net": float(res[0] - (COST_RT + side * fsum) / risk_pct),
                        "r_stress": float(res[0] - (COST_STRESS + side * fsum) / risk_pct), "strength": float(touches),
                        "funding_sum": fsum, "touches": int(touches)})
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
    symbols = [s for s in (listed or F.FALLBACK) if F.usable_symbol(s) and not non_crypto(s)]
    print(f"symbols from {source}: {len(symbols)}", flush=True)
    selected, ranked = F.rank_symbols(symbols)
    print(f"selected {len(selected)} symbols: {', '.join(selected)}", flush=True)
    L = ["BOX SETUP BACKTEST - consolidation box + liquidity sweep + reclaim, 4h bars, Binance USDT-M perpetuals, real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source} (TradFi names removed)",
         "selected: " + ", ".join(selected),
         "costs: fee+slippage 0.20% round trip (stress 0.40%) plus REAL funding while open; max 8 open positions; one signal per reclaim bar",
         "Pass rule fixed before the run: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (SL assumed first), news not in data,",
         "the box / sweep / HTF definitions are one mechanical reading of a discretionary setup (a fail rejects this reading, not the trader)."]
    if len(selected) < 20:
        L.append("ABORTED: too few symbols could be downloaded; the download site may be blocked or changed.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_box_result.txt", "w", encoding="utf-8") as fh:
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
        for kind, s, data in ex.map(job4, tasks):
            cov[kind][1] += 1
            if data:
                cov[kind][0] += 1
                (kl if kind == "k4" else fu)[s].extend(data)
            done += 1
            if done % 1000 == 0:
                print(f"  downloaded {done}/{len(tasks)} files ({time.time() - t_run:.0f}s)", flush=True)
    L.append(f"files found/requested: 4h klines {cov['k4'][0]}/{cov['k4'][1]} | funding {cov['fu'][0]}/{cov['fu'][1]}")
    cands = {True: [], False: []}
    used = 0
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = build_frame4(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) < 600:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        for use_htf in (True, False):
            cands[use_htf] += [x for x in make_candidates(s, df, ft, fr, use_htf) if x["entry_t"] >= start_t]
    L.append(f"symbols with usable data: {used}")
    L.append("")
    sets = [("BOX SETUP - box + equal lows + sweep of a 20-day extreme + reclaim, long and short (eligible)", True, True),
            ("CONTROL - same setup WITHOUT the 20-day extreme condition (information only)", False, False)]
    taken_all = []
    for title, use_htf, eligible in sets:
        cap = B.portfolio(cands[use_htf])
        L.append(f"signals before the position cap: {len(cands[use_htf])} | taken: {len(cap)}")
        L += B.block(title, cap, eligible)
        if eligible:
            taken_all += cap
        if cap:
            fsm = np.array([x["funding_sum"] * (1 if x["side"] == "LONG" else -1) for x in cap])
            L.append(f"  average real funding cost per trade: {100 * fsm.mean():+.3f}% of position (negative = funding was income)")
            L.append("")
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_box_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_box_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "funding_sum", "touches"]
        w.writerow(cols)
        for x in taken_all:
            w.writerow([x[k] for k in cols])


# ---------------------------------------------------------------- self-tests (offline)
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


T0 = 1_650_000_000 // DAY * DAY


def crafted_long(spike=False):
    """decay to 102.5, a 16-bar box (lows 100.0 touched at bars 202/207/212, high 104), sweep+reclaim bar 216, rally to the target at 220."""
    n = 400
    o, h, l, c = (np.zeros(n) for _ in range(4))
    for j in range(200):
        c[j] = 150 - (47.5 / 199) * j
        o[j] = c[j - 1] if j else c[0] + 0.2
        h[j] = max(o[j], c[j]) + 0.5
        l[j] = min(o[j], c[j]) - 0.5
    for j in range(200, 216):
        o[j] = c[j] = 102.5
        h[j] = 104.0 if j == 204 else 103.6
        l[j] = 100.0 if j in (202, 207, 212) else 101.2
    o[216], h[216], l[216], c[216] = 102.5, 102.8, 98.5, 101.0
    o[217], h[217], l[217], c[217] = 101.2, 102.5, 100.8, 102.0
    o[218], h[218], l[218], c[218] = 102.0, 104.5, 101.8, 104.0
    o[219], h[219], l[219], c[219] = 104.0, 106.5, 103.8, 106.0
    o[220], h[220], l[220], c[220] = 106.0, 108.5, 105.8, 108.0
    for j in range(221, n):
        o[j] = c[j] = 108.0
        h[j], l[j] = 108.5, 107.5
    if spike:
        l[130] = 94.0
    df = pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": o, "high": h, "low": l, "close": c, "qv": 5e7})
    return df


def mirror(df):
    m = df.copy()
    m["open"], m["close"] = 200 - df["open"], 200 - df["close"]
    m["high"], m["low"] = 200 - df["low"], 200 - df["high"]
    return m


def fund_for(df, rate=0.0001):
    return np.array([float(x) for x in df["time"] + 3600]), np.full(len(df), rate)


def run_selftest():
    # parsing
    rows = [["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"],
            ["1667275200000", "1", "2", "0.5", "1.5", "10", "1667289599999", "15.5"], ["bad"]]
    check(parse_klines4(rows) == [(1667275200, 1.0, 2.0, 0.5, 1.5, 15.5)], "4h kline time is not floored to the day")
    df = build_frame4([(100, 1, 2, 0.5, 1.5, 1.0), (100, 1, 2, 0.5, 1.5, 1.0), (50, 1, 2, 0.5, 1.5, 1.0), (200, 0, 2, 0.5, 1.5, 1.0)])
    check(list(df["time"]) == [50, 100], "frame: duplicates removed, sorted, invalid rows dropped")
    check(non_crypto("XAUUSDT") and non_crypto("SOXLUSDT") and not non_crypto("BTCUSDT"), "TradFi names filtered")
    # touches and funding boundaries
    check(count_touches([1, 5, 5, 1, 5, 5, 1], 1.2, 1) == 3, "three distinct low touches")
    check(count_touches([1, 1, 1, 5, 5, 5, 5], 1.2, 1) == 1, "adjacent touches count once")
    check(count_touches([1, 5, 1, 5, 1], 1.2, 1) == 3 and count_touches([1, 1, 1, 1], 1.2, 1) == 1, "separated dips count, a flat bottom counts once")
    check(count_touches([9, 5, 5, 9, 5, 5, 9], 8.8, -1) == 3, "high touches for the short side")
    ft, fr = np.array([100.0, 200.0, TF + 100.0, 2 * TF]), np.array([1.0, 2.0, 4.0, 8.0])
    check(funding_sum4(ft, fr, 100.0, 0) == 2.0, "entry-time settlement excluded, rest of the entry bar counted")
    check(funding_sum4(ft, fr, 100.0, TF) == 14.0, "settlement exactly at the end of the exit bar is counted (conservative)")
    # higher time frame levels exclude the current day
    t = T0 + np.arange(30 * BPD) * TF
    hh, ll = np.full(len(t), 10.0), np.full(len(t), 5.0)
    ll[25 * BPD + 1] = 1.0
    lo20, hi20 = htf_levels(t, hh, ll)
    check(np.isnan(lo20[10 * BPD]), "no 20-day level before 20 completed days exist")
    check(lo20[25 * BPD + 2] == 5.0 and lo20[26 * BPD] == 1.0, "today's low is not in today's level, tomorrow's level includes it")
    # crafted long setup
    df = crafted_long()
    ft, fr = fund_for(df)
    got = F_cands(df, ft, fr, True)
    check(len(got) == 1, f"one long setup found, got {len(got)}: {[(x['side'], x['entry_t']) for x in got]}")
    x = got[0]
    check(x["side"] == "LONG" and x["entry_t"] == int(df["time"].iloc[217]) and x["touches"] == 3, "entry is the open after the reclaim bar")
    atr = indicators4(df)["atr"][216]
    entry, stop = df["open"].iloc[217], 98.5 - STOP_BUF * atr
    check(abs(x["risk_pct"] - (entry - stop) / entry) < 1e-9, "stop = sweep low - 0.25 ATR")
    check(x["reason"] == "TP" and abs(x["r_gross"] - 2.0) < 1e-9 and x["exit_t"] == int(df["time"].iloc[220]), f"target reached {x['reason']} {x['r_gross']}")
    check(abs(x["funding_sum"] - 4 * 0.0001) < 1e-12, f"funding of bars 217-220 charged: {x['funding_sum']}")
    expect = 2.0 - (COST_RT + x["funding_sum"]) / x["risk_pct"]
    check(abs(x["r_net"] - expect) < 1e-9 and x["r_stress"] < x["r_net"] < x["r_gross"], "net R = gross - (cost + funding) / risk")
    # mirror image gives the mirrored short with the same R
    dm = mirror(df)
    gm = F_cands(dm, ft, fr, True)
    check(len(gm) == 1 and gm[0]["side"] == "SHORT" and abs(gm[0]["r_gross"] - 2.0) < 1e-9 and gm[0]["entry_t"] == x["entry_t"], "mirror = short with the same R")
    check(abs(gm[0]["funding_sum"] + 0) >= 0 and gm[0]["touches"] == 3, "short touches counted on highs")
    # higher time frame condition
    sp = crafted_long(spike=True)
    check(F_cands(sp, ft, fr, True) == [] and len(F_cands(sp, ft, fr, False)) == 1, "20-day extreme condition separates the setup from the control")
    # sweep rules
    d2 = df.copy()
    d2.loc[216, "low"] = 90.0
    check(F_cands(d2, ft, fr, False) == [], "sweep deeper than 1.5 ATR is cancelled")
    d3 = df.copy()
    d3.loc[216, "low"] = 99.9
    check(F_cands(d3, ft, fr, False) == [], "wick shallower than 0.10 ATR is not a sweep")
    d4 = df.copy()
    d4.loc[216, "close"] = 99.0
    d4.loc[217, ["open", "high", "low", "close"]] = [99.2, 101.5, 98.9, 101.0]
    d4.loc[218, ["open", "high", "low", "close"]] = [101.0, 104.5, 100.8, 104.0]
    g4 = F_cands(d4, ft, fr, False)
    check(len(g4) == 1 and g4[0]["entry_t"] == int(d4["time"].iloc[218]), "late reclaim: the signal bar is the first close back above the box low")
    d5 = df.copy()
    d5.loc[216, ["open", "high", "low", "close"]] = [102.5, 105.0, 101.5, 104.8]
    check(F_cands(d5, ft, fr, False) == [], "no sweep + close above the box high cancels the setup (the flat top is one touch, not equal highs)")
    d6 = df.copy()
    d6.loc[202, "low"] = 101.2
    check(F_cands(d6, ft, fr, False) == [], "only two touches of the box low are not enough")
    d7 = df.copy()
    d7["qv"] = 1e6
    check(F_cands(d7, ft, fr, False) == [], "30-day volume filter")
    d8 = df.copy()
    d8.loc[205:, "time"] += TF
    check(F_cands(d8, ft, fr, False) == [], "a missing bar inside the box blocks the setup")
    global RISK_LO
    old = RISK_LO
    RISK_LO = 0.5
    try:
        check(F_cands(df, ft, fr, False) == [], "risk bounds applied")
    finally:
        RISK_LO = old
    # no look-ahead: results that finish before the cut are identical when the future is changed
    rng = np.random.default_rng(7)
    nb = 6000
    steps = rng.normal(0, 0.004, nb)
    cc = 100 * np.exp(np.cumsum(steps + 0.002 * np.sin(np.arange(nb) / 6.0)))
    oo = np.concatenate([[100.0], cc[:-1]])
    big = pd.DataFrame({"time": T0 + np.arange(nb) * TF, "open": oo, "high": np.maximum(oo, cc) * (1 + np.abs(rng.normal(0, 0.002, nb))),
                        "low": np.minimum(oo, cc) * (1 - np.abs(rng.normal(0, 0.002, nb))), "close": cc, "qv": 5e7})
    fb, rb = fund_for(big, 0.00005)
    full = F_cands(big, fb, rb, False)
    check(len(full) >= 3, f"random series produce setups ({len(full)})")
    cut = 4000
    fut = big.copy()
    fut.loc[cut:, ["open", "high", "low", "close"]] *= 3.0
    alt = F_cands(fut, fb, rb, False)
    tcut = int(big["time"].iloc[cut])
    a1 = {(x["entry_t"], x["side"]): x for x in full if x["exit_t"] < tcut}
    a2 = {(x["entry_t"], x["side"]): x for x in alt if x["exit_t"] < tcut}
    check(len(a1) >= 2 and a1.keys() == a2.keys(), "future prices do not change earlier trades")
    for k_, y in a1.items():
        z = a2[k_]
        check(abs(y["risk_pct"] - z["risk_pct"]) < 1e-12 and abs(y["r_gross"] - z["r_gross"]) < 1e-12 and y["exit_t"] == z["exit_t"], "identical trade data")
    # portfolio and report integration
    cap = B.portfolio(full)
    check(0 < len(cap) <= len(full), "portfolio keeps a subset")
    check(all(y["pair"] == "R" for y in cap), "pair name kept")
    text = "\n".join(B.block("test", cap, True))
    check("VERDICT" in text and "bootstrap" in text, "report block renders")
    print(f"ALL BOX SELF-TESTS PASSED ({PASSED} checks)")


def F_cands(df, ft, fr, use_htf):
    return make_candidates("R", df, ft, fr, use_htf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
