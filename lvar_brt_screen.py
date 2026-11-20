"""
BRT-S SCREEN BACKTEST - Binance USDT-M perpetuals, 4-hour bars, real funding. Hypotheses and thresholds fixed IN ADVANCE, nothing tuned.
This is a SCREEN to reject bad ideas quickly, NOT proof of an edge. The forward paper test stays the final judge.
Needs lvar_box_backtest.py, lvar_daily_backtest.py and lvar_fut_backtest.py in the repository (download, cost and report code are reused).

BASE = the break-and-retest rule of the live BRT-S bot (same level definition, same entry/stop/target/costs as lvar_sr_backtest.py mode B).
  Level = cluster of >= 3 pivots (PIV 5, LOOKBACK 180, TOL 0.25 ATR14); SHORT: close back below the level, bar high in [level-0.25 ATR, level+0.5 ATR],
  a close >= level within the last 10 bars (last such bar k <= i-2) and a close <= level-0.5 ATR after it; LONG is the exact mirror.
  Entry = next open, stop = level/bar extreme +- 0.25 ATR, target 2R, time stop = close of the 30th bar, risk 1%-12% of price, costs 0.20% + real funding.
FILTERS (each is a generic market-logic rule, thresholds fixed before the run):
  REG  market direction: SHORT only if the BTCUSDT 4h close of the signal bar is BELOW its EMA50; LONG only if ABOVE it (EMA50 of 4h closes, adjust=False,
       needs 200 BTC bars of history; unknown = rejected).
  EXT  no chasing: skip if side * (close[i] - close[i-6]) > 3.0 x ATR14[i]  (price already moved more than 3 ATR in the signal direction in the last 6 bars).
  CAP  stop distance: risk must be <= 6% of price (instead of 12%).
VARIANTS: BASE, REG, EXT, CAP, ALL (= REG + EXT + CAP). Five variants are tested: a pass of exactly one is weak evidence, never proof.
Universe, liquidity filter (30-day average dollar volume >= 20M), one position per coin, at most 8 open positions: identical to lvar_sr_backtest.py.
Pass rule (fixed before the run, applied to each variant): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
Rejected-trade statistics per filter are diagnostics only (not eligible for selection).
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
REG_EMA = 50
REG_MIN_BARS = 200
EXT_BARS = 6
EXT_ATR = 3.0
CAP_RISK = 0.06
BTC = "BTCUSDT"
VARIANTS = ("BASE", "REG", "EXT", "CAP", "ALL")


# ---------------------------------------------------------------- levels (same as lvar_sr_backtest.py)
def find_pivots(h, l):
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


def make_candidates(sym, df, ft, fr):
    """break-and-retest candidates (mode B of lvar_sr_backtest.py) with the extra fields used by the filters."""
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
            prev = c[i - FRESH:i]
            if side == 1:
                if not (c[i] > lvl and lvl - ZONE_BELOW * a <= l[i] <= lvl + tol):
                    continue
                dist = abs(l[i] - lvl)
                bad = np.nonzero(prev <= lvl)[0]
                if len(bad) == 0:
                    continue
                kk = i - FRESH + int(bad[-1])
                ok = kk <= i - 2 and c[kk + 1:i].max() >= lvl + BREAK_CLEAN * a
            else:
                if not (c[i] < lvl and lvl - tol <= h[i] <= lvl + ZONE_BELOW * a):
                    continue
                dist = abs(h[i] - lvl)
                bad = np.nonzero(prev >= lvl)[0]
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
        out.append({"pair": sym, "side": "LONG" if side == 1 else "SHORT", "sig_t": int(t[i]), "entry_t": int(t[k]),
                    "exit_t": int(t[res[1]]), "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY),
                    "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                    "r_net": float(res[0] - (COST_RT + side * fsum) / risk_pct),
                    "r_stress": float(res[0] - (COST_STRESS + side * fsum) / risk_pct), "strength": float(npv),
                    "funding_sum": fsum, "touches": int(npv),
                    "ext_atr": float(side * (c[i] - c[i - EXT_BARS]) / a)})
    return out


# ---------------------------------------------------------------- filters
def btc_regime(btc_df):
    """{bar_time: (close, ema)} for the BTC 4h frame; the EMA of bar t uses closes up to and including bar t only."""
    c = btc_df["close"].astype(float)
    ema = c.ewm(span=REG_EMA, adjust=False).mean()
    t = btc_df["time"].to_numpy(np.int64)
    cv, ev = c.to_numpy(float), ema.to_numpy(float)
    return {int(t[j]): (float(cv[j]), float(ev[j])) for j in range(REG_MIN_BARS, len(t))}


def reg_ok(x, reg):
    v = reg.get(x["sig_t"])
    if v is None:
        return False
    return v[0] < v[1] if x["side"] == "SHORT" else v[0] > v[1]


def ext_ok(x):
    return x["ext_atr"] <= EXT_ATR


def cap_ok(x):
    return x["risk_pct"] <= CAP_RISK


def keep(variant, x, reg):
    if variant == "BASE":
        return True
    if variant == "REG":
        return reg_ok(x, reg)
    if variant == "EXT":
        return ext_ok(x)
    if variant == "CAP":
        return cap_ok(x)
    return reg_ok(x, reg) and ext_ok(x) and cap_ok(x)


def summary_row(name, rows):
    if not rows:
        return f"{name:6s} n=0"
    rows = sorted(rows, key=lambda x: x["entry_t"])
    mid_t = rows[len(rows) // 2]["entry_t"]
    sn = B.stats([x["r_net"] for x in rows])
    sa = B.stats([x["r_net"] for x in rows if x["entry_t"] < mid_t])
    sb = B.stats([x["r_net"] for x in rows if x["entry_t"] >= mid_t])
    ok = bool(sn and sa and sb and sn["avgR"] >= 0.10 and sn["PF"] >= 1.2 and sn["n"] >= 300 and sa["avgR"] > 0 and sb["avgR"] > 0)
    return (f"{name:6s} n={sn['n']:5d} avgR={sn['avgR']:+.3f} PF={sn['PF']:.2f} halves={sa['avgR']:+.3f}/{sb['avgR']:+.3f} "
            f"LONG={B.stats([x['r_net'] for x in rows if x['side'] == 'LONG'])['avgR'] if any(x['side'] == 'LONG' for x in rows) else float('nan'):+.3f} "
            f"SHORT={B.stats([x['r_net'] for x in rows if x['side'] == 'SHORT'])['avgR'] if any(x['side'] == 'SHORT' for x in rows) else float('nan'):+.3f} "
            f"{'PASS' if ok else 'FAIL'}")


def build_report(cands, reg):
    """returns (lines, taken_by_variant). cands = all BASE candidates before the position cap."""
    L = []
    taken = {}
    for v in VARIANTS:
        taken[v] = B.portfolio([x for x in cands if keep(v, x, reg)])
    L.append("SUMMARY (net R after 0.20% costs + real funding, one position per coin, max 8 open)")
    for v in VARIANTS:
        L.append("  " + summary_row(v, taken[v]))
    L.append("")
    L.append("DIAGNOSTIC - trades REJECTED by each filter (before the position cap; not eligible for selection)")
    for name, pred in (("REG", lambda x: not reg_ok(x, reg)), ("EXT", lambda x: not ext_ok(x)), ("CAP", lambda x: not cap_ok(x))):
        rej = [x["r_net"] for x in cands if pred(x)]
        L.append(B.fmt(f"  rejected by {name} (net)", rej))
        for side in ("LONG", "SHORT"):
            L.append(B.fmt(f"    {side} rejected by {name}", [x["r_net"] for x in cands if x["side"] == side and pred(x)]))
    L.append(B.fmt("  all candidates BASE (net)", [x["r_net"] for x in cands]))
    L.append("")
    for v in VARIANTS:
        L += B.block(f"{v} - break and retest, filter set {v} (eligible)", taken[v], True)
    return L, taken


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
    if BTC not in selected and BTC in symbols:
        selected = list(selected) + [BTC]
    print(f"selected {len(selected)} symbols", flush=True)
    L = ["BRT-S SCREEN BACKTEST - break+retest with market-direction / no-chasing / stop-cap filters, 4h bars, Binance USDT-M perpetuals, real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source} (TradFi names removed)",
         "selected: " + ", ".join(selected),
         "costs: fee+slippage 0.20% round trip plus REAL funding while open; max 8 open positions",
         f"filters (fixed in advance): REG = BTC 4h close vs EMA{REG_EMA} (short below, long above); EXT = move > {EXT_ATR} ATR in {EXT_BARS} bars in signal direction skipped; CAP = risk <= {100 * CAP_RISK:.0f}%",
         "Pass rule fixed before the run, applied to each variant: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "This is a SCREEN: five variants are tested, a pass of exactly one is weak evidence; the forward paper test stays the final judge.",
         "Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (SL assumed first), news not in data."]
    if len(selected)
