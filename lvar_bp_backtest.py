"""
BLUEPRINT BACKTEST ("Institutional LVAR / value rejection": liquidity sweep + volume absorption, BTC 4H macro gate, news gate)
Binance USDT-M perpetuals, 15-minute bars. Rules fixed IN ADVANCE as numbers, nothing tuned afterwards (read-only research).
Reuses the download / report code of lvar_fut_backtest.py, lvar_box_backtest.py and lvar_daily_backtest.py (all must stay in the repository).

Universe: the 70 most liquid USDT-M perpetuals of the download site (delisted contracts included, TradFi names removed); a coin is tradable
  only if its 30-day average dollar volume is >= 20M USDT. BTCUSDT is always downloaded for the gates. Period: 2022-11-01 .. end of last full month.

Signal (15m bar i, evaluated when the bar is CLOSED):
  1. MACRO GATE: BTC 4H (last COMPLETED 4h bar): EMA20 > EMA50 and close > EMA50 = BULLISH (longs only); EMA20 < EMA50 and close < EMA50 =
     BEARISH (shorts only); anything else = NEUTRAL = no trade.
  2. NEWS GATE: no trade while BTC shows a 15m range >= 3 x ATR14 in any of the last 4 bars, or a move >= 2.5% over the last 4 bars,
     or BTC data is missing (a scheduled-event blackout file cannot be reproduced historically).
  3. SWEEP: SHORT: high of bar i > highest high of the previous 96 bars, close back below that level, bearish bar, close in the lower 40% of
     its range. LONG is the exact mirror.
  4. ABSORPTION: bar volume >= 2.5 x the average volume of the previous 20 bars.
  Entry = open of the next bar (must be within 1.5% of the signal close).
Order sets:
  V2 (eligible, the optimised rule): stop = the sweep extreme (signal bar high / low) +/- 0.25 x ATR14; risk must be 0.6%..4% of price;
     TP1 = 2.5R, TP2 = 4.5R. As the bot works: after TP1 the stop moves to break-even and the WHOLE position runs to TP2 (no partial close).
  FIXED (information only): the original blueprint, stop 1.2%, TP1 4%, TP2 7%, same signals.
  V2P (information only): V2 with half of the position closed at TP1 and the other half running to TP2 with a break-even stop.
Common: one position per coin, at most 8 open positions (stronger volume wins a tie), SL before TP inside a bar, a bar that reaches TP1 cannot also
  count as TP2, gaps against the stop fill at the open, time stop after 960 bars (10 days). Costs: fee + slippage 0.20% round trip (stress 0.40%)
  plus the REAL funding paid or received while the position was open (long pays positive funding, short receives it).
Pass rule (fixed before the run, V2 only): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
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
import lvar_box_backtest as X

DAY = B.DAY
TF = 900
H4 = 14400
STRUCT = 96
VOLAVG = 20
VOL_MULT = 2.5
CLOSE_FRAC = 0.40
ATR_N = 14
STOP_BUF = 0.25
RISK_LO, RISK_HI = 0.006, 0.04
TP1_R, TP2_R = 2.5, 4.5
HOLD = 960
FIX_SL, FIX_TP1, FIX_TP2 = 0.012, 0.04, 0.07
MAX_DEV = 0.015
MIN_VOL30 = 20_000_000
VOL30_BARS = 30 * 96
EMA_F, EMA_S = 20, 50
NEWS_RANGE_ATR = 3.0
NEWS_HOUR_MOVE = 0.025
NEWS_BARS = 4
COST_RT = X.COST_RT
COST_STRESS = X.COST_STRESS

VARIANTS = {"V2": {"mode": "atr", "partial": False},
            "V2P": {"mode": "atr", "partial": True},
            "FIXED": {"mode": "fixed", "partial": False}}


# ---------------------------------------------------------------- BTC context (macro gate + news gate)
def _atr(h, l, c):
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean().to_numpy(float)


def btc_context(df):
    """macro bias per completed 4h bar and a 'trading blocked by the news gate' flag per BTC 15m bar."""
    t = df["time"].to_numpy(np.int64)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    b = t // H4
    g = pd.DataFrame({"b": b, "c": c}).groupby("b").agg(close=("c", "last"), n=("c", "size"))
    g = g[g["n"] == 16]
    if len(g) < EMA_S + 5:
        return None
    full = g["close"].reindex(np.arange(int(g.index.min()), int(g.index.max()) + 1))
    cl = full.ffill()
    fast = cl.ewm(span=EMA_F, adjust=False, min_periods=EMA_F).mean()
    slow = cl.ewm(span=EMA_S, adjust=False, min_periods=EMA_S).mean()
    bias = np.where(fast.isna() | slow.isna() | full.isna(), np.nan,
                    np.where((fast > slow) & (cl > slow), 1.0, np.where((fast < slow) & (cl < slow), -1.0, 0.0)))
    a = _atr(h, l, c)
    ref = np.concatenate([[np.nan], a[:-1]])
    with np.errstate(invalid="ignore"):
        spike = np.isfinite(ref) & (ref > 0) & ((h - l) >= NEWS_RANGE_ATR * ref)
    n = len(t)
    blk = np.zeros(n, bool)
    for j in range(n):
        lo = j - NEWS_BARS + 1
        if lo < 0:
            blk[j] = True
            continue
        if t[j] - t[lo] != (NEWS_BARS - 1) * TF:
            blk[j] = True                       # a data gap inside the window: low confirmation
            continue
        if spike[lo:j + 1].any() or abs(c[j] / o[lo] - 1.0) >= NEWS_HOUR_MOVE:
            blk[j] = True
    blk[:30] = True
    return {"b0": int(g.index.min()), "bias": bias, "bt": t, "blocked": blk}


def align_context(t, ctx):
    """bias (+1 / -1 / 0 / nan) and news-blocked flag for every bar open time in t (bar i is known at t[i] + 900)."""
    t = np.asarray(t, np.int64)
    pos = np.searchsorted(ctx["bt"], t)
    pos_c = np.minimum(pos, len(ctx["bt"]) - 1)
    ok = (pos < len(ctx["bt"])) & (ctx["bt"][pos_c] == t)
    blocked = np.where(ok, ctx["blocked"][pos_c], True)
    bidx = (t + TF) // H4 - 1 - ctx["b0"]
    valid = (bidx >= 0) & (bidx < len(ctx["bias"]))
    bias = np.where(valid, ctx["bias"][np.clip(bidx, 0, len(ctx["bias"]) - 1)], np.nan)
    return bias, blocked


# ---------------------------------------------------------------- simulation
def sim_bp(o, h, l, c, k, side, entry, stop, tp1, tp2, hold):
    """-> (r_full, r_partial, exit_bar, reason) or None. side +1 long, -1 short. Stage 1 = stop active, stage 2 = after TP1 (stop at break-even)."""
    n = len(c)
    risk = side * (entry - stop)
    if risk <= 0:
        return None
    r1 = side * (tp1 - entry) / risk
    end = k + hold - 1
    stage, cur = 1, stop
    for m in range(k, min(n - 1, end) + 1):
        ae = l[m] if side == 1 else h[m]       # adverse extreme
        fe = h[m] if side == 1 else l[m]       # favourable extreme
        if stage == 1:
            if m > k and side * (o[m] - cur) <= 0:
                r = side * (o[m] - entry) / risk
                return r, r, m, "SL"
            if side * (ae - cur) <= 0:
                return -1.0, -1.0, m, "SL"
            if side * (fe - tp1) >= 0:
                stage, cur = 2, entry
            elif m == end:
                r = side * (c[m] - entry) / risk
                return r, r, m, "TIME"
        else:
            if side * (o[m] - cur) <= 0:
                leg = side * (o[m] - entry) / risk
                return leg, 0.5 * r1 + 0.5 * leg, m, "BE"
            if side * (ae - cur) <= 0:
                return 0.0, 0.5 * r1, m, "BE"
            if side * (fe - tp2) >= 0:
                r2 = side * (tp2 - entry) / risk
                return r2, 0.5 * r1 + 0.5 * r2, m, "TP2"
            if m == end:
                leg = side * (c[m] - entry) / risk
                return leg, 0.5 * r1 + 0.5 * leg, m, "TIME2"
    return None


def funding_sum15(ft, fr, t_entry, t_exit_bar):
    lo = int(np.searchsorted(ft, t_entry, side="right"))
    hi = int(np.searchsorted(ft, t_exit_bar + TF, side="right"))
    return float(fr[lo:hi].sum())


# ---------------------------------------------------------------- signals
def signal_list(df, bias, blocked):
    """[(bar_index, side, strength)] for every closed 15m bar that satisfies the sweep + absorption + gates."""
    t = df["time"].to_numpy(np.int64)
    o, h, l, c, qv = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "qv"))
    n = len(df)
    if n < VOL30_BARS + STRUCT + 5:
        return [], None
    a = _atr(h, l, c)
    s_qv = pd.Series(qv)
    avg = s_qv.rolling(VOLAVG).mean().shift(1).to_numpy(float)
    sh = pd.Series(h).rolling(STRUCT).max().shift(1).to_numpy(float)
    sl = pd.Series(l).rolling(STRUCT).min().shift(1).to_numpy(float)
    vol30 = (s_qv.rolling(VOL30_BARS, min_periods=VOL30_BARS).mean() * 96).to_numpy(float)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(rng > 0, (c - l) / rng, np.nan)
        cont = np.concatenate([np.zeros(STRUCT, bool), (t[STRUCT:] - t[:-STRUCT]) == STRUCT * TF])
        base = (~np.asarray(blocked, bool)) & cont & np.isfinite(a) & (vol30 >= MIN_VOL30) & (avg > 0) & (qv >= VOL_MULT * avg)
        short = base & (bias == -1) & (h > sh) & (c < sh) & (c < o) & (pos <= CLOSE_FRAC)
        long_ = base & (bias == 1) & (l < sl) & (c > sl) & (c > o) & (pos >= 1 - CLOSE_FRAC)
        strength = qv / avg
    out = [(int(i), -1, float(strength[i])) for i in np.nonzero(short)[0]] + [(int(i), 1, float(strength[i])) for i in np.nonzero(long_)[0]]
    return sorted(out), a


def build(sym, df, ft, fr, sigs, a, variant):
    t = df["time"].to_numpy(np.int64)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    spec = VARIANTS[variant]
    out = []
    for i, side, strength in sigs:
        k = i + 1
        if k >= n or t[k] - t[i] != TF:
            continue
        entry = o[k]
        if not (entry > 0) or abs(entry / c[i] - 1.0) > MAX_DEV:
            continue
        if spec["mode"] == "atr":
            stop = h[i] + STOP_BUF * a[i] if side == -1 else l[i] - STOP_BUF * a[i]
            risk = side * (entry - stop)
            rp = risk / entry
            if risk <= 0 or not (RISK_LO <= rp <= RISK_HI):
                continue
            tp1, tp2 = entry + side * TP1_R * risk, entry + side * TP2_R * risk
        else:
            stop = entry * (1 - side * FIX_SL)
            risk = side * (entry - stop)
            rp = risk / entry
            tp1, tp2 = entry * (1 + side * FIX_TP1), entry * (1 + side * FIX_TP2)
        if min(stop, tp1, tp2) <= 0:
            continue
        res = sim_bp(o, h, l, c, k, side, entry, stop, tp1, tp2, HOLD)
        if res is None:
            continue
        r = res[1] if spec["partial"] else res[0]
        fsum = funding_sum15(ft, fr, t[k], t[res[2]])
        out.append({"pair": sym, "side": "LONG" if side == 1 else "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[2]]), "reason": res[3],
                    "days": float((res[2] - k + 1) * TF / DAY), "risk_pct": float(rp), "r_gross": float(r),
                    "r_net": float(r - (COST_RT + side * fsum) / rp), "r_stress": float(r - (COST_STRESS + side * fsum) / rp),
                    "strength": float(strength), "funding_sum": fsum})
    return out


def make_candidates(sym, df, ft, fr, ctx, variants=("V2", "V2P", "FIXED")):
    bias, blocked = align_context(df["time"].to_numpy(np.int64), ctx)
    sigs, a = signal_list(df, bias, blocked)
    return {v: build(sym, df, ft, fr, sigs, a, v) for v in variants}, len(sigs)


# ---------------------------------------------------------------- main run
def frame15(arrs):
    d = np.concatenate(arrs) if arrs else np.zeros((0, 6))
    df = pd.DataFrame(d, columns=["time", "open", "high", "low", "close", "qv"])
    df["time"] = df["time"].astype(np.int64)
    df = df.drop_duplicates("time").sort_values("time")
    return df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.qv >= 0)].reset_index(drop=True)


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
    selected, ranked = F.rank_symbols(symbols)
    print(f"selected {len(selected)} symbols: {', '.join(selected)}", flush=True)
    L = ["BLUEPRINT BACKTEST - LVAR value rejection (liquidity sweep + volume absorption), 15m bars, Binance USDT-M perpetuals, real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source} (TradFi names removed); BTCUSDT always loaded for the gates",
         "selected: " + ", ".join(selected),
         "costs: fee+slippage 0.20% round trip (stress 0.40%) plus REAL funding while open; max 8 open positions; one position per coin",
         "Pass rule fixed before the run (V2 only): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Known problems: top-70 liquid contracts, entry at the next bar open, 15m bars hide intrabar order (SL assumed first), the scheduled-event",
         "blackout file is not reproducible in history (only the BTC volatility gate is), Binance data instead of KuCoin spot."]
    if len(selected) < 20:
        L.append("ABORTED: too few symbols could be downloaded; the download site may be blocked or changed.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_bp_result.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return
    load = list(selected) + (["BTCUSDT"] if "BTCUSDT" not in selected else [])
    tasks = []
    for s in load:
        for m in dl_months:
            tasks.append(("k4", s, f"{F.BASE}/monthly/klines/{s}/15m/{s}-15m-{m}.zip"))
            tasks.append(("fu", s, f"{F.BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"))
    kl, fu = {s: [] for s in load}, {s: [] for s in load}
    cov = {"k4": [0, 0], "fu": [0, 0]}
    done = 0
    with ThreadPoolExecutor(F.WORKERS) as ex:
        for kind, s, data in ex.map(X.job4, tasks):
            cov[kind][1] += 1
            if data:
                cov[kind][0] += 1
                if kind == "k4":
                    kl[s].append(np.array(data, float))
                else:
                    fu[s].extend(data)
            done += 1
            if done % 1000 == 0:
                print(f"  downloaded {done}/{len(tasks)} files ({time.time() - t_run:.0f}s)", flush=True)
    L.append(f"files found/requested: 15m klines {cov['k4'][0]}/{cov['k4'][1]} | funding {cov['fu'][0]}/{cov['fu'][1]}")
    btc = frame15(kl.get("BTCUSDT", []))
    btc = btc[btc["time"] < end_t].reset_index(drop=True)
    ctx = btc_context(btc) if len(btc) > 1000 else None
    if ctx is None:
        L.append("ABORTED: BTCUSDT 15m data is missing, the macro gate cannot be built.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_bp_result.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return
    gate_share = float(np.mean(ctx["blocked"]))
    bias_v = ctx["bias"][np.isfinite(ctx["bias"])]
    L.append(f"BTC gates: news gate closed on {100 * gate_share:.1f}% of bars | macro bias BULLISH {100 * np.mean(bias_v == 1):.0f}% / BEARISH {100 * np.mean(bias_v == -1):.0f}% / NEUTRAL {100 * np.mean(bias_v == 0):.0f}% of 4h bars")
    cands = {v: [] for v in VARIANTS}
    used, nsig = 0, 0
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = frame15(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) < VOL30_BARS + 500:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        got, ns = make_candidates(s, df, ft, fr, ctx)
        nsig += ns
        for v in VARIANTS:
            cands[v] += [x for x in got[v] if x["entry_t"] >= start_t]
        kl[s] = None
    L.append(f"symbols with usable data: {used} | raw signals (gates passed, before order filters): {nsig}")
    L.append("")
    sets = [("V2 - sweep + absorption, ATR-based stop beyond the sweep, TP1 2.5R / TP2 4.5R, full position (eligible)", "V2", True),
            ("V2P - same, half of the position closed at TP1 (information only)", "V2P", False),
            ("FIXED - original blueprint: stop 1.2%, TP1 4%, TP2 7%, same signals (information only)", "FIXED", False)]
    taken = []
    for title, v, eligible in sets:
        cap = B.portfolio(cands[v])
        L.append(f"[{v}] signals before the position cap: {len(cands[v])} | taken: {len(cap)}")
        L += B.block(title, cap, eligible)
        if eligible:
            taken = cap
        if cap:
            fsm = np.array([x["funding_sum"] * (1 if x["side"] == "LONG" else -1) for x in cap])
            L.append(f"  average real funding cost per trade: {100 * fsm.mean():+.3f}% of position (negative = funding was income)")
            L.append("")
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_bp_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_bp_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "funding_sum", "strength"]
        w.writerow(cols)
        for x in taken:
            w.writerow([x[k] for k in cols])


# ---------------------------------------------------------------- self-tests (offline)
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


T0 = 1_650_000_000 // DAY * DAY


def naive_signals(df, bias, blocked):
    """independent loop implementation of the signal rule."""
    t = df["time"].to_numpy(np.int64)
    o, h, l, c, qv = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "qv"))
    a = _atr(h, l, c)
    out = set()
    for i in range(STRUCT, len(df)):
        if blocked[i] or not np.isfinite(a[i]) or t[i] - t[i - STRUCT] != STRUCT * TF or i < VOL30_BARS - 1:
            continue
        if qv[i - VOL30_BARS + 1:i + 1].mean() * 96 < MIN_VOL30:
            continue
        avg = qv[i - VOLAVG:i].mean()
        if not (avg > 0 and qv[i] >= VOL_MULT * avg) or h[i] == l[i]:
            continue
        hi, lo = h[i - STRUCT:i].max(), l[i - STRUCT:i].min()
        p = (c[i] - l[i]) / (h[i] - l[i])
        if bias[i] == -1 and h[i] > hi and c[i] < hi and c[i] < o[i] and p <= CLOSE_FRAC:
            out.add((i, -1))
        if bias[i] == 1 and l[i] < lo and c[i] > lo and c[i] > o[i] and p >= 1 - CLOSE_FRAC:
            out.add((i, 1))
    return out


def fat_frame(seed, n=9000, qv0=3e5):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(3, n) * 0.004
    cl = 100 * np.exp(np.cumsum(r))
    op = np.concatenate([[100.0], cl[:-1]])
    hi = np.maximum(op, cl) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(op, cl) * (1 - np.abs(rng.normal(0, 0.002, n)))
    qv = qv0 * np.exp(rng.normal(0, 0.9, n))
    return pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": op, "high": hi, "low": lo, "close": cl, "qv": qv})


def btc_frame(n, close_fn, rng_=0.2):
    t = np.arange(n) * TF + (T0 // H4) * H4
    c = np.array([close_fn(i) for i in range(n)], float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"time": t, "open": o, "high": np.maximum(o, c) + rng_ / 2, "low": np.minimum(o, c) - rng_ / 2, "close": c, "qv": 1e6})


def run_selftest():
    # --- simulator
    def bars(rows):
        a = np.array(rows, float)
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    # SHORT entry 100, stop 102 (risk 2), tp1 95 (2.5R), tp2 91 (4.5R)
    S = dict(side=-1, entry=100.0, stop=102.0, tp1=95.0, tp2=91.0, hold=50)

    def go(rows):
        o, h, l, c = bars(rows)
        return sim_bp(o, h, l, c, 0, S["side"], S["entry"], S["stop"], S["tp1"], S["tp2"], S["hold"])
    pad = [(100, 100.5, 99.5, 100)] * 3
    r = go([(100, 100.5, 99.5, 100), (100, 102.5, 99, 101)] + pad)
    check(r[3] == "SL" and r[0] == -1.0 and r[1] == -1.0 and r[2] == 1, f"stop hit = -1R: {r}")
    r = go([(100, 100.5, 99.5, 100), (103.0, 103.5, 100, 101)] + pad)
    check(r[3] == "SL" and abs(r[0] - (-1.5)) < 1e-12, f"gap through the stop fills at the open (-1.5R): {r}")
    r = go([(100, 100.5, 94.0, 96), (96, 102.5, 95, 101)] + pad)
    check(r[3] == "BE" and r[0] == 0.0 and abs(r[1] - 1.25) < 1e-12, f"TP1 then break-even: full 0R, partial +1.25R: {r}")
    r = go([(100, 100.5, 90.0, 92), (92, 93, 90.5, 91)] + pad)
    check(r[3] == "TP2" and r[2] == 1, f"a bar reaching TP1 and TP2 together counts only as TP1; TP2 exits one bar later at the earliest: {r}")
    r = go([(100, 100.5, 94.0, 96), (96, 97, 90.5, 91)] + pad)
    check(r[3] == "TP2" and abs(r[0] - 4.5) < 1e-12 and abs(r[1] - (1.25 + 2.25)) < 1e-12, f"TP1 then TP2: full 4.5R, partial 3.5R: {r}")
    r = go([(100, 100.5, 94.0, 96), (96, 100.2, 95, 99)] + pad)
    check(r[3] == "BE" and r[0] == 0.0, f"touching exactly the break-even level exits: {r}")
    S2 = dict(S, hold=3)
    o, h, l, c = bars([(100, 100.5, 99.5, 99.0)] * 6)
    r = sim_bp(o, h, l, c, 0, -1, 100.0, 102.0, 95.0, 91.0, 3)
    check(r[3] == "TIME" and r[2] == 2 and abs(r[0] - 0.5) < 1e-12, f"time stop at the close of the 3rd bar: {r}")
    check(sim_bp(o, h, l, c, 4, -1, 100.0, 102.0, 95.0, 91.0, 30) is None, "series ends before an exit = no trade")
    # LONG mirror
    o, h, l, c = bars([(100, 100.5, 99.5, 100), (100, 106.0, 99, 105), (105, 111.0, 104, 110)] + pad)
    r = sim_bp(o, h, l, c, 0, 1, 100.0, 98.0, 105.0, 109.0, 50)
    check(r[3] == "TP2" and r[2] == 2 and abs(r[0] - 4.5) < 1e-12, f"LONG mirror: {r}")

    # --- BTC context: macro bias by completed 4h bar, news gate
    n = 16 * 120
    bear = btc_frame(n, lambda i: 100 - 0.02 * i)
    cx = btc_context(bear)
    bias, blk = align_context(bear["time"].to_numpy(np.int64), cx)
    check(bias[-1] == -1 and np.nanmax(bias[16 * 70:]) == -1, "falling BTC = BEARISH")
    bull = btc_frame(n, lambda i: 100 + 0.02 * i)
    b2, _ = align_context(bull["time"].to_numpy(np.int64), btc_context(bull))
    check(b2[-1] == 1, "rising BTC = BULLISH")
    # no look-ahead in the bias: a bar uses only 4h bars that ended at or before its close
    tb = bear["time"].to_numpy(np.int64)
    j = 16 * 100 + 5
    cut = bear.iloc[:j + 1].copy()
    b3, _ = align_context(cut["time"].to_numpy(np.int64), btc_context(cut))
    check(b3[j] == bias[j] and np.array_equal(b3[:j + 1][np.isfinite(b3[:j + 1])], bias[:j + 1][np.isfinite(b3[:j + 1])]), "bias at a bar does not depend on later data")
    flip = btc_frame(n, lambda i: 100 + 0.02 * i if i < 16 * 90 else 100 + 0.02 * 16 * 90 - 0.2 * (i - 16 * 90))
    bf, _ = align_context(flip["time"].to_numpy(np.int64), btc_context(flip))
    check(bf[16 * 90 - 1] == 1 and bf[-1] == -1, "bias flips after the trend turns")
    # the bias is read from the last COMPLETED 4h bar: the bar at the start of a new 4h block still uses the previous block
    k0 = 16 * 100
    check(bias[k0] == bias[k0 - 1] or np.isfinite(bias[k0]), "first bar of a 4h block uses the finished block")
    # news gate
    calm = btc_frame(n, lambda i: 100.0)
    _, bl = align_context(calm["time"].to_numpy(np.int64), btc_context(calm))
    check(not bl[500:].any(), "calm BTC: gate open")
    sp = calm.copy()
    sp.loc[1000, ["high", "low"]] = [105.0, 99.0]
    _, bl = align_context(sp["time"].to_numpy(np.int64), btc_context(sp))
    check(bl[1000] and bl[1003] and not bl[1004] and not bl[999], "a 3 ATR BTC bar closes the gate for that bar and the next 3")
    mv = btc_frame(n, lambda i: 100.0 if i < 1000 else 100.0 + (i - 999) * 0.9)
    _, bl = align_context(mv["time"].to_numpy(np.int64), btc_context(mv))
    check(bl[1003], "a >= 2.5% move over 4 bars closes the gate")
    gap = calm.drop(index=range(1000, 1002)).reset_index(drop=True)
    tg = gap["time"].to_numpy(np.int64)
    _, bl = align_context(tg, btc_context(gap))
    check(bl[1000], "a gap inside the 4-bar window closes the gate")
    missing = np.array([calm["time"].iloc[1001]], np.int64)
    _, bl = align_context(missing, btc_context(gap))
    check(bl[0], "a coin bar with no BTC bar = blocked")

    # --- signal rule against the independent loop implementation (random + crafted)
    rng = np.random.default_rng(5)
    total = 0
    for sd in range(1, 6):
        df = fat_frame(sd)
        bias_r = rng.choice([-1.0, 0.0, 1.0, np.nan], len(df), p=[0.4, 0.1, 0.4, 0.1])
        blk_r = rng.random(len(df)) < 0.1
        got = {(i, s) for i, s, _ in signal_list(df, bias_r, blk_r)[0]}
        ref = naive_signals(df, bias_r, blk_r)
        check(got == ref, f"seed {sd}: vectorised signals equal the loop implementation ({len(got)} vs {len(ref)})")
        total += len(got)
    check(total >= 20, f"random data produce signals ({total})")
    # crafted short
    d = fat_frame(9, n=4000, qv0=5e6)
    d.loc[:, ["open", "high", "low", "close"]] = 100.0
    d["high"] += 0.2
    d["low"] -= 0.2
    d["qv"] = 5e6
    i = 3500
    d.loc[d.index[i], ["open", "high", "low", "close", "qv"]] = [100.0, 101.0, 99.0, 99.2, 5e6 * 3]
    bias_s = np.full(len(d), -1.0)
    blk0 = np.zeros(len(d), bool)
    sg, a = signal_list(d, bias_s, blk0)
    check(sg == [(i, -1, 3.0)], f"crafted sweep-and-reject short: {sg}")
    for label, mod in (("volume below 2.5x", dict(qv=5e6 * 2.4)), ("close not back below the level", dict(close=100.9)),
                       ("close in the upper part of the bar", dict(close=100.3, open=100.5)), ("bullish bar", dict(open=99.0, close=99.2))):
        e = d.copy()
        for k_, v_ in mod.items():
            e.loc[e.index[i], k_] = v_
        check(signal_list(e, bias_s, blk0)[0] == [], f"no signal: {label}")
    check(signal_list(d, np.full(len(d), 1.0), blk0)[0] == [], "wrong macro bias = no signal")
    check(signal_list(d, np.full(len(d), 0.0), blk0)[0] == [], "NEUTRAL bias = no signal")
    bb = blk0.copy()
    bb[i] = True
    check(signal_list(d, bias_s, bb)[0] == [], "news gate closed = no signal")
    check(signal_list(d.assign(qv=d["qv"] * 0.01), bias_s, blk0)[0] == [], "30-day volume below 20M = no signal")
    d_gap = d.copy()
    d_gap.loc[d_gap.index[3460]:, "time"] += TF
    check(signal_list(d_gap, bias_s, blk0)[0] == [], "a missing bar inside the 96-bar window blocks the signal")
    # order math, both variants
    ft, fr = np.array([0.0]), np.array([0.0])
    d.loc[d.index[i + 1], ["open", "high", "low", "close"]] = [99.0, 99.2, 90.0, 91.0]
    rows = build("X", d, ft, fr, sg, a, "V2")
    check(len(rows) == 1, f"V2 builds one trade: {rows}")
    stop = 101.0 + STOP_BUF * a[i]
    risk = stop - 99.0
    x = rows[0]
    check(abs(x["risk_pct"] - risk / 99.0) < 1e-12 and 0.006 <= x["risk_pct"] <= 0.04, "V2 stop = signal-bar high + 0.25 ATR")
    check(x["entry_t"] == int(d["time"].iloc[i + 1]) and x["reason"] in ("BE", "TP2", "SL", "TIME", "TIME2"), "entry at the next bar open")
    expect_tp1_hit = (99.0 - 2.5 * risk) >= 90.0
    check(expect_tp1_hit, "test bar reaches TP1")
    check(abs(x["r_net"] - (x["r_gross"] - COST_RT / x["risk_pct"])) < 1e-12 and x["r_stress"] < x["r_net"], "net = gross - cost / risk")
    xf = build("X", d, ft, fr, sg, a, "FIXED")[0]
    check(abs(xf["risk_pct"] - FIX_SL) < 1e-12, "FIXED stop = 1.2%")
    check(abs(xf["r_gross"] - (FIX_TP2 / FIX_SL if xf["reason"] == "TP2" else xf["r_gross"])) < 1e-9 and xf["reason"] in ("TP2", "BE"), f"FIXED: 1.2% stop, TP2 at 7% = 5.83R when reached: {xf}")
    # risk bounds: a huge wick is too wide, a tiny bar is too tight
    wide = d.copy()
    wide.loc[wide.index[i], "high"] = 110.0
    sgw, aw = signal_list(wide, bias_s, blk0)
    check(len(sgw) == 1 and build("X", wide, ft, fr, sgw, aw, "V2") == [], "risk above 4% is skipped")
    check(build("X", d, ft, fr, [(i, -1, 3.0)], a, "V2") != [] and RISK_LO > 0.0005, "risk lower bound constant sanity")
    narrow = a.copy()
    narrow[i] = 0.0
    d_n = d.copy()
    d_n.loc[d_n.index[i], "high"] = 99.4
    check(build("X", d_n, ft, fr, [(i, -1, 3.0)], narrow, "V2") == [], "risk below 0.6% (or non-positive) is skipped")
    far = d.copy()
    far.loc[far.index[i + 1], "open"] = 104.0
    check(build("X", far, ft, fr, sg, a, "V2") == [], "entry open more than 1.5% away from the signal close is skipped")
    # funding: short receives positive funding
    ft2, fr2 = np.array([float(d["time"].iloc[i + 1]) + 60.0]), np.array([0.001])
    xs = build("X", d, ft2, fr2, sg, a, "V2")[0]
    check(abs(xs["funding_sum"] - 0.001) < 1e-12 and xs["r_net"] > build("X", d, ft, fr, sg, a, "V2")[0]["r_net"], "a short receives positive funding")

    # --- no look-ahead: changing the future does not change earlier trades
    df = fat_frame(3, n=12000, qv0=1e6)
    bias_r = np.where(np.arange(len(df)) % 2000 < 1000, -1.0, 1.0)
    blk_r = np.zeros(len(df), bool)
    ctx_like = None
    sg1, a1 = signal_list(df, bias_r, blk_r)
    full = build("X", df, ft, fr, sg1, a1, "V2")
    cut = 9000
    fut = df.copy()
    fut.loc[cut:, ["open", "high", "low", "close"]] *= 3.0
    fut.loc[cut:, "qv"] *= 10.0
    sg2, a2 = signal_list(fut, bias_r, blk_r)
    alt = build("X", fut, ft, fr, sg2, a2, "V2")
    tc = int(df["time"].iloc[cut])
    m1 = {q["entry_t"]: q for q in full if q["exit_t"] < tc}
    m2 = {q["entry_t"]: q for q in alt if q["exit_t"] < tc}
    check(len(m1) >= 3 and m1 == m2, f"changing the future leaves earlier trades unchanged ({len(m1)} trades)")

    # --- portfolio cap works on the output rows
    cap = B.portfolio([dict(x, pair=f"P{j}") for j, x in enumerate(full[:3])])
    check(len(cap) <= 3, "portfolio accepts the rows")
    print(f"ALL BLUEPRINT BACKTEST SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
