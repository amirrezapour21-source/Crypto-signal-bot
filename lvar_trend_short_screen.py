"""
SHORT TREND-PULLBACK SCREEN (4h, Binance USDT-M perpetuals, real funding, costs 0.20% round trip, one position per coin, max 8 open).
Exact mirror of the LONG trend-pullback screen (lvar_trend_screen.py, result: net avgR -0.003). Purpose: catch the strong one-way DROPS that the
live BRT-S short bot misses because it needs a clean retest. Nothing is tuned: every number is the mirror of the long rule.
  TREND    close < EMA50 < EMA200 on the signal bar (EMA of 4h closes, adjust=False)
  DIP      the highest high of the last 3 bars (signal bar included) touched or went above EMA20
  TRIGGER  the signal bar closes RED and back BELOW EMA20
  ENTRY    next bar open
  STOP     highest high of the last 5 bars + 0.25 ATR14;  risk 1%-12% of price (same limits as the BRT bot)
  TRAIL    after every bar the stop is lowered to (lowest low since entry + 3 ATR14), never raised;  exit at the stop price (gap: at the open);  time stop after 60 bars
Variants: TRAIL = the rule above (verdict variant).  TP2R = same entries, fixed target 2R, 30 bars (informational).
Pass rule (fixed before the run): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
This is a SCREEN, not proof. Two variants are tested; a pass of exactly one is weak evidence. The forward paper test stays the final judge.
Usage: python lvar_trend_short_screen.py [--selftest]
"""
import re
import sys

src = open("lvar_brt_screen.py", encoding="utf-8").read()


def rep(a, b):
    global src
    if src.count(a) != 1:
        raise SystemExit(f"PATCH FAILED (count={src.count(a)}): {a[:60]}")
    src = src.replace(a, b)


TREND = r'''
# ---------------------------------------------------------------- SHORT trend-pullback rule (replaces the break+retest candidates)
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
DIP_BARS = 3
STOP_BARS = 5
TRAIL_ATR = 3.0
HOLD_TRAIL = 60
HOLD_TP = 30
VARIANTS = ("TRAIL", "TP2R")


def sim_trail(o, h, l, c, atr, k, entry, stop0, hold):
    """SHORT position, entered at the open of bar k. returns (r_gross, exit_bar, reason) or None.
    Per bar: first the CURRENT stop is checked (gap through it = exit at the open), then the stop is lowered using this bar's low."""
    risk = stop0 - entry
    if risk <= 0:
        return None
    n = len(c)
    end = min(n - 1, k + hold - 1)
    stop, ll = stop0, entry
    for m in range(k, end + 1):
        if m > k and o[m] >= stop:
            return (entry - o[m]) / risk, m, "SL"
        if h[m] >= stop:
            return (entry - stop) / risk, m, "SL"
        ll = min(ll, l[m])
        if np.isfinite(atr[m]):
            stop = min(stop, ll + TRAIL_ATR * atr[m])
        if m == end:
            return (entry - c[m]) / risk, m, "TIME"
    return None


def make_candidates(sym, df, ft, fr):
    n = len(df)
    if n < EMA_SLOW + 50:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy(np.int64)
    ind = X.indicators4(df)
    atr, vol = ind["atr"], ind["vol"]
    cs = df["close"].astype(float)
    e20 = cs.ewm(span=EMA_FAST, adjust=False).mean().to_numpy(float)
    e50 = cs.ewm(span=EMA_MID, adjust=False).mean().to_numpy(float)
    e200 = cs.ewm(span=EMA_SLOW, adjust=False).mean().to_numpy(float)
    out = []
    for i in range(EMA_SLOW, n - 2):
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(vol[i])) or vol[i] < MIN_VOL30:
            continue
        if not (c[i] < e50[i] < e200[i]):
            continue
        if not (h[i - DIP_BARS + 1:i + 1].max() >= e20[i] and c[i] < e20[i] and c[i] < o[i]):
            continue
        if t[i] - t[i - STOP_BARS] != STOP_BARS * TF or t[i + 1] - t[i] != TF:
            continue
        k = i + 1
        entry = o[k]
        stop0 = h[i - STOP_BARS + 1:i + 1].max() + STOP_BUF * a
        risk = stop0 - entry
        risk_pct = risk / entry if entry > 0 else 0.0
        if risk <= 0 or not (RISK_LO <= risk_pct <= RISK_HI):
            continue
        for var in VARIANTS:
            if var == "TRAIL":
                res = sim_trail(o, h, l, c, atr, k, entry, stop0, HOLD_TRAIL)
            else:
                res = F.sim(o, h, l, c, k, -1, entry, stop0, entry - RR * risk, HOLD_TP)
            if res is None:
                continue
            fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
            out.append({"pair": sym, "side": "SHORT", "var": var, "sig_t": int(t[i]), "entry_t": int(t[k]),
                        "exit_t": int(t[res[1]]), "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY),
                        "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                        "r_net": float(res[0] - (COST_RT - fsum) / risk_pct),
                        "r_stress": float(res[0] - (COST_STRESS - fsum) / risk_pct), "strength": 0.0,
                        "funding_sum": fsum, "touches": 0,
                        "ext_atr": float(-(c[i] - c[i - EXT_BARS]) / a)})
    return out


def keep(variant, x, reg):
    return x["var"] == variant


def summary_row(name, rows):
    if not rows:
        return f"{name:6s} n=0"
    rows = sorted(rows, key=lambda x: x["entry_t"])
    mid_t = rows[len(rows) // 2]["entry_t"]
    sn = B.stats([x["r_net"] for x in rows])
    sa = B.stats([x["r_net"] for x in rows if x["entry_t"] < mid_t])
    sb = B.stats([x["r_net"] for x in rows if x["entry_t"] >= mid_t])
    ok = bool(sn and sa and sb and sn["avgR"] >= 0.10 and sn["PF"] >= 1.2 and sn["n"] >= 300 and sa["avgR"] > 0 and sb["avgR"] > 0)
    return (f"{name:6s} n={sn['n']:5d} win%={sn['win%']:5.1f} avgR={sn['avgR']:+.3f} PF={sn['PF']:.2f} "
            f"halves={sa['avgR']:+.3f}/{sb['avgR']:+.3f} {'PASS' if ok else 'FAIL'}")


def build_report(cands, reg):
    L, taken = [], {}
    for v in VARIANTS:
        taken[v] = B.portfolio([x for x in cands if x["var"] == v])
    L.append("SUMMARY (net R after 0.20% costs + real funding, one position per coin, max 8 open)")
    for v in VARIANTS:
        L.append("  " + summary_row(v, taken[v]))
    L.append("")
    L.append("Reading help: avgR = average net profit per trade in units of the initial risk; PF = gross wins / gross losses; win% is expected to be low for a trailing exit.")
    L.append("")
    for v in VARIANTS:
        L += B.block(f"{v} - SHORT trend pullback (eligible)", taken[v], True)
    return L, taken


# ---------------------------------------------------------------- main run'''
rep("\n# ---------------------------------------------------------------- main run", TREND)

# files and headings
src = src.replace("lvar_brt_screen_result.txt", "lvar_trendS_result.txt")
src = src.replace("lvar_brt_screen_trades.csv", "lvar_trendS_trades.csv")
src = src.replace('L = ["BRT-S SCREEN BACKTEST - break+retest with market-direction / no-chasing / stop-cap filters, 4h bars',
                  'L = ["SHORT TREND-PULLBACK SCREEN - downtrend (close<EMA50<EMA200), bounce to EMA20, red close below EMA20, trailing stop 3 ATR, 4h bars')
src = re.sub(r'         f"filters \(fixed in advance\):[^\n]*\n',
             '         "rule (fixed in advance, mirror of the long rule): TREND close<EMA50<EMA200; DIP high of last 3 bars >= EMA20; TRIGGER red close below EMA20; stop = 5-bar high + 0.25 ATR (risk 1%-12%); TRAIL = lowest low + 3 ATR, 60 bars max; TP2R = fixed 2R variant",\n', src, count=1)
src = src.replace("This is a SCREEN: five variants are tested, a pass of exactly one is weak evidence;", "This is a SCREEN: two variants are tested, a pass of exactly one is weak evidence;")
src = src.replace("Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (SL assumed first), news not in data.",
                  "Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (stop assumed hit first), short only, news not in data.")

# self-test (own)
a = src.index("def run_selftest():")
b = src.index('\n\nif __name__ == "__main__":')
src = src[:a] + '''def run_selftest():
    # --- sim_trail on crafted bars
    n = 12
    atr = np.full(n, 1.0)
    base = np.full(n, 100.0)
    # (1) immediate hit of the initial stop = -1R
    o, h, l, c = base.copy(), base + 0.5, base - 0.5, base.copy()
    h[1] = 103.0
    r = sim_trail(o, h, l, c, atr, 1, 100.0, 102.0, 10)
    check(r is not None and abs(r[0] + 1.0) < 1e-9 and r[2] == "SL" and r[1] == 1, "stop hit on the entry bar = -1R")
    # (2) gap below the stop: exit at the open (worse than -1R)
    o, h, l, c = base.copy(), base + 0.5, base - 0.5, base.copy()
    o[2], h[2], l[2], c[2] = 105.0, 105.5, 104.5, 105.0
    r = sim_trail(o, h, l, c, atr, 1, 100.0, 102.0, 10)
    check(r is not None and abs(r[0] - (-2.5)) < 1e-9 and r[1] == 2, "gap through the stop exits at the open")
    # (3) rally then drop: exit above entry at the trailed stop, trail never falls
    o = np.array([100, 100, 96, 92, 88, 88, 90, 100, 100, 100, 100, 100], float)
    h = np.array([100.5, 100.5, 97, 93, 89, 91, 95, 101, 100, 100, 100, 100], float)
    l = np.array([99.5, 96, 92, 88, 84, 87, 89, 99, 99, 99, 99, 99], float)
    c = np.array([100, 96.5, 92.5, 88.5, 88, 90, 94, 100, 100, 100, 100, 100], float)
