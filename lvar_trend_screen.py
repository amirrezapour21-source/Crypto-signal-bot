"""
LONG TREND-PULLBACK SCREEN (4h, Binance USDT-M perpetuals, real funding, costs 0.20% round trip, one position per coin, max 8 open).
Idea in plain words: when a coin is in a clear uptrend, buy the short dip back to its 20-EMA and ride the move with a trailing stop
(no fixed target), so that a multi-day rise (for example +25% in 3 days) is not cut off at 2R. Same 70 coins and 4 years as the BRT screens.
All numbers are fixed IN ADVANCE and nothing is tuned after seeing results:
  TREND    close > EMA50 > EMA200 on the signal bar (EMA of 4h closes, adjust=False)
  DIP      the lowest low of the last 3 bars (signal bar included) touched or went below EMA20
  TRIGGER  the signal bar closes GREEN and back ABOVE EMA20
  ENTRY    next bar open
  STOP     lowest low of the last 5 bars - 0.25 ATR14;  risk 1%-12% of price (same limits as the BRT bot)
  TRAIL    after every bar the stop is raised to (highest high since entry - 3 ATR14), never lowered;  exit at the stop price (gap: at the open);  time stop after 60 bars
Variants: TRAIL = the rule above (verdict variant).  TP2R = same entries, fixed target 2R, 30 bars (informational: shows whether the trailing exit is what matters).
Pass rule (fixed before the run): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both time halves net-positive.
This is a SCREEN, not proof. Two variants are tested; a pass of exactly one is weak evidence. The forward paper test stays the final judge.
Usage: python lvar_trend_screen.py [--selftest]
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
# ---------------------------------------------------------------- LONG trend-pullback rule (replaces the break+retest candidates)
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
DIP_BARS = 3
STOP_BARS = 5
TRAIL_ATR = 3.0
HOLD_TRAIL = 60
HOLD_TP = 30
VARIANTS = ("TRAIL", "TP2R")


def sim_trail(o, h, l, c, atr, k, entry, stop0, hold):
    """long position, entered at the open of bar k. returns (r_gross, exit_bar, reason) or None.
    Per bar: first the CURRENT stop is checked (gap through it = exit at the open), then the stop is raised using this bar's high."""
    risk = entry - stop0
    if risk <= 0:
        return None
    n = len(c)
    end = min(n - 1, k + hold - 1)
    stop, hh = stop0, entry
    for m in range(k, end + 1):
        if m > k and o[m] <= stop:
            return (o[m] - entry) / risk, m, "SL"
        if l[m] <= stop:
            return (stop - entry) / risk, m, "SL"
        hh = max(hh, h[m])
        if np.isfinite(atr[m]):
            stop = max(stop, hh - TRAIL_ATR * atr[m])
        if m == end:
            return (c[m] - entry) / risk, m, "TIME"
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
        if not (c[i] > e50[i] > e200[i]):
            continue
        if not (l[i - DIP_BARS + 1:i + 1].min() <= e20[i] and c[i] > e20[i] and c[i] > o[i]):
            continue
        if t[i] - t[i - STOP_BARS] != STOP_BARS * TF or t[i + 1] - t[i] != TF:
            continue
        k = i + 1
        entry = o[k]
        stop0 = l[i - STOP_BARS + 1:i + 1].min() - STOP_BUF * a
        risk = entry - stop0
        risk_pct = risk / entry if entry > 0 else 0.0
        if risk <= 0 or not (RISK_LO <= risk_pct <= RISK_HI):
            continue
        for var in VARIANTS:
            if var == "TRAIL":
                res = sim_trail(o, h, l, c, atr, k, entry, stop0, HOLD_TRAIL)
            else:
                res = F.sim(o, h, l, c, k, 1, entry, stop0, entry + RR * risk, HOLD_TP)
            if res is None:
                continue
            fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
            out.append({"pair": sym, "side": "LONG", "var": var, "sig_t": int(t[i]), "entry_t": int(t[k]),
                        "exit_t": int(t[res[1]]), "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY),
                        "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                        "r_net": float(res[0] - (COST_RT + fsum) / risk_pct),
                        "r_stress": float(res[0] - (COST_STRESS + fsum) / risk_pct), "strength": 0.0,
                        "funding_sum": fsum, "touches": 0,
                        "ext_atr": float((c[i] - c[i - EXT_BARS]) / a)})
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
        L += B.block(f"{v} - LONG trend pullback (eligible)", taken[v], True)
    return L, taken


# ---------------------------------------------------------------- main run'''
rep("\n# ---------------------------------------------------------------- main run", TREND)

# files and headings
src = src.replace("lvar_brt_screen_result.txt", "lvar_trend_result.txt")
src = src.replace("lvar_brt_screen_trades.csv", "lvar_trend_trades.csv")
src = src.replace('L = ["BRT-S SCREEN BACKTEST - break+retest with market-direction / no-chasing / stop-cap filters, 4h bars',
                  'L = ["LONG TREND-PULLBACK SCREEN - uptrend (close>EMA50>EMA200), dip to EMA20, green close above EMA20, trailing stop 3 ATR, 4h bars')
src = re.sub(r'         f"filters \(fixed in advance\):[^\n]*\n',
             '         "rule (fixed in advance): TREND close>EMA50>EMA200; DIP low of last 3 bars <= EMA20; TRIGGER green close above EMA20; stop = 5-bar low - 0.25 ATR (risk 1%-12%); TRAIL = highest high - 3 ATR, 60 bars max; TP2R = fixed 2R variant",\n', src, count=1)
src = src.replace("This is a SCREEN: five variants are tested, a pass of exactly one is weak evidence;", "This is a SCREEN: two variants are tested, a pass of exactly one is weak evidence;")
src = src.replace("Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (SL assumed first), news not in data.",
                  "Known problems: top-70 liquid contracts, entry at the next bar open, 4h bars hide intrabar order (stop assumed hit first), long only, news not in data.")

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
    l[1] = 97.0
    r = sim_trail(o, h, l, c, atr, 1, 100.0, 98.0, 10)
    check(r is not None and abs(r[0] + 1.0) < 1e-9 and r[2] == "SL" and r[1] == 1, "stop hit on the entry bar = -1R")
    # (2) gap below the stop: exit at the open (worse than -1R)
    o, h, l, c = base.copy(), base + 0.5, base - 0.5, base.copy()
    o[2], h[2], l[2], c[2] = 95.0, 95.5, 94.5, 95.0
    r = sim_trail(o, h, l, c, atr, 1, 100.0, 98.0, 10)
    check(r is not None and abs(r[0] - (-2.5)) < 1e-9 and r[1] == 2, "gap through the stop exits at the open")
    # (3) rally then drop: exit above entry at the trailed stop, trail never falls
    o = np.array([100, 100, 104, 108, 112, 112, 110, 100, 100, 100, 100, 100], float)
    h = np.array([100.5, 104, 108, 112, 116, 113, 111, 101, 100, 100, 100, 100], float)
    l = np.array([99.5, 99.5, 103, 107, 111, 109, 105, 99, 99, 99, 99, 99], float)
    c = np.array([100, 103.5, 107.5, 111.5, 112, 110, 106, 100, 100, 100, 100, 100], float)
    r = sim_trail(o, h, l, c, atr, 1, 100.0, 98.0, 10)
    # highest high up to bar 4 = 116 -> stop 113; bar 5 opens at 112 <= 113 -> gap exit at the open 112 = (112-100)/2 = +6R
    check(r is not None and r[2] == "SL" and r[1] == 5 and abs(r[0] - 6.0) < 1e-9, f"trailing stop locks in profit (got {r})")
    # same rally but bar 5 opens above the trailed stop and trades down through it: exit at the stop price 113 = +6.5R
    o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
    o2[5], h2[5], l2[5], c2[5] = 114.0, 114.5, 109.0, 110.0
    r2 = sim_trail(o2, h2, l2, c2, atr, 1, 100.0, 98.0, 10)
    check(r2 is not None and r2[1] == 5 and abs(r2[0] - 6.5) < 1e-9, f"exit at the trailed stop price (got {r2})")
    # (4) time stop at the close of the last bar
    o = np.full(n, 100.0); h = o + 0.3; l = o - 0.3; c = o.copy()
    r = sim_trail(o, h, l, c, atr, 1, 100.0, 95.0, 4)
    check(r is not None and r[2] == "TIME" and r[1] == 4, "time stop after the hold limit")
    # --- candidates on synthetic walks with a drift
    total = 0
    for seed in (1, 2, 3):
        df = walk(seed, n=9000, drift=0.0004)
        cs = cands_of(df)
        total += len(cs)
        t = df["time"].to_numpy(np.int64)
        o_, h_, l_, c_ = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        cc = df["close"].astype(float)
        e20 = cc.ewm(span=20, adjust=False).mean().to_numpy(float)
        e50 = cc.ewm(span=50, adjust=False).mean().to_numpy(float)
        e200 = cc.ewm(span=200, adjust=False).mean().to_numpy(float)
        atr_ = X.indicators4(df)["atr"]
        tidx = {int(v): j for j, v in enumerate(t)}
        for x in cs:
            check(x["sig_t"] == x["entry_t"] - TF and x["sig_t"] in tidx, "signal bar is the bar before the entry bar")
            i = tidx[x["sig_t"]]
            check(c_[i] > e50[i] > e200[i], "uptrend on the signal bar")
            check(l_[i - 2:i + 1].min() <= e20[i] and c_[i] > e20[i] and c_[i] > o_[i], "dip + green close above EMA20")
            check(abs(x["risk_pct"] - (o_[i + 1] - (l_[i - 4:i + 1].min() - STOP_BUF * atr_[i])) / o_[i + 1]) < 1e-9, "stop from the last 5 bars only")
            check(RISK_LO <= x["risk_pct"] <= RISK_HI, "risk limits")
            check(x["exit_t"] >= x["entry_t"] and x["var"] in VARIANTS and x["side"] == "LONG", "fields")
            check(x["r_net"] < x["r_gross"] + 1e-9 or x["funding_sum"] < 0, "costs reduce r unless funding is received")
        trail = [x for x in cs if x["var"] == "TRAIL"]
        tp = [x for x in cs if x["var"] == "TP2R"]
        ent_t, ent_p = {x["entry_t"] for x in trail}, {x["entry_t"] for x in tp}
        last_ok = int(t[-1]) - (HOLD_TRAIL + 2) * TF
        check(all(e >= last_ok for e in ent_t ^ ent_p), "variants share the same entries (they differ only when the data ends before the exit)")
        check(all(x["days"] <= HOLD_TRAIL * TF / DAY + 1e-9 for x in trail), "trail hold limit")
        check(all(x["r_gross"] <= RR + 1e-9 for x in tp), "TP2R never exceeds 2R")
    check(total >= 40, f"enough synthetic signals ({total})")
    allc = []
    for seed in (4, 5, 6):
        allc += [dict(x, pair=f"P{seed}") for x in cands_of(walk(seed, n=9000, drift=0.0004))]
    rep_, taken = build_report(allc, {})
    text = "\\n".join(rep_)
    check("SUMMARY" in text and all(f"{v} - LONG trend pullback" in text for v in VARIANTS), "report renders all variants")
    check(all(x["var"] == "TRAIL" for x in taken["TRAIL"]) and all(x["var"] == "TP2R" for x in taken["TP2R"]), "variants are separated")
    print(f"ALL TREND-SCREEN SELF-TESTS PASSED ({PASSED} checks)")''' + src[b:]

sys.argv = ["lvar_trend_screen.py"] + sys.argv[1:]
exec(compile(src, "lvar_trend_generated", "exec"), {"__name__": "__main__"})
