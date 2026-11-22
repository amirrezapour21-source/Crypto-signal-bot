"""
BRT-S LATE-ENTRY SCREEN. Takes the committed lvar_brt_screen.py (4h BRT screen, same rule, same data, same costs),
adds two measurements to every candidate and tests ONE question fixed in advance:
  Are SHORT entries made after price has ALREADY fallen far (late entries) worse than early ones?
Measure (primary, fixed): drop_atr = (highest high of the last 60 bars up to and including the signal bar - entry price) / ATR14 of the signal bar.
Secondary (diagnostic only): frac = (that 60-bar high - entry) / (that 60-bar high - lowest low of the same 60 bars)  [share of the 60-bar range already travelled].
Groups: terciles of drop_atr, cut points taken from TRAIN (entries before 2025-01-01) only; test = 2025-01-01 and later.
Verdict (fixed before the run): LATE_WEAK only if, for the LATE tercile versus the EARLY tercile, net avgR is lower by at least 0.10 R in BOTH train and test AND n(LATE) >= 200 in each period.
Otherwise: NO EVIDENCE that late entries are worse (nothing is changed in the bot).
Extra, informational: SHORT variant NOLATE (drop LATE tercile) judged by the usual pass rule. Nothing is tuned after seeing results.
Usage: python lvar_late_screen.py [--selftest]
"""
import sys

src = open("lvar_brt_screen.py", encoding="utf-8").read()


def rep(a, b):
    global src
    if src.count(a) != 1:
        raise SystemExit(f"PATCH FAILED (count={src.count(a)}): {a[:60]}")
    src = src.replace(a, b)


# 1) variants: BASE (both sides) and SHORT only
rep('VARIANTS = ("BASE", "REG", "EXT", "CAP", "ALL")', 'VARIANTS = ("BASE", "SHORT")')
rep('''    if variant == "BASE":
        return True
''', '''    if variant == "BASE":
        return True
    if variant == "SHORT":
        return x["side"] == "SHORT"
''')

# 2) the two new measurements on every candidate
rep('''                    "ext_atr": float(side * (c[i] - c[i - EXT_BARS]) / a)})''',
    '''                    "ext_atr": float(side * (c[i] - c[i - EXT_BARS]) / a),
                    "entry_px": float(entry),
                    "drop_atr": float((h[i - LATE_BARS:i + 1].max() - entry) / a),
                    "frac": float((h[i - LATE_BARS:i + 1].max() - entry) / max(h[i - LATE_BARS:i + 1].max() - l[i - LATE_BARS:i + 1].min(), 1e-12))})''')
rep("REG_EMA = 50\n", "REG_EMA = 50\nLATE_BARS = 60\nSPLIT_T = 1735689600  # 2025-01-01 00:00 UTC\nMIN_LATE_N = 200\nGAP = 0.10\n")

# 3) the late-entry analysis, appended to the report
LATE = '''
def late_groups(cands):
    """returns (q33, q67) of drop_atr over TRAIN short candidates, groups by tercile"""
    sh = [x for x in cands if x["side"] == "SHORT"]
    tr = [x["drop_atr"] for x in sh if x["entry_t"] < SPLIT_T]
    if len(tr) < 30:
        return None
    q33, q67 = (float(v) for v in np.quantile(tr, [1 / 3, 2 / 3]))
    return q33, q67


def grp_of(x, q):
    return "EARLY" if x["drop_atr"] < q[0] else ("LATE" if x["drop_atr"] >= q[1] else "MID")


def late_report(cands):
    L = ["", "=" * 60, "LATE-ENTRY TEST (SHORT candidates, before the position cap)",
         "measure: drop_atr = (60-bar high - entry) / ATR14; terciles from TRAIN (entries < 2025-01-01); test = 2025-01-01 and later"]
    q = late_groups(cands)
    if q is None:
        L.append("not enough train short candidates")
        return L, None
    sh = [x for x in cands if x["side"] == "SHORT"]
    L.append(f"cut points (ATR): EARLY < {q[0]:.2f} <= MID < {q[1]:.2f} <= LATE")
    res = {}
    for g in ("EARLY", "MID", "LATE"):
        for per, f in (("train", lambda x: x["entry_t"] < SPLIT_T), ("test", lambda x: x["entry_t"] >= SPLIT_T), ("all", lambda x: True)):
            rows = [x["r_net"] for x in sh if grp_of(x, q) == g and f(x)]
            res[(g, per)] = B.stats(rows)
            L.append(B.fmt(f"  {g:5s} {per}", rows))
    def avg(g, per):
        s = res[(g, per)]
        return s["avgR"] if s else float("nan")
    def cnt(g, per):
        s = res[(g, per)]
        return s["n"] if s else 0
    gap_tr = avg("EARLY", "train") - avg("LATE", "train")
    gap_te = avg("EARLY", "test") - avg("LATE", "test")
    L.append(f"  EARLY minus LATE (net avgR): train {gap_tr:+.3f} | test {gap_te:+.3f}   (rule: both >= {GAP:.2f} and n(LATE) >= {MIN_LATE_N} in each period)")
    weak = bool(gap_tr >= GAP and gap_te >= GAP and cnt("LATE", "train") >= MIN_LATE_N and cnt("LATE", "test") >= MIN_LATE_N)
    L.append("  VERDICT (pre-registered): " + ("LATE_WEAK - late short entries are worse in train AND test" if weak
                                               else "NO EVIDENCE that late short entries are worse (rule not met) - nothing to change"))
    L.append("")
    L.append("DIAGNOSTIC (not eligible for selection): share of the 60-bar range already travelled (frac), same 3 groups by train terciles")
    trf = [x["frac"] for x in sh if x["entry_t"] < SPLIT_T]
    f1, f2 = (float(v) for v in np.quantile(trf, [1 / 3, 2 / 3]))
    L.append(f"  cut points: EARLY < {f1:.2f} <= MID < {f2:.2f} <= LATE")
    for g, f in (("EARLY", lambda v: v < f1), ("MID", lambda v: f1 <= v < f2), ("LATE", lambda v: v >= f2)):
        for per, pf in (("train", lambda x: x["entry_t"] < SPLIT_T), ("test", lambda x: x["entry_t"] >= SPLIT_T)):
            L.append(B.fmt(f"  frac {g:5s} {per}", [x["r_net"] for x in sh if f(x["frac"]) and pf(x)]))
    L.append("")
    nolate = [x for x in cands if x["side"] == "SHORT" and grp_of(x, q) != "LATE"]
    tk = B.portfolio(nolate)
    L.append("INFORMATIONAL - SHORT without the LATE tercile, one position per coin, max 8 open, pass rule applied:")
    L.append("  " + summary_row("NOLATE", tk))
    L.append("  " + summary_row("SHORT ", B.portfolio([x for x in cands if x["side"] == "SHORT"])))
    return L, res


_orig_build_report = build_report


def build_report(cands, reg):
    L, taken = _orig_build_report(cands, reg)
    extra, _ = late_report(cands)
    return L + extra, taken


# ---------------------------------------------------------------- main run'''
rep("\n# ---------------------------------------------------------------- main run", LATE)

# 4) output files and headings
src = src.replace("lvar_brt_screen_result.txt", "lvar_late_result.txt")
src = src.replace("lvar_brt_screen_trades.csv", "lvar_late_trades.csv")
src = src.replace('L = ["BRT-S SCREEN BACKTEST - break+retest with market-direction / no-chasing / stop-cap filters, 4h bars',
                  'L = ["BRT-S LATE-ENTRY SCREEN - same 4h break+retest rule, nothing changed; question: are late SHORT entries worse? 4h bars')
src = src.replace('cols = ["pair", "side", "sig_t", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "ext_atr", "funding_sum", "touches"]',
                  'cols = ["pair", "side", "sig_t", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "ext_atr", "funding_sum", "touches", "drop_atr", "frac"]')

# 5) self-test: replace the old end part and add checks for the new measurements
a = src.index("    rep, taken = build_report(allc, breg)")
b = src.index('    print(f"ALL BRT-SCREEN SELF-TESTS PASSED')
src = src[:a] + '''    rep, taken = build_report(allc, breg)
    text = "\\n".join(rep)
    check(all(f"{v} - break and retest" in text for v in VARIANTS) and "SUMMARY" in text, "report renders all variants")
    check("LATE-ENTRY TEST" in text and "VERDICT (pre-registered)" in text, "late-entry section renders")
    check(all(x["side"] == "SHORT" for x in taken["SHORT"]), "SHORT variant contains only shorts")
    q = late_groups(allc)
    check(q is not None and q[0] <= q[1], "tercile cut points exist and are ordered")
    chg = [dict(x, r_net=x["r_net"] + (5.0 if x["entry_t"] >= SPLIT_T else 0.0)) for x in allc]
    check(late_groups(chg) == q, "cut points do not depend on outcomes or on the test period")
    check(all(grp_of(x, q) in ("EARLY", "MID", "LATE") for x in allc), "every candidate has a group")
''' + src[b:]
rep('check(keep("BASE", lg, reg) and keep("ALL", sh, reg) and not keep("ALL", lg, reg), "variant ALL = REG and EXT and CAP")',
    'check(keep("BASE", lg, reg) and keep("SHORT", sh, reg) and not keep("SHORT", lg, reg), "variant SHORT = shorts only")')
rep('''            check(abs(x["ext_atr"] - s * (c[i] - c[i - EXT_BARS]) / atr[i]) < 1e-9, "ext_atr uses only closed bars up to the signal bar")''',
    '''            check(abs(x["ext_atr"] - s * (c[i] - c[i - EXT_BARS]) / atr[i]) < 1e-9, "ext_atr uses only closed bars up to the signal bar")
            hh, ll, oo = df["high"].to_numpy(float), df["low"].to_numpy(float), df["open"].to_numpy(float)
            check(abs(x["entry_px"] - oo[i + 1]) < 1e-9, "entry price is the open of the bar after the signal bar")
            check(abs(x["drop_atr"] - (hh[i - 60:i + 1].max() - x["entry_px"]) / atr[i]) < 1e-9, "drop_atr uses only bars up to the signal bar")
            check(abs(x["frac"] - (hh[i - 60:i + 1].max() - x["entry_px"]) / (hh[i - 60:i + 1].max() - ll[i - 60:i + 1].min())) < 1e-9, "frac uses only bars up to the signal bar")''')

sys.argv = ["lvar_late_screen.py"] + sys.argv[1:]
exec(compile(src, "lvar_late_generated", "exec"), {"__name__": "__main__"})
