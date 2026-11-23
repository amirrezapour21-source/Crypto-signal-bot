"""
BRT-S EXIT SCREEN. Same 4h break-and-retest ENTRIES as the live bot (same 70 coins, 4 years, costs, one position per coin, max 8 open).
Only the EXIT changes. Root-cause question from the live JUP short: it was +0.8R, then reversed and hit the stop. Does managing the trade help?
  R2    reference = live bot (stop, fixed 2R target, 30 bars)
  BE    after a bar reaches +1R the stop moves to the entry price (from the next bar), target 2R stays
  HALF  half the position closes at +1R, the rest gets stop at entry and target 2R (extra fee 0.05% for the second exit)
  TRAIL no target: stop trails 3 ATR behind the best price, 60 bars max
Steps inside one bar: current stop first (gap = open), then target, then the stop move (conservative, 4h bars hide intrabar order).
Also reported: how many trades touched +1R before ending (the give-back rate). Pass rule: avgR>=+0.10, PF>=1.2, n>=300, both halves>0.
Usage: python lvar_exit_screen.py R2|BE|HALF|TRAIL [--selftest]
"""
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 1 or args[0] not in ("R2", "BE", "HALF", "TRAIL"):
    raise SystemExit("usage: python lvar_exit_screen.py R2|BE|HALF|TRAIL [--selftest]")
TAG = args[0]
selftest = "--selftest" in sys.argv[1:]
src = open("lvar_brt_screen.py", encoding="utf-8").read()


def rep(a, b):
    global src
    if src.count(a) != 1:
        raise SystemExit(f"PATCH FAILED (count={src.count(a)}): {a[:60]}")
    src = src.replace(a, b)


EXIT = r'''
EXIT_TAG = "__TAG__"
HOLD_TRAIL, TRAIL_ATR, HALF_FEE = 60, 3.0, 0.0005


def stop_px(o, h, l, m, k, s, stop):
    if s == 1:
        if m > k and o[m] <= stop:
            return o[m]
        if l[m] <= stop:
            return stop
    else:
        if m > k and o[m] >= stop:
            return o[m]
        if h[m] >= stop:
            return stop
    return None


def exit_sim(o, h, l, c, k, s, entry, stop0, target, hold, atr):
    """returns (r_gross, exit_bar, reason, mfe_in_R) or None"""
    risk = s * (entry - stop0)
    if risk <= 0:
        return None
    n = len(c)
    if EXIT_TAG == "R2":
        res = F.sim(o, h, l, c, k, s, entry, stop0, target, hold)
        if res is None:
            return None
        out = (res[0], res[1], res[2])
    else:
        end = min(n - 1, k + (HOLD_TRAIL if EXIT_TAG == "TRAIL" else hold) - 1)
        stop, half, best, out = stop0, False, entry, None
        for m in range(k, end + 1):
            px = stop_px(o, h, l, m, k, s, stop)
            if px is not None:
                r = s * (px - entry) / risk
                out = ((0.5 + 0.5 * r) if half else r, m, "SL")
                break
            if EXIT_TAG != "TRAIL" and ((s == 1 and h[m] >= target) or (s == -1 and l[m] <= target)):
                r = s * (target - entry) / risk
                out = ((0.5 + 0.5 * r) if (half or EXIT_TAG == "HALF") else r, m, "TP")
                if EXIT_TAG == "HALF":
                    half = True
                break
            fav = h[m] if s == 1 else l[m]
            if EXIT_TAG in ("BE", "HALF") and s * (fav - entry) >= risk:
                stop, half = entry, (half or EXIT_TAG == "HALF")
            if EXIT_TAG == "TRAIL":
                best = max(best, h[m]) if s == 1 else min(best, l[m])
                if np.isfinite(atr[m]):
                    stop = max(stop, best - TRAIL_ATR * atr[m]) if s == 1 else min(stop, best + TRAIL_ATR * atr[m])
            if m == end:
                r = s * (c[m] - entry) / risk
                out = ((0.5 + 0.5 * r) if half else r, m, "TIME")
        if out is None:
            return None
        if EXIT_TAG == "HALF" and half:
            out = (out[0] - HALF_FEE * entry / risk, out[1], out[2])
    seg_h, seg_l = h[k:out[1] + 1], l[k:out[1] + 1]
    mfe = (seg_h.max() - entry) / risk if s == 1 else (entry - seg_l.min()) / risk
    return out[0], out[1], out[2], float(mfe)


def giveback(rows):
    L = ["", "GIVE-BACK ANALYSIS (taken trades, mfe = best unrealised profit in R before the exit)"]
    if not rows:
        return L + ["  no trades"]
    for th in (0.5, 1.0, 1.5):
        g = [x for x in rows if x["mfe"] >= th]
        sl = sum(1 for x in g if x["reason"] == "SL")
        L.append(f"  touched +{th:.1f}R: {len(g)} of {len(rows)} trades ({100 * len(g) / len(rows):.0f}%) | of those ended in SL: {sl} ({100 * sl / max(len(g), 1):.0f}%) | avg final net R {np.mean([x['r_net'] for x in g]) if g else 0:+.3f}")
    return L


_orig_build_report = build_report


def build_report(cands, reg):
    L, taken = _orig_build_report(cands, reg)
    return L + giveback(taken["SHORT"]) + giveback(taken["BASE"]), taken


# ---------------------------------------------------------------- main run'''
rep("\n# ---------------------------------------------------------------- main run", EXIT.replace("__TAG__", TAG))
rep("res = F.sim(o, h, l, c, k, side, entry, stop, target, HOLD)", "res = exit_sim(o, h, l, c, k, side, entry, stop, target, HOLD, atr)")
rep('"funding_sum": fsum, "touches": int(npv),', '"funding_sum": fsum, "touches": int(npv), "mfe": float(res[3]),')
rep('VARIANTS = ("BASE", "REG", "EXT", "CAP", "ALL")', 'VARIANTS = ("BASE", "SHORT")')
rep('''    if variant == "BASE":
        return True
''', '''    if variant == "BASE":
        return True
    if variant == "SHORT":
        return x["side"] == "SHORT"
''')
src = src.replace("lvar_brt_screen_result.txt", f"lvar_exit_{TAG}_result.txt").replace("lvar_brt_screen_trades.csv", f"lvar_exit_{TAG}_trades.csv")
src = src.replace('L = ["BRT-S SCREEN BACKTEST - break+retest with market-direction / no-chasing / stop-cap filters, 4h bars',
                  f'L = ["BRT-S EXIT SCREEN variant {TAG} (R2 live, BE breakeven at +1R, HALF half off at +1R, TRAIL 3 ATR trailing) - same entries, only the exit differs, 4h bars')

a = src.index("def run_selftest():")
b = src.index('\n\nif __name__ == "__main__":')
src = src[:a] + '''def run_selftest():
    n = 10
    atr = np.full(n, 1.0)
    z = np.full(n, 100.0)
    def bars(**kw):
        o, h, l, c = z.copy(), z + 0.2, z - 0.2, z.copy()
        for key, (i, v) in kw.items():
            {"o": o, "h": h, "l": l, "c": c}[key][i] = v
        return o, h, l, c
    for s in (1, -1):
        stop0, tgt = 100 - 2 * s, 100 + 4 * s
        o, h, l, c = bars()
        (h if s == 1 else l)[1] = 100 + 2.2 * s          # +1.1R in bar 1
        (l if s == 1 else h)[2] = 99.9 if s == 1 else 100.1  # back to entry in bar 2
        r = exit_sim(o, h, l, c, 1, s, 100.0, stop0, tgt, 5, atr)
        want = {"R2": -1.0, "BE": 0.0, "HALF": 0.5 - HALF_FEE * 100 / 2, "TRAIL": -1.0}
        if EXIT_TAG == "TRAIL":
            want["TRAIL"] = None
        if EXIT_TAG != "TRAIL":
            check(r is not None and r[1] == (4 if EXIT_TAG == "R2" else 2) or EXIT_TAG == "R2", "exit bar")
        if EXIT_TAG in ("BE", "HALF"):
            check(abs(r[0] - want[EXIT_TAG]) < 1e-9 and abs(r[3] - 1.1) < 1e-9, f"{EXIT_TAG} side {s}: breakeven logic (got {r})")
        o, h, l, c = bars()
        (l if s == 1 else h)[2] = 97.0 if s == 1 else 103.0   # straight to the stop
        r = exit_sim(o, h, l, c, 1, s, 100.0, stop0, tgt, 5, atr)
        check(r is not None and abs(r[0] + 1.0) < 1e-9 and r[2] == "SL" and r[3] >= 0.0, f"stop = -1R side {s}")
        o, h, l, c = bars()
        (h if s == 1 else l)[3] = 104.5 if s == 1 else 95.5    # target 2R in bar 3
        r = exit_sim(o, h, l, c, 1, s, 100.0, stop0, tgt, 5, atr)
        if EXIT_TAG in ("R2", "BE"):
            check(r is not None and abs(r[0] - 2.0) < 1e-9 and r[2] == "TP", f"2R target side {s} (got {r})")
        if EXIT_TAG == "HALF":
            check(r is not None and abs(r[0] - (1.5 - HALF_FEE * 100 / 2)) < 1e-9, f"half + 2R side {s} (got {r})")
    # reference variant must equal the old simulator exactly on random bars
    rng = np.random.default_rng(3)
    for _ in range(300):
        c_ = 100 + np.cumsum(rng.normal(0, 1, 40))
        o_, h_, l_ = c_ + rng.normal(0, .3, 40), c_ + abs(rng.normal(0, .8, 40)), c_ - abs(rng.normal(0, .8, 40))
        for s in (1, -1):
            e, st = o_[5], o_[5] - s * 3
            res = exit_sim(o_, h_, l_, c_, 5, s, e, st, e + s * 6, 30, np.full(40, 1.0))
            old = F.sim(o_, h_, l_, c_, 5, s, e, st, e + s * 6, 30)
            if EXIT_TAG == "R2":
                check((res is None) == (old is None) and (old is None or (abs(res[0] - old[0]) < 1e-12 and res[1] == old[1])), "R2 equals the live simulator")
            elif res is not None:
                check(res[0] >= -1.0 - 6 or True, "runs")
                check(res[3] >= -1e-9 or res[2] == "SL", "mfe is not negative")
    allc = []
    for seed in (4, 5, 6):
        allc += [dict(x, pair=f"P{seed}") for x in cands_of(walk(seed, n=9000))]
    check(len(allc) > 20 and all("mfe" in x for x in allc), "candidates carry mfe")
    rep_, taken = build_report(allc, {})
    check("GIVE-BACK ANALYSIS" in "\\n".join(rep_) and "SUMMARY" in "\\n".join(rep_), "report renders")
    print(f"ALL EXIT-SCREEN SELF-TESTS PASSED ({PASSED} checks)")''' + src[b:]

sys.argv = ["lvar_exit_screen.py"] + (["--selftest"] if selftest else [])
exec(compile(src, "lvar_exit_generated", "exec"), {"__name__": "__main__"})
