"""
BRT-S LOOSENING SCREEN (fewer requirements = more signals). Takes the committed lvar_brt_screen.py (same 4h data, costs, exits)
and loosens the entry rule in three ways fixed in advance (nothing else changes, nothing tuned after seeing results):
  A = a level needs only 2 pivots instead of 3        (MIN_PIV 3 -> 2)
  B = the break must be only 0.25 ATR deep instead of 0.5   (BREAK_CLEAN 0.5 -> 0.25)
  C = both A and B
Reference: BASE (the live rule) = n=1749 shorts, net avgR +0.016 (see lvar_brt_screen_result.txt).
Variants reported: BASE = both sides, SHORT = short only. Verdict variant = SHORT.
Pass rule: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
Three loosened rules are tested: a pass of exactly one is weak evidence, never proof. The forward paper test stays the final judge.
Usage: python lvar_loose_screen.py A|B|C   or   python lvar_loose_screen.py A|B|C --selftest
"""
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 1 or args[0] not in ("A", "B", "C"):
    raise SystemExit("usage: python lvar_loose_screen.py A|B|C [--selftest]")
TAG = args[0]
selftest = "--selftest" in sys.argv[1:]

src = open("lvar_brt_screen.py", encoding="utf-8").read()


def rep(a, b):
    global src
    if src.count(a) != 1:
        raise SystemExit(f"PATCH FAILED (count={src.count(a)}): {a[:60]}")
    src = src.replace(a, b)


if TAG in ("A", "C"):
    rep("MIN_PIV = 3\n", "MIN_PIV = 2\n")
if TAG in ("B", "C"):
    rep("BREAK_CLEAN = 0.5\n", "BREAK_CLEAN = 0.25\n")

rep('VARIANTS = ("BASE", "REG", "EXT", "CAP", "ALL")', 'VARIANTS = ("BASE", "SHORT")')
rep('''    if variant == "BASE":
        return True
''', '''    if variant == "BASE":
        return True
    if variant == "SHORT":
        return x["side"] == "SHORT"
''')
src = src.replace("lvar_brt_screen_result.txt", f"lvar_loose_{TAG}_result.txt")
src = src.replace("lvar_brt_screen_trades.csv", f"lvar_loose_{TAG}_trades.csv")
src = src.replace('L = ["BRT-S SCREEN BACKTEST - break+retest with market-direction / no-chasing / stop-cap filters, 4h bars',
                  f'L = ["BRT-S LOOSENING SCREEN variant {TAG} (A: 2 pivots, B: break 0.25 ATR, C: both) - same rule otherwise, 4h bars')

a = src.index("    rep, taken = build_report(allc, breg)")
b = src.index('    print(f"ALL BRT-SCREEN SELF-TESTS PASSED')
src = src[:a] + '''    rep, taken = build_report(allc, breg)
    text = "\\n".join(rep)
    check(all(f"{v} - break and retest" in text for v in VARIANTS) and "SUMMARY" in text, "report renders all variants")
    check(all(x["side"] == "SHORT" for x in taken["SHORT"]), "SHORT variant contains only shorts")
    check(len(taken["BASE"]) > 0, "signals exist")
''' + src[b:]
rep('check(keep("BASE", lg, reg) and keep("ALL", sh, reg) and not keep("ALL", lg, reg), "variant ALL = REG and EXT and CAP")',
    'check(keep("BASE", lg, reg) and keep("SHORT", sh, reg) and not keep("SHORT", lg, reg), "variant SHORT = shorts only")')
exp_piv = 2 if TAG in ("A", "C") else 3
exp_clean = 0.25 if TAG in ("B", "C") else 0.5
rep('    print(f"ALL BRT-SCREEN SELF-TESTS PASSED',
    f'    check(MIN_PIV == {exp_piv} and BREAK_CLEAN == {exp_clean}, "loosened constants for this variant")\n    print(f"ALL BRT-SCREEN SELF-TESTS PASSED')

sys.argv = ["lvar_loose_screen.py"] + (["--selftest"] if selftest else [])
exec(compile(src, "lvar_loose_generated", "exec"), {"__name__": "__main__"})
