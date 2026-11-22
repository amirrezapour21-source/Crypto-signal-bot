"""
BRT-S 1-HOUR SCREEN. Takes the committed lvar_brt_screen.py (4h screen), changes ONLY the timeframe
to 1h (TF, bars/day, 30-day liquidity window, download URLs) and runs it.
All bar-count rule parameters stay identical (PIV 5, LOOKBACK 180, FRESH 10, HOLD 30, risk 1%-12%, TP 2R).
Variants fixed in advance: BASE = both sides, SHORT = short only (verdict variant).
Pass rule: net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
Usage: python lvar_brt1h_screen.py [--selftest]
"""
import sys

src = open("lvar_brt_screen.py", encoding="utf-8").read()


def rep(a, b):
    global src
    if src.count(a) != 1:
        raise SystemExit(f"PATCH FAILED (count={src.count(a)}): {a[:60]}")
    src = src.replace(a, b)


rep("TF = 4 * 3600", "TF = 3600")
rep("BPD = 6\n", "BPD = 24\n")
rep("VOL_BARS = 180\n", "VOL_BARS = 720\n")
rep("/monthly/klines/{s}/4h/{s}-4h-{m}.zip", "/monthly/klines/{s}/1h/{s}-1h-{m}.zip")
rep('VARIANTS = ("BASE", "REG", "EXT", "CAP", "ALL")', 'VARIANTS = ("BASE", "SHORT")')
rep('''    if variant == "BASE":
        return True
''', '''    if variant == "BASE":
        return True
    if variant == "SHORT":
        return x["side"] == "SHORT"
''')
src = src.replace("lvar_brt_screen_result.txt", "lvar_brt1h_result.txt")
src = src.replace("lvar_brt_screen_trades.csv", "lvar_brt1h_trades.csv")
src = src.replace("4h klines", "1h klines").replace("BTCUSDT 4h data", "BTCUSDT 1h data")
src = src.replace(", 4h bars, Binance", ", 1h bars (SAME rule as the 4h bot, only the timeframe changed), Binance")

a = src.index("    rep, taken = build_report(allc, breg)")
b = src.index('    print(f"ALL BRT-SCREEN SELF-TESTS PASSED')
src = src[:a] + '''    rep, taken = build_report(allc, breg)
    text = "\\n".join(rep)
    check(all(f"{v} - break and retest" in text for v in VARIANTS) and "SUMMARY" in text, "report renders all variants")
    check(all(x["side"] == "SHORT" for x in taken["SHORT"]), "SHORT variant contains only shorts")
    check(len(taken["BASE"]) > 0 and len(taken["SHORT"]) <= len(taken["BASE"]) + 5, "SHORT is a subset-sized view of BASE")
    check(TF == 3600 and X.BPD == 24 and X.VOL_BARS == 720, "1h constants")
''' + src[b:]
rep('check(keep("BASE", lg, reg) and keep("ALL", sh, reg) and not keep("ALL", lg, reg), "variant ALL = REG and EXT and CAP")',
    'check(keep("BASE", lg, reg) and keep("SHORT", sh, reg) and not keep("SHORT", lg, reg), "variant SHORT = shorts only")')

sys.argv = ["lvar_brt1h_screen.py"] + sys.argv[1:]
exec(compile(src, "lvar_brt1h_generated", "exec"), {"__name__": "__main__"})
