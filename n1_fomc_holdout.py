"""N1 FOMC pilot - HOLDOUT, single registered run. Reuses n1_fomc_dev.py unchanged (hash-checked, so sections 2-5 of the SPEC are untouched);
only the event list (the 15 holdout dates inside the hash-checked SPEC) and the month range differ. Read-only; prints only."""
import hashlib, re, sys
sys.path.insert(0, ".")
import n1_fomc_dev as n

DEV_CODE_SHA256 = "3d7459bc4b19672844065492bd09bf97c0197846d5d1f6d2f5977676a37a7ab0"
MON = [f"{y}-{m:02d}" for y in (2024, 2025, 2026) for m in range(1, 13) if "2024-12" <= f"{y}-{m:02d}" <= "2026-10"]   # 2024-12 only feeds baselines

def main():
    assert hashlib.sha256(n.SPEC.encode()).hexdigest() == n.SPEC_SHA256, "REGISTRATION MISMATCH"
    assert hashlib.sha256(open("n1_fomc_dev.py", "rb").read()).hexdigest() == DEV_CODE_SHA256, "n1_fomc_dev.py changed"
    hold = re.search(r"Holdout \(15[^\n]*?\): ([^\n]+)", n.SPEC).group(1).split()
    assert len(hold) == 15 and min(hold) >= "2025-01-01" and max(hold) <= "2026-10-31" and max(MON) == "2026-10", hold
    n.DEV = hold
    print("SPEC sha256", n.SPEC_SHA256, "(verified)\ndev code sha256", DEV_CODE_SHA256, "(verified)\nholdout script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    print("HOLDOUT RUN: events", hold[0], "..", hold[-1], f"({len(hold)}); months loaded", MON[0], "..", MON[-1])
    print("Registered development result for comparison: H1 SUPPORTED (17/18, p 0.0001); H2 NOT supported (6/18 reversals, rate 0.333, p 0.9519); H3 not testable")
    n.analyse({s: n.load(s, MON) for s in n.SYMS}); print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
