#!/usr/bin/env python3
# Synthetic-data tests for evaluate_u67_v2.py and evaluate_u67_long_v2.py (no market data, no real state).
import json, os, re, subprocess, sys, tempfile, hashlib
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.abspath(os.path.join(HERE, ".."))
REPO = "/home/claude/amirrezapour21-source/crypto-signal-bot/"
fails = []
def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c: fails.append(msg)
H = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
BAR = pd.Timedelta(hours=4)
SYMS = ["AAAUSDTM", "BBBUSDTM", "CCCUSDTM", "DDDUSDTM", "EEEUSDTM", "FFFUSDTM", "GGGUSDTM"]
FRESH = {"AAAUSDTM", "BBBUSDTM", "CCCUSDTM"}   # recent last_ts; the other symbols are stale (60+ days)

def run(cmd, repo, env=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=e)
def new_repo():
    d = tempfile.mkdtemp()
    run(["git", "init", "-q"], d); run(["git", "config", "user.name", "t"], d); run(["git", "config", "user.email", "t@t"], d)
    return d
def commit(repo, SF, state, when, msg):
    state["last_run_at"] = msg
    with open(os.path.join(repo, SF), "w") as f: json.dump(state, f, sort_keys=True)
    run(["git", "add", SF], repo)
    d = pd.Timestamp(when).isoformat()
    r = run(["git", "commit", "-qm", msg], repo, {"GIT_COMMITTER_DATE": d, "GIT_AUTHOR_DATE": d}); assert r.returncode == 0, r.stderr

def closed_trade(sig, sym, reason, bars, entry=10.0, risk_pct=0.02):
    risk = entry * risk_pct; sl = entry + risk; tp = entry - 2 * risk
    if reason == "SL": gross = -1.0; xp = sl
    elif reason == "TP": gross = 2.0; xp = tp
    else: xp = entry - 0.3 * risk; gross = 0.3
    ent = sig + BAR
    return dict(id="%s|%s" % (sym, sig.isoformat()), symbol=sym, signal_ts=sig.isoformat(), entry_ts=ent.isoformat(), entry=entry, risk=risk,
                risk_pct=risk_pct, sl=sl, tp=tp, bars=bars, status="CLOSED", exit_ts=(ent + (bars - 1) * BAR).isoformat(), exit_price=xp,
                exit_reason=reason, gross_r=gross, bars_held=bars)
def open_pos(sig, sym, entry=10.0, risk_pct=0.02):
    risk = entry * risk_pct
    return dict(id="%s|%s" % (sym, sig.isoformat()), symbol=sym, signal_ts=sig.isoformat(), entry_ts=(sig + BAR).isoformat(), entry=entry,
                risk=risk, risk_pct=risk_pct, sl=entry + risk, tp=entry - 2 * risk, bars=3, status="OPEN")
def net_s3(c):
    xp = c["exit_price"]; e = c["entry"]
    return c["gross_r"] - (0.0015 * (1 + xp / e) + 0.00010 * c["bars_held"] * 0.5) / c["risk_pct"]
def state_for(created, elig, commit_time, specs):
    """specs: list of (sig_ts, [(sym, kind, bars)]) kind in TP SL TO OPEN CEN."""
    sy = {s: dict(last_ts=(commit_time - pd.Timedelta(hours=4) if s in FRESH else pd.Timestamp("2026-09-01T00:00:00Z")).isoformat(),
                  baseline_done=True, last_status="OK") for s in SYMS}
    st = dict(state_version=1, strategy="x", created_at=created, eligible_from=elig, last_run_at=None, symbols=sy,
              open={}, closed=[], signals=[], missed=[], errors=[])
    for sig, trs in specs:
        for sym, kind, bars in trs:
            if kind in ("OPEN", "CEN"):
                o = open_pos(sig, sym); st["open"][o["id"]] = o
            else:
                st["closed"].append(closed_trade(sig, sym, kind, bars))
    return st

def suite(NAME, EVF, SF, ELIG, SEED, C1, C2, REGF, ENGF, V1REG, V1SHA):
    print("\n===== %s =====" % NAME)
    sys.path.insert(0, HERE)
    modname = os.path.splitext(os.path.basename(EVF))[0]
    E = __import__(modname)
    EV = os.path.join(HERE, EVF)
    T0 = pd.Timestamp(ELIG)
    created = (T0 + pd.Timedelta(minutes=5)).isoformat()
    def scan(repo, cp, mode="scan"):
        r = run([sys.executable, EV, "--repo", repo, "--checkpoint", cp, "--mode", mode, "--state", SF], repo)
        return r
    def mkrepo(specs, when, extra_first=True):
        repo = new_repo()
        first = state_for(created, created, T0 + pd.Timedelta(hours=1), [])
        commit(repo, SF, first, T0 + pd.Timedelta(hours=1), "first")
        st = state_for(created, created, when, specs)
        commit(repo, SF, st, when, "final")
        return repo
    def cl(i, kinds, off=8):
        sig = T0 + pd.Timedelta(hours=off + 4 * i * 2)    # gaps of 8h (non-adjacent)
        fresh = [x for x in SYMS if x in FRESH]; stale = [x for x in SYMS if x not in FRESH]
        out = []
        for j, k in enumerate(kinds):
            sym = stale.pop(0) if k == "CEN" else fresh.pop(0)
            out.append((sym, k, 5 + j))
        return (sig, out)

    # ---- constants / isolation / counter 0
    check(E.REG_ELIGIBLE == T0 and E.SEED == int(SEED) and E.CENSOR_DAYS == 45, "%s constants: eligibility %s, seed %s, 45-day censoring" % (NAME, ELIG, SEED))
    src = open(EV).read()
    other = "u67_long" if "short" in SF else "u67_short"
    check(other not in src and ("U67-S" in src) != ("U67-L" in src), "evaluator never references the other track's state file or label")
    empty = new_repo(); open(os.path.join(empty, "x.txt"), "w").write("x"); run(["git", "add", "x.txt"], empty); run(["git", "commit", "-qm", "x"], empty)
    r = scan(empty, C1)
    check(json.loads(r.stdout)["counts"]["commits"] == 0 and json.loads(r.stdout)["trigger_holds"] is False, "counter 0: repository with no state file -> 0 state commits, trigger False")
    # ---- count-only / refusal before trigger
    repo = mkrepo([cl(i, ["TP"]) for i in range(3)], T0 + pd.Timedelta(days=3))
    r = scan(repo, C1); o = json.loads(r.stdout)
    check(o["trigger_holds"] is False and not re.search(r'net_r|cluster_mean|"S3|gross|ci95|verdict', r.stdout), "scan below N: counts only, no R field in output")
    r = scan(repo, C1, "evaluate")
    check(r.returncode == 3 and "REFUSED" in r.stderr and not re.search(r"cluster_mean|verdict", r.stdout), "evaluate before trigger refuses (exit 3), no R printed")
    check("--force" not in src and "override" not in src.lower(), "no override flag")
    # ---- 45-day boundary, frozen status
    ct = T0 + pd.Timedelta(days=60)
    st = state_for(created, created, ct, [cl(0, ["TP", "CEN"])])
    st["symbols"]["DDDUSDTM"]["last_ts"] = (ct - pd.Timedelta(days=45)).isoformat()
    tr, pend = E.build_trades(st, ct, with_r=False)
    check(any(t["censored"] for t in tr) and not pend, "last_ts exactly 45 days before the evaluation instant -> CENSORED")
    st["symbols"]["DDDUSDTM"]["last_ts"] = (ct - pd.Timedelta(days=45) + pd.Timedelta(minutes=1)).isoformat()
    tr, pend = E.build_trades(st, ct, with_r=False)
    check(not any(t["censored"] for t in tr) and len(pend) == 1, "44d23h59m -> NOT censored (open)")
    st["symbols"]["DDDUSDTM"]["last_ts"] = (ct - pd.Timedelta(days=45)).isoformat()
    tr2, _ = E.build_trades(st, ct + pd.Timedelta(days=30), with_r=False)
    check(any(t["censored"] for t in tr2), "status is a function of the evaluation instant only (recomputation at another instant is the evaluator's, never stored)")
    tr3, _ = E.build_trades(st, ct, with_r=False)
    check(all("S3" not in t and "S3_m1" not in t for t in tr3), "with_r=False computes no R or cost")
    # ---- cluster status, censored cluster skipped, first N non-censored ordering
    specs = []
    for i in range(18):
        specs.append(cl(i, ["TP", "SL"]))
    specs[2] = cl(2, ["CEN", "CEN"])           # fully censored cluster inside the range
    specs[5] = cl(5, ["CEN", "CEN"])
    when = T0 + pd.Timedelta(days=10)
    repo = mkrepo(specs, when)
    o = json.loads(scan(repo, C1).stdout)
    check(o["trigger_holds"] is True and o["counts"]["cohort_size"] == 15 and o["counts"]["censored_clusters_snapshot"] == 2,
          "cohort = 15 non-censored clusters although 2 fully censored clusters sit inside the range; censored clusters counted (2)")
    out = json.loads(run([sys.executable, EV, "--repo", repo, "--checkpoint", C1, "--mode", "evaluate", "--state", SF], repo).stdout.split("RESULT_SHA256")[0])
    exp_keys = [cl(i, [])[0].isoformat() for i in range(18) if i not in (2, 5)][:15]
    check(out["cluster_keys"] == [str(pd.Timestamp(k)) if False else pd.Timestamp(k).isoformat() for k in exp_keys], "first 15 non-censored clusters in signal_ts order (censored clusters skipped)")
    check(out["n_trades_resolved"] == 30 and out["censored_clusters_snapshot"] == 2 and "fully censored" in out["limitation"].lower(), "counts + mandatory limitation text printed")
    check(all("UNVALIDATED" in " ".join(out["notes"]) for _ in [0]) and any("exchange event time" in n for n in out["notes"]), "notes: funding UNVALIDATED, committer time not exchange time")
    # ---- resolved-trade cluster mean and cost formulas
    c0 = [closed_trade(cl(0, [])[0], SYMS[0], "TP", 5), closed_trade(cl(0, [])[0], SYMS[1], "SL", 6)]
    exp0 = float(np.mean([net_s3(c) for c in c0]))
    ks, means = E.cluster_outcomes(E.build_trades(state_for(created, created, when, [cl(0, ["TP", "SL"])]), when)[0], {cl(0, [])[0].tz_convert("UTC") if False else pd.Timestamp(cl(0, [])[0])}, "S3")
    check(abs(means[0] - exp0) < 1e-12, "cluster mean = equal-weight mean of resolved S3 net R (%.6f)" % exp0)
    # ---- sensitivity on SAME cohort; censored trade in cohort cluster
    specs = [cl(i, ["TP", "CEN"]) for i in range(15)]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=10))
    o = json.loads(run([sys.executable, EV, "--repo", repo, "--checkpoint", C1, "--mode", "evaluate", "--state", SF], repo).stdout.split("RESULT_SHA256")[0])
    tp = closed_trade(cl(0, [])[0], SYMS[0], "TP", 5); tpn = net_s3(tp)
    c3 = (0.0015 * (1 + 10.2 / 10.0) + 0.00010 * 30 * 0.5) / 0.02
    m1 = (tpn + (-1.0 - c3)) / 2; m2 = (tpn + (-2.0 - c3)) / 2
    check(abs(o["stats"]["S3"]["cluster_mean"] - tpn) < 1e-9, "primary uses resolved trades only (censored carry no R)")
    check(abs(o["S3_sensitivity_minus1R_cluster_mean"] - m1) < 1e-9 and abs(o["S3_sensitivity_minus2R_cluster_mean"] - m2) < 1e-9,
          "-1R / -2R sensitivities on the same cohort, S3 cost formula unchanged (%.4f / %.4f)" % (m1, m2))
    v1 = __import__("evaluate_u67") if False else None
    # cross-check against the v1 rule value for the -1R case (same cost inputs)
    sys.path.insert(0, os.path.join(ST, "u67" if "short" in SF else "u67l", "evaluator"))
    v1mod = __import__("evaluate_u67" if "short" in SF else "evaluate_u67_long")
    st1 = state_for(created, created, T0 + pd.Timedelta(days=10), [cl(0, ["CEN"])])
    t1, _ = v1mod.build_trades(st1, T0 + pd.Timedelta(days=10))
    check(abs(t1[0]["S3"] - (-1.0 - c3)) < 1e-12 and abs(t1[0]["S3"] - (-1.0 - c3)) < 1e-12, "v1 stop-out S3 value equals the v2 -1R stress input (cost formula unchanged)")
    check(o["censored_trades_in_cohort"] == 15 and o["censored_clusters_snapshot"] == 0, "censored trade count in cohort = 15, no fully censored cluster")
    # ---- b2 guard at UC1 / UC2, rejection rule
    def ev(repo, cp):
        r = run([sys.executable, EV, "--repo", repo, "--checkpoint", cp, "--mode", "evaluate", "--state", SF], repo)
        return json.loads(r.stdout.split("RESULT_SHA256")[0]), r
    specs = [cl(i, ["TP", "CEN", "CEN", "CEN"]) for i in range(15)]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=10))
    o, _ = ev(repo, C1)
    check(o["stats"]["S3"]["cluster_mean"] > 0 and o["S3_sensitivity_minus1R_cluster_mean"] <= 0 and o["guard_b2"], "b2 fires: primary S3 > 0, -1R sensitivity <= 0")
    check(o["verdict"].startswith("CONTINUE_TO_%s" % C2) and "GUARD FLAGGED" in o["verdict"], "at %s a guard is flagged but does not convert the continuation (interpretation I1)" % C1)
    specs = [cl(i, ["TP", "CEN", "CEN", "CEN"]) for i in range(30)]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=20))
    o, _ = ev(repo, C2)
    check(o["verdict"] == "NOT ESTABLISHED (b2: -1R sensitivity <= 0)", "at %s b2 gives NOT ESTABLISHED (b2), PAPER GATE PASSED prohibited" % C2)
    specs = [cl(i, ["SL", "SL"]) for i in range(15)]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=10))
    o, _ = ev(repo, C1)
    check(o["verdict"] == "REJECTED", "primary S3 cluster-mean <= 0 -> REJECTED at %s" % C1)
    specs = [cl(i, ["SL", "SL"]) for i in range(30)] + [cl(30, ["CEN", "CEN"])]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=22))
    o, _ = ev(repo, C2)
    check(o["verdict"] == "REJECTED" and o["guard_censoring_limitation"], "guards never replace the primary rejection (REJECTED even with a fully censored cluster)")
    # ---- PASS baseline, then censoring-limitation guard
    specs = [cl(i, ["TP", "TP", "TP"]) for i in range(30)]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=20))
    o, _ = ev(repo, C2)
    check(o["verdict"] == "PAPER GATE PASSED", "baseline: strong positive 30-cluster sample passes (no guard)")
    specs = [cl(i, ["TP", "TP", "TP"]) for i in range(30)] + [cl(31, ["CEN", "CEN"])]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=22))
    o, _ = ev(repo, C2)
    check(o["verdict"].startswith("NOT ESTABLISHED - CENSORING LIMITATION") and o["guard_censoring_limitation"] and o["censored_clusters_snapshot"] == 1,
          "fully censored cluster anywhere in the snapshot -> NOT ESTABLISHED - CENSORING LIMITATION, no PASS")
    check(abs(o["S3_sensitivity_minus1R_cluster_mean"] - o["stats"]["S3"]["cluster_mean"]) < 1e-12, "fully censored cluster cannot reduce the sensitivity (limitation is real and disclosed)")
    # ---- open trade: in cohort blocks, in later cluster does not
    specs = [cl(i, ["TP", "SL"]) for i in range(14)] + [cl(14, ["TP", "OPEN"])]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=10))
    o = json.loads(scan(repo, C1).stdout)
    check(o["trigger_holds"] is False and o["counts"]["trigger_state"] == "BLOCKED_OPEN", "open (non-censored) trade inside the cohort blocks the checkpoint")
    specs = [cl(i, ["TP", "SL"]) for i in range(15)] + [cl(15, ["TP", "OPEN"])]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=10))
    o = json.loads(scan(repo, C1).stdout)
    check(o["trigger_holds"] is True, "open trade in a LATER cluster does not block")
    specs = [cl(i, ["TP", "SL"]) for i in range(14)] + [cl(14, ["TP", "CEN"])]
    repo = mkrepo(specs, T0 + pd.Timedelta(days=10))
    o = json.loads(scan(repo, C1).stdout)
    check(o["trigger_holds"] is True, "censored trade inside the cohort does not block")
    # ---- completeness (90 min)
    sig = cl(14, [])[0]
    specs = [cl(i, ["TP", "SL"]) for i in range(15)]
    when_early = sig + BAR + pd.Timedelta(minutes=89)
    repo = mkrepo(specs, when_early)
    o = json.loads(scan(repo, C1).stdout)
    check(o["trigger_holds"] is False and o["counts"]["trigger_state"] == "BLOCKED_INCOMPLETE", "cluster not complete before entry + 90 min")
    repo = mkrepo(specs, sig + BAR + pd.Timedelta(minutes=90))
    check(json.loads(scan(repo, C1).stdout)["trigger_holds"] is True, "complete at entry + 90 min")
    # ---- rate cap
    repo = mkrepo([cl(i, ["TP"]) for i in range(3)], T0 + pd.Timedelta(days=121))
    o = json.loads(scan(repo, C1).stdout)["counts"]["rate_cap"]
    check(o["cap_reached"] is True and o["outcome"] == "OBSERVED RATE REPORTED; NO VERDICT", "%s cap (120 d) reached with N never reached -> observed rate, no verdict" % C1)
    repo = mkrepo([cl(i, ["TP"]) for i in range(3)], T0 + pd.Timedelta(days=119))
    check(json.loads(scan(repo, C1).stdout)["counts"]["rate_cap"]["cap_reached"] is False, "%s cap not reached at 119 d" % C1)
    repo = mkrepo([cl(i, ["TP"]) for i in range(3)], T0 + pd.Timedelta(days=239))
    check(json.loads(scan(repo, C2).stdout)["counts"]["rate_cap"]["cap_reached"] is False, "%s cap not reached at 239 d" % C2)
    repo = mkrepo([cl(i, ["TP"]) for i in range(3)], T0 + pd.Timedelta(days=241))
    r = scan(repo, C2, "evaluate")
    check(json.loads(r.stdout)["counts"]["rate_cap"]["cap_reached"] is True and r.returncode == 3, "%s cap reached: no evaluation, no verdict" % C2)
    # ---- determinism
    repo = mkrepo([cl(i, ["TP", "SL"]) for i in range(15)], T0 + pd.Timedelta(days=10))
    a = run([sys.executable, EV, "--repo", repo, "--checkpoint", C1, "--mode", "evaluate", "--state", SF], repo).stdout
    b = run([sys.executable, EV, "--repo", repo, "--checkpoint", C1, "--mode", "evaluate", "--state", SF], repo).stdout
    check(a == b and "RESULT_SHA256" in a, "deterministic: two runs identical")
    # ---- registration / eligibility waiver / engine hashes / v1 preserved
    reg = open(os.path.join(HERE, REGF)).read()
    check(("ELIGIBLE_FROM (UTC, literal, equal to the engine constant REGISTRATION_ELIGIBLE_FROM, UNCHANGED from v1): %s" % ELIG) in reg and "WAIVER (explicit)" in reg and "DISCLOSURE: at the v2 freeze no" in reg,
          "v2 registration holds the exact eligibility constant, WAIVER and DISCLOSURE text")
    check(("starts at counter 0" in reg) and ("supersedes" in reg) and (V1SHA in reg), "v2 registration: counter 0, supersession of the recorded v1 sha")
    eng = open(os.path.join(REPO, ENGF)).read()
    m = re.search(r'REGISTRATION_ELIGIBLE_FROM\s*=\s*pd\.Timestamp\(\s*"([^"]+)"', eng)
    check(m is not None and pd.Timestamp(m.group(1)) == T0, "engine constant REGISTRATION_ELIGIBLE_FROM == v2 ELIGIBLE_FROM (%s)" % (m.group(1) if m else None))
    check(("engine sha256 %s" % H(os.path.join(REPO, ENGF))) in reg, "engine hash in v2 registration equals the engine file hash in the repository root")
    check(H(os.path.join(ST, V1REG)) == V1SHA, "v1 registration file unchanged (sha256 %s...)" % V1SHA[:8])
    check(("sha256: %s" % H(os.path.join(HERE, REGF))) in src or H(os.path.join(HERE, REGF)) in src, "evaluator header carries the v2 registration sha256")
    check("UNRESOLVED stop-out rule, which is withdrawn" in reg and "-2R sensitivity is informational and never decides a verdict" in reg, "v2 registration: v1 rule withdrawn, -2R informational")

S1SHA = "9efe88d1b07c166be1e4e371e3c2339fc941194b22f8cf03fd5c8d8fe879d8be"
L1SHA = "3f0298f2314d1dc54c7768dbe181b41839e5223f9903356b7a1b903f2af5b02a"
suite("U67-S v2", "evaluate_u67_v2.py", "u67_short_paper_state.json", "2026-10-28T12:00:00Z", "20261028", "UC1", "UC2",
      "U67_S_REGISTRATION_v2.txt", "crypto_signal_bot_u67_short_v1.py", "u67/U67_S_REGISTRATION_v1.txt", S1SHA)
suite("U67-L v2", "evaluate_u67_long_v2.py", "u67_long_paper_state.json", "2026-10-30T12:00:00Z", "20261029", "UL1", "UL2",
      "U67_L_REGISTRATION_v2.txt", "crypto_signal_bot_u67_long_v1.py", "u67l/U67_L_REGISTRATION_v1.txt", L1SHA)
# isolation between the two v2 packages
a = open(os.path.join(HERE, "evaluate_u67_v2.py")).read(); b = open(os.path.join(HERE, "evaluate_u67_long_v2.py")).read()
check(a != b and "u67_long" not in a and "u67_short" not in b, "S and L v2 evaluators are separate files, no cross reference")
print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
