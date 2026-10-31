#!/usr/bin/env python3
# Synthetic-data tests for evaluate_u67_long.py (no market data, no real state).
import json, os, subprocess, sys, tempfile, hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "evaluate_u67_long.py")
sys.path.insert(0, HERE)
import evaluate_u67_long as E

fails = []
def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c: fails.append(msg)

SF = "u67_long_paper_state.json"
BAR = pd.Timedelta(hours=4)

def run(cmd, repo, env=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=e)

def commit(repo, state, when, msg="s"):
    state["last_run_at"] = str(msg) + str(when)
    with open(os.path.join(repo, SF), "w") as f:
        json.dump(state, f, sort_keys=True)
    run(["git", "add", SF], repo)
    d = pd.Timestamp(when).isoformat()
    r = run(["git", "commit", "-qm", msg], repo, {"GIT_COMMITTER_DATE": d, "GIT_AUTHOR_DATE": d})
    assert r.returncode == 0, r.stderr

def new_repo():
    d = tempfile.mkdtemp()
    run(["git", "init", "-q"], d); run(["git", "config", "user.name", "t"], d); run(["git", "config", "user.email", "t@t"], d)
    return d

SYMS = ["AAAUSDTM", "BBBUSDTM", "CCCUSDTM", "DDDUSDTM"]
T0 = pd.Timestamp("2026-10-30T12:00:00Z")

def closed_trade(i, sig, sym, reason, bars, entry=10.0, risk_pct=0.02):
    risk = entry * risk_pct
    sl = entry - risk; tp = entry + 2 * risk
    if reason == "SL": gross = -1.0; xp = sl
    elif reason == "TP": gross = 2.0; xp = tp
    else: xp = entry + 0.3 * risk; gross = 0.3
    ent = sig + BAR
    s3c = 0.0015 * (1 + xp / entry) + 0.0001 * bars * 0.5
    return dict(id="%s|%s" % (sym, sig.isoformat()), symbol=sym, side="LONG", signal_ts=sig.isoformat(), entry_ts=ent.isoformat(),
                entry=entry, risk=risk, risk_pct=risk_pct, sl=sl, tp=tp, bars=bars, status="CLOSED",
                exit_ts=(ent + (bars - 1) * BAR).isoformat(), exit_price=xp, exit_reason=reason, gross_r=gross,
                net_r=gross - 0.002 / risk_pct, net_r_s3=gross - s3c / risk_pct, bars_held=bars, stop_gap=False)

def empty_state(created, eligible):
    return dict(state_version=1, strategy="u67_short_only_frozen_v1", created_at=created, eligible_from=eligible,
                last_run_at=None, symbols={s: dict(last_ts="2026-10-30T08:00:00+00:00", baseline_done=True, last_status="OK") for s in SYMS},
                open={}, closed=[], signals=[], missed=[], errors=[])

def open_pos(sig, sym, entry=10.0, risk_pct=0.02):
    risk = entry * risk_pct
    return dict(id="%s|%s" % (sym, sig.isoformat()), symbol=sym, side="LONG", signal_ts=sig.isoformat(), entry_ts=(sig + BAR).isoformat(),
                entry=entry, risk=risk, risk_pct=risk_pct, sl=entry - risk, tp=entry + 2 * risk, bars=3, status="OPEN")

# ---------------------------------------------------------------- scenario
rng = np.random.default_rng(7)
repo = new_repo()
created = "2026-10-30T12:05:00+00:00"
st = empty_state(created, "2026-10-30T12:05:00+00:00")
sigs = [T0 + pd.Timedelta(hours=8 + 4 * i * 1) for i in range(0, 40)]   # consecutive 4H candles -> exercises adjacent merge
# create 15 clusters (UL1) with 1-3 trades each; make cluster 14 still open at first, close later
plan = []
for i in range(40):
    k = int(rng.integers(1, 4))
    for j in range(k):
        plan.append((sigs[i], SYMS[j], rng.choice(["TP", "SL", "TIMEOUT"], p=[.3, .5, .2]), int(rng.integers(2, 31))))
# commit 0: first state
commit(repo, st, T0 + pd.Timedelta(hours=6), "first")
cur = json.loads(json.dumps(st)); trades_all = []
# progressively add clusters, closing them 1 day after signal; leave cluster index 14 open until a later commit
commit_times = []
for i in range(15):
    for (s, sym, rs, b) in [p for p in plan if p[0] == sigs[i]]:
        if i == 14 and sym == "BBBUSDTM":
            cur["open"][ "%s|%s" % (sym, s.isoformat())] = open_pos(s, sym)
            continue
        tr = closed_trade(0, s, sym, rs, b); cur["closed"].append(tr); trades_all.append(tr)
    when = sigs[i] + pd.Timedelta(hours=5)
    commit(repo, cur, when, "c%d" % i); commit_times.append(when)
# After cluster 14 with an open trade: trigger must NOT hold yet
r = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL1", "--mode", "scan"], repo)
out = json.loads(r.stdout)
cluster14_has_b = any(p[0] == sigs[14] and p[1] == "BBBUSDTM" for p in plan)
check(out["trigger_holds"] is (not cluster14_has_b), "trigger respects open trade in cluster 15 (has_open=%s)" % cluster14_has_b)
if cluster14_has_b:
    # close it in a later commit
    s = sigs[14]
    pend = cur["open"].pop("BBBUSDTM|%s" % s.isoformat())
    p = [x for x in plan if x[0] == s and x[1] == "BBBUSDTM"][0]
    tr = closed_trade(0, s, "BBBUSDTM", p[2], p[3]); cur["closed"].append(tr); trades_all.append(tr)
    close_when = sigs[14] + pd.Timedelta(hours=40)
    commit(repo, cur, close_when, "close14")
else:
    close_when = commit_times[-1]
# add extra later clusters (must be excluded from UL1)
for i in range(15, 18):
    for (s, sym, rs, b) in [p for p in plan if p[0] == sigs[i]]:
        tr = closed_trade(0, s, sym, rs, b); cur["closed"].append(tr)
    commit(repo, cur, sigs[i] + pd.Timedelta(hours=5), "late%d" % i)

r1 = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL1", "--mode", "scan"], repo)
o1 = json.loads(r1.stdout)
check(o1["trigger_holds"], "UL1 trigger holds")
check(o1["instant"] == pd.Timestamp(close_when).isoformat().replace("+00:00", "+00:00") or pd.Timestamp(o1["instant"]) == pd.Timestamp(close_when),
      "evaluation instant = commit time of FIRST satisfying commit (not HEAD)")
txt = r1.stdout
check(all(w not in txt for w in ["cluster_mean", "net_r", "S3", "ci95"]), "scan output contains counts only (no R fields)")

# evaluate
r2 = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL1", "--mode", "evaluate"], repo)
check(r2.returncode == 0, "evaluate runs")
body = r2.stdout[:r2.stdout.index("RESULT_SHA256")]
res = json.loads(body)
# ---- independent recomputation (plain loops)
sel_sig = sigs[:15]
tr15 = [t for t in trades_all if pd.Timestamp(t["signal_ts"]) in set(sel_sig)]
def indep(scn):
    cm = []
    for s in sel_sig:
        v = []
        for t in tr15:
            if pd.Timestamp(t["signal_ts"]) != s: continue
            e, xp, rp, b, rs = t["entry"], t["exit_price"], t["risk_pct"], t["bars_held"], t["exit_reason"]
            ratio = 1 + xp / e
            if scn == "S0": c = 0.002
            elif scn == "S3": c = 0.0015 * ratio + 0.0001 * b * .5
            elif scn == "S3a": c = 0.0015 * ratio + 0.0001 * b * .5 + 0.00025 * ratio
            elif scn == "S3b": c = 0.0015 * ratio + 0.0001 * b * .5 + 0.0005 * ratio
            elif scn == "S1": c = 0.0006 * ratio
            else: c = (0.0004 if rs == "TP" else 0.0006) * ratio
            v.append(t["gross_r"] - c / rp)
        cm.append(sum(v) / len(v))
    return np.array(cm)
for scn in ["S0", "S3", "S3a", "S3b", "S1", "S2"]:
    cm = indep(scn)
    check(abs(res["stats"][scn]["cluster_mean"] - cm.mean()) < 1e-12, "cluster mean %s matches independent" % scn)
cm3 = indep("S3")
g = np.random.default_rng(20261029); idx = g.integers(0, 15, size=(20000, 15)); mm = cm3[idx].mean(axis=1)
check(abs(res["stats"]["S3"]["ci95_lo"] - np.percentile(mm, 2.5)) < 1e-12 and abs(res["stats"]["S3"]["ci95_hi"] - np.percentile(mm, 97.5)) < 1e-12, "bootstrap CI matches independent (seed 20261029, 20000 draws)")
check(abs(res["stats"]["S3"]["p_mean_gt0"] - (mm > 0).mean()) < 1e-12, "P(mean>0) matches")
# jackknife independent
big = max(range(15), key=lambda i: (cm3[i], -i))   # highest, ties earlier
rem = [cm3[i] for i in range(15) if i != big]
check(abs(res["jackknife_S3"]["mean_without_largest"] - np.mean(rem)) < 1e-12 and res["jackknife_S3"]["largest_cluster"] == sel_sig[big].isoformat(), "jackknife largest cluster")
months = {}
for i, s in enumerate(sel_sig): months.setdefault(s.strftime("%Y-%m"), []).append(cm3[i])
bm = sorted(months, key=lambda m: (-sum(months[m]), m))[0]
remm = [cm3[i] for i, s in enumerate(sel_sig) if s.strftime("%Y-%m") != bm]
check(res["jackknife_S3"]["best_month"] == bm and (abs(res["jackknife_S3"]["mean_without_best_month"] - np.mean(remm)) < 1e-12 if remm else res["jackknife_S3"]["mean_without_best_month"] is None), "jackknife best month")
check(res["n_clusters"] == 15 and res["n_trades"] == len(tr15), "only first 15 clusters, later clusters excluded")
check(res["created_at_check"] == "OK", "created_at check OK")
check(res["verdict"].startswith("REJECTED") == (cm3.mean() <= 0), "UL1 verdict rule consistent with S3 cluster-mean sign")
# adjacent merge (all 15 consecutive -> a single merged group)
check(res["stats"]["S3"]["merged_n"] == 1, "adjacent-merge: 15 consecutive 4H clusters chain into 1 group")
# determinism
r3 = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL1", "--mode", "evaluate"], repo)
check(r3.stdout == r2.stdout, "deterministic: byte-identical output on rerun")

# ---- refusal when trigger does not hold
r4 = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL2", "--mode", "evaluate"], repo)
check(r4.returncode == 3 and "cluster_mean" not in r4.stdout, "UL2 evaluate refused (trigger false), no R shown")

# ---- UNRESOLVED rule
repo2 = new_repo()
st2 = empty_state(created, "2026-10-30T12:05:00+00:00")
commit(repo2, st2, T0 + pd.Timedelta(hours=6), "first")
cur2 = json.loads(json.dumps(st2))
for i in range(15):
    s = sigs[i]
    if i == 3:
        # open forever on symbol AAA; its cursor stops advancing at 2026-10-30T08:00
        cur2["open"]["AAAUSDTM|%s" % s.isoformat()] = open_pos(s, "AAAUSDTM")
        cur2["closed"].append(closed_trade(0, s, "BBBUSDTM", "SL", 5))
    else:
        cur2["closed"].append(closed_trade(0, s, "BBBUSDTM", "TP", 9))
    commit(repo2, cur2, s + pd.Timedelta(hours=5), "u%d" % i)
# not yet 45 days after last_ts -> not triggered
o = json.loads(run([sys.executable, EV, "--repo", repo2, "--checkpoint", "UL1"], repo2).stdout)
check(not o["trigger_holds"], "open position blocks trigger before 45 days")
# commit at last_ts + 44d23h -> still not; +45d -> UNRESOLVED allows trigger
lt = pd.Timestamp("2026-10-30T08:00:00Z")
commit(repo2, cur2, lt + pd.Timedelta(days=44, hours=23), "d44")
o = json.loads(run([sys.executable, EV, "--repo", repo2, "--checkpoint", "UL1"], repo2).stdout)
check(not o["trigger_holds"], "44d23h: not yet UNRESOLVED")
commit(repo2, cur2, lt + pd.Timedelta(days=45), "d45")
o = json.loads(run([sys.executable, EV, "--repo", repo2, "--checkpoint", "UL1"], repo2).stdout)
check(o["trigger_holds"] and pd.Timestamp(o["instant"]) == lt + pd.Timedelta(days=45), "exactly 45d: UNRESOLVED -> trigger holds at that commit")
r = run([sys.executable, EV, "--repo", repo2, "--checkpoint", "UL1", "--mode", "evaluate"], repo2)
res2 = json.loads(r.stdout[:r.stdout.index("RESULT_SHA256")])
check(res2["n_unresolved"] == 1, "one UNRESOLVED trade counted")
# check its S3 value: gross -1, exit=sl, bars 30
e = 10.0; rp = 0.02; sl = e * (1 - rp)
exp_s3 = -1.0 - (0.0015 * (1 + sl / e) + 0.0001 * 30 * .5) / rp
exp_s0 = -1.0 - 0.002 / rp
# cluster 3 mean = (that BBB closed SL trade + unresolved)/2
b = closed_trade(0, sigs[3], "BBBUSDTM", "SL", 5)
exp_c3 = (b["net_r_s3"] + exp_s3) / 2
others = []
for i in range(15):
    if i == 3: others.append(exp_c3)
    else: others.append(closed_trade(0, sigs[i], "BBBUSDTM", "TP", 9)["net_r_s3"])
check(abs(res2["stats"]["S3"]["cluster_mean"] - np.mean(others)) < 1e-12, "UNRESOLVED counted as -1R stop-out with S3 cost, bars 30")
ex = [closed_trade(0, sigs[i], "BBBUSDTM", "TP", 9)["net_r_s3"] if i != 3 else b["net_r_s3"] for i in range(15)]
check(abs(res2["S3_cluster_mean_excluding_unresolved"] - np.mean(ex)) < 1e-12, "sensitivity excluding UNRESOLVED")
exp_s0c = np.mean([ (b["net_r"] + exp_s0) / 2 if i == 3 else closed_trade(0, sigs[i], "BBBUSDTM", "TP", 9)["net_r"] for i in range(15)])
check(abs(res2["stats"]["S0"]["cluster_mean"] - exp_s0c) < 1e-12, "UNRESOLVED applies to S0 too")

# ---- created_at mismatch
repo3 = new_repo()
a = empty_state("2026-10-30T12:05:00+00:00", "2026-10-30T12:05:00+00:00")
commit(repo3, a, T0 + pd.Timedelta(hours=6), "first")
c3 = json.loads(json.dumps(a)); c3["created_at"] = "2026-11-02T00:00:00+00:00"; c3["eligible_from"] = "2026-11-02T00:00:00+00:00"
for i in range(15):
    s = pd.Timestamp("2026-11-02T04:00:00Z") + 4 * i * BAR / 4 * 1
    c3["closed"].append(closed_trade(0, s, "AAAUSDTM", "TP", 8))
commit(repo3, c3, pd.Timestamp("2026-11-10T00:00:00Z"), "recreated")
r = run([sys.executable, EV, "--repo", repo3, "--checkpoint", "UL1", "--mode", "evaluate"], repo3)
res3 = json.loads(r.stdout[:r.stdout.index("RESULT_SHA256")])
check(res3["created_at_check"].startswith("DEFECT"), "created_at mismatch labelled DEFECT")

# ---- eligibility filter: trade before eligible_from is ignored
repo4 = new_repo()
a = empty_state("2026-10-30T12:05:00+00:00", "2026-10-30T12:05:00+00:00")
early = pd.Timestamp("2026-10-30T04:00:00Z")   # entry 08:00 < eligibility
a["closed"].append(closed_trade(0, early, "AAAUSDTM", "TP", 8))
commit(repo4, a, T0 + pd.Timedelta(hours=6), "first")
o = json.loads(run([sys.executable, EV, "--repo", repo4, "--checkpoint", "UL1"], repo4).stdout)
check(o["counts"]["trades"] == 0, "entry before eligibility boundary is not evidence")

# ---- UL2 decision rule table (unit level)
class A: btc_start = None; btc_end = None
def synth_state(vals_by_cluster, n=30):
    s = empty_state("2026-10-30T12:05:00+00:00", "2026-10-30T12:05:00+00:00")
    sg = [pd.Timestamp("2026-10-30T20:00:00Z") + pd.Timedelta(days=i) for i in range(n)]
    for i, sig in enumerate(sg):
        tr = closed_trade(0, sig, "AAAUSDTM", "TP" if vals_by_cluster[i] > 0 else "SL", 8)
        s["closed"].append(tr)
    return s, sg
def verdict(vals):
    s, sg = synth_state(vals)
    ct = sg[-1] + pd.Timedelta(days=2)
    return E.evaluate(s, "x", ct, s["created_at"], "UL2", A)
check(verdict([1] * 30)["verdict"] == "PAPER GATE PASSED", "UL2 all winners -> PAPER GATE PASSED")
check(verdict([-1] * 30)["verdict"] == "REJECTED", "UL2 all losers -> REJECTED")
# mean>0 but CI lower bound <= 0: 12 winners (+2R) 18 losers (-1R): mean ~ +0.2.. after costs maybe >0 ; just assert logic consistency
v = verdict([1] * 12 + [-1] * 18)
check(v["verdict"] in ("REJECTED", "NOT ESTABLISHED (final for U67-L v1)"), "UL2 mixed -> REJECTED or NOT ESTABLISHED (never PASSED with CI<=0): " + v["verdict"])

# ---- premature-R guard: the scan/trigger path must never compute any R or cost
def _boom(*a, **k):
    raise AssertionError("cost_fraction called on count-only path")
_orig_cost = E.cost_fraction
E.cost_fraction = _boom
try:
    h1 = E.scan(repo, SF, "UL1")      # trigger holds
    h2 = E.scan(repo, SF, "UL2")      # trigger does not hold
    h3 = E.scan(repo4, SF, "UL1")     # no eligible trades
    ok_scan = (h1[0] is not None) and (h2[0] is None)
except AssertionError as ex:
    ok_scan = False
finally:
    E.cost_fraction = _orig_cost
check(ok_scan, "scan/trigger path computes NO R or cost (cost_fraction patched to raise; trigger true and false cases)")
# CLI refuses evaluate when the trigger is false, shows no R, and has no override flag
r = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL2", "--mode", "evaluate"], repo)
check(r.returncode == 3 and "cluster_mean" not in r.stdout + r.stderr and "net" not in r.stdout.lower(), "evaluate refused before trigger: exit 3, no R in output")
r = run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL1", "--mode", "evaluate", "--at-commit", "HEAD"], repo)
check(r.returncode != 0, "no override flag exists (unknown option rejected)")
# evaluation commit is the first satisfying commit even when called later with more commits present
o_a = json.loads(run([sys.executable, EV, "--repo", repo, "--checkpoint", "UL1"], repo).stdout)
check(o_a["first_satisfying_commit"] == o1["first_satisfying_commit"], "first satisfying commit stable")

print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
