#!/usr/bin/env python3
# U67-S EVALUATOR v2.1 (read-only). Supersedes evaluator v2 (UC1 censoring-guard correction, designer Stage 15).
# Implements U67_S_REGISTRATION_v2_1.txt (sections 0e, 6, 7, 7A, 8) - CENSORING rule replaces the v1 UNRESOLVED stop-out rule.
# Registration v2.1 sha256: a10b12becf49a65877a4c33066e06a5c473ee141cd039b25b04192f25d528a70
#
# Properties (unchanged from v1):
#  - Never modifies the repository or any state file (git show / git log only).
#  - Phase 1 "scan": prints COUNTS ONLY (no R value anywhere).
#  - Phase 2 "evaluate": refuses to run unless the trigger holds in git history; always evaluates the FIRST
#    committed state that satisfies the trigger, never HEAD. Evaluation instant = that commit's committer time.
#  - Deterministic: no clock reads, fixed seed, fixed ordering.
# v2 rules (see registration section 7A):
#  - CENSORED trade: still in the state's open set and (evaluation instant - symbols[sym].last_ts) >= 45 days; status fixed
#    at the evaluation instant. It stays in its cluster, carries NO R in the primary analysis.
#  - Cluster mean = equal-weight mean over the cluster's RESOLVED (closed) trades. A cluster with zero resolved trades,
#    at least one censored trade and no open trade = CENSORED CLUSTER: excluded from primary N, thresholds, bootstrap, jackknife.
#  - Cohort = the first N non-censored clusters in signal_ts order. Trigger: they exist, are complete (commit time >=
#    entry_ts + 90 min) and no OPEN (non-censored) trade belongs to a cluster met before the cohort is complete.
#  - Sensitivities (S3 only): censored trades inside the SAME cohort get gross R = -1 (S3_m1) / -2 (S3_m2) with the S3 cost
#    formula, stored entry/risk/stop, 30 bars. They are fixed stress assumptions, not loss bounds. Fully censored clusters
#    are outside the cohort and cannot reduce either sensitivity.
#  - Guards: they never replace the primary rejection (S3 cluster-mean <= 0 -> REJECTED).
#    CL (censoring limitation): primary S3 > 0 and any fully censored cluster exists in the evaluation snapshot ->
#    verdict NOT ESTABLISHED - CENSORING LIMITATION at BOTH UC1 and UC2 (at UC1 monitoring may continue toward UC2, but the
#    reported checkpoint verdict stays NOT ESTABLISHED and is never a passed checkpoint); PAPER GATE PASSED is prohibited.
#    b2: primary S3 > 0 and S3_m1 <= 0: at UC2 -> NOT ESTABLISHED (b2); at UC1 it is printed as GUARD FLAGGED (open question to the
#    designer: apply b2 at UC1 like CL?).
#  - Trigger/scan path is count-only (with_r=False): no R is computed before the trigger holds.
# Implementation choices carried over from v1 (covered by this script's sha256; change only under registration section 9):
#  1. Bootstrap: fresh numpy default_rng(SEED) per scenario, integers(0, n, size=(20000, n)) over the cohort cluster means.
#  2. Commit time = git committer date (%cI); it is NOT exchange event time.
#  3. Cluster complete = commit_time >= entry_ts + 90 minutes.
#  4. Max concurrent open: [entry_ts, exit_ts + 4h); censored: [entry_ts, entry_ts + 30 x 4h).
#  5. Adjacent-merge sensitivity: cohort clusters 4h apart chained; merged outcome = equal-weight mean of resolved trades.
#  6. Profit factor and drawdown use S3 net R of resolved trades.
#  7. BTC change printed only if --btc-start and --btc-end are supplied.
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

STATE_FILE_DEFAULT = "u67_short_paper_state.json"
REG_ELIGIBLE = pd.Timestamp("2026-10-28T12:00:00Z")
SEED = 20261028
DRAWS = 20000
CENSOR_DAYS = 45
MAX_ENTRY_DELAY_MIN = 90
BAR = pd.Timedelta(hours=4)
SCEN = ["S0", "S3", "S3a", "S3b", "S1", "S2"]
CHECKPOINT_N = {"UC1": 15, "UC2": 30}
CAP_DAYS = {"UC1": 120, "UC2": 240}
TRACK = "U67-S"
LIMITATION_TEXT = ("Fully censored clusters are excluded from the sensitivity cohort and cannot reduce the -1R/-2R "
                   "sensitivities or the b2 guard; they appear only as counts. -1R and -2R are fixed stress assumptions, "
                   "not worst-case or conservative bounds.")
NOTES = [
    "Funding-adjusted S3/S3a/S3b performance remains UNVALIDATED (flat funding proxy, no actual funding history).",
    "evaluation_instant is the git committer time of the state commit, NOT exchange event time.",
    "Jackknife results of this track must not be compared directly with LONG-M1 (different frozen definition).",
    "Rate-cap outcomes are descriptive only (no verdict).",
]


# ---------------------------------------------------------------- git helpers
def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), p.stderr.strip()[:300]))
    return p.stdout


def state_commits(repo, state_file):
    out = git(repo, "log", "--first-parent", "--reverse", "--format=%H%x09%cI", "--", state_file)
    rows = []
    for line in out.splitlines():
        if line.strip():
            h, t = line.split("\t")
            rows.append((h, pd.Timestamp(t).tz_convert("UTC")))
    return rows


def load_state_at(repo, commit, state_file):
    try:
        return json.loads(git(repo, "show", "%s:%s" % (commit, state_file)))
    except RuntimeError:
        return None  # file absent at that commit (deletion) -> defect label by caller


# ------------------------------------------------------------ trade building
def ts(x):
    return pd.Timestamp(x).tz_convert("UTC") if pd.Timestamp(x).tzinfo else pd.Timestamp(x, tz="UTC")


def cost_fraction(scn, entry, exit_price, bars, reason):
    ratio = 1.0 + exit_price / entry
    if scn == "S0":
        return 0.002
    s3 = 0.0015 * ratio + 0.00010 * bars * 0.5
    if scn == "S3":
        return s3
    if scn == "S3a":
        return s3 + 0.00025 * ratio
    if scn == "S3b":
        return s3 + 0.0005 * ratio
    if scn == "S1":
        return 0.0006 * ratio
    if scn == "S2":
        return (0.0004 if reason == "TP" else 0.0006) * ratio
    raise ValueError(scn)


def build_trades(state, commit_time, with_r=True):
    """Return (trades, pending). Trades = resolved (closed) + CENSORED eligible trades; pending = open and NOT censored.
    Only eligible trades (entry_ts >= later of constant and state eligible_from). with_r=False computes no R/cost."""
    elig = max(REG_ELIGIBLE, ts(state["eligible_from"]))
    syms = state["symbols"]
    trades, pending = [], []
    seen = set()
    for c in state["closed"]:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        if ts(c["entry_ts"]) < elig:
            continue
        t = dict(id=c["id"], symbol=c["symbol"], signal_ts=ts(c["signal_ts"]), entry_ts=ts(c["entry_ts"]),
                 exit_ts=ts(c["exit_ts"]), censored=False)
        if with_r:
            entry = float(c["entry"]); risk_pct = float(c["risk_pct"])
            bars = int(c["bars_held"]); xp = float(c["exit_price"]); reason = c["exit_reason"]
            gross = float(c["gross_r"])
            t.update(reason=reason, bars=bars, gross=gross)
            for s in SCEN:
                t[s] = gross - cost_fraction(s, entry, xp, bars, reason) / risk_pct
        trades.append(t)
    for pid, o in state["open"].items():
        oid = o.get("id", pid)
        if oid in seen:
            continue
        seen.add(oid)
        if ts(o["entry_ts"]) < elig:
            continue
        last_ts = syms.get(o["symbol"], {}).get("last_ts")
        cens = last_ts is not None and (commit_time - ts(last_ts)) >= pd.Timedelta(days=CENSOR_DAYS)
        base = dict(id=oid, symbol=o["symbol"], signal_ts=ts(o["signal_ts"]), entry_ts=ts(o["entry_ts"]))
        if not cens:
            pending.append(base)
            continue
        t = dict(base, exit_ts=ts(o["entry_ts"]) + 29 * BAR, censored=True)
        if with_r:
            entry = float(o["entry"]); risk_pct = float(o["risk_pct"]); sl = float(o["sl"])
            c3 = cost_fraction("S3", entry, sl, 30, "SL") / risk_pct
            t["S3_m1"] = -1.0 - c3
            t["S3_m2"] = -2.0 - c3
        trades.append(t)
    return trades, pending


def cluster_status(trades, pending):
    """signal_ts -> dict(resolved, censored, open, status). status: OPEN / NONCENSORED / CENSORED_CLUSTER."""
    by = {}
    for t in trades:
        d = by.setdefault(t["signal_ts"], dict(resolved=0, censored=0, open=0))
        d["censored" if t["censored"] else "resolved"] += 1
    for p in pending:
        by.setdefault(p["signal_ts"], dict(resolved=0, censored=0, open=0))["open"] += 1
    for k, d in by.items():
        d["status"] = "OPEN" if d["open"] > 0 else ("NONCENSORED" if d["resolved"] > 0 else "CENSORED_CLUSTER")
    return by


def select_cohort(trades, pending, n, commit_time):
    """Return (state, cohort_keys, census). state: HOLDS / BLOCKED_OPEN / BLOCKED_INCOMPLETE / SHORT."""
    st = cluster_status(trades, pending)
    census = dict(clusters_total=len(st),
                  clusters_noncensored=sum(1 for d in st.values() if d["status"] == "NONCENSORED"),
                  clusters_open=sum(1 for d in st.values() if d["status"] == "OPEN"),
                  censored_clusters_snapshot=sum(1 for d in st.values() if d["status"] == "CENSORED_CLUSTER"),
                  censored_trades_snapshot=sum(1 for t in trades if t["censored"]),
                  open_trades=len(pending))
    cohort = []
    for k in sorted(st):
        if len(cohort) == n:
            break
        s = st[k]["status"]
        if s == "CENSORED_CLUSTER":
            continue
        if s == "OPEN":
            return "BLOCKED_OPEN", cohort, census
        if commit_time < k + BAR + pd.Timedelta(minutes=MAX_ENTRY_DELAY_MIN):
            return "BLOCKED_INCOMPLETE", cohort, census
        cohort.append(k)
    if len(cohort) < n:
        return "SHORT", cohort, census
    return "HOLDS", cohort, census


def trigger_holds(state, commit_time, n):
    """Counts-only trigger. COUNT-ONLY: with_r=False, so no net R / cost value is computed here."""
    trades, pending = build_trades(state, commit_time, with_r=False)
    stt, cohort, census = select_cohort(trades, pending, n, commit_time)
    info = dict(trigger_state=stt, cohort_size=len(cohort), **census)
    return stt == "HOLDS", info


# ------------------------------------------------------------------ statistics
def cluster_outcomes(trades, keys, scn):
    """Equal-weight mean of RESOLVED trades per cluster; censored trades carry no R here."""
    by = {}
    for t in trades:
        if t["signal_ts"] in keys and not t["censored"]:
            by.setdefault(t["signal_ts"], []).append(t[scn])
    ks = sorted(by)
    return ks, np.array([float(np.mean(by[k])) for k in ks])


def stress_cluster_mean(sel, keys, field):
    """S3 cluster-mean on the SAME cohort with censored trades valued at t[field] (S3_m1 / S3_m2)."""
    by = {}
    for t in sel:
        v = t["S3_m1" if field == "m1" else "S3_m2"] if t["censored"] else t["S3"]
        by.setdefault(t["signal_ts"], []).append(v)
    return float(np.mean([np.mean(by[k]) for k in sorted(by)]))


def boot_ci(means):
    n = len(means)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(DRAWS, n))
    m = means[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def merged_adjacent(trades, keys, scn):
    ks = sorted(keys)
    groups, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b - a == BAR:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)
    outs = []
    for g in groups:
        vals = [t[scn] for t in trades if t["signal_ts"] in g and not t["censored"]]
        outs.append(float(np.mean(vals)))
    return len(groups), float(np.mean(outs))


def max_dd(vals_in_order):
    peak = 0.0; cum = 0.0; dd = 0.0
    for v in vals_in_order:
        cum += v; peak = max(peak, cum); dd = min(dd, cum - peak)
    return dd


def max_concurrent(trades):
    ev = []
    for t in trades:
        end = t["entry_ts"] + 30 * BAR if t["censored"] else t["exit_ts"] + BAR
        ev.append((t["entry_ts"], end))
    best = 0
    for s, _ in ev:
        best = max(best, sum(1 for a, b in ev if a <= s < b))
    return best


def evaluate(state, commit, commit_time, first_created_at, cp, args):
    n = CHECKPOINT_N[cp]
    trades, pending = build_trades(state, commit_time)
    stt, keys, census = select_cohort(trades, pending, n, commit_time)
    assert stt == "HOLDS"
    keyset = set(keys)
    sel = [t for t in trades if t["signal_ts"] in keyset]
    resolved = [t for t in sel if not t["censored"]]
    res = dict(track=TRACK, checkpoint=cp, n_clusters=n, commit=commit, evaluation_instant=commit_time.isoformat(),
               created_at_state=state["created_at"], created_at_first_commit=first_created_at,
               created_at_check="OK" if state["created_at"] == first_created_at else "DEFECT_MISMATCH_LABEL_PERIOD",
               eligible_from=state["eligible_from"], n_trades_resolved=len(resolved),
               censored_trades_in_cohort=sum(1 for t in sel if t["censored"]),
               censored_trades_snapshot=census["censored_trades_snapshot"],
               censored_clusters_snapshot=census["censored_clusters_snapshot"],
               open_trades_outside_cohort=len([p for p in pending if p["signal_ts"] not in keyset]),
               limitation=LIMITATION_TEXT, notes=NOTES)
    res["cluster_keys"] = [k.isoformat() for k in keys]
    stats = {}
    for s in SCEN:
        ks, means = cluster_outcomes(sel, keyset, s)
        lo, hi, pgt = boot_ci(means)
        mg = merged_adjacent(sel, keyset, s)
        stats[s] = dict(cluster_mean=float(means.mean()), ci95_lo=lo, ci95_hi=hi, p_mean_gt0=pgt,
                        trade_mean=float(np.mean([t[s] for t in resolved])), merged_n=mg[0], merged_mean=mg[1])
    res["stats"] = stats
    ks, m3 = cluster_outcomes(sel, keyset, "S3")
    order = sorted(range(len(ks)), key=lambda i: (-m3[i], ks[i]))
    big = order[0]
    rem = [m3[i] for i in range(len(ks)) if i != big]
    jk_cluster = float(np.mean(rem)) if rem else None
    month = {}
    for k, v in zip(ks, m3):
        month.setdefault(k.strftime("%Y-%m"), []).append(v)
    best_m = sorted(month, key=lambda mth: (-sum(month[mth]), mth))[0]
    remm = [v for k, v in zip(ks, m3) if k.strftime("%Y-%m") != best_m]
    jk_month = float(np.mean(remm)) if remm else None
    res["jackknife_S3"] = dict(largest_cluster=ks[big].isoformat(), mean_without_largest=jk_cluster,
                               best_month=best_m, mean_without_best_month=jk_month)
    res["S3_sensitivity_minus1R_cluster_mean"] = stress_cluster_mean(sel, keyset, "m1")
    res["S3_sensitivity_minus2R_cluster_mean"] = stress_cluster_mean(sel, keyset, "m2")
    ordered = sorted(resolved, key=lambda t: (t["exit_ts"], t["signal_ts"], t["symbol"]))
    v = [t["S3"] for t in ordered]
    wins = sum(x for x in v if x > 0); losses = -sum(x for x in v if x < 0)
    res["profit_factor_S3"] = "n/a" if losses == 0 else wins / losses
    res["max_drawdown_R_S3"] = max_dd(v)
    res["max_concurrent_open"] = max_concurrent(sel)
    res["by_month_S3"] = {m: dict(n_clusters=len(x), cluster_mean=float(np.mean(x))) for m, x in sorted(month.items())}
    bysym = {}
    for t in resolved:
        bysym.setdefault(t["symbol"], []).append(t["S3"])
    res["by_symbol_S3"] = {s: dict(n=len(x), mean=float(np.mean(x))) for s, x in sorted(bysym.items())}
    if args.btc_start and args.btc_end:
        res["btc_change_pct"] = (args.btc_end / args.btc_start - 1) * 100
    else:
        res["btc_change_pct"] = "NOT PROVIDED"
    s3 = stats["S3"]
    g_b2 = s3["cluster_mean"] > 0 and res["S3_sensitivity_minus1R_cluster_mean"] <= 0
    g_cl = census["censored_clusters_snapshot"] > 0
    res["guard_b2"] = g_b2
    res["guard_censoring_limitation"] = g_cl
    flags = []
    if g_cl:
        flags.append("CENSORING LIMITATION (fully censored cluster in snapshot)")
    if g_b2:
        flags.append("b2 (primary S3 > 0 but -1R sensitivity <= 0)")
    if s3["cluster_mean"] <= 0:
        res["verdict"] = "REJECTED"
    elif g_cl:
        res["verdict"] = "NOT ESTABLISHED - CENSORING LIMITATION" + (" | also b2 (primary S3 > 0 but -1R sensitivity <= 0)" if g_b2 else "")
        if cp == "UC1":
            res["verdict"] += " | UC1: monitoring continues toward UC2; this is NOT a passed checkpoint"
    elif cp == "UC1":
        res["verdict"] = "CONTINUE_TO_UC2 (no verdict; CI and jackknife descriptive)" + ((" | GUARD FLAGGED: " + "; ".join(flags)) if flags else "")
    elif g_b2:
        res["verdict"] = "NOT ESTABLISHED (b2: -1R sensitivity <= 0)"
    else:
        def pos(x):
            return x is not None and x > 0
        if s3["ci95_lo"] > 0 and pos(jk_cluster) and pos(jk_month):
            res["verdict"] = "PAPER GATE PASSED"
        else:
            res["verdict"] = "NOT ESTABLISHED (final for U67-S v2.1)"
    return res


# ------------------------------------------------------------------------ main
def scan(repo, state_file, cp):
    n = CHECKPOINT_N[cp]
    commits = state_commits(repo, state_file)
    if not commits:
        return None, dict(commits=0), None
    first_state = load_state_at(repo, commits[0][0], state_file)
    first_created = first_state["created_at"] if first_state else None
    missing = []
    last_info = {}
    for h, t in commits:
        st = load_state_at(repo, h, state_file)
        if st is None:
            missing.append(h)
            continue
        ok, info = trigger_holds(st, t, n)
        last_info = dict(commit=h, commit_time=t.isoformat(), **info)
        if ok:
            return (h, t, st, first_created), dict(commits=len(commits), missing_state_commits=missing, **last_info), first_created
    last_t = commits[-1][1]
    elapsed = (last_t - REG_ELIGIBLE) / pd.Timedelta(days=1)
    cap = dict(elapsed_days=round(float(elapsed), 2), cap_days=CAP_DAYS[cp], cap_reached=bool(elapsed >= CAP_DAYS[cp]),
               outcome=("OBSERVED RATE REPORTED; NO VERDICT" if elapsed >= CAP_DAYS[cp] else "cap not reached"))
    return None, dict(commits=len(commits), missing_state_commits=missing, rate_cap=cap, **last_info), first_created


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--state", default=STATE_FILE_DEFAULT)
    ap.add_argument("--checkpoint", choices=["UC1", "UC2"], required=True)
    ap.add_argument("--mode", choices=["scan", "evaluate"], default="scan")
    ap.add_argument("--btc-start", type=float)
    ap.add_argument("--btc-end", type=float)
    a = ap.parse_args()
    hit, info, first_created = scan(a.repo, a.state, a.checkpoint)
    if a.mode == "scan" or hit is None:
        out = dict(checkpoint=a.checkpoint, trigger_holds=hit is not None, counts=info)
        if hit is not None:
            out["first_satisfying_commit"] = hit[0]
            out["instant"] = hit[1].isoformat()
        print(json.dumps(out, sort_keys=True, indent=1))
        if a.mode == "evaluate" and hit is None:
            print("REFUSED: trigger does not hold; no R value is computed or shown.", file=sys.stderr)
            sys.exit(3)
        return
    h, t, st, fc = hit
    res = evaluate(st, h, t, fc, a.checkpoint, a)
    txt = json.dumps(res, sort_keys=True, indent=1, default=str)
    print(txt)
    print("RESULT_SHA256 " + hashlib.sha256(txt.encode()).hexdigest())


if __name__ == "__main__":
    main()
