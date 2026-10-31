#!/usr/bin/env python3
# U67-S EVALUATOR v1 (read-only).
# Implements U67_S_REGISTRATION_v1.txt sections 0e, 6, 7, 8 and the UNRESOLVED rule.
# Registration sha256: 9efe88d1b07c166be1e4e371e3c2339fc941194b22f8cf03fd5c8d8fe879d8be
#
# Properties:
#  - Never modifies the repository or any state file (git show / git log only).
#  - Phase 1 "scan": prints COUNTS ONLY (no R value anywhere).
#  - Phase 2 "evaluate": refuses to run unless the trigger holds in git history;
#    always evaluates the FIRST committed state that satisfies the trigger,
#    never HEAD. Evaluation instant = that commit's committer time.
#  - Deterministic: no clock reads, fixed seed, fixed ordering.
#
# Implementation choices NOT spelled out in the registration (declared here,
# covered by this script's sha256, changeable only under registration section 9):
#  1. Bootstrap: for each scenario a fresh numpy default_rng(20261028) draws
#     rng.integers(0, n, size=(20000, n)); statistic = mean over the drawn
#     cluster means. Identical resample indices across scenarios (paired).
#  2. "Commit time" = git committer date (%cI) of the state commit.
#  3. Cluster complete = commit_time >= entry_ts + 90 minutes.
#  4. Max concurrent open trades: a trade occupies [entry_ts, exit_ts + 4h)
#     (UNRESOLVED: [entry_ts, entry_ts + 30 x 4h)); counted at interval starts.
#  5. Adjacent-merge sensitivity: strict clusters whose signal_ts differ by
#     exactly 4h are chained; merged outcome = equal-weight mean of all trades
#     of the chained clusters; statistic = mean of merged outcomes.
#  6. Profit factor and drawdown use the S3 net R of trades.
#  7. BTC change is only printed if --btc-start and --btc-end are supplied.
#  8. Trigger detection (scan, and the trigger check inside evaluate) is COUNT-ONLY:
#     it builds trades with with_r=False, so no gross/net R or cost is computed before
#     the trigger holds. The state JSON is parsed whole (it contains R fields as plain
#     data) but none of them is read by the scan path.
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
UNRESOLVED_DAYS = 45
MAX_ENTRY_DELAY_MIN = 90
BAR = pd.Timedelta(hours=4)
SCEN = ["S0", "S3", "S3a", "S3b", "S1", "S2"]
CHECKPOINT_N = {"UC1": 15, "UC2": 30}
CAP_DAYS = {"UC1": 120, "UC2": 240}


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
    """Return list of trade dicts (closed + UNRESOLVED) and list of still-open trades
    that are NOT unresolved. Only eligible trades (entry_ts >= later of constant,
    state eligible_from)."""
    elig = max(REG_ELIGIBLE, ts(state["eligible_from"]))
    syms = state["symbols"]
    trades, open_pending = [], []
    seen = set()
    for c in state["closed"]:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        if ts(c["entry_ts"]) < elig:
            continue
        t = dict(id=c["id"], symbol=c["symbol"], signal_ts=ts(c["signal_ts"]), entry_ts=ts(c["entry_ts"]),
                 exit_ts=ts(c["exit_ts"]), unresolved=False)
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
        age_ok = last_ts is not None and (commit_time - ts(last_ts)) >= pd.Timedelta(days=UNRESOLVED_DAYS)
        if not age_ok:
            open_pending.append(dict(id=oid, symbol=o["symbol"], signal_ts=ts(o["signal_ts"]), entry_ts=ts(o["entry_ts"])))
            continue
        t = dict(id=oid, symbol=o["symbol"], signal_ts=ts(o["signal_ts"]), entry_ts=ts(o["entry_ts"]),
                 exit_ts=ts(o["entry_ts"]) + 29 * BAR, unresolved=True)
        if with_r:
            entry = float(o["entry"]); risk_pct = float(o["risk_pct"]); sl = float(o["sl"])
            t.update(reason="SL", bars=30, gross=-1.0)
            for s in SCEN:
                t[s] = -1.0 - cost_fraction(s, entry, sl, 30, "SL") / risk_pct
        trades.append(t)
    return trades, open_pending


def first_n_clusters(trades, pending, n):
    """Strict clusters ordered by signal_ts, over OPENED eligible trades (closed, unresolved, still open)."""
    by = {}
    for t in trades:
        by.setdefault(t["signal_ts"], []).append(t)
    for p in pending:
        by.setdefault(p["signal_ts"], [])
    keys = sorted(by)
    if len(keys) < n:
        return None
    return keys[:n], by


def trigger_holds(state, commit_time, n):
    """Counts-only trigger. Returns (bool, info dict of counts)."""
    # COUNT-ONLY: with_r=False, so no net R / cost value is computed here.
    trades, pending = build_trades(state, commit_time, with_r=False)
    info = dict(trades=len(trades), pending_open=len(pending), clusters=0)
    keys_by = first_n_clusters(trades, pending, n)
    allk = sorted({t["signal_ts"] for t in trades} | {p["signal_ts"] for p in pending})
    info["clusters"] = len(allk)
    if keys_by is None:
        return False, info
    keys, by = keys_by
    pend_ts = {p["signal_ts"] for p in pending}
    for k in keys:
        entry_ts = k + BAR
        if commit_time < entry_ts + pd.Timedelta(minutes=MAX_ENTRY_DELAY_MIN):
            return False, info
        if k in pend_ts:
            return False, info
    return True, info


# ------------------------------------------------------------------ statistics
def cluster_outcomes(trades, keys, scn):
    by = {}
    for t in trades:
        if t["signal_ts"] in keys:
            by.setdefault(t["signal_ts"], []).append(t[scn])
    ks = sorted(by)
    return ks, np.array([float(np.mean(by[k])) for k in ks])


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
        vals = [t[scn] for t in trades if t["signal_ts"] in g]
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
        end = t["entry_ts"] + 30 * BAR if t["unresolved"] else t["exit_ts"] + BAR
        ev.append((t["entry_ts"], end))
    best = 0
    for s, _ in ev:
        best = max(best, sum(1 for a, b in ev if a <= s < b))
    return best


def evaluate(state, commit, commit_time, first_created_at, cp, args):
    n = CHECKPOINT_N[cp]
    trades, pending = build_trades(state, commit_time)
    keys, _ = first_n_clusters(trades, pending, n)
    keyset = set(keys)
    sel = [t for t in trades if t["signal_ts"] in keyset]
    res = dict(checkpoint=cp, n_clusters=n, commit=commit, evaluation_instant=commit_time.isoformat(),
               created_at_state=state["created_at"], created_at_first_commit=first_created_at,
               created_at_check="OK" if state["created_at"] == first_created_at else "DEFECT_MISMATCH_LABEL_PERIOD",
               eligible_from=state["eligible_from"], n_trades=len(sel),
               n_unresolved=sum(1 for t in sel if t["unresolved"]),
               open_outside_selected_clusters=len([p for p in pending if p["signal_ts"] not in keyset]))
    res["cluster_keys"] = [k.isoformat() for k in keys]
    stats = {}
    for s in SCEN:
        ks, means = cluster_outcomes(sel, keyset, s)
        lo, hi, pgt = boot_ci(means)
        stats[s] = dict(cluster_mean=float(means.mean()), ci95_lo=lo, ci95_hi=hi, p_mean_gt0=pgt,
                        trade_mean=float(np.mean([t[s] for t in sel])),
                        merged_n=merged_adjacent(sel, keyset, s)[0], merged_mean=merged_adjacent(sel, keyset, s)[1])
    res["stats"] = stats
    # S3 jackknife
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
    # UNRESOLVED exclusion sensitivity (S3)
    nu = [t for t in sel if not t["unresolved"]]
    if nu and res["n_unresolved"]:
        ks2 = {t["signal_ts"] for t in nu}
        _, m2 = cluster_outcomes(nu, ks2, "S3")
        res["S3_cluster_mean_excluding_unresolved"] = float(m2.mean())
    else:
        res["S3_cluster_mean_excluding_unresolved"] = res["stats"]["S3"]["cluster_mean"]
    # descriptive
    ordered = sorted(sel, key=lambda t: (t["exit_ts"], t["signal_ts"], t["symbol"]))
    v = [t["S3"] for t in ordered]
    wins = sum(x for x in v if x > 0); losses = -sum(x for x in v if x < 0)
    res["profit_factor_S3"] = "n/a" if losses == 0 else wins / losses
    res["max_drawdown_R_S3"] = max_dd(v)
    res["max_concurrent_open"] = max_concurrent(sel)
    res["by_month_S3"] = {m: dict(n_clusters=len(x), cluster_mean=float(np.mean(x))) for m, x in sorted(month.items())}
    bysym = {}
    for t in sel:
        bysym.setdefault(t["symbol"], []).append(t["S3"])
    res["by_symbol_S3"] = {s: dict(n=len(x), mean=float(np.mean(x))) for s, x in sorted(bysym.items())}
    if args.btc_start and args.btc_end:
        res["btc_change_pct"] = (args.btc_end / args.btc_start - 1) * 100
    else:
        res["btc_change_pct"] = "NOT PROVIDED"
    # decision
    s3 = stats["S3"]
    if cp == "UC1":
        res["verdict"] = "REJECTED" if s3["cluster_mean"] <= 0 else "CONTINUE_TO_UC2 (no verdict; CI and jackknife descriptive)"
    else:
        if s3["cluster_mean"] <= 0:
            res["verdict"] = "REJECTED"
        else:
            def pos(x):
                return x is not None and x > 0
            if s3["ci95_lo"] > 0 and pos(jk_cluster) and pos(jk_month):
                res["verdict"] = "PAPER GATE PASSED"
            else:
                res["verdict"] = "NOT ESTABLISHED (final for U67-S v1)"
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
    return None, dict(commits=len(commits), missing_state_commits=missing, **last_info), first_created


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
        # COUNTS ONLY
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
