"""
C11 POST-BOUNDARY EXECUTION-REALISM DIAGNOSTIC - ALL SIGNALS (READ-ONLY)
Writes NO file, edits NO state, commits NOTHING, places NO orders, changes NO rule. Output: log only.
Inputs : candidate11_paper_state.json (frozen records), GitHub Actions API (run list + run logs, read-only),
         git history of the state file (first commit that contains each signal), PUBLIC KuCoin Futures klines.
Per signal: exact RUN_TIME_UTC from the Actions log of the run that recorded it (commit-time proxy only if the log is unavailable, labelled),
         the 1-minute OHLCV bar containing RUN_TIME_UTC (PROXY bracket, not a quote), diff vs frozen entry in % and frozen-R,
         classification, and R-to-SL / R-to-TP from the processing-time price with the UNCHANGED absolute SL/TP.
SHORT convention: shift_R = (entry - price)/risk. shift_R > 0: real short worse than frozen entry (frozen fill OPTIMISTIC = 'favorable' to the trade);
         shift_R < 0: real short better (frozen fill CONSERVATIVE = 'unfavorable' to the trade). Bracket containing the entry = INDETERMINATE.
"""
import io, json, os, re, statistics, subprocess, sys, time, zipfile
from datetime import datetime, timedelta, timezone
import requests

STATE = "candidate11_paper_state.json"
WORKFLOW = "paper_v4.yml"
BOUNDARY = "2026-10-21T08:00:00+00:00"          # first post-B signal candle (signal_ts >= this)
AUDITED = {"ATOMUSDTM|2026-11-05T12:00:00+00:00", "BCHUSDTM|2026-11-05T12:00:00+00:00"}
NORMAL_MAX_MIN = 10.0                            # lag <= 10 min = normal window
KLINE = "https://api-futures.kucoin.com/api/v1/kline/query"
GH = "https://api.github.com"
REPO = os.environ.get("GITHUB_REPOSITORY", "")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
S = requests.Session()
S.headers.update({"User-Agent": "c11-exec-diag-all/1.0 (read-only)"})


def utc(msv): return datetime.fromtimestamp(msv / 1000, tz=timezone.utc)
def ms(dt): return int(dt.timestamp() * 1000)
def P(dt): return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)


def kget(symbol, gran, start, end, tries=4):
    last = None
    for k in range(tries):
        try:
            r = S.get(KLINE, params={"symbol": symbol, "granularity": gran, "from": ms(start), "to": ms(end)}, timeout=25)
            if r.status_code == 200:
                p = r.json()
                if p.get("code") == "200000":
                    out = {}
                    for x in p.get("data") or []:
                        t = int(x[0]); t = t * 1000 if t < 10_000_000_000 else t
                        out[t] = tuple(float(v) for v in x[1:6])
                    return out
                last = str(p)[:120]
            else:
                last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(2 ** k, 6))
    print("FETCH_FAILED", symbol, gran, last)
    return None


def gh(path, params=None, raw=False):
    h = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    r = requests.get(GH + path, headers=h, params=params, timeout=60, allow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API {path} HTTP {r.status_code}")
    return r.content if raw else r.json()


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, check=False).stdout


def first_commit_of_ids():
    """first commit (time) whose state file contains each signal id"""
    seen = {}
    rows = [l.split("|") for l in sh("git", "log", "--reverse", "--format=%H|%cI", "--", STATE).splitlines() if l]
    for h, ct in rows:
        try:
            d = json.loads(sh("git", "show", f"{h}:{STATE}"))
        except Exception:
            continue
        for x in d.get("signals", []):
            seen.setdefault(x["id"], (h[:7], P(ct)))
    return seen, len(rows)


def list_runs(since):
    runs, page = [], 1
    while page <= 30:
        j = gh(f"/repos/{REPO}/actions/workflows/{WORKFLOW}/runs",
               {"per_page": 100, "page": page, "created": f">={since:%Y-%m-%d}"})
        rs = j.get("workflow_runs", [])
        runs += rs
        if len(rs) < 100:
            break
        page += 1
    return runs


LOGCACHE = {}
def run_log_info(run):
    rid = run["id"]
    if rid in LOGCACHE:
        return LOGCACHE[rid]
    info = None
    try:
        z = zipfile.ZipFile(io.BytesIO(gh(f"/repos/{REPO}/actions/runs/{rid}/logs", raw=True)))
        txt = "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".txt"))
        m = re.search(r"RUN_TIME_UTC\s*=\s*(\S+)", txt)
        o = re.search(r"THIS RUN: NEW_SIGNALS=(\d+) OPENED=(\d+)", txt)
        if m:
            info = {"t": P(m.group(1)), "opened": int(o.group(2)) if o else None, "new": int(o.group(1)) if o else None}
    except Exception as e:
        print(f"  log unavailable run {rid}: {e!r}")
    LOGCACHE[rid] = info
    return info


def classify(e, lo, hi):
    if lo > e: return "UNFAVORABLE(frozen conservative; whole bracket above entry)"
    if hi < e: return "FAVORABLE(frozen optimistic; whole bracket below entry)"
    return "INDETERMINATE(bracket contains entry)"


def med(v): return statistics.median(v) if v else float("nan")


def main():
    d = json.load(open(STATE, encoding="utf-8"))
    sigs = sorted([x for x in d["signals"] if x["signal_ts"] >= BOUNDARY], key=lambda x: (x["entry_ts"], x["symbol"]))
    print(f"C11 POST-BOUNDARY EXECUTION-REALISM (read-only). signals in state with signal_ts >= {BOUNDARY}: {len(sigs)} (state total {len(d['signals'])})")
    first, ncommits = first_commit_of_ids()
    print(f"state-file commits scanned: {ncommits}")
    since = P(min(x["entry_ts"] for x in sigs)) - timedelta(days=1)
    try:
        runs = list_runs(since)
        print(f"workflow runs listed since {since:%Y-%m-%d}: {len(runs)}")
    except Exception as e:
        runs = []
        print("RUN LIST UNAVAILABLE:", repr(e))
    runs_t = []
    for r in runs:
        st = r.get("run_started_at") or r.get("created_at")
        if st and r.get("conclusion") in ("success", "failure"):
            runs_t.append((P(st), r))
    runs_t.sort(key=lambda x: x[0])

    rows = []
    for x in sigs:
        sym, e, risk, sl, tp = x["symbol"], float(x["entry"]), float(x["risk"]), float(x["sl"]), float(x["tp"])
        ets = P(x["entry_ts"])
        fc = first.get(x["id"])
        T, tsrc, rid = None, None, None
        if fc:
            cands = [r for st, r in runs_t if ets <= st <= fc[1] + timedelta(minutes=1)]
            best = None
            for r in cands:
                inf = run_log_info(r)
                if inf and ets <= inf["t"] <= fc[1] + timedelta(seconds=5):
                    if best is None or inf["t"] > best[0]:
                        best = (inf["t"], r["id"])
            if best:
                T, rid, tsrc = best[0], best[1], "EXACT(Actions log RUN_TIME_UTC)"
            else:
                T, tsrc = fc[1] - timedelta(minutes=0.5), "PROXY(first state commit time - 30s; log not matched)"
        if T is None:
            print(f"{sym} {x['signal_ts']}: no processing time resolvable - skipped"); continue
        lag = (T - ets).total_seconds() / 60
        k1 = kget(sym, 1, T.replace(second=0, microsecond=0) - timedelta(minutes=2), T + timedelta(minutes=2))
        k4 = kget(sym, 240, ets - timedelta(hours=4), ets + timedelta(hours=4))
        tm = ms(T.replace(second=0, microsecond=0))
        bar = k1.get(tm) if k1 else None
        o4 = k4.get(ms(ets)) if k4 else None
        match = "N/A" if not o4 else ("MATCH" if abs(o4[0] - e) <= 1e-9 * max(1, e) else f"DIFF(exch {o4[0]})")
        rec = {"id": x["id"], "sym": sym, "sig": x["signal_ts"][5:16], "T": T, "lag": lag, "src": tsrc, "run": rid, "commit": fc[0] if fc else None,
               "match": match, "audited": x["id"] in AUDITED}
        if bar:
            o, h, l, c, v = bar
            rec.update(bar=bar, d_pct={k: 100 * (p / e - 1) for k, p in (("o", o), ("c", c), ("l", l), ("h", h))},
                       sh={k: (e - p) / risk for k, p in (("o", o), ("c", c), ("l", l), ("h", h))},
                       rtp={k: (p - tp) / risk for k, p in (("o", o), ("c", c), ("l", l), ("h", h))},
                       rsl={k: (sl - p) / risk for k, p in (("o", o), ("c", c), ("l", l), ("h", h))},
                       cls=classify(e, l, h), risk_pct=100 * risk / e)
        rows.append(rec)
        print("-" * 78)
        print(f"{sym} sig {x['signal_ts']} entry_ts {x['entry_ts']}{'  [ALREADY AUDITED]' if rec['audited'] else ''}")
        print(f"  frozen entry {e} risk {risk:.8g} ({100 * risk / e:.3f}%) SL {sl:.8g} TP {tp:.8g}; exchange 4H open check: {match}")
        print(f"  T = {T.isoformat()} [{tsrc}] run_id {rid} first-state-commit {rec['commit']}  lag {lag:.2f} min")
        if bar:
            print(f"  1-min bar {utc(tm):%H:%MZ} O {o} H {h} L {l} C {c} (PROXY)")
            print(f"  diff% vs entry: open {rec['d_pct']['o']:+.3f} close {rec['d_pct']['c']:+.3f} low {rec['d_pct']['l']:+.3f} high {rec['d_pct']['h']:+.3f}")
            print(f"  shift_R: open {rec['sh']['o']:+.3f} close {rec['sh']['c']:+.3f} low {rec['sh']['l']:+.3f} high {rec['sh']['h']:+.3f}  (bracket {min(rec['sh'].values()):+.3f}..{max(rec['sh'].values()):+.3f})")
            print(f"  R-to-TP from price: {min(rec['rtp'].values()):.2f}..{max(rec['rtp'].values()):.2f} (frozen 2.00)  R-to-SL: {min(rec['rsl'].values()):.2f}..{max(rec['rsl'].values()):.2f} (frozen 1.00)")
            print(f"  class: {rec['cls']}")
        else:
            print("  1-min bar unavailable - not classified")
        time.sleep(0.2)

    ok = [r for r in rows if "bar" in r]
    print("=" * 78)
    print(f"COHORT: {len(rows)} signals resolved, {len(ok)} with a 1-min bar; exact-T {sum(r['src'].startswith('EXACT') for r in rows)}, proxy-T {sum(r['src'].startswith('PROXY') for r in rows)}; 4H-open MATCH {sum(r['match']=='MATCH' for r in rows)}/{len(rows)}")

    def block(name, grp):
        print(f"--- {name}: n={len(grp)}")
        if not grp: return
        for cl in ("UNFAV", "FAV", "INDET"):
            pass
        n_un = sum(r["cls"].startswith("UNFAV") for r in grp)
        n_fa = sum(r["cls"].startswith("FAV") for r in grp)
        n_in = sum(r["cls"].startswith("INDET") for r in grp)
        print(f"  UNFAVORABLE(frozen conservative)={n_un}  FAVORABLE(frozen optimistic)={n_fa}  INDETERMINATE={n_in}")
        dc = [r["d_pct"]["c"] for r in grp]; sc = [r["sh"]["c"] for r in grp]
        lo_d = [min(r["d_pct"].values()) for r in grp]; hi_d = [max(r["d_pct"].values()) for r in grp]
        lo_s = [min(r["sh"].values()) for r in grp]; hi_s = [max(r["sh"].values()) for r in grp]
        print(f"  diff% (close proxy): median {med(dc):+.3f}  range {min(dc):+.3f}..{max(dc):+.3f};  full bracket envelope {min(lo_d):+.3f}..{max(hi_d):+.3f}")
        print(f"  shift_R (close proxy): median {med(sc):+.3f}  range {min(sc):+.3f}..{max(sc):+.3f};  full bracket envelope {min(lo_s):+.3f}..{max(hi_s):+.3f}")
        print(f"  mean shift_R (close) {statistics.mean(sc):+.3f}; positive(close) {sum(v>0 for v in sc)}, negative(close) {sum(v<0 for v in sc)}")
        print(f"  R-to-SL median {med([statistics.mean(r['rsl'].values()) for r in grp]):.2f}; R-to-TP median {med([statistics.mean(r['rtp'].values()) for r in grp]):.2f}")
    normal = [r for r in ok if r["lag"] <= NORMAL_MAX_MIN]
    delayed = [r for r in ok if r["lag"] > NORMAL_MAX_MIN]
    block("ALL", ok)
    block(f"NORMAL windows (lag <= {NORMAL_MAX_MIN:g} min)", normal)
    block(f"DELAYED windows (lag > {NORMAL_MAX_MIN:g} min)", delayed)
    block("DELAYED excluding the 2 already-audited (ATOM,BCH)", [r for r in delayed if not r["audited"]])
    block("ALL excluding the 2 already-audited", [r for r in ok if not r["audited"]])
    print("lags (min): " + ", ".join(f"{r['sym'][:-4]}@{r['sig']}:{r['lag']:.1f}" for r in rows))

    print("=" * 78)
    print("INTEGRITY CHECKS (git history, read-only)")
    wf = sh("git", "ls-files", ".github/workflows").split()
    refs = [f for f in wf if f != ".github/workflows/c11_exec_diag.yml" and f != ".github/workflows/c11_exec_diag_all.yml"
            and any(s in open(f, encoding="utf-8", errors="replace").read() for s in ("c11_exec_diag", "Execution Diagnostic"))]
    print("workflow files referencing the diagnostic (other than itself):", refs or "NONE")
    for f in (".github/workflows/c11_exec_diag.yml", ".github/workflows/c11_exec_diag_all.yml"):
        if os.path.exists(f):
            t = open(f, encoding="utf-8").read()
            on_block = t.split("jobs:")[0]
            has_push = bool(re.search(r"^\s*push\s*:", on_block, re.M))
            perms = re.findall(r"(contents|actions):\s*(\w+)", on_block)
            gitw = ("git push" in t) or ("git commit" in t)
            print(f"{f}: schedule={'schedule' in on_block}, cron={'cron' in on_block}, push={has_push}, workflow_dispatch={'workflow_dispatch' in on_block}, permissions={perms}, git commit/push in file={gitw}")
    for f in ("c11_exec_diag.py", ".github/workflows/c11_exec_diag.yml", "c11_exec_diag_all.py", ".github/workflows/c11_exec_diag_all.yml"):
        logs = sh("git", "log", "--format=%h %cI %an", "--", f).strip().splitlines()
        print(f"history of {f}: {len(logs)} commit(s) {logs[:3]}")
        for l in logs:
            h = l.split()[0]
            files = sh("git", "show", "--name-only", "--format=", h).split()
            print(f"   commit {h} touches: {files}")
    track = ["crypto_signal_bot_v4.py", "candidate11_paper_state.json", "long_m1_paper_state.json", "u67_short_paper_state.json", "u67_long_paper_state.json",
            "crypto_signal_bot_long_v1.py", "crypto_signal_bot_u67_long_v1.py", "crypto_signal_bot_u67_short_v1.py",
            ".github/workflows/paper_v4.yml", ".github/workflows/paper_long_v1.yml", ".github/workflows/paper_u67_long_v1.yml", ".github/workflows/paper_u67_short_v1.yml"]
    diag_commits = set()
    for f in ("c11_exec_diag.py", ".github/workflows/c11_exec_diag.yml", "c11_exec_diag_all.py", ".github/workflows/c11_exec_diag_all.yml"):
        diag_commits |= {l.split()[0] for l in sh("git", "log", "--format=%h", "--", f).splitlines() if l}
    bad = []
    for h in diag_commits:
        files = set(sh("git", "show", "--name-only", "--format=", h).split())
        bad += [(h, f) for f in files if f in track]
    print("diagnostic-adding commits that also touch any track file:", bad or "NONE")
    print("engine sha256:", sh("sha256sum", "crypto_signal_bot_v4.py").split()[0] if os.path.exists("crypto_signal_bot_v4.py") else "n/a")
    print("Nothing was written to the repository by this script.")


if __name__ == "__main__":
    main()
