import importlib.util, sys, os, json, time, hashlib, subprocess, tempfile, shutil, copy, re, io, contextlib
import numpy as np, pandas as pd, yaml

REPO = "/home/claude/amirrezapour21-source/crypto-signal-bot/"
BASE = "/mnt/user-data/outputs/micromap/stage3/u67l/"
ENG = os.environ.get("U67L_ENG", BASE + "engine/crypto_signal_bot_u67_long_v1.py")
WF = os.environ.get("U67L_WF", BASE + "engine/paper_u67_long_v3.yml")
REG = os.environ.get("U67L_REG", BASE + "U67_L_REGISTRATION_v1.txt")
TMP = tempfile.mkdtemp(prefix="u67lt_")
os.environ["U67L_STATE_FILE"] = os.path.join(TMP, "u67_long_paper_state.json")
os.environ["PAPER_STATE_FILE"] = os.path.join(TMP, "c11_dummy.json")
os.environ["LONG_M1_STATE_FILE"] = os.path.join(TMP, "lm1_dummy.json")
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
C = load("c11", REPO + "crypto_signal_bot_v4.py")
M = load("lm1", REPO + "crypto_signal_bot_long_v1.py")
L = load("u67l", ENG)
fails = []
def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c: fails.append(msg)

# ---------- 1 constants / universe / registration ----------
for k in ["ZSCORE_N","EMA_N","RANGE_N","VOL_N","ATR_N","SL_ATR","TP_R","HOLD","MAX_BARS","MAX_ENTRY_DELAY_MIN","MIN_CANDLES","BAR","GRANULARITY","CHUNK","N_CHUNKS","OVERLAP","FEE_RT","RETRIES","MAX_FAIL_FRAC","MAX_SIGNALS","MAX_MISSED","MAX_ERRORS","MAX_CLOSED","PAUSE","REQUEST_TIMEOUT","BAR_SECONDS","S3_FEE_PER_SIDE","S3_FUNDING_PER_8H"]:
    check(getattr(M, k) == getattr(L, k), "const equals LONG-M1: " + k)
fin = json.load(open(REPO + "u67_universe_final.json"))
check(L.SYMS == fin["selected_symbols"] and len(L.SYMS) == 67 and len(set(L.SYMS)) == 67, "67 symbols == final.json order")
check(hashlib.sha256("\n".join(L.SYMS).encode()).hexdigest() == "5123e95389eaab880b202b740ce9c7bc54f5a40451e4d40b741ac210e4d21192", "list sha256 equals registered")
check(not (set(L.SYMS) & set(C.SYMS)), "disjoint from C11/LONG-M1 33 symbols")
sh = importlib.util.spec_from_file_location("x", BASE + "../u67/engine/crypto_signal_bot_u67_short_v1.py"); SH = importlib.util.module_from_spec(sh)
os.environ["U67_STATE_FILE"] = os.path.join(TMP, "u67s_dummy.json"); sh.loader.exec_module(SH)
check(L.SYMS == SH.SYMS, "same 67 symbols, same order as U67-S engine")
bad = set(fin["designer_excluded"]) | set(fin["non_crypto_evidenced"])
check(not (bad & set(L.SYMS)), "no excluded/non-crypto symbol in list")
check(L.STATE_FILE.endswith("u67_long_paper_state.json") and L.STATE_VERSION == 1 and L.STATE_FILE != SH.STATE_FILE and L.STATE_FILE != M.STATE_FILE, "own state file, distinct from U67-S/LONG-M1")
reg = open(REG).read()
m = re.search(r"Freeze timestamp \(UTC\): (\S+)\. ELIGIBLE_FROM .*?: (\S+)\.", reg)
check(m and pd.Timestamp(m.group(2)) > pd.Timestamp(m.group(1)) and pd.Timestamp(m.group(2)) == L.REGISTRATION_ELIGIBLE_FROM, "registration literals: ELIGIBLE_FROM strictly after freeze and equal to engine constant")
src = open(ENG).read()
reg_sha = hashlib.sha256(reg.encode()).hexdigest()
check(("(sha256 " + reg_sha + ")") in src, "engine header carries the sha256 of this exact registration text")
check("candidate11_paper_state" not in src and "long_m1_paper_state" not in src and "u67_short" not in src and "LONG_M1_STATE_FILE" not in src, "engine source references no other track's state file")
# engine = LONG-M1 engine except registered differences
import difflib
a = open(REPO + "crypto_signal_bot_long_v1.py").read().split("\n"); b = src.split("\n")
d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if (l.startswith("+") or l.startswith("-")) and not l.startswith("+++") and not l.startswith("---")]
d_nosym = [l for l in d if not re.match(r"^[+-]\s+\"[A-Z0-9]+USDTM\",$", l)]
allowed = ("U67-L", "LONG-M1", "u67", "U67L", "long_m1", "Copy of the LONG-M1", "Mirror of Candidate 11", "REGISTRATION_ELIGIBLE", "2026-10-", "sha256", "(sha256", "LONG_M1_REG", "Registration section 0", "LONG-M1 rules", "Pre-registration", "LONG_M1_STATE_FILE", "long-m1-paper-v1", "long_m1_long_only", "long_m1_paper_state", "no R values are reported before")
bad_d = [l for l in d_nosym if not any(t in l for t in allowed)]
check(not bad_d, "diff vs LONG-M1 engine limited to registered differences (symbols, names, eligibility literal, headers)" + ("" if not bad_d else " :: " + str(bad_d[:5])))
check(len([l for l in d if re.match(r"^\+\s+\"[A-Z0-9]+USDTM\",$", l)]) >= 60, "symbol list replaced (>=60 added symbol lines)")

# ---------- 2 synth + equivalence ----------
def synth(seed, n, start="2020-01-01"):
    r = np.random.default_rng(seed)
    ret = r.standard_t(3, n) * 0.012 + 0.0004
    close = 100 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]] * (1 + r.normal(0, 0.002, n))
    hi = np.maximum(op, close) * (1 + np.abs(r.normal(0, 0.006, n)))
    lo = np.minimum(op, close) * (1 - np.abs(r.normal(0, 0.006, n)))
    vol = np.exp(r.normal(10, 0.6, n))
    ts = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(dict(ts=ts, open=op, high=hi, low=lo, close=close, volume=vol))
def mirror(df):
    Mx = 3 * df["high"].max(); m = df.copy()
    m["open"] = Mx - df["open"]; m["high"] = Mx - df["low"]; m["low"] = Mx - df["high"]; m["close"] = Mx - df["close"]; return m
X = synth(1, 1500)
check(C.prepare(X).equals(L.prepare(X)) and M.prepare(X).equals(L.prepare(X)), "prepare identical to C11 and LONG-M1")
tot = 0; ok = True; ok2 = True
for seed in range(2, 14):
    X = synth(seed, 4000); dl = L.prepare(X); dm = M.prepare(X); dc = C.prepare(mirror(X))
    a1 = [L.signal_reason(dl, i) for i in range(len(X))]; a2 = [M.signal_reason(dm, i) for i in range(len(X))]; a3 = [C.signal_reason(dc, i) for i in range(len(X))]
    tot += a1.count("SIGNAL"); ok &= (a1 == a2); ok2 &= ([i for i, x in enumerate(a1) if x == "SIGNAL"] == [i for i, x in enumerate(a3) if x == "SIGNAL"])
check(ok and tot >= 30, f"signal_reason identical to LONG-M1 on 12 series ({tot} LONG signals)")
check(ok2, "LONG signals on X == C11 SHORT signals on mirror(X) (12 series)")
d0 = L.prepare(synth(3, 600)).copy(); i = 400
d0.loc[i-1, "z20"] = -2.0; d0.loc[i, "z20"] = -1.99
d0.loc[i, "close"] = d0.loc[i, "ema200"] + 1; d0.loc[i, "range20"] = d0.loc[i, "range_median20"]; d0.loc[i, "volume"] = d0.loc[i, "vol_median20"]
check(L.signal_reason(d0, i) == "SIGNAL", "boundary: prev z==-2.0, cur>-2, range==median, vol==median -> SIGNAL")
for nm, mod, exp in [("prev z -1.99", lambda d: d.__setitem__((i-1, "z20"), -1.99), "NO_Z_CROSS"), ("cur z == -2.0", lambda d: d.__setitem__((i, "z20"), -2.0), "NO_Z_CROSS"),
                     ("close == ema200", lambda d: d.__setitem__((i, "close"), d.loc[i, "ema200"]), "BELOW_EMA200"),
                     ("range below median", lambda d: d.__setitem__((i, "range20"), d.loc[i, "range_median20"] - 1e-9), "LOW_RANGE"),
                     ("volume below median", lambda d: d.__setitem__((i, "volume"), d.loc[i, "vol_median20"] - 1e-9), "LOW_VOLUME")]:
    dd = d0.copy()
    if nm == "prev z -1.99": dd.loc[i-1, "z20"] = -1.99
    elif nm == "cur z == -2.0": dd.loc[i, "z20"] = -2.0
    elif nm == "close == ema200": dd.loc[i, "close"] = dd.loc[i, "ema200"]
    elif nm == "range below median": dd.loc[i, "range20"] = dd.loc[i, "range_median20"] - 1e-9
    else: dd.loc[i, "volume"] = dd.loc[i, "vol_median20"] - 1e-9
    check(L.signal_reason(dd, i) == exp, f"boundary: {nm} -> {exp}")

OLD = "2000-01-01T00:00:00+00:00"
def simulate(mod, df, sym, eligible=None, N0=300):
    st = mod.new_state()
    if eligible is not None: st["eligible_from"] = eligible
    for k in range(N0, len(df) - 1):
        raw = df.iloc[:k+1].copy().reset_index(drop=True)
        now = df["ts"].iloc[k] + pd.Timedelta(minutes=10)
        mod.process_symbol(st, sym, raw, now)
    return st
ncmp = 0; ok = True; tp = sl = to = 0; okm = True
for seed in (21, 22, 23, 24, 25, 26, 27):
    X = synth(seed, 2600)
    sL = simulate(L, X, "ZECUSDTM", eligible=OLD); sM = simulate(M, X, "XBTUSDTM", eligible=OLD); sC = simulate(C, mirror(X), "XBTUSDTM")
    for ref, nm in ((sM, "LONG-M1"), (sC, "C11-mirror")):
        cl, cc = sL["closed"], ref["closed"]
        if len(cl) != len(cc) or len(sL["missed"]) != len(ref["missed"]) or len(sL["open"]) != len(ref["open"]): ok = False; print("count mismatch", nm, seed, len(cl), len(cc))
        for x, y in zip(cl, cc):
            for f in ["signal_ts", "entry_ts", "exit_ts", "exit_reason", "bars_held"]:
                if x[f] != y[f]: ok = False; print("diff", nm, f)
            for f in ["gross_r", "risk", "atr"] + (["entry", "sl", "tp", "exit_price", "net_r", "net_r_s3"] if nm == "LONG-M1" else []):
                if abs(x[f] - y[f]) > 1e-9: ok = False; print("diff", nm, f, x[f], y[f])
    for a in sL["closed"]:
        ncmp += 1
        tp += a["exit_reason"] == "TP"; sl += a["exit_reason"] == "SL"; to += a["exit_reason"] == "TIMEOUT"
        cf = 0.0015 * (1 + a["exit_price"] / a["entry"]) + 0.0001 * a["bars_held"] * 0.5
        if abs(a["gross_r"] - cf / a["risk_pct"] - a["net_r_s3"]) > 1e-12: ok = False; print("S3 mismatch")
        if abs(a["net_r"] - (a["gross_r"] - 0.002 / a["risk_pct"])) > 1e-12: ok = False; print("S0 mismatch")
        if a["side"] != "LONG" or not (a["sl"] < a["entry"] < a["tp"]): ok = False; print("geometry")
        if abs((a["tp"] - a["entry"]) - 2 * (a["entry"] - a["sl"])) > 1e-9: ok = False; print("2R")
        if a["exit_reason"] == "SL" and abs(a["exit_price"] - a["sl"]) > 1e-12: ok = False
        if a["exit_reason"] == "TP" and abs(a["exit_price"] - a["tp"]) > 1e-12: ok = False
check(ok, f"incremental sim parity U67-L vs LONG-M1 (byte-level fields) and vs C11(mirror X): {ncmp} trades, TP={tp} SL={sl} TIMEOUT={to}; S0/S3 independent recomputation; LONG geometry")
check(ncmp >= 15 and tp > 0 and sl > 0, "parity test power")
check("net_r_s3" in sL["closed"][0] and "stop_gap" in sL["closed"][0], "new fields present on closed records")

# ---------- 3 monitor unit tests (LONG) ----------
def run(bars, entry=100.0, risk=2.0):
    st = L.new_state(); et = pd.Timestamp("2026-11-01T00:00:00Z")
    st["open"]["p"] = dict(id="p", symbol="X", side="LONG", signal_ts=(et - L.BAR).isoformat(), entry_ts=et.isoformat(), entry=entry, atr=1.6, risk=risk, risk_pct=risk/entry, sl=entry-risk, tp=entry+2*risk, bars=0, last_bar_ts=None, source="n", status="OPEN")
    for k, (o, h, l, c) in enumerate(bars):
        r = dict(ts=et + k*L.BAR, open=o, high=h, low=l, close=c, volume=1.0)
        if L.monitor_bar(st, "X", r): return st["closed"][0]
    return st
a = run([(100, 105, 97, 99)]); check(a["exit_reason"] == "SL" and a["gross_r"] == -1 and a["exit_price"] == 98 and a["stop_gap"] is False, "both hit same candle -> SL first")
b = run([(97, 98, 96, 97)]); check(b["exit_reason"] == "SL" and b["stop_gap"] is True, "open below stop -> SL + stop_gap flag")
c = run([(100, 105, 99, 104)]); check(c["exit_reason"] == "TP" and c["gross_r"] == 2 and c["exit_price"] == 104, "TP at 2R")
d = run([(100, 101, 99, 100.5)] * 30); check(d["exit_reason"] == "TIMEOUT" and d["bars_held"] == 30 and abs(d["gross_r"] - 0.25) < 1e-12, "30-bar timeout, entry candle = bar 1")
e = run([(100, 101, 99, 100.5)] * 29); check("exit_reason" not in e, "29 bars -> still open")
check(abs(d["net_r_s3"] - (0.25 - (0.0015 * (1 + 100.5/100) + 0.0001 * 30 * 0.5) / 0.02)) < 1e-12, "S3 formula on timeout (long)")
st = L.new_state(); et = pd.Timestamp("2026-11-01T00:00:00Z")
st["open"]["p"] = dict(id="p", symbol="X", side="LONG", signal_ts=(et - L.BAR).isoformat(), entry_ts=et.isoformat(), entry=100.0, atr=1.6, risk=2.0, risk_pct=0.02, sl=98.0, tp=104.0, bars=0, last_bar_ts=None, source="n", status="OPEN")
r = dict(ts=et, open=100, high=101, low=99, close=100, volume=1.0); L.monitor_bar(st, "X", r); L.monitor_bar(st, "X", r)
check(st["open"]["p"]["bars"] == 1, "monitor idempotent for same candle")
X = synth(21, 1800); d1 = L.prepare(X); idx = [i for i in range(300, 1790) if L.signal_reason(d1, i) == "SIGNAL"][0]
st = L.new_state(); st["eligible_from"] = OLD; st["symbols"]["ZECUSDTM"].update(baseline_done=True, last_ts=X["ts"].iloc[idx-1].isoformat())
L.process_symbol(st, "ZECUSDTM", X.iloc[:idx+2].copy(), X["ts"].iloc[idx+1] + pd.Timedelta(minutes=10))
pos = list(st["open"].values())[0]
check(all(k in pos for k in ["symbol", "entry", "risk", "sl", "signal_ts"]) and "last_ts" in st["symbols"]["ZECUSDTM"] and pos["side"] == "LONG", "state has fields used by the UNRESOLVED rule; side LONG")

# ---------- 4 candle timing ----------
ts0 = pd.Timestamp("2026-11-02T08:00:00Z")
df = pd.DataFrame(dict(ts=pd.date_range(end=ts0, periods=5, freq="4h"), open=1, high=1, low=1, close=1, volume=1))
for now, exp in [("2026-11-02T11:59:59Z", 4), ("2026-11-02T12:00:00Z", 5), ("2026-11-02T12:00:01Z", 5), ("2026-11-02T08:00:00Z", 4)]:
    check(len(L.completed_only(df, pd.Timestamp(now))) == exp == len(C.completed_only(df, pd.Timestamp(now))), f"completed_only at {now} -> {exp}, equals C11")
g = pd.DataFrame(dict(ts=pd.date_range(end=ts0, periods=300, freq="4h").delete(200)))
check(L.check_contiguous(g)[0] is False and L.check_contiguous(g) == C.check_contiguous(g), "gap in last 250 -> data-quality failure (same as C11)")

# ---------- 5 eligibility, first run, delay, duplicates ----------
X = synth(21, 1800)
base = simulate(L, X, "ZECUSDTM", eligible=OLD)
first = base["signals"][0]; et = pd.Timestamp(first["entry_ts"])
late = simulate(L, X, "ZECUSDTM", eligible=(et + pd.Timedelta(seconds=1)).isoformat())
check(first["id"] not in [s["id"] for s in late["signals"]] and any(m["reason"] == "BEFORE_REGISTRATION_START" and m["signal_ts"] == first["signal_ts"] for m in late["missed"]), "entry before eligible_from -> BEFORE_REGISTRATION_START, not opened")
exact = simulate(L, X, "ZECUSDTM", eligible=et.isoformat())
check(first["id"] in [s["id"] for s in exact["signals"]], "entry_ts == eligible_from is eligible")
ns = L.new_state()
check(pd.Timestamp(ns["eligible_from"]) >= L.REGISTRATION_ELIGIBLE_FROM and pd.Timestamp(ns["eligible_from"]) >= pd.Timestamp(ns["created_at"]), "new_state eligible_from >= max(constant, created_at)")
st = L.new_state(); st["eligible_from"] = OLD; raw = X.iloc[:1500].copy(); now = X["ts"].iloc[1499] + pd.Timedelta(minutes=10)
L.process_symbol(st, "ZECUSDTM", raw, now)
check(len(st["signals"]) + len(st["missed"]) <= 1, "first run does not replay history")
s1 = copy.deepcopy(st); L.process_symbol(st, "ZECUSDTM", raw, now); check(s1 == st, "idempotent re-run on same data")
d = L.prepare(X); idx = [i for i in range(300, 1790) if L.signal_reason(d, i) == "SIGNAL"][0]; raw = X.iloc[:idx+2].copy()
def run_delay(minutes):
    s = L.new_state(); s["eligible_from"] = OLD; s["symbols"]["ZECUSDTM"].update(baseline_done=True, last_ts=X["ts"].iloc[idx-1].isoformat())
    L.process_symbol(s, "ZECUSDTM", raw, X["ts"].iloc[idx+1] + pd.Timedelta(minutes=minutes)); return s
for mnt, exp in [(0, "open"), (89, "open"), (90, "open"), (90.5, "miss"), (91, "miss"), (600, "miss")]:
    s = run_delay(mnt)
    got = "open" if len(s["signals"]) == 1 else ("miss" if any(m["reason"] == "ENTRY_DELAY_EXCEEDED" for m in s["missed"]) else "none")
    check(got == exp, f"entry delay {mnt} min -> {exp}")
s = run_delay(10); n0 = len(s["signals"]); s["symbols"]["ZECUSDTM"]["last_ts"] = X["ts"].iloc[idx-1].isoformat()
L.process_symbol(s, "ZECUSDTM", raw, X["ts"].iloc[idx+1] + pd.Timedelta(minutes=10))
check(len(s["signals"]) == n0 and len(s["open"]) == 1, "duplicate signal not recorded twice (cursor reset)")
s = run_delay(10)
r = L.handle_signal(s, "ZECUSDTM", d, raw, idx, X["ts"].iloc[idx+1] + pd.Timedelta(minutes=10))
check(r == "DUPLICATE", "same signal -> DUPLICATE")
s = run_delay(10); s["signals"] = []
for _p in s["open"].values(): _p["signal_ts"] = (pd.Timestamp(_p["signal_ts"]) - L.BAR * 5).isoformat()
r = L.handle_signal(s, "ZECUSDTM", d, raw, idx, X["ts"].iloc[idx+1] + pd.Timedelta(minutes=10))
check(r == "MISSED" and any(m["reason"] == "SYMBOL_ALREADY_OPEN" for m in s["missed"]) and len(s["open"]) == 1, "second signal while open -> SYMBOL_ALREADY_OPEN, no second position")

# ---------- 6 API retries ----------
class R:
    def __init__(s, code, payload=None, text=""): s.status_code = code; s._p = payload; s.text = text
    def json(s): return s._p
good = R(200, {"code": "200000", "data": [[1700000000000, 1, 2, 0.5, 1.5, 10, 10]]})
def mk(seq):
    calls = []
    def get(url, params=None, timeout=None):
        calls.append(params); x = seq[min(len(calls)-1, len(seq)-1)]
        if isinstance(x, Exception): raise x
        return x
    return get, calls
sleeps = []; L.time.sleep = lambda s: sleeps.append(s)
for name, seq, exp_ok, exp_calls in [
 ("500,500,500 then 200", [R(500), R(500), R(500), good], True, 4), ("429 then 200", [R(429), good], True, 2),
 ("timeout exception then 200", [Exception("timeout"), good], True, 2), ("HTTP200 bad payload then 200", [R(200, {"code": "400100"}), good], True, 2),
 ("4 failures -> FETCH_FAILED", [R(500)] * 4, False, 4), ("404 repeated -> FETCH_FAILED", [R(404, text="nf")] * 4, False, 4)]:
    get, calls = mk(seq); L.SESSION.get = get; sleeps.clear()
    try:
        L.fetch_chunk("ZECUSDTM", 0, 1); okk = True
    except RuntimeError as e: okk = False; msg = str(e)
    check(okk == exp_ok and len(calls) == exp_calls and (okk or "FETCH_FAILED ZECUSDTM" in msg), f"retry: {name} (calls={len(calls)})")

# ---------- 7 main(): isolation, health threshold, 3x load ----------
def fake_df(sym, now, bars=589):
    seed = abs(hash(sym)) % 10000
    df = synth(seed, bars); end = pd.Timestamp(now).floor("4h")
    df["ts"] = pd.date_range(end=end, periods=bars, freq="4h"); return df
def run_main(symbols, fail=set(), state_name="u67_long_paper_state.json"):
    wd = tempfile.mkdtemp(prefix="u67lm_"); os.chdir(wd)
    sf = os.path.join(wd, state_name); L.STATE_FILE = sf
    L.SYMS = list(symbols)
    calls = {"n": 0}
    def fetch_raw(sym, now):
        calls["n"] += 1
        if sym in fail: raise RuntimeError("FETCH_FAILED " + sym + ": sim")
        return fake_df(sym, now)
    L.fetch_raw = fetch_raw
    buf = io.StringIO(); code = 0; t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf): L.main()
    except SystemExit as e: code = e.code
    return wd, sf, code, buf.getvalue(), calls["n"], time.time() - t0
S67 = list(fin["selected_symbols"])
wd, sf, code, out, n, dt = run_main(S67)
check(code == 0 and os.path.exists(sf) and sorted(os.listdir(wd)) == ["u67_long_paper_state.json"], "main(): clean run, only u67_long_paper_state.json created (state isolation)")
st = json.load(open(sf)); check(len(st["symbols"]) == 67 and all(v["last_status"] == "OK" for v in st["symbols"].values()) and "eligible_from" in st and st["strategy"] == "u67_long_only_frozen_v1", "state structure 67/67 OK, strategy name")
check("NET_EXP" not in out and "NET_PF" not in out and "NET_TOTAL" not in out, "report prints no R figures")
check("SYMBOLS_OK=67/67 FAILED=0" in out and "RUN_STATUS = OK" in out, "health line 67/67")
check(not os.path.exists(os.path.join(REPO, "u67_long_paper_state.json")), "repo checkout untouched by test")
for nf, exp in [(13, 0), (14, 2)]:
    wd, sf, code, out, n, dt = run_main(S67, fail=set(S67[:nf]))
    check(code == exp, f"{nf}/67 symbols failing -> exit {exp} ({'OK' if exp == 0 else 'DEGRADED'})")
    if exp == 2: check(os.path.exists(sf) and "RUN_STATUS = DEGRADED" in out, "degraded run saves state and reports DEGRADED")
S201 = [f"T{i:03d}USDTM" for i in range(201)]
L.PAUSE = 0.0
wd, sf, code, out, n, dt = run_main(S201)
check(code == 0 and n == 201 and "SYMBOLS_OK=201/201" in out, f"3x load: 201 symbols processed in one run (fetch_raw calls={n}, compute {dt:.1f}s)")
st = json.load(open(sf)); check(st["errors"] == [], "3x load: no errors")
before = open(sf).read(); L.STATE_FILE = sf; L.SYMS = S201
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try: L.main()
    except SystemExit: pass
check(open(sf).read() == before, "second run on same data leaves state byte-identical (no commit churn)")

# ---------- 8 workflow ----------
w = yaml.safe_load(open(WF)); steps = w["jobs"]["paper"]["steps"]; wtxt = open(WF).read()
check(w["concurrency"]["group"] == "u67-long-paper" and w["name"] == "U67-L Paper Trading", "workflow name/group distinct")
check(w["concurrency"]["cancel-in-progress"] is False, "cancel-in-progress false")
check(any("crypto_signal_bot_u67_long_v1.py" in (s.get("run") or "") for s in steps) and not re.search(r"candidate11|long_m1|long-m1|u67_short|u67-short|crypto_signal_bot_long_v1|crypto_signal_bot_v4", wtxt, re.I), "workflow references only U67-L files")
check(steps[0].get("with", {}).get("fetch-depth") == 0, "checkout full history")
guard = [s for s in steps if s.get("name") == "Guard state continuity"][0]["run"]
def guard_run(setup):
    dd = tempfile.mkdtemp(prefix="g_")
    cmds = "cd " + dd + " && git init -q . && git config user.email a@b && git config user.name n && " + setup
    return subprocess.run(["bash", "-c", "set -e; " + cmds + "\nset +e\n" + guard], capture_output=True, text=True).returncode
rc1 = guard_run("true")
rc2 = guard_run("echo '{}' > u67_long_paper_state.json && git add . && git commit -qm x && git rm -q u67_long_paper_state.json && git commit -qm del")
rc3 = guard_run("echo '{}' > u67_long_paper_state.json && git add . && git commit -qm x")
check(rc1 == 0 and rc2 != 0 and rc3 == 0, f"guard: fresh repo passes ({rc1}), committed-then-missing fails ({rc2}), present passes ({rc3})")
# workflow == U67-S workflow with only registered name changes
sw = open(BASE + "../u67/engine/paper_u67_short_v3.yml").read()
tr = sw.replace("U67-S Paper Trading", "U67-L Paper Trading").replace("u67-short-paper", "u67-long-paper").replace("u67_short_paper_state.json", "u67_long_paper_state.json").replace("crypto_signal_bot_u67_short_v1.py", "crypto_signal_bot_u67_long_v1.py").replace("Update U67-S paper state", "Update U67-L paper state").replace("Run U67-S", "Run U67-L")
check(tr == wtxt, "workflow identical to the U67-S workflow after the registered name substitutions")
check("RANDOM % 8" in wtxt and "for i in 1 2 3 4 5 6" in wtxt and "PUSH_FAILED_AFTER_6_ATTEMPTS" in wtxt and "git pull --rebase origin main && git push" in wtxt, "jittered 6-attempt rebase/push loop present")
check("schedule" not in w[True] and "workflow_dispatch" in w[True], "workflow_dispatch only, no schedule trigger (Stage 4)")
# schedule/dispatch triggers documented
check(w[True]["workflow_dispatch"] is None or "workflow_dispatch" in w[True], "workflow_dispatch trigger present")
print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
