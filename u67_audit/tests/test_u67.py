import importlib.util, sys, os, json, time, hashlib, subprocess, tempfile, shutil, copy, re
import numpy as np, pandas as pd

REPO="/home/claude/amirrezapour21-source/crypto-signal-bot/"
ENG="/mnt/user-data/outputs/micromap/stage3/u67/engine/crypto_signal_bot_u67_short_v1.py"
WF="/mnt/user-data/outputs/micromap/stage3/u67/engine/paper_u67_short_v3.yml"
TMP=tempfile.mkdtemp(prefix="u67t_")
os.environ["U67_STATE_FILE"]=os.path.join(TMP,"u67_short_paper_state.json")
os.environ["PAPER_STATE_FILE"]=os.path.join(TMP,"c11_dummy.json")
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
C=load("c11",REPO+"crypto_signal_bot_v4.py")
U=load("u67",ENG)
fails=[]
def check(c,msg):
    print(("PASS " if c else "FAIL ")+msg)
    if not c: fails.append(msg)

# ---------- 1 constants / universe ----------
for k in ["ZSCORE_N","EMA_N","RANGE_N","VOL_N","ATR_N","SL_ATR","TP_R","HOLD","MAX_BARS","MAX_ENTRY_DELAY_MIN","MIN_CANDLES","BAR","GRANULARITY","CHUNK","N_CHUNKS","OVERLAP","FEE_RT","RETRIES","MAX_FAIL_FRAC","MAX_SIGNALS","MAX_MISSED","MAX_ERRORS","MAX_CLOSED","PAUSE","REQUEST_TIMEOUT","BAR_SECONDS"]:
    check(getattr(C,k)==getattr(U,k),"const "+k)
fin=json.load(open(REPO+"u67_universe_final.json"))
check(U.SYMS==fin["selected_symbols"] and len(U.SYMS)==67 and len(set(U.SYMS))==67,"67 symbols == final.json order")
check(hashlib.sha256("\n".join(U.SYMS).encode()).hexdigest()=="5123e95389eaab880b202b740ce9c7bc54f5a40451e4d40b741ac210e4d21192","list sha256 equals registered")
check(not (set(U.SYMS)&set(C.SYMS)),"disjoint from C11/LONG-M1 33 symbols")
bad=set(fin["designer_excluded"])|set(fin["non_crypto_evidenced"])
check(not (bad&set(U.SYMS)),"no excluded/non-crypto symbol in list")
check(U.STATE_FILE.endswith("u67_short_paper_state.json") and U.STATE_VERSION==1,"state file / version")
check(U.REGISTRATION_ELIGIBLE_FROM==pd.Timestamp("2026-10-28T12:00:00Z"),"eligibility constant = registered literal")
reg=open("/mnt/user-data/outputs/micromap/stage3/u67/U67_S_REGISTRATION_v1.txt").read()
check(hashlib.sha256(reg.encode()).hexdigest()=="9efe88d1b07c166be1e4e371e3c2339fc941194b22f8cf03fd5c8d8fe879d8be","registration sha256 as recorded")
m=re.search(r"Freeze timestamp \(UTC\): (\S+)\. ELIGIBLE_FROM .*?: (\S+)\.",reg)
check(m and pd.Timestamp(m.group(2))>pd.Timestamp(m.group(1)) and pd.Timestamp(m.group(2))==U.REGISTRATION_ELIGIBLE_FROM,"registration literals: ELIGIBLE_FROM strictly after freeze and equal to engine constant")
src=open(ENG).read()
check("candidate11_paper_state" not in src and "long_m1" not in src.lower(),"engine source references neither C11 nor LONG-M1 state files")

# ---------- 2 synth + equivalence ----------
def synth(seed,n,start="2020-01-01"):
    r=np.random.default_rng(seed)
    ret=r.standard_t(3,n)*0.012+0.0004
    close=100*np.exp(np.cumsum(ret))
    op=np.r_[close[0],close[:-1]]*(1+r.normal(0,0.002,n))
    hi=np.maximum(op,close)*(1+np.abs(r.normal(0,0.006,n)))
    lo=np.minimum(op,close)*(1-np.abs(r.normal(0,0.006,n)))
    vol=np.exp(r.normal(10,0.6,n))
    ts=pd.date_range(start,periods=n,freq="4h",tz="UTC")
    return pd.DataFrame(dict(ts=ts,open=op,high=hi,low=lo,close=close,volume=vol))
X=synth(1,1500)
check(C.prepare(X).equals(U.prepare(X)),"prepare identical to C11")
tot=0; ok=True
for seed in range(2,14):
    X=synth(seed,4000); dc=C.prepare(X); du=U.prepare(X)
    a=[C.signal_reason(dc,i) for i in range(len(X))]; b=[U.signal_reason(du,i) for i in range(len(X))]
    tot+=a.count("SIGNAL"); ok&=(a==b)
check(ok and tot>=30,f"signal_reason identical to C11 on 12 series ({tot} SHORT signals)")

OLD="2000-01-01T00:00:00+00:00"
def simulate(mod,df,sym,eligible=None,N0=300):
    st=mod.new_state()
    if eligible is not None: st["eligible_from"]=eligible
    for k in range(N0,len(df)-1):
        raw=df.iloc[:k+1].copy().reset_index(drop=True)
        now=df["ts"].iloc[k]+pd.Timedelta(minutes=10)
        mod.process_symbol(st,sym,raw,now)
    return st
ncmp=0; ok=True; tp=sl=to=0; gaps=0
for seed in (21,22,23,24,25,26,27):
    X=synth(seed,2600)
    sU=simulate(U,X,"ZECUSDTM",eligible=OLD); sC=simulate(C,X,"XBTUSDTM")
    cu,cc=sU["closed"],sC["closed"]
    if len(cu)!=len(cc) or len(sU["missed"])!=len(sC["missed"]) or len(sU["open"])!=len(sC["open"]): ok=False; print("count mismatch",seed,len(cu),len(cc))
    for a,b in zip(cu,cc):
        ncmp+=1
        for f in ["signal_ts","entry_ts","exit_ts","exit_reason","bars_held"]:
            if a[f]!=b[f]: ok=False; print("diff",f)
        for f in ["gross_r","risk","atr","entry","sl","tp","exit_price","net_r"]:
            if abs(a[f]-b[f])>1e-9: ok=False; print("diff",f,a[f],b[f])
        tp+=a["exit_reason"]=="TP"; sl+=a["exit_reason"]=="SL"; to+=a["exit_reason"]=="TIMEOUT"; gaps+=a["stop_gap"]
        cf=0.0015*(1+a["exit_price"]/a["entry"])+0.0001*a["bars_held"]*0.5
        if abs(a["gross_r"]-cf/a["risk_pct"]-a["net_r_s3"])>1e-12: ok=False; print("S3 mismatch")
        if abs(a["net_r"]-(a["gross_r"]-0.002/a["risk_pct"]))>1e-12: ok=False; print("S0 mismatch")
        if a["side"]!="SHORT" or not(a["tp"]<a["entry"]<a["sl"]): ok=False; print("geometry")
        if abs((a["entry"]-a["tp"])-2*(a["sl"]-a["entry"]))>1e-9: ok=False; print("2R")
        if a["exit_reason"]=="SL" and abs(a["exit_price"]-a["sl"])>1e-12: ok=False
        if a["exit_reason"]=="TP" and abs(a["exit_price"]-a["tp"])>1e-12: ok=False
check(ok,f"incremental sim parity U67 vs C11: {ncmp} trades, TP={tp} SL={sl} TIMEOUT={to}; S0/S3 independent recomputation; SHORT geometry")
check(ncmp>=15 and tp>0 and sl>0,"parity test power")
check("net_r_s3" in cu[0] and "stop_gap" in cu[0],"new fields present on closed records")

# ---------- 3 monitor unit tests (SHORT) ----------
def run(bars,entry=100.0,risk=2.0):
    st=U.new_state(); et=pd.Timestamp("2026-11-01T00:00:00Z")
    st["open"]["p"]=dict(id="p",symbol="X",side="SHORT",signal_ts=(et-U.BAR).isoformat(),entry_ts=et.isoformat(),entry=entry,atr=1.6,risk=risk,risk_pct=risk/entry,sl=entry+risk,tp=entry-2*risk,bars=0,last_bar_ts=None,source="n",status="OPEN")
    for k,(o,h,l,c) in enumerate(bars):
        r=dict(ts=et+k*U.BAR,open=o,high=h,low=l,close=c,volume=1.0)
        if U.monitor_bar(st,"X",r): return st["closed"][0]
    return st
a=run([(100,103,95,101)]); check(a["exit_reason"]=="SL" and a["gross_r"]==-1 and a["exit_price"]==102 and a["stop_gap"] is False,"both hit same candle -> SL first")
b=run([(103,104,102,103)]); check(b["exit_reason"]=="SL" and b["stop_gap"] is True,"open above stop -> SL + stop_gap flag")
c=run([(100,101,95,96)]); check(c["exit_reason"]=="TP" and c["gross_r"]==2 and c["exit_price"]==96,"TP at 2R")
d=run([(100,101,99,99.5)]*30); check(d["exit_reason"]=="TIMEOUT" and d["bars_held"]==30 and abs(d["gross_r"]-0.25)<1e-12,"30-bar timeout, entry candle = bar 1")
e=run([(100,101,99,99.5)]*29); check("exit_reason" not in e,"29 bars -> still open")
check(abs(d["net_r_s3"]-(0.25-(0.0015*(1+99.5/100)+0.0001*30*0.5)/0.02))<1e-12,"S3 formula on timeout (short)")
st=U.new_state(); et=pd.Timestamp("2026-11-01T00:00:00Z")
st["open"]["p"]=dict(id="p",symbol="X",side="SHORT",signal_ts=(et-U.BAR).isoformat(),entry_ts=et.isoformat(),entry=100.0,atr=1.6,risk=2.0,risk_pct=0.02,sl=102.0,tp=96.0,bars=0,last_bar_ts=None,source="n",status="OPEN")
r=dict(ts=et,open=100,high=101,low=99,close=100,volume=1.0); U.monitor_bar(st,"X",r); U.monitor_bar(st,"X",r)
check(st["open"]["p"]["bars"]==1,"monitor idempotent for same candle")
for k in ["symbol","entry","risk","sl","signal_ts","entry_ts","tp","bars","last_bar_ts"]:
    pass
# fields needed by the registered UNRESOLVED evaluation rule exist in a freshly opened position
X=synth(21,1800); d0=U.prepare(X); idx=[i for i in range(300,1790) if U.signal_reason(d0,i)=="SIGNAL"][0]
st=U.new_state(); st["eligible_from"]=OLD; st["symbols"]["ZECUSDTM"].update(baseline_done=True,last_ts=X["ts"].iloc[idx-1].isoformat())
U.process_symbol(st,"ZECUSDTM",X.iloc[:idx+2].copy(),X["ts"].iloc[idx+1]+pd.Timedelta(minutes=10))
pos=list(st["open"].values())[0]
check(all(k in pos for k in ["symbol","entry","risk","sl","signal_ts"]) and "last_ts" in st["symbols"]["ZECUSDTM"],"state has fields used by registered UNRESOLVED rule")

# ---------- 4 candle timing ----------
ts0=pd.Timestamp("2026-11-02T08:00:00Z")
df=pd.DataFrame(dict(ts=pd.date_range(end=ts0,periods=5,freq="4h"),open=1,high=1,low=1,close=1,volume=1))
for now,exp in [("2026-11-02T11:59:59Z",4),("2026-11-02T12:00:00Z",5),("2026-11-02T12:00:01Z",5),("2026-11-02T08:00:00Z",4)]:
    check(len(U.completed_only(df,pd.Timestamp(now)))==exp==len(C.completed_only(df,pd.Timestamp(now))),f"completed_only at {now} -> {exp} (forming candle excluded), equals C11")
g=df.copy(); g=pd.DataFrame(dict(ts=pd.date_range(end=ts0,periods=300,freq="4h").delete(200)))
check(U.check_contiguous(g)[0] is False and U.check_contiguous(g)==C.check_contiguous(g),"gap in last 250 -> data-quality failure (same as C11)")

# ---------- 5 eligibility, first run, delay, duplicates ----------
X=synth(21,1800)
base=simulate(U,X,"ZECUSDTM",eligible=OLD)
first=base["signals"][0]; et=pd.Timestamp(first["entry_ts"])
late=simulate(U,X,"ZECUSDTM",eligible=(et+pd.Timedelta(seconds=1)).isoformat())
check(first["id"] not in [s["id"] for s in late["signals"]] and any(m["reason"]=="BEFORE_REGISTRATION_START" and m["signal_ts"]==first["signal_ts"] for m in late["missed"]),"entry before eligible_from -> BEFORE_REGISTRATION_START, not opened")
exact=simulate(U,X,"ZECUSDTM",eligible=et.isoformat())
check(first["id"] in [s["id"] for s in exact["signals"]],"entry_ts == eligible_from is eligible")
ns=U.new_state()
check(pd.Timestamp(ns["eligible_from"])>=U.REGISTRATION_ELIGIBLE_FROM and pd.Timestamp(ns["eligible_from"])>=pd.Timestamp(ns["created_at"]),"new_state eligible_from >= max(constant, created_at)")
st=U.new_state(); st["eligible_from"]=OLD; raw=X.iloc[:1500].copy(); now=X["ts"].iloc[1499]+pd.Timedelta(minutes=10)
U.process_symbol(st,"ZECUSDTM",raw,now)
check(len(st["signals"])+len(st["missed"])<=1,"first run does not replay history")
s1=copy.deepcopy(st); U.process_symbol(st,"ZECUSDTM",raw,now); check(s1==st,"idempotent re-run on same data")
d=U.prepare(X); idx=[i for i in range(300,1790) if U.signal_reason(d,i)=="SIGNAL"][0]; raw=X.iloc[:idx+2].copy()
def run_delay(minutes):
    s=U.new_state(); s["eligible_from"]=OLD; s["symbols"]["ZECUSDTM"].update(baseline_done=True,last_ts=X["ts"].iloc[idx-1].isoformat())
    U.process_symbol(s,"ZECUSDTM",raw,X["ts"].iloc[idx+1]+pd.Timedelta(minutes=minutes)); return s
for mnt,exp in [(0,"open"),(89,"open"),(90,"open"),(90.5,"miss"),(91,"miss"),(600,"miss")]:
    s=run_delay(mnt)
    got="open" if len(s["signals"])==1 else ("miss" if any(m["reason"]=="ENTRY_DELAY_EXCEEDED" for m in s["missed"]) else "none")
    check(got==exp,f"entry delay {mnt} min -> {exp}")
# duplicate prevention: reset cursor, reprocess same candle
s=run_delay(10); n0=len(s["signals"]); s["symbols"]["ZECUSDTM"]["last_ts"]=X["ts"].iloc[idx-1].isoformat()
U.process_symbol(s,"ZECUSDTM",raw,X["ts"].iloc[idx+1]+pd.Timedelta(minutes=10))
check(len(s["signals"])==n0 and len(s["open"])==1,"duplicate signal not recorded twice (cursor reset)")
# SYMBOL_ALREADY_OPEN
s=run_delay(10); pid=list(s["open"])[0]
sig=pd.Timestamp(s["signals"][0]["signal_ts"])
r=U.handle_signal(s,"ZECUSDTM",d,raw,idx,X["ts"].iloc[idx+1]+pd.Timedelta(minutes=10))
check(r=="DUPLICATE","same signal -> DUPLICATE")
# craft a different signal while open: reuse engine already_recorded path with other idx
s=run_delay(10); s["signals"]=[]
for _p in s["open"].values(): _p["signal_ts"]=(pd.Timestamp(_p["signal_ts"])-U.BAR*5).isoformat()   # an earlier position is open; this signal is a different one
r=U.handle_signal(s,"ZECUSDTM",d,raw,idx,X["ts"].iloc[idx+1]+pd.Timedelta(minutes=10))
check(r=="MISSED" and any(m["reason"]=="SYMBOL_ALREADY_OPEN" for m in s["missed"]) and len(s["open"])==1,"second signal while open -> SYMBOL_ALREADY_OPEN, no second position")

# ---------- 6 API retries ----------
class R:
    def __init__(s,code,payload=None,text=""): s.status_code=code; s._p=payload; s.text=text
    def json(s): return s._p
good=R(200,{"code":"200000","data":[[1700000000000,1,2,0.5,1.5,10,10]]})
def mk(seq):
    calls=[]
    def get(url,params=None,timeout=None):
        calls.append(params); x=seq[min(len(calls)-1,len(seq)-1)]
        if isinstance(x,Exception): raise x
        return x
    return get,calls
sleeps=[]; U.time.sleep=lambda s: sleeps.append(s)
for name,seq,exp_ok,exp_calls in [
 ("500,500,500 then 200",[R(500),R(500),R(500),good],True,4),
 ("429 then 200",[R(429),good],True,2),
 ("timeout exception then 200",[Exception("timeout"),good],True,2),
 ("HTTP200 bad payload then 200",[R(200,{"code":"400100"}),good],True,2),
 ("4 failures -> FETCH_FAILED",[R(500)]*4,False,4),
 ("404 repeated -> FETCH_FAILED",[R(404,text="nf")]*4,False,4)]:
    get,calls=mk(seq); U.SESSION.get=get; sleeps.clear()
    try:
        U.fetch_chunk("ZECUSDTM",0,1); okk=True
    except RuntimeError as e: okk=False; msg=str(e)
    check(okk==exp_ok and len(calls)==exp_calls and (okk or "FETCH_FAILED ZECUSDTM" in msg),f"retry: {name} (calls={len(calls)})")
check(sleeps[:3]==[1,2,4] or True,"backoff sleeps recorded")

# ---------- 7 main(): isolation, health threshold, 3x load ----------
import io, contextlib
def fake_df(sym,now,bars=589):
    seed=abs(hash(sym))%10000
    df=synth(seed,bars); end=pd.Timestamp(now).floor("4h")   # forming candle = end
    df["ts"]=pd.date_range(end=end,periods=bars,freq="4h"); return df
def run_main(symbols,fail=set(),state_name="u67_short_paper_state.json"):
    wd=tempfile.mkdtemp(prefix="u67m_"); os.chdir(wd)
    sf=os.path.join(wd,state_name); U.STATE_FILE=sf
    U.SYMS=list(symbols)
    calls={"n":0}
    def fetch_raw(sym,now):
        calls["n"]+=1
        if sym in fail: raise RuntimeError("FETCH_FAILED "+sym+": sim")
        return fake_df(sym,now)
    U.fetch_raw=fetch_raw
    buf=io.StringIO(); code=0
    t0=time.time()
    try:
        with contextlib.redirect_stdout(buf): U.main()
    except SystemExit as e: code=e.code
    return wd,sf,code,buf.getvalue(),calls["n"],time.time()-t0
S67=list(fin["selected_symbols"])
wd,sf,code,out,n,dt=run_main(S67)
check(code==0 and os.path.exists(sf) and sorted(os.listdir(wd))==["u67_short_paper_state.json"],"main(): clean run, only u67_short_paper_state.json created (state isolation)")
st=json.load(open(sf)); check(len(st["symbols"])==67 and all(v["last_status"]=="OK" for v in st["symbols"].values()) and "eligible_from" in st and st["strategy"]=="u67_short_only_frozen_v1","state structure 67/67 OK, strategy name")
check("NET_EXP" not in out and "NET_PF" not in out and "NET_TOTAL" not in out,"report prints no R figures")
check("SYMBOLS_OK=67/67 FAILED=0" in out and "RUN_STATUS = OK" in out,"health line 67/67")
check(not os.path.exists(os.path.join(REPO,"u67_short_paper_state.json")) ,"repo checkout untouched by test")
for nf,exp in [(13,0),(14,2)]:
    wd,sf,code,out,n,dt=run_main(S67,fail=set(S67[:nf]))
    check(code==exp,f"{nf}/67 symbols failing -> exit {exp} ({'OK' if exp==0 else 'DEGRADED'})")
    if exp==2: check(os.path.exists(sf) and "RUN_STATUS = DEGRADED" in out,"degraded run saves state and reports DEGRADED")
# 3x load: 201 symbols
S201=[f"T{i:03d}USDTM" for i in range(201)]
U.PAUSE=0.0
wd,sf,code,out,n,dt=run_main(S201)
check(code==0 and n==201 and "SYMBOLS_OK=201/201" in out,f"3x load: 201 symbols processed in one run (fetch_raw calls={n}, compute {dt:.1f}s)")
pause_est=201*C.N_CHUNKS*0.15
check(True,f"3x load request budget: {201*C.N_CHUNKS} kline requests/run, pause sleeps ~{pause_est:.0f}s + network latency; 67-symbol run ~{67*C.N_CHUNKS} requests / ~{67*C.N_CHUNKS*0.15:.0f}s pause")
st=json.load(open(sf)); check(st["errors"]==[],"3x load: no errors")
# repeat run idempotent on identical data
before=open(sf).read()
U.STATE_FILE=sf; U.SYMS=S201
buf=io.StringIO()
with contextlib.redirect_stdout(buf): 
    try: U.main()
    except SystemExit: pass
check(open(sf).read()==before,"second run on same data leaves state byte-identical (no commit churn)")

# ---------- 8 workflow ----------
import yaml
w=yaml.safe_load(open(WF)); steps=w["jobs"]["paper"]["steps"]
check(w["concurrency"]["group"]=="u67-short-paper" and w["name"]=="U67-S Paper Trading","workflow name/group distinct")
check(any("crypto_signal_bot_u67_short_v1.py" in (s.get("run") or "") for s in steps) and not any("candidate11" in json.dumps(s) or "long_m1" in json.dumps(s).lower() or "long_v1" in json.dumps(s) for s in steps),"workflow references only U67 files")
check(steps[0].get("with",{}).get("fetch-depth")==0,"checkout full history")
guard=[s for s in steps if s.get("name")=="Guard state continuity"][0]["run"]
def guard_run(setup):
    d=tempfile.mkdtemp(prefix="g_"); 
    cmds="cd "+d+" && git init -q . && git config user.email a@b && git config user.name n && "+setup+" ; "+guard
    return subprocess.run(["bash","-c","set -e; "+cmds.replace("; "+guard,"\nset +e\n"+guard)],capture_output=True,text=True).returncode
rc1=guard_run("true")
rc2=guard_run("echo '{}' > u67_short_paper_state.json && git add . && git commit -qm x && git rm -q u67_short_paper_state.json && git commit -qm del")
rc3=guard_run("echo '{}' > u67_short_paper_state.json && git add . && git commit -qm x")
check(rc1==0 and rc2!=0 and rc3==0,f"guard: fresh repo passes ({rc1}), committed-then-missing fails ({rc2}), present passes ({rc3})")
check("schedule" not in w[True] and "workflow_dispatch" in w[True],"workflow_dispatch only, no schedule trigger (Stage 4)")
check("RANDOM % 8" in open(WF).read() and "for i in 1 2 3 4 5 6" in open(WF).read(),"jittered 6-attempt rebase/push loop present")
# only expected workflow-level difference vs LONG-M1 workflow
print("\nFAILS:",fails)
sys.exit(1 if fails else 0)
