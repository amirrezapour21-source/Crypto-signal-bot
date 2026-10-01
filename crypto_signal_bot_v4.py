import time, requests, pandas as pd, numpy as np

URL="https://api.kucoin.com/api/v1/market/candles"
SYMS="""BTC ETH SOL BNB XRP DOGE ADA LINK AVAX DOT SUI TRX NEAR AAVE OP ARB APT ATOM FIL LTC BCH ETC UNI INJ SEI VET HBAR ALGO XLM ICP WIF PEPE FLOKI""".split()
SYMS=[x+"-USDT" for x in SYMS]

END=pd.Timestamp("2026-09-30 00:00",tz="UTC")
START=pd.Timestamp("2024-09-30 04:00",tz="UTC")
STEP=pd.Timedelta(hours=4)
N=4380
COST=.003
SL_ATR=1.25
HOLD=30
TPS=[1,1.5,2,3]

def fetch(s):
    rows=[]; cur=START
    while cur<=END:
        e=min(END,cur+STEP*1399)
        for z in range(4):
            try:
                r=requests.get(URL,params={
                    "symbol":s,"type":"4hour",
                    "startAt":int(cur.timestamp()),
                    "endAt":int(e.timestamp())},timeout=20).json()
                if r["code"]!="200000": raise Exception(r.get("msg"))
                a=r["data"]
                rows+=a
                if not a: cur=e+STEP
                else: cur=pd.Timestamp(max(int(x[0]) for x in a),unit="s",tz="UTC")+STEP
                break
            except Exception as ex:
                if z==3: raise Exception(f"{s}: {ex}")
                time.sleep(1)
        time.sleep(.08)

    d=pd.DataFrame(rows,columns=["ts","open","close","high","low","vol","turn"])
    d.ts=pd.to_datetime(d.ts.astype(int),unit="s",utc=True)
    for c in ["open","close","high","low","vol"]: d[c]=pd.to_numeric(d[c])
    return d.drop_duplicates("ts").sort_values("ts").set_index("ts")

def prep(d):
    c,h,l,v=d.close,d.high,d.low,d.vol
    m=c.rolling(20).mean()
    sd=c.rolling(20).std()
    d["z"]=(c-m)/sd.replace(0,np.nan)
    d["ema"]=c.ewm(span=200,adjust=False,min_periods=200).mean()
    d["rng"]=h.rolling(20).max()-l.rolling(20).min()
    d["rm"]=d.rng.rolling(20).median()
    d["vm"]=v.rolling(20).median()
    pc=c.shift()
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/20,adjust=False,min_periods=20).mean()
    return d

def events(d,s):
    e=[]
    for i in range(1,len(d)):
        r,p=d.iloc[i],d.iloc[i-1]
        if pd.isna(r[["z","ema","rng","rm","vm","atr"]]).any() or pd.isna(p.z): continue

        if r.close>r.ema and p.z<=-2 and r.z>-2 and r.rng>=r.rm and r.vol>=r.vm:
            e.append((s,d.index[i],i,"LONG",r.close,r.atr))

        if r.close<r.ema and p.z>=2 and r.z<2 and r.rng>=r.rm and r.vol>=r.vm:
            e.append((s,d.index[i],i,"SHORT",r.close,r.atr))
    return e

def trade(e,d,tp):
    s,t,i,side,entry,atr=e
    risk=atr*SL_ATR
    if side=="LONG": sl,tpv=entry-risk,entry+risk*tp
    else: sl,tpv=entry+risk,entry-risk*tp

    for j in range(i+1,min(i+HOLD,len(d)-1)+1):
        h,l=d.high.iloc[j],d.low.iloc[j]

        if side=="LONG":
            if l<=sl: return -1-COST
            if h>=tpv: return tp-COST
        else:
            if h>=sl: return -1-COST
            if l<=tpv: return tp-COST

    return -COST

def report(label,x):
    if not x:
        print(label,"N=0"); return
    x=np.array(x)
    win=x[x>0].sum()
    loss=-x[x<0].sum()
    pf=win/loss if loss else float("inf")
    print(f"{label}: N={len(x)} WR={(x>0).mean():.4f} "
          f"Exp={x.mean():.4f} Total={x.sum():.2f}R PF={pf:.3f}")

def concentration(results):
    if not results: return
    q=pd.DataFrame(results,columns=["s","r"]).groupby("s").r.sum().sort_values(ascending=False)
    total=q.sum()
    print(f"Portfolio={total:.2f}R | -Top1={total-q.head(1).sum():.2f}R "
          f"| -Top3={total-q.head(3).sum():.2f}R | -Top5={total-q.head(5).sum():.2f}R")
    print("Top:",", ".join(f"{k}={v:.2f}R" for k,v in q.head(5).items()))

print("="*58)
print("CANDIDATE 11 — DIRECTION ROBUSTNESS V2")
print("="*58)

raw={}
for s in SYMS:
    try:
        raw[s]=fetch(s)
        print(s,len(raw[s]))
    except Exception as e:
        print("FAIL",e)

# -------- exact common window --------
common=set(raw[SYMS[0]].index)
for s in SYMS[1:]:
    common &= set(raw[s].index)

common=sorted(common)

if len(common)<N:
    raise SystemExit(f"ABORTED: common timestamps={len(common)} < {N}")

common=pd.DatetimeIndex(common[-N:])

data={}
for s in raw:
    d=raw[s].loc[common].copy()
    if len(d)!=N:
        raise SystemExit(f"ABORTED: {s}={len(d)}")
    data[s]=prep(d)

print("\nVALID SYMBOLS =",len(data))
print("COMMON TIMESTAMPS =",len(common))
print("WINDOW =",common[0],"->",common[-1])
print("DATA AUDIT = PASS")

# -------- events --------
E=[]
for s,d in data.items():
    E+=events(d,s)

print("\nEVENT AUDIT")
print("TOTAL =",len(E))
print("LONG  =",sum(x[3]=="LONG" for x in E))
print("SHORT =",sum(x[3]=="SHORT" for x in E))

# -------- direction test --------
for tp in TPS:
    print(f"\n{'='*58}\nTP{tp} DIRECTION\n{'='*58}")

    L=[]; S=[]

    for e in E:
        r=trade(e,data[e[0]],tp)
        (L if e[3]=="LONG" else S).append((e[0],r))

    report("LONG ",[r for _,r in L])
    report("SHORT",[r for _,r in S])
    report("BOTH ",[r for _,r in L+S])

    print("\nLONG concentration")
    concentration(L)

    print("SHORT concentration")
    concentration(S)

print("\n"+"="*58)
print("DIRECTION ROBUSTNESS V2 COMPLETED")
print("="*58)
