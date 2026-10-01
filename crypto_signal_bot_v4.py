import time, requests, numpy as np, pandas as pd

BASE="https://api.kucoin.com/api/v1/market/candles"
SYMBOLS=[
"BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT",
"ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT","SUI-USDT","TRX-USDT",
"NEAR-USDT","AAVE-USDT","OP-USDT","ARB-USDT","APT-USDT","ATOM-USDT",
"FIL-USDT","LTC-USDT","BCH-USDT","ETC-USDT","UNI-USDT","INJ-USDT",
"SEI-USDT","VET-USDT","HBAR-USDT","ALGO-USDT","XLM-USDT","ICP-USDT",
"WIF-USDT","PEPE-USDT","FLOKI-USDT"
]

STEP=14400
ROWS=1450
HOLD=30
COST=.003
SL_ATR=1.25
TPs=[1,1.5,2,3]

# Fixed comparison window: UTC
END=pd.Timestamp("2026-09-30 00:00:00",tz="UTC").timestamp()
START=END-730*86400

def fetch(sym):
    out=[]; cur=int(START)
    while cur<int(END):
        e=min(cur+ROWS*STEP,int(END))
        p={"symbol":sym,"type":"4hour","startAt":cur,"endAt":e}
        j=requests.get(BASE,params=p,timeout=20).json()
        if j.get("code")!="200000":
            return pd.DataFrame()
        x=j.get("data",[])
        if not x: break
        out += x
        mx=max(int(r[0]) for r in x)
        cur=mx+1
        time.sleep(.08)

    if not out: return pd.DataFrame()

    d=pd.DataFrame(out,columns=[
        "time","open","close","high","low","volume","turnover"
    ])
    d["time"]=pd.to_numeric(d["time"]).astype("int64")
    d=d[(d.time>=START)&(d.time<=END)]
    d=d.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    gaps=int((d.time.diff().dropna()!=STEP).sum())
    if gaps!=0: return pd.DataFrame()

    d["open"]=d.open.astype(float)
    d["close"]=d.close.astype(float)
    d["high"]=d.high.astype(float)
    d["low"]=d.low.astype(float)
    d["volume"]=d.volume.astype(float)

    c=d.close
    m=c.rolling(20).mean()
    s=c.rolling(20).std()

    d["z"]=(c-m)/s.replace(0,np.nan)
    d["ema"]=c.ewm(span=200,adjust=False).mean()
    d["rng"]=d.high.rolling(20).max()-d.low.rolling(20).min()
    d["rngmed"]=d.rng.rolling(20).median()
    d["vmed"]=d.volume.rolling(20).median()

    pc=c.shift()
    tr=pd.concat([
        d.high-d.low,
        (d.high-pc).abs(),
        (d.low-pc).abs()
    ],axis=1).max(axis=1)

    d["atr"]=tr.ewm(
        alpha=1/20,adjust=False,min_periods=20
    ).mean()

    return d

def events(d):
    out=[]
    for i in range(1,len(d)):
        r,p=d.iloc[i],d.iloc[i-1]

        if not np.isfinite(r.atr+r.z+p.z):
            continue

        common=(
            r.rng>=r.rngmed and
            r.volume>=r.vmed
        )

        if common and r.close>r.ema and p.z<=-2 and r.z>-2:
            out.append((i,"LONG"))

        elif common and r.close<r.ema and p.z>=2 and r.z<2:
            out.append((i,"SHORT"))

    return out

def trade(d,i,side,tp):
    e=float(d.close.iloc[i])
    atr=float(d.atr.iloc[i])

    sl=SL_ATR*atr
    target=tp*sl

    if i+HOLD>=len(d):
        return None

    for j in range(i+1,i+HOLD+1):
        h=float(d.high.iloc[j])
        l=float(d.low.iloc[j])

        if side=="LONG":
            if l<=e-sl:
                return -1-COST
            if h>=e+target:
                return tp-COST
        else:
            if h>=e+sl:
                return -1-COST
            if l<=e-target:
                return tp-COST

    return -COST

print("="*60)
print("CANDIDATE 11 — FIXED END + CONCENTRATION")
print("="*60)

data={}
for s in SYMBOLS:
    d=fetch(s)

    if len(d)<4300:
        print(s,"INVALID")
        continue

    data[s]=d
    print(s,len(d))

print("\nVALID SYMBOLS =",len(data))

if len(data)!=len(SYMBOLS):
    print("DATA AUDIT = FAIL")
    print("STOP")
    raise SystemExit

# Common timestamps
common=set.intersection(*[
    set(d.time) for d in data.values()
])

print("COMMON TIMESTAMPS =",len(common))

if len(common)<4300:
    print("COMMON AUDIT = FAIL")
    raise SystemExit

for s in data:
    data[s]=data[s][data[s].time.isin(common)].reset_index(drop=True)

print("DATA AUDIT = PASS")

# ------------------------------------------------------------
# Trades by symbol
# ------------------------------------------------------------

results={tp:{} for tp in TPs}

for s,d in data.items():

    ev=events(d)

    for tp in TPs:
        results[tp][s]=[]

        for i,side in ev:
            r=trade(d,i,side,tp)
            if r is not None:
                results[tp][s].append(r)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def total(x):
    return sum(x)

def pf(x):
    pos=sum(v for v in x if v>0)
    neg=-sum(v for v in x if v<0)
    return pos/neg if neg else 999

# ------------------------------------------------------------
# Fixed-window performance
# ------------------------------------------------------------

print("\n"+"="*60)
print("FIXED WINDOW PERFORMANCE")
print("="*60)

for tp in TPs:

    allr=[]
    for s in SYMBOLS:
        allr += results[tp][s]

    print(
        f"TP{tp}: N={len(allr)} "
        f"Exp={np.mean(allr):.4f} "
        f"Total={total(allr):.2f}R "
        f"PF={pf(allr):.3f}"
    )

# ------------------------------------------------------------
# Symbol concentration
# ------------------------------------------------------------

print("\n"+"="*60)
print("SYMBOL CONCENTRATION")
print("="*60)

for tp in TPs:

    sym_total={
        s:total(results[tp][s])
        for s in SYMBOLS
    }

    ordered=sorted(
        sym_total.items(),
        key=lambda x:x[1],
        reverse=True
    )

    portfolio=sum(sym_total.values())

    top1=sum(v for _,v in ordered[:1])
    top3=sum(v for _,v in ordered[:3])
    top5=sum(v for _,v in ordered[:5])

    print(f"\nTP{tp}")
    print(f"PORTFOLIO = {portfolio:.2f}R")
    print(f"REMOVE TOP1 = {portfolio-top1:.2f}R")
    print(f"REMOVE TOP3 = {portfolio-top3:.2f}R")
    print(f"REMOVE TOP5 = {portfolio-top5:.2f}R")

    print("TOP SYMBOLS:")
    for s,v in ordered[:5]:
        print(f"  {s}: {v:.2f}R")

print("\n"+"="*60)
print("FIXED-END + CONCENTRATION TEST COMPLETED")
print("="*60)
