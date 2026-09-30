import requests,pandas as pd,numpy as np,time

S=["BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT",
"ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT","SUI-USDT","APT-USDT",
"NEAR-USDT","OP-USDT","ARB-USDT","ATOM-USDT","FIL-USDT","LTC-USDT",
"BCH-USDT","ETC-USDT","UNI-USDT","AAVE-USDT","INJ-USDT","SEI-USDT",
"WIF-USDT","PEPE-USDT","FET-USDT","RENDER-USDT","TAO-USDT","TIA-USDT",
"IMX-USDT","STX-USDT","ALGO-USDT","HBAR-USDT","ICP-USDT"]

URL="https://api.kucoin.com/api/v1/market/candles"
INT=14400;DAYS=365;HOLD=30;COST=.003
TPs=[1,1.5,2,3]

def get(s):
    end=int(time.time())//INT*INT-1
    cur=end-DAYS*86400;out=[]
    while cur<end:
        e=min(cur+INT*1499,end)
        ok=False
        for _ in range(3):
            try:
                r=requests.get(URL,params={"symbol":s,"type":"4hour",
                "startAt":cur,"endAt":e},timeout=20).json()
                if r.get("code")=="200000":
                    out+=r["data"];ok=True;break
            except:pass
            time.sleep(1)
        if not ok:return None
        if not r["data"]:break
        cur=max(int(x[0]) for x in r["data"])+INT
    if not out:return None
    d=pd.DataFrame(out,columns=["t","o","c","h","l","v","turn"])
    d=d.drop_duplicates("t").sort_values("t")
    for x in ["o","c","h","l","v"]:d[x]=pd.to_numeric(d[x])
    d["t"]=pd.to_datetime(d.t.astype(int),unit="s",utc=True)
    return d.reset_index(drop=True)

def rma(x,n):
    return x.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def prep(d):
    h,l,c,v=d.h,d.l,d.c,d.v
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,20)

    mid=c.rolling(20).mean()
    sd=c.rolling(20).std()
    z=(c-mid)/sd

    rng=h.rolling(20).max()-l.rolling(20).min()
    rng_med=rng.rolling(20).median()
    vr=v.rolling(20).median()

    ema=c.ewm(span=200,adjust=False).mean()

    long=(c>ema)&(z.shift(1)<=-2)&(z>-2)&(rng>=rng_med)&(v>=vr)
    short=(c<ema)&(z.shift(1)>=2)&(z<2)&(rng>=rng_med)&(v>=vr)

    d=d.copy()
    d["atr"]=atr
    d["sig"]=np.where(long,1,np.where(short,-1,0))
    return d

def trade(d,i,sg,tp):
    if i+HOLD>=len(d):return None
    e=d.c.iloc[i];a=d.atr.iloc[i]
    sl=e-sg*1.25*a
    target=e+sg*1.25*a*tp
    for j in range(i+1,i+HOLD+1):
        hi,lo=d.h.iloc[j],d.l.iloc[j]
        slhit=(lo<=sl) if sg==1 else (hi>=sl)
        tphit=(hi>=target) if sg==1 else (lo<=target)
        if slhit:return -1-COST
        if tphit:return 1.25*tp-COST
    return -COST

def stats(x):
    x=np.array(x,float)
    if not len(x):return 0,0,0,0,0
    win=(x>0).mean()
    ex=x.mean();total=x.sum()
    pos=x[x>0].sum();neg=x[x<0].sum()
    pf=pos/abs(neg) if neg else np.inf
    eq=np.cumsum(x);dd=eq-np.maximum.accumulate(eq)
    return len(x),win,ex,total,pf,dd.min()

print("="*70)
print("SETUP V4 — CANDIDATE 11")
print("RANGE EXTENSION → REVERSION")
print("="*70)
print("4H | ZSCORE20 ±2 | EMA200 | RANGE20>=MEDIAN | VOL>=MEDIAN")
print("SL=1.25ATR | HOLD=30 | TP=1/1.5/2/3R | COST=.003R")
print("LONG + SHORT | CAUSAL | NO OPTIMIZATION | NO EXTRA FILTERS")

D={};failed=[]

for n,s in enumerate(S,1):
    d=get(s)
    if d is None or len(d)<180*2:
        print(f"[{n:02}/{len(S)}] {s} FAILED")
        failed.append(s);continue
    D[s]=prep(d)
    print(f"[{n:02}/{len(S)}] {s} OK {len(d)} candles")

if len(D)<10:raise SystemExit("NOT ENOUGH VALID SYMBOLS")

common=set(D[next(iter(D))].t)
for d in D.values():common &= set(d.t)
common=sorted(common)

events=[]
for s,d in D.items():
    ix={t:i for i,t in enumerate(d.t)}
    for t in common:
        i=ix[t];sg=d.sig.iloc[i]
        if sg:events.append((t,s,i,int(sg)))

events.sort()

print("\nVALID SYMBOLS =",len(D))
print("COMMON TIMESTAMPS =",len(common))
print("EVENTS =",len(events))
print("FAILED =",failed)

for tp in TPs:
    rows=[]
    for t,s,i,sg in events:
        r=trade(D[s],i,sg,tp)
        if r is not None:rows.append(r)
    z=stats(rows)
    print(f"TP{tp}R n={z[0]} WR={z[1]:.4f} "
          f"NetExp={z[2]:.4f} Total={z[3]:.3f} PF={z[4]:.4f} DD={z[5]:.3f}")

for tp in [2]:
    L=[];Sh=[]
    for t,s,i,sg in events:
        r=trade(D[s],i,sg,tp)
        if r is None:continue
        (L if sg==1 else Sh).append(r)
    a=stats(L);b=stats(Sh)
    print("\nTP2R DIRECTION")
    print(f"LONG  n={a[0]} WR={a[1]:.4f} NetExp={a[2]:.4f} PF={a[4]:.4f} Total={a[3]:.3f}")
    print(f"SHORT n={b[0]} WR={b[1]:.4f} NetExp={b[2]:.4f} PF={b[4]:.4f} Total={b[3]:.3f}")

print("\n"+"="*70)
print("DISCOVERY GATE")
print("="*70)
print("This is DISCOVERY ONLY.")
print("No TP selection. No direction selection.")
print("No parameter optimization.")
print("No production deployment.")
