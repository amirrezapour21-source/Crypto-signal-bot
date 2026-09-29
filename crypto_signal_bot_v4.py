import requests,pandas as pd,numpy as np,time

S=[
"BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT",
"ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT","TON-USDT","SUI-USDT",
"APT-USDT","NEAR-USDT","OP-USDT","ARB-USDT","ATOM-USDT","FIL-USDT",
"LTC-USDT","BCH-USDT","ETC-USDT","UNI-USDT","AAVE-USDT","INJ-USDT",
"SEI-USDT","WIF-USDT","PEPE-USDT","FET-USDT","RENDER-USDT","TAO-USDT",
"TIA-USDT","IMX-USDT","MKR-USDT","MATIC-USDT","STX-USDT","ALGO-USDT",
"HBAR-USDT","ICP-USDT"]

URL="https://api.kucoin.com/api/v1/market/candles"
INT=14400; DAYS=730; HOLD=30; COST=.003
TPs=[1,1.5,2,3]

def get(sym):
    end=int(time.time())//INT*INT-1
    cur=end-DAYS*86400; out=[]
    while cur<end:
        e=min(cur+INT*1499,end)
        ok=False
        for _ in range(3):
            try:
                r=requests.get(URL,params={"symbol":sym,"type":"4hour",
                    "startAt":cur,"endAt":e},timeout=20).json()
                if r.get("code")=="200000":
                    out+=r["data"];ok=True;break
            except: pass
            time.sleep(1)
        if not ok:return None
        if not r["data"]:break
        mx=max(int(x[0]) for x in r["data"])
        if mx<cur:break
        cur=mx+INT
    if not out:return None
    df=pd.DataFrame(out,columns=["t","o","c","h","l","v","turn"])
    df=df.drop_duplicates("t").sort_values("t")
    for x in ["o","c","h","l","v"]:df[x]=pd.to_numeric(df[x])
    df["t"]=pd.to_datetime(df.t.astype(int),unit="s",utc=True)
    return df.reset_index(drop=True)

def rma(x,n):
    return x.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def prep(d):
    h,l,c,v=d.h,d.l,d.c,d.v
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,20)
    up=h.diff();dn=-l.diff()
    plus=rma(up.where((up>dn)&(up>0),0),14)
    minus=rma(dn.where((dn>up)&(dn>0),0),14)
    pdi=100*plus/atr;mdi=100*minus/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi)
    adx=rma(dx,14)
    er=(c-c.shift(20))/c.diff().abs().rolling(20).sum()
    ema=c.ewm(span=200,adjust=False).mean()
    am=atr.rolling(20).median()
    vm=v.rolling(20).median()

    long=(c>ema)&(er.shift(1)<=.55)&(er>.55)&(adx>=20)&(adx>adx.shift(1))&(pdi>mdi)&(atr>=am)&(v>=vm)
    short=(c<ema)&(er.shift(1)>=-.55)&(er<-.55)&(adx>=20)&(adx>adx.shift(1))&(mdi>pdi)&(atr>=am)&(v>=vm)

    d=d.copy();d["atr"]=atr
    d["sig"]=np.where(long,1,np.where(short,-1,0))
    return d

def trade(d,i,direction,tp):
    if i+HOLD>=len(d):return None
    e=d.c.iloc[i]; a=d.atr.iloc[i]
    sl=e-direction*1.25*a
    target=e+direction*1.25*a*tp
    for j in range(i+1,i+HOLD+1):
        hi,lo=d.h.iloc[j],d.l.iloc[j]
        slhit=(lo<=sl) if direction==1 else (hi>=sl)
        tphit=(hi>=target) if direction==1 else (lo<=target)
        if slhit:return -1-COST
        if tphit:return 1.25*tp-COST
    return -COST

def stats(rows):
    x=np.array(rows,float)
    if not len(x):return (0,0,0,0,0)
    win=(x>0).mean(); ex=x.mean(); total=x.sum()
    pos=x[x>0].sum();neg=x[x<0].sum()
    pf=pos/abs(neg) if neg else np.inf
    eq=np.cumsum(x);dd=eq-np.maximum.accumulate(eq)
    return len(x),win,ex,total,pf,dd.min()

print("="*80)
print("SETUP V4 — CANDIDATE 10")
print("5-FOLD WALK-FORWARD VALIDATION")
print("="*80)
print("FROZEN: 4H | ER20 ±0.55 | EMA200 | ADX14>=20 | ATR20")
print("SL=1.25ATR | HOLD=30 | TP=1/1.5/2/3R | COST=.003R")
print("LONG + SHORT | NO OPTIMIZATION | NO EXTRA FILTERS")

D={}; failed=[]
for n,s in enumerate(S,1):
    d=get(s)
    if d is None or len(d)<180*6:
        print(f"[{n:02}/{len(S)}] {s} FAILED");failed.append(s);continue
    d=prep(d);D[s]=d
    print(f"[{n:02}/{len(S)}] {s} OK {len(d)} candles")

if len(D)<10:raise SystemExit("NOT ENOUGH VALID SYMBOLS")

common=set(D[next(iter(D))].t)
for d in D.values():common &= set(d.t)
common=sorted(common)
lo,hi=common[0],common[-1]

events=[]
for s,d in D.items():
    ix={t:i for i,t in enumerate(d.t)}
    for t in common:
        i=ix[t];sg=d.sig.iloc[i]
        if sg:
            events.append((t,s,i,int(sg)))

events.sort()
print("\nVALID SYMBOLS =",len(D))
print("COMMON TIMESTAMPS =",len(common))
print("EVENTS =",len(events))
print("FAILED =",failed)

# 5 chronological 10%-wide test folds: 50-60 ... 90-100
N=len(common); results={tp:[] for tp in TPs}

for f in range(5):
    a=int(N*(.50+.10*f));b=int(N*(.60+.10*f))
    ws,we=common[a],common[min(b,N)-1]
    fe=[e for e in events if ws<=e[0]<=we]
    print(f"\nWF{f+1}: {ws} -> {we} | events={len(fe)}")

    for tp in TPs:
        rows=[]
        for t,s,i,sg in fe:
            r=trade(D[s],i,sg,tp)
            if r is not None:rows.append(r)
        z=stats(rows);results[tp].append(z)
        print(f"TP{tp}R n={z[0]} WR={z[1]:.4f} "
              f"NetExp={z[2]:.4f} Total={z[3]:.3f} PF={z[4]:.4f} DD={z[5]:.3f}")

print("\n"+"="*80)
print("AGGREGATE WALK-FORWARD")
print("="*80)

for tp in TPs:
    z=results[tp]
    x=np.concatenate([np.array([]) if q[0]==0 else [] for q in []]) if False else None
    n=sum(q[0] for q in z)
    total=sum(q[3] for q in z)
    exp=total/n if n else 0
    pos=sum(q[2]>0 for q in z)
    pf=sum(q[4]*0 for q in z) # placeholder replaced below
    # aggregate PF from fold totals is not valid; recompute from fold PF/exp is unavailable.
    # Use fold-positive stability as primary WF gate.
    print(f"TP{tp}R | n={n} | PositiveFolds={pos}/5 | "
          f"AvgFoldExp={np.mean([q[2] for q in z]):.4f} | Total={total:.3f}")

print("\n"+"="*80)
print("WF RESEARCH GATE")
print("="*80)
print("REQUIRED: >=4/5 positive folds")
print("No TP selection from a single fold.")
print("No direction selection.")
print("No parameter optimization.")
print("No production deployment from this run.")

for tp in TPs:
    pos=sum(q[2]>0 for q in results[tp])
    print(f"TP{tp}R PositiveFolds={pos}/5",
          "PASS" if pos>=4 else "FAIL")
