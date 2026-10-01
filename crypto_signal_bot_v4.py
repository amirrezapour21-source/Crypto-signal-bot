import requests,pandas as pd,numpy as np,time

BASE="https://api.kucoin.com/api/v1/market/candles"
SYMS=["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT","SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC","BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP","WIF","PEPE","FLOKI"]
N=4380; HOLD=30; COST=.003; ATR_M=1.25; STEP=14400

def fetch(s):
    rows=[]; end=int(pd.Timestamp("2026-09-30T00:00:00Z").timestamp())
    while len(rows)<N+50:
        r=requests.get(BASE,params={"symbol":f"{s}-USDT","type":"4hour","endAt":end},timeout=20)
        r.raise_for_status(); x=r.json()["data"]
        if not x: break
        rows+=x
        end=min(int(z[0]) for z in x)-STEP
        time.sleep(.08)
    d=pd.DataFrame(rows,columns=["ts","open","close","high","low","vol","turn"])
    d["ts"]=pd.to_datetime(pd.to_numeric(d.ts),unit="s",utc=True)
    for c in ["open","close","high","low","vol"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.drop_duplicates("ts").sort_values("ts").tail(N).reset_index(drop=True)
    if len(d)!=N: raise ValueError(f"{s}: received {len(d)}")
    if d.ts.diff().dropna().dt.total_seconds().ne(STEP).any():
        raise ValueError(f"{s}: timestamp gap")
    return d

def ind(d):
    c,h,l,v=d.close,d.high,d.low,d.vol
    d["z"]=(c-c.rolling(20).mean())/c.rolling(20).std()
    d["ema"]=c.ewm(span=200,adjust=False).mean()
    d["range20"]=h.rolling(20).max()-l.rolling(20).min()
    d["rmed"]=d.range20.rolling(20).median()
    d["vmed"]=v.rolling(20).median()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/20,adjust=False).mean()
    return d

def events(d,tp):
    out=[]
    for i in range(21,len(d)-HOLD):
        if not(d.z.iloc[i-1]>=2 and d.z.iloc[i]<2): continue
        if not(d.close.iloc[i]<d.ema.iloc[i]): continue
        if not(d.range20.iloc[i]>=d.rmed.iloc[i]): continue
        if not(d.vol.iloc[i]>=d.vmed.iloc[i]): continue
        e=float(d.close.iloc[i]); a=float(d.atr.iloc[i])
        if not np.isfinite(a) or a<=0: continue
        sl=e+ATR_M*a; target=e-tp*ATR_M*a
        r=-COST; ex=i+HOLD-1
        for j in range(i+1,i+HOLD):
            hi,lo=float(d.high.iloc[j]),float(d.low.iloc[j])
            if hi>=sl: r=-1-COST; ex=j; break
            if lo<=target: r=tp-COST; ex=j; break
        out.append([d.ts.iloc[i],d.ts.iloc[ex],r])
    return pd.DataFrame(out,columns=["entry","exit","r"])

def stats(x):
    if len(x)==0:return 0,np.nan,np.nan,0,0
    r=x.r.to_numpy(float)
    win=r[r>0].sum(); loss=-r[r<0].sum()
    eq=np.cumsum(r)
    dd=(eq-np.maximum.accumulate(eq)).min()
    return len(r),r.mean(),win/loss if loss else np.inf,r.sum(),dd

print("="*70)
print("CANDIDATE 11 — SHORT ONLY — FIXED R ENGINE — 5-FOLD")
print("="*70)

data={}
for s in SYMS:
    try:
        data[s]=ind(fetch(s))
        print(f"{s:<6} {len(data[s])} candles")
    except Exception as e:
        print(f"{s:<6} ERROR: {e}")

if len(data)!=len(SYMS):
    raise SystemExit(f"ABORT: {len(SYMS)-len(data)} symbols failed")

common=set(data[SYMS[0]].ts)
for s in SYMS[1:]: common &= set(data[s].ts)
common=sorted(common)

if len(common)!=N:
    raise SystemExit(f"ABORT: COMMON TIMESTAMPS={len(common)} EXPECTED={N}")

print(f"\nDATA AUDIT = PASS")
print(f"VALID SYMBOLS = {len(data)}")
print(f"COMMON TIMESTAMPS = {len(common)}")
print(f"START = {common[0]}")
print(f"END   = {common[-1]}")

folds=np.array_split(np.array(common),5)

for tp in [1.5,2.0]:
    evs=[]
    for s,d in data.items():
        e=events(d,tp)
        if len(e):
            e["symbol"]=s
            evs.append(e)

    ev=pd.concat(evs,ignore_index=True).sort_values("entry").reset_index(drop=True)
    oos=[]

    print("\n"+"="*70)
    print(f"TP{tp:g}R — SHORT ONLY")
    print(f"TOTAL GENERATED EVENTS = {len(ev)}")
    print("="*70)

    for k in range(5):
        start=pd.Timestamp(folds[k][0])
        end=pd.Timestamp(folds[k][-1])+pd.Timedelta(hours=4)

        isx=ev[(ev.entry<start)&(ev.exit<start)]
        ox=ev[(ev.entry>=start)&(ev.entry<end)&(ev.exit<end)]

        a=stats(isx); b=stats(ox)
        print(f"FOLD {k+1}")
        print(f"IS  N={a[0]:4d} Exp={a[1]:+.4f} PF={a[2]:.3f} Total={a[3]:+.2f}R DD={a[4]:+.2f}R")
        print(f"OOS N={b[0]:4d} Exp={b[1]:+.4f} PF={b[2]:.3f} Total={b[3]:+.2f}R DD={b[4]:+.2f}R")
        oos.append(ox)

    z=stats(pd.concat(oos,ignore_index=True))
    positive=sum(stats(x)[3]>0 for x in oos)

    print("-"*70)
    print(f"TP{tp:g}R AGGREGATE OOS")
    print(f"N={z[0]} Exp={z[1]:+.4f} PF={z[2]:.3f} Total={z[3]:+.2f}R DD={z[4]:+.2f}R")
    print(f"Positive OOS folds = {positive}/5")
    print("TEMPORAL VALIDATION =",
          "PASS" if positive>=4 and z[1]>0 and z[2]>1 else "FAIL")

print("\nRUN COMPLETE")
