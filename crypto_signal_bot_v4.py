# crypto_signal_bot_v4.py
import requests, pandas as pd, numpy as np, time

BASE="https://api.kucoin.com/api/v1/market/candles"
SYMS=["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT","SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC","BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP","WIF","PEPE","FLOKI"]
N=4380; HOLD=30; ATR_M=1.25; COST=.003
TFS="4hour"; STEP=4*3600

def fetch(sym):
    rows=[]; end=int(pd.Timestamp("2026-09-30T00:00:00Z").timestamp())
    for _ in range(10):
        r=requests.get(BASE,params={"symbol":f"{sym}-USDT","type":TFS,"endAt":end},timeout=20)
        r.raise_for_status()
        x=r.json()["data"]
        if not x: break
        rows += x
        mn=min(int(z[0]) for z in x)
        end=mn-STEP
        if len(rows)>=N+20: break
        time.sleep(.05)

    d=pd.DataFrame(rows,columns=["ts","open","close","high","low","vol","turn"])
    d["ts"]=pd.to_datetime(pd.to_numeric(d.ts),unit="s",utc=True)
    for c in ["open","close","high","low","vol"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    d=d.tail(N).reset_index(drop=True)

    if len(d)!=N: raise ValueError(f"{sym}: received {len(d)}")
    gap=d.ts.diff().dropna().dt.total_seconds().ne(STEP).any()
    if gap: raise ValueError(f"{sym}: timestamp gap")
    return d

def indicators(d):
    c,h,l,v=d.close,d.high,d.low,d.vol
    mean=c.rolling(20).mean()
    sd=c.rolling(20).std(ddof=0)
    d["z"]=(c-mean)/sd.replace(0,np.nan)
    d["ema"]=c.ewm(span=200,adjust=False).mean()
    d["range20"]=h.rolling(20).max()-l.rolling(20).min()
    d["rmed"]=d.range20.rolling(20).median()
    d["vmed"]=v.rolling(20).median()

    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/20,adjust=False).mean()
    return d

def events(d,tp_mult):
    out=[]
    for i in range(21,len(d)-HOLD):
        if not (d.z.iloc[i-1]>=2 and d.z.iloc[i]<2): continue
        if not (d.close.iloc[i]<d.ema.iloc[i]): continue
        if not (d.range20.iloc[i]>=d.rmed.iloc[i]): continue
        if not (d.vol.iloc[i]>=d.vmed.iloc[i]): continue
        e=float(d.close.iloc[i]); a=float(d.atr.iloc[i])
        if not np.isfinite(a) or a<=0: continue
        sl=e+ATR_M*a; tp=e-TP_mult*ATR_M*a
        result=-COST; exit_i=i+HOLD-1
        for j in range(i+1,i+HOLD):
            hi,lo=float(d.high.iloc[j]),float(d.low.iloc[j])
            if hi>=sl:                 # SL first
                result=-1-COST; exit_i=j; break
            if lo<=tp:
                result=tp_mult-COST; exit_i=j; break
        out.append({"entry":d.ts.iloc[i],"exit":d.ts.iloc[exit_i],"r":result})
    return pd.DataFrame(out)

def stats(x):
    if len(x)==0: return (0,np.nan,np.inf,0,0)
    r=x.r.to_numpy(float)
    exp=r.mean(); pos=r[r>0].sum(); neg=-r[r<0].sum()
    pf=pos/neg if neg>0 else np.inf
    eq=np.cumsum(r); dd=(eq-np.maximum.accumulate(eq)).min()
    return len(r),exp,pf,r.sum(),dd

print("="*72)
print("CANDIDATE 11 — SHORT-ONLY — FIXED R ENGINE — 5-FOLD")
print("="*72)

data={}; bad={}
for s in SYMS:
    try:
        data[s]=indicators(fetch(s))
        print(f"{s:<6} {len(data[s])} candles  {data[s].ts.iloc[0]} -> {data[s].ts.iloc[-1]}")
    except Exception as e:
        bad[s]=str(e); print(f"{s:<6} ERROR: {e}")

if bad:
    raise SystemExit(f"ABORT: {len(bad)} symbols failed")

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

all_ts=pd.Series(common)
folds=np.array_split(all_ts.to_numpy(),5)

for tp_mult in [1.5,2.0]:
    all_events=[]
    for s,d in data.items():
        e=events(d,tp_mult)
        if not e.empty:
            e["symbol"]=s
            all_events.append(e)

    ev=pd.concat(all_events,ignore_index=True).sort_values("entry").reset_index(drop=True)
    print("\n"+"="*72)
    print(f"TP{tp_mult:g}R — SHORT ONLY")
    print("="*72)
    print(f"TOTAL EVENTS = {len(ev)}")

    oos_all=[]

    for k in range(5):
        start=pd.Timestamp(folds[k][0])
        end=pd.Timestamp(folds[k][-1])+pd.Timedelta(hours=4)

        isx=ev[(ev.entry<start)&(ev.exit<start)]
        oos=ev[(ev.entry>=start)&(ev.entry<end)&(ev.exit<end)]

        a=stats(isx); b=stats(oos)
        print(f"\nFOLD {k+1}")
        print(f"IS  {start} -> {start if k==0 else start}")
        print(f"IS      N={a[0]:4d} Exp={a[1]:+.4f} PF={a[2]:.3f} Total={a[3]:+.2f}R DD={a[4]:+.2f}R")
        print(f"OOS {start} -> {end}")
        print(f"OOS     N={b[0]:4d} Exp={b[1]:+.4f} PF={b[2]:.3f} Total={b[3]:+.2f}R DD={b[4]:+.2f}R")
        oos_all.append(oos)

    oos=pd.concat(oos_all,ignore_index=True)
    z=stats(oos)
    positive=sum(stats(x)[3]>0 for x in oos_all)

    print("\n"+"-"*72)
    print(f"TP{tp_mult:g}R AGGREGATE OOS")
    print(f"OOS ALL N={z[0]} Exp={z[1]:+.4f} PF={z[2]:.3f} Total={z[3]:+.2f}R DD={z[4]:+.2f}R")
    print(f"Positive OOS folds = {positive}/5")
    print("TEMPORAL VALIDATION =", "PASS" if positive>=4 and z[1]>0 and z[2]>1 else "FAIL")

print("\n"+"="*72)
print("RUN COMPLETE")
print("="*72)
