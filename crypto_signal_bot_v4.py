# CANDIDATE 6 — 5-FOLD WALK-FORWARD VALIDATION
import time, json, urllib.request
import numpy as np
import pandas as pd

SYMBOLS = ['BTC-USDT','ETH-USDT','SOL-USDT','BNB-USDT','XRP-USDT','DOGE-USDT','ADA-USDT','LINK-USDT','AVAX-USDT','DOT-USDT','LTC-USDT','BCH-USDT','UNI-USDT','AAVE-USDT','ATOM-USDT','NEAR-USDT','FIL-USDT','ETC-USDT','ICP-USDT','APT-USDT','ARB-USDT','OP-USDT','SUI-USDT','SEI-USDT','INJ-USDT','TIA-USDT','JUP-USDT','PEPE-USDT','SHIB-USDT','TRX-USDT','HBAR-USDT','VET-USDT','ALGO-USDT','STX-USDT','RUNE-USDT']
INTERVAL=14400; DAYS=730; MAX_PAGE=1500; SL_ATR=1.25; TP_R=2.0; HOLD=30; COST_R=0.003

def fetch(symbol):
    now=int(time.time()); cur=now-DAYS*86400; out=[]
    while cur<now:
        end=min(cur+(MAX_PAGE-1)*INTERVAL,now)
        url=f'https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type=4hour&startAt={cur}&endAt={end}'
        ok=False
        for a in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req,timeout=20) as r: x=json.loads(r.read().decode())
                if x.get('code')=='200000': out+=x.get('data',[]); ok=True; break
            except Exception:
                if a<2: time.sleep(1+a)
        if not ok: return pd.DataFrame()
        cur=end+INTERVAL; time.sleep(.08)
    if not out: return pd.DataFrame()
    df=pd.DataFrame(out,columns=['time','open','close','high','low','volume','turnover'])
    df['time']=pd.to_numeric(df['time'])
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    cutoff=(now//INTERVAL)*INTERVAL
    return df[df.time<cutoff][['time','open','high','low','close','volume']].reset_index(drop=True)

def indicators(d):
    d=d.copy(); pc=d.close.shift(1)
    mid=d.close.rolling(20).mean(); sd=d.close.rolling(20).std(ddof=0)
    d['bb_low']=mid-2*sd; d['bb_high']=mid+2*sd
    delta=d.close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['rsi']=100-100/(1+ag/al.replace(0,np.nan))
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d['atr']=tr.ewm(alpha=1/20,adjust=False,min_periods=20).mean()
    up=d.high.diff(); down=-d.low.diff()
    pdm=up.where((up>down)&(up>0),0.0); mdm=down.where((down>up)&(down>0),0.0)
    a14=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    pdi=100*pdm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/a14
    mdi=100*mdm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/a14
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['adx']=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    return d

def events(d):
    out=[]
    for i in range(1,len(d)):
        r,p=d.iloc[i],d.iloc[i-1]
        if any(pd.isna(r[x]) for x in ['atr','adx','bb_low','bb_high']) or pd.isna(p.rsi): continue
        if p.close<p.bb_low and r.close>=r.bb_low and p.rsi<=30 and r.adx<30: out.append((i,'LONG'))
        elif p.close>p.bb_high and r.close<=r.bb_high and p.rsi>=70 and r.adx<30: out.append((i,'SHORT'))
    return out

def trade(d,i,side,end):
    entry=float(d.iloc[i].close); risk=SL_ATR*float(d.iloc[i].atr)
    sl,tp=(entry-risk,entry+TP_R*risk) if side=='LONG' else (entry+risk,entry-TP_R*risk)
    last=min(i+HOLD,end)
    if last<i+HOLD: return None
    for j in range(i+1,last+1):
        c=d.iloc[j]
        slhit,tp_hit=(c.low<=sl,c.high>=tp) if side=='LONG' else (c.high>=sl,c.low<=tp)
        if slhit: return (-1-COST_R,int(c.time))
        if tp_hit: return (TP_R-COST_R,int(c.time))
    return (-COST_R,int(d.iloc[last].time))

def metrics(ts):
    if not ts: return (0,0,0,0,0,0)
    v=np.array([x[0] for x in sorted(ts,key=lambda z:(z[1],z[2],z[3]))]); n=len(v)
    pos=v[v>0].sum(); neg=abs(v[v<0].sum()); pf=pos/neg if neg else 0
    c=np.cumsum(v); dd=np.max(np.maximum.accumulate(c)-c) if len(c) else 0
    return n,(v>0).sum()/n,v.mean(),pf,v.sum(),dd

print('='*72); print('CANDIDATE 6 — 5-FOLD WALK-FORWARD VALIDATION'); print('='*72)
cache={}
for k,s in enumerate(SYMBOLS,1):
    d=fetch(s)
    if len(d)>=180: cache[s]=indicators(d)
    print(f'{k:02d}/{len(SYMBOLS)} {s}: {len(d)} candles')
if len(cache)<30: raise RuntimeError(f'Insufficient valid symbols: {len(cache)}')
common=None
for d in cache.values():
    t=set(d.time.astype(int)); common=t if common is None else common&t
common=sorted(common)
if len(common)<4000: raise RuntimeError(f'Common timestamp count too low: {len(common)}')
for s in list(cache): cache[s]=cache[s][cache[s].time.isin(common)].reset_index(drop=True)
print(f'VALID SYMBOLS = {len(cache)}'); print(f'COMMON 4H TIMESTAMPS = {len(common)}')
print(f'START = {pd.to_datetime(common[0],unit="s",utc=True)}'); print(f'END   = {pd.to_datetime(common[-1],unit="s",utc=True)}')

folds=[(.50,.60),(.60,.70),(.70,.80),(.80,.90),(.90,1.00)]; results=[]; all_trades=[]
for k,(a,b) in enumerate(folds,1):
    start=int(len(common)*a); end=int(len(common)*b)-1; ft=[]; op=0; ev=0
    for s,d in cache.items():
        for i,side in events(d):
            if start<=i<=end:
                ev+=1; z=trade(d,i,side,end)
                if z is None: op+=1; continue
                ft.append((z[0],z[1],s,side))
    m=metrics(ft); results.append(m); all_trades.extend(ft)
    print(f'\n--- OOS FOLD {k} ({a*100:.0f}%-{b*100:.0f}%) ---')
    print(f'events            = {ev}'); print(f'traded            = {m[0]}'); print(f'open_at_fold_end  = {op}')
    print(f'WR                = {m[1]:.4f}'); print(f'NetExp            = {m[2]:+.4f}R'); print(f'NetPF             = {m[3]:.3f}'); print(f'NetTotalR         = {m[4]:+.2f}R'); print(f'NetMaxDD          = -{m[5]:.2f}R')
agg=metrics(all_trades); pos=sum(r[2]>0 for r in results)
print('\n'+'='*72); print('AGGREGATE OOS RESULTS'); print('='*72)
print(f'traded            = {agg[0]}'); print(f'WR                = {agg[1]:.4f}'); print(f'Aggregate NetExp  = {agg[2]:+.4f}R'); print(f'Aggregate NetPF   = {agg[3]:.3f}'); print(f'Aggregate TotalR  = {agg[4]:+.2f}R'); print(f'Aggregate MaxDD   = -{agg[5]:.2f}R'); print(f'Positive Folds    = {pos}/5')
status='PASS' if agg[2]>0 and agg[3]>1 and pos>=4 else 'FAIL'; print(f'WF STATUS         = {status}'); print('='*72)
