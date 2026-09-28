# CANDIDATE 7 — DAILY REGIME + 4H PULLBACK MOMENTUM SCREEN
import time, json, urllib.request
import numpy as np
import pandas as pd

SYMBOLS = [
'BTC-USDT','ETH-USDT','SOL-USDT','BNB-USDT','XRP-USDT','DOGE-USDT','ADA-USDT',
'LINK-USDT','AVAX-USDT','DOT-USDT','LTC-USDT','BCH-USDT','UNI-USDT','AAVE-USDT',
'ATOM-USDT','NEAR-USDT','FIL-USDT','ETC-USDT','ICP-USDT','APT-USDT','ARB-USDT',
'OP-USDT','SUI-USDT','SEI-USDT','INJ-USDT','TIA-USDT','JUP-USDT','PEPE-USDT',
'SHIB-USDT','TRX-USDT','HBAR-USDT','VET-USDT','ALGO-USDT','STX-USDT','RUNE-USDT']

INT=14400; DAYS=365; PAGE=1500; HOLD=30; SL_ATR=1.25; TP_R=2.0; COST=.003

def fetch(s):
    now=int(time.time()); cur=now-DAYS*86400; out=[]
    while cur<now:
        end=min(cur+(PAGE-1)*INT,now)
        u=f"https://api.kucoin.com/api/v1/market/candles?symbol={s}&type=4hour&startAt={cur}&endAt={end}"
        for k in range(3):
            try:
                q=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(q,timeout=20) as r: x=json.loads(r.read().decode())
                if x.get('code')=='200000': out+=x.get('data',[]); break
            except Exception:
                if k==2:return pd.DataFrame()
                time.sleep(1+k)
        cur=end+INT
    if not out:return pd.DataFrame()
    d=pd.DataFrame(out,columns=['time','open','close','high','low','volume','turnover'])
    for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d['time']=pd.to_numeric(d['time']); d=d.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    d=d[d.time < (now//INT)*INT].reset_index(drop=True)
    return d[['time','open','high','low','close','volume']]

def ind(d):
    d=d.copy()
    d['e20']=d.close.ewm(span=20,adjust=False).mean()
    d['e50']=d.close.ewm(span=50,adjust=False).mean()
    d['e200']=d.close.ewm(span=200,adjust=False).mean()
    de=d.close.diff(); g=de.clip(lower=0); l=-de.clip(upper=0)
    ag=g.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    al=l.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['rsi']=100-100/(1+ag/al.replace(0,np.nan))
    pc=d.close.shift()
    tr=pd.concat([(d.high-d.low),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d['atr']=tr.ewm(alpha=1/20,adjust=False,min_periods=20).mean()
    d['atr_ma']=d.atr.rolling(20).mean()
    d['vol_ma']=d.volume.rolling(20).median()
    return d

def daily_bias(d):
    x=d.set_index(pd.to_datetime(d.time,unit='s',utc=True))
    q=x.resample('1D').agg({'close':'last'})
    q['ema200']=q.close.ewm(span=200,adjust=False).mean()
    return q.ema200.reindex(x.index,method='ffill').values

def signal(d,i,bias):
    if i<2 or any(pd.isna(d.iloc[i][c]) for c in ['e20','e50','e200','rsi','atr','atr_ma','vol_ma']):
        return None
    r=d.iloc[i]; p=d.iloc[i-1]
    long_ok=(bias[i] < r.close and r.close>r.e50 and
             p.close<=p.e20 and r.close>r.e20 and
             p.rsi<50<=r.rsi and r.atr>=r.atr_ma and r.volume>=r.vol_ma)
    short_ok=(bias[i] > r.close and r.close<r.e50 and
              p.close>=p.e20 and r.close<r.e20 and
              p.rsi>50>=r.rsi and r.atr>=r.atr_ma and r.volume>=r.vol_ma)
    return 'LONG' if long_ok else ('SHORT' if short_ok else None)

def run(d,i,direction,end):
    e=float(d.iloc[i].close); a=float(d.iloc[i].atr); risk=SL_ATR*a
    sl=e-risk if direction=='LONG' else e+risk
    tp=e+TP_R*risk if direction=='LONG' else e-TP_R*risk
    last=min(i+HOLD,end)
    if last<i+HOLD:return None
    for j in range(i+1,last+1):
        x=d.iloc[j]
        sh=x.low<=sl if direction=='LONG' else x.high>=sl
        th=x.high>=tp if direction=='LONG' else x.low<=tp
        if sh:return -1-COST,int(x.time)
        if th:return TP_R-COST,int(x.time)
    return -COST,int(d.iloc[last].time)

print('='*68);print('CANDIDATE 7 — 365D DISCOVERY SCREEN');print('='*68)
cache={}
for s in SYMBOLS:
    d=fetch(s)
    if len(d)>=500:cache[s]=ind(d)
print('VALID SYMBOLS =',len(cache))

res=[]
for s,d in cache.items():
    bias=daily_bias(d)
    for i in range(200,len(d)-HOLD):
        z=signal(d,i,bias)
        if z:
            t=run(d,i,z,len(d)-1)
            if t:res.append((t[0],t[1],s,z))

if res:
    v=np.array([x[0] for x in sorted(res,key=lambda x:(x[1],x[2]))])
    wins=(v>0).sum(); total=v.sum()
    pos=v[v>0].sum(); neg=abs(v[v<0].sum())
    pf=pos/neg if neg else 0
    print('EVENTS/TRADES =',len(v))
    print(f'WR = {wins/len(v):.4f}')
    print(f'NetExp = {v.mean():+.4f}R')
    print(f'NetPF = {pf:.3f}')
    print(f'NetTotalR = {total:+.2f}R')
else:
    print('NO VALID EVENTS')

print('='*68)
