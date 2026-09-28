# CANDIDATE 8 — VOLUME BREAKOUT + RETEST SCREEN
import time,json,urllib.request
import numpy as np,pandas as pd

S=['BTC-USDT','ETH-USDT','SOL-USDT','BNB-USDT','XRP-USDT','DOGE-USDT','ADA-USDT',
'LINK-USDT','AVAX-USDT','DOT-USDT','LTC-USDT','BCH-USDT','UNI-USDT','AAVE-USDT',
'ATOM-USDT','NEAR-USDT','FIL-USDT','ETC-USDT','ICP-USDT','APT-USDT','ARB-USDT',
'OP-USDT','SUI-USDT','SEI-USDT','INJ-USDT','TIA-USDT','JUP-USDT','PEPE-USDT',
'SHIB-USDT','TRX-USDT','HBAR-USDT','VET-USDT','ALGO-USDT','STX-USDT','RUNE-USDT']

INT=14400; DAYS=365; PAGE=1500; RETEST=5; HOLD=30
SLATR=1.25; TPR=2.; COST=.003

def fetch(s):
    now=int(time.time());cur=now-DAYS*86400;out=[]
    while cur<now:
        end=min(cur+(PAGE-1)*INT,now)
        u=f"https://api.kucoin.com/api/v1/market/candles?symbol={s}&type=4hour&startAt={cur}&endAt={end}"
        for k in range(3):
            try:
                q=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(q,timeout=20) as r:x=json.loads(r.read().decode())
                if x.get('code')=='200000':out+=x['data'];break
            except Exception:
                if k==2:return pd.DataFrame()
                time.sleep(1+k)
        cur=end+INT
    if not out:return pd.DataFrame()
    d=pd.DataFrame(out,columns=['time','open','close','high','low','volume','turnover'])
    d['time']=pd.to_numeric(d.time)
    for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    return d[d.time<(now//INT)*INT].reset_index(drop=True)

def prep(d):
    d=d.copy()
    d['ema200']=d.close.ewm(span=200,adjust=False).mean()
    d['atr']=pd.concat([(d.high-d.low),
        (d.high-d.close.shift()).abs(),
        (d.low-d.close.shift()).abs()],axis=1).max(axis=1).ewm(
        alpha=1/20,adjust=False,min_periods=20).mean()
    d['vmed']=d.volume.rolling(20).median()
    d['atrmed']=d.atr.rolling(20).median()
    d['hi20']=d.high.shift(1).rolling(20).max()
    d['lo20']=d.low.shift(1).rolling(20).min()
    return d

def signals(d):
    out=[]
    for i in range(220,len(d)-RETEST):
        r=d.iloc[i]
        if pd.isna(r.hi20) or pd.isna(r.lo20) or pd.isna(r.atrmed):continue

        # Breakout candle: causal, based only on prior 20 bars
        if r.close>r.hi20 and r.volume>=1.5*r.vmed and r.atr>=r.atrmed and r.close>r.ema200:
            level=r.hi20
            for j in range(i+1,min(i+1+RETEST,len(d))):
                x=d.iloc[j]
                if x.low<=level and x.close>level:
                    out.append((j,'LONG',level))
                    break

        elif r.close<r.lo20 and r.volume>=1.5*r.vmed and r.atr>=r.atrmed and r.close<r.ema200:
            level=r.lo20
            for j in range(i+1,min(i+1+RETEST,len(d))):
                x=d.iloc[j]
                if x.high>=level and x.close<level:
                    out.append((j,'SHORT',level))
                    break
    return out

def trade(d,i,side):
    e=float(d.iloc[i].close);a=float(d.iloc[i].atr);risk=SLATR*a
    sl=e-risk if side=='LONG' else e+risk
    tp=e+TPR*risk if side=='LONG' else e-TPR*risk
    end=min(i+HOLD,len(d)-1)
    if end<i+HOLD:return None
    for j in range(i+1,end+1):
        x=d.iloc[j]
        if side=='LONG':
            if x.low<=sl:return -1-COST
            if x.high>=tp:return TPR-COST
        else:
            if x.high>=sl:return -1-COST
            if x.low<=tp:return TPR-COST
    return -COST

print('='*65)
print('CANDIDATE 8 — VOLUME BREAKOUT + RETEST')
print('='*65)

alltr=[];valid=0
for s in S:
    d=fetch(s)
    if len(d)<500:continue
    valid+=1;d=prep(d)
    for i,side,level in signals(d):
        x=trade(d,i,side)
        if x is not None:alltr.append(x)

print('VALID SYMBOLS =',valid)
print('TRADES =',len(alltr))

if alltr:
    v=np.array(alltr);w=(v>0).sum()
    pos=v[v>0].sum();neg=abs(v[v<0].sum())
    print(f'WR = {w/len(v):.4f}')
    print(f'NetExp = {v.mean():+.4f}R')
    print(f'NetPF = {pos/neg:.3f}' if neg else 'NetPF = INF')
    print(f'NetTotalR = {v.sum():+.2f}R')
else:
    print('NO VALID TRADES')

print('='*65)
