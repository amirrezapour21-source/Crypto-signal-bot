import time, requests, math
import pandas as pd
import numpy as np

# SETUP V4 — CANDIDATE 4 — CAUSAL DONCHIAN VOLATILITY BREAKOUT
# Research only. Frozen parameters; no optimization.
# 4H entry: close breaks prior 20-bar high/low + ATR expansion + volume confirmation.
# HTF protection: daily 20-bar EMA slope must agree with trade direction.
# Trade: entry close, SL=1 ATR, TP=1/1.5/2/3R, HOLD=30, SL-first.

SYMS=['ETH','SOL','BNB','XRP','DOGE','ADA','LINK','AVAX','DOT','NEAR','APT','ARB','OP','SUI','INJ','TIA','SEI','FIL','ATOM','LTC','ETC','TRX','ICP','AAVE','UNI','MKR','RUNE','FTM','GRT','ALGO','VET','HBAR','EGLD','XLM','THETA','SAND','MANA','AXS','CHZ']
BASE='https://api.kucoin.com/api/v1/market/candles'
TARGET_DAYS=730; MIN_DAYS=180
BREAKOUT=20; ATR_N=20; VOL_N=20; HOLD=30
FEE=0.10; SLIPPAGE=0.05
TP_SCENARIOS=[1,1.5,2,3]

S=requests.Session(); S.headers.update({'User-Agent':'V4-Candidate4-Validation/1.0'})

def fetch(sym):
    end=int(time.time()); rows=[]
    while True:
        p={'symbol':sym+'-USDT','type':'4hour','endAt':end}
        r=S.get(BASE,params=p,timeout=20); r.raise_for_status(); data=r.json().get('data',[])
        if not data: break
        rows += data
        mn=min(int(x[0]) for x in data)
        if len(rows)>=TARGET_DAYS*6 or mn>=end: break
        end=mn-1; time.sleep(.05)
    if not rows: return None
    # KuCoin candle order: time, open, close, high, low, volume, turnover
    df=pd.DataFrame(rows,columns=['ts','open','close','high','low','volume','turnover'])
    for c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    now=int(time.time())
    if len(df) and now < int(df.iloc[-1].ts)+14400: df=df.iloc[:-1]
    df['atr']=(df.high-df.low).rolling(ATR_N).mean()
    df['vol_ma']=df.volume.rolling(VOL_N).mean()
    df['prior_hi']=df.high.shift(1).rolling(BREAKOUT).max()
    df['prior_lo']=df.low.shift(1).rolling(BREAKOUT).min()
    days=(df.ts.iloc[-1]-df.ts.iloc[0])/86400 if len(df)>1 else 0
    return df,days

def daily_bias(sym):
    end=int(time.time()); rows=[]
    while len(rows)<220:
        p={'symbol':sym+'-USDT','type':'1day','endAt':end}
        r=S.get(BASE,params=p,timeout=20); r.raise_for_status(); d=r.json().get('data',[])
        if not d: break
        rows+=d; mn=min(int(x[0]) for x in d); end=mn-1; time.sleep(.03)
    if not rows:return None
    df=pd.DataFrame(rows,columns=['ts','open','close','high','low','volume','turnover'])
    for c in df.columns:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    if len(df) and int(time.time())<int(df.iloc[-1].ts)+86400:df=df.iloc[:-1]
    df['ema']=df.close.ewm(span=20,adjust=False).mean()
    return df[['ts','ema']]

def bias_at(daily,ts):
    x=daily[daily.ts<ts]
    if len(x)<2:return 0
    return 1 if x.ema.iloc[-1]>x.ema.iloc[-2] else -1

def simulate(df,idx,side,tp_mult):
    entry=float(df.close.iloc[idx]); atr=float(df.atr.iloc[idx]); sl=entry-side*atr; tp=entry+side*atr*tp_mult
    cost=(FEE+SLIPPAGE)/100/(atr/entry)
    last=idx+HOLD
    if last>=len(df):return ('OPEN_AT_DATASET_END',None,None)
    for j in range(idx+1,last+1):
        hi=float(df.high.iloc[j]);lo=float(df.low.iloc[j])
        hit_sl=(lo<=sl) if side==1 else (hi>=sl)
        hit_tp=(hi>=tp) if side==1 else (lo<=tp)
        if hit_sl and hit_tp:return ('SL',-1,-1-cost)
        if hit_sl:return ('SL',-1,-1-cost)
        if hit_tp:return ('TP',tp_mult,tp_mult-cost)
    return ('TIMEOUT',0,-cost)

def main():
    print('SETUP V4 — CANDIDATE 4 — DONCHIAN VOLATILITY BREAKOUT')
    print('Frozen: 4H | breakout=20 | ATR=20 | volume=20 | daily EMA20 bias | HOLD=30 | Long/Short')
    data={}; audits={}
    for s in SYMS:
        try:
            z=fetch(s)
            if z:data[s],audits[s]=z
        except Exception as e: print('FETCH_ERROR',s,str(e)[:100])
    usable={s:d for s,(d,days) in data.items() if days>=MIN_DAYS}
    print('DATA',len(data),'loaded;',len(usable),'sufficient')
    events=[]
    for s,df in usable.items():
        daily=daily_bias(s)
        if daily is None: continue
        # keep only events with full future HOLD available
        for i in range(max(BREAKOUT,ATR_N,VOL_N),len(df)-HOLD):
            c=float(df.close.iloc[i]); atr=float(df.atr.iloc[i]); vm=float(df.vol_ma.iloc[i]); v=float(df.volume.iloc[i])
            if not np.isfinite(atr) or atr<=0 or not np.isfinite(vm) or vm<=0: continue
            b=bias_at(daily,int(df.ts.iloc[i]));
            long_ok=c>float(df.prior_hi.iloc[i]) and (c-float(df.open.iloc[i]))>atr and v>vm and b==1
            short_ok=c<float(df.prior_lo.iloc[i]) and (float(df.open.iloc[i])-c)>atr and v>vm and b==-1
            if long_ok: events.append((s,i,1))
            elif short_ok: events.append((s,i,-1))
    print('EVENTS',len(events))
    for tp in TP_SCENARIOS:
        outcomes=[]
        for s,i,side in events:
            outcomes.append((s,)+simulate(usable[s],i,side,tp))
        closed=[x for x in outcomes if x[1]!='OPEN_AT_DATASET_END']
        gross=[x[2] for x in closed]; net=[x[3] for x in closed]
        wins=sum(x>0 for x in gross); losses=sum(x<0 for x in gross)
        gp=sum(x for x in gross if x>0); gl=-sum(x for x in gross if x<0)
        ngp=sum(x for x in net if x>0); ngl=-sum(x for x in net if x<0)
        eq=np.cumsum(net); dd=eq-np.maximum.accumulate(eq); maxdd=float(dd.min()) if len(dd) else 0
        print(f'\nTP {tp}R | TRADED {len(closed)} TP {sum(x[1]=="TP" for x in closed)} SL {sum(x[1]=="SL" for x in closed)} TIMEOUT {sum(x[1]=="TIMEOUT" for x in closed)} OPEN_AT_END {len(outcomes)-len(closed)}')
        print(f'GROSS_EXP {np.mean(gross) if gross else 0:.4f} NET_EXP {np.mean(net) if net else 0:.4f} GROSS_TOTAL {sum(gross):.2f} NET_TOTAL {sum(net):.2f} PF {gp/gl if gl else float("inf"):.3f} NET_PF {ngp/ngl if ngl else float("inf"):.3f} MAXDD {maxdd:.2f} WR {wins/len(closed)*100 if closed else 0:.2f}%')
    print('\nDONE — Candidate 4 baseline IS run; no optimization.')

if __name__=='__main__':main()
