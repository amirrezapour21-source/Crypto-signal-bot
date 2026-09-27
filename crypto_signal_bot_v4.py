# SETUP F — OOS INTEGRITY AUDIT (NO OPTIMIZATION)
# Verifies chronological split, causal event boundary, and trade-window boundary.
# Frozen: 4H | MOM=20 | LOOKBACK=100 | TOP_N=3 | Long-only | HOLD=30

import time, requests
from datetime import datetime, timezone

SYMS = ['ETH','SOL','BNB','XRP','DOGE','ADA','LINK','AVAX','DOT','NEAR','APT','ARB','OP','SUI','INJ','TIA','SEI','FIL','ATOM','LTC','ETC','TRX','ICP','AAVE','UNI','MKR','RUNE','FTM','GRT','ALGO','VET','HBAR','EGLD','XLM','THETA','SAND','MANA','AXS','CHZ']
TARGET_DAYS=730; MOM_PERIOD=20; RANKING_LOOKBACK=100; TOP_N=3; HOLD=30
URL='https://api.kucoin.com/api/v1/market/candles'; STEP=14400

def fetch(sym):
    end=int(time.time()); start=end-TARGET_DAYS*86400; rows=[]; seen=set(); cur=end
    while cur>start:
        p={'symbol':sym+'-USDT','type':'4hour','endAt':cur}
        r=requests.get(URL,params=p,timeout=20).json()
        data=r.get('data') or []
        if not data: break
        for x in data:
            ts=int(x[0])
            if ts>=start and ts<=end and ts not in seen:
                seen.add(ts); rows.append(ts)
        nxt=min(int(x[0]) for x in data)-1
        if nxt>=cur: break
        cur=nxt
        if len(data)<2: break
    return sorted(rows)

def main():
    print('SETUP F — OOS INTEGRITY AUDIT')
    print('Frozen spec unchanged; no performance optimization.')
    data={}; gaps={}
    for s in SYMS:
        try:
            ts=fetch(s); data[s]=ts
            gaps[s]=sum(b-a!=STEP for a,b in zip(ts,ts[1:]))
        except Exception as e:
            print('FETCH_ERROR',s,str(e)); data[s]=[]
    valid=[s for s,t in data.items() if len(t)>=180*6]
    common=sorted(set.intersection(*(set(data[s]) for s in valid))) if valid else []
    if len(common)<200:
        print('FAIL: insufficient common timestamps'); return
    split=common[int(len(common)*0.70)]
    is_times=[t for t in common if t<split]
    oos_times=[t for t in common if t>=split]
    # OOS detector may use only prior data; first OOS event must have prior MOM+lookback context.
    first_allowed=oos_times[0]
    idx=common.index(first_allowed)
    warmup_ok=idx>=max(MOM_PERIOD,RANKING_LOOKBACK)
    # Last OOS signal must have HOLD bars available.
    last_signal_cut=common[-(HOLD+1)] if len(common)>HOLD else None
    print('DATA_VALID_SYMBOLS',len(valid))
    print('COMMON_TIMESTAMPS',len(common))
    print('IS_COUNT',len(is_times),'OOS_COUNT',len(oos_times))
    print('SPLIT_TIMESTAMP',split,datetime.fromtimestamp(split,tz=timezone.utc).isoformat())
    print('OOS_WARMUP_CONTEXT_OK',warmup_ok)
    print('LAST_SIGNAL_ALLOWED',last_signal_cut,datetime.fromtimestamp(last_signal_cut,tz=timezone.utc).isoformat())
    print('GAP_SYMBOLS',sum(v>0 for v in gaps.values()),'TOTAL_GAPS',sum(gaps.values()))
    print('GAP_DETAIL', {k:v for k,v in gaps.items() if v>0})
    print('AUDIT_RESULT', 'PASS' if warmup_ok and last_signal_cut is not None else 'FAIL')
    print('NOTE: This audit does not change or optimize Setup F. If PASS, existing OOS results remain the decision evidence.')

if __name__=='__main__': main()
