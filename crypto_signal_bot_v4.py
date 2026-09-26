# Setup F — Unified Validation Engine (Corrected, frozen spec)
# Cross-Sectional Momentum Ranking | 4H | Long-only
# Replace crypto_signal_bot_v4.py with this file and run it directly.

import time, requests, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

SYMS = ["ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT","NEAR","APT","ARB","OP","SUI","INJ","TIA","SEI","FIL","ATOM","LTC","ETC","TRX","ICP","AAVE","UNI","MKR","RUNE","FTM","GRT","ALGO","VET","HBAR","EGLD","XLM","THETA","SAND","MANA","AXS","CHZ"]
TARGET_DAYS=730; MIN_HISTORY_DAYS=180
MOM_PERIOD=20; RANKING_LOOKBACK=100; TOP_N=3; HOLD=30
ATR_LOOKBACK=20
FEE_PCT=0.10; SLIPPAGE_PCT=0.05
TP_SCENARIOS=[1,1.5,2,3]
HORIZONS=[1,3,6,12,24]; HIT_LEVELS=[.5,1,1.5,2,3]
MAX_TRACK_BARS=44
URL="https://api.kucoin.com/api/v1/market/candles"
S=requests.Session()

def fetch(sym):
    end=int(time.time()); rows=[]; seen=set()
    target=TARGET_DAYS*6
    while len(rows)<target:
        p={"symbol":sym+"-USDT","type":"4hour","endAt":end}
        try:
            r=S.get(URL,params=p,timeout=20); r.raise_for_status()
            data=r.json().get("data",[])
        except Exception as e:
            print("FETCH_ERROR",sym,e); break
        if not data: break
        before=len(rows)
        for x in data:
            ts=int(x[0])
            if ts not in seen:
                seen.add(ts); rows.append([ts,float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])])
        oldest=min(int(x[0]) for x in data)
        end=oldest-14400
        if len(rows)==before: break
        time.sleep(.08)
    rows.sort()
    now=time.time()
    if rows and now < rows[-1][0]+14400: rows.pop()
    return rows

def prep(rows):
    if not rows: return None
    ts=[x[0] for x in rows]
    dup=len(ts)-len(set(ts))
    rows=sorted({x[0]:x for x in rows}.values(), key=lambda x:x[0])
    mono=all(rows[i][0]>rows[i-1][0] for i in range(1,len(rows)))
    gaps=sum(1 for i in range(1,len(rows)) if rows[i][0]-rows[i-1][0]!=14400)
    cov=(rows[-1][0]-rows[0][0])/86400 if len(rows)>1 else 0
    atr=[]
    for i in range(len(rows)):
        atr.append(sum(x[2]-x[3] for x in rows[max(0,i-ATR_LOOKBACK+1):i+1])/min(i+1,ATR_LOOKBACK))
    return {"r":rows,"dup":dup,"mono":mono,"gaps":gaps,"days":cov,"atr":atr}

def ret(df,i,n): return (df["r"][i][4]-df["r"][i-n][4])/df["r"][i-n][4]

def detect(dfs):
    common=set.intersection(*(set(x["map"]) for x in dfs.values()))
    times=sorted(common); scan=times[-RANKING_LOOKBACK:]
    pos={t:i for i,t in enumerate(times)}
    events=[]; prev=set()
    for t in scan:
        i=pos[t]
        if i<MOM_PERIOD: continue
        ranks=[]
        for sym,d in dfs.items():
            if t not in d["map"] or times[i-MOM_PERIOD] not in d["map"]: continue
            j=d["map"][t]; k=d["map"][times[i-MOM_PERIOD]]
            ranks.append((ret(d,j,j-k),sym,j))
        # fixed frozen rule: first scanned timestamp starts with an empty prev_top.
        ranks.sort(reverse=True)
        top={x[1] for x in ranks[:TOP_N]}
        for sym in sorted(top-prev):
            d=dfs[sym]; j=d["map"][t]
            events.append({"sym":sym,"idx":j,"time":t})
        prev=top
    return events,times

def layer_b(events,dfs):
    out=[]; hits=defaultdict(list)
    for e in events:
        d=dfs[e["sym"]]; i=e["idx"]; rows=d["r"]
        if i>=len(rows) or d["atr"][i]<=0: continue
        entry=rows[i][4]; a=d["atr"][i]
        maxb=min(MAX_TRACK_BARS,len(rows)-1-i)
        mfe=-1e99; mae=1e99; tm=ta=0
        fwd={}
        for h in HORIZONS:
            if i+h<len(rows): fwd[h]=rows[i+h][4]/entry-1
        for n in range(1,maxb+1):
            hi,lo=rows[i+n][2],rows[i+n][3]
            up=(hi-entry)/a; dn=(entry-lo)/a
            if up>mfe: mfe=up; tm=n
            if dn>mae: mae=dn; ta=n
        hit={}
        for h in HORIZONS:
            for lv in HIT_LEVELS:
                hit[(h,lv)] = any((rows[i+k][2]-entry)/a>=lv for k in range(1,min(h,maxb)+1))
        out.append({"event":e,"entry":entry,"atr":a,"mfe":mfe,"mae":mae,"tmfe":tm,"tmae":ta,"fwd":fwd,"hit":hit})
    return out

def simulate(m,dfs,tp):
    e=m["event"]; d=dfs[e["sym"]]; rows=d["r"]; i=e["idx"]
    entry=m["entry"]; a=m["atr"]; sl=entry-a; target=entry+a*tp
    cost=(FEE_PCT+SLIPPAGE_PCT)/100/(a/entry)
    full=(i+HOLD)<len(rows)
    if not full: return None,"OPEN_AT_DATASET_END",0,0,False
    for k in range(i+1,i+HOLD+1):
        hi,lo=rows[k][2],rows[k][3]
        hit_sl=lo<=sl; hit_tp=hi>=target
        if hit_sl: return -1,"SL",(-1-cost),cost,hit_tp
        if hit_tp: return tp,"TP",(tp-cost),cost,False
    return 0,"TIMEOUT",-cost,cost,False

def main():
    print("SETUP F — UNIFIED VALIDATION ENGINE — CORRECTED")
    print("Spec: 4H | 40 symbols | MOM=20 | lookback=100 | TOP_N=3 | new-entry only | Long-only | HOLD=30")
    raw={}; dfs={}; audits={}
    for sym in SYMS:
        rows=fetch(sym); raw[sym]=rows
        d=prep(rows)
        if not d: audits[sym]={"status":"FETCH_FAIL"}; continue
        audits[sym]={k:d[k] for k in ["dup","mono","gaps","days"]}
        if d["days"]>=MIN_HISTORY_DAYS:
            d["map"]={x[0]:i for i,x in enumerate(d["r"])}; dfs[sym]=d
        else: audits[sym]["status"]="INSUFFICIENT_HISTORY"
    print("DATA",len(raw),"loaded;",len(dfs),"sufficient")
    print("AUDIT", "bad_gap",sum(a.get("gaps",0)>0 for a in audits.values()),
          "duplicates",sum(a.get("dup",0)>0 for a in audits.values()))
    events,times=detect(dfs)
    print("COMMON_TIMESTAMPS",len(times),"SCAN",min(times) if times else None,"->",max(times) if times else None)
    print("DETECTED_EVENTS",len(events))
    meas=layer_b(events,dfs)
    print("LAYER_B_MEASURED",len(meas),"rejected",len(events)-len(meas))
    if meas:
        print("MFE_MEAN %.3f MAE_MEAN %.3f MFE_MED %.3f MAE_MED %.3f" %
              (statistics.mean(x["mfe"] for x in meas),statistics.mean(x["mae"] for x in meas),
               statistics.median(x["mfe"] for x in meas),statistics.median(x["mae"] for x in meas)))
        print("TIME_TO_MFE_MED",statistics.median(x["tmfe"] for x in meas),
              "TIME_TO_MAE_MED",statistics.median(x["tmae"] for x in meas))
        for h in HORIZONS:
            print("FWD_%s_MEAN_PCT %.4f"%(h,100*statistics.mean(x["fwd"][h] for x in meas if h in x["fwd"])))
            print("HIT_%s"%h,{lv:round(100*sum(x["hit"][(h,lv)] for x in meas)/len(meas),2) for lv in HIT_LEVELS})
    for tp in TP_SCENARIOS:
        alltr=[]; symm=defaultdict(lambda:Counter()); amb=0; openend=0
        for m in meas:
            r,status,net,cost,isamb=simulate(m,dfs,tp)
            sym=m["event"]["sym"]; symm[sym][status]+=1
            if status=="OPEN_AT_DATASET_END": openend+=1; continue
            alltr.append((r,net,status,isamb)); amb+=int(isamb)
        wins=sum(x[2]=="TP" for x in alltr); losses=sum(x[2]=="SL" for x in alltr); tout=sum(x[2]=="TIMEOUT" for x in alltr)
        gross=sum(x[0] for x in alltr); net=sum(x[1] for x in alltr)
        gp=sum(x[0] for x in alltr if x[0]>0); gl=-sum(x[0] for x in alltr if x[0]<0)
        np_=sum(x[1] for x in alltr if x[1]>0); nl=-sum(x[1] for x in alltr if x[1]<0)
        pf=gp/gl if gl else float("inf"); npf=np_/nl if nl else float("inf")
        vals=[]; peak=0; dd=0; npeak=0
        for x in alltr:
            npeak+=x[0]; peak=max(peak,npeak); dd=min(dd,npeak-peak)
        n=len(alltr)
        print("\nTP",tp,"R | TRADED",n,"TP",wins,"SL",losses,"TIMEOUT",tout,"OPEN_AT_END",openend,"AMBIGUOUS",amb)
        print("GROSS_EXP %.4f NET_EXP %.4f GROSS_TOTAL %.2f NET_TOTAL %.2f PF %.3f NET_PF %.3f MAXDD %.2f" %
              (gross/n if n else 0,net/n if n else 0,gross,net,pf,npf,dd))
        print("WR %.2f%%"% (100*wins/n if n else 0))
        print("PER_SYMBOL")
        for sym in sorted(symm):
            c=symm[sym]; print(sym,dict(c))
    print("\nDONE — send the complete Test RUN output back for independent review.")

if __name__=="__main__": main()
