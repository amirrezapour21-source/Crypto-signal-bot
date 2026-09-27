# SETUP F — OOS VALIDATION
# Frozen detector/trade rules. Chronological 70/30 split; no parameter optimization.
import time, requests, statistics
from collections import Counter, defaultdict

SYMS=["ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT","NEAR","APT","ARB","OP","SUI","INJ","TIA","SEI","FIL","ATOM","LTC","ETC","TRX","ICP","AAVE","UNI","MKR","RUNE","FTM","GRT","ALGO","VET","HBAR","EGLD","XLM","THETA","SAND","MANA","AXS","CHZ"]
TARGET_DAYS=730; MIN_HISTORY_DAYS=180
MOM_PERIOD=20; RANKING_LOOKBACK=100; TOP_N=3; HOLD=30; ATR_LOOKBACK=20
FEE_PCT=.10; SLIPPAGE_PCT=.05
TP_SCENARIOS=[1,1.5,2,3]; HORIZONS=[1,3,6,12,24]; HIT_LEVELS=[.5,1,1.5,2,3]; MAX_TRACK_BARS=44
URL="https://api.kucoin.com/api/v1/market/candles"; S=requests.Session()

def fetch(sym):
    end=int(time.time()); rows=[]; seen=set(); target=TARGET_DAYS*6
    while len(rows)<target:
        try:
            r=S.get(URL,params={"symbol":sym+"-USDT","type":"4hour","endAt":end},timeout=20); r.raise_for_status(); data=r.json().get("data",[])
        except Exception as e: print("FETCH_ERROR",sym,e); break
        if not data: break
        before=len(rows)
        for x in data:
            ts=int(x[0])
            if ts not in seen:
                seen.add(ts); rows.append([ts,float(x[1]),float(x[3]),float(x[4]),float(x[2]),float(x[5])])
        oldest=min(int(x[0]) for x in data); end=oldest-14400
        if len(rows)==before: break
        time.sleep(.08)
    rows.sort()
    if rows and time.time()<rows[-1][0]+14400: rows.pop()
    return rows

def prep(rows):
    if not rows:return None
    ts=[x[0] for x in rows]; dup=len(ts)-len(set(ts)); rows=sorted({x[0]:x for x in rows}.values(),key=lambda x:x[0])
    gaps=sum(1 for i in range(1,len(rows)) if rows[i][0]-rows[i-1][0]!=14400)
    days=(rows[-1][0]-rows[0][0])/86400 if len(rows)>1 else 0
    atr=[]
    for i in range(len(rows)):
        n=min(i+1,ATR_LOOKBACK); atr.append(sum(x[2]-x[3] for x in rows[i-n+1:i+1])/n)
    return {"r":rows,"dup":dup,"gaps":gaps,"days":days,"atr":atr,"map":{x[0]:i for i,x in enumerate(rows)}}

def detect_oos(dfs,times):
    # 70/30 chronological split. OOS is the final 30%; last HOLD bars are excluded.
    cut=int(len(times)*.70); oos=times[cut:-HOLD] if len(times)>cut+HOLD else []
    pos={t:i for i,t in enumerate(times)}; events=[]; prev=None
    # Warm-up top state from the immediately preceding timestamp so the first OOS event is causal.
    if oos:
        first=pos[oos[0]]
        if first>0:
            t0=times[first-1]; ranks=[]
            if first-1>=MOM_PERIOD:
                for sym,d in dfs.items():
                    if t0 in d["map"] and times[first-1-MOM_PERIOD] in d["map"]:
                        j=d["map"][t0]; k=d["map"][times[first-1-MOM_PERIOD]]
                        ranks.append(((d["r"][j][4]-d["r"][k][4])/d["r"][k][4],sym))
                ranks.sort(reverse=True); prev={x[1] for x in ranks[:TOP_N]}
        else: prev=set()
    for t in oos:
        i=pos[t]
        if i<MOM_PERIOD: continue
        ranks=[]
        for sym,d in dfs.items():
            if t in d["map"] and times[i-MOM_PERIOD] in d["map"]:
                j=d["map"][t]; k=d["map"][times[i-MOM_PERIOD]]
                ranks.append(((d["r"][j][4]-d["r"][k][4])/d["r"][k][4],sym,j))
        ranks.sort(reverse=True); top={x[1] for x in ranks[:TOP_N]}
        for sym in sorted(top-(prev or set())):
            events.append({"sym":sym,"idx":dfs[sym]["map"][t],"time":t})
        prev=top
    return events,cut,oos

def layer_b(events,dfs):
    out=[]; rej=Counter()
    for e in events:
        d=dfs.get(e["sym"]); i=e["idx"]
        if d is None: rej["symbol_not_in_dfs"]+=1; continue
        a=d["atr"][i]; rows=d["r"]
        if not a>0: rej["invalid_atr"]+=1; continue
        entry=rows[i][4]
        maxb=min(MAX_TRACK_BARS,len(rows)-1-i)
        if maxb<1: rej["no_forward_data"]+=1; continue
        mfe=mae=0; tm=ta=0; fwd={}
        for h in HORIZONS:
            if i+h<len(rows): fwd[h]=rows[i+h][4]/entry-1
        for n in range(1,maxb+1):
            up=(rows[i+n][2]-entry)/a; dn=(entry-rows[i+n][3])/a
            if up>mfe:mfe=up;tm=n
            if dn>mae:mae=dn;ta=n
        hit={(h,l):any((rows[i+k][2]-entry)/a>=l for k in range(1,min(h,maxb)+1)) for h in HORIZONS for l in HIT_LEVELS}
        out.append({"event":e,"entry":entry,"atr":a,"mfe":mfe,"mae":mae,"tmfe":tm,"tmae":ta,"fwd":fwd,"hit":hit})
    return out,rej

def sim(m,dfs,tp):
    d=dfs[m["event"]["sym"]]; rows=d["r"]; i=m["event"]["idx"]; entry=m["entry"]; a=m["atr"]; sl=entry-a; target=entry+a*tp
    cost=(FEE_PCT+SLIPPAGE_PCT)/100/(a/entry)
    if i+HOLD>=len(rows): return None,"OPEN_AT_DATASET_END",0,False
    for k in range(i+1,i+HOLD+1):
        hi,lo=rows[k][2],rows[k][3]; hs=lo<=sl; ht=hi>=target
        if hs:return -1,"SL",-1-cost,hs and ht
        if ht:return tp,"TP",tp-cost,False
    return 0,"TIMEOUT",-cost,False

def main():
    print("SETUP F — OOS VALIDATION — FROZEN SPEC")
    print("Chronological split 70/30 | 4H | MOM=20 | TOP_N=3 | Long-only | HOLD=30 | no optimization")
    dfs={}; audits={}; loaded=0
    for sym in SYMS:
        d=prep(fetch(sym))
        if d:
            loaded+=1; audits[sym]=d
            if d["days"]>=MIN_HISTORY_DAYS: dfs[sym]=d
    print("DATA",loaded,"loaded;",len(dfs),"sufficient")
    print("AUDIT bad_gap",sum(d["gaps"]>0 for d in audits.values()),"duplicates",sum(d["dup"]>0 for d in audits.values()))
    common=sorted(set.intersection(*(set(d["map"]) for d in dfs.values())))
    events,cut,oos=detect_oos(dfs,common)
    print("COMMON_TIMESTAMPS",len(common),"IS_END",common[cut-1] if cut else None,"OOS_START",oos[0] if oos else None,"OOS_END",oos[-1] if oos else None)
    print("OOS_EVENTS",len(events))
    meas,rej=layer_b(events,dfs); print("LAYER_B_MEASURED",len(meas),"rejected",len(events)-len(meas))
    if rej:print("REJECT_REASONS",dict(rej))
    if not meas: print("NO_VALID_OOS_MEASUREMENTS"); return
    print("MFE_MEAN %.3f MAE_MEAN %.3f MFE_MED %.3f MAE_MED %.3f"%(statistics.mean(x["mfe"] for x in meas),statistics.mean(x["mae"] for x in meas),statistics.median(x["mfe"] for x in meas),statistics.median(x["mae"] for x in meas)))
    for tp in TP_SCENARIOS:
        tr=[]; symm=defaultdict(Counter)
        for m in meas:
            r,s,n,a=sim(m,dfs,tp); sym=m["event"]["sym"]; symm[sym][s]+=1
            if s!="OPEN_AT_DATASET_END": tr.append((r,n,s,a))
        w=sum(x[2]=="TP" for x in tr); l=sum(x[2]=="SL" for x in tr); to=sum(x[2]=="TIMEOUT" for x in tr); gross=sum(x[0] for x in tr); net=sum(x[1] for x in tr)
        gp=sum(x[0] for x in tr if x[0]>0); gl=-sum(x[0] for x in tr if x[0]<0); np_=sum(x[1] for x in tr if x[1]>0); nl=-sum(x[1] for x in tr if x[1]<0)
        peak=0;dd=0;eq=0
        for x in tr:eq+=x[0];peak=max(peak,eq);dd=min(dd,eq-peak)
        n=len(tr)
        print("\nTP",tp,"R | TRADED",n,"TP",w,"SL",l,"TIMEOUT",to,"OPEN_AT_END",sum(x[2]=="OPEN_AT_DATASET_END" for x in tr),"AMBIGUOUS",sum(x[3] for x in tr))
        print("GROSS_EXP %.4f NET_EXP %.4f GROSS_TOTAL %.2f NET_TOTAL %.2f PF %.3f NET_PF %.3f MAXDD %.2f"%(gross/n,net/n,gross,net,gp/gl if gl else float('inf'),np_/nl if nl else float('inf'),dd))
        print("WR %.2f%%"%(100*w/n))
        print("PER_SYMBOL")
        for sym in sorted(symm):print(sym,dict(symm[sym]))
    print("\nDONE — send complete OOS RUN output for review")

if __name__=="__main__":main()
