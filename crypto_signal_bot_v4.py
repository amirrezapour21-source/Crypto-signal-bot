"""Setup D: Fresh BOS -> Retest — Cycle 2 Edge Discovery"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
"ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
"COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT","GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
"1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT","ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
"ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT","OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT"]
LB, RETEST_TOL_ATR, RETEST_MAX_BARS, HOLD, FEE = 120, 0.3, 5, 30, 0.10

def g(sym, end_at=None):
    r = requests.get(BASE, params={"symbol":sym,"type":"4hour",**({"endAt":int(end_at)} if end_at else {})}, timeout=20)
    if r.status_code != 200: return None
    d = r.json()
    if d.get("code") != "200000" or not d.get("data"): return None
    rows = [{"time":int(x[0]),"open":float(x[1]),"close":float(x[2]),"high":float(x[3]),"low":float(x[4]),"volume":float(x[5])} for x in d["data"]]
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

def get_df(sym, n=250):
    dfs, end_at, rem = [], None, n
    while rem > 0:
        d = g(sym, end_at)
        if d is None or d.empty: break
        dfs.append(d); rem -= len(d); end_at = int(d["time"].min())-1
        time.sleep(0.12)
        if len(d) < 100: break
    if not dfs: return None
    f = pd.concat(dfs).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    f["dt"] = pd.to_datetime(f["time"], unit="s", utc=True)
    if int(time.time()) < int(f["time"].iloc[-1])+14400: f = f.iloc[:-1]
    return f.tail(n).reset_index(drop=True)

def swings(df, l=3, r=3):
    h, lo = df["high"].values, df["low"].values
    sh, sl = [], []
    for i in range(l, len(df)-r):
        if h[i]==h[i-l:i+r+1].max() and (h[i-l:i+r+1]==h[i]).sum()==1: sh.append({"index":i,"price":h[i]})
        if lo[i]==lo[i-l:i+r+1].min() and (lo[i-l:i+r+1]==lo[i]).sum()==1: sl.append({"index":i,"price":lo[i]})
    return sh, sl

def fresh_bos(df, sh, sl, lb=LB):
    ev, lh, ll = [], None, None
    for i in range(max(0,len(df)-lb), len(df)):
        c = df["close"].iloc[i]
        rh = [s for s in sh if s["index"]<i]
        if rh and c>rh[-1]["price"] and rh[-1]["price"]!=lh:
            ev.append({"dir":"bullish","idx":i,"level":rh[-1]["price"]}); lh=rh[-1]["price"]
        rl = [s for s in sl if s["index"]<i]
        if rl and c<rl[-1]["price"] and rl[-1]["price"]!=ll:
            ev.append({"dir":"bearish","idx":i,"level":rl[-1]["price"]}); ll=rl[-1]["price"]
    return ev

def find_retest(df, bos, atr_s):
    i, lvl, d = bos["idx"], bos["level"], bos["dir"]
    atr = atr_s.iloc[i]
    if pd.isna(atr) or atr==0: return None
    tol = RETEST_TOL_ATR*atr
    for j in range(i+1, min(i+RETEST_MAX_BARS+1, len(df))):
        row = df.iloc[j]
        near = (d=="bullish" and row["low"]<=lvl+tol) or (d=="bearish" and row["high"]>=lvl-tol)
        if near:
            for k in range(j, min(j+2, len(df))):
                c = df["close"].iloc[k]
                if (d=="bullish" and c>lvl) or (d=="bearish" and c<lvl):
                    return {"retest_idx":j, "confirm_idx":k}
            return None
    return None

def excursion(df, direction, entry, atr, idx, hold=HOLD):
    end = min(idx+1+hold, len(df))
    mfe=mae=0; pl=[0.5,1,1.5,2,3]; nl=[0.5,0.75,1,1.25,1.5]
    rp={v:None for v in pl}; rn={v:None for v in nl}
    for step,i in enumerate(range(idx+1,end),1):
        row=df.iloc[i]
        fav,adv=(row["high"]-entry,entry-row["low"]) if direction=="bullish" else (entry-row["low"],row["high"]-entry)
        mfe,mae=max(mfe,fav),max(mae,adv)
        fa,aa=fav/atr,adv/atr
        for v in pl:
            if rp[v] is None and fa>=v: rp[v]=step
        for v in nl:
            if rn[v] is None and aa>=v: rn[v]=step
    return {"mfe":mfe/atr,"mae":mae/atr,"rp":rp,"rn":rn}

def sim(df, direction, entry, sl_atr, tp_mult, atr, idx, hold=HOLD):
    if direction=="bullish": sl=entry-sl_atr*atr; tp=entry+sl_atr*atr*tp_mult
    else: sl=entry+sl_atr*atr; tp=entry-sl_atr*atr*tp_mult
    end=min(idx+1+hold,len(df))
    for i in range(idx+1,end):
        row=df.iloc[i]
        sh_,th_=(row["low"]<=sl,row["high"]>=tp) if direction=="bullish" else (row["high"]>=sl,row["low"]<=tp)
        if sh_: return -1.0
        if th_: return tp_mult
    return None

def pct(v,p):
    if not v: return None
    s=sorted(v); k=(len(s)-1)*p; f=int(k); c=f+1 if f+1<len(s) else f
    return s[f] if f==c else s[f]+(s[c]-s[f])*(k-f)

if __name__ == "__main__":
    print("SETUP D CYCLE 2 — Edge Discovery")
    entries = []
    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80: continue
        sh, sl = swings(df)
        ar = (df["high"]-df["low"]).rolling(20).mean()
        for b in fresh_bos(df, sh, sl):
            rt = find_retest(df, b, ar)
            if not rt: continue
            atr = ar.iloc[rt["confirm_idx"]]
            if pd.isna(atr) or atr == 0: continue
            entry = df["close"].iloc[rt["confirm_idx"]]
            exc = excursion(df, b["dir"], entry, atr, rt["confirm_idx"])
            entries.append({"sym":sym,"dir":b["dir"],"entry":entry,"atr":atr,"idx":rt["confirm_idx"],"df":df,"exc":exc})
        time.sleep(0.25)

    N = len(entries)
    print(f"N={N} | Symbols={len(set(e['sym'] for e in entries))}")
    if N > 0:
        mfe=[e["exc"]["mfe"] for e in entries]; mae=[e["exc"]["mae"] for e in entries]
        print(f"MFE mean={sum(mfe)/N:.2f} med={pct(mfe,.5):.2f} | MAE mean={sum(mae)/N:.2f} med={pct(mae,.5):.2f}")
        print(f"P(MFE>=1)={round(sum(1 for v in mfe if v>=1)/N*100,1)}% P(MFE>=2)={round(sum(1 for v in mfe if v>=2)/N*100,1)}%")
        print(f"P(MAE<=1)={round(sum(1 for v in mae if v<=1)/N*100,1)}% P(MAE<=1.5)={round(sum(1 for v in mae if v<=1.5)/N*100,1)}%")
        for pl,nl in [(1,0.75),(1,1),(2,1),(2,1.25),(3,1.5)]:
            cp=cv=0
            for e in entries:
                r1,r2=e["exc"]["rp"].get(pl),e["exc"]["rn"].get(nl)
                if r1 is None and r2 is None: continue
                cv+=1
                if r1 is not None and (r2 is None or r1<=r2): cp+=1
            print(f"  P(+{pl} before -{nl}): {round(cp/cv*100,1) if cv else None}% (n={cv})")
        print("Fixed SL=1.0 ATR:")
        for tp in [1,1.5,2,3]:
            outs=[sim(e["df"],e["dir"],e["entry"],1.0,tp,e["atr"],e["idx"]) for e in entries]
            outs=[o for o in outs if o is not None]
            if outs:
                wins=[o for o in outs if o>0]; loss=[o for o in outs if o<0]
                exp=sum(outs)/len(outs)
                pf=(sum(wins)/abs(sum(loss))) if loss and sum(loss)!=0 else None
                print(f"  TP={tp}R N={len(outs)} WR={round(len(wins)/len(outs)*100,1)}% Exp={round(exp,3)} PF={round(pf,2) if pf else 'N/A'} NetExp={round(exp-FEE/100,3)}")
        by_s={}
        for e in entries: by_s.setdefault(e["sym"],0); by_s[e["sym"]]+=1
        top=sorted(by_s.items(),key=lambda x:-x[1])[:6]
        print("Top symbols:", top)
