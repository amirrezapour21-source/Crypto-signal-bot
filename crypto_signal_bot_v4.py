"""Candidate 1: SMT->BOS->OB+FVG — Cycle 2 IS Edge Discovery (full universe, frozen logic)"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
"ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
"COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT","GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
"1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT","ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
"ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT","OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT"]
LB, RETEST_MAX_BARS, HOLD = 120, 5, 30

def g(sym, end_at=None):
    r = requests.get(BASE, params={"symbol":sym,"type":"4hour",**({"endAt":int(end_at)} if end_at else {})}, timeout=20)
    if r.status_code != 200: return None
    d = r.json()
    if d.get("code") != "200000" or not d.get("data"): return None
    rows = [{"time":int(x[0]),"open":float(x[1]),"close":float(x[2]),"high":float(x[3]),"low":float(x[4])} for x in d["data"]]
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
    if int(time.time()) < int(f["time"].iloc[-1])+14400: f = f.iloc[:-1]
    f["range"] = f["high"]-f["low"]; f["atr20"] = f["range"].rolling(20).mean()
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
        rh = [s for s in sh if s["index"]+3<i]
        if rh and c>rh[-1]["price"] and rh[-1]["price"]!=lh:
            ev.append({"dir":"bullish","idx":i,"level":rh[-1]["price"]}); lh=rh[-1]["price"]
        rl = [s for s in sl if s["index"]+3<i]
        if rl and c<rl[-1]["price"] and rl[-1]["price"]!=ll:
            ev.append({"dir":"bearish","idx":i,"level":rl[-1]["price"]}); ll=rl[-1]["price"]
    return ev

def check_smt(sh, sl, btc_sh, btc_sl, bos_idx, direction):
    if direction == "bullish":
        rel = [s for s in sl if s["index"]+3 < bos_idx]
        btc_rel = [s for s in btc_sl if s["index"]+3 < bos_idx]
        if len(rel) < 2 or len(btc_rel) < 2: return False
        return rel[-1]["price"]>rel[-2]["price"] and btc_rel[-1]["price"]<btc_rel[-2]["price"]
    else:
        rel = [s for s in sh if s["index"]+3 < bos_idx]
        btc_rel = [s for s in btc_sh if s["index"]+3 < bos_idx]
        if len(rel) < 2 or len(btc_rel) < 2: return False
        return rel[-1]["price"]<rel[-2]["price"] and btc_rel[-1]["price"]>btc_rel[-2]["price"]

def find_ob(df, bos_idx, direction):
    for j in range(bos_idx-1, max(0,bos_idx-10), -1):
        row = df.iloc[j]
        if direction=="bullish" and row["close"]<row["open"]: return {"low":row["low"],"high":row["high"]}
        if direction=="bearish" and row["close"]>row["open"]: return {"low":row["low"],"high":row["high"]}
    return None

def find_fvg(df, bos_idx, direction):
    for k in range(bos_idx, max(1,bos_idx-5), -1):
        if k < 2: continue
        c0, c2 = df.iloc[k-2], df.iloc[k]
        if direction=="bullish" and c0["high"]<c2["low"]: return {"low":c0["high"],"high":c2["low"]}
        if direction=="bearish" and c0["low"]>c2["high"]: return {"low":c2["high"],"high":c0["low"]}
    return None

def find_entry(df, bos_idx, zone, direction, max_bars=RETEST_MAX_BARS):
    for j in range(bos_idx+1, min(bos_idx+max_bars+1, len(df))):
        row = df.iloc[j]
        touched = (direction=="bullish" and row["low"]<=zone["high"]) or (direction=="bearish" and row["high"]>=zone["low"])
        if touched:
            for k in range(j, min(j+2, len(df))):
                c = df["close"].iloc[k]
                if (direction=="bullish" and c>zone["low"]) or (direction=="bearish" and c<zone["high"]):
                    return k
            return None
    return None

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

def excursion(df, direction, entry, atr, idx, hold=HOLD):
    end=min(idx+1+hold,len(df)); mfe=mae=0
    for i in range(idx+1,end):
        row=df.iloc[i]
        fav,adv=(row["high"]-entry,entry-row["low"]) if direction=="bullish" else (entry-row["low"],row["high"]-entry)
        mfe,mae=max(mfe,fav),max(mae,adv)
    return mfe/atr, mae/atr

if __name__ == "__main__":
    print("CANDIDATE 1: SMT->BOS->OB+FVG — CYCLE 2 IS Edge")
    btc_df = get_df("BTC-USDT")
    btc_sh, btc_sl = swings(btc_df)
    entries = []
    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80: continue
        sh, sl = swings(df)
        for b in fresh_bos(df, sh, sl):
            if not check_smt(sh, sl, btc_sh, btc_sl, b["idx"], b["dir"]): continue
            ob = find_ob(df, b["idx"], b["dir"])
            fvg = find_fvg(df, b["idx"], b["dir"])
            zone = ob if ob else fvg
            if not zone: continue
            entry_idx = find_entry(df, b["idx"], zone, b["dir"])
            if entry_idx is None: continue
            atr = df["atr20"].iloc[entry_idx]
            if pd.isna(atr) or atr == 0: continue
            entries.append({"sym":sym,"dir":b["dir"],"df":df,"idx":entry_idx,"atr":atr})
        time.sleep(0.2)

    N = len(entries)
    print(f"N={N} | Symbols={len(set(e['sym'] for e in entries))}")
    if N < 30:
        print("SAMPLE TOO SMALL for meaningful IS inference — reporting raw numbers only, no conclusion.")
    if N > 0:
        mfe_l, mae_l = [], []
        for e in entries:
            entry_price = e["df"]["close"].iloc[e["idx"]]
            mfe, mae = excursion(e["df"], e["dir"], entry_price, e["atr"], e["idx"])
            mfe_l.append(mfe); mae_l.append(mae)
        print(f"MFE mean={sum(mfe_l)/N:.2f} | MAE mean={sum(mae_l)/N:.2f}")
        for tp in [1,1.5,2,3]:
            outs=[]
            for e in entries:
                entry_price = e["df"]["close"].iloc[e["idx"]]
                r = sim(e["df"], e["dir"], entry_price, 1.0, tp, e["atr"], e["idx"])
                if r is not None: outs.append(r)
            if outs:
                wins=[o for o in outs if o>0]; loss=[o for o in outs if o<0]
                exp=sum(outs)/len(outs)
                pf=(sum(wins)/abs(sum(loss))) if loss and sum(loss)!=0 else None
                print(f"  TP={tp}R N={len(outs)} WR={round(len(wins)/len(outs)*100,1)}% Exp={round(exp,3)} PF={round(pf,2) if pf else 'N/A'}")
        by_s={}
        for e in entries: by_s.setdefault(e["sym"],0); by_s[e["sym"]]+=1
        print("By symbol:", by_s)
