"""Candidate 1 — Cycle 2 FINAL FIX: require OB+FVG both, single-candle causal retest confirm"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
"ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
"COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT","GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
"1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT","ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
"ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT","OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT"]
LB, RETEST_MAX_BARS, HOLD, ZONE_BUFFER_ATR, MIN_RR = 120, 5, 30, 0.1, 2.0

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
    h, lo, t = df["high"].values, df["low"].values, df["time"].values
    sh, sl = [], []
    for i in range(l, len(df)-r):
        if h[i]==h[i-l:i+r+1].max() and (h[i-l:i+r+1]==h[i]).sum()==1: sh.append({"index":i,"price":h[i],"time":t[i]})
        if lo[i]==lo[i-l:i+r+1].min() and (lo[i-l:i+r+1]==lo[i]).sum()==1: sl.append({"index":i,"price":lo[i],"time":t[i]})
    return sh, sl

def fresh_bos(df, sh, sl, lb=LB):
    ev, lh, ll = [], None, None
    for i in range(max(0,len(df)-lb), len(df)):
        c = df["close"].iloc[i]; t = df["time"].iloc[i]
        rh = [s for s in sh if s["index"]+3<i]
        if rh and c>rh[-1]["price"] and rh[-1]["price"]!=lh:
            ev.append({"dir":"bullish","idx":i,"time":t}); lh=rh[-1]["price"]
        rl = [s for s in sl if s["index"]+3<i]
        if rl and c<rl[-1]["price"] and rl[-1]["price"]!=ll:
            ev.append({"dir":"bearish","idx":i,"time":t}); ll=rl[-1]["price"]
    return ev

def check_smt_by_time(sh, sl, btc_sh, btc_sl, bos_time, direction):
    if direction == "bullish":
        rel = [s for s in sl if s["time"] < bos_time]
        btc_rel = [s for s in btc_sl if s["time"] < bos_time]
        if len(rel) < 2 or len(btc_rel) < 2: return False
        return rel[-1]["price"]>rel[-2]["price"] and btc_rel[-1]["price"]<btc_rel[-2]["price"]
    else:
        rel = [s for s in sh if s["time"] < bos_time]
        btc_rel = [s for s in btc_sh if s["time"] < bos_time]
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
    """FIX: تأیید فقط با اطلاعات خودِ همون کندلی که Zone رو لمس کرده -
    بدون نگاه به کندل بعدی (j+1 حذف شد)."""
    for j in range(bos_idx+1, min(bos_idx+max_bars+1, len(df))):
        row = df.iloc[j]
        touched = (direction=="bullish" and row["low"]<=zone["high"]) or (direction=="bearish" and row["high"]>=zone["low"])
        if touched:
            c = row["close"]
            if (direction=="bullish" and c>zone["low"]) or (direction=="bearish" and c<zone["high"]):
                return j
    return None

def build_trade(direction, entry, zone, atr):
    buf = ZONE_BUFFER_ATR * atr
    if direction == "bullish":
        sl = zone["low"] - buf; risk = entry - sl
        if risk <= 0: return None
        tp = entry + risk * MIN_RR
    else:
        sl = zone["high"] + buf; risk = sl - entry
        if risk <= 0: return None
        tp = entry - risk * MIN_RR
    return {"entry":entry,"sl":sl,"tp":tp,"risk":risk}

def sim_trade(df, direction, trade, idx, hold=HOLD):
    end = min(idx+1+hold, len(df))
    for i in range(idx+1, end):
        row = df.iloc[i]
        sl_hit = row["low"]<=trade["sl"] if direction=="bullish" else row["high"]>=trade["sl"]
        tp_hit = row["high"]>=trade["tp"] if direction=="bullish" else row["low"]<=trade["tp"]
        if sl_hit and tp_hit: return -1.0, "ambiguous_same_candle_SL_assumed"
        if sl_hit: return -1.0, "SL"
        if tp_hit: return MIN_RR, "TP"
    return None, "open_no_outcome"

def excursion(df, direction, entry, atr, idx, hold=HOLD):
    end=min(idx+1+hold,len(df)); mfe=mae=0
    for i in range(idx+1,end):
        row=df.iloc[i]
        fav,adv=(row["high"]-entry,entry-row["low"]) if direction=="bullish" else (entry-row["low"],row["high"]-entry)
        mfe,mae=max(mfe,fav),max(mae,adv)
    return mfe/atr, mae/atr

if __name__ == "__main__":
    print("CANDIDATE 1 — CYCLE 2 FINAL (OB+FVG required, single-candle causal entry)")
    btc_df = get_df("BTC-USDT")
    btc_sh, btc_sl = swings(btc_df)

    raw, ob_fvg_qualified, valid, rejected, overlap_removed = 0, 0, [], {}, 0
    last_exit_time = {}

    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80: continue
        sh, sl = swings(df)
        for b in fresh_bos(df, sh, sl):
            if not check_smt_by_time(sh, sl, btc_sh, btc_sl, b["time"], b["dir"]): continue
            raw += 1
            ob = find_ob(df, b["idx"], b["dir"])
            fvg = find_fvg(df, b["idx"], b["dir"])
            if not (ob and fvg):
                continue
            ob_fvg_qualified += 1
            zone = ob
            entry_idx = find_entry(df, b["idx"], zone, b["dir"])
            if entry_idx is None: continue
            entry_time = df["time"].iloc[entry_idx]

            if sym in last_exit_time and entry_time < last_exit_time[sym]:
                overlap_removed += 1; continue

            atr = df["atr20"].iloc[entry_idx]
            if pd.isna(atr) or atr == 0:
                rejected["atr_invalid"] = rejected.get("atr_invalid",0)+1; continue
            entry_price = df["close"].iloc[entry_idx]
            trade = build_trade(b["dir"], entry_price, zone, atr)
            if trade is None:
                rejected["invalid_risk"] = rejected.get("invalid_risk",0)+1; continue

            r_mult, outcome = sim_trade(df, b["dir"], trade, entry_idx)
            exit_idx = min(entry_idx+1+HOLD, len(df)-1)
            last_exit_time[sym] = df["time"].iloc[exit_idx]

            mfe, mae = excursion(df, b["dir"], trade["entry"], atr, entry_idx)
            valid.append({"sym":sym,"dir":b["dir"],"entry":trade["entry"],"sl":trade["sl"],"tp":trade["tp"],
                          "r":r_mult,"outcome":outcome,"mfe":mfe,"mae":mae})
        time.sleep(0.2)

    print(f"\nRaw SMT+BOS entries: {raw}")
    print(f"OB+FVG qualified: {ob_fvg_qualified}")
    print(f"Overlapping removed: {overlap_removed}")
    print(f"Rejected: {rejected}")
    print(f"Final independent trades: {len(valid)}")

    N = len(valid)
    if N < 30: print("INSUFFICIENT SAMPLE for meaningful IS inference.")
    if N > 0:
        closed = [v for v in valid if v["r"] is not None]
        ambiguous = sum(1 for v in valid if v["outcome"]=="ambiguous_same_candle_SL_assumed")
        print(f"Closed: {len(closed)} | Ambiguous: {ambiguous}")
        if closed:
            rs=[v["r"] for v in closed]; wins=[r for r in rs if r>0]; losses=[r for r in rs if r<0]
            exp=sum(rs)/len(rs)
            pf=(sum(wins)/abs(sum(losses))) if losses and sum(losses)!=0 else None
            cum=peak=maxdd=0
            for r in rs: cum+=r; peak=max(peak,cum); maxdd=min(maxdd,cum-peak)
            print(f"WR={round(len(wins)/len(rs)*100,1)}% Exp={round(exp,3)}R PF={round(pf,2) if pf else 'N/A'} TotalR={round(sum(rs),2)} MaxDD={round(maxdd,2)}R")
        mfe_l=[v["mfe"] for v in valid]; mae_l=[v["mae"] for v in valid]
        print(f"MFE mean={sum(mfe_l)/N:.2f} | MAE mean={sum(mae_l)/N:.2f}")
