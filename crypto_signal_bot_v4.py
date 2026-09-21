"""Candidate 2: SVA/Value-Area Mean-Reversion — Cycle 2 IS Edge (full universe, causal SL/TP)"""
import requests, pandas as pd, numpy as np, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
"ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
"COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT","GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
"1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT","ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
"ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT","OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT"]
VP_WINDOW, VA_PCT, LB, HOLD, MIN_RR, SL_BUFFER_PCT, MAX_EXCURSION_LOOKBACK = 40, 0.70, 120, 30, 2.0, 0.001, 10

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
    if int(time.time()) < int(f["time"].iloc[-1])+14400: f = f.iloc[:-1]
    return f.tail(n).reset_index(drop=True)

def compute_value_area(window_df, bins=24):
    tp = (window_df["high"]+window_df["low"]+window_df["close"])/3
    lo, hi = window_df["low"].min(), window_df["high"].max()
    if hi <= lo: return None
    edges = np.linspace(lo, hi, bins+1)
    idx = np.clip(np.digitize(tp, edges)-1, 0, bins-1)
    vol = np.zeros(bins)
    for j, v in zip(idx, window_df["volume"].values): vol[j] += v
    centers = (edges[:-1]+edges[1:])/2
    total_vol = vol.sum()
    if total_vol == 0: return None
    poc_i = int(np.argmax(vol))
    lo_i = hi_i = poc_i
    cum = vol[poc_i]
    while cum/total_vol < VA_PCT and (lo_i>0 or hi_i<bins-1):
        left = vol[lo_i-1] if lo_i>0 else -1
        right = vol[hi_i+1] if hi_i<bins-1 else -1
        if right >= left: hi_i += 1; cum += vol[hi_i]
        else: lo_i -= 1; cum += vol[lo_i]
    return {"poc":centers[poc_i], "vah":centers[hi_i], "val":centers[lo_i]}

def detect_events(df, lb=LB, window=VP_WINDOW):
    events = []
    for i in range(max(window+1, len(df)-lb), len(df)):
        win = df.iloc[i-window:i]
        va = compute_value_area(win)
        if va is None: continue
        prev_close, cur_close = df["close"].iloc[i-1], df["close"].iloc[i]
        if prev_close > va["vah"] and cur_close <= va["vah"]:
            events.append({"idx":i, "dir":"bearish", "va":va})
        elif prev_close < va["val"] and cur_close >= va["val"]:
            events.append({"idx":i, "dir":"bullish", "va":va})
    return events

def build_trade(df, event):
    idx, direction, va = event["idx"], event["dir"], event["va"]
    entry = df["close"].iloc[idx]
    lookback_start = max(0, idx-MAX_EXCURSION_LOOKBACK)
    if direction == "bullish":
        extreme = df["low"].iloc[lookback_start:idx].min()
        sl = extreme * (1 - SL_BUFFER_PCT)
        risk = entry - sl
        if risk <= 0: return None
        tp = va["poc"]
        reward = tp - entry
    else:
        extreme = df["high"].iloc[lookback_start:idx].max()
        sl = extreme * (1 + SL_BUFFER_PCT)
        risk = sl - entry
        if risk <= 0: return None
        tp = va["poc"]
        reward = entry - tp
    if reward <= 0 or reward/risk < MIN_RR:
        return None
    return {"entry":entry, "sl":sl, "tp":tp, "risk":risk, "rr":reward/risk}

def sim_trade(df, direction, trade, idx, hold=HOLD):
    end = min(idx+1+hold, len(df))
    for i in range(idx+1, end):
        row = df.iloc[i]
        sl_hit = row["low"]<=trade["sl"] if direction=="bullish" else row["high"]>=trade["sl"]
        tp_hit = row["high"]>=trade["tp"] if direction=="bullish" else row["low"]<=trade["tp"]
        if sl_hit and tp_hit: return -1.0, "ambiguous_same_candle_SL_assumed"
        if sl_hit: return -1.0, "SL"
        if tp_hit: return trade["rr"], "TP"
    return None, "open_no_outcome"

if __name__ == "__main__":
    print("CANDIDATE 2 — CYCLE 2 IS Edge Discovery")
    raw, valid, rejected, overlap_removed = 0, [], {}, 0
    last_exit_time = {}
    is_period_start, is_period_end = None, None

    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80: continue
        if is_period_start is None or df["time"].iloc[max(0,len(df)-LB)] < is_period_start:
            is_period_start = df["time"].iloc[max(0,len(df)-LB)]
        if is_period_end is None or df["time"].iloc[-1] > is_period_end:
            is_period_end = df["time"].iloc[-1]

        for e in detect_events(df):
            raw += 1
            entry_time = df["time"].iloc[e["idx"]]
            if sym in last_exit_time and entry_time < last_exit_time[sym]:
                overlap_removed += 1; continue
            trade = build_trade(df, e)
            if trade is None:
                rejected["rr_or_risk_invalid"] = rejected.get("rr_or_risk_invalid",0)+1; continue
            r_mult, outcome = sim_trade(df, e["dir"], trade, e["idx"])
            exit_idx = min(e["idx"]+1+HOLD, len(df)-1)
            last_exit_time[sym] = df["time"].iloc[exit_idx]
            valid.append({"sym":sym,"dir":e["dir"],"r":r_mult,"outcome":outcome,"rr":trade["rr"]})
        time.sleep(0.2)

    print(f"IS Period: {pd.to_datetime(is_period_start,unit='s')} to {pd.to_datetime(is_period_end,unit='s')}")
    print(f"Raw events: {raw} | Overlap removed: {overlap_removed} | Rejected: {rejected}")
    N = len(valid)
    print(f"Final independent trades: {N}")

    if N < 30:
        print("INSUFFICIENT SAMPLE — NO EDGE CONCLUSION.")
    if N > 0:
        closed = [v for v in valid if v["r"] is not None]
        ambiguous = sum(1 for v in valid if v["outcome"]=="ambiguous_same_candle_SL_assumed")
        print(f"Closed: {len(closed)} | Ambiguous same-candle: {ambiguous}")
        if closed:
            rs = [v["r"] for v in closed]
            wins=[r for r in rs if r>0]; losses=[r for r in rs if r<0]
            exp = sum(rs)/len(rs)
            pf = (sum(wins)/abs(sum(losses))) if losses and sum(losses)!=0 else None
            cum=peak=maxdd=0
            for r in rs: cum+=r; peak=max(peak,cum); maxdd=min(maxdd,cum-peak)
            print(f"WR={round(len(wins)/len(rs)*100,1)}% Exp={round(exp,3)}R PF={round(pf,2) if pf else 'N/A'} TotalR={round(sum(rs),2)} MaxDD={round(maxdd,2)}R")
        by_s={}
        for v in valid: by_s.setdefault(v["sym"],0); by_s[v["sym"]]+=1
        top=sorted(by_s.items(),key=lambda x:-x[1])[:8]
        print("Top symbols:", top)
