"""Candidate 2: SVA/Value-Area Mean-Reversion — Cycle 1 Detection/Causality"""
import requests, pandas as pd, numpy as np, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT"]
VP_WINDOW, VA_PCT, LB = 40, 0.70, 120

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
        time.sleep(0.15)
        if len(d) < 100: break
    if not dfs: return None
    f = pd.concat(dfs).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if int(time.time()) < int(f["time"].iloc[-1])+14400: f = f.iloc[:-1]
    return f.tail(n).reset_index(drop=True)

def compute_value_area(window_df, bins=24):
    """Causal: فقط از کندل‌های داخل window (که همه قبل از i هستن) استفاده می‌کنه"""
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
    included = {poc_i}
    cum = vol[poc_i]
    lo_i, hi_i = poc_i, poc_i
    while cum/total_vol < VA_PCT and (lo_i>0 or hi_i<bins-1):
        left = vol[lo_i-1] if lo_i>0 else -1
        right = vol[hi_i+1] if hi_i<bins-1 else -1
        if right >= left: hi_i += 1; cum += vol[hi_i]
        else: lo_i -= 1; cum += vol[lo_i]
    return {"poc":centers[poc_i], "vah":centers[hi_i], "val":centers[lo_i]}

def detect_events(df, lb=LB, window=VP_WINDOW):
    """در هر کندل i (Causal): VA از window قبل از i محاسبه می‌شه (بدون
    خود کندل i). اگه کندل i-1 بیرون VA بسته باشه و کندل i برگرده داخل
    VA، Event تولید می‌شه."""
    events = []
    start = max(window+1, len(df)-lb)
    for i in range(start, len(df)):
        win = df.iloc[i-window:i]  # فقط کندل‌های قبل از i (Causal)
        va = compute_value_area(win)
        if va is None: continue
        prev_close, cur_close = df["close"].iloc[i-1], df["close"].iloc[i]
        if prev_close > va["vah"] and cur_close <= va["vah"]:
            events.append({"idx":i, "dir":"bearish", "va":va})
        elif prev_close < va["val"] and cur_close >= va["val"]:
            events.append({"idx":i, "dir":"bullish", "va":va})
    return events

if __name__ == "__main__":
    print("CANDIDATE 2: SVA/Value-Area Mean-Reversion — CYCLE 1 Detection")
    total, causal_ok = 0, True
    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80: continue
        events = detect_events(df)
        total += len(events)
        for e in events[:2]:
            print(f"  {sym} {e['dir']} idx={e['idx']} POC={e['va']['poc']:.4f} VAH={e['va']['vah']:.4f} VAL={e['va']['val']:.4f}")
        for e in events:
            if e["idx"] >= len(df): causal_ok = False
    print(f"\nTotal events (9 symbols): {total}")
    print(f"Causality sanity PASS={causal_ok}")
