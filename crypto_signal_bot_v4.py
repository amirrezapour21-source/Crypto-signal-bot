"""Setup D: Fresh BOS -> Retest — Cycle 1 Detection/Causality"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT"]
LB, RETEST_TOL_ATR, RETEST_MAX_BARS = 120, 0.3, 5

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

if __name__ == "__main__":
    print("SETUP D CYCLE 1 — Detection/Causality")
    total, causal_ok, dup_ok, events = 0, True, True, []
    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df)<80: continue
        sh, sl = swings(df)
        ar = (df["high"]-df["low"]).rolling(20).mean()
        bos_list = fresh_bos(df, sh, sl)
        for b in bos_list:
            rt = find_retest(df, b, ar)
            if rt:
                if not (b["idx"] < rt["retest_idx"] <= rt["confirm_idx"]): causal_ok = False
                events.append({"sym":sym,**b,**rt})
        total += len(bos_list)
    seen = set()
    for e in events:
        k = (e["sym"], e["level"])
        if k in seen: dup_ok = False
        seen.add(k)
    print(f"Total Fresh BOS: {total} | Retest-confirmed events: {len(events)}")
    print(f"Causality PASS={causal_ok} | Duplicate PASS={dup_ok}")
    for e in events[:8]:
        print(f"  {e['sym']} {e['dir']} lvl={e['level']:.4f} bos={e['idx']} retest={e['retest_idx']} confirm={e['confirm_idx']}")
