"""Candidate 1: SMT Divergence -> BOS -> OB+FVG Entry — Cycle 1 Detection"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT"]
LB, RETEST_MAX_BARS = 120, 5

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
        time.sleep(0.15)
        if len(d) < 100: break
    if not dfs: return None
    f = pd.concat(dfs).drop_duplicates("time").sort_values("time").reset_index(drop=True)
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
        rh = [s for s in sh if s["index"]+3<i]
        if rh and c>rh[-1]["price"] and rh[-1]["price"]!=lh:
            ev.append({"dir":"bullish","idx":i,"level":rh[-1]["price"]}); lh=rh[-1]["price"]
        rl = [s for s in sl if s["index"]+3<i]
        if rl and c<rl[-1]["price"] and rl[-1]["price"]!=ll:
            ev.append({"dir":"bearish","idx":i,"level":rl[-1]["price"]}); ll=rl[-1]["price"]
    return ev

def check_smt(df, btc_df, sh, sl, btc_sh, btc_sl, bos_idx, direction):
    """SMT: نزدیک‌ترین Swing قبل از BOS رو با نزدیک‌ترین Swing هم‌زمان BTC مقایسه می‌کنه"""
    if direction == "bullish":
        rel = [s for s in sl if s["index"]+3 < bos_idx]
        btc_rel = [s for s in btc_sl if s["index"]+3 < bos_idx]
        if len(rel) < 2 or len(btc_rel) < 2: return False
        sym_higher_low = rel[-1]["price"] > rel[-2]["price"]
        btc_lower_low = btc_rel[-1]["price"] < btc_rel[-2]["price"]
        return sym_higher_low and btc_lower_low
    else:
        rel = [s for s in sh if s["index"]+3 < bos_idx]
        btc_rel = [s for s in btc_sh if s["index"]+3 < bos_idx]
        if len(rel) < 2 or len(btc_rel) < 2: return False
        sym_lower_high = rel[-1]["price"] < rel[-2]["price"]
        btc_higher_high = btc_rel[-1]["price"] > btc_rel[-2]["price"]
        return sym_lower_high and btc_higher_high

def find_ob(df, bos_idx, direction):
    """آخرین کندل مخالف قبل از BOS (Causal، فقط ایندکس قبل از BOS)"""
    for j in range(bos_idx-1, max(0,bos_idx-10), -1):
        row = df.iloc[j]
        if direction == "bullish" and row["close"] < row["open"]:
            return {"idx": j, "low": row["low"], "high": row["high"]}
        if direction == "bearish" and row["close"] > row["open"]:
            return {"idx": j, "low": row["low"], "high": row["high"]}
    return None

def find_fvg(df, bos_idx, direction):
    """FVG سه‌کندلی حول BOS (Causal، فقط کندل‌های <= bos_idx)"""
    for k in range(bos_idx, max(1,bos_idx-5), -1):
        if k < 2: continue
        c0, c2 = df.iloc[k-2], df.iloc[k]
        if direction == "bullish" and c0["high"] < c2["low"]:
            return {"low": c0["high"], "high": c2["low"]}
        if direction == "bearish" and c0["low"] > c2["high"]:
            return {"low": c2["high"], "high": c0["low"]}
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

if __name__ == "__main__":
    print("CANDIDATE 1: SMT->BOS->OB+FVG — CYCLE 1 Detection")
    btc_df = get_df("BTC-USDT")
    btc_sh, btc_sl = swings(btc_df)
    total_bos, smt_pass, ob_found, fvg_found, entries, causal_ok = 0,0,0,0,[],True

    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80: continue
        sh, sl = swings(df)
        bos_list = fresh_bos(df, sh, sl)
        total_bos += len(bos_list)
        for b in bos_list:
            if not check_smt(df, btc_df, sh, sl, btc_sh, btc_sl, b["idx"], b["dir"]): continue
            smt_pass += 1
            ob = find_ob(df, b["idx"], b["dir"])
            fvg = find_fvg(df, b["idx"], b["dir"])
            if ob: ob_found += 1
            if fvg: fvg_found += 1
            zone = ob if ob else fvg
            if not zone: continue
            if not (zone.get("idx", -1) < b["idx"] if ob else True): causal_ok = False
            entry_idx = find_entry(df, b["idx"], zone, b["dir"])
            if entry_idx:
                entries.append({"sym":sym,"dir":b["dir"],"bos_idx":b["idx"],"entry_idx":entry_idx})

    print(f"Total Fresh BOS: {total_bos}")
    print(f"SMT-confirmed BOS: {smt_pass}")
    print(f"OB found: {ob_found} | FVG found: {fvg_found}")
    print(f"Final causal entries: {len(entries)}")
    print(f"Causality PASS={causal_ok}")
    for e in entries[:8]:
        print(f"  {e['sym']} {e['dir']} bos={e['bos_idx']} entry={e['entry_idx']}")
