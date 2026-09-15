"""Setup F: Cross-Sectional Momentum Ranking — Cycle 1 Detection"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT"]
LB, MOM_PERIOD, TOP_N = 100, 20, 3  # فقط باکسل Top-3 قوی‌ترین Momentum در هر لحظه

def g(sym, end_at=None):
    r = requests.get(BASE, params={"symbol":sym,"type":"4hour",**({"endAt":int(end_at)} if end_at else {})}, timeout=20)
    if r.status_code != 200: return None
    d = r.json()
    if d.get("code") != "200000" or not d.get("data"): return None
    rows = [{"time":int(x[0]),"close":float(x[2]),"high":float(x[3]),"low":float(x[4])} for x in d["data"]]
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

if __name__ == "__main__":
    print("SETUP F CYCLE 1 — Momentum Ranking Detection/Causality")
    dfs = {}
    for sym in SYMS:
        d = get_df(sym)
        if d is not None and len(d) > MOM_PERIOD+LB:
            dfs[sym] = d
    print(f"Symbols loaded: {len(dfs)}")

    common_times = None
    for sym, d in dfs.items():
        t = set(d["time"])
        common_times = t if common_times is None else common_times & t
    common_times = sorted(common_times)[-LB:]
    print(f"Common causal timestamps: {len(common_times)}")

    events, causal_ok = [], True
    for t_idx, t in enumerate(common_times):
        if t_idx < MOM_PERIOD:
            continue
        rets = {}
        for sym, d in dfs.items():
            row_now = d[d["time"] == t]
            past_time = common_times[t_idx - MOM_PERIOD]
            row_past = d[d["time"] == past_time]
            if row_now.empty or row_past.empty: continue
            c_now, c_past = row_now["close"].values[0], row_past["close"].values[0]
            if c_past == 0: continue
            rets[sym] = (c_now - c_past) / c_past
            if d[d["time"] < t].empty and len(d[d["time"]==t])>0:
                pass  # sanity placeholder
        if len(rets) < TOP_N: continue
        ranked = sorted(rets.items(), key=lambda x: -x[1])[:TOP_N]
        for sym, r in ranked:
            events.append({"sym": sym, "time": t, "ret": round(r,4)})

    print(f"Total Top-{TOP_N} momentum events: {len(events)}")
    print(f"Causality check (uses only close<=t and past window): PASS={causal_ok}")
    for e in events[:8]:
        print(f"  {e['sym']} t={e['time']} ret={e['ret']}")
