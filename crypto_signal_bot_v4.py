"""Setup F Cycle 2: IS Edge (independent events = NEW entries into Top-N only)"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
"ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT"]
LB, MOM_PERIOD, TOP_N, HOLD = 100, 20, 3, 30

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
    f["range"] = f["high"] - f["low"]
    f["atr20"] = f["range"].rolling(20).mean()
    return f.tail(n).reset_index(drop=True)

def sim(df, t_idx, sl_atr, tp_mult, hold=HOLD):
    entry = df["close"].iloc[t_idx]
    atr = df["atr20"].iloc[t_idx]
    if pd.isna(atr) or atr == 0: return None
    sl, tp = entry - sl_atr*atr, entry + sl_atr*atr*tp_mult
    end = min(t_idx+1+hold, len(df))
    for i in range(t_idx+1, end):
        row = df.iloc[i]
        if row["low"] <= sl: return -1.0
        if row["high"] >= tp: return tp_mult
    return None

def excursion(df, t_idx, hold=HOLD):
    entry = df["close"].iloc[t_idx]; atr = df["atr20"].iloc[t_idx]
    if pd.isna(atr) or atr == 0: return None
    end = min(t_idx+1+hold, len(df)); mfe = mae = 0
    for i in range(t_idx+1, end):
        row = df.iloc[i]
        mfe = max(mfe, row["high"]-entry); mae = max(mae, entry-row["low"])
    return {"mfe": mfe/atr, "mae": mae/atr}

if __name__ == "__main__":
    print("SETUP F CYCLE 2 — IS Edge (independent new-entry events only)")
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

    prev_top = set()
    entries = []
    for t_idx, t in enumerate(common_times):
        if t_idx < MOM_PERIOD: continue
        rets = {}
        past_time = common_times[t_idx - MOM_PERIOD]
        for sym, d in dfs.items():
            row_now = d[d["time"] == t]; row_past = d[d["time"] == past_time]
            if row_now.empty or row_past.empty: continue
            c_now, c_past = row_now["close"].values[0], row_past["close"].values[0]
            if c_past == 0: continue
            rets[sym] = (c_now - c_past) / c_past
        if len(rets) < TOP_N: continue
        top_now = set(s for s, r in sorted(rets.items(), key=lambda x: -x[1])[:TOP_N])
        new_entries = top_now - prev_top  # فقط نمادهایی که تازه وارد Top-N شدن (Event مستقل)
        for sym in new_entries:
            d = dfs[sym]
            idx_in_d = d[d["time"] == t].index
            if len(idx_in_d) == 0: continue
            idx = idx_in_d[0]
            entries.append({"sym": sym, "df": d, "idx": idx})
        prev_top = top_now

    N = len(entries)
    print(f"Independent events (new Top-{TOP_N} entries): {N}")

    if N > 0:
        exc = [excursion(e["df"], e["idx"]) for e in entries]
        exc = [x for x in exc if x is not None]
        mfe = [x["mfe"] for x in exc]; mae = [x["mae"] for x in exc]
        if mfe:
            print(f"MFE mean={sum(mfe)/len(mfe):.2f} | MAE mean={sum(mae)/len(mae):.2f}")
        for tp in [1, 1.5, 2, 3]:
            outs = [sim(e["df"], e["idx"], 1.0, tp) for e in entries]
            outs = [o for o in outs if o is not None]
            if outs:
                wins = [o for o in outs if o > 0]; loss = [o for o in outs if o < 0]
                exp = sum(outs)/len(outs)
                pf = (sum(wins)/abs(sum(loss))) if loss and sum(loss) != 0 else None
                print(f"  TP={tp}R N={len(outs)} WR={round(len(wins)/len(outs)*100,1)}% Exp={round(exp,3)} PF={round(pf,2) if pf else 'N/A'}")
        by_s = {}
        for e in entries: by_s.setdefault(e["sym"],0); by_s[e["sym"]] += 1
        print("By symbol:", by_s)
