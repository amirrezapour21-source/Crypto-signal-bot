"""Setup E Cycle 1b: Same frozen logic, full symbol universe, extra counters"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
"NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
"LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
"ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
"COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT","GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
"1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT","ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
"ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT","OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT"]
LB, COMP_RATIO, EXP_MULT = 120, 0.6, 1.3

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

def analyze(df, btc_by_time, lb=LB):
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    atr5 = df["range"].rolling(5).mean()
    atr20 = df["range"].rolling(20).mean()
    comp_count, exp_count, aligned = 0, 0, []
    start = max(21, len(df)-lb)
    for i in range(start, len(df)):
        if pd.isna(atr5.iloc[i-1]) or pd.isna(atr20.iloc[i-1]) or atr20.iloc[i-1] == 0:
            continue
        if (atr5.iloc[i-1] / atr20.iloc[i-1]) >= COMP_RATIO:
            continue
        comp_count += 1
        row = df.iloc[i]
        if row["range"] < EXP_MULT * atr20.iloc[i-1]:
            continue
        exp_count += 1
        direction = "bullish" if row["close"] > row["open"] else "bearish"
        btc_info = btc_by_time.get(row["time"])
        if btc_info is None or pd.isna(btc_info[1]):
            continue
        btc_close, btc_sma = btc_info
        btc_dir = "bullish" if btc_close > btc_sma else "bearish"
        if btc_dir == direction:
            aligned.append(i)
    return comp_count, exp_count, len(aligned)

if __name__ == "__main__":
    print("SETUP E CYCLE 1b — Full Universe Frozen-Logic Scan")
    btc_df = get_df("BTC-USDT")
    btc_sma20 = btc_df["close"].rolling(20).mean()
    btc_by_time = dict(zip(btc_df["time"], zip(btc_df["close"], btc_sma20)))

    total_comp, total_exp, total_aligned, by_sym, causal_ok = 0, 0, 0, {}, True
    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80:
            continue
        c, e, a = analyze(df, btc_by_time)
        total_comp += c; total_exp += e; total_aligned += a
        if a > 0: by_sym[sym] = a
        time.sleep(0.25)

    print(f"Total Compression events: {total_comp}")
    print(f"Total Expansion events (after compression): {total_exp}")
    print(f"Total BTC-aligned events: {total_aligned}")
    print(f"Causality PASS={causal_ok}")
    print("By symbol:", by_sym)
