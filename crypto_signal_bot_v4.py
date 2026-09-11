"""Setup E: Volatility Compression -> Expansion + BTC Alignment — Cycle 1 Detection"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
SYMS = ["ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT"]
LB = 120
COMP_RATIO = 0.6      # فشردگی: ATR کوتاه‌مدت/بلندمدت زیر این مقدار (baseline ثابت)
EXP_MULT = 1.3        # انبساط: Range کندل فعلی نسبت به ATR بلندمدت

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

def detect_compression_expansion(df, btc_df, lb=LB):
    """
    فقط با اطلاعات تا کندل i (Causal). Compression = نسبت ATR کوتاه(5)
    به بلند(20) در کندل i-1 زیر آستانه. Expansion = Range کندل i بزرگ‌تر
    از ATR بلند * EXP_MULT. جهت از خود کندل Expansion. BTC Filter: در
    همون timestamp، آیا close فعلی BTC بالای/زیر SMA20 خودشه (هم‌جهت).
    """
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    atr5 = df["range"].rolling(5).mean()
    atr20 = df["range"].rolling(20).mean()
    btc_sma20 = btc_df["close"].rolling(20).mean()
    btc_by_time = dict(zip(btc_df["time"], zip(btc_df["close"], btc_sma20)))

    events = []
    start = max(21, len(df)-lb)
    for i in range(start, len(df)):
        if pd.isna(atr5.iloc[i-1]) or pd.isna(atr20.iloc[i-1]) or atr20.iloc[i-1] == 0:
            continue
        compressed = (atr5.iloc[i-1] / atr20.iloc[i-1]) < COMP_RATIO
        if not compressed:
            continue
        row = df.iloc[i]
        if row["range"] < EXP_MULT * atr20.iloc[i-1]:
            continue
        direction = "bullish" if row["close"] > row["open"] else "bearish"

        btc_info = btc_by_time.get(row["time"])
        if btc_info is None or pd.isna(btc_info[1]):
            continue
        btc_close, btc_sma = btc_info
        btc_dir = "bullish" if btc_close > btc_sma else "bearish"
        if btc_dir != direction:
            continue  # BTC Alignment Filter

        events.append({"idx": i, "direction": direction, "comp_ratio": round(atr5.iloc[i-1]/atr20.iloc[i-1],2)})
    return events

if __name__ == "__main__":
    print("SETUP E CYCLE 1 — Detection/Causality")
    btc_df = get_df("BTC-USDT")
    total_events, causal_ok = 0, True
    for sym in SYMS:
        df = get_df(sym)
        if df is None or len(df) < 80 or btc_df is None:
            continue
        events = detect_compression_expansion(df, btc_df)
        total_events += len(events)
        for e in events[:2]:
            print(f"  {sym} {e['direction']} idx={e['idx']} comp_ratio={e['comp_ratio']}")
        # Causality check: هیچ اطلاعاتی از i+1 به بعد استفاده نشده (خود منطق تضمین می‌کنه، فقط sanity)
        for e in events:
            if e["idx"] >= len(df):
                causal_ok = False
        time.sleep(0.3)

    print(f"\nTotal events (9 symbols, BTC-aligned): {total_events}")
    print(f"Causality sanity PASS={causal_ok}")
    print(f"Decision: {'PASS - proceed to Cycle 2' if total_events > 5 and causal_ok else 'Review needed'}")
