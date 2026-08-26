"""
Crypto Signal Bot V4 — Phase 5e: close_position Distribution + Directional Audit
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "1d": "1day"}
TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
    "NEAR-USDT", "APT-USDT", "ARB-USDT", "OP-USDT"
]
LOOKBACK_RECENT = 40


def safe_get(url, params=None, retries=3):
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(5)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        return resp
    return resp


def fetch_kucoin_candles(symbol, interval_key, end_at=None):
    params = {"symbol": symbol, "type": KUCOIN_INTERVALS[interval_key]}
    if end_at:
        params["endAt"] = int(end_at)
    resp = safe_get(KUCOIN_BASE, params=params)
    if resp is None or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if data.get("code") != "200000":
        return None
    raw = data.get("data", [])
    if not raw:
        return None
    rows = []
    for r in raw:
        if len(r) < 6:
            continue
        rows.append({"time": int(r[0]), "open": float(r[1]), "close": float(r[2]),
                     "high": float(r[3]), "low": float(r[4]), "volume": float(r[5])})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def get_ohlcv_v4(symbol, interval_key, total_candles=200):
    all_dfs = []
    end_at = None
    remaining = total_candles
    while remaining > 0:
        df = fetch_kucoin_candles(symbol, interval_key, end_at=end_at)
        if df is None or df.empty:
            break
        all_dfs.append(df)
        remaining -= len(df)
        end_at = int(df["time"].min()) - 1
        time.sleep(0.4)
        if len(df) < 100:
            break
    if not all_dfs:
        return None
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    full_df["dt"] = pd.to_datetime(full_df["time"], unit="s", utc=True)
    return full_df.tail(total_candles).reset_index(drop=True)


def drop_unclosed_candle(df, interval_key):
    if df is None or df.empty:
        return df
    interval_seconds = {"4h": 4 * 3600, "1d": 86400}
    now_ts = int(time.time())
    last_candle_time = int(df["time"].iloc[-1])
    duration = interval_seconds.get(interval_key, 0)
    if duration and now_ts < last_candle_time + duration:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def find_swings(df, left=3, right=3):
    highs = df["high"].values
    lows = df["low"].values
    swing_highs, swing_lows = [], []
    for i in range(left, len(df) - right):
        window_h = highs[i-left:i+right+1]
        window_l = lows[i-left:i+right+1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            swing_highs.append({"index": i, "price": highs[i], "time": df["dt"].iloc[i]})
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swing_lows.append({"index": i, "price": lows[i], "time": df["dt"].iloc[i]})
    return swing_highs, swing_lows


def classify_market_structure(df, left=3, right=3):
    swing_highs, swing_lows = find_swings(df, left, right)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"regime": "range", "swing_highs": swing_highs, "swing_lows": swing_lows}
    h1, h2 = swing_highs[-2:]
    l1, l2 = swing_lows[-2:]
    if h2["price"] > h1["price"] and l2["price"] > l1["price"]:
        regime = "up"
    elif h2["price"] < h1["price"] and l2["price"] < l1["price"]:
        regime = "down"
    else:
        regime = "range"
    return {"regime": regime, "swing_highs": swing_highs, "swing_lows": swing_lows}


def detect_bos(df, swing_highs, swing_lows, lookback_candles=LOOKBACK_RECENT):
    events = []
    recent_start = max(0, len(df) - lookback_candles)
    for i in range(recent_start, len(df)):
        close_price = df["close"].iloc[i]
        relevant_highs = [s for s in swing_highs if s["index"] < i]
        relevant_lows = [s for s in swing_lows if s["index"] < i]
        if relevant_highs and close_price > relevant_highs[-1]["price"]:
            events.append({"type": "bullish_bos", "index": i, "time": df["dt"].iloc[i], "close": close_price})
        if relevant_lows and close_price < relevant_lows[-1]["price"]:
            events.append({"type": "bearish_bos", "index": i, "time": df["dt"].iloc[i], "close": close_price})
    return events


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def get_daily_regime(symbol):
    df_daily = get_ohlcv_v4(symbol, "1d", total_candles=100)
    if df_daily is None or len(df_daily) < 30:
        return None
    df_daily = drop_unclosed_candle(df_daily, "1d")
    structure = classify_market_structure(df_daily)
    mapping = {"up": "BULLISH", "down": "BEARISH", "range": "CHOPPY"}
    return mapping[structure["regime"]]


def analyze_displacement(df, idx, avg_body, avg_volume, direction,
                          body_multiplier=1.2, volume_multiplier=1.3, close_threshold=0.75):
    if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]) or avg_body.iloc[idx] == 0:
        return None
    row = df.iloc[idx]
    body_size = abs(row["close"] - row["open"])
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return None
    body_ratio = body_size / avg_body.iloc[idx]
    volume_ratio = row["volume"] / avg_volume.iloc[idx]
    if direction == "bullish":
        close_position = (row["close"] - row["low"]) / candle_range
    else:
        close_position = (row["high"] - row["close"]) / candle_range
    return {
        "direction": direction, "body_ratio": round(body_ratio, 3),
        "body_ok": body_ratio >= body_multiplier,
        "volume_ratio": round(volume_ratio, 3), "volume_ok": volume_ratio >= volume_multiplier,
        "close_position": round(close_position, 3), "close_ok": close_position >= close_threshold,
        "all_pass": body_ratio >= body_multiplier and volume_ratio >= volume_multiplier and close_position >= close_threshold,
    }


def bucket_close_position(cp):
    buckets = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
               (0.50, 0.60), (0.60, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in buckets:
        if lo <= cp < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "out_of_range"


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5e: close_position Distribution + Directional Audit")
    print("=" * 70)

    all_records = []

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=200)
        if df4h is None or len(df4h) < 60:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        structure = classify_market_structure(df4h)
        bos_events = detect_bos(df4h, structure["swing_highs"], structure["swing_lows"])
        daily_regime = get_daily_regime(symbol)
        avg_body = compute_avg_body(df4h)
        avg_volume = compute_avg_volume(df4h)

        for bos in bos_events:
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
            d = analyze_displacement(df4h, idx, avg_body, avg_volume, direction)
            if d is None:
                continue
            d["symbol"] = symbol
            d["daily_regime"] = daily_regime
            d["h4_regime"] = structure["regime"]
            all_records.append(d)
        time.sleep(1)

    total = len(all_records)
    bullish = [r for r in all_records if r["direction"] == "bullish"]
    bearish = [r for r in all_records if r["direction"] == "bearish"]

    print(f"\nکل نمونه: {total} (Bullish={len(bullish)}, Bearish={len(bearish)})")

    print("\n--- توزیع close_position (کل) ---")
    dist = {}
    for r in all_records:
        b = bucket_close_position(r["close_position"])
        dist[b] = dist.get(b, 0) + 1
    for b in sorted(dist.keys()):
        print(f"  {b}: {dist[b]}")

    print("\n--- توزیع close_position (فقط Bullish) ---")
    dist_b = {}
    for r in bullish:
        b = bucket_close_position(r["close_position"])
        dist_b[b] = dist_b.get(b, 0) + 1
    for b in sorted(dist_b.keys()):
        print(f"  {b}: {dist_b[b]}")

    print("\n--- توزیع close_position (فقط Bearish) ---")
    dist_s = {}
    for r in bearish:
        b = bucket_close_position(r["close_position"])
        dist_s[b] = dist_s.get(b, 0) + 1
    for b in sorted(dist_s.keys()):
        print(f"  {b}: {dist_s[b]}")

    candidates_60_75 = [r for r in all_records if 0.60 <= r["close_position"] < 0.75]
    print(f"\nتعداد نمونه در بازه Candidate (0.60-0.75): {len(candidates_60_75)}")
    for r in candidates_60_75:
        print(f"  {r['symbol']} | {r['direction']} | close_pos={r['close_position']} | body_ok={r['body_ok']} | vol_ok={r['volume_ok']} | daily={r['daily_regime']} | 4h={r['h4_regime']}")

    all_pass_count = sum(1 for r in all_records if r["all_pass"])
    print(f"\nکل نمونه‌هایی که همه ۳ شرط Displacement را پاس کردند: {all_pass_count}")
    print("=" * 70)
