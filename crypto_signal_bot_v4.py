"""
Crypto Signal Bot V4 — Phase 5g: extension_atr Deep Audit (no threshold change)
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
MAX_EXTENSION_ATR = 2.5


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
            events.append({"type": "bullish_bos", "index": i, "time": df["dt"].iloc[i],
                          "close": close_price, "broken_level": relevant_highs[-1]["price"]})
        if relevant_lows and close_price < relevant_lows[-1]["price"]:
            events.append({"type": "bearish_bos", "index": i, "time": df["dt"].iloc[i],
                          "close": close_price, "broken_level": relevant_lows[-1]["price"]})
    return events


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def detect_displacement(df, idx, avg_body, avg_volume, direction,
                          body_multiplier=1.2, volume_multiplier=1.3, close_position_pct=0.25):
    if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]) or avg_body.iloc[idx] == 0:
        return False
    row = df.iloc[idx]
    body_size = abs(row["close"] - row["open"])
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    body_ok = body_size >= body_multiplier * avg_body.iloc[idx]
    volume_ok = row["volume"] >= volume_multiplier * avg_volume.iloc[idx]
    if direction == "bullish":
        close_position = (row["close"] - row["low"]) / candle_range
    else:
        close_position = (row["high"] - row["close"]) / candle_range
    close_ok = close_position >= (1 - close_position_pct)
    return body_ok and volume_ok and close_ok


def extension_at_index(df, eval_idx, direction, avg_range, structure):
    """extension_atr رو در یک نقطه زمانی مشخص (eval_idx) محاسبه می‌کنه"""
    if pd.isna(avg_range.iloc[eval_idx]) or avg_range.iloc[eval_idx] == 0:
        return None
    current_price = df["close"].iloc[eval_idx]
    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < eval_idx]
        if not relevant:
            return None
        origin_price = relevant[-1]["price"]
        extension = current_price - origin_price
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < eval_idx]
        if not relevant:
            return None
        origin_price = relevant[-1]["price"]
        extension = origin_price - current_price
    return extension / avg_range.iloc[eval_idx]


def bucket_ext(v):
    if v < 1:
        return "<1"
    if v < 2:
        return "1-2"
    if v < 2.5:
        return "2-2.5"
    if v < 3:
        return "2.5-3"
    if v < 4:
        return "3-4"
    if v < 5:
        return "4-5"
    if v < 6:
        return "5-6"
    if v < 8:
        return "6-8"
    return ">8"


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5g: extension_atr Deep Audit")
    print("=" * 70)

    records = []

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=200)
        if df4h is None or len(df4h) < 60:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        structure = classify_market_structure(df4h)
        bos_events = detect_bos(df4h, structure["swing_highs"], structure["swing_lows"])
        avg_body = compute_avg_body(df4h)
        avg_volume = compute_avg_volume(df4h)
        avg_range = compute_avg_range(df4h)

        for bos in bos_events:
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
            if not detect_displacement(df4h, idx, avg_body, avg_volume, direction):
                continue

            # مؤلفه A: فاصله Swing تا سطح BOS (broken_level) بر حسب ATR
            if direction == "bullish":
                relevant = [s for s in structure["swing_lows"] if s["index"] < idx]
            else:
                relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
            if not relevant or pd.isna(avg_range.iloc[idx]) or avg_range.iloc[idx] == 0:
                continue
            origin_price = relevant[-1]["price"]
            dist_swing_to_bos = abs(bos["broken_level"] - origin_price) / avg_range.iloc[idx]
            dist_bos_to_close = abs(bos["close"] - bos["broken_level"]) / avg_range.iloc[idx]

            ext_at_bos = extension_at_index(df4h, idx, direction, avg_range, structure)

            # extension در ۲ کندل بعد (زمان ارزیابی follow-through)
            eval_idx = min(idx + 2, len(df4h) - 1)
            ext_at_eval = extension_at_index(df4h, eval_idx, direction, avg_range, structure)

            records.append({
                "symbol": symbol, "direction": direction,
                "dist_swing_to_bos": round(dist_swing_to_bos, 2),
                "dist_bos_to_close": round(dist_bos_to_close, 2),
                "ext_at_bos": round(ext_at_bos, 2) if ext_at_bos else None,
                "ext_at_eval_2bars_later": round(ext_at_eval, 2) if ext_at_eval else None,
                "bars_evaluated_later": eval_idx - idx,
            })
        time.sleep(1)

    print(f"\nکل نمونه بررسی‌شده (Displacement-passed): {len(records)}\n")

    for r in records:
        print(f"{r['symbol']} | {r['direction']} | SwingToBOS={r['dist_swing_to_bos']} | "
              f"BOSToClose={r['dist_bos_to_close']} | ExtAtBOS={r['ext_at_bos']} | "
              f"ExtAt+{r['bars_evaluated_later']}bars={r['ext_at_eval_2bars_later']}")

    ext_bos_values = [r["ext_at_bos"] for r in records if r["ext_at_bos"] is not None]
    ext_eval_values = [r["ext_at_eval_2bars_later"] for r in records if r["ext_at_eval_2bars_later"] is not None]

    print("\n--- توزیع extension_atr در لحظه BOS ---")
    dist = {}
    for v in ext_bos_values:
        b = bucket_ext(v)
        dist[b] = dist.get(b, 0) + 1
    for b in ["<1", "1-2", "2-2.5", "2.5-3", "3-4", "4-5", "5-6", "6-8", ">8"]:
        print(f"  {b}: {dist.get(b, 0)}")
    if ext_bos_values:
        sorted_v = sorted(ext_bos_values)
        print(f"  Min={min(ext_bos_values):.2f} Median={sorted_v[len(sorted_v)//2]:.2f} Mean={sum(ext_bos_values)/len(ext_bos_values):.2f} Max={max(ext_bos_values):.2f}")

    print("\n--- توزیع extension_atr در زمان ارزیابی (2 کندل بعد از BOS) ---")
    dist2 = {}
    for v in ext_eval_values:
        b = bucket_ext(v)
        dist2[b] = dist2.get(b, 0) + 1
    for b in ["<1", "1-2", "2-2.5", "2.5-3", "3-4", "4-5", "5-6", "6-8", ">8"]:
        print(f"  {b}: {dist2.get(b, 0)}")
    if ext_eval_values:
        sorted_v2 = sorted(ext_eval_values)
        print(f"  Min={min(ext_eval_values):.2f} Median={sorted_v2[len(sorted_v2)//2]:.2f} Mean={sum(ext_eval_values)/len(ext_eval_values):.2f} Max={max(ext_eval_values):.2f}")

    already_extended_at_bos = sum(1 for v in ext_bos_values if v > MAX_EXTENSION_ATR)
    extended_only_later = sum(1 for r in records if r["ext_at_bos"] and r["ext_at_bos"] <= MAX_EXTENSION_ATR
                               and r["ext_at_eval_2bars_later"] and r["ext_at_eval_2bars_later"] > MAX_EXTENSION_ATR)

    print(f"\nبیش‌ازحد‌کشیده در لحظه BOS (ext_at_bos > 2.5): {already_extended_at_bos}")
    print(f"در لحظه BOS نرمال بود ولی 2 کندل بعد بیش‌ازحد شد: {extended_only_later}")
    print("=" * 70)
