"""
Crypto Signal Bot V4 — Phase 4-5c: PATH B Direction from BOS (not h4_regime)
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "15m": "15min", "1h": "1hour", "1d": "1day"}

TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
    "NEAR-USDT", "APT-USDT", "ARB-USDT", "OP-USDT"
]

MIN_SL_DISTANCE_PCT = 0.005
MAX_EXTENSION_ATR = 2.5
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
    interval_seconds = {"4h": 4 * 3600, "15m": 15 * 60, "1h": 3600, "1d": 86400}
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
    higher_high = h2["price"] > h1["price"]
    higher_low = l2["price"] > l1["price"]
    lower_high = h2["price"] < h1["price"]
    lower_low = l2["price"] < l1["price"]
    if higher_high and higher_low:
        regime = "up"
    elif lower_high and lower_low:
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
        if relevant_highs:
            last_high = relevant_highs[-1]
            if close_price > last_high["price"]:
                events.append({"type": "bullish_bos", "index": i, "time": df["dt"].iloc[i],
                              "close": close_price, "broken_level": last_high["price"]})
        if relevant_lows:
            last_low = relevant_lows[-1]
            if close_price < last_low["price"]:
                events.append({"type": "bearish_bos", "index": i, "time": df["dt"].iloc[i],
                              "close": close_price, "broken_level": last_low["price"]})
    return events


def get_daily_regime(symbol):
    df_daily = get_ohlcv_v4(symbol, "1d", total_candles=100)
    if df_daily is None or len(df_daily) < 30:
        return None
    df_daily = drop_unclosed_candle(df_daily, "1d")
    structure = classify_market_structure(df_daily)
    mapping = {"up": "BULLISH", "down": "BEARISH", "range": "CHOPPY"}
    return mapping[structure["regime"]]


def global_regime_filter(daily_regime, requested_direction):
    """
    فقط بر اساس Daily. مطابق تصمیم طراح: 4H دیگر در این فیلتر دخیل نیست،
    4H فقط برای پیدا کردن خود BOS/Setup استفاده می‌شود.
    """
    if daily_regime is None:
        return False, "daily_regime_unavailable"
    direction_regime = "BULLISH" if requested_direction == "bullish" else "BEARISH"
    if daily_regime == "BULLISH" and direction_regime == "BEARISH":
        return False, "daily_bullish_short_forbidden"
    if daily_regime == "BEARISH" and direction_regime == "BULLISH":
        return False, "daily_bearish_long_forbidden"
    return True, "allowed"


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def detect_compression(df, idx, lookback=20, threshold=0.75):
    if idx < lookback * 2:
        return False, None
    recent_range = (df["high"] - df["low"]).iloc[idx-lookback:idx].mean()
    longer_range = (df["high"] - df["low"]).iloc[idx-lookback*2:idx].mean()
    if longer_range == 0:
        return False, None
    ratio = recent_range / longer_range
    return ratio < threshold, ratio


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


def detect_follow_through(df, breakout_idx, direction, candles_after=2):
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return None
    breakout_close = df["close"].iloc[breakout_idx]
    after = df.iloc[breakout_idx+1:end_idx]
    if direction == "bullish":
        return bool((after["close"] > breakout_close).any())
    return bool((after["close"] < breakout_close).any())


def check_anti_chasing(df, idx, direction, avg_range, structure, max_extension_atr=MAX_EXTENSION_ATR):
    if pd.isna(avg_range.iloc[idx]) or avg_range.iloc[idx] == 0:
        return False, None
    current_price = df["close"].iloc[idx]
    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < idx]
        if not relevant:
            return False, None
        origin_price = relevant[-1]["price"]
        extension = current_price - origin_price
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
        if not relevant:
            return False, None
        origin_price = relevant[-1]["price"]
        extension = origin_price - current_price
    extension_atr = extension / avg_range.iloc[idx]
    return extension_atr <= max_extension_atr, extension_atr


def scan_symbol(symbol):
    log = {"symbol": symbol, "setups": [], "path_a": 0, "path_b": 0,
           "reject_daily": 0, "reject_compression": 0, "reject_followthru": 0,
           "reject_antichase": 0, "reject_displacement": 0, "bullish_bos": 0, "bearish_bos": 0}
    df4h = get_ohlcv_v4(symbol, "4h", total_candles=200)
    if df4h is None or len(df4h) < 60:
        log["error"] = "4h_data_unavailable"
        return log
    df4h = drop_unclosed_candle(df4h, "4h")
    structure = classify_market_structure(df4h)
    bos_events = detect_bos(df4h, structure["swing_highs"], structure["swing_lows"])
    daily_regime = get_daily_regime(symbol)
    log["daily_regime"] = daily_regime
    log["h4_regime"] = structure["regime"]
    log["bos_count"] = len(bos_events)
    avg_body = compute_avg_body(df4h)
    avg_volume = compute_avg_volume(df4h)
    avg_range = compute_avg_range(df4h)

    for bos in bos_events:
        idx = bos["index"]
        direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
        if direction == "bullish":
            log["bullish_bos"] += 1
        else:
            log["bearish_bos"] += 1

        allowed, reason = global_regime_filter(daily_regime, direction)
        if not allowed:
            log["reject_daily"] += 1
            continue

        compression_ok, _ = detect_compression(df4h, idx)
        displacement_ok = detect_displacement(df4h, idx, avg_body, avg_volume, direction)
        follow_through_ok = detect_follow_through(df4h, idx, direction)
        if follow_through_ok is None:
            continue
        anti_chasing_ok, extension_atr = check_anti_chasing(df4h, idx, direction, avg_range, structure)

        path_a_valid = compression_ok and displacement_ok and follow_through_ok
        # PATH B: جهت از خود BOS میاد (نه از h4_regime) - طبق تصمیم طراح
        path_b_valid = displacement_ok and follow_through_ok and anti_chasing_ok

        if path_a_valid:
            log["path_a"] += 1
            log["setups"].append({"time": bos["time"], "direction": direction, "path": "A",
                                   "close": bos["close"], "extension_atr": extension_atr})
        elif path_b_valid:
            log["path_b"] += 1
            log["setups"].append({"time": bos["time"], "direction": direction, "path": "B",
                                   "close": bos["close"], "extension_atr": extension_atr})
        else:
            if not displacement_ok:
                log["reject_displacement"] += 1
            elif not follow_through_ok:
                log["reject_followthru"] += 1
            elif not anti_chasing_ok:
                log["reject_antichase"] += 1
            else:
                log["reject_compression"] += 1
    return log


def run_unit_tests():
    """طبق بند ۱۰ Spec - سناریوهای A تا F"""
    results = []
    # A: Daily Bullish + Bullish BOS -> ALLOW
    allowed, _ = global_regime_filter("BULLISH", "bullish")
    results.append(("A_DailyBullish_BullishBOS_ALLOW", allowed))
    # B: Daily Bearish + Bearish BOS -> ALLOW
    allowed, _ = global_regime_filter("BEARISH", "bearish")
    results.append(("B_DailyBearish_BearishBOS_ALLOW", allowed))
    # C: Daily Bearish + Bullish BOS -> REJECT
    allowed, _ = global_regime_filter("BEARISH", "bullish")
    results.append(("C_DailyBearish_BullishBOS_MUST_REJECT", not allowed))
    # D: Daily Bullish + Bearish BOS -> REJECT
    allowed, _ = global_regime_filter("BULLISH", "bearish")
    results.append(("D_DailyBullish_BearishBOS_MUST_REJECT", not allowed))
    # E: Daily Bullish + Bullish BOS -> ALLOW (4H bullish context, همون A)
    allowed, _ = global_regime_filter("BULLISH", "bullish")
    results.append(("E_DailyBullish_4hBullish_BullishBOS_ALLOW", allowed))
    # F: Daily Bearish + Bearish BOS -> ALLOW (همون B)
    allowed, _ = global_regime_filter("BEARISH", "bearish")
    results.append(("F_DailyBearish_4hBearish_BearishBOS_ALLOW", allowed))
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4-5c: PATH B Direction from Fresh BOS (Daily = Safety Filter Only)")
    print("=" * 70)

    print("\n--- Unit Tests A-F ---")
    all_passed = True
    for name, passed in run_unit_tests():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"{status} | {name}")
    print(f"نتیجه: {'✅ همه موفق' if all_passed else '❌ شکست خورد'}")

    print("\n" + "-" * 70)
    print("--- اسکن چندنمادی ---")

    agg = {"total_bos": 0, "bullish_bos": 0, "bearish_bos": 0, "path_a": 0, "path_b": 0,
           "reject_daily": 0, "reject_compression": 0, "reject_followthru": 0,
           "reject_antichase": 0, "reject_displacement": 0}

    for symbol in TEST_SYMBOLS:
        result = scan_symbol(symbol)
        if "error" in result:
            print(f"\n{symbol} | ERROR | {result['error']}")
            continue

        print(f"\n{symbol} | Daily={result['daily_regime']} | 4H={result['h4_regime']} | BOS={result['bos_count']}")
        for s in result["setups"]:
            print(f"  ✅ SETUP PATH {s['path']} | {s['direction']} | {s['time']} | close={s['close']:.4f} | ext={s['extension_atr']}")

        for k in agg:
            if k == "total_bos":
                agg[k] += result.get("bos_count", 0)
            elif k in result:
                agg[k] += result[k]

        time.sleep(1)

    print("\n" + "=" * 70)
    print("خلاصه کلی:")
    print(f"  کل BOS: {agg['total_bos']} (Bullish={agg['bullish_bos']}, Bearish={agg['bearish_bos']})")
    print(f"  PATH A معتبر: {agg['path_a']}")
    print(f"  PATH B معتبر: {agg['path_b']}")
    print(f"  رد به دلیل Daily: {agg['reject_daily']}")
    print(f"  رد به دلیل Displacement: {agg['reject_displacement']}")
    print(f"  رد به دلیل Follow-through: {agg['reject_followthru']}")
    print(f"  رد به دلیل Anti-Chasing: {agg['reject_antichase']}")
    print(f"  رد به دلیل Compression (فقط برای Path A رد شده‌ها): {agg['reject_compression']}")
    print("=" * 70)
