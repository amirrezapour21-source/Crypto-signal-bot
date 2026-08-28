"""
Crypto Signal Bot V4 — Phase 5k: Fix detect_bos duplicate emission (Fresh BOS only)
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


def detect_bos_fresh_only(df, swing_highs, swing_lows, lookback_candles=LOOKBACK_RECENT):
    """
    نسخه اصلاح‌شده (Phase 5k): فقط اولین کندلی که یه سطح رو می‌شکنه
    به‌عنوان BOS ثبت می‌شه (Fresh Break). قبلاً هر کندل بعدی که بالای
    همون سطح می‌موند هم دوباره BOS ثبت می‌شد که باعث می‌شد BOS های
    "دیر" (چند کندل بعد از شکست واقعی) وارد سیستم بشن و Extension
    بزرگ به‌طور مصنوعی افزایش پیدا کنه.
    قانون جدید: هر broken_level فقط یک‌بار می‌تونه BOS تولید کنه؛ بعد
    از اون، باید یه Swing جدید (بالاتر/پایین‌تر) شکل بگیره تا BOS بعدی
    معتبر باشه.
    """
    events = []
    recent_start = max(0, len(df) - lookback_candles)
    last_broken_high = None
    last_broken_low = None

    for i in range(recent_start, len(df)):
        close_price = df["close"].iloc[i]
        relevant_highs = [s for s in swing_highs if s["index"] < i]
        relevant_lows = [s for s in swing_lows if s["index"] < i]

        if relevant_highs:
            level = relevant_highs[-1]["price"]
            if close_price > level and level != last_broken_high:
                events.append({"type": "bullish_bos", "index": i, "time": df["dt"].iloc[i],
                              "close": close_price, "broken_level": level})
                last_broken_high = level

        if relevant_lows:
            level = relevant_lows[-1]["price"]
            if close_price < level and level != last_broken_low:
                events.append({"type": "bearish_bos", "index": i, "time": df["dt"].iloc[i],
                              "close": close_price, "broken_level": level})
                last_broken_low = level

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


def run_regime_filter_unit_tests():
    tests = []
    allowed, _ = global_regime_filter("BULLISH", "bearish")
    tests.append(("DailyBullish_ShortBOS_MUST_REJECT", not allowed))
    allowed, _ = global_regime_filter("BEARISH", "bullish")
    tests.append(("DailyBearish_LongBOS_MUST_REJECT", not allowed))
    allowed, _ = global_regime_filter("BULLISH", "bullish")
    tests.append(("DailyBullish_LongBOS_MUST_ALLOW", allowed))
    allowed, _ = global_regime_filter("BEARISH", "bearish")
    tests.append(("DailyBearish_ShortBOS_MUST_ALLOW", allowed))
    return tests


def scan_symbol(symbol):
    log = {"symbol": symbol, "setups": [], "bos_count_old": 0, "bos_count_new": 0}
    df4h = get_ohlcv_v4(symbol, "4h", total_candles=200)
    if df4h is None or len(df4h) < 60:
        log["error"] = "4h_data_unavailable"
        return log
    df4h = drop_unclosed_candle(df4h, "4h")
    structure = classify_market_structure(df4h)

    bos_events = detect_bos_fresh_only(df4h, structure["swing_highs"], structure["swing_lows"])
    log["bos_count_new"] = len(bos_events)

    daily_regime = get_daily_regime(symbol)
    log["daily_regime"] = daily_regime
    log["h4_regime"] = structure["regime"]
    avg_body = compute_avg_body(df4h)
    avg_volume = compute_avg_volume(df4h)
    avg_range = compute_avg_range(df4h)

    for bos in bos_events:
        idx = bos["index"]
        direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

        allowed, _ = global_regime_filter(daily_regime, direction)
        if not allowed:
            continue

        displacement_ok = detect_displacement(df4h, idx, avg_body, avg_volume, direction)
        if not displacement_ok:
            continue
        follow_through_ok = detect_follow_through(df4h, idx, direction)
        if follow_through_ok is None or not follow_through_ok:
            continue
        anti_chasing_ok, extension_atr = check_anti_chasing(df4h, idx, direction, avg_range, structure)
        if not anti_chasing_ok:
            continue
        compression_ok, _ = detect_compression(df4h, idx)
        path = "A" if compression_ok else "B"

        log["setups"].append({
            "time": bos["time"], "direction": direction, "path": path,
            "close": bos["close"], "extension_atr": round(extension_atr, 2),
            "daily_regime": daily_regime, "h4_regime": structure["regime"],
        })
    return log


if __name__ == "__main__":
    print("PHASE: 5k")
    print("STATUS: EXECUTING")
    print("=" * 70)

    print("\n--- Regression: Unit Tests ---")
    all_passed = True
    for name, passed in run_regime_filter_unit_tests():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{status} | {name}")
    print(f"Unit Tests: {'ALL PASS' if all_passed else 'FAILED'}")

    print("\n--- Multi-Symbol Scan with Fresh-BOS-Only Fix ---")
    total_setups = 0
    total_bos = 0

    for symbol in TEST_SYMBOLS:
        result = scan_symbol(symbol)
        if "error" in result:
            print(f"\n{symbol} | ERROR")
            continue
        total_bos += result["bos_count_new"]
        print(f"\n{symbol} | Daily={result['daily_regime']} | 4H={result['h4_regime']} | Fresh_BOS_count={result['bos_count_new']}")
        for s in result["setups"]:
            total_setups += 1
            print(f"  SETUP PATH {s['path']} | {s['direction']} | {s['time']} | close={s['close']:.4f} | ext={s['extension_atr']}")
        time.sleep(1)

    print("\n" + "=" * 70)
    print(f"WHAT WAS TESTED: detect_bos با فیلتر Fresh-Break-Only (هر level فقط یک‌بار BOS تولید می‌کنه)")
    print(f"FINDINGS: کل Fresh BOS شناسایی‌شده در 14 نماد: {total_bos} (قبلاً با روش قدیمی روی همین بازه ~229 بود)")
    print(f"ROOT CAUSE: detect_bos قدیمی هر کندلی که بالای یه سطح می‌موند رو دوباره BOS حساب می‌کرد، نه فقط اولین شکست")
    print(f"DECISION: تغییر منطق BOS به Fresh-Break-Only (بدون تغییر هیچ Threshold ای)")
    print(f"CHANGES MADE: تابع detect_bos_fresh_only جایگزین detect_bos شد")
    print(f"TEST RESULTS: SETUP نهایی معتبر = {total_setups}")
    print(f"REGRESSION STATUS: {'ALL PASS' if all_passed else 'FAILED'}")
    print("=" * 70)
