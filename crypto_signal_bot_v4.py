"""
Crypto Signal Bot V4 — Regime-Aware Structure & Breakout Engine
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "15m": "15min", "1h": "1hour", "1d": "1day"}

TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT"
]

MIN_SL_DISTANCE_PCT = 0.005  # حداقل فاصله SL از Entry - طبق بند ۹ Spec، محافظتی، قابل بازبینی بعدی


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


def get_ohlcv_v4(symbol, interval_key, total_candles=150):
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


# ============================================================
# STRUCTURE ENGINE (مشترک بین 4H و Daily)
# ============================================================

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
    """خروجی رگیم: 'up' / 'down' / 'range' (choppy)"""
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


def detect_bos(df, swing_highs, swing_lows, lookback_candles=30):
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


# ============================================================
# PHASE 1-2: DAILY REGIME + GLOBAL REGIME FILTER
# ============================================================

def get_daily_regime(symbol):
    """
    Daily Regime رو با همون منطق Structure Engine محاسبه می‌کنه (طبق
    بند ۲ Spec: از منطق/اندیکاتور جدید استفاده نشه، فقط تایم‌فریم عوض بشه).
    خروجی: 'BULLISH' / 'BEARISH' / 'CHOPPY'
    """
    df_daily = get_ohlcv_v4(symbol, "1d", total_candles=100)
    if df_daily is None or len(df_daily) < 30:
        return None
    df_daily = drop_unclosed_candle(df_daily, "1d")
    structure = classify_market_structure(df_daily)
    mapping = {"up": "BULLISH", "down": "BEARISH", "range": "CHOPPY"}
    return mapping[structure["regime"]]


def global_regime_filter(daily_regime, h4_regime_raw, requested_direction):
    """
    طبق بند ۳ و ۴ Spec: تصمیم می‌گیره که آیا `requested_direction`
    (bullish/bearish) اجازه بررسی بیشتر داره یا باید رد بشه.

    h4_regime_raw: خروجی خام Structure Engine روی 4H ('up'/'down'/'range')
    daily_regime: 'BULLISH'/'BEARISH'/'CHOPPY' یا None (داده در دسترس نبود)

    خروجی: (allowed: bool, reason: str)
    """
    h4_map = {"up": "BULLISH", "down": "BEARISH", "range": "CHOPPY"}
    h4_regime = h4_map.get(h4_regime_raw, "CHOPPY")

    if daily_regime is None:
        return False, "daily_regime_unavailable"

    direction_regime = "BULLISH" if requested_direction == "bullish" else "BEARISH"

    # بند ۴: تناقض مستقیم Daily/4H
    if daily_regime == "BULLISH" and h4_regime == "BEARISH" and direction_regime == "BEARISH":
        return False, "conflict_daily_bullish_4h_bearish_short_setup"
    if daily_regime == "BEARISH" and h4_regime == "BULLISH" and direction_regime == "BULLISH":
        return False, "conflict_daily_bearish_4h_bullish_long_setup"

    # بند ۳: قانون اصلی - Daily جهت میده، خلافش ممنوعه
    if daily_regime == "BULLISH" and direction_regime == "BEARISH":
        return False, "daily_bullish_short_forbidden"
    if daily_regime == "BEARISH" and direction_regime == "BULLISH":
        return False, "daily_bearish_long_forbidden"

    # Daily CHOPPY: خودش به‌تنهایی سیگنال تولید نمی‌کنه، ولی مانع 4H هم نیست
    # (تصمیم بر عهده 4H regime و قوانین PATH گذاشته می‌شه)
    return True, "allowed"


# ============================================================
# PHASE 3: UNIT TESTS (V3 Failure Reproduction Tests)
# ============================================================

def run_regime_filter_unit_tests():
    """
    طبق بند ۵ Spec: باید این سناریوها تست بشن. چون تست‌ها منطق
    global_regime_filter رو مستقیم صدا می‌زنن (نه از طریق API واقعی)،
    بدون نیاز به داده زنده اجرا می‌شن.
    """
    tests = []

    # Test 1: Daily Bullish + 4H Choppy + Short Setup -> MUST REJECT
    allowed, reason = global_regime_filter("BULLISH", "range", "bearish")
    tests.append(("Test1_DailyBullish_4hChoppy_Short_MUST_REJECT", not allowed, reason))

    # Test 2: Daily Bearish + 4H Choppy + Long Setup -> MUST REJECT
    allowed, reason = global_regime_filter("BEARISH", "range", "bullish")
    tests.append(("Test2_DailyBearish_4hChoppy_Long_MUST_REJECT", not allowed, reason))

    # Test 3: Daily Bullish + 4H Choppy + Long Setup -> MUST ALLOW (ادامه Validation)
    allowed, reason = global_regime_filter("BULLISH", "range", "bullish")
    tests.append(("Test3_DailyBullish_4hChoppy_Long_MUST_ALLOW", allowed, reason))

    # Test 4: Daily Bearish + 4H Choppy + Short Setup -> MUST ALLOW
    allowed, reason = global_regime_filter("BEARISH", "range", "bearish")
    tests.append(("Test4_DailyBearish_4hChoppy_Short_MUST_ALLOW", allowed, reason))

    # Test 5 (Conflict): Daily Bullish + 4H Bearish + Short Setup -> MUST REJECT
    allowed, reason = global_regime_filter("BULLISH", "down", "bearish")
    tests.append(("Test5_Conflict_DailyBullish_4hBearish_Short_MUST_REJECT", not allowed, reason))

    # Test 6 (Conflict): Daily Bearish + 4H Bullish + Long Setup -> MUST REJECT
    allowed, reason = global_regime_filter("BEARISH", "up", "bullish")
    tests.append(("Test6_Conflict_DailyBearish_4hBullish_Long_MUST_REJECT", not allowed, reason))

    # Test 7 (Sanity): Daily Bullish + 4H Bullish + Long Setup -> MUST ALLOW
    allowed, reason = global_regime_filter("BULLISH", "up", "bullish")
    tests.append(("Test7_DailyBullish_4hBullish_Long_MUST_ALLOW", allowed, reason))

    # Test 8 (Sanity): Daily Bearish + 4H Bearish + Short Setup -> MUST ALLOW
    allowed, reason = global_regime_filter("BEARISH", "down", "bearish")
    tests.append(("Test8_DailyBearish_4hBearish_Short_MUST_ALLOW", allowed, reason))

    return tests


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1-3: Daily Regime + Global Filter + Unit Tests")
    print("=" * 70)

    print("\n--- Unit Tests (منطقی، بدون نیاز به داده زنده) ---")
    tests = run_regime_filter_unit_tests()
    all_passed = True
    for name, passed, reason in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"{status} | {name} | reason={reason}")

    print(f"\nنتیجه کلی Unit Tests: {'✅ همه موفق' if all_passed else '❌ حداقل یک تست شکست خورد'}")

    print("\n" + "-" * 70)
    print("--- تست Daily Regime واقعی روی چند نماد ---")
    for symbol in TEST_SYMBOLS[:5]:
        daily_regime = get_daily_regime(symbol)
        print(f"{symbol} | Daily Regime = {daily_regime}")
        time.sleep(0.5)
