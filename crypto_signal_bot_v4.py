"""
Crypto Signal Bot V4 — Phase 5h: New Extension Metric (pre-BOS + entry chasing)
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
LEGACY_MAX_EXTENSION_ATR = 2.5
NEW_MAX_PRE_BOS_EXTENSION_ATR = 2.5
NEW_MAX_ENTRY_EXTENSION_ATR = 1.5


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


def legacy_extension(df, idx, direction, avg_range, structure, max_atr=LEGACY_MAX_EXTENSION_ATR):
    """Metric قدیمی: از Swing تا Close همون کندل BOS (شامل خود Displacement)"""
    if pd.isna(avg_range.iloc[idx]) or avg_range.iloc[idx] == 0:
        return None, None
    current_price = df["close"].iloc[idx]
    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < idx]
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
    if not relevant:
        return None, None
    origin_price = relevant[-1]["price"]
    if direction == "bullish":
        extension = current_price - origin_price
    else:
        extension = origin_price - current_price
    ext_atr = extension / avg_range.iloc[idx]
    return ext_atr, (ext_atr <= max_atr)


def new_extension_metrics(df, idx, direction, avg_range, structure,
                           max_pre_bos=NEW_MAX_PRE_BOS_EXTENSION_ATR,
                           max_entry_ext=NEW_MAX_ENTRY_EXTENSION_ATR):
    """
    Metric جدید طبق تصمیم طراح - دو مؤلفه جدا:
    1. pre_bos_extension: فاصله از Swing تا Open کندل BOS (قبل از خود
       Displacement) - اندازه‌گیری اینکه آیا حرکت قبل از این کندل خودش
       زیادی کشیده بوده یا نه.
    2. entry_extension: فاصله از سطح BOS (broken_level) تا Close کندل
       BOS - اندازه‌گیری اینکه Entry (که در Close این کندل قرار می‌گیره)
       چقدر از سطح شکسته‌شده دور شده.
    """
    if pd.isna(avg_range.iloc[idx]) or avg_range.iloc[idx] == 0:
        return None, None, None, None
    row = df.iloc[idx]
    open_price = row["open"]
    close_price = row["close"]

    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < idx]
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
    if not relevant:
        return None, None, None, None
    origin_price = relevant[-1]["price"]

    if direction == "bullish":
        pre_bos_extension = (open_price - origin_price) / avg_range.iloc[idx]
    else:
        pre_bos_extension = (origin_price - open_price) / avg_range.iloc[idx]

    pre_bos_ok = pre_bos_extension <= max_pre_bos
    return pre_bos_extension, pre_bos_ok, open_price, origin_price


def entry_extension_from_bos(broken_level, close_price, direction, atr):
    if direction == "bullish":
        ext = (close_price - broken_level) / atr
    else:
        ext = (broken_level - close_price) / atr
    return ext, (ext <= NEW_MAX_ENTRY_EXTENSION_ATR)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5h: New Extension Metric vs Legacy — 39-sample comparison")
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

            legacy_ext, legacy_pass = legacy_extension(df4h, idx, direction, avg_range, structure)
            pre_bos_ext, pre_bos_pass, open_price, origin_price = new_extension_metrics(
                df4h, idx, direction, avg_range, structure)
            if pre_bos_ext is None:
                continue

            atr_val = avg_range.iloc[idx]
            entry_ext, entry_pass = entry_extension_from_bos(bos["broken_level"], bos["close"], direction, atr_val)

            new_pass = pre_bos_pass and entry_pass

            records.append({
                "symbol": symbol, "direction": direction,
                "legacy_ext": round(legacy_ext, 2) if legacy_ext else None,
                "legacy_pass": legacy_pass,
                "pre_bos_ext": round(pre_bos_ext, 2),
                "entry_ext": round(entry_ext, 2),
                "new_pass": new_pass,
                "idx": idx,
            })
        time.sleep(1)

    print(f"\nکل نمونه (Displacement-passed): {len(records)}\n")

    for r in records:
        legacy_str = "PASS" if r["legacy_pass"] else "REJECT"
        new_str = "PASS" if r["new_pass"] else "REJECT"
        print(f"{r['symbol']} | {r['direction']} | LegacyExt={r['legacy_ext']}({legacy_str}) | "
              f"PreBOS={r['pre_bos_ext']} | EntryExt={r['entry_ext']} | New={new_str}")

    legacy_pass_count = sum(1 for r in records if r["legacy_pass"])
    new_pass_count = sum(1 for r in records if r["new_pass"])
    freed = sum(1 for r in records if not r["legacy_pass"] and r["new_pass"])
    still_rejected = sum(1 for r in records if not r["new_pass"])

    print(f"\nLegacy Metric PASS: {legacy_pass_count} / {len(records)}")
    print(f"New Metric PASS: {new_pass_count} / {len(records)}")
    print(f"نمونه‌هایی که با Metric جدید آزاد شدند (قبلاً رد بودند): {freed}")
    print(f"نمونه‌هایی که همچنان با Metric جدید رد شدند: {still_rejected}")
    print("=" * 70)
