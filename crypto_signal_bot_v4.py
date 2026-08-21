
Crypto Signal Bot V4 — Multi-Symbol Phase 3 Debug Test
--------------------------------------------------
نسخه دیباگ‌دار: برای هر Setup رد‌شده، دقیقاً مشخص می‌کنه کدوم شرط(ها)
شکست خوردن، تا بفهمیم کدوم فیلتر بیش‌ازحد سخت‌گیره.
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"

KUCOIN_INTERVALS = {
    "4h": "4hour",
    "15m": "15min",
    "1h": "1hour",
    "1d": "1day",
}

TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT"
]


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
    if interval_key not in KUCOIN_INTERVALS:
        raise ValueError(f"interval نامعتبر: {interval_key}")
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
        rows.append({
            "time": int(r[0]), "open": float(r[1]), "close": float(r[2]),
            "high": float(r[3]), "low": float(r[4]), "volume": float(r[5]),
        })
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


def detect_displacement_debug(df, idx, avg_body, avg_volume, direction,
                                body_multiplier=1.2, volume_multiplier=1.3, close_position_pct=0.25):
    """نسخه دیباگ: برمی‌گردونه (نتیجه_کلی, جزئیات هر شرط)"""
    if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]) or avg_body.iloc[idx] == 0:
        return False, {"body_ok": None, "volume_ok": None, "close_ok": None, "reason": "insufficient_lookback_data"}

    row = df.iloc[idx]
    body_size = abs(row["close"] - row["open"])
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False, {"body_ok": None, "volume_ok": None, "close_ok": None, "reason": "zero_range"}

    body_ratio = body_size / avg_body.iloc[idx]
    volume_ratio = row["volume"] / avg_volume.iloc[idx]
    body_ok = body_ratio >= body_multiplier
    volume_ok = volume_ratio >= volume_multiplier

    if direction == "bullish":
        close_position = (row["close"] - row["low"]) / candle_range
    else:
        close_position = (row["high"] - row["close"]) / candle_range
    close_ok = close_position >= (1 - close_position_pct)

    details = {
        "body_ratio": round(body_ratio, 2), "body_ok": body_ok,
        "volume_ratio": round(volume_ratio, 2), "volume_ok": volume_ok,
        "close_position": round(close_position, 2), "close_ok": close_ok,
    }
    return (body_ok and volume_ok and close_ok), details


def detect_follow_through(df, breakout_idx, direction, candles_after=2):
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return False
    breakout_close = df["close"].iloc[breakout_idx]
    after = df.iloc[breakout_idx+1:end_idx]
    if direction == "bullish":
        return (after["close"] > breakout_close).any()
    return (after["close"] < breakout_close).any()


def check_anti_chasing(df, idx, direction, avg_range, structure, max_extension_atr=2.5):
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


def scan_breakout_setups_debug(df, structure, bos_events, lookback_recent=15):
    avg_body = compute_avg_body(df)
    avg_volume = compute_avg_volume(df)
    avg_range = compute_avg_range(df)
    results = []
    recent_bos = [e for e in bos_events if e["index"] >= len(df) - lookback_recent]

    for bos in recent_bos:
        idx = bos["index"]
        direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

        compression_ok, compression_ratio = detect_compression(df, idx)
        displacement_ok, disp_details = detect_displacement_debug(df, idx, avg_body, avg_volume, direction)
        follow_through_ok = detect_follow_through(df, idx, direction)
        anti_chasing_ok, extension_atr = check_anti_chasing(df, idx, direction, avg_range, structure)

        regime_aligned = (structure["regime"] == "up" and direction == "bullish") or \
                          (structure["regime"] == "down" and direction == "bearish")

        path_a_valid = compression_ok and displacement_ok and follow_through_ok
        path_b_valid = regime_aligned and displacement_ok and follow_through_ok and anti_chasing_ok
        valid_setup = path_a_valid or path_b_valid

        results.append({
            "time": bos["time"], "direction": direction, "close": bos["close"],
            "compression_ok": compression_ok, "compression_ratio": compression_ratio,
            "displacement_ok": displacement_ok, "disp_details": disp_details,
            "follow_through_ok": follow_through_ok,
            "anti_chasing_ok": anti_chasing_ok, "extension_atr": extension_atr,
            "regime_aligned": regime_aligned,
            "path_a_valid": path_a_valid, "path_b_valid": path_b_valid,
            "valid_setup": valid_setup,
        })
    return results


def scan_symbol_debug(symbol):
    try:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=150)
        if df4h is None or len(df4h) < 60:
            return {"symbol": symbol, "error": "داده کافی دریافت نشد"}
        df4h = drop_unclosed_candle(df4h, "4h")
        if df4h is None or len(df4h) < 60:
            return {"symbol": symbol, "error": "بعد از حذف کندل باز، داده کافی نیست"}
        structure = classify_market_structure(df4h)
        bos_events = detect_bos(df4h, structure["swing_highs"], structure["swing_lows"])
        results = scan_breakout_setups_debug(df4h, structure, bos_events)
        return {"symbol": symbol, "regime": structure["regime"], "results": results}
    except Exception as e:
        return {"symbol": symbol, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print("=" * 70)
    print(f"PHASE 3 DEBUG SCAN — دلیل دقیق رد شدن هر Setup — {len(TEST_SYMBOLS)} نماد")
    print("=" * 70)

    # شمارنده‌ی کلی برای فهمیدن کدوم شرط بیشترین رد شدن رو داره
    fail_counts = {
        "compression": 0, "displacement_body": 0, "displacement_volume": 0,
        "displacement_close_pos": 0, "follow_through": 0, "anti_chasing": 0,
        "regime_not_aligned": 0,
    }
    total_checked = 0
    total_valid = 0

    for symbol in TEST_SYMBOLS:
        result = scan_symbol_debug(symbol)
        if "error" in result:
            print(f"\n{symbol} | ERROR | {result['error']}")
            continue

        print(f"\n### {symbol} | REGIME={result['regime'].upper()} ###")
        for r in result["results"]:
            total_checked += 1
            if r["valid_setup"]:
                total_valid += 1
                print(f"  ✅ VALID | {r['direction']} | {r['time']} | close={r['close']:.4f}")
                continue

            if not r["compression_ok"]:
                fail_counts["compression"] += 1
            dd = r["disp_details"]
            if dd.get("body_ok") is False:
                fail_counts["displacement_body"] += 1
            if dd.get("volume_ok") is False:
                fail_counts["displacement_volume"] += 1
            if dd.get("close_ok") is False:
                fail_counts["displacement_close_pos"] += 1
            if not r["follow_through_ok"]:
                fail_counts["follow_through"] += 1
            if not r["anti_chasing_ok"]:
                fail_counts["anti_chasing"] += 1
            if not r["regime_aligned"]:
                fail_counts["regime_not_aligned"] += 1

            print(f"  ❌ {r['direction']} | {r['time']} | close={r['close']:.4f}")
            print(f"     Compression={r['compression_ok']}({r['compression_ratio']}) | "
                  f"Body={dd.get('body_ok')}({dd.get('body_ratio')}) | "
                  f"Vol={dd.get('volume_ok')}({dd.get('volume_ratio')}) | "
                  f"ClosePos={dd.get('close_ok')}({dd.get('close_position')}) | "
                  f"FollowThru={r['follow_through_ok']} | "
                  f"AntiChase={r['anti_chasing_ok']}({r['extension_atr']}) | "
                  f"RegimeAligned={r['regime_aligned']}")

        time.sleep(1)

    print("\n" + "=" * 70)
    print(f"خلاصه نهایی: {total_valid} معتبر از {total_checked} بررسی‌شده")
    print("تعداد رد شدن به تفکیک دلیل:")
    for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print("=" * 70)
