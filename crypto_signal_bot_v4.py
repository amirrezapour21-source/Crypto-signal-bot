"""
Crypto Signal Bot V4 — Phase 5y: Displacement Discrimination Audit (Shadow Only)
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "1d": "1day"}
TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
    "NEAR-USDT", "APT-USDT", "ARB-USDT", "OP-USDT",
    "SUI-USDT", "INJ-USDT", "TIA-USDT", "SEI-USDT", "FIL-USDT",
    "ATOM-USDT", "LTC-USDT", "ETC-USDT", "TRX-USDT", "ICP-USDT",
    "AAVE-USDT", "UNI-USDT", "MKR-USDT", "RUNE-USDT", "FTM-USDT", "GRT-USDT",
    "ALGO-USDT", "VET-USDT", "HBAR-USDT", "EGLD-USDT", "XLM-USDT",
    "THETA-USDT", "SAND-USDT", "MANA-USDT", "AXS-USDT", "CHZ-USDT",
    "COMP-USDT", "SNX-USDT", "CRV-USDT", "LDO-USDT", "DYDX-USDT",
    "GMX-USDT", "STX-USDT", "KAVA-USDT", "ZIL-USDT", "ONE-USDT"
]
LOOKBACK_RECENT = 60
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


def get_ohlcv_v4(symbol, interval_key, total_candles=250):
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
        time.sleep(0.2)
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
        return False
    direction_regime = "BULLISH" if requested_direction == "bullish" else "BEARISH"
    if daily_regime == "BULLISH" and direction_regime == "BEARISH":
        return False
    if daily_regime == "BEARISH" and direction_regime == "BULLISH":
        return False
    return True


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def raw_displacement_metrics(df, idx, avg_body, avg_volume, direction):
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
    return {"body_ratio": body_ratio, "volume_ratio": volume_ratio, "close_position": close_position}


def detect_follow_through(df, breakout_idx, direction, candles_after=2):
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return None
    breakout_close = df["close"].iloc[breakout_idx]
    after = df.iloc[breakout_idx+1:end_idx]
    if direction == "bullish":
        return bool((after["close"] > breakout_close).any())
    return bool((after["close"] < breakout_close).any())


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def median(values):
    return percentile(values, 0.5)


def bucket_success_rate(records, metric_key, n_buckets=4):
    """رکوردها رو بر اساس مقدار metric_key به n_buckets تقسیم می‌کنه و
    Follow-through Success Rate هر باکت رو برمی‌گردونه."""
    valid = [r for r in records if r[metric_key] is not None and r["follow_through"] is not None]
    if len(valid) < n_buckets * 3:
        return []
    sorted_recs = sorted(valid, key=lambda r: r[metric_key])
    bucket_size = len(sorted_recs) // n_buckets
    results = []
    for i in range(n_buckets):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_buckets - 1 else len(sorted_recs)
        bucket = sorted_recs[start:end]
        if not bucket:
            continue
        ft_vals = [r["follow_through"] for r in bucket]
        success_rate = sum(ft_vals) / len(ft_vals) * 100
        min_val = bucket[0][metric_key]
        max_val = bucket[-1][metric_key]
        results.append({"range": f"{min_val:.2f}-{max_val:.2f}", "n": len(bucket), "success_rate": round(success_rate, 1)})
    return results


if __name__ == "__main__":
    print("PHASE: 5y")
    print("STATUS: EXECUTING - Displacement Discrimination Audit")
    print("=" * 70)

    records = []

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=250)
        if df4h is None or len(df4h) < 80:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        structure = classify_market_structure(df4h)
        bos_events = detect_bos_fresh_only(df4h, structure["swing_highs"], structure["swing_lows"])
        daily_regime = get_daily_regime(symbol)
        avg_body = compute_avg_body(df4h)
        avg_volume = compute_avg_volume(df4h)

        for bos in bos_events:
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

            if not global_regime_filter(daily_regime, direction):
                continue

            dm = raw_displacement_metrics(df4h, idx, avg_body, avg_volume, direction)
            if dm is None:
                continue

            ft = detect_follow_through(df4h, idx, direction)

            records.append({
                "symbol": symbol, "direction": direction,
                "body_ratio": dm["body_ratio"], "volume_ratio": dm["volume_ratio"],
                "close_position": dm["close_position"], "follow_through": ft,
            })
        time.sleep(0.5)

    print(f"\nTotal BOS analyzed (Daily-passed): {len(records)}")
    ft_valid = [r for r in records if r["follow_through"] is not None]
    print(f"Follow-through valid samples: {len(ft_valid)}\n")

    for metric in ["body_ratio", "volume_ratio", "close_position"]:
        vals = [r[metric] for r in records if r[metric] is not None]
        print(f"\n{'=' * 50}")
        print(f"METRIC: {metric}")
        print(f"{'=' * 50}")
        print(f"N={len(vals)} | Min={min(vals):.2f} | P25={percentile(vals,0.25):.2f} | "
              f"Median={median(vals):.2f} | P75={percentile(vals,0.75):.2f} | Max={max(vals):.2f}")

        current_threshold = {"body_ratio": 1.2, "volume_ratio": 1.3, "close_position": 0.75}[metric]
        fail_rate = sum(1 for v in vals if v < current_threshold) / len(vals) * 100
        print(f"Current threshold={current_threshold} | Fail Rate={round(fail_rate,1)}%")

        buckets = bucket_success_rate(records, metric, n_buckets=4)
        print("Follow-through Success Rate by bucket (low->high):")
        for b in buckets:
            print(f"  range={b['range']} | n={b['n']} | FT_success_rate={b['success_rate']}%")

        # False reject: زیر Threshold ولی Follow-through=True
        false_reject = sum(1 for r in records if r[metric] is not None and r[metric] < current_threshold
                            and r["follow_through"] is True)
        true_ft_total = sum(1 for r in records if r["follow_through"] is True)
        if true_ft_total:
            print(f"False Reject Rate (زیر Threshold ولی FT=True): {false_reject}/{true_ft_total} "
                  f"({round(false_reject/true_ft_total*100,1)}%)")

    print("\n" + "=" * 70)
    print("COMBINED SUMMARY (Anti-Chasing از Phase 5x در برابر Displacement):")
    print("  Anti-Chasing (Group A->C): 46.2% -> 38.0% -> 11.1% (تفکیک قوی، یک‌طرفه)")
    print("=" * 70)
