"""
Setup Family B — Cycle 2: Fix duplicate-trigger bug (Fresh Sweep Only)
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "1d": "1day"}
TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
]
LOOKBACK_RECENT = 120
EQUAL_LEVEL_TOLERANCE_ATR = 0.15
RECLAIM_MAX_CANDLES = 3


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


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def find_equal_level_clusters(swings, atr_series, ref_idx, tolerance_atr=EQUAL_LEVEL_TOLERANCE_ATR):
    valid_swings = [s for s in swings if s["index"] < ref_idx]
    if len(valid_swings) < 2:
        return []
    atr = atr_series.iloc[ref_idx]
    if pd.isna(atr) or atr == 0:
        return []
    sorted_swings = sorted(valid_swings, key=lambda s: s["price"])
    groups = []
    current_group = [sorted_swings[0]]
    for s in sorted_swings[1:]:
        if abs(s["price"] - current_group[-1]["price"]) / atr <= tolerance_atr:
            current_group.append(s)
        else:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [s]
    if len(current_group) >= 2:
        groups.append(current_group)
    return groups


def detect_sweep_and_reclaim_v2(df, swing_highs, swing_lows, avg_range, lookback=LOOKBACK_RECENT):
    """
    Fix Cycle 2: هر سطح Liquidity فقط یک‌بار می‌تونه Sweep+Reclaim معتبر
    تولید کنه (Fresh Sweep Only) - دقیقاً همون منطق Fresh-BOS که قبلاً
    استفاده کردیم. بعد از یه Reclaim موفق روی یه سطح، اون سطح دیگه
    نمی‌تونه دوباره سیگنال بده مگر این‌که یه سطح Cluster جدید (با
    قیمت متفاوت) تشکیل بشه.
    """
    events = []
    recent_start = max(0, len(df) - lookback)
    last_swept_level_bull = None
    last_swept_level_bear = None

    for i in range(recent_start, len(df)):
        row = df.iloc[i]

        eh_groups = find_equal_level_clusters(swing_highs, avg_range, i)
        if eh_groups:
            nearest_group = eh_groups[-1]  # نزدیک‌ترین Cluster به قیمت فعلی (آخرین در لیست مرتب‌شده صعودی که هنوز بالای قیمته - ساده‌سازی: از انتها می‌گیریم)
            level_price = max(s["price"] for s in nearest_group)
            if row["high"] > level_price and level_price != last_swept_level_bear:
                sweep_idx = i
                for j in range(sweep_idx, min(sweep_idx + RECLAIM_MAX_CANDLES + 1, len(df))):
                    if df["close"].iloc[j] < level_price:
                        events.append({
                            "type": "bearish_sweep_reclaim", "sweep_idx": sweep_idx,
                            "reclaim_idx": j, "level_price": level_price,
                            "level_group_size": len(nearest_group), "time": df["dt"].iloc[j],
                            "bars_between": j - sweep_idx,
                        })
                        last_swept_level_bear = level_price
                        break

        el_groups = find_equal_level_clusters(swing_lows, avg_range, i)
        if el_groups:
            nearest_group = el_groups[0]
            level_price = min(s["price"] for s in nearest_group)
            if row["low"] < level_price and level_price != last_swept_level_bull:
                sweep_idx = i
                for j in range(sweep_idx, min(sweep_idx + RECLAIM_MAX_CANDLES + 1, len(df))):
                    if df["close"].iloc[j] > level_price:
                        events.append({
                            "type": "bullish_sweep_reclaim", "sweep_idx": sweep_idx,
                            "reclaim_idx": j, "level_price": level_price,
                            "level_group_size": len(nearest_group), "time": df["dt"].iloc[j],
                            "bars_between": j - sweep_idx,
                        })
                        last_swept_level_bull = level_price
                        break

    return events


if __name__ == "__main__":
    print("SETUP FAMILY B — CYCLE 2: Fresh-Sweep-Only Fix")
    print("=" * 70)

    total_events = 0
    same_candle_count = 0
    multi_bar_count = 0

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=250)
        if df4h is None or len(df4h) < 80:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        swing_highs, swing_lows = find_swings(df4h, 3, 3)
        avg_range = compute_avg_range(df4h)

        events = detect_sweep_and_reclaim_v2(df4h, swing_highs, swing_lows, avg_range)
        total_events += len(events)
        same_candle_count += sum(1 for e in events if e["bars_between"] == 0)
        multi_bar_count += sum(1 for e in events if e["bars_between"] > 0)

        print(f"\n{symbol}: {len(events)} رویداد (بعد از Fresh-Sweep-Only)")
        for e in events:
            print(f"  {e['type']} | Level={e['level_price']:.6f} | GroupSize={e['level_group_size']} | "
                  f"idx={e['sweep_idx']}->{e['reclaim_idx']} | BarsBetween={e['bars_between']}")
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"مجموع رویداد بعد از Fix: {total_events} (قبلاً 134 بود)")
    print(f"Same-Candle Reclaim (Wick-Reject): {same_candle_count}")
    print(f"Multi-Bar Reclaim: {multi_bar_count}")
    print("=" * 70)
