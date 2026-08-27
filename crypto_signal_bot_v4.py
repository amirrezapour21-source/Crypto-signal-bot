"""
Crypto Signal Bot V4 — Phase 5j: Anti-Chasing Concept / Entry Model Audit
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour"}
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
    interval_seconds = {"4h": 4 * 3600}
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
            swing_highs.append({"index": i, "price": highs[i]})
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swing_lows.append({"index": i, "price": lows[i]})
    return swing_highs, swing_lows


def detect_bos(df, swing_highs, swing_lows, lookback_candles=LOOKBACK_RECENT):
    events = []
    recent_start = max(0, len(df) - lookback_candles)
    for i in range(recent_start, len(df)):
        close_price = df["close"].iloc[i]
        relevant_highs = [s for s in swing_highs if s["index"] < i]
        relevant_lows = [s for s in swing_lows if s["index"] < i]
        if relevant_highs and close_price > relevant_highs[-1]["price"]:
            events.append({"type": "bullish_bos", "index": i, "broken_level": relevant_highs[-1]["price"]})
        if relevant_lows and close_price < relevant_lows[-1]["price"]:
            events.append({"type": "bearish_bos", "index": i, "broken_level": relevant_lows[-1]["price"]})
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


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5j: Entry Model / Anti-Chasing Concept Audit")
    print("=" * 70)

    print("""
--- تعریف فعلی Entry Model در کد V4 (تا این Phase) ---
PATH B فعلاً "Entry" مستقلی تعریف نکرده - در تمام Phase های قبلی
(5-5i)، Entry معادل Close همان کندل BOS فرض شده (چون check_anti_chasing
و entry_extension_from_bos هر دو از df['close'].iloc[idx] یعنی Close
کندل BOS استفاده می‌کنند - نه یک کندل بعدتر، نه Retest، نه Pullback).

یعنی: BOS_to_Entry_ATR در Phase 5h همان مقدار entry_ext بود که از
broken_level تا Close کندل BOS اندازه‌گیری شد - این عملاً معادل
"BOS Displacement" (بند 2-B همین دستور) است، نه یک Entry جدا بعد
از BOS.

نتیجه مهم: در معماری فعلی V4، سه مفهوم (Pre-BOS Extension /
BOS Displacement / Post-BOS Entry Extension) در عمل به دو مقدار
تقلیل یافته‌اند چون Entry = Close BOS candle، یعنی:
  - BOS Displacement == Post-BOS/Entry Extension (چون Entry همان
    Close BOS است)
  - این دقیقاً همان چیزی است که باعث شد "Displacement قوی" خودش
    به‌طور خودکار به "Chasing بزرگ" تبدیل شود.
""")

    print("-" * 70)
    print("--- Audit سه‌مؤلفه‌ای روی نمونه‌های Displacement-passed ---")

    records = []
    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=200)
        if df4h is None or len(df4h) < 60:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        swing_highs, swing_lows = find_swings(df4h, 3, 3)
        bos_events = detect_bos(df4h, swing_highs, swing_lows)
        avg_body = compute_avg_body(df4h)
        avg_volume = compute_avg_volume(df4h)
        avg_range = compute_avg_range(df4h)

        for bos in bos_events:
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
            if not detect_displacement(df4h, idx, avg_body, avg_volume, direction):
                continue
            if pd.isna(avg_range.iloc[idx]) or avg_range.iloc[idx] == 0:
                continue

            row = df4h.iloc[idx]
            atr = avg_range.iloc[idx]
            broken_level = bos["broken_level"]

            if direction == "bullish":
                relevant = [s for s in swing_lows if s["index"] < idx]
            else:
                relevant = [s for s in swing_highs if s["index"] < idx]
            if not relevant:
                continue
            swing_price = relevant[-1]["price"]

            if direction == "bullish":
                swing_to_bos_level = (broken_level - swing_price) / atr
                pre_bos_extension = (row["open"] - swing_price) / atr
                bos_displacement = (row["close"] - row["open"]) / atr
                bos_to_close = (row["close"] - broken_level) / atr
            else:
                swing_to_bos_level = (swing_price - broken_level) / atr
                pre_bos_extension = (swing_price - row["open"]) / atr
                bos_displacement = (row["open"] - row["close"]) / atr
                bos_to_close = (broken_level - row["close"]) / atr

            records.append({
                "symbol": symbol, "direction": direction,
                "swing_to_bos_level": round(swing_to_bos_level, 2),
                "pre_bos_extension": round(pre_bos_extension, 2),
                "bos_displacement_atr": round(bos_displacement, 2),
                "bos_to_close_entry": round(bos_to_close, 2),
            })
        time.sleep(1)

    print(f"\nکل نمونه: {len(records)}\n")
    for r in records:
        print(f"{r['symbol']} | {r['direction']} | Swing->BOSLevel={r['swing_to_bos_level']} | "
              f"PreBOSExt={r['pre_bos_extension']} | BOSDisplacement={r['bos_displacement_atr']} | "
              f"BOSToCloseEntry={r['bos_to_close_entry']}")

    disp_vals = [r["bos_displacement_atr"] for r in records]
    entry_vals = [r["bos_to_close_entry"] for r in records]
    pre_vals = [r["pre_bos_extension"] for r in records]

    if disp_vals:
        print(f"\nBOS Displacement (خود کندل): Mean={sum(disp_vals)/len(disp_vals):.2f} | Max={max(disp_vals):.2f} | Min={min(disp_vals):.2f}")
    if entry_vals:
        print(f"BOS-to-Close Entry Extension: Mean={sum(entry_vals)/len(entry_vals):.2f} | Max={max(entry_vals):.2f} | Min={min(entry_vals):.2f}")
    if pre_vals:
        print(f"Pre-BOS Extension: Mean={sum(pre_vals)/len(pre_vals):.2f} | Max={max(pre_vals):.2f} | Min={min(pre_vals):.2f}")

    print(f"\nنسبت میانگین BOS_Displacement به BOS_to_Close_Entry: "
          f"{(sum(disp_vals)/len(disp_vals)) / (sum(entry_vals)/len(entry_vals)) if entry_vals and sum(entry_vals) else 'N/A':.2f}"
          if disp_vals and entry_vals else "")
    print("=" * 70)
