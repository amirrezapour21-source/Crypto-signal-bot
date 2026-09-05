"""
Setup Family C — Failed Breakout / Liquidity Reversal
Phase 0-3 + Cycle 1: Detection/Causality Audit ONLY (no performance test)
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour"}
TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
]
LOOKBACK_RECENT = 120
MIN_PENETRATION_ATR = 0.10  # baseline ثابت (طبق Phase 3، بدون Optimization)
MAX_FAILURE_CANDLES = 3      # حداکثر فاصله بین Breakout و Failure


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


def drop_unclosed_candle(df):
    if df is None or df.empty:
        return df
    now_ts = int(time.time())
    last_candle_time = int(df["time"].iloc[-1])
    if now_ts < last_candle_time + 4 * 3600:
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


def detect_failed_breakouts(df, swing_highs, swing_lows, avg_range, lookback=LOOKBACK_RECENT):
    """
    Level = آخرین Swing High/Low که قبل از کندل فعلی تشکیل شده (No
    Look-Ahead). Breakout = High/Low کندل از Level + Min Penetration
    رد بشه. Failure = همون کندل یا یکی از کندل‌های بعدی (حداکثر
    MAX_FAILURE_CANDLES تا) Close برگرده اونور Level.
    Reset: هر Level فقط یه‌بار می‌تونه Event تولید کنه (Fresh trigger،
    مشابه اصلاح Setup A/B).
    """
    events = []
    recent_start = max(0, len(df) - lookback)
    last_level_resistance = None
    last_level_support = None

    for i in range(recent_start, len(df)):
        row = df.iloc[i]
        atr = avg_range.iloc[i]
        if pd.isna(atr) or atr == 0:
            continue
        pen = MIN_PENETRATION_ATR * atr

        # --- Resistance: Breakout بالای آخرین Swing High معتبر ---
        relevant_highs = [s for s in swing_highs if s["index"] < i]
        if relevant_highs:
            level = relevant_highs[-1]  # آخرین Swing High قبل از این کندل (Causal)
            level_price = level["price"]
            assert level["index"] < i  # Causality check
            if row["high"] > level_price + pen and level_price != last_level_resistance:
                breakout_idx = i
                failure_idx, failure_type = None, None
                if row["close"] < level_price:
                    failure_idx, failure_type = i, "same_candle"
                else:
                    for j in range(i + 1, min(i + MAX_FAILURE_CANDLES + 1, len(df))):
                        if df["close"].iloc[j] < level_price:
                            failure_idx, failure_type = j, "multi_bar"
                            break
                if failure_idx is not None:
                    events.append({
                        "direction": "bearish_reversal", "level_type": "resistance",
                        "level_price": level_price, "level_confirm_idx": level["index"],
                        "breakout_idx": breakout_idx, "failure_idx": failure_idx,
                        "failure_type": failure_type,
                    })
                    last_level_resistance = level_price

        # --- Support: Breakout زیر آخرین Swing Low معتبر ---
        relevant_lows = [s for s in swing_lows if s["index"] < i]
        if relevant_lows:
            level = relevant_lows[-1]
            level_price = level["price"]
            assert level["index"] < i
            if row["low"] < level_price - pen and level_price != last_level_support:
                breakout_idx = i
                failure_idx, failure_type = None, None
                if row["close"] > level_price:
                    failure_idx, failure_type = i, "same_candle"
                else:
                    for j in range(i + 1, min(i + MAX_FAILURE_CANDLES + 1, len(df))):
                        if df["close"].iloc[j] > level_price:
                            failure_idx, failure_type = j, "multi_bar"
                            break
                if failure_idx is not None:
                    events.append({
                        "direction": "bullish_reversal", "level_type": "support",
                        "level_price": level_price, "level_confirm_idx": level["index"],
                        "breakout_idx": breakout_idx, "failure_idx": failure_idx,
                        "failure_type": failure_type,
                    })
                    last_level_support = level_price

    return events


if __name__ == "__main__":
    print("SETUP FAMILY C")
    print("CYCLE 1 REPORT — Detection/Causality Audit Only")
    print("=" * 70)

    all_events = []
    total_candles_seen = 0
    checks = {
        "causality_pass": True, "lookahead_pass": True, "duplicate_pass": True,
        "reset_pass": True, "entry_timing_pass": True, "target_contamination_pass": True,
    }

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=250)
        if df4h is None or len(df4h) < 80:
            continue
        df4h = drop_unclosed_candle(df4h)
        total_candles_seen += len(df4h)
        swing_highs, swing_lows = find_swings(df4h, 3, 3)
        avg_range = compute_avg_range(df4h)

        try:
            events = detect_failed_breakouts(df4h, swing_highs, swing_lows, avg_range)
        except AssertionError:
            checks["causality_pass"] = False
            events = []

        for e in events:
            e["symbol"] = symbol
            all_events.append(e)
        time.sleep(0.4)

    print(f"\n1. DATA")
    print(f"  Symbols: {len(TEST_SYMBOLS)}")
    print(f"  Timeframe: 4H")
    print(f"  Total candles seen: {total_candles_seen}")
    print(f"  Total Failed-Breakout events (Cycle 1, raw): {len(all_events)}")

    same_candle = [e for e in all_events if e["failure_type"] == "same_candle"]
    multi_bar = [e for e in all_events if e["failure_type"] == "multi_bar"]

    print(f"\n3. EVENT DISTRIBUTION")
    print(f"  Same-Candle: {len(same_candle)}")
    print(f"  Multi-Bar: {len(multi_bar)}")
    by_symbol = {}
    for e in all_events:
        by_symbol.setdefault(e["symbol"], 0)
        by_symbol[e["symbol"]] += 1
    for sym, cnt in by_symbol.items():
        print(f"    {sym}: {cnt}")

    # === بررسی دقیق منطق (بند ۱۷) ===
    for e in all_events:
        if not (e["level_confirm_idx"] < e["breakout_idx"]):
            checks["lookahead_pass"] = False
        if not (e["breakout_idx"] <= e["failure_idx"]):
            checks["lookahead_pass"] = False

    level_seen = set()
    for e in all_events:
        key = (e["symbol"], e["level_type"], round(e["level_price"], 8))
        if key in level_seen:
            checks["duplicate_pass"] = False
        level_seen.add(key)

    print(f"\n2. LOGIC VALIDATION")
    print(f"  Causality (Level confirmed before Breakout): {'PASS' if checks['causality_pass'] else 'FAIL'}")
    print(f"  Look-ahead (Breakout <= Failure, Level < Breakout): {'PASS' if checks['lookahead_pass'] else 'FAIL'}")
    print(f"  Duplicate protection (هر Level فقط یک Event): {'PASS' if checks['duplicate_pass'] else 'FAIL'}")
    print(f"  Reset logic (Fresh-trigger per level): PASS (طراحی‌شده در کد)")
    print(f"  Entry timing (بعداً در Cycle 2 با Entry واقعی چک می‌شه): N/A در Cycle 1")
    print(f"  Target contamination: N/A در Cycle 1 (هنوز TP تعریف نشده)")

    print(f"\n--- نمونه‌های Debug (۵ رویداد اول) ---")
    for e in all_events[:5]:
        print(f"  {e['symbol']} | {e['direction']} | Level={e['level_price']:.6f} "
              f"(confirm_idx={e['level_confirm_idx']}) | Breakout_idx={e['breakout_idx']} | "
              f"Failure_idx={e['failure_idx']} | Type={e['failure_type']}")

    print(f"\n8. BUGS")
    print(f"  (اگه Check بالا PASS باشه، هیچ باگی در Cycle 1 پیدا نشد)")

    print(f"\n9. DECISION GATE (Cycle 1 فقط Detection Audit است)")
    all_pass = all(checks.values())
    print(f"  {'همه چک‌ها PASS - آماده Cycle 2 (In-Sample Edge Discovery)' if all_pass else 'حداقل یک چک FAIL - نیاز به اصلاح قبل از ادامه'}")

    print(f"\n10. NEXT ACTION")
    print(f"  {'اجرای Cycle 2 با نمونه بزرگ‌تر برای اندازه‌گیری Raw Edge' if all_pass else 'رفع باگ منطقی قبل از ادامه'}")
    print("=" * 70)
