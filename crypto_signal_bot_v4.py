"""
Crypto Signal Bot V4 — Institutional Liquidity Reversal Strategy
--------------------------------------------------------------------
نسخه مستقل و جدید، طبق V4 Final Specification + V4 Directive.

⚠️ این فایل کاملاً جدا از crypto_signal_bot.py (نسخه V3) است.

PHASE 1: Real OHLCV Data Layer (KuCoin) - تکمیل‌شده
PHASE 2: Structure Engine (Swing High/Low, BOS با تأیید Close) - تکمیل‌شده
PHASE 3: Breakout / Continuation Engine
------------------------------------------
طبق بند ۱۱ و ۱۲ Specification: تشخیص حرکت‌های قدرتمند که با Compression
+ Volume Expansion + Structure Break + Displacement تأیید می‌شن - برای
جلوگیری از مشکل قبلی (از دست رفتن فرصت‌هایی مثل حرکت +6% تا +9% XRP).
"""

import requests
import pandas as pd
import numpy as np
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"

KUCOIN_INTERVALS = {
    "4h": "4hour",
    "15m": "15min",
    "1h": "1hour",
    "1d": "1day",
}


def safe_get(url, params=None, retries=3):
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(5)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        return resp
    return resp


def fetch_kucoin_candles(symbol, interval_key, limit=100, end_at=None):
    if interval_key not in KUCOIN_INTERVALS:
        raise ValueError(f"interval نامعتبر: {interval_key}")

    params = {
        "symbol": symbol,
        "type": KUCOIN_INTERVALS[interval_key],
    }
    if end_at:
        params["endAt"] = int(end_at)

    resp = safe_get(KUCOIN_BASE, params=params)
    if resp is None or resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("code") != "200000":
        return None
    raw = data.get("data", [])
    if not raw:
        return None

    rows = []
    for r in raw:
        rows.append({
            "time": int(r[0]),
            "open": float(r[1]),
            "close": float(r[2]),
            "high": float(r[3]),
            "low": float(r[4]),
            "volume": float(r[5]),
        })
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


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
        end_at = df["time"].min() - 1
        time.sleep(0.5)
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
    last_candle_time = df["time"].iloc[-1]
    if now_ts < last_candle_time + interval_seconds.get(interval_key, 0):
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ============================================================
# PHASE 2: STRUCTURE ENGINE
# ============================================================

def find_swings(df, left=3, right=3):
    highs, lows = df["high"].values, df["low"].values
    swing_highs, swing_lows = [], []

    for i in range(left, len(df) - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            swing_highs.append({"index": i, "price": highs[i], "time": df["dt"].iloc[i]})
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swing_lows.append({"index": i, "price": lows[i], "time": df["dt"].iloc[i]})

    return swing_highs, swing_lows


def classify_market_structure(df, left=3, right=3):
    swing_highs, swing_lows = find_swings(df, left, right)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"regime": "range", "swing_highs": swing_highs, "swing_lows": swing_lows}

    last_two_highs = swing_highs[-2:]
    last_two_lows = swing_lows[-2:]

    higher_high = last_two_highs[1]["price"] > last_two_highs[0]["price"]
    higher_low = last_two_lows[1]["price"] > last_two_lows[0]["price"]
    lower_high = last_two_highs[1]["price"] < last_two_highs[0]["price"]
    lower_low = last_two_lows[1]["price"] < last_two_lows[0]["price"]

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
        current_time_idx = i
        close_price = df["close"].iloc[i]

        relevant_highs = [s for s in swing_highs if s["index"] < current_time_idx]
        relevant_lows = [s for s in swing_lows if s["index"] < current_time_idx]

        if relevant_highs:
            last_high = relevant_highs[-1]
            if close_price > last_high["price"]:
                events.append({
                    "type": "bullish_bos",
                    "index": i,
                    "time": df["dt"].iloc[i],
                    "close": close_price,
                    "broken_level": last_high["price"],
                    "broken_level_time": last_high["time"],
                })

        if relevant_lows:
            last_low = relevant_lows[-1]
            if close_price < last_low["price"]:
                events.append({
                    "type": "bearish_bos",
                    "index": i,
                    "time": df["dt"].iloc[i],
                    "close": close_price,
                    "broken_level": last_low["price"],
                    "broken_level_time": last_low["time"],
                })

    return events


# ============================================================
# PHASE 3: BREAKOUT / CONTINUATION ENGINE
# ============================================================

def compute_avg_body(df, lookback=20):
    """میانگین اندازه بدنه کندل‌ها (Body = |Close - Open|) در `lookback` کندل اخیر"""
    body = (df["close"] - df["open"]).abs()
    return body.rolling(window=lookback).mean()


def compute_avg_volume(df, lookback=20):
    """میانگین حجم در `lookback` کندل اخیر (بدون احتساب کندل فعلی)"""
    return df["volume"].rolling(window=lookback).mean()


def detect_compression(df, idx, lookback=20, threshold=0.75):
    """
    تشخیص فشردگی (Compression) قبل از یه کندل مشخص: میانگین دامنه نوسان
    (High-Low) در `lookback` کندل قبل از idx، نسبت به دامنه بلندمدت‌تر.
    اگه نسبت کمتر از threshold باشه، یعنی بازار قبل از این لحظه توی
    یه محدوده فشرده/بی‌حرکت بوده - پیش‌نیاز Breakout واقعی.
    """
    if idx < lookback * 2:
        return False, None
    recent_range = (df["high"] - df["low"]).iloc[idx - lookback:idx].mean()
    longer_range = (df["high"] - df["low"]).iloc[idx - lookback * 2:idx].mean()
    if longer_range == 0:
        return False, None
    ratio = recent_range / longer_range
    return ratio < threshold, ratio


def detect_displacement(df, idx, avg_body, avg_volume, direction,
                          body_multiplier=1.2, volume_multiplier=1.3, close_position_pct=0.25):
    """
    تشخیص کندل Displacement (حرکت قدرتمند): بدنه بزرگ‌تر از میانگین،
    حجم بالاتر از میانگین، و Close نزدیک به یک سر کندل.
    طبق بند ۱۱ Specification.
    """
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
        close_ok = close_position >= (1 - close_position_pct)
    else:
        close_position = (row["high"] - row["close"]) / candle_range
        close_ok = close_position >= (1 - close_position_pct)

    return body_ok and volume_ok and close_ok


def detect_follow_through(df, breakout_idx, direction, candles_after=2):
    """
    بررسی می‌کنه که بعد از کندل Breakout، حرکت واقعاً ادامه پیدا کرده یا
    فوری برگشته (که نشونه Fakeout بوده). طبق بند ۱۲: بعد از Breakout باید
    Confirmation/Continuation باشه.
    """
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return False  # هنوز کندل کافی بعدش نیومده

    breakout_close = df["close"].iloc[breakout_idx]
    after_candles = df.iloc[breakout_idx + 1:end_idx]

    if direction == "bullish":
        # حداقل یکی از کندل‌های بعدی باید بالاتر از close کندل breakout بسته بشه
        return (after_candles["close"] > breakout_close).any()
    else:
        return (after_candles["close"] < breakout_close).any()


def scan_breakout_setups(df, structure, bos_events, lookback_recent=15):
    """
    ترکیب تمام اجزا برای پیدا کردن Setup های معتبر Breakout/Continuation
    در `lookback_recent` کندل اخیر. برای هر BOS اخیر، چک می‌کنه که آیا
    Compression + Displacement + Volume + Follow-through همه تأیید شدن.

    خروجی: لیستی از Setup های Breakout معتبر با جزئیات کامل (برای Missed
    Opportunity Log هم لازمه بدونیم چرا هر کاندید رد شده).
    """
    avg_body = compute_avg_body(df)
    avg_volume = compute_avg_volume(df)
    results = []

    recent_bos = [e for e in bos_events if e["index"] >= len(df) - lookback_recent]

    for bos in recent_bos:
        idx = bos["index"]
        direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

        compression_ok, compression_ratio = detect_compression(df, idx)
        displacement_ok = detect_displacement(df, idx, avg_body, avg_volume, direction)
        follow_through_ok = detect_follow_through(df, idx, direction)

        reasons_failed = []
        if not compression_ok:
            reasons_failed.append("no_compression_before_breakout")
        if not displacement_ok:
            reasons_failed.append("weak_displacement_or_volume")
        if not follow_through_ok:
            reasons_failed.append("no_follow_through")

        valid = compression_ok and displacement_ok and follow_through_ok

        results.append({
            "index": idx,
            "time": bos["time"],
            "direction": direction,
            "close": bos["close"],
            "broken_level": bos["broken_level"],
            "compression_ratio": compression_ratio,
            "compression_ok": compression_ok,
            "displacement_ok": displacement_ok,
            "follow_through_ok": follow_through_ok,
            "valid_breakout": valid,
            "reasons_failed": reasons_failed,
        })

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 TEST — Breakout/Continuation Engine روی BTC-USDT (4H)")
    print("=" * 60)

    df4h = get_ohlcv_v4("BTC-USDT", "4h", total_candles=150)
    if df4h is None:
        print("❌ دریافت داده ناموفق بود")
    else:
        df4h = drop_unclosed_candle(df4h, "4h")
        print(f"✅ {len(df4h)} کندل بسته‌شده دریافت شد\n")

        structure = classify_market_structure(df4h)
        print(f"Market Regime (4H): {structure['regime'].upper()}\n")

        bos_events = detect_bos(df4h, structure["swing_highs"], structure["swing_lows"])
        print(f"تعداد رویداد BOS اخیر: {len(bos_events)}\n")

        print("-" * 60)
        print("بررسی Breakout Setups:")
        print("-" * 60)
        breakout_results = scan_breakout_setups(df4h, structure, bos_events)

        if not breakout_results:
            print("هیچ رویداد BOS اخیری برای بررسی وجود نداشت.")
        for r in breakout_results:
            status = "✅ معتبر" if r["valid_breakout"] else "❌ رد شد"
            print(f"\n{r['time']} | {r['direction']} | close={r['close']:.2f} | {status}")
            print(f"  Compression: {r['compression_ok']} (ratio={r['compression_ratio']})")
            print(f"  Displacement: {r['displacement_ok']}")
            print(f"  Follow-through: {r['follow_through_ok']}")
            if r["reasons_failed"]:
                print(f"  دلایل رد شدن: {', '.join(r['reasons_failed'])}")
