"""
Crypto Signal Bot V4 — Institutional Liquidity Reversal Strategy
--------------------------------------------------------------------
نسخه مستقل و جدید، طبق V4 Final Specification + V4 Directive.

⚠️ این فایل کاملاً جدا از crypto_signal_bot.py (نسخه V3) است.
V3 «کنار گذاشته شده» تلقی می‌شه (طبق Directive) اما فایلش دست‌نخورده
باقی می‌مونه فقط برای آرشیو/مقایسه تاریخی، و منطقش وارد V4 نمی‌شه.

PHASE 1: Real OHLCV Data Layer (KuCoin)
------------------------------------------
منبع داده: KuCoin (تست شد و موفق بود - Binance از GitHub Actions
مسدود است، کد خطا 451 - طبق تست PHASE 0).

PHASE 2: Structure Engine
------------------------------------------
تشخیص مکانیکی Swing High/Low و BOS (Break of Structure) با تأیید
Close کندل (نه فقط Wick - طبق بند ۶ Specification: "Wick-only break
به‌تنهایی BOS معتبر محسوب نشود").
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

    # حذف آخرین کندل اگه هنوز کامل/بسته نشده (طبق Directive بند ۴:
    # "فقط کندل‌های بسته‌شده قابل استفاده باشند")
    return full_df.tail(total_candles).reset_index(drop=True)


def drop_unclosed_candle(df, interval_key):
    """
    آخرین کندل رو اگه هنوز به پایان نرسیده حذف می‌کنه - جلوگیری از
    Look-Ahead Bias (طبق Directive بند ۴).
    """
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
    """
    Swing High/Low مکانیکی: یه کندل Swing High محسوب می‌شه اگه High اون
    از `left` کندل قبل و `right` کندل بعدش بیشتر باشه (و برعکس برای Low).
    خروجی: لیستی از دیکشنری‌ها با ایندکس، قیمت، و زمان.
    """
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
    """
    وضعیت ساختار بازار رو بر پایه توالی Swing High/Low اخیر تشخیص می‌ده:
    'up' (HH+HL), 'down' (LH+LL), یا 'range' (نامشخص).
    این تشخیص فقط بر پایه Structure است، نه یک Indicator تنها
    (طبق بند ۵ Specification).
    """
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
    """
    BOS (Break of Structure) رو با تأیید Close کندل (نه فقط Wick) تشخیص
    می‌ده - طبق بند ۶: "Wick-only break به‌تنهایی BOS معتبر محسوب نشود".

    Bullish BOS: قیمت با Close (نه فقط Wick) از بالای آخرین Swing High
    معتبر رد می‌شه.
    Bearish BOS: قیمت با Close از پایین آخرین Swing Low معتبر رد می‌شه.

    خروجی: لیستی از رویدادهای BOS که در `lookback_candles` اخیر رخ دادن.
    """
    events = []
    recent_start = max(0, len(df) - lookback_candles)

    for i in range(recent_start, len(df)):
        current_time_idx = i
        close_price = df["close"].iloc[i]

        # آخرین Swing High/Low معتبر قبل از این کندل رو پیدا کن
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


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 TEST — Structure Engine روی BTC-USDT (4H)")
    print("=" * 60)

    df4h = get_ohlcv_v4("BTC-USDT", "4h", total_candles=150)
    if df4h is None:
        print("❌ دریافت داده ناموفق بود")
    else:
        df4h = drop_unclosed_candle(df4h, "4h")
        print(f"✅ {len(df4h)} کندل بسته‌شده دریافت شد")
        print(f"از {df4h['dt'].iloc[0]} تا {df4h['dt'].iloc[-1]}\n")

        structure = classify_market_structure(df4h)
        print(f"Market Regime (4H): {structure['regime'].upper()}")
        print(f"تعداد Swing High: {len(structure['swing_highs'])}")
        print(f"تعداد Swing Low: {len(structure['swing_lows'])}\n")

        if structure["swing_highs"]:
            print("۳ Swing High اخیر:")
            for s in structure["swing_highs"][-3:]:
                print(f"  {s['time']} → {s['price']:.2f}")
        if structure["swing_lows"]:
            print("\n۳ Swing Low اخیر:")
            for s in structure["swing_lows"][-3:]:
                print(f"  {s['time']} → {s['price']:.2f}")

        print("\n" + "-" * 60)
        bos_events = detect_bos(df4h, structure["swing_highs"], structure["swing_lows"])
        print(f"تعداد رویداد BOS در ۳۰ کندل اخیر: {len(bos_events)}")
        for e in bos_events[-5:]:
            print(f"  {e['time']} | {e['type']} | close={e['close']:.2f} | سطح شکسته‌شده={e['broken_level']:.2f}")
