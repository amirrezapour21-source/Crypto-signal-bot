"""
Crypto Signal Bot V4 — Institutional Liquidity Reversal Strategy
--------------------------------------------------------------------
نسخه مستقل و جدید، طبق V4 Final Specification + Directive + Breakout Patch.

⚠️ این فایل کاملاً جدا از crypto_signal_bot.py (نسخه V3) است.

PHASE 1: Real OHLCV Data Layer (KuCoin) - تکمیل‌شده
PHASE 2: Structure Engine (Swing High/Low, BOS با تأیید Close) - تکمیل‌شده
PHASE 3: Breakout / Continuation Engine با دو مسیر مستقل (PATH A/B) - تکمیل‌شده

این نسخه چند نماد رو به‌جای فقط BTC اسکن می‌کنه تا ببینیم آیا سیستم
اصلاً می‌تونه Setup معتبر (نه فقط Reject) پیدا کنه یا نه.
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

# لیست نمادهای تست (چند تا از ارزهای برتر با نقدشوندگی بالا روی KuCoin)
TEST_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
                "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT"]


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
    body = (df["close"] - df["open"]).abs()
    return body.rolling(window=lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(window=lookback).mean()


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(window=lookback).mean()


def detect_compression(df, idx, lookback=20, threshold=0.75):
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
    if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]) or avg_body.iloc[idx] == 0:
        return False

    row = df.iloc[idx]
    body_size = abs(row["close"] - row["open"])
    candle_range =
