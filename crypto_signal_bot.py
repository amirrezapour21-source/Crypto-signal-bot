"""
Elite Crypto Volume Profile — Scan & Analysis Bot (Fully Automated)
--------------------------------------------------------------------
بر اساس پرامپت "ELITE CRYPTO VOLUME PROFILE" ساخته شده، با این تفاوت که
به‌جای دریافت اسکرین‌شات چارت از کاربر، خودش داده قیمت/حجم واقعی رو از
API می‌گیره و Volume Profile / Market Structure رو محاسبه می‌کنه.

منبع داده اصلی: CoinGecko (رایگان، بدون نیاز به کلید API)
منبع اعتبارسنجی قیمت لحظه‌ای: KuCoin

نسخه v8 - اصلاح منطق ضد تکرار سیگنال: قبلاً has_recent_duplicate فقط
سیگنال‌های باز در ۲۴ ساعت اخیر رو چک می‌کرد که یه باگ واقعی داشت -
سیگنال‌هایی که بیشتر از ۲۴ ساعت باز مونده بودن (مثل PAXG) دیگه محافظت
نمی‌شدن. حالا این تابع صرف‌نظر از زمان، هر سیگنال باز برای همون
نماد+جهت رو تشخیص می‌ده و از صدور سیگنال تکراری جلوگیری می‌کنه.

نصب پیش‌نیاز:
    pip install requests pandas numpy

تنظیمات: TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID از متغیر محیطی خونده میشه.
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime, timezone

# ============ تنظیمات ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

SCAN_TOP_N_COINS = 60
REQUEST_DELAY = 1.5
MIN_RR = 2.0
PRICE_DEVIATION_THRESHOLD = 0.01


# ============ ابزار درخواست امن ============
def safe_get(url, params=None, headers=None, retries=3):
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
        except requests.RequestException:
            time.sleep(5)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        return resp
    return resp


# ============ قیمت لحظه‌ای واقعی (KuCoin) ============
def get_live_price_kucoin(symbol):
    try:
        url = "https://api.kucoin.com/api/v1/market/orderbook/level1"
        params = {"symbol": f"{symbol}-USDT"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("code") != "200000":
            return None
        price = data.get("data", {}).get("price")
        return float(price) if price else None
    except Exception:
        return None


def validate_and_adjust_prices(symbol, entry, sl, tp1, tp2, direction):
    live_price = get_live_price_kucoin(symbol)
    if live_price is None or entry == 0:
        return entry, sl, tp1, tp2, False, None

    deviation = abs(live_price - entry) / entry
    if deviation <= PRICE_DEVIATION_THRESHOLD:
        return entry, sl, tp1, tp2, False, live_price

    shift = live_price - entry
    new_entry = live_price
    new_sl = sl + shift
    new_tp1 = tp1 + shift
    new_tp2 = tp2 + shift
    return new_entry, new_sl, new_tp1, new_tp2, True, live_price


# ============ فیلتر ضد تکرار سیگنال ============
def has_recent_duplicate(symbol, direction):
    """
    اگه برای این نماد/جهت یه سیگنال باز (ENTRY_PENDING/OPEN/TP1_HIT) وجود
    داشته باشه، سیگنال جدید رد می‌شه - صرف‌نظر از اینکه چقدر پیش صادر شده.
    (قبلاً این محدودیت فقط ۲۴ ساعت بود که یه باگ واقعی داشت: سیگنال‌هایی
    که بیشتر از ۲۴ ساعت باز مونده بودن، مثل PAXG، دیگه محافظت نمی‌شدن.)
    """
    log = load_log()
    open_statuses = ("ENTRY_PENDING", "OPEN", "TP1_HIT")
    for s in log:
        if s.get("symbol") == symbol and s.get("direction") == direction:
            if s.get("status") in open_statuses:
                return True
    return False


# ============ دریافت لیست ارزها (CoinGecko) ============
def get_top_coins(n):
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd", "order": "market_cap_desc",
        "per_page": n, "page": 1, "sparkline": "false",
    }
    resp = safe_get(url, params=params)
    if resp is None or resp.status_code != 200:
        return []
    data = resp.json()

    def is_stablecoin(c):
        price = c.get("current_price") or 0
        chg = c.get("price_change_percentage_24h") or 0
        return 0.95 <= price <= 1.05 and abs(chg) < 0.5

    return [c for c in data if not is_stablecoin(c)]


# ============ دریافت سری قیمت/حجم و ساخت کندل مصنوعی (CoinGecko) ============
def fetch_market_chart(coin_id, days):
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    resp = safe_get(url, params=params)
    if resp is None or resp.status_code != 200:
        return None
    data = resp.json()
    prices = data.get("prices")
    volumes = data.get("total_volumes")
    if not prices or not volumes:
        return None
    df = pd.DataFrame({"time": [p[0] // 1000 for p in prices], "price": [p[1] for p in prices]})
    vol_df = pd.DataFrame({"time": [v[0] // 1000 for v in volumes], "volume": [v[1] for v in volumes]})
    df = df.merge(vol_df, on="time", how="left").sort_values("time").reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0)
    return df


def resample_to_ohlcv(df, bucket_size):
    n = len(df) // bucket_size
    if n < 1:
        return None
    rows = []
    for i in range(n):
        chunk = df.iloc[i * bucket_size:(i + 1) * bucket_size]
        rows.append({
            "time": chunk["time"].iloc[-1],
            "open": chunk["price"].iloc[0],
            "high": chunk["price"].max(),
            "low": chunk["price"].min(),
            "close": chunk["price"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
    return pd.DataFrame(rows)


def get_ohlcv(coin_id, timeframe="hour", aggregate=4, limit=150):
    if timeframe == "hour":
        hours_needed = limit * aggregate
        days = min(90, max(2, (hours_needed // 24) + 3))
        raw = fetch_market_chart(coin_id, days)
        if raw is None or len(raw) < aggregate * 10:
            return None
        candles = resample_to_ohlcv(raw, aggregate)
    else:
        raw = fetch_market_chart(coin_id, 1)
        if raw is None or len(raw) < 10:
            return None
        bucket = max(1, round(aggregate / 5))
        candles = resample_to_ohlcv(raw, bucket)

    if candles is None or len(candles) < 10:
        return None
    return candles.tail(limit).reset_index(drop=True)


# ============ تشخیص پیوت‌ها ============
def find_pivots(df, left=2, right=2):
    highs, lows = df["high"].values, df["low"].values
    pivot_highs, pivot_lows = [], []
    for i in range(left, len(df) - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            pivot_highs.append((i, highs[i]))
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            pivot_lows.append((i, lows[i]))
    return pivot_highs, pivot_lows


# ============ تشخیص ساختار بازار ============
def classify_structure(df):
    pivot_highs, pivot_lows = find_pivots(df)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return {"trend": "choppy", "pivot_highs": pivot_highs, "pivot_lows": pivot_lows}

    last_two_highs = pivot_highs[-2:]
    last_two_lows = pivot_lows[-2:]
    higher_high = last_two_highs[1][1] > last_two_highs[0][1]
    higher_low = last_two_lows[1][1] > last_two_lows[0][1]
    lower_high = last_two_highs[1][1] < last_two_highs[0][1]
    lower_low = last_two_lows[1][1] < last_two_lows[0][1]

    if higher_high and higher_low:
        trend = "up"
    elif lower_high and lower_low:
        trend = "down"
    else:
        trend = "choppy"

    return {"trend": trend, "pivot_highs": pivot_highs, "pivot_lows": pivot_lows}


# ============ تأیید روند در تایم‌فریم بالاتر (روزانه) ============
def get_daily_trend_bias(coin_id):
    df_daily = get_ohlcv(coin_id, timeframe="hour", aggregate=24, limit=45)
    if df_daily is None or len(df_daily) < 30:
        return None
    structure = classify_structure(df_daily)
    return structure["trend"]


# ============ Volume Profile ============
def compute_volume_profile(df, bins=24):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    lo, hi = df["low"].min(), df["high"].max()
    if hi <= lo:
        return None
    bin_edges = np.linspace(lo, hi, bins + 1)
    bin_idx = np.clip(np.digitize(typical_price, bin_edges) - 1, 0, bins - 1)
    vol_per_bin = np.zeros(bins)
    for idx, vol in zip(bin_idx, df["volume"].values):
        vol_per_bin[idx] += vol
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    poc_i = int(np.argmax(vol_per_bin))
    poc_price = bin_centers[poc_i]
    threshold_hvn = np.percentile(vol_per_bin, 70)
    hvn_prices = bin_centers[vol_per_bin >= threshold_hvn]
    return {"poc": poc_price, "hvn": hvn_prices, "bin_centers": bin_centers, "vol_per_bin": vol_per_bin}


# ============ RVOL ============
def compute_rvol(df, lookback=20):
    if len(df) < lookback + 1:
        return 1.0
    recent = df["volume"].iloc[-1]
    avg = df["volume"].iloc[-lookback - 1:-1].mean()
    if avg == 0:
        return 1.0
    return recent / avg


# ============ کیفیت زون ============
def zone_quality_ok(df):
    tr = (df["high"] - df["low"]).abs()
    if len(tr) < 50:
        return False
    atr_short = tr.iloc[-10:].mean()
    atr_long = tr.iloc[-50:].mean()
    if atr_long == 0:
        return False
    return (atr_short / atr_long) < 0.85


# ============ شکار نقدینگی ============
def liquidity_sweep_detected(df, pivot_highs, pivot_lows):
    if len(df) < 10:
        return False
    recent = df.iloc[-10:]
    if pivot_highs:
        last_high = pivot_highs[-1][1]
        for _, row in recent.iterrows():
            if row["high"] > last_high and row["close"] < last_high:
                return True
    if pivot_lows:
        last_low = pivot_lows[-1][1]
        for _, row in recent.iterrows():
            if row["low"] < last_low and row["close"] > last_low:
                return True
    return False


# ============ RSI ============
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============ Bollinger Bands ============
def compute_bollinger(df, period=20, num_std=2):
    mid = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


# ============ PART A: امتیازدهی — روند ============
def score_trend_candidate(symbol, coin_id, df4h, structure, vp, rvol):
    zq = zone_quality_ok(df4h)
    liq = liquidity_sweep_detected(df4h, structure["pivot_highs"], structure["pivot_lows"])

    current_price = df4h["close"].iloc[-1]
    price_near_poc = abs(current_price - vp["poc"]) / current_price < 0.06

    daily_trend = get_daily_trend_bias(coin_id)
    daily_confirmed = daily_trend is not None and daily_trend == structure["trend"]
    if not daily_confirmed:
        return None

    checks = {
        "Volume Profile": price_near_poc,
        "Volume/RVOL": rvol > 1.25,
        "Zone Quality": zq,
        "Structure Clarity": structure["trend"] in ("up", "down"),
        "Liquidity Context": liq,
    }
    score = sum(checks.values())

    if score >= 4:
        confidence = "بالا" if score == 5 else "متوسط"
    else:
        return None

    return {
        "symbol": symbol, "coin_id": coin_id, "strategy": "trend", "score": score, "confidence": confidence,
        "checks": checks, "trend": structure["trend"], "rvol": rvol,
        "price": current_price, "df4h": df4h, "structure": structure, "vp": vp,
    }


# ============ PART A: امتیازدهی — Mean Reversion ============
def score_mean_reversion_candidate(symbol, coin_id, df4h, structure, vp, rvol):
    if structure["trend"] in ("up", "down"):
        return None
    if len(df4h) < 45:
        return None

    lookback = df4h.iloc[-40:]
    range_high = lookback["high"].max()
    range_low = lookback["low"].min()
    if range_high <= range_low:
        return None

    mid_bb, upper_bb, lower_bb = compute_bollinger(df4h)
    last_mid = mid_bb.iloc[-1]
    last_upper = upper_bb.iloc[-1]
    last_lower = lower_bb.iloc[-1]
    if any(pd.isna(x) for x in [last_mid, last_upper, last_lower]):
        return None

    rsi = compute_rsi(df4h["close"])
    last_rsi = rsi.iloc[-1]
    if pd.isna(last_rsi):
        return None

    last_close = df4h["close"].iloc[-1]
    last_low = df4h["low"].iloc[-1]
    last_high = df4h["high"].iloc[-1]

    dist_to_lower = last_close - range_low
    dist_to_upper = range_high - last_close
    direction_bias = "LONG" if dist_to_lower < dist_to_upper else "SHORT"

    touches_high = (lookback["high"] >= range_high * 0.99).sum()
    touches_low = (lookback["low"] <= range_low * 1.01).sum()
    range_confirmed = touches_high >= 2 and touches_low >= 2

    if direction_bias == "LONG":
        rsi_extreme = last_rsi < 28
        bb_touch = last_low <= last_lower
        reversal = last_low <= last_lower and last_close > last_lower
    else:
        rsi_extreme = last_rsi > 72
        bb_touch = last_high >= last_upper
        reversal = last_high >= last_upper and last_close < last_upper

    checks = {
        "Range Confirmed": range_confirmed,
        "RSI Extreme": rsi_extreme,
        "Volume Confirmation": rvol > 1.3,
        "Bollinger Band Touch": bb_touch,
        "Reversal Rejection": reversal,
    }
    score = sum(checks.values())

    if score >= 4:
        confidence = "بالا" if score == 5 else "متوسط"
    else:
        return None

    return {
        "symbol": symbol, "coin_id": coin_id, "strategy": "mean_reversion", "score": score, "confidence": confidence,
        "checks": checks, "direction_bias": direction_bias, "rvol": rvol,
        "price": last_close, "df4h": df4h, "range_high": range_high, "range_low": range_low,
        "range_mid": (range_high + range_low) / 2, "vp": vp,
    }


def evaluate_symbol(coin):
    symbol = coin["symbol"].upper()
    coin_id = coin["id"]
    df4h = get_ohlcv(coin_id, timeframe="hour", aggregate=4, limit=150)
    if df4h is None or len(df4h) < 60:
        return None
    structure = classify_structure(df4h)
    vp = compute_volume_profile(df4h)
    if vp is None:
        return None
    rvol = compute_rvol(df4h)

    if structure["trend"] in ("up", "down"):
        return score_trend_candidate(symbol, coin_id, df4h, structure, vp, rvol)
    else:
        return score_mean_reversion_candidate(symbol, coin_id, df4h, structure, vp, rvol)


def part_a_scan(coins):
    candidates = []
    for c in coins:
        symbol = c["symbol"].upper()
        try:
            res = evaluate_symbol(c)
            if res:
                candidates.append(res)
        except Exception as e:
            print(f"خطا در اسکن {symbol}: {e}")
        time.sleep(REQUEST_DELAY)
    candidates.sort(key=lambda x: (x["score"], x["rvol"]), reverse=True)
    return candidates[:2]


# ============ انتخاب هدف معتبر ============
def get_valid_targets(pivots, entry, direction):
    if direction == "LONG":
        valid = sorted([p[1] for p in pivots if p[1] > entry])
    else:
        valid = sorted([p[1] for p in pivots if p[1] < entry], reverse=True)

    if len(valid) >= 2:
        return valid[0], valid[1]
    if len(valid) == 1:
        tp1 = valid[0]
        tp2 = tp1 * 1.02 if direction == "LONG" else tp1 * 0.98
        return tp1, tp2

    if len(pivots) >= 2:
        last_two = sorted(pivots, key=lambda p: p[0])[-2:]
        leg_size = abs(last_two[1][1] - last_two[0][1])
        if leg_size > 0:
            if direction == "LONG":
                return entry + leg_size * 0.5, entry + leg_size * 1.0
            else:
                return entry - leg_size * 0.5, entry - leg_size * 1.0

    return None, None


# ============ PART B: اجرای معامله — روند ============
def execute_trend(candidate):
    symbol = candidate["symbol"]
    coin_id = candidate["coin_id"]
    trend = candidate["trend"]
    structure = candidate["structure"]

    df15m = get_ohlcv(coin_id, timeframe="minute", aggregate=15, limit=90)
    if df15m is None or len(df15m) < 20:
        return {"symbol": symbol, "rejected": True}

    vp15 = compute_volume_profile(df15m.iloc[-20:])
    if vp15 is None:
        return {"symbol": symbol, "rejected": True}

    pivot_highs, pivot_lows = structure["pivot_highs"], structure["pivot_lows"]

    if trend == "up" and len(pivot_lows) >= 2 and len(pivot_highs) >= 1:
        direction = "LONG"
        entry = vp15["poc"]
        invalidation = pivot_lows[-1][1]
        sl = invalidation * 0.997
        tp1, tp2 = get_valid_targets(pivot_highs, entry, "LONG")
        if tp1 is None:
            return {"symbol": symbol, "rejected": True}
        risk = entry - sl
        reward1 = tp1 - entry
        if risk <= 0 or reward1 <= 0:
            return {"symbol": symbol, "rejected": True}
        rr = reward1 / risk

    elif trend == "down" and len(pivot_highs) >= 2 and len(pivot_lows) >= 1:
        direction = "SHORT"
        entry = vp15["poc"]
        invalidation = pivot_highs[-1][1]
        sl = invalidation * 1.003
        tp1, tp2 = get_valid_targets(pivot_lows, entry, "SHORT")
        if tp1 is None:
            return {"symbol": symbol, "rejected": True}
        risk = sl - entry
        reward1 = entry - tp1
        if risk <= 0 or reward1 <= 0:
            return {"symbol": symbol, "rejected": True}
        rr = reward1 / risk
    else:
        return {"symbol": symbol, "rejected": True}

    if rr < MIN_RR:
        return {"symbol": symbol, "rejected": True}

    if has_recent_duplicate(symbol, direction):
        return {"symbol": symbol, "rejected": True, "duplicate": True}

    entry, sl, tp1, tp2, adjusted, live_price = validate_and_adjust_prices(
        symbol, entry, sl, tp1, tp2, direction
    )
    risk = abs(entry - sl)
    reward1 = abs(tp1 - entry)
    if risk <= 0 or reward1 <= 0:
        return {"symbol": symbol, "rejected": True}
    rr = reward1 / risk
    if rr < MIN_RR:
        return {"symbol": symbol, "rejected": True}

    risk_level = "LOW" if (candidate["confidence"] == "بالا" and rr >= 3) else "MEDIUM"

    return {
        "symbol": symbol, "coin_id": coin_id, "rejected": False, "direction": direction,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
        "risk_level": risk_level, "confidence": candidate["confidence"],
        "price_adjusted": adjusted, "live_price": live_price,
        "strategy_type": "Trend + Volume Profile + Liquidity",
    }


# ============ PART B: اجرای معامله — Mean Reversion ============
def execute_mean_reversion(candidate):
    symbol = candidate["symbol"]
    coin_id = candidate["coin_id"]
    direction_bias = candidate["direction_bias"]
    range_high, range_low, range_mid = candidate["range_high"], candidate["range_low"], candidate["range_mid"]

    df15m = get_ohlcv(coin_id, timeframe="minute", aggregate=15, limit=90)
    if df15m is None or len(df15m) < 20:
        return {"symbol": symbol, "rejected": True}

    vp15 = compute_volume_profile(df15m.iloc[-20:])
    if vp15 is None:
        return {"symbol": symbol, "rejected": True}

    entry = vp15["poc"]

    if direction_bias == "LONG":
        direction = "LONG"
        sl = range_low * 0.99
        tp1 = range_mid
        tp2 = range_high
        risk = entry - sl
        reward1 = tp1 - entry
    else:
        direction = "SHORT"
        sl = range_high * 1.01
        tp1 = range_mid
        tp2 = range_low
        risk = sl - entry
        reward1 = entry - tp1

    if risk <= 0 or reward1 <= 0:
        return {"symbol": symbol, "rejected": True}
    rr = reward1 / risk

    if rr < MIN_RR:
        return {"symbol": symbol, "rejected": True}

    if has_recent_duplicate(symbol, direction):
        return {"symbol": symbol, "rejected": True, "duplicate": True}

    entry, sl, tp1, tp2, adjusted, live_price = validate_and_adjust_prices(
        symbol, entry, sl, tp1, tp2, direction
    )
    risk = abs(entry - sl)
    reward1 = abs(tp1 - entry)
    if risk <= 0 or reward1 <= 0:
        return {"symbol": symbol, "rejected": True}
    rr = reward1 / risk
    if rr < MIN_RR:
        return {"symbol": symbol, "rejected": True}

    risk_level = "LOW" if (candidate["confidence"] == "بالا" and rr >= 3) else "MEDIUM"

    return {
        "symbol": symbol, "coin_id": coin_id, "rejected": False, "direction": direction,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
        "risk_level": risk_level, "confidence": candidate["confidence"],
        "price_adjusted": adjusted, "live_price": live_price,
        "strategy_type": "Mean Reversion",
    }


def part_b_execute(candidate):
    if candidate["strategy"] == "trend":
        return execute_trend(candidate)
    else:
        return execute_mean_reversion(candidate)


# ============ فرمت خروجی — تمیز و ساده (برای اسکرین‌شات) ============
def fmt_price(p):
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def format_single_signal(ex):
    direction_emoji = "🟢" if ex["direction"] == "LONG" else "🔴"
    lines = [
        f"📊 {ex['symbol']}/USDT — SIGNAL",
        "",
        f"{direction_emoji} Direction: {ex['direction']}",
        f"📈 Strategy: {ex.get('strategy_type', '-')}",
        "",
        "📍 Entry",
        f"{fmt_price(ex['entry'])}",
        "",
        "🛑 Stop Loss",
        f"{fmt_price(ex['sl'])}",
        "",
        "🎯 Take Profit",
        f"TP1 → {fmt_price(ex['tp1'])}",
        f"TP2 → {fmt_price(ex['tp2'])}",
        "",
        "⚖️ Risk/Reward",
        f"1 : {ex['rr']:.2f}",
        "",
        "⚠️ Risk Level",
        f"{ex['risk_level']}",
        "",
        "🏦 Exchange",
        "KuCoin" if ex.get("live_price") else "CoinGecko",
    ]
    return "\n".join(lines)


def format_output(candidates, executions):
    if not candidates:
        return None

    messages = []
    for ex in executions:
        if ex.get("rejected"):
            continue
        messages.append(format_single_signal(ex))

    if not messages:
        return None

    return messages


# ============ لاگ سیگنال‌ها ============
LOG_PATH = "signals_log.json"


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_log(data):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_signal(ex):
    log = load_log()
    entry_id = f"{ex['symbol']}_{int(time.time())}"
    log.append({
        "id": entry_id,
        "symbol": ex["symbol"],
        "coin_id": ex.get("coin_id", ""),
        "direction": ex["direction"],
        "strategy_type": ex.get("strategy_type", ""),
        "entry": ex["entry"],
        "sl": ex["sl"],
        "tp1": ex["tp1"],
        "tp2": ex["tp2"],
        "signal_time": datetime.now(timezone.utc).isoformat(),
        "status": "ENTRY_PENDING",
        "entered": False,
        "entry_time": None,
        "closed_time": None,
        "result": None,
    })
    save_log(log)


# ============ تلگرام ============
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    resp = requests.post(url, data=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ============ اجرای اصلی ============
def main():
    print("در حال دریافت لیست ارزها...")
    coins = get_top_coins(SCAN_TOP_N_COINS)
    print(f"در حال اسکن {len(coins)} ارز (PART A)...")

    candidates = part_a_scan(coins)

    executions = []
    if candidates:
        print("در حال اجرای تحلیل چارت خودکار (PART B)...")
        for c in candidates:
            try:
                executions.append(part_b_execute(c))
            except Exception as e:
                print(f"خطا در Part B برای {c['symbol']}: {e}")
                executions.append({"symbol": c["symbol"], "rejected": True})
            time.sleep(REQUEST_DELAY)

        for ex in executions:
            if not ex.get("rejected"):
                log_signal(ex)

    messages = format_output(candidates, executions)

    if messages:
        for msg in messages:
            print(msg)
            print("-" * 30)
            if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
                send_telegram_message(msg)
        print("ارسال شد.")
    else:
        print("سیگنال معتبری برای ارسال به تلگرام وجود نداشت.")


if __name__ == "__main__":
    main()
