"""
Elite Crypto Volume Profile — Scan & Analysis Bot (Fully Automated)
--------------------------------------------------------------------
بر اساس پرامپت "ELITE CRYPTO VOLUME PROFILE" ساخته شده، با این تفاوت که
به‌جای دریافت اسکرین‌شات چارت از کاربر، خودش داده قیمت/حجم واقعی رو از
API می‌گیره و Volume Profile / Market Structure رو محاسبه می‌کنه.

⚠️ توجه مهم:
این یک تقریب الگوریتمی از تحلیل Volume Profile حرفه‌ای است، نه جایگزین
کامل نرم‌افزارهایی که به داده tick/order-book دسترسی دارند.

منبع داده: CoinGecko (رایگان، بدون نیاز به کلید API)

نسخه بهینه‌شده (v3 - تعادلی): بین نسخه اولیه (نرخ برد ۲۴.۱٪، خیلی سست) و
نسخه سخت‌گیرانه (تقریباً هیچ سیگنالی تولید نمی‌کرد) یک نقطه تعادل انتخاب شده:
  1. استراتژی روند: امتیاز لازم ۴ از ۵ (نه ۵ کامل)، RVOL=1.25، Zone Quality=0.85
  2. تأیید روند در تایم‌فریم روزانه (حفظ شده)
  3. Mean Reversion: امتیاز لازم ۴ از ۵، RSI=28/72، touches=2

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

SCAN_TOP_N_COINS = 60       # تعداد ارزهایی که در فاز اسکن بررسی میشن
REQUEST_DELAY = 1.5         # فاصله بین درخواست‌ها (رعایت rate limit رایگان CoinGecko)
MIN_RR = 2.0                # حداقل نسبت ریسک به ریوارد قابل قبول


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
    """هر `bucket_size` نقطه قیمتی پیاپی رو به یک کندل OHLCV تبدیل می‌کنه"""
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
    """
    timeframe='hour': کندل چندساعته (مثلاً ۴H یا ۲۴H) از داده ساعتی CoinGecko می‌سازه.
    timeframe='minute': برای نقطه ورود دقیق‌تر، از داده ریزدانه‌تر ۲۴ساعت اخیر
    (~۵ دقیقه‌ای) استفاده می‌کنه و به بازه دلخواه (مثلاً ۱۵دقیقه) می‌سازه.
    """
    if timeframe == "hour":
        hours_needed = limit * aggregate
        days = min(90, max(2, (hours_needed // 24) + 3))
        raw = fetch_market_chart(coin_id, days)
        if raw is None or len(raw) < aggregate * 10:
            return None
        candles = resample_to_ohlcv(raw, aggregate)
    else:
        raw = fetch_market_chart(coin_id, 1)  # ۲۴ساعت اخیر با گرانولاریتی ~۵دقیقه
        if raw is None or len(raw) < 10:
            return None
        bucket = max(1, round(aggregate / 5))
        candles = resample_to_ohlcv(raw, bucket)

    if candles is None or len(candles) < 10:
        return None
    return candles.tail(limit).reset_index(drop=True)


# ============ تشخیص پیوت‌ها (سوئینگ‌های تأییدشده) ============
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


# ============ تشخیص ساختار بازار (HH/HL/LH/LL) ============
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
    """
    برای فیلتر کردن سیگنال‌های خلاف‌جهت روند اصلی، ساختار بازار رو روی
    کندل روزانه (۲۴ساعته) هم می‌سنجیم. سیگنال روند فقط وقتی تأیید میشه
    که جهت ۴H با جهت روزانه هم‌راستا باشه.
    """
    df_daily = get_ohlcv(coin_id, timeframe="hour", aggregate=24, limit=45)
    if df_daily is None or len(df_daily) < 30:
        return None
    structure = classify_structure(df_daily)
    return structure["trend"]


# ============ Volume Profile (POC / HVN / LVN) ============
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


# ============ کیفیت زون (انقباض نوسان = تشکیل زون) ============
def zone_quality_ok(df):
    tr = (df["high"] - df["low"]).abs()
    if len(tr) < 50:
        return False
    atr_short = tr.iloc[-10:].mean()
    atr_long = tr.iloc[-50:].mean()
    if atr_long == 0:
        return False
    return (atr_short / atr_long) < 0.85


# ============ شکار نقدینگی (Liquidity Sweep) ============
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


# ============ RSI (برای Mean Reversion) ============
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============ Bollinger Bands (برای Mean Reversion) ============
def compute_bollinger(df, period=20, num_std=2):
    mid = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


# ============ PART A: امتیازدهی کاندید — استراتژی روند (Trend/Smart Money) ============
def score_trend_candidate(symbol, coin_id, df4h, structure, vp, rvol):
    zq = zone_quality_ok(df4h)
    liq = liquidity_sweep_detected(df4h, structure["pivot_highs"], structure["pivot_lows"])

    current_price = df4h["close"].iloc[-1]
    price_near_poc = abs(current_price - vp["poc"]) / current_price < 0.06

    # فیلتر: تأیید روند در تایم‌فریم روزانه (فقط اگه با ۴H هم‌جهت باشه ادامه می‌دیم)
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

    # نسخه تعادلی: امتیاز ۴ از ۵ کافیه (نه لزوماً ۵ کامل)
    if score >= 4:
        confidence = "بالا" if score == 5 else "متوسط"
    else:
        return None

    return {
        "symbol": symbol, "coin_id": coin_id, "strategy": "trend", "score": score, "confidence": confidence,
        "checks": checks, "trend": structure["trend"], "rvol": rvol,
        "price": current_price, "df4h": df4h, "structure": structure, "vp": vp,
    }


# ============ PART A: امتیازدهی کاندید — استراتژی بازگشت به میانگین (Mean Reversion) ============
def score_mean_reversion_candidate(symbol, coin_id, df4h, structure, vp, rvol):
    # فقط وقتی بازار روند مشخصی نداره (رنج/بی‌جهت) این مسیر رو بررسی می‌کنیم
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

    # تشخیص سوگیری جهت: نزدیک‌تر به کدوم مرز رنج/باند هست
    dist_to_lower = last_close - range_low
    dist_to_upper = range_high - last_close
    direction_bias = "LONG" if dist_to_lower < dist_to_upper else "SHORT"

    # تأیید واقعی بودن رنج: قیمت باید چند بار هر دو مرز رو لمس کرده باشه
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


# ============ انتخاب هدف معتبر (بالاتر/پایین‌تر از قیمت فعلی، نه صرفاً آخرین پیوت) ============
def get_valid_targets(pivots, entry, direction):
    """
    اشکال قبلی: آخرین پیوت تأییدشده معمولاً از قبل توسط قیمت رد شده (چون
    تشخیص پیوت با تأخیر کار می‌کنه). این تابع فقط سطوحی رو به‌عنوان هدف
    برمی‌گردونه که هنوز جلوی قیمته (دست‌نخورده).
    """
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

    # هیچ سطح دست‌نخورده‌ای جلوی قیمت نیست (روند خیلی قوی، از همه سقف/کف‌های
    # قبلی رد شده) — به‌جای رد کردن کامل، از اندازه آخرین موج قیمتی برای
    # پیش‌بینی هدف بعدی استفاده می‌کنیم (روش «Measured Move»)
    if len(pivots) >= 2:
        last_two = sorted(pivots, key=lambda p: p[0])[-2:]
        leg_size = abs(last_two[1][1] - last_two[0][1])
        if leg_size > 0:
            if direction == "LONG":
                return entry + leg_size * 0.5, entry + leg_size * 1.0
            else:
                return entry - leg_size * 0.5, entry - leg_size * 1.0

    return None, None


# ============ PART B: اجرای معامله — مسیر روند (Trend) ============
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

    if candidate["confidence"] == "بالا" and rr >= 3:
        risk_level = "پایین"
    else:
        risk_level = "متوسط"

    return {
        "symbol": symbol, "coin_id": coin_id, "rejected": False, "direction": direction, "strategy": "روند (Smart Money)",
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
        "risk_level": risk_level, "confidence": candidate["confidence"],
        "checks": candidate["checks"],
    }


# ============ PART B: اجرای معامله — مسیر بازگشت به میانگین (Mean Reversion) ============
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

    if candidate["confidence"] == "بالا" and rr >= 3:
        risk_level = "پایین"
    else:
        risk_level = "متوسط"

    return {
        "symbol": symbol, "coin_id": coin_id, "rejected": False, "direction": direction, "strategy": "بازگشت به میانگین (Mean Reversion)",
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
        "risk_level": risk_level, "confidence": candidate["confidence"],
        "checks": candidate["checks"],
    }


def part_b_execute(candidate):
    if candidate["strategy"] == "trend":
        return execute_trend(candidate)
    else:
        return execute_mean_reversion(candidate)


# ============ فرمت خروجی (فارسی، ساختاریافته) ============
def fmt_price(p):
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def format_output(candidates, executions):
    if not candidates:
        return "فاقد ارز مستعد"

    lines = ["<b>PART A — اسکن مارکت</b>"]
    lines.append("وضعیت اسکن: انجام‌شده ✅ (منبع: CoinGecko)\n")
    for c in candidates:
        strategy_label = "روند (Smart Money)" if c["strategy"] == "trend" else "بازگشت به میانگین (Mean Reversion)"
        lines.append(f"<b>{c['symbol']}</b> | استراتژی: {strategy_label} | اعتبار: {c['confidence']}")
        for k, v in c["checks"].items():
            lines.append(f"  {k}: {'✓' if v else '✗'}")
        lines.append("")

    lines.append("<b>PART B — اجرای معامله</b>")
    for ex in executions:
        if ex["rejected"]:
            lines.append(f"\n<b>{ex['symbol']}</b>: عدم وجود موقعیت کم‌ریسک")
            continue
        lines.append(f"\nنماد: <b>{ex['symbol']}</b>")
        lines.append(f"استراتژی: {ex['strategy']}")
        lines.append(f"جهت: {ex['direction']}")
        lines.append(f"ورود دقیق: {fmt_price(ex['entry'])}$")
        lines.append(f"حد ضرر: {fmt_price(ex['sl'])}$")
        lines.append(f"هدف اول: {fmt_price(ex['tp1'])}$")
        lines.append(f"هدف دوم: {fmt_price(ex['tp2'])}$")
        lines.append(f"R:R = 1:{ex['rr']:.2f}")
        lines.append(f"سطح ریسک: {ex['risk_level']}")
        lines.append(f"اعتبار: {ex['confidence']}")

    lines.append(
        "\n⚠️ این سطوح بر اساس تفسیر بصری چارت است و ممکن است دقت قیمتی کامل نداشته باشد. "
        "پیش از اجرا با چارت زنده تطبیق دهید."
    )
    lines.append(
        "⚠️ میزان لوریج باید متناسب با سرمایه و تحمل ریسک شخصی شما تعیین شود؛ "
        "این خروجی توصیه لوریج مشخص ارائه نمی‌دهد."
    )
    return "\n".join(lines)


# ============ لاگ سیگنال‌ها (برای پیگیری خودکار TP/SL) ============
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

    if not candidates:
        message = "فاقد ارز مستعد"
    else:
        print("در حال اجرای تحلیل چارت خودکار (PART B)...")
        executions = []
        for c in candidates:
            try:
                executions.append(part_b_execute(c))
            except Exception as e:
                print(f"خطا در Part B برای {c['symbol']}: {e}")
                executions.append({"symbol": c["symbol"], "rejected": True})
            time.sleep(REQUEST_DELAY)
        message = format_output(candidates, executions)

        for ex in executions:
            if not ex.get("rejected"):
                log_signal(ex)

    print(message)

    should_send = candidates and any(not ex.get("rejected") for ex in executions) if candidates else False

    if should_send:
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
            send_telegram_message(message)
            print("ارسال شد.")
        else:
            print("\n[توجه] توکن تنظیم نشده.")
    else:
        print("سیگنال معتبری برای ارسال به تلگرام وجود نداشت (فقط در لاگ ثبت شد).")


if __name__ == "__main__":
    main()
