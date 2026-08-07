"""
Crypto Daily Signal Telegram Bot
---------------------------------
هر روز، تاپ ارزها (بر اساس ارزش بازار، از CoinGecko) رو اسکن می‌کنه،
با ترکیب RSI + MACD + Moving Average یه امتیاز به هر ارز میده،
و بهترین سیگنال‌ها (خرید یا فروش) رو به تلگرام ارسال می‌کنه.

نصب پیش‌نیاز:
    pip install requests pandas numpy

تنظیمات:
    توکن و chat_id از متغیرهای محیطی TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID خونده میشه
    (روی GitHub Actions از Secrets تنظیم میشه؛ برای تست دستی می‌تونی مستقیم پایین پر کنی).
"""

import requests
import pandas as pd
import numpy as np
import time
import os

# ============ تنظیمات ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
TOP_N_COINS = 60           # تعداد ارزهایی که اسکن میشن (بر اساس ارزش بازار)
HISTORY_DAYS = 100         # تعداد روز تاریخچه قیمت برای محاسبه اندیکاتورها
TOP_SIGNALS_TO_SEND = 5    # چند تا سیگنال برتر روزانه ارسال بشه
REQUEST_DELAY = 1.6        # فاصله بین درخواست‌ها برای رعایت rate limit رایگان CoinGecko


# ============ درخواست با retry برای rate limit ============
def safe_get(url, params=None, retries=3):
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"Rate limit، صبر {wait} ثانیه...")
            time.sleep(wait)
            continue
        return resp
    return resp


# ============ دریافت لیست ارزها ============
def get_top_coins(n=60):
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": "false",
    }
    resp = safe_get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    # حذف استیبل‌کوین‌ها چون سیگنال معنی‌داری ندارن
    stablecoins = {"tether", "usd-coin", "dai", "true-usd", "first-digital-usd",
                   "usdd", "frax", "paypal-usd", "binance-usd"}
    return [c for c in data if c["id"] not in stablecoins]


# ============ دریافت تاریخچه قیمت ============
def get_price_history(coin_id, days=HISTORY_DAYS):
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    resp = safe_get(url, params=params)
    if resp.status_code != 200:
        return None
    data = resp.json()
    prices = data.get("prices")
    if not prices or len(prices) < 60:
        return None
    close = pd.Series([p[1] for p in prices])
    return close


# ============ اندیکاتورها ============
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_ma(series, period):
    return series.rolling(window=period).mean()


# ============ منطق امتیازدهی سیگنال ============
def analyze_coin(coin):
    coin_id = coin["id"]
    symbol = coin["symbol"].upper()

    close = get_price_history(coin_id)
    if close is None:
        return None

    rsi = compute_rsi(close)
    macd_line, signal_line, hist = compute_macd(close)
    ma20 = compute_ma(close, 20)
    ma50 = compute_ma(close, 50)

    last_rsi = rsi.iloc[-1]
    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    last_ma20 = ma20.iloc[-1]
    last_ma50 = ma50.iloc[-1]
    last_close = close.iloc[-1]

    if any(pd.isna(x) for x in [last_rsi, last_hist, prev_hist, last_ma20, last_ma50]):
        return None

    score = 0
    reasons = []

    if last_rsi < 30:
        score += 2
        reasons.append(f"RSI اشباع فروش ({last_rsi:.1f})")
    elif last_rsi > 70:
        score -= 2
        reasons.append(f"RSI اشباع خرید ({last_rsi:.1f})")

    if prev_hist < 0 and last_hist > 0:
        score += 2
        reasons.append("کراس صعودی MACD")
    elif prev_hist > 0 and last_hist < 0:
        score -= 2
        reasons.append("کراس نزولی MACD")

    if last_close > last_ma20 > last_ma50:
        score += 1
        reasons.append("روند صعودی (قیمت بالای MA20 و MA50)")
    elif last_close < last_ma20 < last_ma50:
        score -= 1
        reasons.append("روند نزولی (قیمت زیر MA20 و MA50)")

    direction = "BUY" if score > 0 else ("SELL" if score < 0 else "NEUTRAL")

    return {
        "symbol": symbol,
        "score": score,
        "direction": direction,
        "price": last_close,
        "rsi": last_rsi,
        "reasons": reasons,
    }


# ============ ارسال به تلگرام ============
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, data=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def format_signal_message(results):
    if not results:
        return "امروز سیگنال قابل‌توجهی در مارکت پیدا نشد."

    lines = ["<b>📊 سیگنال روزانه کریپتو</b>\n"]
    for r in results:
        emoji = "🟢" if r["direction"] == "BUY" else "🔴"
        price_fmt = f"{r['price']:.6f}" if r["price"] < 1 else f"{r['price']:.4f}"
        lines.append(
            f"{emoji} <b>{r['symbol']}</b> — {r['direction']} (امتیاز: {r['score']})\n"
            f"قیمت: {price_fmt}$ | RSI: {r['rsi']:.1f}\n"
            f"دلایل: {', '.join(r['reasons'])}\n"
        )
    return "\n".join(lines)


# ============ اجرای اصلی ============
def main():
    print("در حال دریافت لیست ارزها از CoinGecko...")
    coins = get_top_coins(TOP_N_COINS)
    print(f"در حال تحلیل {len(coins)} ارز...")

    results = []
    for coin in coins:
        try:
            res = analyze_coin(coin)
            if res and res["direction"] != "NEUTRAL":
                results.append(res)
        except Exception as e:
            print(f"خطا در {coin.get('id')}: {e}")
        time.sleep(REQUEST_DELAY)

    results.sort(key=lambda x: abs(x["score"]), reverse=True)
    top_results = results[:TOP_SIGNALS_TO_SEND]

    message = format_signal_message(top_results)
    print(message)

    if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
        send_telegram_message(message)
        print("پیام به تلگرام ارسال شد.")
    else:
        print("\n[توجه] TELEGRAM_BOT_TOKEN تنظیم نشده — پیام فقط چاپ شد، ارسال نشد.")


if __name__ == "__main__":
    main()
