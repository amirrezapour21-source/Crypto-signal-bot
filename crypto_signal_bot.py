"""
Crypto Daily Signal Telegram Bot
---------------------------------
هر روز، کل مارکت (تاپ ۱۰۰ ارز بر اساس حجم معاملات از Binance) رو اسکن می‌کنه،
با ترکیب RSI + MACD + Moving Average یه امتیاز به هر ارز میده،
و بهترین سیگنال (خرید یا فروش) رو به تلگرام ارسال می‌کنه.

نصب پیش‌نیاز:
    pip install requests pandas numpy

تنظیمات:
    پایین همین فایل، مقدار TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID رو پر کن.

اجرای روزانه:
    این اسکریپت یک‌بار اجرا میشه و خارج میشه. برای اجرای خودکار روزانه از cron استفاده کن:
        crontab -e
        0 9 * * * /usr/bin/python3 /path/to/crypto_signal_bot.py
    (نمونه بالا هر روز ساعت ۹ صبح اجرا میشه)
"""

import requests
import pandas as pd
import numpy as np
import time

# ============ تنظیمات ============
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"   # از @BotFather بگیر
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"        # آیدی عددی چت یا کانال

BINANCE_BASE = "https://api.binance.com"
TOP_N_COINS = 100          # تعداد ارزهایی که اسکن میشن
KLINE_INTERVAL = "4h"      # تایم‌فریم کندل‌ها
KLINE_LIMIT = 200          # تعداد کندل برای محاسبه اندیکاتورها
TOP_SIGNALS_TO_SEND = 3    # چند تا سیگنال برتر روزانه ارسال بشه


# ============ دریافت لیست ارزها ============
def get_top_symbols(n=100):
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    usdt_pairs = [
        d for d in data
        if d["symbol"].endswith("USDT")
        and not any(x in d["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])
    ]
    usdt_pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [d["symbol"] for d in usdt_pairs[:n]]


# ============ دریافت کندل‌ها ============
def get_klines(symbol, interval=KLINE_INTERVAL, limit=KLINE_LIMIT):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df


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
def analyze_symbol(symbol):
    df = get_klines(symbol)
    if df is None or len(df) < 60:
        return None

    close = df["close"]
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

    # RSI
    if last_rsi < 30:
        score += 2
        reasons.append(f"RSI اشباع فروش ({last_rsi:.1f})")
    elif last_rsi > 70:
        score -= 2
        reasons.append(f"RSI اشباع خرید ({last_rsi:.1f})")

    # MACD کراس
    if prev_hist < 0 and last_hist > 0:
        score += 2
        reasons.append("کراس صعودی MACD")
    elif prev_hist > 0 and last_hist < 0:
        score -= 2
        reasons.append("کراس نزولی MACD")

    # روند میانگین متحرک
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
        lines.append(
            f"{emoji} <b>{r['symbol']}</b> — {r['direction']} (امتیاز: {r['score']})\n"
            f"قیمت: {r['price']:.4f} | RSI: {r['rsi']:.1f}\n"
            f"دلایل: {', '.join(r['reasons'])}\n"
        )
    return "\n".join(lines)


# ============ اجرای اصلی ============
def main():
    print("در حال دریافت لیست ارزها...")
    symbols = get_top_symbols(TOP_N_COINS)

    print(f"در حال تحلیل {len(symbols)} ارز...")
    results = []
    for sym in symbols:
        try:
            res = analyze_symbol(sym)
            if res and res["direction"] != "NEUTRAL":
                results.append(res)
        except Exception as e:
            print(f"خطا در {sym}: {e}")
        time.sleep(0.15)  # رعایت rate limit بایننس

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
