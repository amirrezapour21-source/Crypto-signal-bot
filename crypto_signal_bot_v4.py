"""
Crypto Signal Bot V4 — Institutional Liquidity Reversal Strategy
--------------------------------------------------------------------
نسخه مستقل و جدید، طبق V4 Master Strategy Specification.

⚠️ این فایل کاملاً جدا از crypto_signal_bot.py (نسخه V3) است و
جایگزین آن نمی‌شود. V3 دست‌نخورده باقی می‌ماند.

PHASE 1: Real OHLCV Data Layer (KuCoin)
------------------------------------------
منبع داده: KuCoin (تست شد و موفق بود - Binance از GitHub Actions
مسدود است، کد خطا 451).

این بخش فقط توابع پایه دریافت کندل واقعی رو داره. منطق استراتژی
(Regime, Sweep, MSS/BOS, ...) در فازهای بعدی اضافه می‌شه.
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"

# KuCoin این بازه‌های زمانی رو پشتیبانی می‌کنه
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
    """
    یک بار درخواست به KuCoin می‌زنه و حداکثر ۱۰۰ کندل برمی‌گردونه.
    symbol: مثل 'BTC-USDT'
    interval_key: یکی از کلیدهای KUCOIN_INTERVALS مثل '4h' یا '15m'
    end_at: timestamp (ثانیه) برای گرفتن کندل‌های قبل از این زمان (صفحه‌بندی)
    """
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

    # فرمت هر ردیف KuCoin: [time, open, close, high, low, volume, turnover]
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
    """
    با صفحه‌بندی (چند بار درخواست پشت‌سرهم)، تعداد کندل بیشتر از ۱۰۰ تا
    رو از KuCoin جمع می‌کنه. برای بک‌تست و تحلیل ۴H/15M استفاده می‌شه.
    """
    all_dfs = []
    end_at = None
    remaining = total_candles

    while remaining > 0:
        df = fetch_kucoin_candles(symbol, interval_key, end_at=end_at)
        if df is None or df.empty:
            break
        all_dfs.append(df)
        remaining -= len(df)
        # برای صفحه بعد، از قدیمی‌ترین timestamp این دسته به عقب بریم
        end_at = df["time"].min() - 1
        time.sleep(0.5)  # رعایت rate limit
        if len(df) < 100:
            # یعنی دیگه داده قدیمی‌تری نیست
            break

    if not all_dfs:
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    full_df["dt"] = pd.to_datetime(full_df["time"], unit="s", utc=True)
    return full_df.tail(total_candles).reset_index(drop=True)


if __name__ == "__main__":
    # تست دستی سریع - این بخش فقط برای بررسی صحت کارکرد Data Layer است
    print("در حال تست دریافت ۱۵۰ کندل ۴ساعته BTC-USDT از KuCoin...")
    df4h = get_ohlcv_v4("BTC-USDT", "4h", total_candles=150)
    if df4h is not None:
        print(f"✅ موفق - {len(df4h)} کندل دریافت شد")
        print(f"از {df4h['dt'].iloc[0]} تا {df4h['dt'].iloc[-1]}")
        print(f"نمونه آخرین کندل:\n{df4h.iloc[-1]}")
    else:
        print("❌ ناموفق")

    print("\nدر حال تست دریافت ۱۵۰ کندل ۱۵دقیقه‌ای BTC-USDT از KuCoin...")
    df15m = get_ohlcv_v4("BTC-USDT", "15m", total_candles=150)
    if df15m is not None:
        print(f"✅ موفق - {len(df15m)} کندل دریافت شد")
        print(f"از {df15m['dt'].iloc[0]} تا {df15m['dt'].iloc[-1]}")
    else:
        print("❌ ناموفق")
