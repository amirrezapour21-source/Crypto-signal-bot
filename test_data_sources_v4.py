"""
PHASE 0 — تست دسترسی به داده تاریخی (Historical OHLCV) واقعی صرافی
برای پروژه V4. هدف: مشخص کردن اینکه KuCoin یا Binance، کدوم از
GitHub Actions به کندل‌های ۴ساعته و ۱۵دقیقه‌ای واقعی دسترسی داره.
"""
import requests
import time


def test_kucoin_historical():
    print("=" * 50)
    print("تست KuCoin — کندل تاریخی ۴ساعته BTC-USDT")
    print("=" * 50)
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {
        "symbol": "BTC-USDT",
        "type": "4hour",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"کد وضعیت: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "200000":
                candles = data.get("data", [])
                print(f"✅ موفق - تعداد کندل دریافتی: {len(candles)}")
                if candles:
                    print(f"نمونه آخرین کندل: {candles[0]}")
            else:
                print(f"⚠️ خطای داخلی: {data}")
        else:
            print(f"❌ ناموفق: {resp.text[:300]}")
    except Exception as e:
        print(f"❌ خطا در اتصال: {e}")
    print()


def test_kucoin_15m():
    print("=" * 50)
    print("تست KuCoin — کندل تاریخی ۱۵دقیقه‌ای BTC-USDT")
    print("=" * 50)
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {
        "symbol": "BTC-USDT",
        "type": "15min",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"کد وضعیت: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "200000":
                candles = data.get("data", [])
                print(f"✅ موفق - تعداد کندل دریافتی: {len(candles)}")
                if candles:
                    print(f"نمونه آخرین کندل: {candles[0]}")
            else:
                print(f"⚠️ خطای داخلی: {data}")
        else:
            print(f"❌ ناموفق: {resp.text[:300]}")
    except Exception as e:
        print(f"❌ خطا در اتصال: {e}")
    print()


def test_binance_historical():
    print("=" * 50)
    print("تست Binance — کندل تاریخی ۴ساعته BTCUSDT")
    print("=" * 50)
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "4h",
        "limit": 10,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"کد وضعیت: {resp.status_code}")
        if resp.status_code == 200:
            candles = resp.json()
            print(f"✅ موفق - تعداد کندل دریافتی: {len(candles)}")
            if candles:
                print(f"نمونه آخرین کندل: {candles[0]}")
        else:
            print(f"❌ ناموفق (احتمالاً IP مسدود): {resp.text[:300]}")
    except Exception as e:
        print(f"❌ خطا در اتصال: {e}")
    print()


def test_binance_futures():
    print("=" * 50)
    print("تست Binance Futures — کندل تاریخی ۴ساعته BTCUSDT")
    print("=" * 50)
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "4h",
        "limit": 10,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"کد وضعیت: {resp.status_code}")
        if resp.status_code == 200:
            candles = resp.json()
            print(f"✅ موفق - تعداد کندل دریافتی: {len(candles)}")
            if candles:
                print(f"نمونه آخرین کندل: {candles[0]}")
        else:
            print(f"❌ ناموفق (احتمالاً IP مسدود): {resp.text[:300]}")
    except Exception as e:
        print(f"❌ خطا در اتصال: {e}")
    print()


if __name__ == "__main__":
    test_kucoin_historical()
    time.sleep(1)
    test_kucoin_15m()
    time.sleep(1)
    test_binance_historical()
    time.sleep(1)
    test_binance_futures()

    print("=" * 50)
    print("تست کامل شد.")
    print("=" * 50)
