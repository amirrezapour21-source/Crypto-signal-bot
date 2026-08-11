"""
تست دسترسی به KuCoin API از سرورهای GitHub Actions
"""
import requests

def test_kucoin():
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {
        "symbol": "BTC-USDT",
        "type": "1hour",
        "startAt": 1755000000,
        "endAt": 1755010000,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"کد وضعیت: {resp.status_code}")
        print(f"پاسخ: {resp.text[:500]}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "200000":
                print("\n✅ KuCoin در دسترس است و داده برمی‌گرداند.")
            else:
                print(f"\n⚠️ پاسخ دریافت شد ولی کد داخلی خطا: {data.get('code')}")
        else:
            print("\n❌ KuCoin در دسترس نیست یا خطا برگرداند.")
    except requests.RequestException as e:
        print(f"\n❌ خطا در اتصال: {e}")

if __name__ == "__main__":
    test_kucoin()
