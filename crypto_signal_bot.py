"""
Elite Crypto Volume Profile — Scan & Analysis Bot (Fully Automated)
--------------------------------------------------------------------
بر اساس پرامپت "ELITE CRYPTO VOLUME PROFILE" ساخته شده، با این تفاوت که
به‌جای دریافت اسکرین‌شات چارت از کاربر، خودش داده کندل ۴ساعته و ۱۵دقیقه
واقعی رو از API می‌گیره و Volume Profile / Market Structure رو محاسبه می‌کنه.

⚠️ توجه مهم:
این یک تقریب الگوریتمی از تحلیل Volume Profile حرفه‌ای است (بر پایه OHLCV)،
نه جایگزین کامل نرم‌افزارهایی که به داده tick/order-book دسترسی دارند.
دقت آن به کیفیت داده منبع وابسته است.

منابع داده:
    - لیست ارزها و رتبه‌بندی بازار: CoinGecko
    - کندل‌های ۴H و ۱۵M (OHLCV واقعی): CryptoCompare (min-api.cryptocompare.com)

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
CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2"

SCAN_TOP_N_COINS = 60       # تعداد ارزهایی که در فاز اسکن بررسی میشن
REQUEST_DELAY = 1.2         # فاصله بین درخواست‌ها
MIN_RR = 2.0                # حداقل نسبت ریسک به ریوارد قابل قبول


# ============ ابزار درخواست امن ============
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


# ============ دریافت کندل (CryptoCompare) ============
def get_ohlcv(symbol, timeframe="hour", aggregate=4, limit=150):
    """timeframe: 'hour' برای ساعتی/۴ساعته، 'minute' برای دقیقه‌ای/۱۵دقیقه"""
    endpoint = "histohour" if timeframe == "hour" else "histominute"
    url = f"{CRYPTOCOMPARE_BASE}/{endpoint}"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit, "aggregate": aggregate}
    resp = safe_get(url, params=params)
    if resp is None or resp.status_code != 200:
        return None
    payload = resp.json()
    if payload.get("Response") != "Success":
        return None
    rows = payload["Data"]["Data"]
    df = pd.DataFrame(rows)
    if df.empty or df["close"].eq(0).all():
        return None
    df["volume"] = df["volumeto"]
    return df[["time", "open", "high", "low", "close", "volume"]]


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
    return (atr_short / atr_long) < 0.90


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


# ============ PART A: امتیازدهی کاندید ============
def score_candidate(symbol):
    df4h = get_ohlcv(symbol, timeframe="hour", aggregate=4, limit=150)
    if df4h is None or len(df4h) < 60:
        return None

    structure = classify_structure(df4h)
    vp = compute_volume_profile(df4h)
    if vp is None:
        return None
    rvol = compute_rvol(df4h)
    zq = zone_quality_ok(df4h)
    liq = liquidity_sweep_detected(df4h, structure["pivot_highs"], structure["pivot_lows"])

    current_price = df4h["close"].iloc[-1]
    price_near_poc = abs(current_price - vp["poc"]) / current_price < 0.06

    checks = {
        "Volume Profile": price_near_poc,
        "Volume/RVOL": rvol > 1.15,
        "Zone Quality": zq,
        "Structure Clarity": structure["trend"] in ("up", "down"),
        "Liquidity Context": liq,
    }
    score = sum(checks.values())

    if score >= 4:
        confidence = "بالا"
    elif score == 3:
        confidence = "متوسط"
    else:
        return None  # ≤2 → حذف

    return {
        "symbol": symbol, "score": score, "confidence": confidence,
        "checks": checks, "trend": structure["trend"], "rvol": rvol,
        "price": current_price, "df4h": df4h, "structure": structure, "vp": vp,
    }


def part_a_scan(coins):
    candidates = []
    for c in coins:
        symbol = c["symbol"].upper()
        try:
            res = score_candidate(symbol)
            if res:
                candidates.append(res)
        except Exception as e:
            print(f"خطا در اسکن {symbol}: {e}")
        time.sleep(REQUEST_DELAY)
    candidates.sort(key=lambda x: (x["score"], x["rvol"]), reverse=True)
    return candidates[:2]


# ============ PART B: اجرای معامله (بدون اسکرین‌شات، خودکار) ============
def part_b_execute(candidate):
    symbol = candidate["symbol"]
    trend = candidate["trend"]
    df4h = candidate["df4h"]
    structure = candidate["structure"]

    df15m = get_ohlcv(symbol, timeframe="minute", aggregate=15, limit=150)
    if df15m is None or len(df15m) < 60:
        return {"symbol": symbol, "rejected": True}

    vp15 = compute_volume_profile(df15m.iloc[-40:])
    if vp15 is None:
        return {"symbol": symbol, "rejected": True}

    pivot_highs, pivot_lows = structure["pivot_highs"], structure["pivot_lows"]
    current_price = df15m["close"].iloc[-1]

    if trend == "up" and len(pivot_lows) >= 2 and len(pivot_highs) >= 1:
        direction = "LONG"
        entry = vp15["poc"]
        invalidation = pivot_lows[-1][1]
        sl = invalidation * 0.997
        tp1 = pivot_highs[-1][1]
        tp2 = pivot_highs[-2][1] if len(pivot_highs) >= 2 else tp1 * 1.02
        if tp2 <= tp1:
            tp2 = tp1 * 1.02
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
        tp1 = pivot_lows[-1][1]
        tp2 = pivot_lows[-2][1] if len(pivot_lows) >= 2 else tp1 * 0.98
        if tp2 >= tp1:
            tp2 = tp1 * 0.98
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
    elif rr >= MIN_RR:
        risk_level = "متوسط"
    else:
        risk_level = "بالا - بدون معامله"

    return {
        "symbol": symbol, "rejected": False, "direction": direction,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
        "risk_level": risk_level, "confidence": candidate["confidence"],
        "checks": candidate["checks"],
    }


# ============ فرمت خروجی (فارسی، ساختاریافته) ============
def fmt_price(p):
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def format_output(candidates, executions):
    if not candidates:
        return "فاقد ارز مستعد"

    lines = ["<b>PART A — اسکن مارکت</b>"]
    lines.append("وضعیت اسکن: انجام‌شده ✅ (منبع: CoinGecko + CryptoCompare)\n")
    for c in candidates:
        lines.append(f"<b>{c['symbol']}</b> | اعتبار: {c['confidence']}")
        for k, v in c["checks"].items():
            lines.append(f"  {k}: {'✓' if v else '✗'}")
        lines.append("")

    lines.append("<b>PART B — اجرای معامله</b>")
    for ex in executions:
        if ex["rejected"]:
            lines.append(f"\n<b>{ex['symbol']}</b>: عدم وجود موقعیت کم‌ریسک")
            continue
        lines.append(f"\nنماد: <b>{ex['symbol']}</b>")
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
