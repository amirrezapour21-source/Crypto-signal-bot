"""
Signal Outcome Tracker
------------------------
این اسکریپت هر ساعت اجرا میشه، سیگنال‌های باز رو با قیمت واقعی چک می‌کنه،
و مشخص می‌کنه که Limit Entry پر شده یا نه، و بعدش TP یا SL اول لمس شده.

⚠️ محدودیت: بر پایه کندل ۱ساعته کار می‌کنه (نه قیمت لحظه‌به‌لحظه).
اگه SL و TP تو یه کندل هم‌زمان لمس بشن، برای احتیاط SL رو اول‌خورده فرض می‌کنیم.
"""

import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, timezone

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2"
LOG_PATH = "signals_log.json"
ENTRY_TIMEOUT_HOURS = 72
TRADE_TIMEOUT_HOURS = 240


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(data):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def get_hourly_candles_since(symbol, since_dt):
    hours_ago = int((datetime.now(timezone.utc) - since_dt).total_seconds() // 3600) + 2
    limit = min(hours_ago, 2000)
    url = f"{CRYPTOCOMPARE_BASE}/histohour"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit}
    resp = safe_get(url, params=params)
    if resp is None or resp.status_code != 200:
        return None
    payload = resp.json()
    if payload.get("Response") != "Success":
        return None
    df = pd.DataFrame(payload["Data"]["Data"])
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[df["dt"] >= since_dt].reset_index(drop=True)
    return df if not df.empty else None


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    resp = requests.post(url, data=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def hours_since(dt):
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


def process_signal(sig):
    symbol = sig["symbol"]
    direction = sig["direction"]
    entry, sl, tp1, tp2 = sig["entry"], sig["sl"], sig["tp1"], sig["tp2"]
    signal_time = datetime.fromisoformat(sig["signal_time"])

    df = get_hourly_candles_since(symbol, signal_time)
    if df is None:
        return sig, None

    notify = None

    if not sig["entered"]:
        for _, row in df.iterrows():
            if row["low"] <= entry <= row["high"]:
                sig["entered"] = True
                sig["status"] = "OPEN"
                sig["entry_time"] = row["dt"].isoformat()
                notify = f"✅ ورود انجام شد: {symbol} در قیمت {entry}"
                break
        if not sig["entered"]:
            if hours_since(signal_time) > ENTRY_TIMEOUT_HOURS:
                sig["status"] = "EXPIRED"
                sig["closed_time"] = datetime.now(timezone.utc).isoformat()
                notify = f"⌛ سفارش منقضی شد (Limit پر نشد): {symbol}"
            return sig, notify

    entry_time = datetime.fromisoformat(sig["entry_time"])
    df_after = df[df["dt"] >= pd.Timestamp(entry_time)]

    tp1_done = sig["status"] == "TP1_HIT"

    for _, row in df_after.iterrows():
        low, high = row["low"], row["high"]
        if direction == "LONG":
            sl_hit = low <= sl
            tp1_hit = high >= tp1
            tp2_hit = high >= tp2
        else:
            sl_hit = high >= sl
            tp1_hit = low <= tp1
            tp2_hit = low <= tp2

        if not tp1_done:
            if sl_hit:
                sig["status"] = "SL_HIT"
                sig["result"] = "LOSS"
                sig["closed_time"] = row["dt"].isoformat()
                notify = f"🔴 حد ضرر خورد: {symbol} ({direction}) — نتیجه: ضرر"
                break
            if tp1_hit:
                sig["status"] = "TP1_HIT"
                sig["result"] = "WIN (TP1)"
                tp1_done = True
                notify = f"🟢 هدف اول خورد: {symbol} ({direction}) — نتیجه: سود جزئی"
        if tp1_done and tp2_hit:
            sig["status"] = "TP2_HIT"
            sig["result"] = "WIN (TP2 - Full)"
            sig["closed_time"] = row["dt"].isoformat()
            notify = f"🟢🟢 هدف دوم خورد: {symbol} ({direction}) — نتیجه: سود کامل"
            break

    if sig["status"] not in ("SL_HIT", "TP2_HIT") and hours_since(entry_time) > TRADE_TIMEOUT_HOURS:
        sig["status"] = "TIMEOUT"
        sig["result"] = sig.get("result") or "OPEN_TIMEOUT"
        sig["closed_time"] = datetime.now(timezone.utc).isoformat()
        notify = f"⏱️ معامله {symbol} پس از ۱۰ روز بدون نتیجه قطعی بسته شد."

    return sig, notify


def main():
    log = load_log()
    open_statuses = ("ENTRY_PENDING", "OPEN", "TP1_HIT")
    open_signals = [s for s in log if s["status"] in open_statuses]

    if not open_signals:
        print("هیچ سیگنال بازی برای پیگیری وجود ندارد.")
        return

    print(f"در حال بررسی {len(open_signals)} سیگنال باز...")
    updated_log = {s["id"]: s for s in log}

    for sig in open_signals:
        try:
            new_sig, notify = process_signal(sig)
            updated_log[sig["id"]] = new_sig
            if notify:
                print(notify)
                if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
                    send_telegram_message(notify)
        except Exception as e:
            print(f"خطا در بررسی {sig['symbol']}: {e}")
        time.sleep(1.2)

    save_log(list(updated_log.values()))
    print("پیگیری به‌روزرسانی و ذخیره شد.")


if __name__ == "__main__":
    main()
