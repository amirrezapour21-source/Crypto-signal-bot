"""
Signal Outcome Tracker
------------------------
این اسکریپت به‌صورت دوره‌ای (best-effort، طبق مستندات GitHub Actions) اجرا
می‌شه، سیگنال‌های باز رو با قیمت واقعی چک می‌کنه، و مشخص می‌کنه که Limit
Entry پر شده یا نه، و بعدش TP یا SL اول لمس شده.

⚠️ محدودیت: بر پایه کندل ساعتی مصنوعی (از CoinGecko) کار می‌کنه، نه قیمت
لحظه‌به‌لحظه. اگه SL و TP تو یه کندل هم‌زمان لمس بشن، برای احتیاط SL رو
اول‌خورده فرض می‌کنیم.

نسخه v3 - Scheduler Awareness: چون GitHub Actions تضمین نمی‌کنه دقیقاً
هر ۱۵ دقیقه اجرا بشه (Best-Effort Scheduler)، این نسخه:
1. زمان دقیق شروع اجرا (UTC) رو ثبت می‌کنه
2. زمان آخرین اجرای موفق قبلی رو از فایل last_run.json می‌خونه
3. فاصله واقعی از اجرای قبلی رو محاسبه و لاگ می‌کنه
4. یک گزارش خلاصه (SCHEDULER REPORT) در پایان هر اجرا چاپ می‌کنه
هیچ تغییری در منطق تصمیم‌گیری (Entry/SL/TP/Strategy) ایجاد نشده - این
فقط برای شفافیت و مانیتورینگ زمان‌بندیه.
"""

import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, timezone

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
LOG_PATH = "signals_log.json"
LAST_RUN_PATH = "last_run.json"
ENTRY_TIMEOUT_HOURS = 72
TRADE_TIMEOUT_HOURS = 240
DATA_AGE_WARNING_MINUTES = 90  # اگه داده از این قدیمی‌تر بود، هشدار بده


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(data):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_last_run():
    if not os.path.exists(LAST_RUN_PATH):
        return None
    try:
        with open(LAST_RUN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return datetime.fromisoformat(data["last_successful_run"])
    except Exception:
        return None


def save_last_run(dt):
    with open(LAST_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_successful_run": dt.isoformat()}, f)


def fmt_price(p):
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def safe_get(url, params=None, retries=3):
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            print(f"    [دیباگ] خطای شبکه در تلاش {attempt+1}: {e}", flush=True)
            time.sleep(5)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            print(f"    [دیباگ] rate limit (429) - صبر و تلاش مجدد", flush=True)
            time.sleep(15 * (attempt + 1))
            continue
        print(f"    [دیباگ] کد وضعیت غیرمنتظره: {resp.status_code}", flush=True)
        return resp
    return resp


def get_hourly_candles_since(coin_id, since_dt):
    hours_ago = int((datetime.now(timezone.utc) - since_dt).total_seconds() // 3600) + 2
    days = min(90, max(2, (hours_ago // 24) + 2))
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    resp = safe_get(url, params=params)
    if resp is None or resp.status_code != 200:
        print(f"    [دیباگ] دریافت داده ناموفق برای coin_id={coin_id} (status={resp.status_code if resp else 'None'})", flush=True)
        return None, None
    data = resp.json()
    prices = data.get("prices")
    volumes = data.get("total_volumes")
    if not prices or not volumes:
        print(f"    [دیباگ] prices/volumes خالی برای coin_id={coin_id}", flush=True)
        return None, None
    df = pd.DataFrame({"time": [p[0] // 1000 for p in prices], "price": [p[1] for p in prices]})
    vol_df = pd.DataFrame({"time": [v[0] // 1000 for v in volumes], "volume": [v[1] for v in volumes]})
    df = df.merge(vol_df, on="time", how="left").sort_values("time").reset_index(drop=True)

    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour_bucket"] = df["dt"].dt.floor("h")
    grouped = df.groupby("hour_bucket").agg(
        open=("price", "first"), high=("price", "max"),
        low=("price", "min"), close=("price", "last"),
        volume=("volume", "sum"),
    ).reset_index().rename(columns={"hour_bucket": "dt"})

    grouped = grouped[grouped["dt"] >= since_dt].reset_index(drop=True)
    if grouped.empty:
        print(f"    [دیباگ] بعد از فیلتر زمانی (since={since_dt}) هیچ کندلی نموند", flush=True)
        return None, None
    latest_data_timestamp = grouped["dt"].max()
    return grouped, latest_data_timestamp


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    resp = requests.post(url, data=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def hours_since(dt):
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


def find_entry_row(df, entry):
    prev_close = None
    for idx, row in df.iterrows():
        if row["low"] <= entry <= row["high"]:
            return row, "candle_range"
        if prev_close is not None:
            crossed_down = prev_close > entry and row["close"] < entry
            crossed_up = prev_close < entry and row["close"] > entry
            if crossed_down or crossed_up:
                return row, "gap_cross"
        prev_close = row["close"]
    return None, None


def build_entry_message(symbol, direction, entry, sl, tp1, tp2):
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    return (
        f"🟢 ENTRY EXECUTED — {symbol}/USDT\n\n"
        f"📍 Entry: {fmt_price(entry)}\n"
        f"{dir_emoji} {direction} ACTIVE\n\n"
        f"🎯 TP1: {fmt_price(tp1)}\n"
        f"🎯 TP2: {fmt_price(tp2)}\n"
        f"🛑 SL: {fmt_price(sl)}"
    )


def pct_change(entry, target, direction):
    if direction == "LONG":
        return (target - entry) / entry * 100
    else:
        return (entry - target) / entry * 100


def build_tp1_message(symbol, direction, entry, tp1):
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    profit = pct_change(entry, tp1, direction)
    return (
        f"🎯 TP1 HIT — {symbol}/USDT\n\n"
        f"{dir_emoji} {direction}\n"
        f"✅ TP1: {fmt_price(tp1)}\n"
        f"💰 Profit so far: +{profit:.2f}%\n\n"
        f"⏳ TP2 در حال پیگیری (SL ثابت)"
    )


def build_tp2_message(symbol, direction, entry, tp1, tp2, sl):
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    total_profit = pct_change(entry, tp2, direction)
    risk_pct = abs(pct_change(entry, sl, direction))
    rr_final = total_profit / risk_pct if risk_pct > 0 else 0
    return (
        f"🏆 TP2 HIT — {symbol}/USDT\n\n"
        f"{dir_emoji} {direction}\n"
        f"✅ TP1 + TP2 خورد\n"
        f"💰 Total Profit: +{total_profit:.2f}%\n"
        f"📊 Final R:R: 1:{rr_final:.2f}\n\n"
        f"✅ TRADE CLOSED"
    )


def build_sl_message(symbol, direction, entry, sl):
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    loss = pct_change(entry, sl, direction)
    return (
        f"🛑 STOP LOSS HIT — {symbol}/USDT\n\n"
        f"{dir_emoji} {direction}\n"
        f"📍 Entry: {fmt_price(entry)}\n"
        f"🛑 SL: {fmt_price(sl)} ❌\n"
        f"📉 Loss: {loss:.2f}%\n\n"
        f"🔴 TRADE CLOSED"
    )


def build_expired_message(symbol, entry):
    return f"⌛ سفارش منقضی شد (Limit پر نشد): {symbol} — ورودی: {fmt_price(entry)}$"


def build_timeout_message(symbol, direction, entry):
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    return (
        f"⏱️ TRADE TIMEOUT — {symbol}/USDT\n\n"
        f"{dir_emoji} {direction}\n"
        f"📍 Entry: {fmt_price(entry)}\n\n"
        f"پس از ۱۰ روز بدون نتیجه قطعی بسته شد."
    )


def process_signal(sig, max_data_age_minutes):
    symbol = sig["symbol"]
    coin_id = sig.get("coin_id", "")
    direction = sig["direction"]
    entry, sl, tp1, tp2 = sig["entry"], sig["sl"], sig["tp1"], sig["tp2"]
    signal_time = datetime.fromisoformat(sig["signal_time"])

    print(f"\n--- بررسی {symbol} (id={sig['id']}, status={sig['status']}) ---", flush=True)
    print(f"  coin_id={coin_id} | entry={fmt_price(entry)} | signal_time={signal_time}", flush=True)

    if not coin_id:
        print("  [دیباگ] coin_id خالی است - غیرقابل پیگیری", flush=True)
        return sig, None

    df, latest_data_ts = get_hourly_candles_since(coin_id, signal_time)
    if df is None:
        print("  [دیباگ] df=None - داده‌ای دریافت نشد، این سیگنال این‌بار رد می‌شود", flush=True)
        return sig, None

    data_age_min = (datetime.now(timezone.utc) - latest_data_ts).total_seconds() / 60
    max_data_age_minutes[0] = max(max_data_age_minutes[0], data_age_min)
    print(f"  [دیباگ] {len(df)} کندل دریافت شد از {df['dt'].iloc[0]} تا {df['dt'].iloc[-1]}", flush=True)
    print(f"  [دیباگ] بازه Low/High کل داده: {df['low'].min():.6f} تا {df['high'].max():.6f}", flush=True)
    print(f"  [دیباگ] DATA_AGE: {data_age_min:.1f} دقیقه" + (" ⚠️ DATA_TOO_OLD" if data_age_min > DATA_AGE_WARNING_MINUTES else ""), flush=True)

    notify = None

    if not sig["entered"]:
        entry_row, method = find_entry_row(df, entry)
        if entry_row is not None:
            sig["entered"] = True
            sig["status"] = "OPEN"
            sig["entry_time"] = entry_row["dt"].isoformat()
            notify = build_entry_message(symbol, direction, entry, sl, tp1, tp2)
            print(f"  [دیباگ] ✅ Entry پیدا شد در کندل {entry_row['dt']} (روش={method})", flush=True)
        else:
            print(f"  [دیباگ] هیچ کندلی شامل entry={fmt_price(entry)} نبود و پرشی هم دیده نشد - هنوز پر نشده", flush=True)
            if hours_since(signal_time) > ENTRY_TIMEOUT_HOURS:
                sig["status"] = "EXPIRED"
                sig["closed_time"] = datetime.now(timezone.utc).isoformat()
                notify = build_expired_message(symbol, entry)
            return sig, notify

    entry_time = datetime.fromisoformat(sig["entry_time"])
    df_after = df[df["dt"] >= pd.Timestamp(entry_time)]
    print(f"  [دیباگ] {len(df_after)} کندل بعد از entry_time={entry_time} برای چک SL/TP", flush=True)

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
                notify = build_sl_message(symbol, direction, entry, sl)
                print(f"  [دیباگ] 🔴 SL خورد در کندل {row['dt']}", flush=True)
                break
            if tp1_hit:
                sig["status"] = "TP1_HIT"
                sig["result"] = "WIN (TP1)"
                tp1_done = True
                notify = build_tp1_message(symbol, direction, entry, tp1)
                print(f"  [دیباگ] 🟢 TP1 خورد در کندل {row['dt']}", flush=True)
        if tp1_done and tp2_hit:
            sig["status"] = "TP2_HIT"
            sig["result"] = "WIN (TP2 - Full)"
            sig["closed_time"] = row["dt"].isoformat()
            notify = build_tp2_message(symbol, direction, entry, tp1, tp2, sl)
            print(f"  [دیباگ] 🟢🟢 TP2 خورد در کندل {row['dt']}", flush=True)
            break

    if notify is None:
        print(f"  [دیباگ] هیچ رویداد جدیدی (SL/TP) در این بازه پیدا نشد. sl={fmt_price(sl)} tp1={fmt_price(tp1)} tp2={fmt_price(tp2)}", flush=True)

    if sig["status"] not in ("SL_HIT", "TP2_HIT") and hours_since(entry_time) > TRADE_TIMEOUT_HOURS:
        sig["status"] = "TIMEOUT"
        sig["result"] = sig.get("result") or "OPEN_TIMEOUT"
        sig["closed_time"] = datetime.now(timezone.utc).isoformat()
        notify = build_timeout_message(symbol, direction, entry)

    return sig, notify


def main():
    actual_start = datetime.now(timezone.utc)
    last_run = load_last_run()
    elapsed_str = "نامشخص (اولین اجرا)"
    if last_run:
        elapsed_min = (actual_start - last_run).total_seconds() / 60
        elapsed_str = f"{elapsed_min:.1f} دقیقه"

    print("=" * 40, flush=True)
    print("SCHEDULER INFO", flush=True)
    print("=" * 40, flush=True)
    print(f"Actual Start (UTC): {actual_start.isoformat()}", flush=True)
    print(f"Last Successful Run: {last_run.isoformat() if last_run else 'ندارد'}", flush=True)
    print(f"Elapsed Since Last Run: {elapsed_str}", flush=True)
    print("=" * 40 + "\n", flush=True)

    log = load_log()
    open_statuses = ("ENTRY_PENDING", "OPEN", "TP1_HIT")
    open_signals = [s for s in log if s["status"] in open_statuses]

    max_data_age = [0.0]  # لیست به‌عنوان trick برای mutable closure

    if not open_signals:
        print("هیچ سیگنال بازی برای پیگیری وجود ندارد.", flush=True)
        scanner_status = "SUCCESS"
        signal_generated = "NO"
    else:
        print(f"در حال بررسی {len(open_signals)} سیگنال باز...", flush=True)
        updated_log = {s["id"]: s for s in log}
        any_notify = False

        for sig in open_signals:
            try:
                new_sig, notify = process_signal(sig, max_data_age)
                updated_log[sig["id"]] = new_sig
                if notify:
                    any_notify = True
                    print(f"  >>> نوتیفیکیشن ارسال شد", flush=True)
                    if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
                        send_telegram_message(notify)
            except Exception as e:
                print(f"خطا در بررسی {sig['symbol']}: {e}", flush=True)
            time.sleep(1.2)

        save_log(list(updated_log.values()))
        scanner_status = "SUCCESS"
        signal_generated = "YES" if any_notify else "NO"
        print("\nپیگیری به‌روزرسانی و ذخیره شد.", flush=True)

    actual_finish = datetime.now(timezone.utc)
    duration = (actual_finish - actual_start).total_seconds()
    save_last_run(actual_finish)

    print("\n" + "=" * 40, flush=True)
    print("SCHEDULER REPORT", flush=True)
    print("=" * 40, flush=True)
    print(f"Actual Start:            {actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
    print(f"Actual Finish:           {actual_finish.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
    print(f"Execution Duration:      {duration:.1f} seconds", flush=True)
    print(f"Elapsed Since Last Scan: {elapsed_str}", flush=True)
    print(f"Max Data Age Seen:       {max_data_age[0]:.1f} minutes" + (" ⚠️ DATA_TOO_OLD" if max_data_age[0] > DATA_AGE_WARNING_MINUTES else ""), flush=True)
    print(f"Scanner Status:          {scanner_status}", flush=True)
    print(f"Signal Notification:     {signal_generated}", flush=True)
    print("=" * 40, flush=True)


if __name__ == "__main__":
    main()
</parameter>
