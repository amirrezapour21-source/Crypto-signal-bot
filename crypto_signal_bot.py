"""
Crypto Signal Bot — Signal + Price Alerts
KuCoin Spot data. Paper tracking only; no real orders.
"""

import html
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

KUCOIN_BASE = "https://api.kucoin.com"
COINGECKO_BASE = "https://api.coingecko.com"

SCAN_TOP_N_COINS = 40
REQUEST_DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3

MIN_RR = 2.0
MIN_SL_DISTANCE_PCT = 0.005
STOP_ATR_BUFFER = 0.20
PULLBACK_EMA_TOLERANCE_ATR = 0.35
MAX_ENTRY_DEVIATION_PCT = 0.015
MIN_4H_CANDLES = 70
MIN_DAILY_CANDLES = 35
MIN_15M_CANDLES = 60
MAX_SIGNALS_PER_RUN = 2
TIMEOUT_DAYS = 10

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
LOG_PATH = Path(os.getenv("SIGNALS_LOG_PATH", "signals_log.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoSignalBot/1.2"})


def utc_now():
    return datetime.now(timezone.utc)


def parse_time(value):
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def safe_get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            response = SESSION.get(
                url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    return None
                return payload if isinstance(payload, (dict, list)) else None

            if response.status_code == 429 or 500 <= response.status_code < 600:
                time.sleep(min(2 ** attempt * 2, 12))
                continue

            logging.warning("HTTP %s from %s", response.status_code, url)
            return None

        except requests.RequestException as exc:
            logging.warning("Request error for %s: %s", url, exc)
            time.sleep(min(2 ** attempt, 8))

    return None


def get_top_coins(limit: int) -> list:
    payload = safe_get_json(
        f"{COINGECKO_BASE}/coins/markets",
        {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(max(int(limit), 1), 250),
            "page": 1,
            "sparkline": "false",
        },
    )

    if not isinstance(payload, list):
        return []

    coins = []
    seen = set()

    for coin in payload:
        if not isinstance(coin, dict):
            continue

        symbol = str(coin.get("symbol", "")).upper().strip()
        coin_id = str(coin.get("id", "")).strip()
        price = coin.get("current_price") or 0
        change = coin.get("price_change_percentage_24h") or 0

        if not symbol.isalnum() or not coin_id:
            continue

        if 0.97 <= float(price or 0) <= 1.03 and abs(float(change or 0)) < 0.5:
            continue

        pair = f"{symbol}-USDT"
        if pair in seen:
            continue

        seen.add(pair)
        coins.append({"symbol": symbol, "pair": pair, "coin_id": coin_id})

    return coins


def get_kucoin_candles(
    pair: str, candle_type: str, limit: int
) -> Optional[pd.DataFrame]:
    intervals = {
        "15min": 15 * 60,
        "4hour": 4 * 3600,
        "1day": 24 * 3600,
    }

    if candle_type not in intervals:
        raise ValueError(f"Unsupported candle type: {candle_type}")

    now = int(time.time())
    interval = intervals[candle_type]
    start_at = now - (int(limit) + 3) * interval

    payload = safe_get_json(
        f"{KUCOIN_BASE}/api/v1/market/candles",
        {
            "symbol": pair,
            "type": candle_type,
            "startAt": start_at,
            "endAt": now,
        },
    )

    if not payload or payload.get("code") != "200000":
        return None

    raw = payload.get("data")
    if not isinstance(raw, list) or not raw:
        return None

    rows = []
    for item in raw:
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            rows.append({
                "time": int(item[0]),
                "open": float(item[1]),
                "close": float(item[2]),
                "high": float(item[3]),
                "low": float(item[4]),
                "volume": float(item[5]),
            })
        except (TypeError, ValueError, OverflowError):
            continue

    if not rows:
        return None

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
    )

    df = df[
        (df["open"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
        & (df["volume"] >= 0)
    ]

    df = df[df["time"] + interval <= now - 2]
    df = df.tail(int(limit)).reset_index(drop=True)
    return None if df.empty else df


def get_live_price(pair: str) -> Optional[float]:
    payload = safe_get_json(
        f"{KUCOIN_BASE}/api/v1/market/orderbook/level1",
        {"symbol": pair},
    )

    if not payload or payload.get("code") != "200000":
        return None

    try:
        price = float(payload["data"]["price"])
        return price if np.isfinite(price) and price > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(
        span=period, adjust=False, min_periods=period
    ).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()


def classify_trend(df: pd.DataFrame) -> str:
    if df is None or len(df) < 55:
        return "choppy"

    close = df["close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)

    if pd.isna(e20.iloc[-1]) or pd.isna(e50.iloc[-1]):
        return "choppy"

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    pivot_highs = []
    pivot_lows = []

    for i in range(2, len(df) - 2):
        wh = highs[i - 2:i + 3]
        wl = lows[i - 2:i + 3]

        if highs[i] == np.max(wh) and np.sum(wh == highs[i]) == 1:
            pivot_highs.append((i, highs[i]))
        if lows[i] == np.min(wl) and np.sum(wl == lows[i]) == 1:
            pivot_lows.append((i, lows[i]))

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "choppy"

    hh = pivot_highs[-1][1] > pivot_highs[-2][1]
    hl = pivot_lows[-1][1] > pivot_lows[-2][1]
    lh = pivot_highs[-1][1] < pivot_highs[-2][1]
    ll = pivot_lows[-1][1] < pivot_lows[-2][1]

    if e20.iloc[-1] > e50.iloc[-1] and close.iloc[-1] > e50.iloc[-1] and hh and hl:
        return "up"

    if e20.iloc[-1] < e50.iloc[-1] and close.iloc[-1] < e50.iloc[-1] and lh and ll:
        return "down"

    return "choppy"


def recent_swing(
    df: pd.DataFrame, direction: str, lookback: int = 12
) -> Optional[float]:
    recent = df.iloc[-lookback - 1:-1]
    if recent.empty:
        return None
    if direction == "LONG":
        return float(recent["low"].min())
    return float(recent["high"].max())


def detect_setup(pair: str, symbol: str) -> Optional[dict]:
    daily = get_kucoin_candles(pair, "1day", 100)
    time.sleep(REQUEST_DELAY_SECONDS)
    h4 = get_kucoin_candles(pair, "4hour", 120)
    time.sleep(REQUEST_DELAY_SECONDS)
    m15 = get_kucoin_candles(pair, "15min", 100)

    if daily is None or len(daily) < MIN_DAILY_CANDLES:
        return None
    if h4 is None or len(h4) < MIN_4H_CANDLES:
        return None
    if m15 is None or len(m15) < MIN_15M_CANDLES:
        return None

    daily_trend = classify_trend(daily)
    h4_trend = classify_trend(h4)

    if daily_trend not in ("up", "down") or daily_trend != h4_trend:
        return None

    direction = "LONG" if h4_trend == "up" else "SHORT"

    close = m15["close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    atr15 = atr(m15, 14)

    if any(pd.isna(x) for x in (
        e20.iloc[-1], e20.iloc[-2], e50.iloc[-1], atr15.iloc[-1]
    )):
        return None

    last = m15.iloc[-1]
    prev = m15.iloc[-2]
    a = float(atr15.iloc[-1])

    if a <= 0 or not np.isfinite(a):
        return None

    tolerance = a * PULLBACK_EMA_TOLERANCE_ATR

    if direction == "LONG":
        touched_zone = (
            float(last["low"]) <= float(e20.iloc[-1]) + tolerance
            or float(prev["low"]) <= float(e20.iloc[-2]) + tolerance
        )
        confirmed = (
            float(last["close"]) > float(last["open"])
            and float(last["close"]) > float(e20.iloc[-1])
            and float(last["close"]) > float(prev["high"])
            and float(last["close"]) > float(e50.iloc[-1])
        )
    else:
        touched_zone = (
            float(last["high"]) >= float(e20.iloc[-1]) - tolerance
            or float(prev["high"]) >= float(e20.iloc[-2]) - tolerance
        )
        confirmed = (
            float(last["close"]) < float(last["open"])
            and float(last["close"]) < float(e20.iloc[-1])
            and float(last["close"]) < float(prev["low"])
            and float(last["close"]) < float(e50.iloc[-1])
        )

    if not touched_zone or not confirmed:
        return None

    setup_close = float(last["close"])
    live_price = get_live_price(pair)
    if live_price is None:
        return None

    deviation = abs(live_price - setup_close) / setup_close
    if deviation > MAX_ENTRY_DEVIATION_PCT:
        logging.info("%s rejected: live price too far from setup close", pair)
        return None

    entry = live_price
    swing = recent_swing(m15, direction, lookback=12)
    if swing is None:
        return None

    if direction == "LONG":
        sl = min(swing, float(last["low"])) - STOP_ATR_BUFFER * a
        risk = entry - sl
        if risk <= 0:
            return None
        tp1 = entry + MIN_RR * risk
        tp2 = entry + 3.0 * risk
    else:
        sl = max(swing, float(last["high"])) + STOP_ATR_BUFFER * a
        risk = sl - entry
        if risk <= 0:
            return None
        tp1 = entry - MIN_RR * risk
        tp2 = entry - 3.0 * risk

    if entry <= 0 or abs(entry - sl) / entry < MIN_SL_DISTANCE_PCT:
        return None

    rr = abs(tp1 - entry) / abs(entry - sl)
    if not np.isfinite(rr) or rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "pair": pair,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "strategy_type": "Daily + 4H Trend / 15m Pullback",
        "daily_trend": daily_trend,
        "h4_trend": h4_trend,
        "setup_close": setup_close,
        "live_price": live_price,
        "signal_time": utc_now().isoformat(),
    }


def load_log() -> list:
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logging.error("Signal log is unreadable.")
        return []


def save_log(records: list) -> None:
    tmp = LOG_PATH.with_suffix(LOG_PATH.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, LOG_PATH)
    except OSError as exc:
        logging.error("Could not save signal log: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def has_open_duplicate(records: list, pair: str, direction: str) -> bool:
    open_statuses = {"ENTRY_PENDING", "OPEN", "TP1_HIT"}
    return any(
        r.get("pair", f"{r.get('symbol', '')}-USDT") == pair
        and r.get("direction") == direction
        and r.get("status") in open_statuses
        for r in records if isinstance(r, dict)
    )


def fmt_price(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}"
    if value >= 0.01:
        return f"{value:.6f}"
    return f"{value:.8f}"


def format_signal(signal: dict) -> str:
    direction = signal["direction"]
    icon = "🟢" if direction == "LONG" else "🔴"

    return "\n".join([
        f"📊 <b>{html.escape(signal['symbol'])}/USDT — SIGNAL</b>",
        "",
        f"{icon} <b>Direction:</b> {direction}",
        f"📈 <b>Strategy:</b> {html.escape(signal['strategy_type'])}",
        "",
        f"📍 <b>Entry:</b> <code>{fmt_price(signal['entry'])}</code>",
        f"🛑 <b>Stop Loss:</b> <code>{fmt_price(signal['sl'])}</code>",
        "",
        "🎯 <b>Take Profit:</b>",
        f"TP1 → <code>{fmt_price(signal['tp1'])}</code>",
        f"TP2 → <code>{fmt_price(signal['tp2'])}</code>",
        "",
        f"⚖️ <b>Risk/Reward:</b> 1:{signal['rr']:.2f}",
        "⚠️ <b>Risk Level:</b> MEDIUM",
        "",
        "🏦 <b>Exchange:</b> CoinGecko",
    ])


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials missing.")
        return False

    try:
        response = SESSION.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            logging.error("Telegram HTTP %s: %s", response.status_code, response.text[:300])
            return False

        payload = response.json()
        if not payload.get("ok"):
            logging.error("Telegram rejected message: %s", payload)
            return False

        return True

    except (requests.RequestException, ValueError) as exc:
        logging.error("Telegram send failed: %s", exc)
        return False


def monitor_signals(records: list) -> bool:
    changed = False
    now = utc_now()
    active_statuses = {"ENTRY_PENDING", "OPEN", "TP1_HIT"}

    for record in records:
        if not isinstance(record, dict):
            continue

        status = record.get("status")
        if status not in active_statuses:
            continue

        pair = record.get("pair")
        if not pair:
            continue

        symbol = html.escape(str(record.get("symbol", pair.replace("-USDT", ""))))
        direction = record.get("direction")

        try:
            entry = float(record["entry"])
            sl = float(record["sl"])
            tp1 = float(record["tp1"])
            tp2 = float(record["tp2"])
        except (KeyError, TypeError, ValueError):
            logging.warning("Invalid levels in signal record: %s", pair)
            continue

        if direction not in {"LONG", "SHORT"}:
            continue

        if status == "ENTRY_PENDING":
            signal_time = parse_time(record.get("signal_time"))
            if signal_time and now - signal_time >= timedelta(days=TIMEOUT_DAYS):
                message = (
                    f"⌛ <b>سفارش منقضی شد (Limit پر نشد): "
                    f"{symbol} — ورودی: {fmt_price(entry)}$</b>"
                )
                if send_telegram_message(message):
                    record["status"] = "CLOSED_EXPIRED"
                    record["result"] = "EXPIRED"
                    record["closed_time"] = now.isoformat()
                    changed = True
                continue

        if status in {"OPEN", "TP1_HIT"}:
            entry_time = parse_time(record.get("entry_time"))
            if entry_time and now - entry_time >= timedelta(days=TIMEOUT_DAYS):
                message = (
                    f"⏱️ <b>TRADE TIMEOUT — {symbol}/USDT</b>\n\n"
                    f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                    f"📍 Entry: <code>{fmt_price(entry)}</code>\n\n"
                    "پس از ۱۰ روز بدون نتیجه قطعی بسته شد."
                )
                if send_telegram_message(message):
                    record["status"] = "CLOSED_TIMEOUT"
                    record["result"] = "TIMEOUT"
                    record["closed_time"] = now.isoformat()
                    changed = True
                continue

        price = get_live_price(pair)
        if price is None:
            continue

        if status == "ENTRY_PENDING":
            entry_hit = price <= entry if direction == "LONG" else price >= entry
            if not entry_hit:
                continue

            message = (
                f"➕ <b>ENTRY EXECUTED — {symbol}/USDT</b>\n\n"
                f"📍 Entry: <code>{fmt_price(entry)}</code>\n"
                f"{'🟢' if direction == 'LONG' else '🔴'} "
                f"<b>{direction} ACTIVE</b>\n\n"
                f"🎯 TP1: <code>{fmt_price(tp1)}</code>\n"
                f"🎯 TP2: <code>{fmt_price(tp2)}</code>\n"
                f"🛑 SL: <code>{fmt_price(sl)}</code>"
            )
            if send_telegram_message(message):
                record["status"] = "OPEN"
                record["entered"] = True
                record["entry_time"] = now.isoformat()
                changed = True
            continue

        sl_hit = price <= sl if direction == "LONG" else price >= sl
        if sl_hit:
            loss_pct = abs(entry - price) / entry * 100
            message = (
                f"🛑 <b>STOP LOSS HIT — {symbol}/USDT</b>\n\n"
                f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                f"📍 Entry: <code>{fmt_price(entry)}</code>\n"
                f"🛑 SL: <code>{fmt_price(sl)}</code> ❌\n\n"
                f"📉 <b>Loss: -{loss_pct:.2f}%</b>\n\n"
                "🔴 <b>TRADE CLOSED</b>"
            )
            if send_telegram_message(message):
                record["status"] = "CLOSED_SL"
                record["closed_time"] = now.isoformat()
                record["result"] = "SL"
                changed = True
            continue

        tp2_hit = price >= tp2 if direction == "LONG" else price <= tp2
        if tp2_hit:
            movement_pct = abs(tp2 - entry) / entry * 100
            final_rr = abs(tp2 - entry) / abs(entry - sl) if entry != sl else 0

            tp2_message = (
                f"🏆 <b>TP2 HIT — {symbol}/USDT</b>\n\n"
                f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                "✅ TP1 + TP2 hit\n\n"
                f"📈 <b>Total Profit: +{movement_pct:.2f}%</b>\n"
                f"⚖️ <b>Final R:R: 1:{final_rr:.2f}</b>\n\n"
                "🟢 <b>TRADE CLOSED</b>"
            )
            if send_telegram_message(tp2_message):
                record["status"] = "CLOSED_TP2"
                record["closed_time"] = now.isoformat()
                record["result"] = "TP2"
                changed = True
            continue

        tp1_hit = price >= tp1 if direction == "LONG" else price <= tp1
        if status == "OPEN" and tp1_hit:
            message = (
                f"🎯 <b>TAKE PROFIT 1 HIT — {symbol}/USDT</b>\n\n"
                f"{direction}\n"
                f"TP1: <code>{fmt_price(tp1)}</code> ✅\n"
                f"💹 Current: <code>{fmt_price(price)}</code>\n\n"
                "Monitoring continues for TP2 and SL."
            )
            if send_telegram_message(message):
                record["status"] = "TP1_HIT"
                record["tp1_time"] = now.isoformat()
                changed = True

    return changed


def main() -> None:
    logging.info("Signal scanner started.")

    records = load_log()
    if monitor_signals(records):
        save_log(records)

    coins = get_top_coins(SCAN_TOP_N_COINS)
    if not coins:
        logging.error("Coin universe unavailable. Check CoinGecko connectivity/rate limits.")
        return

    checked_pairs = set()
    emitted = 0

    for coin in coins:
        if emitted >= MAX_SIGNALS_PER_RUN:
            break

        pair = coin["pair"]
        if pair in checked_pairs:
            continue
        checked_pairs.add(pair)

        try:
            signal = detect_setup(pair, coin["symbol"])
            if not signal:
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            if has_open_duplicate(records, pair, signal["direction"]):
                logging.info("%s skipped: duplicate open signal", pair)
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            if not send_telegram_message(format_signal(signal)):
                logging.error("%s signal was not delivered; not recording it.", pair)
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            record = {
                "id": f"{pair}_{int(time.time())}",
                "symbol": signal["symbol"],
                "pair": signal["pair"],
                "direction": signal["direction"],
                "strategy_type": signal["strategy_type"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp1": signal["tp1"],
                "tp2": signal["tp2"],
                "rr": signal["rr"],
                "signal_time": signal["signal_time"],
                "status": "ENTRY_PENDING",
                "entered": False,
                "entry_time": None,
                "closed_time": None,
                "result": None,
            }

            records.append(record)
            save_log(records)
            emitted += 1

            logging.info(
                "Signal sent: %s %s RR=%.2f",
                pair, signal["direction"], signal["rr"]
            )

        except Exception:
            logging.exception("Unexpected scan error for %s", pair)

        time.sleep(REQUEST_DELAY_SECONDS)

    logging.info(
        "Scan finished: %d coins considered, %d new signals sent.",
        len(checked_pairs), emitted
    )


if __name__ == "__main__":
    main()
