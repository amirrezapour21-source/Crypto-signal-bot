"""
Crypto Signal Bot — Institutional LVAR Strategy v3.2 (Liquidity Sweep & Value Area Rejection)
KuCoin Spot. Paper tracking only; no real orders.
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

SCAN_TOP_N_COINS = 50          # top 50 USDT pairs by 24h turnover (liquid coins only)
REQUEST_DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3

MIN_RR = 2.0
MIN_SL_DISTANCE_PCT = 0.005
STOP_ATR_BUFFER = 0.40
MAX_ENTRY_DEVIATION_PCT = 0.015
MAX_SETUP_AGE_SECONDS = 3 * 900   # the sweep candle must have closed within the last 45 minutes
MIN_15M_CANDLES = 60
PROFILE_15M_FETCH = 230          # about 57 hours of 15m candles (covers the whole previous UTC day)
MIN_PREV_DAY_15M_BARS = 80       # previous UTC day must have at least 80 of 96 bars
VOLUME_AVG_BARS = 20
VOLUME_CONFIRM_MULT = 2.5
MIN_24H_TURNOVER_USDT = 5_000_000  # minimum 24h turnover (USDT) for a pair to be scanned
REJECTION_CLOSE_FRACTION = 0.40  # close must sit in the lower 40% (SHORT) / upper 40% (LONG) of the sweep candle
STRATEGY_VERSION = "LVAR-v3.2"
SESSION_START_UTC = 7             # London + New York hours only: new signals are scanned between 07:00 and 20:00 UTC
SESSION_END_UTC = 20
H4_TREND_FETCH = 120
H4_EMA_FAST, H4_EMA_SLOW = 20, 50
EXCLUDED_BASES = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "USDD", "BUSD", "EURT", "XAUT", "PAXG"}
EXCLUDED_SUFFIXES = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")
MAX_SIGNALS_PER_RUN = 2
TIMEOUT_DAYS = 10
PENDING_MAX_15M_BARS = 4
MONITOR_CANDLE_TYPE = "15min"
MONITOR_INTERVAL_SECONDS = 15 * 60
MONITOR_FETCH_LIMIT = 1500

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
LOG_PATH = Path(os.getenv("SIGNALS_LOG_PATH", "signals_log.json"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoSignalBot/LVAR-Pro"})


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
            response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
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
    payload = safe_get_json(f"{KUCOIN_BASE}/api/v1/market/allTickers")
    if not isinstance(payload, dict) or payload.get("code") != "200000":
        return []
    data_block = payload.get("data")
    data = data_block.get("ticker", []) if isinstance(data_block, dict) else []
    if not isinstance(data, list):
        return []
    coins = []
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol_raw = str(item.get("symbol") or "")
        if not symbol_raw.endswith("-USDT"):
            continue
        symbol = symbol_raw[:-len("-USDT")]
        try:
            vol_value = float(item.get("volValue") or 0)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(vol_value):
            continue
        if not symbol or symbol in EXCLUDED_BASES or symbol.endswith(EXCLUDED_SUFFIXES):
            continue
        if vol_value < MIN_24H_TURNOVER_USDT:
            continue
            
        coins.append({
            "symbol": symbol,
            "pair": symbol_raw,
            "volume": vol_value
        })
    
    coins = sorted(coins, key=lambda x: x["volume"], reverse=True)
    return coins[:limit]


def get_kucoin_candles(pair: str, candle_type: str, limit: int,
                       start_at: Optional[int] = None) -> Optional[pd.DataFrame]:
    intervals = {"15min": 900, "4hour": 14400, "1day": 86400}
    if candle_type not in intervals:
        raise ValueError(f"Unsupported candle type: {candle_type}")
    now = int(time.time())
    interval = intervals[candle_type]
    start_at = int(start_at) if start_at is not None else now - (int(limit) + 3) * interval
    payload = safe_get_json(f"{KUCOIN_BASE}/api/v1/market/candles", {
        "symbol": pair, "type": candle_type, "startAt": start_at, "endAt": now,
    })
    if not isinstance(payload, dict) or payload.get("code") != "200000":
        return None
    raw = payload.get("data")
    if not isinstance(raw, list) or not raw:
        return None
    rows = []
    for item in raw:
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            rows.append({"time": int(item[0]), "open": float(item[1]),
                         "close": float(item[2]), "high": float(item[3]),
                         "low": float(item[4]), "volume": float(item[5])})
        except (TypeError, ValueError, OverflowError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["time"]).sort_values("time")
    df = df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.volume >= 0)]
    df = df[df.time + interval <= now - 2]
    return df.tail(int(limit)).reset_index(drop=True) if not df.empty else None


def get_live_price(pair: str) -> Optional[float]:
    payload = safe_get_json(f"{KUCOIN_BASE}/api/v1/market/orderbook/level1", {"symbol": pair})
    if not isinstance(payload, dict) or payload.get("code") != "200000":
        return None
    try:
        price = float(payload["data"]["price"])
        return price if np.isfinite(price) and price > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
    """محاسبه‌ی پویای محدوده ارزش (VAH, VAL) و نقطه کنترل حجم (POC)"""
    if df is None or len(df) < bins:
        return {"poc": 0, "vah": 0, "val": 0}
    
    price_min, price_max = df["low"].min(), df["high"].max()
    if price_min == price_max:
        return {"poc": price_min, "vah": price_max, "val": price_min}
    
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    hist, bin_edges = np.histogram(typical, bins=bins, range=(float(price_min), float(price_max)), weights=df["volume"])
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    
    total_vol = np.sum(hist)
    if total_vol <= 0:
        return {"poc": poc, "vah": price_max, "val": price_min}
    
    target_vol = total_vol * 0.70
    current_vol = hist[poc_idx]
    left, right = poc_idx, poc_idx
    
    while current_vol < target_vol:
        added = False
        if left > 0:
            left -= 1
            current_vol += hist[left]
            added = True
        if right < len(hist) - 1:
            right += 1
            current_vol += hist[right]
            added = True
        if not added:
            break
            
    val = bin_edges[left]
    vah = bin_edges[right + 1]
    return {"poc": float(poc), "vah": float(vah), "val": float(val)}


def previous_day_window(m15: pd.DataFrame) -> Optional[pd.DataFrame]:
    """15m candles of the previous completed UTC day (the reference session). None if incomplete."""
    day_start = int(utc_now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    prev = m15[(m15["time"] >= day_start - 86400) & (m15["time"] < day_start)]
    return prev if len(prev) >= MIN_PREV_DAY_15M_BARS else None


def detect_setup(pair: str, symbol: str) -> Optional[dict]:
    m15 = get_kucoin_candles(pair, "15min", PROFILE_15M_FETCH)
    if m15 is None or len(m15) < MIN_15M_CANDLES:
        return None

    # Reference levels come ONLY from the previous completed UTC day (never from the signal candle's own window).
    prev_day = previous_day_window(m15)
    if prev_day is None:
        return None
    vp = calculate_volume_profile(prev_day)
    vah, val, poc = vp["vah"], vp["val"], vp["poc"]
    if not (vah > poc > val > 0):
        return None
    pdh, pdl = float(prev_day["high"].max()), float(prev_day["low"].min())

    last = m15.iloc[-1]
    if int(utc_now().timestamp()) - (int(last.time) + MONITOR_INTERVAL_SECONDS) > MAX_SETUP_AGE_SECONDS:
        return None   # stale last candle (illiquid pair / data gap): never signal on old data
    atr_val = float(atr(m15, 14).iloc[-1])
    if atr_val <= 0 or not np.isfinite(atr_val):
        return None
    candle_range = float(last.high) - float(last.low)
    if candle_range <= 0:
        return None

    # Absorption volume: last candle vs the mean of the previous 20 candles.
    avg_vol = m15["volume"].iloc[-VOLUME_AVG_BARS - 1:-1].mean()
    if not (avg_vol > 0 and last.volume >= avg_vol * VOLUME_CONFIRM_MULT):
        return None

    direction = None
    swept_level = None
    close_pos = (float(last.close) - float(last.low)) / candle_range   # 0 = at the low, 1 = at the high
    # SHORT: the candle pierced a level above (PDH or VAH) and CLOSED back below that same level, bearish, close in lower part.
    swept_up = [lv for lv in (pdh, vah) if float(last.high) > lv and float(last.close) < lv]
    if swept_up and last.close < last.open and close_pos <= REJECTION_CLOSE_FRACTION:
        direction, swept_level = "SHORT", max(swept_up)
    else:
        # LONG: pierced a level below (PDL or VAL) and CLOSED back above that same level, bullish, close in upper part.
        swept_down = [lv for lv in (pdl, val) if float(last.low) < lv and float(last.close) > lv]
        if swept_down and last.close > last.open and close_pos >= 1 - REJECTION_CLOSE_FRACTION:
            direction, swept_level = "LONG", min(swept_down)
    if not direction:
        return None

    # 4H trend filter: trade only with the 4H trend (EMA20 vs EMA50 and the last closed 4H candle vs EMA50).
    h4 = get_kucoin_candles(pair, "4hour", H4_TREND_FETCH)
    if h4 is None or len(h4) < H4_EMA_SLOW + 5:
        return None
    h4_fast, h4_slow = ema(h4["close"], H4_EMA_FAST).iloc[-1], ema(h4["close"], H4_EMA_SLOW).iloc[-1]
    h4_close = float(h4["close"].iloc[-1])
    if pd.isna(h4_fast) or pd.isna(h4_slow):
        return None
    if direction == "LONG" and not (h4_fast > h4_slow and h4_close > h4_slow):
        return None
    if direction == "SHORT" and not (h4_fast < h4_slow and h4_close < h4_slow):
        return None

    live_price = get_live_price(pair)
    if live_price is None or abs(live_price - float(last.close)) / float(last.close) > MAX_ENTRY_DEVIATION_PCT:
        return None

    entry = live_price
    if direction == "LONG":
        sl = float(last.low) - STOP_ATR_BUFFER * atr_val            # beyond the sweep wick
        risk = entry - sl
        tp1, tp2, tp3 = poc, vah, vah + (vah - val)
        targets_ok = entry < tp1 < tp2
    else:
        sl = float(last.high) + STOP_ATR_BUFFER * atr_val           # beyond the sweep wick
        risk = sl - entry
        tp1, tp2, tp3 = poc, val, val - (vah - val)
        targets_ok = entry > tp1 > tp2
    if risk <= 0 or not targets_ok:
        return None
    if entry <= 0 or abs(entry - sl) / entry < MIN_SL_DISTANCE_PCT:
        return None
    rr = abs(tp1 - entry) / abs(entry - sl)
    if not np.isfinite(rr) or rr < MIN_RR:
        return None

    return {
        "symbol": symbol, "pair": pair, "direction": direction, "entry": entry,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": rr,
        "strategy_type": "Liquidity Sweep & Value Area Rejection (LVAR)",
        "strategy_version": STRATEGY_VERSION, "swept_level": float(swept_level),
        "setup_close": float(last.close), "setup_high": float(last.high),
        "setup_low": float(last.low), "setup_candle_time": int(last.time),
        "live_price": live_price, "signal_time": utc_now().isoformat(),
    }


def load_log() -> list:
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logging.error("Signal log is unreadable; keeping a .corrupt copy and starting fresh.")
        try:
            os.replace(LOG_PATH, LOG_PATH.with_suffix(LOG_PATH.suffix + ".corrupt"))
        except OSError:
            pass
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
    return any(r.get("pair", f"{r.get('symbol', '')}-USDT") == pair and
               r.get("direction") == direction and r.get("status") in open_statuses
               for r in records if isinstance(r, dict))


def has_same_setup(records: list, pair: str, setup_time: int) -> bool:
    return any(isinstance(r, dict) and r.get("pair") == pair and r.get("setup_candle_time") == setup_time
               for r in records)


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
        f"📊 <b>{html.escape(signal['symbol'])}/USDT — SIGNAL</b>", "",
        f"{icon} <b>Direction:</b> {direction}",
        f"📈 <b>Strategy:</b> {html.escape(signal['strategy_type'])}", "",
        f"📍 <b>Entry:</b> <code>{fmt_price(signal['entry'])}</code>",
        f"🛑 <b>Stop Loss:</b> <code>{fmt_price(signal['sl'])}</code>", "",
        "🎯 <b>Take Profit:</b>", f"TP1 (POC) → <code>{fmt_price(signal['tp1'])}</code>",
        f"TP2 → <code>{fmt_price(signal['tp2'])}</code>", "",
        f"⚖️ <b>Risk/Reward:</b> 1:{signal['rr']:.2f}",
        "⚠️ <b>Risk Level:</b> INSTITUTIONAL", "", "🏦 <b>Exchange:</b> KuCoin Spot",
    ])


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials missing.")
        return False
    try:
        response = SESSION.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML",
                  "disable_web_page_preview": True}, timeout=REQUEST_TIMEOUT_SECONDS)
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


def candle_hits_level(candle: dict, level: float, direction: str, kind: str) -> bool:
    if kind == "entry":
        return candle["low"] <= level if direction == "LONG" else candle["high"] >= level
    if kind == "sl":
        return candle["low"] <= level if direction == "LONG" else candle["high"] >= level
    if kind in {"tp1", "tp2", "tp3"}:
        return candle["high"] >= level if direction == "LONG" else candle["low"] <= level
    raise ValueError("Unknown level kind")


def pending_candle_count(record: dict, candle_time: int) -> int:
    setup_time = int(record.get("setup_candle_time") or 0)
    if setup_time <= 0:
        signal_dt = parse_time(record.get("signal_time"))
        setup_time = int(signal_dt.timestamp()) if signal_dt else candle_time
    return max(0, (int(candle_time) - setup_time) // MONITOR_INTERVAL_SECONDS)


def _notify_and_mark(record: dict, message: str, updates: dict) -> None:
    sent = send_telegram_message(message)
    record.update(updates)
    record["last_notification_sent"] = bool(sent)
    if not sent:
        logging.error("State updated but Telegram notification was not delivered: %s", record.get("pair"))


def monitor_signals(records: list) -> bool:
    changed = False
    now = utc_now()
    active_statuses = {"ENTRY_PENDING", "OPEN", "TP1_HIT"}
    for record in records:
        if not isinstance(record, dict) or record.get("status") not in active_statuses:
            continue
        pair = record.get("pair")
        if not pair:
            continue
        symbol = html.escape(str(record.get("symbol", pair.replace("-USDT", ""))))
        direction = record.get("direction")
        try:
            entry, active_sl = float(record["entry"]), float(record["sl"])
            sl = float(record.get("initial_sl") or active_sl)   # original stop (for loss % and final R:R)
            tp1, tp2 = float(record["tp1"]), float(record["tp2"])
        except (KeyError, TypeError, ValueError):
            logging.warning("Invalid levels in signal record: %s", pair)
            continue
        if direction not in {"LONG", "SHORT"}:
            continue

        status = record["status"]
        if status in {"OPEN", "TP1_HIT"}:
            entry_time = parse_time(record.get("entry_time"))
            if entry_time and now - entry_time >= timedelta(days=TIMEOUT_DAYS):
                message = (f"⏱️ <b>TRADE TIMEOUT — {symbol}/USDT</b>\n\n"
                    f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                    f"📍 Entry: <code>{fmt_price(entry)}</code>\n\n"
                    "پس از ۱۰ روز بدون نتیجه قطعی بسته شد.")
                _notify_and_mark(record, message, {"status": "CLOSED_TIMEOUT", "result": "TIMEOUT",
                    "closed_time": now.isoformat()})
                changed = True
                continue

        setup_time = int(record.get("setup_candle_time") or 0)
        if not setup_time:
            sig_dt = parse_time(record.get("signal_time"))
            setup_time = int(sig_dt.timestamp()) if sig_dt else int(now.timestamp()) - MONITOR_INTERVAL_SECONDS
        last_processed = int(record.get("last_monitor_candle_time") or setup_time)
        fetch_start = max(setup_time, last_processed - 2 * MONITOR_INTERVAL_SECONDS)
        candles = get_kucoin_candles(pair, MONITOR_CANDLE_TYPE, MONITOR_FETCH_LIMIT, start_at=fetch_start)
        if candles is None or candles.empty:
            continue
        new_candles = candles[candles.time > last_processed].sort_values("time")
        if new_candles.empty:
            continue

        for _, row in new_candles.iterrows():
            candle = {k: float(row[k]) for k in ("open", "high", "low", "close")}
            candle_time = int(row.time)
            current_status = record.get("status")
            if current_status not in active_statuses:
                break

            if current_status == "ENTRY_PENDING":
                setup_low = record.get("setup_low")
                setup_high = record.get("setup_high")
                invalid = False
                if setup_low is not None and direction == "LONG":
                    invalid = candle["close"] < float(setup_low)
                elif setup_high is not None and direction == "SHORT":
                    invalid = candle["close"] > float(setup_high)
                bars_waited = pending_candle_count(record, candle_time)
                if invalid:
                    msg = (f"⌛ <b>سفارش منقضی شد (Limit پر نشد): "
                           f"{symbol} — ورودی: {fmt_price(entry)}$</b>")
                    _notify_and_mark(record, msg, {"status": "CLOSED_EXPIRED", "result": "INVALIDATED",
                        "closed_time": datetime.fromtimestamp(candle_time, timezone.utc).isoformat(),
                        "last_monitor_candle_time": candle_time})
                    changed = True
                    break
                if candle_hits_level(candle, entry, direction, "entry"):
                    message = (f"➕ <b>ENTRY EXECUTED — {symbol}/USDT</b>\n\n"
                        f"📍 Entry: <code>{fmt_price(entry)}</code>\n"
                        f"{'🟢' if direction == 'LONG' else '🔴'} <b>{direction} ACTIVE</b>\n\n"
                        f"🎯 TP1 (POC): <code>{fmt_price(tp1)}</code>\n"
                        f"🎯 TP2: <code>{fmt_price(tp2)}</code>\n"
                        f"🛑 SL: <code>{fmt_price(sl)}</code>")
                    _notify_and_mark(record, message, {"status": "OPEN", "entered": True,
                        "entry_time": datetime.fromtimestamp(candle_time + MONITOR_INTERVAL_SECONDS, timezone.utc).isoformat(),
                        "last_monitor_candle_time": candle_time})
                    changed = True
                    continue   # entry candle itself is not used for exit checks (entry is effective at its close)
                else:
                    if bars_waited >= PENDING_MAX_15M_BARS:
                        msg = (f"⌛ <b>سفارش منقضی شد (Limit پر نشد): "
                               f"{symbol} — ورودی: {fmt_price(entry)}$</b>")
                        _notify_and_mark(record, msg, {"status": "CLOSED_EXPIRED", "result": "EXPIRED",
                            "closed_time": datetime.fromtimestamp(candle_time + MONITOR_INTERVAL_SECONDS, timezone.utc).isoformat(),
                            "last_monitor_candle_time": candle_time})
                        changed = True
                        break
                    record["last_monitor_candle_time"] = candle_time
                    changed = True
                    continue

            sl_hit = candle_hits_level(candle, active_sl, direction, "sl")
            tp2_hit = candle_hits_level(candle, tp2, direction, "tp2")
            tp1_hit = candle_hits_level(candle, tp1, direction, "tp1")
            
            if sl_hit:
                if current_status == "TP1_HIT":
                    message = (f"🛡️ <b>BREAK-EVEN STOP HIT — {symbol}/USDT</b>\n\n"
                        f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                        f"📍 Entry: <code>{fmt_price(entry)}</code> (Protected)\n\n"
                        "پوزیشن در نقطه سر به سر بسته شد.")
                    _notify_and_mark(record, message, {"status": "CLOSED_BE", "closed_time":
                        datetime.fromtimestamp(candle_time + MONITOR_INTERVAL_SECONDS, timezone.utc).isoformat(),
                        "result": "BREAKEVEN", "last_monitor_candle_time": candle_time})
                else:
                    loss_pct = abs(entry - sl) / entry * 100 if entry else 0.0
                    message = (f"🛑 <b>STOP LOSS HIT — {symbol}/USDT</b>\n\n"
                        f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                        f"📍 Entry: <code>{fmt_price(entry)}</code>\n"
                        f"🛑 SL: <code>{fmt_price(sl)}</code> ❌\n\n"
                        f"📉 <b>Loss: -{loss_pct:.2f}%</b>\n\n🔴 <b>TRADE CLOSED</b>")
                    _notify_and_mark(record, message, {"status": "CLOSED_SL", "closed_time":
                        datetime.fromtimestamp(candle_time + MONITOR_INTERVAL_SECONDS, timezone.utc).isoformat(),
                        "result": "SL", "last_monitor_candle_time": candle_time})
                changed = True
                break
                
            if tp2_hit:
                movement_pct = abs(tp2 - entry) / entry * 100 if entry else 0.0
                final_rr = abs(tp2 - entry) / abs(entry - sl) if entry != sl else 0.0
                message = (f"🏆 <b>TP2 HIT — {symbol}/USDT</b>\n\n"
                    f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
                    "✅ TP1 + TP2 hit\n\n"
                    f"📈 <b>Total Profit: +{movement_pct:.2f}%</b>\n"
                    f"⚖️ <b>Final R:R: 1:{final_rr:.2f}</b>\n\n🟢 <b>TRADE CLOSED</b>")
                _notify_and_mark(record, message, {"status": "CLOSED_TP2", "closed_time":
                    datetime.fromtimestamp(candle_time + MONITOR_INTERVAL_SECONDS, timezone.utc).isoformat(),
                    "result": "TP2", "last_monitor_candle_time": candle_time})
                changed = True
                break
                
            if current_status == "OPEN" and tp1_hit:
                record.setdefault("initial_sl", sl)
                record["sl"] = entry # انتقال حد ضرر به نقطه ورود پس از تاچ شدن TP1
                active_sl = entry    # applies immediately to the next candles of this same run
                message = (f"🎯 <b>TAKE PROFIT 1 HIT (POC) — {symbol}/USDT</b>\n\n"
                    f"{direction}\nTP1: <code>{fmt_price(tp1)}</code> ✅\n"
                    f"🛡️ <b>SL Moved to Break-Even:</b> <code>{fmt_price(entry)}</code>\n\n"
                    "Monitoring continues for TP2.")
                _notify_and_mark(record, message, {"status": "TP1_HIT", "tp1_time":
                    datetime.fromtimestamp(candle_time + MONITOR_INTERVAL_SECONDS, timezone.utc).isoformat(),
                    "last_monitor_candle_time": candle_time})
                changed = True
                continue
                
            record["last_monitor_candle_time"] = candle_time
            changed = True
    return changed


def main() -> None:
    logging.info("LVAR Signal scanner started.")
    records = load_log()
    try:
        changed = monitor_signals(records)
    except Exception:
        logging.exception("Monitoring error (state saved as-is; scan continues).")
        changed = True
    if changed:
        save_log(records)
    hour = utc_now().hour
    if not (SESSION_START_UTC <= hour < SESSION_END_UTC):
        logging.info("Outside the %02d:00-%02d:00 UTC trading session: monitoring only, no new scan.", SESSION_START_UTC, SESSION_END_UTC)
        return
    coins = get_top_coins(SCAN_TOP_N_COINS)
    if not coins:
        logging.error("Coin universe unavailable. Check KuCoin connectivity.")
        return
    checked_pairs, emitted = set(), 0
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
            if has_same_setup(records, pair, signal["setup_candle_time"]):
                logging.info("%s skipped: this setup candle was already signalled", pair)
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
                "id": f"{pair}_{int(time.time())}", "symbol": signal["symbol"],
                "pair": signal["pair"], "direction": signal["direction"],
                "strategy_type": signal["strategy_type"], "strategy_version": signal["strategy_version"],
                "swept_level": signal["swept_level"], "entry": signal["entry"],
                "sl": signal["sl"], "initial_sl": signal["sl"], "tp1": signal["tp1"], "tp2": signal["tp2"],
                "tp3": signal["tp3"], "rr": signal["rr"], "signal_time": signal["signal_time"],
                "setup_candle_time": signal["setup_candle_time"],
                "setup_high": signal["setup_high"], "setup_low": signal["setup_low"],
                "status": "ENTRY_PENDING", "entered": False, "entry_time": None,
                "closed_time": None, "result": None,
                "last_monitor_candle_time": signal["setup_candle_time"],
            }
            records.append(record)
            save_log(records)
            emitted += 1
            logging.info("LVAR Signal sent: %s %s RR=%.2f", pair, signal["direction"], signal["rr"])
        except Exception:
            logging.exception("Unexpected scan error for %s", pair)
        time.sleep(REQUEST_DELAY_SECONDS)
    logging.info("Scan finished: %d coins considered, %d new signals sent.", len(checked_pairs), emitted)


if __name__ == "__main__":
    main()
