"""
Crypto Signal Bot V3 — replacement strategy
Strategy: Daily + 4H trend alignment, followed by a confirmed 15m pullback.
Uses real KuCoin Spot OHLCV candles. No synthetic candles and no order execution.

Dependencies:
    pip install requests pandas numpy

Environment variables:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Important:
- This script only scans and sends signals. It does not place trades.
- Validate on paper before relying on signals. No strategy can guarantee profits.
"""

import html
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------- Configuration ----------------
KUCOIN_BASE = "https://api.kucoin.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

SCAN_TOP_N_COINS = 40
REQUEST_DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3

MIN_RR = 2.0
MIN_SL_DISTANCE_PCT = 0.005       # Reject stops closer than 0.5% to entry.
STOP_ATR_BUFFER = 0.20            # Buffer beyond the 15m swing.
PULLBACK_EMA_TOLERANCE_ATR = 0.35
MAX_ENTRY_DEVIATION_PCT = 0.015   # Reject if live price is >1.5% from setup close.
MIN_4H_CANDLES = 70
MIN_DAILY_CANDLES = 35
MIN_15M_CANDLES = 60
MAX_SIGNALS_PER_RUN = 2

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
LOG_PATH = Path(os.getenv("SIGNALS_LOG_PATH", "signals_log.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoSignalBotV3/1.0"})


def safe_get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """GET JSON with bounded retries, timeout, and rate-limit handling."""
    for attempt in range(MAX_RETRIES):
        try:
            response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    logging.warning("Non-JSON response from %s", url)
                    return None
                return payload if isinstance(payload, (dict, list)) else None
            if response.status_code == 429 or 500 <= response.status_code < 600:
                wait = min(2 ** attempt * 2, 12)
                logging.warning("HTTP %s from %s; retrying in %ss",
                                response.status_code, url, wait)
                time.sleep(wait)
                continue
            logging.warning("HTTP %s from %s", response.status_code, url)
            return None
        except requests.RequestException as exc:
            logging.warning("Request error for %s: %s", url, exc)
            time.sleep(min(2 ** attempt, 8))
    return None


def get_top_coins(limit: int) -> list:
    """CoinGecko is used only for the universe; all candle data comes from KuCoin."""
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
    if not payload or not isinstance(payload, list):
        return []

    coins = []
    seen = set()
    for coin in payload:
        if not isinstance(coin, dict):
            continue
        symbol = str(coin.get("symbol", "")).upper().strip()
        price = coin.get("current_price") or 0
        change = coin.get("price_change_percentage_24h") or 0
        coin_id = str(coin.get("id", "")).strip()
        # Exclude stablecoins and malformed symbols.
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


def get_kucoin_candles(pair: str, candle_type: str, limit: int) -> Optional[pd.DataFrame]:
    """
    KuCoin Spot candles response order:
    [time, open, close, high, low, volume, turnover].
    candle_type: 15min, 4hour, 1day.
    API data is returned newest-first; normalize to oldest-first and remove
    the currently forming candle using its opening timestamp and interval.
    """
    intervals = {"15min": 15 * 60, "4hour": 4 * 3600, "1day": 24 * 3600}
    if candle_type not in intervals:
        raise ValueError(f"Unsupported candle type: {candle_type}")

    now = int(time.time())
    interval = intervals[candle_type]
    # KuCoin's public endpoint returns up to 1500 candles; query a bounded window.
    start_at = now - (int(limit) + 3) * interval
    payload = safe_get_json(
        f"{KUCOIN_BASE}/api/v1/market/candles",
        {"symbol": pair, "type": candle_type, "startAt": start_at, "endAt": now},
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

    df = pd.DataFrame(rows).drop_duplicates(subset=["time"]).sort_values("time")
    df = df[(df["open"] > 0) & (df["close"] > 0) &
            (df["high"] >= df["low"]) & (df["volume"] >= 0)]
    # Keep only completed candles. A candle is complete only after its interval ends.
    df = df[df["time"] + interval <= now - 2]
    df = df.tail(int(limit)).reset_index(drop=True)
    if df.empty:
        return None
    return df


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
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def classify_trend(df: pd.DataFrame) -> str:
    """Trend filter combines EMA alignment with recent confirmed swing structure."""
    if df is None or len(df) < 55:
        return "choppy"
    close = df["close"]
    e20, e50 = ema(close, 20), ema(close, 50)
    if pd.isna(e20.iloc[-1]) or pd.isna(e50.iloc[-1]):
        return "choppy"

    # Confirmed pivots only: pivot at i needs two candles to its right.
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    pivot_highs, pivot_lows = [], []
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


def recent_swing(df: pd.DataFrame, direction: str, lookback: int = 12) -> Optional[float]:
    recent = df.iloc[-lookback - 1:-1]
    if recent.empty:
        return None
    return float(recent["low"].min() if direction == "LONG" else recent["high"].max())


def detect_setup(pair: str, symbol: str) -> Optional[dict]:
    """Return a setup only when daily and 4H direction agree and 15m confirms pullback."""
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
    e20, e50 = ema(close, 20), ema(close, 50)
    atr15 = atr(m15, 14)
    if any(pd.isna(x) for x in (e20.iloc[-1], e20.iloc[-2],
                                e50.iloc[-1], atr15.iloc[-1])):
        return None

    last = m15.iloc[-1]  # last fully closed candle only
    prev = m15.iloc[-2]
    a = float(atr15.iloc[-1])
    if a <= 0 or not np.isfinite(a):
        return None

    tolerance = a * PULLBACK_EMA_TOLERANCE_ATR
    if direction == "LONG":
        # Pullback approaches EMA20/EMA50, then a bullish close reclaims EMA20.
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
        logging.info("%s rejected: live price too far from setup close (%.2f%%)",
                     pair, deviation * 100)
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
        "signal_time": datetime.now(timezone.utc).isoformat(),
    }


def load_log() -> list:
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logging.warning("Signal log is unreadable; starting with an empty in-memory log.")
        return []


def save_log(records: list) -> None:
    """Atomic write prevents a partial JSON file if the process stops mid-write."""
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
    lines = [
        f"<b>{html.escape(signal['symbol'])}/USDT — V3 SIGNAL</b>",
        "",
        f"{icon} <b>Direction:</b> {direction}",
        f"<b>Strategy:</b> {html.escape(signal['strategy_type'])}",
        f"<b>Entry:</b> {fmt_price(signal['entry'])}",
        f"<b>Stop Loss:</b> {fmt_price(signal['sl'])}",
        f"<b>TP1 (minimum 1:{MIN_RR:.0f}):</b> {fmt_price(signal['tp1'])}",
        f"<b>TP2 (3R):</b> {fmt_price(signal['tp2'])}",
        f"<b>Risk/Reward:</b> 1:{signal['rr']:.2f}",
        f"<b>Daily / 4H:</b> {signal['daily_trend'].upper()} / {signal['h4_trend'].upper()}",
        "<b>Exchange:</b> KuCoin Spot",
        "<i>Signal only; no order was placed. Confirm spread and execution before trading.</i>",
    ]
    return "\n".join(lines)


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials missing; message will only be printed.")
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
            logging.error("Telegram API rejected message: %s", payload)
            return False
        return True
    except (requests.RequestException, ValueError) as exc:
        logging.error("Telegram send failed: %s", exc)
        return False


def main() -> None:
    logging.info("V3 replacement scanner started; real KuCoin candles only.")
    coins = get_top_coins(SCAN_TOP_N_COINS)
    if not coins:
        logging.error("Coin universe unavailable. Check CoinGecko connectivity/rate limits.")
        return

    records = load_log()
    candidates = []
    checked_pairs = set()

    for coin in coins:
        pair = coin["pair"]
        if pair in checked_pairs:
            continue
        checked_pairs.add(pair)
        try:
            signal = detect_setup(pair, coin["symbol"])
            if signal:
                if has_open_duplicate(records, pair, signal["direction"]):
                    logging.info("%s skipped: duplicate open signal", pair)
                else:
                    candidates.append(signal)
                    logging.info("Candidate: %s %s RR=%.2f",
                                 pair, signal["direction"], signal["rr"])
        except Exception:
            logging.exception("Unexpected scan error for %s", pair)
        time.sleep(REQUEST_DELAY_SECONDS)

    candidates.sort(key=lambda s: (s["rr"], s["symbol"]), reverse=True)
    selected = candidates[:MAX_SIGNALS_PER_RUN]

    for signal in selected:
        message = format_signal(signal)
        print("\n" + message.replace("<b>", "").replace("</b>", "")
              .replace("<i>", "").replace("</i>", "") + "\n")
        # Log signal before delivery so repeated runs don't emit duplicates.
        records.append({
            "id": f"{signal['pair']}_{int(time.time())}",
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
        })
        save_log(records)
        send_telegram_message(message)

    logging.info("Scan finished: %d coins considered, %d setups, %d signals emitted.",
                 len(coins), len(candidates), len(selected))


if __name__ == "__main__":
    main()
