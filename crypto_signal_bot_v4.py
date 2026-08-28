"""
Crypto Signal Bot V4 — Phase 5m: Full Pipeline with Fresh-BOS Fix (Production Candidate)
"""

import requests
import pandas as pd
import time
import json
import os

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "1d": "1day"}
TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
    "NEAR-USDT", "APT-USDT", "ARB-USDT", "OP-USDT"
]
LOOKBACK_RECENT = 40
MAX_EXTENSION_ATR = 2.5
MIN_SL_DISTANCE_PCT = 0.005


def safe_get(url, params=None, retries=3):
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(5)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        return resp
    return resp


def fetch_kucoin_candles(symbol, interval_key, end_at=None):
    params = {"symbol": symbol, "type": KUCOIN_INTERVALS[interval_key]}
    if end_at:
        params["endAt"] = int(end_at)
    resp = safe_get(KUCOIN_BASE, params=params)
    if resp is None or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if data.get("code") != "200000":
        return None
    raw = data.get("data", [])
    if not raw:
        return None
    rows = []
    for r in raw:
        if len(r) < 6:
            continue
        rows.append({"time": int(r[0]), "open": float(r[1]), "close": float(r[2]),
                     "high": float(r[3]), "low": float(r[4]), "volume": float(r[5])})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def get_ohlcv_v4(symbol, interval_key, total_candles=200):
    all_dfs = []
    end_at = None
    remaining = total_candles
    while remaining > 0:
        df = fetch_kucoin_candles(symbol, interval_key, end_at=end_at)
        if df is None or df.empty:
            break
        all_dfs.append(df)
        remaining -= len(df)
        end_at = int(df["time"].min()) - 1
        time.sleep(0.4)
        if len(df) < 100:
            break
    if not all_dfs:
        return None
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    full_df["dt"] = pd.to_datetime(full_df["time"], unit="s", utc=True)
    return full_df.tail(total_candles).reset_index(drop=True)


def drop_unclosed_candle(df, interval_key):
    if df is None or df.empty:
        return df
    interval_seconds = {"4h": 4 * 3600, "1d": 86400}
    now_ts = int(time.time())
    last_candle_time = int(df["time"].iloc[-1])
    duration = interval_seconds.get(interval_key, 0)
    if duration and now_ts < last_candle_time + duration:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def find_swings(df, left=3, right=3):
    highs = df["high"].values
    lows = df["low"].values
    swing_highs, swing_lows = [], []
    for i in range(left, len(df) - right):
        window_h = highs[i-left:i+right+1]
        window_l = lows[i-left:i+right+1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            swing_highs.append({"index": i, "price": highs[i], "time": df["dt"].iloc[i]})
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swing_lows.append({"index": i, "price": lows[i], "time": df["dt"].iloc[i]})
    return swing_highs, swing_lows


def classify_market_structure(df, left=3, right=3):
    swing_highs, swing_lows = find_swings(df, left, right)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"regime": "range", "swing_highs": swing_highs, "swing_lows": swing_lows}
    h1, h2 = swing_highs[-2:]
    l1, l2 = swing_lows[-2:]
    if h2["price"] > h1["price"] and l2["price"] > l1["price"]:
        regime = "up"
    elif h2["price"] < h1["price"] and l2["price"] < l1["price"]:
        regime = "down"
    else:
        regime = "range"
    return {"regime": regime, "swing_highs": swing_highs, "swing_lows": swing_lows}


def detect_bos_fresh_only(df, swing_highs, swing_lows, lookback_candles=LOOKBACK_RECENT):
    events = []
    recent_start = max(0, len(df) - lookback_candles)
    last_broken_high = None
    last_broken_low = None
    for i in range(recent_start, len(df)):
        close_price = df["close"].iloc[i]
        relevant_highs = [s for s in swing_highs if s["index"] < i]
        relevant_lows = [s for s in swing_lows if s["index"] < i]
        if relevant_highs:
            level = relevant_highs[-1]["price"]
            if close_price > level and level != last_broken_high:
                events.append({"type": "bullish_bos", "index": i, "time": df["dt"].iloc[i],
                              "close": close_price, "broken_level": level})
                last_broken_high = level
        if relevant_lows:
            level = relevant_lows[-1]["price"]
            if close_price < level and level != last_broken_low:
                events.append({"type": "bearish_bos", "index": i, "time": df["dt"].iloc[i],
                              "close": close_price, "broken_level": level})
                last_broken_low = level
    return events


def get_daily_regime(symbol):
    df_daily = get_ohlcv_v4(symbol, "1d", total_candles=100)
    if df_daily is None or len(df_daily) < 30:
        return None
    df_daily = drop_unclosed_candle(df_daily, "1d")
    structure = classify_market_structure(df_daily)
    mapping = {"up": "BULLISH", "down": "BEARISH", "range": "CHOPPY"}
    return mapping[structure["regime"]]


def global_regime_filter(daily_regime, requested_direction):
    if daily_regime is None:
        return False, "daily_regime_unavailable"
    direction_regime = "BULLISH" if requested_direction == "bullish" else "BEARISH"
    if daily_regime == "BULLISH" and direction_regime == "BEARISH":
        return False, "daily_bullish_short_forbidden"
    if daily_regime == "BEARISH" and direction_regime == "BULLISH":
        return False, "daily_bearish_long_forbidden"
    return True, "allowed"


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def detect_compression(df, idx, lookback=20, threshold=0.75):
    if idx < lookback * 2:
        return False, None
    recent_range = (df["high"] - df["low"]).iloc[idx-lookback:idx].mean()
    longer_range = (df["high"] - df["low"]).iloc[idx-lookback*2:idx].mean()
    if longer_range == 0:
        return False, None
    ratio = recent_range / longer_range
    return ratio < threshold, ratio


def detect_displacement(df, idx, avg_body, avg_volume, direction,
                          body_multiplier=1.2, volume_multiplier=1.3, close_position_pct=0.25):
    if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]) or avg_body.iloc[idx] == 0:
        return False
    row = df.iloc[idx]
    body_size = abs(row["close"] - row["open"])
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    body_ok = body_size >= body_multiplier * avg_body.iloc[idx]
    volume_ok = row["volume"] >= volume_multiplier * avg_volume.iloc[idx]
    if direction == "bullish":
        close_position = (row["close"] - row["low"]) / candle_range
    else:
        close_position = (row["high"] - row["close"]) / candle_range
    close_ok = close_position >= (1 - close_position_pct)
    return body_ok and volume_ok and close_ok


def detect_follow_through(df, breakout_idx, direction, candles_after=2):
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return None
    breakout_close = df["close"].iloc[breakout_idx]
    after = df.iloc[breakout_idx+1:end_idx]
    if direction == "bullish":
        return bool((after["close"] > breakout_close).any())
    return bool((after["close"] < breakout_close).any())


def check_anti_chasing(df, idx, direction, avg_range, structure, max_extension_atr=MAX_EXTENSION_ATR):
    if pd.isna(avg_range.iloc[idx]) or avg_range.iloc[idx] == 0:
        return False, None
    current_price = df["close"].iloc[idx]
    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < idx]
        if not relevant:
            return False, None
        origin_price = relevant[-1]["price"]
        extension = current_price - origin_price
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
        if not relevant:
            return False, None
        origin_price = relevant[-1]["price"]
        extension = origin_price - current_price
    extension_atr = extension / avg_range.iloc[idx]
    return extension_atr <= max_extension_atr, extension_atr


def compute_sl_tp(df, idx, direction, structure, entry_price):
    """SL بر اساس آخرین Swing مخالف جهت + Buffer، TP بر اساس ساختار موجود"""
    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < idx]
        if not relevant:
            return None, None, None
        sl = relevant[-1]["price"] * 0.997
        opp = [s for s in structure["swing_highs"] if s["price"] > entry_price]
        if opp:
            tp1 = sorted(opp, key=lambda s: s["price"])[0]["price"]
        else:
            risk = entry_price - sl
            tp1 = entry_price + risk * 2
        tp2 = entry_price + (tp1 - entry_price) * 1.6
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
        if not relevant:
            return None, None, None
        sl = relevant[-1]["price"] * 1.003
        opp = [s for s in structure["swing_lows"] if s["price"] < entry_price]
        if opp:
            tp1 = sorted(opp, key=lambda s: s["price"], reverse=True)[0]["price"]
        else:
            risk = sl - entry_price
            tp1 = entry_price - risk * 2
        tp2 = entry_price - (entry_price - tp1) * 1.6
    return sl, tp1, tp2


def sl_distance_ok(entry, sl):
    if entry == 0:
        return False
    return abs(entry - sl) / entry >= MIN_SL_DISTANCE_PCT


def scan_symbol_full(symbol):
    log = {"symbol": symbol, "setups": []}
    df4h = get_ohlcv_v4(symbol, "4h", total_candles=200)
    if df4h is None or len(df4h) < 60:
        log["error"] = "4h_data_unavailable"
        return log
    df4h = drop_unclosed_candle(df4h, "4h")
    structure = classify_market_structure(df4h)
    bos_events = detect_bos_fresh_only(df4h, structure["swing_highs"], structure["swing_lows"])
    daily_regime = get_daily_regime(symbol)
    log["daily_regime"] = daily_regime
    log["h4_regime"] = structure["regime"]
    avg_body = compute_avg_body(df4h)
    avg_volume = compute_avg_volume(df4h)
    avg_range = compute_avg_range(df4h)

    for bos in bos_events:
        idx = bos["index"]
        direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

        allowed, _ = global_regime_filter(daily_regime, direction)
        if not allowed:
            continue
        if not detect_displacement(df4h, idx, avg_body, avg_volume, direction):
            continue
        ft = detect_follow_through(df4h, idx, direction)
        if ft is None or not ft:
            continue
        ac_ok, ext = check_anti_chasing(df4h, idx, direction, avg_range, structure)
        if not ac_ok:
            continue

        entry_price = bos["close"]
        sl, tp1, tp2 = compute_sl_tp(df4h, idx, direction, structure, entry_price)
        if sl is None or not sl_distance_ok(entry_price, sl):
            continue
        risk = abs(entry_price - sl)
        reward1 = abs(tp1 - entry_price)
        if risk <= 0 or reward1 <= 0:
            continue
        rr = reward1 / risk
        if rr < 1.5:
            continue

        compression_ok, _ = detect_compression(df4h, idx)
        path = "A" if compression_ok else "B"

        log["setups"].append({
            "time": str(bos["time"]), "direction": "LONG" if direction == "bullish" else "SHORT",
            "path": path, "entry": round(entry_price, 6), "sl": round(sl, 6),
            "tp1": round(tp1, 6), "tp2": round(tp2, 6), "rr": round(rr, 2),
            "extension_atr": round(ext, 2), "daily_regime": daily_regime, "h4_regime": structure["regime"],
        })
    return log


if __name__ == "__main__":
    print("PHASE: 5m")
    print("STATUS: EXECUTING")
    print("=" * 70)

    total_setups = 0
    all_setups = []

    for symbol in TEST_SYMBOLS:
        result = scan_symbol_full(symbol)
        if "error" in result:
            print(f"\n{symbol} | ERROR")
            continue
        print(f"\n{symbol} | Daily={result['daily_regime']} | 4H={result['h4_regime']}")
        for s in result["setups"]:
            total_setups += 1
            s["symbol"] = symbol
            all_setups.append(s)
            print(f"  SETUP PATH {s['path']} | {s['direction']} | Entry={s['entry']} | SL={s['sl']} | "
                  f"TP1={s['tp1']} | TP2={s['tp2']} | R:R={s['rr']} | Ext={s['extension_atr']}")
        if not result["setups"]:
            print("  (no valid setup)")
        time.sleep(1)

    print("\n" + "=" * 70)
    print(f"WHAT WAS TESTED: Full Pipeline (BOS-fresh -> Daily -> Displacement -> FollowThru -> AntiChase -> SL/TP/RR)")
    print(f"FINDINGS: {total_setups} Setup نهایی معتبر از 14 نماد")
    print(f"DECISION: در صورت وجود Setup، این Pipeline به‌عنوان V4 Production Candidate ثبت می‌شود")
    print(f"NEXT ACTION: اگر Setup>0، ادامه به Full Backtest (Phase 6). اگر 0، ادامه Audit با نمادهای بیشتر")
    print("=" * 70)
