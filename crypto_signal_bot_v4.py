"""
Crypto Signal Bot V4 — Phase 6: New Production Funnel + Full Trade Parameters
(Displacement moved to Shadow Mode per Phase 5aa decision)

Funnel:
Daily Regime -> Fresh BOS -> Follow-through -> Anti-Chasing ->
Structural Liquidity Target (Equal-Level, Phase 5u) ->
Invalidation-based SL (Phase 5v Model C) -> Valid R:R -> Final Candidate

Displacement metrics همچنان محاسبه و ثبت می‌شن (Shadow Feature)،
ولی دیگه باعث Reject نمی‌شن.
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "1d": "1day"}
TEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
    "NEAR-USDT", "APT-USDT", "ARB-USDT", "OP-USDT",
    "SUI-USDT", "INJ-USDT", "TIA-USDT", "SEI-USDT", "FIL-USDT",
    "ATOM-USDT", "LTC-USDT", "ETC-USDT", "TRX-USDT", "ICP-USDT",
    "AAVE-USDT", "UNI-USDT", "MKR-USDT", "RUNE-USDT", "FTM-USDT", "GRT-USDT",
    "ALGO-USDT", "VET-USDT", "HBAR-USDT", "EGLD-USDT", "XLM-USDT",
    "THETA-USDT", "SAND-USDT", "MANA-USDT", "AXS-USDT", "CHZ-USDT",
    "COMP-USDT", "SNX-USDT", "CRV-USDT", "LDO-USDT", "DYDX-USDT",
    "GMX-USDT", "STX-USDT", "KAVA-USDT", "ZIL-USDT", "ONE-USDT"
]
LOOKBACK_RECENT = 60
MAX_EXTENSION_ATR = 2.5
MIN_SL_DISTANCE_PCT = 0.005
MIN_RR = 1.5
EQUAL_LEVEL_TOLERANCE_ATR = 0.15


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


def get_ohlcv_v4(symbol, interval_key, total_candles=250):
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
        time.sleep(0.2)
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
        return False
    direction_regime = "BULLISH" if requested_direction == "bullish" else "BEARISH"
    if daily_regime == "BULLISH" and direction_regime == "BEARISH":
        return False
    if daily_regime == "BEARISH" and direction_regime == "BULLISH":
        return False
    return True


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def displacement_shadow(df, idx, avg_body, avg_volume, direction):
    """فقط برای ثبت (Shadow) - دیگه باعث Reject نمی‌شه"""
    if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]) or avg_body.iloc[idx] == 0:
        return None
    row = df.iloc[idx]
    body_size = abs(row["close"] - row["open"])
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return None
    body_ratio = body_size / avg_body.iloc[idx]
    volume_ratio = row["volume"] / avg_volume.iloc[idx]
    if direction == "bullish":
        close_position = (row["close"] - row["low"]) / candle_range
    else:
        close_position = (row["high"] - row["close"]) / candle_range
    return {"body_ratio": round(body_ratio, 2), "volume_ratio": round(volume_ratio, 2),
            "close_position": round(close_position, 2)}


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


def is_structurally_strong(swing, pool, atr):
    if atr is None or atr == 0:
        return False
    neighbor_distances = [abs(other["price"] - swing["price"]) / atr
                           for other in pool if other["index"] != swing["index"]]
    min_dist = min(neighbor_distances) if neighbor_distances else None
    return min_dist is not None and min_dist <= EQUAL_LEVEL_TOLERANCE_ATR


def sl_distance_ok(entry, sl):
    if entry == 0 or sl is None:
        return False
    return abs(entry - sl) / entry >= MIN_SL_DISTANCE_PCT


def build_candidate(symbol, direction, idx, bos, df, structure, atr, daily_regime, disp_shadow, ext):
    entry_price = bos["close"]

    # Structural Liquidity Target (Phase 5u: نزدیک‌ترین STRONG equal-level، بدون Look-Ahead)
    if direction == "bullish":
        target_pool = [s for s in structure["swing_highs"] if s["price"] > entry_price and s["index"] < idx]
    else:
        target_pool = [s for s in structure["swing_lows"] if s["price"] < entry_price and s["index"] < idx]

    if not target_pool:
        return None, "no_target_pool"

    strong_targets = [s for s in target_pool if is_structurally_strong(s, target_pool, atr)]
    use_pool = strong_targets if strong_targets else target_pool
    target_type = "STRONG (equal-level)" if strong_targets else "POTENTIAL (isolated swing)"
    sorted_pool = sorted(use_pool, key=lambda s: s["price"], reverse=(direction == "bearish"))
    tp1 = sorted_pool[0]["price"]
    tp2 = sorted_pool[-1]["price"] if len(sorted_pool) > 1 else tp1

    # Invalidation-based SL (Phase 5v Model C: خود broken_level BOS)
    buffer_pct = 1.003 if direction == "bearish" else 0.997
    sl = bos["broken_level"] * buffer_pct

    if not sl_distance_ok(entry_price, sl):
        return None, "sl_too_close"

    risk = abs(entry_price - sl)
    reward1 = abs(tp1 - entry_price)
    if risk == 0 or reward1 == 0:
        return None, "invalid_risk_reward"
    rr1 = reward1 / risk
    if rr1 < MIN_RR:
        return None, f"rr_too_low ({rr1:.2f})"

    reward2 = abs(tp2 - entry_price)
    rr2 = reward2 / risk

    return {
        "symbol": symbol, "direction": "LONG" if direction == "bullish" else "SHORT",
        "entry": round(entry_price, 6), "sl": round(sl, 6),
        "tp1": round(tp1, 6), "tp2": round(tp2, 6),
        "risk_pct": round(risk / entry_price * 100, 3),
        "rr1": round(rr1, 2), "rr2": round(rr2, 2),
        "target_type": target_type, "daily_regime": daily_regime,
        "extension_atr": round(ext, 2), "displacement_shadow": disp_shadow,
        "bos_time": str(bos["time"]), "bos_index": idx,
    }, "PASS"


if __name__ == "__main__":
    print("PHASE: 6")
    print("STATUS: EXECUTING - New Production Funnel (Displacement=Shadow)")
    print("=" * 70)

    final_candidates = []
    reject_reasons = {}
    funnel = {"total_bos": 0, "daily_pass": 0, "ft_pass": 0, "antichase_pass": 0,
              "target_sl_rr_pass": 0}

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=250)
        if df4h is None or len(df4h) < 80:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        structure = classify_market_structure(df4h)
        bos_events = detect_bos_fresh_only(df4h, structure["swing_highs"], structure["swing_lows"])
        daily_regime = get_daily_regime(symbol)
        avg_body = compute_avg_body(df4h)
        avg_volume = compute_avg_volume(df4h)
        avg_range = compute_avg_range(df4h)

        for bos in bos_events:
            funnel["total_bos"] += 1
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

            if not global_regime_filter(daily_regime, direction):
                continue
            funnel["daily_pass"] += 1

            ft = detect_follow_through(df4h, idx, direction)
            if ft is None or not ft:
                continue
            funnel["ft_pass"] += 1

            atr = avg_range.iloc[idx]
            if pd.isna(atr) or atr == 0:
                continue

            ac_ok, ext = check_anti_chasing(df4h, idx, direction, avg_range, structure)
            if not ac_ok:
                continue
            funnel["antichase_pass"] += 1

            disp_shadow = displacement_shadow(df4h, idx, avg_body, avg_volume, direction)

            candidate, reason = build_candidate(symbol, direction, idx, bos, df4h, structure,
                                                  atr, daily_regime, disp_shadow, ext)
            if candidate:
                funnel["target_sl_rr_pass"] += 1
                final_candidates.append(candidate)
            else:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        time.sleep(0.5)

    print("\n--- FUNNEL ---")
    for k, v in funnel.items():
        print(f"  {k}: {v}")

    print(f"\n--- REJECT REASONS (بعد از Anti-Chasing) ---")
    for k, v in reject_reasons.items():
        print(f"  {k}: {v}")

    print(f"\n--- FINAL CANDIDATES: {len(final_candidates)} ---\n")
    for c in final_candidates:
        print(f"{c['symbol']} | {c['direction']} | Entry={c['entry']} | SL={c['sl']} | "
              f"TP1={c['tp1']} | TP2={c['tp2']} | Risk%={c['risk_pct']} | "
              f"RR1={c['rr1']} | RR2={c['rr2']} | Target={c['target_type']} | "
              f"Ext={c['extension_atr']} | Disp(shadow)={c['displacement_shadow']}")

    print("=" * 70)
