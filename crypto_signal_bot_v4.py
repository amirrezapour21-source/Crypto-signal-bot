"""
Crypto Signal Bot V4 — Phase 5p: SL/TP debug for the single surviving candidate
"""

import requests
import pandas as pd
import time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_INTERVALS = {"4h": "4hour", "1d": "1day"}
TEST_SYMBOLS = ["OP-USDT"]
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


def compute_avg_body(df, lookback=20):
    return (df["close"] - df["open"]).abs().rolling(lookback).mean()


def compute_avg_volume(df, lookback=20):
    return df["volume"].rolling(lookback).mean()


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


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


if __name__ == "__main__":
    print("PHASE: 5p")
    print("STATUS: EXECUTING - OP-USDT deep trace")
    print("=" * 70)

    df4h = get_ohlcv_v4("OP-USDT", "4h", total_candles=200)
    df4h = drop_unclosed_candle(df4h, "4h")
    structure = classify_market_structure(df4h)
    bos_events = detect_bos_fresh_only(df4h, structure["swing_highs"], structure["swing_lows"])
    avg_body = compute_avg_body(df4h)
    avg_volume = compute_avg_volume(df4h)
    avg_range = compute_avg_range(df4h)

    print(f"Total candles: {len(df4h)}")
    print(f"4H Regime: {structure['regime']}")
    print(f"Swing Highs: {len(structure['swing_highs'])} | Swing Lows: {len(structure['swing_lows'])}")
    print(f"Fresh BOS events: {len(bos_events)}\n")

    for bos in bos_events:
        idx = bos["index"]
        direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
        disp = detect_displacement(df4h, idx, avg_body, avg_volume, direction)
        ft = detect_follow_through(df4h, idx, direction)
        ac_ok, ext = check_anti_chasing(df4h, idx, direction, avg_range, structure) if not pd.isna(avg_range.iloc[idx]) else (None, None)

        print(f"BOS idx={idx} | {direction} | time={bos['time']} | close={bos['close']:.6f} | "
              f"broken_level={bos['broken_level']:.6f} | Disp={disp} | FT={ft} | AC={ac_ok}({ext})")

        if direction == "bearish" and disp and ft and ac_ok:
            print("\n  --- TRACING SL/TP FOR THIS CANDIDATE ---")
            entry_price = bos["close"]
            relevant = [s for s in structure["swing_highs"] if s["index"] < idx]
            print(f"  Relevant swing_highs before idx={idx}: {len(relevant)}")
            if relevant:
                sl = relevant[-1]["price"] * 1.003
                print(f"  Nearest swing_high used for SL: index={relevant[-1]['index']}, price={relevant[-1]['price']:.6f}")
                print(f"  Computed SL = {sl:.6f}")
                sl_dist_pct = abs(entry_price - sl) / entry_price * 100
                print(f"  SL distance from entry: {sl_dist_pct:.4f}%  (min required: {MIN_SL_DISTANCE_PCT*100}%)")

                opp = [s for s in structure["swing_lows"] if s["price"] < entry_price]
                print(f"  Swing_lows below entry_price ({entry_price:.6f}): {len(opp)}")
                if opp:
                    tp1 = sorted(opp, key=lambda s: s["price"], reverse=True)[0]["price"]
                    print(f"  TP1 (nearest swing_low below entry) = {tp1:.6f}")
                else:
                    risk = sl - entry_price
                    tp1 = entry_price - risk * 2
                    print(f"  No swing_low below entry -> fallback TP1 = {tp1:.6f} (2R fallback)")

                risk = abs(entry_price - sl)
                reward1 = abs(tp1 - entry_price)
                print(f"  Risk={risk:.6f} | Reward1={reward1:.6f}")
                if risk > 0:
                    rr = reward1 / risk
                    print(f"  R:R = {rr:.2f}  (min required: 1.5)")
                else:
                    print("  Risk = 0, invalid")
            else:
                print("  NO relevant swing_high found before this BOS -> SL cannot be computed -> REJECT")
        print()

    print("=" * 70)
