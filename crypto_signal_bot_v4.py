"""
Crypto Signal Bot V4 — Phase 5aa: Controlled Displacement Bypass (A/B Funnel Comparison)
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


def displacement_details(df, idx, avg_body, avg_volume, direction,
                           body_multiplier=1.2, volume_multiplier=1.3, close_position_pct=0.25):
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
    body_ok = body_ratio >= body_multiplier
    volume_ok = volume_ratio >= volume_multiplier
    close_ok = close_position >= (1 - close_position_pct)
    return {"body_ratio": body_ratio, "volume_ratio": volume_ratio,
            "close_position": close_position, "overall_pass": body_ok and volume_ok and close_ok}


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
    print("PHASE: 5aa")
    print("STATUS: EXECUTING - Controlled Displacement Bypass (A/B Comparison)")
    print("=" * 70)

    funnel_a = {"total_bos": 0, "daily_pass": 0, "displacement_pass": 0, "displacement_reject": 0,
                "ft_pass": 0, "ft_reject": 0, "antichase_pass": 0, "antichase_reject": 0}
    funnel_b = {"total_bos": 0, "daily_pass": 0, "ft_pass": 0, "ft_reject": 0,
                "antichase_pass": 0, "antichase_reject": 0}

    # BOS هایی که Displacement رد کرده - برای تحلیل مهم‌ترین معیار
    disp_rejected_but_ft = 0
    disp_rejected_but_ac = 0
    disp_rejected_but_both = 0
    disp_rejected_total = 0

    extension_values_b = []

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
            funnel_a["total_bos"] += 1
            funnel_b["total_bos"] += 1
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"

            if not global_regime_filter(daily_regime, direction):
                continue
            funnel_a["daily_pass"] += 1
            funnel_b["daily_pass"] += 1

            dd = displacement_details(df4h, idx, avg_body, avg_volume, direction)
            ft = detect_follow_through(df4h, idx, direction)
            ac_ok, ext = check_anti_chasing(df4h, idx, direction, avg_range, structure)

            # ===== FUNNEL A: با Displacement (فعلی) =====
            disp_pass_a = dd is not None and dd["overall_pass"]
            if disp_pass_a:
                funnel_a["displacement_pass"] += 1
                if ft is True:
                    funnel_a["ft_pass"] += 1
                    if ac_ok:
                        funnel_a["antichase_pass"] += 1
                    else:
                        funnel_a["antichase_reject"] += 1
                elif ft is False:
                    funnel_a["ft_reject"] += 1
            else:
                funnel_a["displacement_reject"] += 1
                # تحلیل مهم: چند مورد از رد‌شده‌های Displacement بعداً موفق بودن؟
                if dd is not None:
                    disp_rejected_total += 1
                    if ft is True:
                        disp_rejected_but_ft += 1
                    if ac_ok:
                        disp_rejected_but_ac += 1
                    if ft is True and ac_ok:
                        disp_rejected_but_both += 1

            # ===== FUNNEL B: بدون Displacement (Bypass) =====
            if ft is True:
                funnel_b["ft_pass"] += 1
                if ac_ok:
                    funnel_b["antichase_pass"] += 1
                    if ext is not None:
                        extension_values_b.append(ext)
                else:
                    funnel_b["antichase_reject"] += 1
            elif ft is False:
                funnel_b["ft_reject"] += 1
        time.sleep(0.5)

    print("\n" + "=" * 35)
    print("FUNNEL A — WITH Displacement (Current)")
    print("=" * 35)
    for k, v in funnel_a.items():
        print(f"  {k}: {v}")
    print(f"  Final Candidates (Displacement+FT+AntiChase all PASS): {funnel_a['antichase_pass']}")

    print("\n" + "=" * 35)
    print("FUNNEL B — WITHOUT Displacement (Bypass)")
    print("=" * 35)
    for k, v in funnel_b.items():
        print(f"  {k}: {v}")
    print(f"  Final Candidates (FT+AntiChase PASS, no Displacement): {funnel_b['antichase_pass']}")

    print("\n" + "=" * 70)
    print("KEY METRIC: از میان BOS هایی که Displacement رد کرده:")
    print(f"  کل رد‌شده توسط Displacement: {disp_rejected_total}")
    print(f"  از این‌ها، Follow-through موفق داشتن: {disp_rejected_but_ft} "
          f"({round(disp_rejected_but_ft/disp_rejected_total*100,1) if disp_rejected_total else 0}%)")
    print(f"  از این‌ها، Anti-Chasing رو پاس کردن: {disp_rejected_but_ac} "
          f"({round(disp_rejected_but_ac/disp_rejected_total*100,1) if disp_rejected_total else 0}%)")
    print(f"  از این‌ها، هر دو (FT+AC) رو پاس کردن: {disp_rejected_but_both} "
          f"({round(disp_rejected_but_both/disp_rejected_total*100,1) if disp_rejected_total else 0}%)")

    print("\n" + "=" * 70)
    print(f"FUNNEL B Extension distribution (روی Candidate های نهایی، n={len(extension_values_b)}):")
    if extension_values_b:
        s = sorted(extension_values_b)
        print(f"  Min={min(s):.2f} Median={s[len(s)//2]:.2f} Max={max(s):.2f}")

    print(f"\nمقایسه نهایی: FUNNEL A داد {funnel_a['antichase_pass']} Candidate | "
          f"FUNNEL B داد {funnel_b['antichase_pass']} Candidate")
    print("=" * 70)
