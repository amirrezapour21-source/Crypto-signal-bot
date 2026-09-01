"""
Crypto Signal Bot V4 — Phase 8: SL Buffer / Retest-Tolerance Audit + Look-Ahead Timing Check
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
MAX_HOLD_CANDLES = 60
FOLLOW_THROUGH_CANDLES = 2


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


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def detect_follow_through(df, breakout_idx, direction, candles_after=FOLLOW_THROUGH_CANDLES):
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return None, None
    breakout_close = df["close"].iloc[breakout_idx]
    after = df.iloc[breakout_idx+1:end_idx]
    if direction == "bullish":
        confirmed = after["close"] > breakout_close
    else:
        confirmed = after["close"] < breakout_close
    if confirmed.any():
        confirm_idx = breakout_idx + 1 + confirmed.values.argmax()
        return True, confirm_idx
    return False, None


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


def get_target_pool(direction, entry_price, structure, idx):
    if direction == "bullish":
        return [s for s in structure["swing_highs"] if s["price"] > entry_price and s["index"] < idx]
    return [s for s in structure["swing_lows"] if s["price"] < entry_price and s["index"] < idx]


def build_sl_variants(direction, bos, atr):
    """چهار مدل SL: A=Current(0.3%), B1/B2/B3 = ATR-aware با بافر 0.25/0.5/0.75"""
    broken_level = bos["broken_level"]
    variants = {}
    if direction == "bearish":
        variants["A_current_0.3pct"] = broken_level * 1.003
        variants["B_atr_0.25"] = broken_level + atr * 0.25
        variants["B_atr_0.50"] = broken_level + atr * 0.50
        variants["B_atr_0.75"] = broken_level + atr * 0.75
    else:
        variants["A_current_0.3pct"] = broken_level * 0.997
        variants["B_atr_0.25"] = broken_level - atr * 0.25
        variants["B_atr_0.50"] = broken_level - atr * 0.50
        variants["B_atr_0.75"] = broken_level - atr * 0.75
    return variants


def simulate_outcome(df, direction, entry, sl, tp1, tp2, start_idx, max_hold=MAX_HOLD_CANDLES):
    end_idx = min(start_idx + 1 + max_hold, len(df))
    mfe, mae = 0, 0
    tp1_hit = False
    for i in range(start_idx + 1, end_idx):
        row = df.iloc[i]
        if direction == "bullish":
            favorable = row["high"] - entry
            adverse = entry - row["low"]
            sl_hit = row["low"] <= sl
            tp1_now = row["high"] >= tp1
            tp2_now = row["high"] >= tp2
        else:
            favorable = entry - row["low"]
            adverse = row["high"] - entry
            sl_hit = row["high"] >= sl
            tp1_now = row["low"] <= tp1
            tp2_now = row["low"] <= tp2
        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if not tp1_hit:
            if sl_hit:
                return {"result": "SL_HIT", "bars_held": i - start_idx, "mfe": mfe, "mae": mae}
            if tp1_now:
                tp1_hit = True
                if tp2_now:
                    return {"result": "TP2_HIT", "bars_held": i - start_idx, "mfe": mfe, "mae": mae}
        else:
            if sl_hit:
                return {"result": "TP1_THEN_SL", "bars_held": i - start_idx, "mfe": mfe, "mae": mae}
            if tp2_now:
                return {"result": "TP2_HIT", "bars_held": i - start_idx, "mfe": mfe, "mae": mae}
    if tp1_hit:
        return {"result": "TP1_ONLY_OPEN", "bars_held": end_idx - start_idx - 1, "mfe": mfe, "mae": mae}
    return {"result": "OPEN_NO_OUTCOME", "bars_held": end_idx - start_idx - 1, "mfe": mfe, "mae": mae}


def median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid-1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


if __name__ == "__main__":
    print("PHASE: 8")
    print("STATUS: EXECUTING - SL Buffer / Retest-Tolerance Audit + Timing Check")
    print("=" * 70)

    # داده Setup ها رو یک‌بار می‌سازیم (Immutable Snapshot طبق دستور بند ۱۲)
    setups = []
    look_ahead_flags = []

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=250)
        if df4h is None or len(df4h) < 80:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        structure = classify_market_structure(df4h)
        bos_events = detect_bos_fresh_only(df4h, structure["swing_highs"], structure["swing_lows"])
        daily_regime = get_daily_regime(symbol)
        avg_range = compute_avg_range(df4h)

        for bos in bos_events:
            idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
            if not global_regime_filter(daily_regime, direction):
                continue

            ft, confirm_idx = detect_follow_through(df4h, idx, direction)
            if not ft:
                continue

            atr = avg_range.iloc[idx]
            if pd.isna(atr) or atr == 0:
                continue
            ac_ok, ext = check_anti_chasing(df4h, idx, direction, avg_range, structure)
            if not ac_ok:
                continue

            entry_price = bos["close"]
            target_pool = get_target_pool(direction, entry_price, structure, idx)
            if not target_pool:
                continue
            strong_targets = [s for s in target_pool if is_structurally_strong(s, target_pool, atr)]
            use_pool = strong_targets if strong_targets else target_pool
            sorted_pool = sorted(use_pool, key=lambda s: s["price"], reverse=(direction == "bearish"))
            tp1 = sorted_pool[0]["price"]
            tp2 = sorted_pool[-1]["price"] if len(sorted_pool) > 1 else tp1

            # === بررسی Look-Ahead Timing (بند ۱۰-۱۱ دستور) ===
            # آیا Entry واقعاً در زمان idx قابل‌دسترس بود؟ بله - bos.close
            # یعنی Close خودِ کندل idx، که در لحظه بسته‌شدنش قابل مشاهده‌ست.
            # اما تأیید Follow-through (که شرط ورودمونه) نیاز به دیدن
            # کندل‌های idx+1 تا confirm_idx داره. یعنی در عمل، در لحظه‌ای
            # که واقعاً می‌فهمیم "این Setup معتبره"، بازار از قیمت entry_price
            # (Close کندل idx) فاصله گرفته. بررسی می‌کنیم این فاصله چقدره:
            realistic_entry = df4h["close"].iloc[confirm_idx] if confirm_idx else entry_price
            entry_slippage_atr = abs(realistic_entry - entry_price) / atr if atr else 0
            look_ahead_flags.append({
                "symbol": symbol, "claimed_entry": entry_price, "realistic_entry": realistic_entry,
                "bars_to_confirm": confirm_idx - idx if confirm_idx else 0,
                "slippage_atr": round(entry_slippage_atr, 3),
            })

            setups.append({
                "symbol": symbol, "direction": direction, "entry": entry_price,
                "tp1": tp1, "tp2": tp2, "bos": bos, "idx": idx, "atr": atr, "df": df4h,
            })
        time.sleep(0.5)

    print(f"\nTotal Setups (Snapshot, Immutable): {len(setups)}\n")

    # === بخش ۱: بررسی Look-Ahead Timing ===
    print("=" * 60)
    print("LOOK-AHEAD TIMING CHECK (بند 10-11)")
    print("=" * 60)
    slippages = [f["slippage_atr"] for f in look_ahead_flags]
    bars_list = [f["bars_to_confirm"] for f in look_ahead_flags]
    if slippages:
        print(f"Entry اعلام‌شده (Close کندل BOS) در برابر قیمتی که واقعاً در لحظه تأیید Follow-through در دسترس بود:")
        print(f"  Mean Slippage (ATR): {sum(slippages)/len(slippages):.3f} | Median: {median(slippages):.3f} | Max: {max(slippages):.3f}")
        print(f"  Mean Bars to Confirm: {sum(bars_list)/len(bars_list):.2f}")
        significant_slippage = sum(1 for s in slippages if s > 0.3)
        print(f"  تعداد نمونه با Slippage > 0.3 ATR: {significant_slippage} از {len(slippages)} ({round(significant_slippage/len(slippages)*100,1)}%)")
        print("\n  ⚠️ نتیجه: Entry فعلی روی Close کندل BOS ثبت می‌شه، ولی تأیید")
        print("  Follow-through به کندل‌های بعدی نیاز داره. اگه Slippage قابل‌توجه")
        print("  باشه، یعنی SL/TP/RR که حساب کردیم بر پایه یه قیمت Entry ساخته")
        print("  شده که در لحظه تصمیم واقعی (بعد از تأیید FT) دیگه در دسترس نبوده.")

    # === بخش ۲: مقایسه مدل‌های SL ===
    print("\n" + "=" * 60)
    print("SL MODEL COMPARISON (روی همون Setup Snapshot)")
    print("=" * 60)

    model_results = {"A_current_0.3pct": [], "B_atr_0.25": [], "B_atr_0.50": [], "B_atr_0.75": []}

    for s in setups:
        variants = build_sl_variants(s["direction"], s["bos"], s["atr"])
        for model_name, sl_val in variants.items():
            if not sl_distance_ok(s["entry"], sl_val):
                continue
            risk = abs(s["entry"] - sl_val)
            reward1 = abs(s["tp1"] - s["entry"])
            if risk == 0 or reward1 == 0:
                continue
            rr1 = reward1 / risk
            if rr1 < MIN_RR:
                continue
            outcome = simulate_outcome(s["df"], s["direction"], s["entry"], sl_val, s["tp1"], s["tp2"], s["idx"])
            model_results[model_name].append({
                "symbol": s["symbol"], "result": outcome["result"], "bars_held": outcome["bars_held"],
                "mae": outcome["mae"], "mfe": outcome["mfe"], "risk": risk, "rr1": rr1,
                "stop_pct": round(risk / s["entry"] * 100, 3),
            })

    for model_name, results in model_results.items():
        total = len(results)
        if total == 0:
            print(f"\n--- {model_name} --- N=0")
            continue
        sl_first_candle = sum(1 for r in results if r["result"] == "SL_HIT" and r["bars_held"] == 1)
        sl_within_3 = sum(1 for r in results if r["result"] == "SL_HIT" and r["bars_held"] <= 3)
        tp2 = sum(1 for r in results if r["result"] == "TP2_HIT")
        tp1_be = sum(1 for r in results if r["result"] == "TP1_THEN_SL")
        tp1_open = sum(1 for r in results if r["result"] == "TP1_ONLY_OPEN")
        sl_total = sum(1 for r in results if r["result"] == "SL_HIT")
        mae_over_risk = [r["mae"]/r["risk"] for r in results if r["risk"] > 0]
        r_multiples = []
        for r in results:
            if r["result"] == "TP2_HIT":
                r_multiples.append(r["rr1"] * (r["mfe"]/r["risk"] if r["risk"] else 0) if False else r["rr1"])
            elif r["result"] in ("TP1_THEN_SL",):
                r_multiples.append(0)
            elif r["result"] == "SL_HIT":
                r_multiples.append(-1)
        expectancy = sum(r_multiples)/len(r_multiples) if r_multiples else None
        stop_pcts = [r["stop_pct"] for r in results]

        print(f"\n--- {model_name} --- N={total}")
        print(f"  SL_HIT (کندل اول): {sl_first_candle} ({round(sl_first_candle/total*100,1)}%)")
        print(f"  SL_HIT (تا کندل 3): {sl_within_3} ({round(sl_within_3/total*100,1)}%)")
        print(f"  SL_HIT (کل): {sl_total} ({round(sl_total/total*100,1)}%)")
        print(f"  TP2_HIT: {tp2} ({round(tp2/total*100,1)}%)")
        print(f"  TP1_THEN_SL: {tp1_be} ({round(tp1_be/total*100,1)}%)")
        print(f"  TP1_ONLY_OPEN: {tp1_open} ({round(tp1_open/total*100,1)}%)")
        print(f"  Median Stop Distance %: {median(stop_pcts)}")
        print(f"  Mean MAE/Risk ratio: {sum(mae_over_risk)/len(mae_over_risk):.2f}" if mae_over_risk else "  N/A")
        print(f"  Expectancy (R-multiple ساده‌شده): {round(expectancy,2) if expectancy is not None else 'N/A'}")

    print("\n" + "=" * 70)
