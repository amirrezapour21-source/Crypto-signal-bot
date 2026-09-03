"""
Crypto Signal Bot V4 — Phase 10: BOS Core vs SL Model Decision Experiment
(Frozen: BOS/Daily/FollowThrough/Anti-Chasing/Entry Timing — Only SL model varies)
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
    "GMX-USDT", "STX-USDT", "KAVA-USDT", "ZIL-USDT", "ONE-USDT",
    "1INCH-USDT", "YFI-USDT", "BAL-USDT", "ENJ-USDT", "BAT-USDT",
    "ZRX-USDT", "OMG-USDT", "IOTA-USDT", "QTUM-USDT", "WAVES-USDT",
    "ANKR-USDT", "CELR-USDT", "COTI-USDT", "SKL-USDT", "STORJ-USDT",
    "OCEAN-USDT", "RSR-USDT", "CKB-USDT", "IOTX-USDT", "KSM-USDT",
]
LOOKBACK_RECENT = 120
MAX_EXTENSION_ATR = 2.5
FOLLOW_THROUGH_CANDLES = 2
MAX_HOLD_CANDLES = 30
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
        time.sleep(0.15)
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


def detect_follow_through_realtime(df, breakout_idx, direction, candles_after=FOLLOW_THROUGH_CANDLES):
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return None
    breakout_close = df["close"].iloc[breakout_idx]
    for i in range(breakout_idx + 1, end_idx):
        c = df["close"].iloc[i]
        if (direction == "bullish" and c > breakout_close) or (direction == "bearish" and c < breakout_close):
            return i
    return None


def check_anti_chasing_at(df, ref_idx, direction, avg_range, structure, max_extension_atr=MAX_EXTENSION_ATR):
    if pd.isna(avg_range.iloc[ref_idx]) or avg_range.iloc[ref_idx] == 0:
        return False, None
    current_price = df["close"].iloc[ref_idx]
    if direction == "bullish":
        relevant = [s for s in structure["swing_lows"] if s["index"] < ref_idx]
        if not relevant:
            return False, None
        origin_price = relevant[-1]["price"]
        extension = current_price - origin_price
    else:
        relevant = [s for s in structure["swing_highs"] if s["index"] < ref_idx]
        if not relevant:
            return False, None
        origin_price = relevant[-1]["price"]
        extension = origin_price - current_price
    extension_atr = extension / avg_range.iloc[ref_idx]
    return extension_atr <= max_extension_atr, extension_atr


def is_structurally_strong(swing, pool, atr):
    if atr is None or atr == 0:
        return False
    neighbor_distances = [abs(other["price"] - swing["price"]) / atr
                           for other in pool if other["index"] != swing["index"]]
    min_dist = min(neighbor_distances) if neighbor_distances else None
    return min_dist is not None and min_dist <= EQUAL_LEVEL_TOLERANCE_ATR


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def track_excursion(df, direction, entry, atr, start_idx, max_hold=MAX_HOLD_CANDLES):
    """
    برای هر کندل بعد از Entry، MFE/MAE (بر حسب ATR) رو محاسبه می‌کنه
    و مشخص می‌کنه کدوم سطوح +ATR قبل از کدوم سطوح -ATR رسیدن -
    بدون هیچ SL/TP از پیش تعریف‌شده (Pure Directional Excursion).
    """
    end_idx = min(start_idx + 1 + max_hold, len(df))
    mfe, mae = 0, 0
    pos_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    neg_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    reached_pos = {lv: None for lv in pos_levels}  # bars_to_reach
    reached_neg = {lv: None for lv in neg_levels}

    for step, i in enumerate(range(start_idx + 1, end_idx), start=1):
        row = df.iloc[i]
        if direction == "bullish":
            favorable = row["high"] - entry
            adverse = entry - row["low"]
        else:
            favorable = entry - row["low"]
            adverse = row["high"] - entry
        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        fav_atr = favorable / atr
        adv_atr = adverse / atr
        for lv in pos_levels:
            if reached_pos[lv] is None and fav_atr >= lv:
                reached_pos[lv] = step
        for lv in neg_levels:
            if reached_neg[lv] is None and adv_atr >= lv:
                reached_neg[lv] = step

    return {
        "mfe_atr": round(mfe / atr, 3), "mae_atr": round(mae / atr, 3),
        "reached_pos": reached_pos, "reached_neg": reached_neg,
    }


def simulate_fixed_sl_tp(df, direction, entry, sl_dist_atr, tp_r_mult, atr, start_idx, max_hold=MAX_HOLD_CANDLES):
    """SL و TP ثابت بر حسب چند برابر ATR - برای تست Fixed-R Model ها"""
    if direction == "bullish":
        sl = entry - sl_dist_atr * atr
        tp = entry + (sl_dist_atr * atr * tp_r_mult)
    else:
        sl = entry + sl_dist_atr * atr
        tp = entry - (sl_dist_atr * atr * tp_r_mult)

    end_idx = min(start_idx + 1 + max_hold, len(df))
    for i in range(start_idx + 1, end_idx):
        row = df.iloc[i]
        if direction == "bullish":
            sl_hit = row["low"] <= sl
            tp_hit = row["high"] >= tp
        else:
            sl_hit = row["high"] >= sl
            tp_hit = row["low"] <= tp
        if sl_hit:
            return -1.0
        if tp_hit:
            return tp_r_mult
    return None  # باز مونده، بدون نتیجه قطعی


if __name__ == "__main__":
    print("PHASE: 10")
    print("STATUS: EXECUTING - BOS Core vs SL Model Decision Experiment")
    print("=" * 70)

    causal_entries = []

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
            bos_idx = bos["index"]
            direction = "bullish" if bos["type"] == "bullish_bos" else "bearish"
            if not global_regime_filter(daily_regime, direction):
                continue
            confirm_idx = detect_follow_through_realtime(df4h, bos_idx, direction)
            if confirm_idx is None:
                continue
            atr = avg_range.iloc[confirm_idx]
            if pd.isna(atr) or atr == 0:
                continue
            ac_ok, ext = check_anti_chasing_at(df4h, confirm_idx, direction, avg_range, structure)
            if not ac_ok:
                continue
            entry = df4h["close"].iloc[confirm_idx]

            excursion = track_excursion(df4h, direction, entry, atr, confirm_idx)

            causal_entries.append({
                "symbol": symbol, "direction": direction, "entry": entry, "atr": atr,
                "confirm_idx": confirm_idx, "daily_regime": daily_regime,
                "h4_regime": structure["regime"], "df": df4h, "excursion": excursion,
            })
        time.sleep(0.4)

    N = len(causal_entries)
    print(f"\n[A] TOTAL CAUSAL ENTRIES (Frozen Pipeline): N={N}\n")

    if N == 0:
        print("هیچ نمونه‌ای یافت نشد.")
    else:
        mfe_vals = [e["excursion"]["mfe_atr"] for e in causal_entries]
        mae_vals = [e["excursion"]["mae_atr"] for e in causal_entries]

        print("=" * 60)
        print("[B] MAE DISTRIBUTION (ATR)")
        print("=" * 60)
        for p in [0.5, 0.75, 0.9, 0.95]:
            print(f"  P{int(p*100)}: {percentile(mae_vals, p):.2f}")
        print(f"  Mean: {sum(mae_vals)/N:.2f}")

        print("\n" + "=" * 60)
        print("[C] MFE DISTRIBUTION (ATR)")
        print("=" * 60)
        for p in [0.5, 0.75, 0.9, 0.95]:
            print(f"  P{int(p*100)}: {percentile(mfe_vals, p):.2f}")
        print(f"  Mean: {sum(mfe_vals)/N:.2f}")

        print(f"\n  P(MFE>=1 ATR): {round(sum(1 for v in mfe_vals if v>=1)/N*100,1)}%")
        print(f"  P(MFE>=1.5 ATR): {round(sum(1 for v in mfe_vals if v>=1.5)/N*100,1)}%")
        print(f"  P(MFE>=2 ATR): {round(sum(1 for v in mfe_vals if v>=2)/N*100,1)}%")
        print(f"  P(MFE>=3 ATR): {round(sum(1 for v in mfe_vals if v>=3)/N*100,1)}%")

        print(f"\n  P(MAE<=0.5 ATR): {round(sum(1 for v in mae_vals if v<=0.5)/N*100,1)}%")
        print(f"  P(MAE<=0.75 ATR): {round(sum(1 for v in mae_vals if v<=0.75)/N*100,1)}%")
        print(f"  P(MAE<=1.0 ATR): {round(sum(1 for v in mae_vals if v<=1.0)/N*100,1)}%")
        print(f"  P(MAE<=1.25 ATR): {round(sum(1 for v in mae_vals if v<=1.25)/N*100,1)}%")
        print(f"  P(MAE<=1.5 ATR): {round(sum(1 for v in mae_vals if v<=1.5)/N*100,1)}%")

        print("\n" + "=" * 60)
        print("[D] +ATR BEFORE -ATR PROBABILITIES")
        print("=" * 60)
        combos = [(1.0, 0.75), (1.0, 1.0), (2.0, 1.0), (2.0, 1.25), (3.0, 1.5)]
        for pos_lv, neg_lv in combos:
            count_pos_first = 0
            count_valid = 0
            for e in causal_entries:
                rp = e["excursion"]["reached_pos"].get(pos_lv)
                rn = e["excursion"]["reached_neg"].get(neg_lv)
                if rp is None and rn is None:
                    continue
                count_valid += 1
                if rp is not None and (rn is None or rp <= rn):
                    count_pos_first += 1
            pct = round(count_pos_first / count_valid * 100, 1) if count_valid else None
            print(f"  P(+{pos_lv}ATR before -{neg_lv}ATR): {pct}% (n={count_valid})")

        print("\n" + "=" * 60)
        print("[E] FIXED SL MODEL COMPARISON (SL dist in ATR: 0.75/1.0/1.25/1.5)")
        print("=" * 60)
        for sl_dist in [0.75, 1.0, 1.25, 1.5]:
            win_counts = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0, 3.0: 0}
            total_valid = 0
            for e in causal_entries:
                rn_sl = None
                for lv in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    if abs(lv - sl_dist) < 0.01:
                        rn_sl = e["excursion"]["reached_neg"].get(lv)
                if rn_sl is None:
                    rn_sl = e["excursion"]["reached_neg"].get(1.5)  # نزدیک‌ترین موجود بالاتر
                total_valid += 1
                for win_lv in win_counts:
                    rp = e["excursion"]["reached_pos"].get(win_lv)
                    if rp is not None and (rn_sl is None or rp <= rn_sl):
                        win_counts[win_lv] += 1
            print(f"\n  SL={sl_dist} ATR:")
            for win_lv, cnt in win_counts.items():
                print(f"    Win-to-{win_lv}R: {round(cnt/total_valid*100,1)}% ({cnt}/{total_valid})")

        print("\n" + "=" * 60)
        print("[F] FIXED R:R TP OUTCOMES (SL=1.0 ATR baseline, TP=1R/1.5R/2R/3R)")
        print("=" * 60)
        for tp_mult in [1.0, 1.5, 2.0, 3.0]:
            outcomes = []
            for e in causal_entries:
                r = simulate_fixed_sl_tp(e["df"], e["direction"], e["entry"], 1.0, tp_mult, e["atr"], e["confirm_idx"])
                if r is not None:
                    outcomes.append(r)
            if outcomes:
                wins = [o for o in outcomes if o > 0]
                losses = [o for o in outcomes if o < 0]
                expectancy = sum(outcomes) / len(outcomes)
                pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
                print(f"  TP={tp_mult}R | N={len(outcomes)} | WinRate={round(len(wins)/len(outcomes)*100,1)}% | "
                      f"Expectancy={round(expectancy,3)} | ProfitFactor={round(pf,2) if pf else 'N/A'}")

        print("\n" + "=" * 60)
        print("[H] REGIME BREAKDOWN")
        print("=" * 60)
        for regime in ["BULLISH", "BEARISH", "CHOPPY"]:
            subset = [e for e in causal_entries if e["daily_regime"] == regime]
            if subset:
                mean_mfe = sum(e["excursion"]["mfe_atr"] for e in subset) / len(subset)
                mean_mae = sum(e["excursion"]["mae_atr"] for e in subset) / len(subset)
                print(f"  Daily={regime}: N={len(subset)} | Mean MFE={mean_mfe:.2f} | Mean MAE={mean_mae:.2f}")

    print("\n" + "=" * 70)
