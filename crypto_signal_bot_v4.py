"""
Crypto Signal Bot V4 — Phase 9 Part B/C/D/E: Realistic SL/TP + Full Forward Simulation
Entry = Causal (BOS -> Follow-through confirmation -> first executable price)
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
FOLLOW_THROUGH_CANDLES = 2
MAX_HOLD_CANDLES = 30
MIN_SL_DISTANCE_PCT = 0.005
MIN_RR = 2.0
EQUAL_LEVEL_TOLERANCE_ATR = 0.15
FEE_PCT_ROUND_TRIP = 0.10  # فرض: ۰.۰۵٪ هر طرف (تیکر معمول اسپات/فیوچرز) = ۰.۱٪ رفت‌وبرگشت


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


def sl_distance_ok(entry, sl):
    if entry == 0 or sl is None:
        return False
    return abs(entry - sl) / entry >= MIN_SL_DISTANCE_PCT


def simulate_trade(df, direction, entry, sl, tp1, tp2, start_idx, max_hold=MAX_HOLD_CANDLES):
    end_idx = min(start_idx + 1 + max_hold, len(df))
    mfe, mae = 0, 0
    tp1_hit_flag = False
    bars_to_tp1, bars_to_tp2, bars_to_sl = None, None, None

    for step, i in enumerate(range(start_idx + 1, end_idx), start=1):
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

        if not tp1_hit_flag:
            # قانون محافظه‌کارانه: اگه SL و TP هر دو تو یه کندل باشن، SL اول فرض می‌شه
            if sl_hit:
                bars_to_sl = step
                return {"result": "SL_HIT", "bars_to_sl": bars_to_sl, "bars_to_tp1": None,
                        "bars_to_tp2": None, "mfe": mfe, "mae": mae, "r_multiple": -1.0}
            if tp1_now:
                tp1_hit_flag = True
                bars_to_tp1 = step
                if tp2_now:
                    bars_to_tp2 = step
                    r_mult = abs(tp2 - entry) / abs(entry - sl)
                    return {"result": "TP2_HIT", "bars_to_sl": None, "bars_to_tp1": bars_to_tp1,
                            "bars_to_tp2": bars_to_tp2, "mfe": mfe, "mae": mae, "r_multiple": r_mult}
        else:
            if sl_hit:
                return {"result": "TP1_THEN_SL", "bars_to_sl": step, "bars_to_tp1": bars_to_tp1,
                        "bars_to_tp2": None, "mfe": mfe, "mae": mae, "r_multiple": 0.0}
            if tp2_now:
                bars_to_tp2 = step
                r_mult = abs(tp2 - entry) / abs(entry - sl)
                return {"result": "TP2_HIT", "bars_to_sl": None, "bars_to_tp1": bars_to_tp1,
                        "bars_to_tp2": bars_to_tp2, "mfe": mfe, "mae": mae, "r_multiple": r_mult}

    if tp1_hit_flag:
        return {"result": "TP1_ONLY_OPEN", "bars_to_sl": None, "bars_to_tp1": bars_to_tp1,
                "bars_to_tp2": None, "mfe": mfe, "mae": mae, "r_multiple": None}
    return {"result": "OPEN_NO_OUTCOME", "bars_to_sl": None, "bars_to_tp1": None,
            "bars_to_tp2": None, "mfe": mfe, "mae": mae, "r_multiple": None}


def median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid-1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


if __name__ == "__main__":
    print("PHASE: 9 (Part B/C/D/E)")
    print("STATUS: EXECUTING - Realistic SL/TP + Full Forward Simulation")
    print("=" * 70)

    all_pre_rr = []  # همه Candidate ها قبل از فیلتر R:R (بند D)
    final_trades = []

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

            # SL: broken_level (سطح BOS)، محاسبه‌شده مستقل از Entry جدید ولی خودش تغییر نکرده
            buffer_pct = 1.003 if direction == "bearish" else 0.997
            sl = bos["broken_level"] * buffer_pct
            if not sl_distance_ok(entry, sl):
                all_pre_rr.append({"symbol": symbol, "reject_stage": "invalid_risk_sl_too_close"})
                continue

            # Target: فقط Swing هایی که قبل از confirm_idx تشکیل شدن (No Look-Ahead)
            if direction == "bullish":
                target_pool = [s for s in structure["swing_highs"] if s["price"] > entry and s["index"] < confirm_idx]
            else:
                target_pool = [s for s in structure["swing_lows"] if s["price"] < entry and s["index"] < confirm_idx]

            if not target_pool:
                all_pre_rr.append({"symbol": symbol, "reject_stage": "no_target_pool"})
                continue

            strong_targets = [s for s in target_pool if is_structurally_strong(s, target_pool, atr)]
            use_pool = strong_targets if strong_targets else target_pool
            target_type = "STRONG (equal-level)" if strong_targets else "POTENTIAL (isolated)"
            sorted_pool = sorted(use_pool, key=lambda s: s["price"], reverse=(direction == "bearish"))
            tp1 = sorted_pool[0]["price"]
            tp2 = sorted_pool[-1]["price"] if len(sorted_pool) > 1 else tp1

            risk = abs(entry - sl)
            reward1 = abs(tp1 - entry)
            reward2 = abs(tp2 - entry)
            rr1 = reward1 / risk if risk else 0
            rr2 = reward2 / risk if risk else 0

            candidate_record = {
                "symbol": symbol, "direction": direction, "entry": entry, "sl": sl,
                "tp1": tp1, "tp2": tp2, "risk_pct": round(risk/entry*100, 3),
                "risk_atr": round(risk/atr, 2), "rr1": round(rr1, 2), "rr2": round(rr2, 2),
                "target_type": target_type, "confirm_idx": confirm_idx,
            }
            all_pre_rr.append({**candidate_record, "reject_stage": "PASSED_TO_RR_GATE" if rr1 >= MIN_RR else f"rr_below_{MIN_RR}"})

            if rr1 < MIN_RR:
                continue

            outcome = simulate_trade(df4h, direction, entry, sl, tp1, tp2, confirm_idx)

            gross_pct = None
            if outcome["result"] == "TP2_HIT":
                gross_pct = reward2 / entry * 100
            elif outcome["result"] == "TP1_THEN_SL":
                gross_pct = 0.0
            elif outcome["result"] == "SL_HIT":
                gross_pct = -risk / entry * 100
            net_pct = (gross_pct - FEE_PCT_ROUND_TRIP) if gross_pct is not None else None

            final_trades.append({**candidate_record, **outcome, "gross_pct": gross_pct, "net_pct": net_pct})
        time.sleep(0.5)

    print(f"\nAll pre-R:R candidates logged: {len(all_pre_rr)}")
    reject_counts = {}
    for c in all_pre_rr:
        stage = c["reject_stage"]
        reject_counts[stage] = reject_counts.get(stage, 0) + 1
    print("Breakdown:")
    for k, v in reject_counts.items():
        print(f"  {k}: {v}")

    print(f"\n--- LEVEL 2/3: Final Trades (R:R>=2 gate passed): {len(final_trades)} ---\n")
    for t in final_trades:
        print(f"{t['symbol']} | {t['direction']} | Entry={t['entry']:.6f} | SL={t['sl']:.6f} | "
              f"TP1={t['tp1']:.6f} | TP2={t['tp2']:.6f} | RiskATR={t['risk_atr']} | RR1={t['rr1']} | "
              f"Result={t['result']} | BarsToSL={t['bars_to_sl']} | BarsToTP1={t['bars_to_tp1']} | "
              f"BarsToTP2={t['bars_to_tp2']} | Gross%={t['gross_pct']} | Net%={t['net_pct']}")

    total = len(final_trades)
    if total:
        tp2 = sum(1 for t in final_trades if t["result"] == "TP2_HIT")
        tp1_be = sum(1 for t in final_trades if t["result"] == "TP1_THEN_SL")
        sl_hit = sum(1 for t in final_trades if t["result"] == "SL_HIT")
        open_trades = total - tp2 - tp1_be - sl_hit

        r_mults = [t["r_multiple"] for t in final_trades if t["r_multiple"] is not None]
        wins = [r for r in r_mults if r > 0]
        losses = [r for r in r_mults if r < 0]
        expectancy = sum(r_mults) / len(r_mults) if r_mults else None
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None

        gross_pcts = [t["gross_pct"] for t in final_trades if t["gross_pct"] is not None]
        net_pcts = [t["net_pct"] for t in final_trades if t["net_pct"] is not None]

        print("\n" + "=" * 70)
        print("LEVEL 3 — NET REALISTIC PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"N (Final Trades) = {total}")
        print(f"TP2_HIT: {tp2} ({round(tp2/total*100,1)}%) | TP1_THEN_SL: {tp1_be} ({round(tp1_be/total*100,1)}%) | "
              f"SL_HIT: {sl_hit} ({round(sl_hit/total*100,1)}%) | Open: {open_trades}")
        print(f"Expectancy (R-multiple, on closed trades): {round(expectancy,3) if expectancy is not None else 'N/A'}")
        print(f"Profit Factor: {round(profit_factor,2) if profit_factor else 'N/A'}")
        if gross_pcts:
            print(f"Gross Return sum: {sum(gross_pcts):.2f}% | Mean per trade: {sum(gross_pcts)/len(gross_pcts):.2f}%")
        if net_pcts:
            print(f"Net Return sum (after {FEE_PCT_ROUND_TRIP}% fee/trade): {sum(net_pcts):.2f}% | Mean per trade: {sum(net_pcts)/len(net_pcts):.2f}%")

        print("\n" + "=" * 70)
        print("COMPARISON TABLE: Phase 7/8 (Immutable) vs Phase 9 (Realistic Entry)")
        print("=" * 70)
        print(f"{'Metric':<25} {'Phase 7/8':<20} {'Phase 9 Real Entry'}")
        print(f"{'Entry timing':<25} {'BOS close (flawed)':<20} {'Confirmation close (causal)'}")
        print(f"{'N (final trades)':<25} {'11':<20} {total}")
        print(f"{'TP2_HIT %':<25} {'9.1%':<20} {round(tp2/total*100,1)}%")
        print(f"{'SL_HIT %':<25} {'72.7%':<20} {round(sl_hit/total*100,1)}%")
        print(f"{'Expectancy':<25} {'-0.61 to -0.91':<20} {round(expectancy,3) if expectancy is not None else 'N/A'}")
    else:
        print("\nهیچ Trade نهایی (بعد از R:R>=2) وجود نداره.")
    print("=" * 70)
