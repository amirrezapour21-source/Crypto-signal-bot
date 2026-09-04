"""
Setup Family B — Cycle 3: Full Experiment + Decision Gate
Liquidity Sweep -> Reclaim -> Reversal
(Detection logic FROZEN from Cycle 2 - no changes)
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
EQUAL_LEVEL_TOLERANCE_ATR = 0.15
RECLAIM_MAX_CANDLES = 3
MAX_HOLD_CANDLES = 30


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


def compute_avg_range(df, lookback=20):
    return (df["high"] - df["low"]).rolling(lookback).mean()


def find_equal_level_clusters(swings, atr_series, ref_idx, tolerance_atr=EQUAL_LEVEL_TOLERANCE_ATR):
    valid_swings = [s for s in swings if s["index"] < ref_idx]
    if len(valid_swings) < 2:
        return []
    atr = atr_series.iloc[ref_idx]
    if pd.isna(atr) or atr == 0:
        return []
    sorted_swings = sorted(valid_swings, key=lambda s: s["price"])
    groups = []
    current_group = [sorted_swings[0]]
    for s in sorted_swings[1:]:
        if abs(s["price"] - current_group[-1]["price"]) / atr <= tolerance_atr:
            current_group.append(s)
        else:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [s]
    if len(current_group) >= 2:
        groups.append(current_group)
    return groups


def detect_sweep_and_reclaim_v2(df, swing_highs, swing_lows, avg_range, lookback=LOOKBACK_RECENT):
    events = []
    recent_start = max(0, len(df) - lookback)
    last_swept_level_bull = None
    last_swept_level_bear = None

    for i in range(recent_start, len(df)):
        row = df.iloc[i]

        eh_groups = find_equal_level_clusters(swing_highs, avg_range, i)
        if eh_groups:
            nearest_group = eh_groups[-1]
            level_price = max(s["price"] for s in nearest_group)
            if row["high"] > level_price and level_price != last_swept_level_bear:
                sweep_idx = i
                for j in range(sweep_idx, min(sweep_idx + RECLAIM_MAX_CANDLES + 1, len(df))):
                    if df["close"].iloc[j] < level_price:
                        events.append({
                            "type": "bearish_sweep_reclaim", "sweep_idx": sweep_idx,
                            "reclaim_idx": j, "level_price": level_price,
                            "level_group_size": len(nearest_group),
                            "bars_between": j - sweep_idx,
                        })
                        last_swept_level_bear = level_price
                        break

        el_groups = find_equal_level_clusters(swing_lows, avg_range, i)
        if el_groups:
            nearest_group = el_groups[0]
            level_price = min(s["price"] for s in nearest_group)
            if row["low"] < level_price and level_price != last_swept_level_bull:
                sweep_idx = i
                for j in range(sweep_idx, min(sweep_idx + RECLAIM_MAX_CANDLES + 1, len(df))):
                    if df["close"].iloc[j] > level_price:
                        events.append({
                            "type": "bullish_sweep_reclaim", "sweep_idx": sweep_idx,
                            "reclaim_idx": j, "level_price": level_price,
                            "level_group_size": len(nearest_group),
                            "bars_between": j - sweep_idx,
                        })
                        last_swept_level_bull = level_price
                        break

    return events


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
    end_idx = min(start_idx + 1 + max_hold, len(df))
    mfe, mae = 0, 0
    pos_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    neg_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    reached_pos = {lv: None for lv in pos_levels}
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

    return {"mfe_atr": round(mfe/atr, 3), "mae_atr": round(mae/atr, 3),
            "reached_pos": reached_pos, "reached_neg": reached_neg}


def simulate_fixed_sl_tp(df, direction, entry, sl_dist_atr, tp_r_mult, atr, start_idx, max_hold=MAX_HOLD_CANDLES):
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
    return None


if __name__ == "__main__":
    print("SETUP FAMILY B — CYCLE 3: Full Experiment + Decision Gate")
    print("=" * 70)

    entries = []

    for symbol in TEST_SYMBOLS:
        df4h = get_ohlcv_v4(symbol, "4h", total_candles=250)
        if df4h is None or len(df4h) < 80:
            continue
        df4h = drop_unclosed_candle(df4h, "4h")
        swing_highs, swing_lows = find_swings(df4h, 3, 3)
        avg_range = compute_avg_range(df4h)

        events = detect_sweep_and_reclaim_v2(df4h, swing_highs, swing_lows, avg_range)

        for e in events:
            direction = "bullish" if e["type"] == "bullish_sweep_reclaim" else "bearish"
            entry_idx = e["reclaim_idx"]  # Entry = Close همون کندلی که Reclaim توش تأیید شد (Causal)
            atr = avg_range.iloc[entry_idx]
            if pd.isna(atr) or atr == 0:
                continue
            entry_price = df4h["close"].iloc[entry_idx]
            excursion = track_excursion(df4h, direction, entry_price, atr, entry_idx)

            entries.append({
                "symbol": symbol, "direction": direction, "entry": entry_price, "atr": atr,
                "entry_idx": entry_idx, "df": df4h, "excursion": excursion,
                "reclaim_type": "same_candle" if e["bars_between"] == 0 else "multi_bar",
            })
        time.sleep(0.4)

    N = len(entries)
    print(f"\n[SAMPLE] N = {N}")
    same_candle_n = sum(1 for e in entries if e["reclaim_type"] == "same_candle")
    multi_bar_n = sum(1 for e in entries if e["reclaim_type"] == "multi_bar")
    print(f"  Same-Candle Reclaim: {same_candle_n} | Multi-Bar Reclaim: {multi_bar_n}\n")

    if N == 0:
        print("هیچ نمونه‌ای یافت نشد.")
    else:
        mfe_vals = [e["excursion"]["mfe_atr"] for e in entries]
        mae_vals = [e["excursion"]["mae_atr"] for e in entries]

        print("=" * 60)
        print("[RAW EDGE] MFE / MAE Distribution (ATR)")
        print("=" * 60)
        print(f"Mean MFE: {sum(mfe_vals)/N:.2f} | Mean MAE: {sum(mae_vals)/N:.2f}")
        print(f"Median MFE: {percentile(mfe_vals,0.5):.2f} | Median MAE: {percentile(mae_vals,0.5):.2f}")

        print(f"\nP(MFE>=1 ATR): {round(sum(1 for v in mfe_vals if v>=1)/N*100,1)}%")
        print(f"P(MFE>=2 ATR): {round(sum(1 for v in mfe_vals if v>=2)/N*100,1)}%")
        print(f"P(MAE<=1.0 ATR): {round(sum(1 for v in mae_vals if v<=1.0)/N*100,1)}%")

        print("\n" + "=" * 60)
        print("[RAW EDGE] +ATR before -ATR Probabilities")
        print("=" * 60)
        combos = [(1.0, 0.75), (1.0, 1.0), (2.0, 1.0), (2.0, 1.25), (3.0, 1.5)]
        for pos_lv, neg_lv in combos:
            count_pos_first, count_valid = 0, 0
            for e in entries:
                rp = e["excursion"]["reached_pos"].get(pos_lv)
                rn = e["excursion"]["reached_neg"].get(neg_lv)
                if rp is None and rn is None:
                    continue
                count_valid += 1
                if rp is not None and (rn is None or rp <= rn):
                    count_pos_first += 1
            pct = round(count_pos_first/count_valid*100, 1) if count_valid else None
            print(f"  P(+{pos_lv}ATR before -{neg_lv}ATR): {pct}% (n={count_valid})")

        print("\n" + "=" * 60)
        print("[TRADE CONSTRUCTION] Fixed SL Models")
        print("=" * 60)
        for sl_dist in [0.75, 1.0, 1.25, 1.5]:
            win_counts = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0, 3.0: 0}
            for e in entries:
                rn_sl = e["excursion"]["reached_neg"].get(sl_dist) if sl_dist in [0.5,0.75,1.0,1.25,1.5] else None
                for win_lv in win_counts:
                    rp = e["excursion"]["reached_pos"].get(win_lv)
                    if rp is not None and (rn_sl is None or rp <= rn_sl):
                        win_counts[win_lv] += 1
            print(f"\n  SL={sl_dist} ATR:")
            for win_lv, cnt in win_counts.items():
                print(f"    Win-to-{win_lv}R: {round(cnt/N*100,1)}% ({cnt}/{N})")

        print("\n" + "=" * 60)
        print("[TRADE CONSTRUCTION] Fixed R:R Outcomes (SL=1.0 ATR)")
        print("=" * 60)
        for tp_mult in [1.0, 1.5, 2.0, 3.0]:
            outcomes = []
            for e in entries:
                r = simulate_fixed_sl_tp(e["df"], e["direction"], e["entry"], 1.0, tp_mult, e["atr"], e["entry_idx"])
                if r is not None:
                    outcomes.append(r)
            if outcomes:
                wins = [o for o in outcomes if o > 0]
                losses = [o for o in outcomes if o < 0]
                expectancy = sum(outcomes)/len(outcomes)
                pf = (sum(wins)/abs(sum(losses))) if losses and sum(losses) != 0 else None
                print(f"  TP={tp_mult}R | N={len(outcomes)} | WinRate={round(len(wins)/len(outcomes)*100,1)}% | "
                      f"Expectancy={round(expectancy,3)} | PF={round(pf,2) if pf else 'N/A'}")

        print("\n" + "=" * 60)
        print("[RECLAIM TYPE BREAKDOWN]")
        print("=" * 60)
        for rt in ["same_candle", "multi_bar"]:
            subset = [e for e in entries if e["reclaim_type"] == rt]
            if subset:
                mean_mfe = sum(e["excursion"]["mfe_atr"] for e in subset)/len(subset)
                mean_mae = sum(e["excursion"]["mae_atr"] for e in subset)/len(subset)
                print(f"  {rt}: N={len(subset)} | Mean MFE={mean_mfe:.2f} | Mean MAE={mean_mae:.2f}")

    print("\n" + "=" * 70)
