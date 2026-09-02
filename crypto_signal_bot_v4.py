"""
Crypto Signal Bot V4 — Phase 9 Part A+F: Realistic Entry Timing + Direct Edge Measurement
(Phase 7/8 نتایج Immutable باقی می‌مونن - این یه Snapshot جدید و مستقله)
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
    """
    نسخه Realistic: به‌جای اینکه فقط بگه Pass/Fail، دقیقاً کندلی که
    توش Follow-through برای اولین‌بار تأیید شده رو برمی‌گردونه - این
    همون کندلیه که سیگنال واقعاً می‌تونه صادر بشه (Signal Decision Time).
    """
    end_idx = min(breakout_idx + candles_after + 1, len(df))
    if end_idx <= breakout_idx + 1:
        return None
    breakout_close = df["close"].iloc[breakout_idx]
    for i in range(breakout_idx + 1, end_idx):
        c = df["close"].iloc[i]
        if (direction == "bullish" and c > breakout_close) or (direction == "bearish" and c < breakout_close):
            return i  # این کندل، لحظه‌ی واقعی تأیید Signal است
    return None


def check_anti_chasing_at(df, ref_idx, direction, avg_range, structure, max_extension_atr=MAX_EXTENSION_ATR):
    """Anti-Chasing رو در لحظه واقعی تصمیم (ref_idx) چک می‌کنه، نه در لحظه BOS"""
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


def measure_edge(df, direction, entry, entry_idx, atr, max_hold=MAX_HOLD_CANDLES):
    """
    اندازه‌گیری مستقیم Edge بدون هیچ SL/TP از پیش تعریف‌شده - فقط
    می‌بینیم قیمت به +1R/+2R/+3R (بر پایه ATR به‌عنوان واحد ریسک فرضی)
    زودتر می‌رسه یا به -1R.
    """
    end_idx = min(entry_idx + 1 + max_hold, len(df))
    mfe, mae = 0, 0
    reached_1r_before_neg1r = None
    reached_2r_before_neg1r = None
    reached_3r_before_neg1r = None
    ret_1, ret_3, ret_6, ret_12 = None, None, None, None

    for step, i in enumerate(range(entry_idx + 1, end_idx), start=1):
        row = df.iloc[i]
        if direction == "bullish":
            favorable = row["high"] - entry
            adverse = entry - row["low"]
        else:
            favorable = entry - row["low"]
            adverse = row["high"] - entry
        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if reached_1r_before_neg1r is None:
            if adverse >= atr:
                reached_1r_before_neg1r = False
                reached_2r_before_neg1r = False
                reached_3r_before_neg1r = False
            elif favorable >= atr:
                reached_1r_before_neg1r = True
        if reached_1r_before_neg1r and reached_2r_before_neg1r is None:
            if adverse >= atr:
                reached_2r_before_neg1r = False
                reached_3r_before_neg1r = False
            elif favorable >= 2 * atr:
                reached_2r_before_neg1r = True
        if reached_2r_before_neg1r and reached_3r_before_neg1r is None:
            if adverse >= atr:
                reached_3r_before_neg1r = False
            elif favorable >= 3 * atr:
                reached_3r_before_neg1r = True

        if step == 1:
            ret_1 = (row["close"] - entry) / entry * 100 if direction == "bullish" else (entry - row["close"]) / entry * 100
        if step == 3:
            ret_3 = (row["close"] - entry) / entry * 100 if direction == "bullish" else (entry - row["close"]) / entry * 100
        if step == 6:
            ret_6 = (row["close"] - entry) / entry * 100 if direction == "bullish" else (entry - row["close"]) / entry * 100
        if step == 12:
            ret_12 = (row["close"] - entry) / entry * 100 if direction == "bullish" else (entry - row["close"]) / entry * 100

    return {
        "mfe_atr": round(mfe / atr, 2) if atr else None, "mae_atr": round(mae / atr, 2) if atr else None,
        "win_1r": reached_1r_before_neg1r, "win_2r": reached_2r_before_neg1r, "win_3r": reached_3r_before_neg1r,
        "ret_1": round(ret_1, 2) if ret_1 is not None else None,
        "ret_3": round(ret_3, 2) if ret_3 is not None else None,
        "ret_6": round(ret_6, 2) if ret_6 is not None else None,
        "ret_12": round(ret_12, 2) if ret_12 is not None else None,
    }


if __name__ == "__main__":
    print("PHASE: 9 (Part A + F)")
    print("STATUS: EXECUTING - Realistic Entry Timing + Direct Edge Measurement")
    print("=" * 70)
    print("توجه: نتایج Phase 7/8 دست‌نخورده و Immutable باقی می‌مونن - این یه")
    print("Snapshot کاملاً جدا و مستقله.\n")

    records = []
    missed_entries = 0

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

            # === PART A: پیدا کردن لحظه واقعی تأیید Follow-through ===
            confirm_idx = detect_follow_through_realtime(df4h, bos_idx, direction)
            if confirm_idx is None:
                continue  # Follow-through هرگز تأیید نشد - سیگنالی صادر نمی‌شد

            atr = avg_range.iloc[confirm_idx]
            if pd.isna(atr) or atr == 0:
                continue

            # Anti-Chasing حالا در لحظه واقعی تصمیم (confirm_idx) چک می‌شه، نه BOS
            ac_ok, ext = check_anti_chasing_at(df4h, confirm_idx, direction, avg_range, structure)
            if not ac_ok:
                missed_entries += 1  # NO TRADE - قیمت خیلی دور شده بود
                continue

            # Entry واقعی = اولین قیمت در دسترس بعد از تأیید = Close همون کندل تأیید
            realistic_entry = df4h["close"].iloc[confirm_idx]

            edge = measure_edge(df4h, direction, realistic_entry, confirm_idx, atr)

            records.append({
                "symbol": symbol, "direction": direction,
                "bos_idx": bos_idx, "confirm_idx": confirm_idx,
                "bars_to_confirm": confirm_idx - bos_idx,
                "realistic_entry": realistic_entry, "extension_atr": round(ext, 2),
                **edge,
            })
        time.sleep(0.5)

    print(f"Total Realistic Setups (بعد از اصلاح Entry Timing): {len(records)}")
    print(f"NO TRADE / MISSED (به‌خاطر Extension در لحظه تأیید): {missed_entries}\n")

    for r in records:
        print(f"{r['symbol']} | {r['direction']} | bars_to_confirm={r['bars_to_confirm']} | "
              f"Entry={r['realistic_entry']:.6f} | MFE={r['mfe_atr']}R | MAE={r['mae_atr']}R | "
              f"Win1R={r['win_1r']} | Win2R={r['win_2r']} | Win3R={r['win_3r']} | "
              f"Ret1={r['ret_1']}% Ret3={r['ret_3']}% Ret6={r['ret_6']}% Ret12={r['ret_12']}%")

    total = len(records)
    if total:
        win1r = sum(1 for r in records if r["win_1r"] is True)
        win2r = sum(1 for r in records if r["win_2r"] is True)
        win3r = sum(1 for r in records if r["win_3r"] is True)
        avg_mfe = sum(r["mfe_atr"] for r in records if r["mfe_atr"] is not None) / total
        avg_mae = sum(r["mae_atr"] for r in records if r["mae_atr"] is not None) / total

        print("\n" + "=" * 70)
        print("SUMMARY (Raw Edge بدون هیچ SL/TP از پیش تعریف‌شده):")
        print(f"  N = {total}")
        print(f"  Win-to-1R (رسیدن به +1R قبل از -1R): {win1r} ({round(win1r/total*100,1)}%)")
        print(f"  Win-to-2R: {win2r} ({round(win2r/total*100,1)}%)")
        print(f"  Win-to-3R: {win3r} ({round(win3r/total*100,1)}%)")
        print(f"  Mean MFE: {avg_mfe:.2f}R | Mean MAE: {avg_mae:.2f}R")

        rets_1 = [r["ret_1"] for r in records if r["ret_1"] is not None]
        rets_3 = [r["ret_3"] for r in records if r["ret_3"] is not None]
        rets_6 = [r["ret_6"] for r in records if r["ret_6"] is not None]
        rets_12 = [r["ret_12"] for r in records if r["ret_12"] is not None]
        if rets_1:
            print(f"\n  Mean Return after 1 candle: {sum(rets_1)/len(rets_1):.2f}%")
        if rets_3:
            print(f"  Mean Return after 3 candles: {sum(rets_3)/len(rets_3):.2f}%")
        if rets_6:
            print(f"  Mean Return after 6 candles: {sum(rets_6)/len(rets_6):.2f}%")
        if rets_12:
            print(f"  Mean Return after 12 candles: {sum(rets_12)/len(rets_12):.2f}%")
    print("=" * 70)
