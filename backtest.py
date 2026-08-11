"""
Backtest Engine — سنجش عملکرد واقعی استراتژی روی داده تاریخی
----------------------------------------------------------------
این اسکریپت دقیقاً همون منطق نسخه تعادلی (v3) crypto_signal_bot.py رو
روی چند ماه گذشته اجرا می‌کنه (walk-forward، بدون نگاه به آینده) و
می‌سنجه که اگه این سیستم اون‌موقع فعال بود، چند درصد سیگنال‌ها
برنده/بازنده می‌شدن.

⚠️ محدودیت‌های مهم (صادقانه):
- منبع داده: CoinGecko (کندل مصنوعی از سری قیمت/حجم، نه OHLCV واقعی صرافی).
- نقطه ورود از روی Volume Profile همون کندل‌های ۴ساعته (۱۰ کندل آخر) تقریب
  زده می‌شه، نه کندل ۱۵دقیقه‌ای دقیق (که برای بک‌تست چندماهه غیرعملیه).
- کارمزد صرافی، اسلیپیج (لغزش قیمت)، و تأخیر اجرا در نظر گرفته نشده.
- تأیید روند روزانه (daily bias) یک‌بار به‌ازای هر ارز محاسبه و کش می‌شه
  (نه به‌ازای هر پنجره)، برای صرفه‌جویی در تعداد درخواست‌های API.
- نتیجه یک "تقریب معقول"ه، نه یک عدد قطعی برای تصمیم مالی.

پارامترهای این نسخه (تعادلی v3):
  - روند: امتیاز≥۴ از ۵، RVOL>1.25، Zone Quality<0.85، تأیید روزانه
  - Mean Reversion: امتیاز≥۴ از ۵، RSI=28/72، touches≥2

اجرا: از طریق GitHub Actions (workflow_dispatch دستی) یا لوکال.
"""

import sys
import os
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_signal_bot import (
    get_top_coins, get_ohlcv, classify_structure, compute_volume_profile,
    compute_rvol, zone_quality_ok, liquidity_sweep_detected, compute_rsi,
    compute_bollinger, get_valid_targets, fmt_price, MIN_RR,
)

BACKTEST_COINS_N = 20      # تعداد ارزهایی که بک‌تست می‌شن
BACKTEST_DAYS = 85         # بازه زمانی بک‌تست (روز) — زیر ۹۰ روز تا گرانولاریتی ساعتی CoinGecko حفظ بشه
WINDOW = 150                # تعداد کندل تاریخچه لازم (مطابق نسخه زنده)
FORWARD_LOOKOUT = 60        # چند کندل ۴ساعته جلوتر رو برای TP/SL چک کنیم (~۱۰ روز)
COOLDOWN_BARS = 6           # بعد از هر سیگنال، چند کندل صبر کنیم قبل از سیگنال بعدی همون ارز

# کش تأیید روند روزانه — به‌ازای هر coin_id فقط یک‌بار در کل بک‌تست محاسبه می‌شه
_daily_trend_cache = {}


def get_daily_trend_cached(coin_id):
    if coin_id in _daily_trend_cache:
        return _daily_trend_cache[coin_id]
    df_daily = get_ohlcv(coin_id, timeframe="hour", aggregate=24, limit=45)
    if df_daily is None or len(df_daily) < 30:
        _daily_trend_cache[coin_id] = None
        return None
    structure = classify_structure(df_daily)
    _daily_trend_cache[coin_id] = structure["trend"]
    return _daily_trend_cache[coin_id]


def score_trend_candidate_bt(symbol, coin_id, df4h, structure, vp, rvol):
    """نسخه بک‌تست از منطق امتیازدهی روند (تعادلی v3، با کش daily bias)"""
    zq = zone_quality_ok(df4h)
    liq = liquidity_sweep_detected(df4h, structure["pivot_highs"], structure["pivot_lows"])
    current_price = df4h["close"].iloc[-1]
    price_near_poc = abs(current_price - vp["poc"]) / current_price < 0.06

    daily_trend = get_daily_trend_cached(coin_id)
    if daily_trend is None or daily_trend != structure["trend"]:
        return None

    checks = {
        "Volume Profile": price_near_poc,
        "Volume/RVOL": rvol > 1.25,
        "Zone Quality": zq,
        "Structure Clarity": structure["trend"] in ("up", "down"),
        "Liquidity Context": liq,
    }
    score = sum(checks.values())
    if score < 4:
        return None

    confidence = "بالا" if score == 5 else "متوسط"

    return {
        "symbol": symbol, "coin_id": coin_id, "strategy": "trend", "score": score, "confidence": confidence,
        "checks": checks, "trend": structure["trend"], "rvol": rvol,
        "price": current_price, "df4h": df4h, "structure": structure, "vp": vp,
    }


def score_mean_reversion_candidate_bt(symbol, coin_id, df4h, structure, vp, rvol):
    if structure["trend"] in ("up", "down"):
        return None
    if len(df4h) < 45:
        return None

    lookback = df4h.iloc[-40:]
    range_high = lookback["high"].max()
    range_low = lookback["low"].min()
    if range_high <= range_low:
        return None

    mid_bb, upper_bb, lower_bb = compute_bollinger(df4h)
    last_mid = mid_bb.iloc[-1]
    last_upper = upper_bb.iloc[-1]
    last_lower = lower_bb.iloc[-1]
    if any(pd.isna(x) for x in [last_mid, last_upper, last_lower]):
        return None

    rsi = compute_rsi(df4h["close"])
    last_rsi = rsi.iloc[-1]
    if pd.isna(last_rsi):
        return None

    last_close = df4h["close"].iloc[-1]
    last_low = df4h["low"].iloc[-1]
    last_high = df4h["high"].iloc[-1]

    dist_to_lower = last_close - range_low
    dist_to_upper = range_high - last_close
    direction_bias = "LONG" if dist_to_lower < dist_to_upper else "SHORT"

    touches_high = (lookback["high"] >= range_high * 0.99).sum()
    touches_low = (lookback["low"] <= range_low * 1.01).sum()
    range_confirmed = touches_high >= 2 and touches_low >= 2

    if direction_bias == "LONG":
        rsi_extreme = last_rsi < 28
        bb_touch = last_low <= last_lower
        reversal = last_low <= last_lower and last_close > last_lower
    else:
        rsi_extreme = last_rsi > 72
        bb_touch = last_high >= last_upper
        reversal = last_high >= last_upper and last_close < last_upper

    checks = {
        "Range Confirmed": range_confirmed,
        "RSI Extreme": rsi_extreme,
        "Volume Confirmation": rvol > 1.3,
        "Bollinger Band Touch": bb_touch,
        "Reversal Rejection": reversal,
    }
    score = sum(checks.values())
    if score < 4:
        return None

    confidence = "بالا" if score == 5 else "متوسط"

    return {
        "symbol": symbol, "coin_id": coin_id, "strategy": "mean_reversion", "score": score, "confidence": confidence,
        "checks": checks, "direction_bias": direction_bias, "rvol": rvol,
        "price": last_close, "df4h": df4h, "range_high": range_high, "range_low": range_low,
        "range_mid": (range_high + range_low) / 2, "vp": vp,
    }


def get_full_history(coin_id, days):
    limit = min(int(days * 6), 500)
    df = get_ohlcv(coin_id, timeframe="hour", aggregate=4, limit=limit)
    return df


def simulate_trend_entry(window_df, structure):
    pivot_highs, pivot_lows = structure["pivot_highs"], structure["pivot_lows"]
    vp_recent = compute_volume_profile(window_df.iloc[-10:])
    if vp_recent is None:
        return None
    entry = vp_recent["poc"]

    if structure["trend"] == "up" and len(pivot_lows) >= 2 and len(pivot_highs) >= 1:
        direction = "LONG"
        sl = pivot_lows[-1][1] * 0.997
        tp1, tp2 = get_valid_targets(pivot_highs, entry, "LONG")
        if tp1 is None:
            return None
        risk, reward1 = entry - sl, tp1 - entry
    elif structure["trend"] == "down" and len(pivot_highs) >= 2 and len(pivot_lows) >= 1:
        direction = "SHORT"
        sl = pivot_highs[-1][1] * 1.003
        tp1, tp2 = get_valid_targets(pivot_lows, entry, "SHORT")
        if tp1 is None:
            return None
        risk, reward1 = sl - entry, entry - tp1
    else:
        return None

    if risk <= 0 or reward1 <= 0:
        return None
    rr = reward1 / risk
    if rr < MIN_RR:
        return None
    return {"direction": direction, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr}


def simulate_mr_entry(window_df, candidate):
    vp_recent = compute_volume_profile(window_df.iloc[-10:])
    if vp_recent is None:
        return None
    entry = vp_recent["poc"]
    range_high, range_low = candidate["range_high"], candidate["range_low"]
    range_mid = candidate["range_mid"]

    if candidate["direction_bias"] == "LONG":
        direction = "LONG"
        sl = range_low * 0.99
        tp1, tp2 = range_mid, range_high
        risk, reward1 = entry - sl, tp1 - entry
    else:
        direction = "SHORT"
        sl = range_high * 1.01
        tp1, tp2 = range_mid, range_low
        risk, reward1 = sl - entry, entry - tp1

    if risk <= 0 or reward1 <= 0:
        return None
    rr = reward1 / risk
    if rr < MIN_RR:
        return None
    return {"direction": direction, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr}


def check_outcome(df, start_idx, trade):
    direction = trade["direction"]
    entry, sl, tp1, tp2 = trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]
    end_idx = min(start_idx + FORWARD_LOOKOUT, len(df))
    entered = False
    tp1_hit = False

    for j in range(start_idx, end_idx):
        low, high = df["low"].iloc[j], df["high"].iloc[j]

        if not entered:
            if low <= entry <= high:
                entered = True
            continue

        if direction == "LONG":
            sl_hit, t1_hit, t2_hit = low <= sl, high >= tp1, high >= tp2
        else:
            sl_hit, t1_hit, t2_hit = high >= sl, low <= tp1, low <= tp2

        if not tp1_hit:
            if sl_hit:
                return "LOSS"
            if t1_hit:
                tp1_hit = True
        if tp1_hit and t2_hit:
            return "WIN_TP2"

    if not entered:
        return "NOT_ENTERED"
    if tp1_hit:
        return "WIN_TP1"
    return "TIMEOUT"


def backtest_symbol(symbol, coin_id):
    df = get_full_history(coin_id, BACKTEST_DAYS)
    if df is None or len(df) < WINDOW + 20:
        return []

    results = []
    i = WINDOW
    while i < len(df) - 1:
        window_df = df.iloc[i - WINDOW + 1:i + 1].reset_index(drop=True)
        structure = classify_structure(window_df)
        vp = compute_volume_profile(window_df)
        if vp is None:
            i += 1
            continue
        rvol = compute_rvol(window_df)

        candidate = None
        strategy = None
        if structure["trend"] in ("up", "down"):
            candidate = score_trend_candidate_bt(symbol, coin_id, window_df, structure, vp, rvol)
            strategy = "trend"
        else:
            candidate = score_mean_reversion_candidate_bt(symbol, coin_id, window_df, structure, vp, rvol)
            strategy = "mean_reversion"

        if candidate is None:
            i += 1
            continue

        if strategy == "trend":
            trade = simulate_trend_entry(window_df, structure)
        else:
            trade = simulate_mr_entry(window_df, candidate)

        if trade is None:
            i += 1
            continue

        outcome = check_outcome(df, i + 1, trade)
        results.append({
            "symbol": symbol, "strategy": strategy, "score": candidate["score"],
            "confidence": candidate["confidence"], "direction": trade["direction"],
            "rr_planned": round(trade["rr"], 2), "outcome": outcome,
        })
        i += COOLDOWN_BARS

    return results


def main():
    print(f"=== شروع بک‌تست ({BACKTEST_DAYS} روز، {BACKTEST_COINS_N} ارز) ===\n", flush=True)
    coins = get_top_coins(BACKTEST_COINS_N)
    all_results = []

    for c in coins:
        symbol = c["symbol"].upper()
        coin_id = c["id"]
        print(f"در حال بک‌تست {symbol}...", flush=True)
        try:
            res = backtest_symbol(symbol, coin_id)
            all_results.extend(res)
            print(f"  {len(res)} سیگنال پیدا شد.", flush=True)
        except Exception as e:
            print(f"  خطا: {e}", flush=True)
        time.sleep(2.0)

    print("\n" + "=" * 50, flush=True)
    print("=== نتیجه نهایی بک‌تست ===", flush=True)
    print("=" * 50, flush=True)

    if not all_results:
        print("هیچ سیگنالی در کل بازه زمانی تولید نشد.", flush=True)
        return

    total = len(all_results)
    wins = [r for r in all_results if r["outcome"] in ("WIN_TP1", "WIN_TP2")]
    losses = [r for r in all_results if r["outcome"] == "LOSS"]
    not_entered = [r for r in all_results if r["outcome"] == "NOT_ENTERED"]
    timeout = [r for r in all_results if r["outcome"] == "TIMEOUT"]

    decided = len(wins) + len(losses)
    win_rate = (len(wins) / decided * 100) if decided > 0 else 0

    print(f"\nکل سیگنال‌های تولیدشده: {total}", flush=True)
    print(f"وارد نشده (Limit پر نشد): {len(not_entered)}", flush=True)
    print(f"برنده (TP1 یا TP2): {len(wins)}", flush=True)
    print(f"بازنده (SL): {len(losses)}", flush=True)
    print(f"بدون نتیجه قطعی (Timeout): {len(timeout)}", flush=True)
    print(f"\n>>> نرخ برد (فقط معاملات با نتیجه قطعی): {win_rate:.1f}% <<<\n", flush=True)

    for strat in ("trend", "mean_reversion"):
        strat_results = [r for r in all_results if r["strategy"] == strat]
        strat_wins = [r for r in strat_results if r["outcome"] in ("WIN_TP1", "WIN_TP2")]
        strat_losses = [r for r in strat_results if r["outcome"] == "LOSS"]
        strat_decided = len(strat_wins) + len(strat_losses)
        strat_wr = (len(strat_wins) / strat_decided * 100) if strat_decided > 0 else 0
        label = "روند (Smart Money)" if strat == "trend" else "بازگشت به میانگین"
        print(f"[{label}] تعداد: {len(strat_results)} | نرخ برد: {strat_wr:.1f}% ({len(strat_wins)}برد/{len(strat_losses)}باخت)", flush=True)

    avg_rr = np.mean([r["rr_planned"] for r in all_results])
    print(f"\nمیانگین R:R برنامه‌ریزی‌شده: 1:{avg_rr:.2f}", flush=True)


if __name__ == "__main__":
    main()
