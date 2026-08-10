"""
Backtest Engine — سنجش عملکرد واقعی استراتژی روی داده تاریخی
----------------------------------------------------------------
این اسکریپت دقیقاً همون منطق crypto_signal_bot.py رو روی چند ماه گذشته
اجرا می‌کنه (walk-forward، بدون نگاه به آینده) و می‌سنجه که اگه این
سیستم اون‌موقع فعال بود، چند درصد سیگنال‌ها برنده/بازنده می‌شدن.

⚠️ محدودیت‌های مهم (صادقانه):
- به‌جای کندل ۱۵دقیقه‌ای (که برای بک‌تست چندماهه حجم درخواست API
  غیرعملی می‌شه)، نقطه ورود از روی Volume Profile همون کندل‌های
  ۴ساعته (۱۰ کندل آخر ≈ ۴۰ ساعت) تقریب زده می‌شه. این دقت ورود رو
  کمی پایین‌تر از نسخه زنده (که از ۱۵دقیقه استفاده می‌کنه) می‌آره.
- کارمزد صرافی، اسلیپیج (لغزش قیمت)، و تأخیر اجرا در نظر گرفته نشده.
- نتیجه یک "تقریب معقول"ه، نه یک عدد قطعی برای تصمیم مالی.

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
    compute_bollinger, score_trend_candidate, score_mean_reversion_candidate,
    get_valid_targets, fmt_price, MIN_RR, safe_get, CRYPTOCOMPARE_BASE,
)

BACKTEST_COINS_N = 20      # تعداد ارزهایی که بک‌تست می‌شن (محدود به‌خاطر سقف درخواست API)
BACKTEST_DAYS = 90         # بازه زمانی بک‌تست (روز)
WINDOW = 150                # تعداد کندل تاریخچه لازم (مطابق نسخه زنده)
FORWARD_LOOKOUT = 60        # چند کندل ۴ساعته جلوتر رو برای TP/SL چک کنیم (~۱۰ روز)
COOLDOWN_BARS = 6           # بعد از هر سیگنال، چند کندل صبر کنیم قبل از سیگنال بعدی همون ارز


def get_full_history(symbol, days):
    limit = min(int(days * 6) + WINDOW + 10, 2000)
    df = get_ohlcv(symbol, timeframe="hour", aggregate=4, limit=limit)
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
    """بررسی کندل‌های جلوتر برای دیدن اینکه SL یا TP اول لمس شده"""
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


def backtest_symbol(symbol):
    df = get_full_history(symbol, BACKTEST_DAYS)
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
            candidate = score_trend_candidate(symbol, window_df, structure, vp, rvol)
            strategy = "trend"
        else:
            candidate = score_mean_reversion_candidate(symbol, window_df, structure, vp, rvol)
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
    print(f"=== شروع بک‌تست ({BACKTEST_DAYS} روز، {BACKTEST_COINS_N} ارز) ===\n")
    coins = get_top_coins(BACKTEST_COINS_N)
    all_results = []

    for c in coins:
        symbol = c["symbol"].upper()
        print(f"در حال بک‌تست {symbol}...")
        try:
            res = backtest_symbol(symbol)
            all_results.extend(res)
            print(f"  {len(res)} سیگنال پیدا شد.")
        except Exception as e:
            print(f"  خطا: {e}")
        time.sleep(1.5)

    print("\n" + "=" * 50)
    print("=== نتیجه نهایی بک‌تست ===")
    print("=" * 50)

    if not all_results:
        print("هیچ سیگنالی در کل بازه زمانی تولید نشد.")
        return

    total = len(all_results)
    wins = [r for r in all_results if r["outcome"] in ("WIN_TP1", "WIN_TP2")]
    losses = [r for r in all_results if r["outcome"] == "LOSS"]
    not_entered = [r for r in all_results if r["outcome"] == "NOT_ENTERED"]
    timeout = [r for r in all_results if r["outcome"] == "TIMEOUT"]

    decided = len(wins) + len(losses)
    win_rate = (len(wins) / decided * 100) if decided > 0 else 0

    print(f"\nکل سیگنال‌های تولیدشده: {total}")
    print(f"وارد نشده (Limit پر نشد): {len(not_entered)}")
    print(f"برنده (TP1 یا TP2): {len(wins)}")
    print(f"بازنده (SL): {len(losses)}")
    print(f"بدون نتیجه قطعی (Timeout): {len(timeout)}")
    print(f"\n>>> نرخ برد (فقط معاملات با نتیجه قطعی): {win_rate:.1f}% <<<\n")

    for strat in ("trend", "mean_reversion"):
        strat_results = [r for r in all_results if r["strategy"] == strat]
        strat_wins = [r for r in strat_results if r["outcome"] in ("WIN_TP1", "WIN_TP2")]
        strat_losses = [r for r in strat_results if r["outcome"] == "LOSS"]
        strat_decided = len(strat_wins) + len(strat_losses)
        strat_wr = (len(strat_wins) / strat_decided * 100) if strat_decided > 0 else 0
        label = "روند (Smart Money)" if strat == "trend" else "بازگشت به میانگین"
        print(f"[{label}] تعداد: {len(strat_results)} | نرخ برد: {strat_wr:.1f}% ({len(strat_wins)}برد/{len(strat_losses)}باخت)")

    avg_rr = np.mean([r["rr_planned"] for r in all_results])
    print(f"\nمیانگین R:R برنامه‌ریزی‌شده: 1:{avg_rr:.2f}")


if __name__ == "__main__":
    main()
