"""
Debug Diagnostic — بررسی نهایی سلامت داده و منطق روی BTC
------------------------------------------------------------
این اسکریپت فقط روی BTC اجرا میشه، اول تأیید می‌کنه داده از CoinGecko
درست میاد، بعد آمار کامل هر ۵ معیار رو نشون می‌ده.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_signal_bot import (
    get_top_coins, get_ohlcv, classify_structure, compute_volume_profile,
    compute_rvol, zone_quality_ok, liquidity_sweep_detected, get_valid_targets, MIN_RR,
)

WINDOW = 150


def main():
    print("در حال پیدا کردن coin_id برای BTC از CoinGecko...")
    coins = get_top_coins(5)
    btc = next((c for c in coins if c["symbol"].upper() == "BTC"), None)
    if btc is None:
        print("❌ BTC تو لیست پیدا نشد!")
        return
    coin_id = btc["id"]
    print(f"coin_id پیدا شد: {coin_id}\n")

    print("در حال دریافت کندل ۴ساعته (۸۵ روز)...")
    df = get_ohlcv(coin_id, timeframe="hour", aggregate=4, limit=500)
    if df is None:
        print("❌ داده دریافت نشد!")
        return
    print(f"تعداد کندل دریافتی: {len(df)}")
    print(f"نمونه آخرین کندل: {df.iloc[-1].to_dict()}\n")

    if len(df) < WINDOW + 20:
        print(f"⚠️ داده کافی نیست (نیاز به حداقل {WINDOW + 20}، فقط {len(df)} موجوده)")
        return

    trend_count = {"up": 0, "down": 0, "choppy": 0}
    check_pass = {"Volume Profile": 0, "Volume/RVOL": 0, "Zone Quality": 0, "Liquidity Context": 0}
    check_total = 0
    score_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    score_ge3 = 0
    target_found = 0
    rr_ok = 0

    i = WINDOW
    total_windows = 0
    while i < len(df) - 1:
        window_df = df.iloc[i - WINDOW + 1:i + 1].reset_index(drop=True)
        structure = classify_structure(window_df)
        trend_count[structure["trend"]] += 1
        total_windows += 1

        if structure["trend"] in ("up", "down"):
            vp = compute_volume_profile(window_df)
            if vp is None:
                i += 1
                continue
            rvol = compute_rvol(window_df)
            zq = zone_quality_ok(window_df)
            liq = liquidity_sweep_detected(window_df, structure["pivot_highs"], structure["pivot_lows"])
            current_price = window_df["close"].iloc[-1]
            price_near_poc = abs(current_price - vp["poc"]) / current_price < 0.06

            check_total += 1
            if price_near_poc:
                check_pass["Volume Profile"] += 1
            if rvol > 1.15:
                check_pass["Volume/RVOL"] += 1
            if zq:
                check_pass["Zone Quality"] += 1
            if liq:
                check_pass["Liquidity Context"] += 1

            score = 1 + sum([price_near_poc, rvol > 1.15, zq, liq])
            score_distribution[score] += 1

            if score >= 3:
                score_ge3 += 1
                pivot_highs, pivot_lows = structure["pivot_highs"], structure["pivot_lows"]
                vp_recent = compute_volume_profile(window_df.iloc[-10:])
                if vp_recent is not None:
                    entry = vp_recent["poc"]
                    if structure["trend"] == "up" and len(pivot_lows) >= 2 and len(pivot_highs) >= 1:
                        sl = pivot_lows[-1][1] * 0.997
                        tp1, tp2 = get_valid_targets(pivot_highs, entry, "LONG")
                        if tp1 is not None:
                            target_found += 1
                            risk, reward1 = entry - sl, tp1 - entry
                            if risk > 0 and reward1 > 0 and (reward1 / risk) >= MIN_RR:
                                rr_ok += 1
                    elif structure["trend"] == "down" and len(pivot_highs) >= 2 and len(pivot_lows) >= 1:
                        sl = pivot_highs[-1][1] * 1.003
                        tp1, tp2 = get_valid_targets(pivot_lows, entry, "SHORT")
                        if tp1 is not None:
                            target_found += 1
                            risk, reward1 = sl - entry, entry - tp1
                            if risk > 0 and reward1 > 0 and (reward1 / risk) >= MIN_RR:
                                rr_ok += 1
        i += 1

    print("=" * 50)
    print("نتیجه تشخیصی کامل")
    print("=" * 50)
    print(f"\nکل پنجره‌های بررسی‌شده: {total_windows}")
    print(f"توزیع روند: صعودی={trend_count['up']} | نزولی={trend_count['down']} | رنج={trend_count['choppy']}")
    print(f"\nاز {check_total} پنجره روند‌دار:")
    for k, v in check_pass.items():
        pct = (v / check_total * 100) if check_total else 0
        print(f"  {k}: {v} بار قبول ({pct:.1f}%)")
    print(f"\nتوزیع امتیاز (از ۵): {score_distribution}")
    print(f"تعداد با امتیاز >= 3: {score_ge3}")
    print(f"تعداد که هدف معتبر (TP) پیدا شد: {target_found}")
    print(f"تعداد که R:R >= {MIN_RR} هم بود (سیگنال نهایی): {rr_ok}")


if __name__ == "__main__":
    main()
