"""Unified Validation Engine — Layer B (CORRECTED): fixed-horizon hit-rates per HIT_LEVEL"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
TARGET_DAYS = 730
ATR_LOOKBACK = 20
HORIZONS = [1, 3, 6, 12, 24]
HIT_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
MAX_TRACK_BARS = 44  # فقط برای MFE/MAE (پنجره پیگیری ثابت)، طبق تأیید صریح

def fetch_page(sym, end_at=None):
    r = requests.get(BASE, params={"symbol":sym,"type":"4hour",**({"endAt":int(end_at)} if end_at else {})}, timeout=20)
    if r.status_code != 200: return None, r.status_code
    d = r.json()
    if d.get("code") != "200000" or not d.get("data"): return None, d.get("code")
    rows = [{"time":int(x[0]),"open":float(x[1]),"close":float(x[2]),"high":float(x[3]),"low":float(x[4]),"volume":float(x[5])} for x in d["data"]]
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True), 200

def fetch_full_history(sym, target_days=TARGET_DAYS):
    target_candles = target_days * 6
    dfs, end_at, fetched, pages = [], None, 0, 0
    prev_min_time, consecutive_errors, stop_reason = None, 0, "target_reached"
    while fetched < target_candles:
        df, status = fetch_page(sym, end_at)
        pages += 1
        if df is None or df.empty:
            consecutive_errors += 1
            if consecutive_errors >= 3: stop_reason = f"repeated_error(status={status})"; break
            time.sleep(1); continue
        consecutive_errors = 0
        new_min_time = int(df["time"].min())
        if prev_min_time is not None and new_min_time >= prev_min_time:
            stop_reason = "no_progress_pagination_stalled"; break
        dfs.append(df); fetched += len(df); prev_min_time = new_min_time
        end_at = new_min_time - 1
        time.sleep(0.1)
        if pages > 100: stop_reason = "page_safety_limit"; break
    else:
        stop_reason = "target_reached"
    if not dfs: return None, {"stop_reason":"no_data_at_all"}
    full = pd.concat(dfs, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if int(time.time()) < int(full["time"].iloc[-1]) + 14400: full = full.iloc[:-1].reset_index(drop=True)
    full["range"] = full["high"] - full["low"]
    full["atr20"] = full["range"].rolling(ATR_LOOKBACK).mean()
    return full, {"symbol":sym, "total_candles":len(full),
                  "coverage_days": round((full["time"].iloc[-1]-full["time"].iloc[0])/86400,1),
                  "stop_reason": stop_reason}

def dummy_detector(df, every=50):
    """DUMMY ONLY — بدون ربط به Candidate 1/2"""
    events = []
    for i in range(ATR_LOOKBACK, len(df) - MAX_TRACK_BARS, every):
        events.append({"idx": i, "dir": "bullish"})
    return events

def measure_signal_level(df, events):
    """Layer B: مستقل از SL/TP/Fee. hit-rate حالا per (HIT_LEVEL, HORIZON)
    از پیش مشخص‌شده - نه انتخاب افق بعد از دیدن نتیجه."""
    results = []
    for e in events:
        idx, direction = e["idx"], e["dir"]
        atr = df["atr20"].iloc[idx]
        if pd.isna(atr) or atr == 0: continue
        entry = df["close"].iloc[idx]
        end = min(idx + 1 + MAX_TRACK_BARS, len(df))

        fwd_ret = {}
        mfe = mae = 0.0
        time_to_mfe = time_to_mae = None
        # برای هر HIT_LEVEL، فاصله (bars) که اولین‌بار بهش رسیده رو ذخیره می‌کنیم
        first_reach_step = {lv: None for lv in HIT_LEVELS}

        for step, i in enumerate(range(idx+1, end), start=1):
            row = df.iloc[i]
            if direction == "bullish":
                fav, adv = row["high"] - entry, entry - row["low"]
            else:
                fav, adv = entry - row["low"], row["high"] - entry
            if fav > mfe: mfe = fav; time_to_mfe = step
            if adv > mae: mae = adv; time_to_mae = step
            fav_atr = fav / atr
            for lv in HIT_LEVELS:
                if first_reach_step[lv] is None and fav_atr >= lv:
                    first_reach_step[lv] = step
            if step in HORIZONS:
                c = row["close"]
                fwd_ret[step] = round(((c-entry)/entry*100) if direction=="bullish" else ((entry-c)/entry*100), 3)

        # hit_at_horizon[lv][h] = True اگه first_reach_step[lv] <= h (یعنی تا اون افق ثابت، بهش رسیده)
        hit_at_horizon = {lv: {h: (first_reach_step[lv] is not None and first_reach_step[lv] <= h) for h in HORIZONS} for lv in HIT_LEVELS}

        results.append({
            "idx": idx, "entry": entry, "atr": atr,
            "fwd_ret": fwd_ret, "mfe_atr": round(mfe/atr,3), "mae_atr": round(mae/atr,3),
            "time_to_mfe": time_to_mfe, "time_to_mae": time_to_mae,
            "hit_at_horizon": hit_at_horizon,
        })
    return results

if __name__ == "__main__":
    print("UNIFIED VALIDATION ENGINE — Layer B (Fixed-Horizon Hit-Rates) — Dummy Detector Self-Test")
    print("="*70)
    df, cov = fetch_full_history("BTC-USDT")
    print(f"Data: {cov}")

    events = dummy_detector(df)
    print(f"\nDummy events: {len(events)}")

    res = measure_signal_level(df, events)
    n = len(res)
    print(f"Signal-level measurements: {n}")

    if n > 0:
        mfe_l = [r["mfe_atr"] for r in res]; mae_l = [r["mae_atr"] for r in res]
        print(f"\nMean MFE_ATR={sum(mfe_l)/n:.2f} | Mean MAE_ATR={sum(mae_l)/n:.2f}")
        for h in HORIZONS:
            vals = [r["fwd_ret"].get(h) for r in res if r["fwd_ret"].get(h) is not None]
            if vals: print(f"  Mean forward_return[{h}bar]: {round(sum(vals)/len(vals),3)}%")

        print("\nHit-Rate Matrix [HIT_LEVEL][HORIZON]:")
        header = "  LEVEL\\HORIZON | " + " | ".join(f"{h}bar" for h in HORIZONS)
        print(header)
        for lv in HIT_LEVELS:
            row_str = f"  +{lv}ATR       | "
            for h in HORIZONS:
                hits = sum(1 for r in res if r["hit_at_horizon"][lv][h])
                row_str += f"{round(hits/n*100,1)}%".ljust(6) + " | "
            print(row_str)
