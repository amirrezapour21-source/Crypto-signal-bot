"""Unified Validation Engine — Data Layer FIX: robust pagination stop + 4H continuity audit"""
import requests, pandas as pd, time

BASE = "https://api.kucoin.com/api/v1/market/candles"
TARGET_DAYS = 730
FEE_PCT, SLIPPAGE_PCT = 0.10, 0.05
EXPECTED_GAP_SEC = 4 * 3600

def fetch_page(sym, end_at=None):
    r = requests.get(BASE, params={"symbol":sym,"type":"4hour",**({"endAt":int(end_at)} if end_at else {})}, timeout=20)
    if r.status_code != 200: return None, r.status_code
    d = r.json()
    if d.get("code") != "200000" or not d.get("data"): return None, d.get("code")
    rows = [{"time":int(x[0]),"open":float(x[1]),"close":float(x[2]),"high":float(x[3]),"low":float(x[4]),"volume":float(x[5])} for x in d["data"]]
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True), 200

def fetch_full_history(sym, target_days=TARGET_DAYS):
    """FIX: توقف فقط بر اساس: هدف تأمین شد / no_more_data واقعی /
    خطای تکرارشونده / صفحه جدید هیچ Timestamp قدیمی‌تری اضافه نکرد."""
    target_candles = target_days * 6
    dfs, end_at, fetched, pages = [], None, 0, 0
    prev_min_time, consecutive_errors, stop_reason = None, 0, "target_reached"

    while fetched < target_candles:
        df, status = fetch_page(sym, end_at)
        pages += 1
        if df is None or df.empty:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                stop_reason = f"repeated_error(status={status})"
                break
            time.sleep(1)
            continue
        consecutive_errors = 0
        new_min_time = int(df["time"].min())
        if prev_min_time is not None and new_min_time >= prev_min_time:
            stop_reason = "no_progress_pagination_stalled"
            break
        dfs.append(df)
        fetched += len(df)
        prev_min_time = new_min_time
        end_at = new_min_time - 1
        time.sleep(0.1)
        if pages > 100:
            stop_reason = "page_safety_limit"
            break
    else:
        stop_reason = "target_reached"

    if not dfs:
        return None, {"pages":pages, "stop_reason":"no_data_at_all"}
    full = pd.concat(dfs, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if int(time.time()) < int(full["time"].iloc[-1]) + 14400:
        full = full.iloc[:-1].reset_index(drop=True)

    coverage = {
        "symbol": sym,
        "first_candle": pd.to_datetime(full["time"].iloc[0], unit="s"),
        "last_candle": pd.to_datetime(full["time"].iloc[-1], unit="s"),
        "total_candles": len(full),
        "coverage_days": round((full["time"].iloc[-1]-full["time"].iloc[0])/86400, 1),
        "pages_fetched": pages,
        "stop_reason": stop_reason,
        "sufficient_history": (full["time"].iloc[-1]-full["time"].iloc[0])/86400 >= 180,
    }
    return full, coverage

def audit_continuity(df, expected_gap=EXPECTED_GAP_SEC):
    """FIX: علاوه بر Monotonic/No-Dup، بررسی کامل پیوستگی ۴ساعته و گزارش Gap"""
    times = df["time"].values
    monotonic = all(times[i] < times[i+1] for i in range(len(times)-1))
    no_dupes = len(times) == len(set(times))
    gaps = []
    for i in range(len(times)-1):
        diff = times[i+1] - times[i]
        if diff != expected_gap:
            gaps.append({"after_time": pd.to_datetime(times[i], unit="s"), "gap_seconds": int(diff), "missing_candles_est": int(diff/expected_gap)-1})
    return {"monotonic": monotonic, "no_dupes": no_dupes, "gap_count": len(gaps), "gaps": gaps[:10], "total_missing_candles_est": sum(g["missing_candles_est"] for g in gaps)}

if __name__ == "__main__":
    print("UNIFIED VALIDATION ENGINE — Data Layer (FIXED) Self-Test")
    print("="*70)
    test_symbols = ["BTC-USDT", "ETH-USDT", "SUI-USDT"]
    coverages = []
    for sym in test_symbols:
        df, cov = fetch_full_history(sym)
        if df is not None:
            cont = audit_continuity(df)
            cov.update(cont)
        else:
            cov.update({"monotonic":False,"no_dupes":False,"gap_count":None})
        coverages.append(cov)
        print(f"\n{sym}:")
        for k, v in cov.items():
            if k == "gaps" and v:
                print(f"  gaps (first 10): {v}")
            elif k != "gaps":
                print(f"  {k}: {v}")

    print("\n" + "="*70)
    print("SELF-VALIDATION SUMMARY")
    print("="*70)
    print(f"Fee (trade-level only): {FEE_PCT}% | Slippage (trade-level only): {SLIPPAGE_PCT}%")
    for c in coverages:
        flag = "OK" if c.get("sufficient_history") else "INSUFFICIENT_HISTORY"
        gap_flag = "NO_GAPS" if c.get("gap_count")==0 else f"GAPS_FOUND({c.get('gap_count')}, missing~{c.get('total_missing_candles_est')})"
        print(f"  {c['symbol']}: {c['coverage_days']}d | {c['total_candles']} candles | {flag} | {gap_flag} | stop={c['stop_reason']}")
