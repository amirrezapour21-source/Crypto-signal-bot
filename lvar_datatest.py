"""
DATA ACCESS TEST (read-only research, no trading logic).
Goal: find out which public futures data can be pulled from GitHub Actions, for how far back, for BTC and for a mid-cap alt (AVAX):
  funding-rate history, open-interest history, long/short ratio, liquidation sizes, real perpetual klines.
Every request is a plain public GET (one POST for Hyperliquid). Nothing is written except lvar_datatest_result.txt.
"""
import json
import time
from datetime import datetime, timezone

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; research-datatest)"}
DAY = 86400
T0 = 1667260800  # 2022-11-01 00:00 UTC
DEEP_UNTIL = T0 + 90 * DAY  # a test is "deep" if its oldest record is within 90 days of T0
TIMEOUT = 25


def norm(v):
    v = float(v)
    return v / 1000.0 if v > 1e11 else v


def dt(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def walk(j, path):
    for p in path:
        j = j[p]
    return j


def series(path, key):
    """extractor: list of records at `path`; timestamp is item[key] (dict) or item[key] (list index)."""
    def f(j):
        items = walk(j, path)
        return [norm(it[key]) for it in items], ""
    return f


def stats_series(j):
    items = j if isinstance(j, list) else []
    ts = [norm(it["time"]) for it in items]
    keys = set(items[0].keys()) if items else set()
    extra = "fields: " + ",".join(sorted(k for k in keys if "liq" in k or "lsr" in k or k == "open_interest"))
    return ts, extra


def snapshot(j):
    d = j["data"]
    if "openInterest" in d:
        return [time.time()], f"snapshot only, openInterest={d['openInterest']}"
    return [], ""


def tests():
    t0ms, t90ms = T0 * 1000, (T0 + 90 * DAY) * 1000
    out = []
    for label, b in (("BTC", {"bn": "BTCUSDT", "okx": "BTC-USDT-SWAP", "ccy": "BTC", "kc": "XBTUSDTM", "gate": "BTC_USDT", "mexc": "BTC_USDT", "hl": "BTC"}),
                     ("AVAX", {"bn": "AVAXUSDT", "okx": "AVAX-USDT-SWAP", "ccy": "AVAX", "kc": "AVAXUSDTM", "gate": "AVAX_USDT", "mexc": "AVAX_USDT", "hl": "AVAX"})):
        out += [
            ("funding", "Binance funding", label, "GET", "https://fapi.binance.com/fapi/v1/fundingRate",
             {"symbol": b["bn"], "startTime": t0ms, "limit": 1000}, series([], "fundingTime")),
            ("oi", "Binance OI history", label, "GET", "https://fapi.binance.com/futures/data/openInterestHist",
             {"symbol": b["bn"], "period": "1d", "limit": 500}, series([], "timestamp")),
            ("lsr", "Binance long/short ratio", label, "GET", "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
             {"symbol": b["bn"], "period": "1d", "limit": 500}, series([], "timestamp")),
            ("kline", "Binance perp klines 1d", label, "GET", "https://fapi.binance.com/fapi/v1/klines",
             {"symbol": b["bn"], "interval": "1d", "startTime": t0ms, "limit": 1500}, series([], 0)),
            ("funding", "Bybit funding", label, "GET", "https://api.bybit.com/v5/market/funding/history",
             {"category": "linear", "symbol": b["bn"], "startTime": t0ms, "endTime": t90ms, "limit": 200}, series(["result", "list"], "fundingRateTimestamp")),
            ("oi", "Bybit OI history 1d", label, "GET", "https://api.bybit.com/v5/market/open-interest",
             {"category": "linear", "symbol": b["bn"], "intervalTime": "1d", "startTime": t0ms, "endTime": (T0 + 190 * DAY) * 1000, "limit": 200}, series(["result", "list"], "timestamp")),
            ("lsr", "Bybit long/short ratio", label, "GET", "https://api.bybit.com/v5/market/account-ratio",
             {"category": "linear", "symbol": b["bn"], "period": "1d", "limit": 500}, series(["result", "list"], "timestamp")),
            ("kline", "Bybit perp klines 1d", label, "GET", "https://api.bybit.com/v5/market/kline",
             {"category": "linear", "symbol": b["bn"], "interval": "D", "start": t0ms, "end": (T0 + 900 * DAY) * 1000, "limit": 1000}, series(["result", "list"], 0)),
            ("funding", "OKX funding (older than 2023-11-14)", label, "GET", "https://www.okx.com/api/v5/public/funding-rate-history",
             {"instId": b["okx"], "after": 1700000000000, "limit": 100}, series(["data"], "fundingTime")),
            ("oi", "OKX OI+volume 1d", label, "GET", "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
             {"ccy": b["ccy"], "period": "1D"}, series(["data"], 0)),
            ("funding", "KuCoin futures funding", label, "GET", "https://api-futures.kucoin.com/api/v1/contract/funding-rates",
             {"symbol": b["kc"], "from": t0ms, "to": t90ms}, series(["data", "dataList"], "timePoint")),
            ("kline", "KuCoin futures klines 1d", label, "GET", "https://api-futures.kucoin.com/api/v1/kline/query",
             {"symbol": b["kc"], "granularity": 1440, "from": t0ms, "to": (T0 + 190 * DAY) * 1000}, series(["data"], 0)),
            ("oi_snapshot", "KuCoin futures contract info", label, "GET", f"https://api-futures.kucoin.com/api/v1/contracts/{b['kc']}",
             {}, snapshot),
            ("funding", "Gate funding", label, "GET", "https://api.gateio.ws/api/v4/futures/usdt/funding_rate",
             {"contract": b["gate"], "from": T0, "to": T0 + 365 * DAY, "limit": 1000}, series([], "t")),
            ("oi", "Gate contract stats 1d (OI, ratios, liquidations)", label, "GET", "https://api.gateio.ws/api/v4/futures/usdt/contract_stats",
             {"contract": b["gate"], "from": T0, "interval": "1d", "limit": 100}, stats_series),
            ("funding", "Bitget funding (page 40)", label, "GET", "https://api.bitget.com/api/v2/mix/market/history-fund-rate",
             {"symbol": b["bn"], "productType": "usdt-futures", "pageSize": 100, "pageNo": 40}, series(["data"], "fundingTime")),
            ("funding", "MEXC funding (page 40)", label, "GET", "https://contract.mexc.com/api/v1/contract/funding_rate/history",
             {"symbol": b["mexc"], "page_num": 40, "page_size": 100}, series(["data", "resultList"], "settleTime")),
            ("funding", "Hyperliquid funding", label, "POST", "https://api.hyperliquid.xyz/info",
             {"type": "fundingHistory", "coin": b["hl"], "startTime": t0ms}, series([], "time")),
        ]
    return out


def run_one(tag, name, label, method, url, params, extractor):
    row = {"tag": tag, "name": name, "label": label, "http": None, "n": 0, "first": None, "last": None, "note": ""}
    try:
        if method == "POST":
            r = requests.post(url, json=params, headers=UA, timeout=TIMEOUT)
        else:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
        row["http"] = r.status_code
        if r.status_code in (403, 451):
            row["note"] = "BLOCKED from this server (likely region/IP block)"
            return row
        if r.status_code != 200:
            row["note"] = r.text[:140].replace("\n", " ")
            return row
        j = r.json()
        ts, extra = extractor(j)
        ts = [t for t in ts if t > 0]
        row["n"] = len(ts)
        if ts:
            row["first"], row["last"] = min(ts), max(ts)
        row["note"] = extra
        if not ts and not extra:
            row["note"] = json.dumps(j)[:140]
    except Exception as exc:  # one failing endpoint must never stop the test
        row["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return row


def fmt(row):
    span = f"{dt(row['first'])} .. {dt(row['last'])}" if row["first"] else "no records"
    deep = " DEEP" if row["first"] and row["first"] <= DEEP_UNTIL and row["tag"] != "oi_snapshot" else ""
    return f"{row['name']:52s} {row['label']:5s} HTTP {str(row['http']):4s} n={row['n']:5d} {span}{deep}  {row['note']}".rstrip()


def summary(rows):
    L = ["", "=" * 60, "SUMMARY (DEEP = history reaches back to about 2022-11 when asked for it)"]
    names = {"funding": "FUNDING-RATE history", "oi": "OPEN-INTEREST / stats history", "lsr": "LONG/SHORT-ratio history",
             "kline": "PERPETUAL klines (daily)", "oi_snapshot": "OI snapshot only"}
    for tag in ("funding", "oi", "lsr", "kline", "oi_snapshot"):
        sel = [r for r in rows if r["tag"] == tag]
        okn = sorted({r["name"] for r in sel if r["n"] > 0})
        deepn = sorted({r["name"] for r in sel if r["n"] > 0 and r["first"] and r["first"] <= DEEP_UNTIL and tag != "oi_snapshot"})
        blocked = sorted({r["name"] for r in sel if r["http"] in (403, 451)})
        L.append(f"{names[tag]}:")
        if tag != "oi_snapshot":
            L.append(f"  deep + reachable: {', '.join(deepn) if deepn else 'NONE'}")
        shallow = [n for n in okn if n not in deepn]
        if shallow:
            L.append(f"  reachable but shallow/snapshot: {', '.join(shallow)}")
        if blocked:
            L.append(f"  blocked: {', '.join(blocked)}")
    return "\n".join(L)


def main():
    rows = []
    for spec in tests():
        row = run_one(*spec)
        rows.append(row)
        print(fmt(row), flush=True)
        time.sleep(0.3)
    text = ["DATA ACCESS TEST from the GitHub Actions runner", f"run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "Asked for history starting 2022-11-01 where the API allows it; BTC and AVAX (a mid-cap alt with a long history).", ""]
    text += [fmt(r) for r in rows]
    text.append(summary(rows))
    out = "\n".join(text) + "\n"
    print(summary(rows))
    with open("lvar_datatest_result.txt", "w", encoding="utf-8") as fh:
        fh.write(out)


if __name__ == "__main__":
    main()
