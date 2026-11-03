"""
DATA ACCESS TEST v2 (read-only research, no trading logic).
v1 result: Binance and Bybit APIs are blocked from GitHub servers; KuCoin futures klines work back to 2022-11; Hyperliquid funding works;
OI / long-short / liquidation history was not found. v2 fixes my own parser and parameter mistakes (KuCoin funding, Gate window, MEXC page)
and adds the one source that is usually NOT blocked: the public Binance download site (data.binance.vision), which holds daily files for
open interest, long/short ratios, funding and liquidations. Nothing is written except lvar_datatest_result.txt.
"""
import csv
import io
import json
import time
import zipfile
from datetime import datetime, timezone

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; research-datatest)"}
DAY = 86400
T0 = 1667260800  # 2022-11-01 00:00 UTC
DEEP_UNTIL = T0 + 90 * DAY  # "deep" = the oldest record is within 90 days of T0
HL_START = 1690000000       # Hyperliquid went live in 2023-05; deep if history reaches 2023-07
TIMEOUT = 40


def norm(v):
    v = float(v)
    return v / 1000.0 if v > 1e11 else v


def dt(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def parse_time(s):
    s = str(s).strip()
    try:
        return norm(float(s))
    except ValueError:
        pass
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, f).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError("not a time: " + s[:30])


def walk(j, path):
    for p in path:
        j = j[p]
    return j


def series(path, key):
    def f(j):
        items = walk(j, path)
        return [norm(it[key]) for it in items], ""
    return f


def kucoin_funding(j):
    d = j["data"]
    items = d if isinstance(d, list) else d.get("dataList", [])
    ts = []
    for it in items:
        v = it.get("timepoint", it.get("timePoint"))
        if v is not None:
            ts.append(norm(v))
    return ts, ""


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


def empty_row(tag, name, label, deep_until=DEEP_UNTIL):
    return {"tag": tag, "name": name, "label": label, "http": None, "n": 0, "first": None, "last": None, "note": "", "deep_until": deep_until}


def api_tests():
    now = int(time.time())
    t0ms = T0 * 1000
    out = []
    for label, b in (("BTC", {"kc": "XBTUSDTM", "gate": "BTC_USDT", "hl": "BTC"}),
                     ("AVAX", {"kc": "AVAXUSDTM", "gate": "AVAX_USDT", "hl": "AVAX"})):
        out += [
            ("funding", "KuCoin futures funding 2022-11", label, "GET", "https://api-futures.kucoin.com/api/v1/contract/funding-rates",
             {"symbol": b["kc"], "from": t0ms, "to": (T0 + 30 * DAY) * 1000}, kucoin_funding, DEEP_UNTIL),
            ("funding", "KuCoin futures funding 2024-06", label, "GET", "https://api-futures.kucoin.com/api/v1/contract/funding-rates",
             {"symbol": b["kc"], "from": 1717200000000, "to": 1719792000000}, kucoin_funding, 1717200000 + 5 * DAY),
            ("kline", "KuCoin futures klines 1d", label, "GET", "https://api-futures.kucoin.com/api/v1/kline/query",
             {"symbol": b["kc"], "granularity": 1440, "from": t0ms, "to": (T0 + 190 * DAY) * 1000}, series(["data"], 0), DEEP_UNTIL),
            ("snapshot", "KuCoin futures contract info", label, "GET", f"https://api-futures.kucoin.com/api/v1/contracts/{b['kc']}",
             {}, snapshot, DEEP_UNTIL),
            ("funding", "Gate funding (latest 1000)", label, "GET", "https://api.gateio.ws/api/v4/futures/usdt/funding_rate",
             {"contract": b["gate"], "limit": 1000}, series([], "t"), DEEP_UNTIL),
            ("oi", "Gate stats 1d (last 170 days)", label, "GET", "https://api.gateio.ws/api/v4/futures/usdt/contract_stats",
             {"contract": b["gate"], "from": now - 170 * DAY, "interval": "1d", "limit": 100}, stats_series, now - 160 * DAY),
            ("funding", "Hyperliquid funding from launch", label, "POST", "https://api.hyperliquid.xyz/info",
             {"type": "fundingHistory", "coin": b["hl"], "startTime": t0ms}, series([], "time"), HL_START),
            ("funding", "Hyperliquid funding last 20 days", label, "POST", "https://api.hyperliquid.xyz/info",
             {"type": "fundingHistory", "coin": b["hl"], "startTime": (now - 20 * DAY) * 1000}, series([], "time"), now - 18 * DAY),
        ]
    return out


def run_one(tag, name, label, method, url, params, extractor, deep_until=DEEP_UNTIL):
    row = empty_row(tag, name, label, deep_until)
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


def run_mexc(label, symbol):
    """MEXC pages go backwards in time: read page 1 for totalPage, then the last page for the oldest records."""
    row = empty_row("funding", "MEXC funding (oldest page)", label)
    url = "https://contract.mexc.com/api/v1/contract/funding_rate/history"
    try:
        r1 = requests.get(url, params={"symbol": symbol, "page_num": 1, "page_size": 100}, headers=UA, timeout=TIMEOUT)
        row["http"] = r1.status_code
        if r1.status_code != 200:
            row["note"] = r1.text[:140].replace("\n", " ")
            return row
        d1 = r1.json()["data"]
        total_pages, total = int(d1["totalPage"]), int(d1["totalCount"])
        r2 = requests.get(url, params={"symbol": symbol, "page_num": total_pages, "page_size": 100}, headers=UA, timeout=TIMEOUT)
        d2 = r2.json()["data"]
        ts = [norm(it["settleTime"]) for it in d1["resultList"] + d2["resultList"]]
        row["n"] = total
        if ts:
            row["first"], row["last"] = min(ts), max(ts)
        row["note"] = f"totalCount={total}"
    except Exception as exc:
        row["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return row


def vision_tests():
    base = "https://data.binance.vision/data/futures/um"
    out = []
    for sym in ("BTCUSDT", "AVAXUSDT", "1000PEPEUSDT"):
        for d in ("2021-12-15", "2022-11-01", "2023-11-01", "2024-11-01", "2026-10-01"):
            out.append(("v_metrics", f"Vision metrics (OI, ratios) {d}", sym, f"{base}/daily/metrics/{sym}/{sym}-metrics-{d}.zip"))
        for m in ("2020-06", "2022-11", "2024-11"):
            out.append(("v_funding", f"Vision fundingRate {m}", sym, f"{base}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip"))
        for d in ("2022-11-01", "2023-11-01"):
            out.append(("v_liq", f"Vision liquidationSnapshot {d}", sym, f"{base}/daily/liquidationSnapshot/{sym}/{sym}-liquidationSnapshot-{d}.zip"))
        out.append(("v_kline", "Vision klines 1d 2022-11", sym, f"{base}/monthly/klines/{sym}/1d/{sym}-1d-2022-11.zip"))
    return out


def run_vision(tag, name, label, url):
    row = empty_row(tag, name, label)
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        row["http"] = r.status_code
        if r.status_code != 200:
            body = r.text[:90].replace("\n", " ")
            row["note"] = f"not available (404 = file does not exist; 403 may mean blocked): {body}"
            return row
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as fh:
            rows = list(csv.reader(io.TextIOWrapper(fh, "utf-8")))
        header = []
        if rows:
            try:
                parse_time(rows[0][0])
            except ValueError:
                header = rows[0]
                rows = rows[1:]
        ts = []
        for rw in rows:
            try:
                ts.append(parse_time(rw[0]))
            except (ValueError, IndexError):
                continue
        row["n"] = len(rows)
        if ts:
            row["first"], row["last"] = min(ts), max(ts)
        row["note"] = "columns: " + ",".join(header)[:150]
    except Exception as exc:
        row["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return row


def fmt(row):
    span = f"{dt(row['first'])} .. {dt(row['last'])}" if row["first"] else "no records"
    deep = " DEEP" if row["first"] and row["first"] <= row["deep_until"] and row["tag"] not in ("snapshot",) and not row["tag"].startswith("v_") else ""
    return f"{row['name']:46s} {row['label']:13s} HTTP {str(row['http']):4s} n={row['n']:6d} {span}{deep}  {row['note']}".rstrip()


def summary(rows):
    L = ["", "=" * 60, "SUMMARY"]
    api_names = {"funding": "FUNDING-RATE history (exchange APIs)", "oi": "OPEN-INTEREST / ratios / liquidations (exchange APIs)",
                 "kline": "PERPETUAL klines daily (exchange APIs)", "snapshot": "Open-interest snapshot only"}
    for tag in ("funding", "oi", "kline", "snapshot"):
        sel = [r for r in rows if r["tag"] == tag]
        if not sel:
            continue
        deep = sorted({r["name"] for r in sel if r["n"] > 0 and r["first"] and r["first"] <= r["deep_until"] and tag != "snapshot"})
        other = sorted({r["name"] for r in sel if r["n"] > 0} - set(deep))
        L.append(f"{api_names[tag]}:")
        if tag != "snapshot":
            L.append(f"  deep: {', '.join(deep) if deep else 'NONE'}")
        if other:
            L.append(f"  reachable but not deep / snapshot: {', '.join(other)}")
    vnames = {"v_metrics": "data.binance.vision METRICS (open interest, long/short ratios, taker ratio)", "v_funding": "data.binance.vision FUNDING",
              "v_liq": "data.binance.vision LIQUIDATIONS", "v_kline": "data.binance.vision KLINES (site reachability check)"}
    for tag in ("v_metrics", "v_funding", "v_liq", "v_kline"):
        sel = [r for r in rows if r["tag"] == tag]
        L.append(f"{vnames[tag]}:")
        for sym in sorted({r["label"] for r in sel}):
            ok = [r["name"].split()[-1] for r in sel if r["label"] == sym and r["http"] == 200 and r["n"] > 0]
            bad = [f"{r['name'].split()[-1]}({r['http']})" for r in sel if r["label"] == sym and not (r["http"] == 200 and r["n"] > 0)]
            L.append(f"  {sym}: found {', '.join(ok) if ok else 'NONE'}" + (f" | missing/blocked {', '.join(bad)}" if bad else ""))
    return "\n".join(L)


def main():
    rows = []

    def add(row):
        rows.append(row)
        print(fmt(row), flush=True)
        time.sleep(0.3)

    for spec in api_tests():
        add(run_one(*spec))
    add(run_mexc("BTC", "BTC_USDT"))
    add(run_mexc("AVAX", "AVAX_USDT"))
    for spec in vision_tests():
        add(run_vision(*spec))
    text = ["DATA ACCESS TEST v2 from the GitHub Actions runner", f"run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "Binance/Bybit exchange APIs were already shown to be blocked in v1 (HTTP 451/403) and are not repeated.", ""]
    text += [fmt(r) for r in rows]
    text.append(summary(rows))
    out = "\n".join(text) + "\n"
    print(summary(rows))
    with open("lvar_datatest_result.txt", "w", encoding="utf-8") as fh:
        fh.write(out)


if __name__ == "__main__":
    main()
