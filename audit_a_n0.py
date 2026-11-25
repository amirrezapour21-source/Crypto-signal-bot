"""READ-ONLY data audit. A: Binance USDT-M archive coverage/quality (klines 1m w/ taker-buy volume, aggTrades w/ is_buyer_maker).
N0: reachability probes for macro/news sources. No strategy, no writes to the repo (prints only)."""
import csv, io, json, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import requests

BASE = "https://data.binance.vision/data/futures/um"
SYMS = ("BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT LTC TRX NEAR UNI ATOM ETC FIL APT ARB OP SUI INJ 1000PEPE 1000SHIB WIF FET AAVE MKR LDO SEI TIA "
        "ORDI 1000BONK 1000FLOKI JUP WLD ENA TAO DYDX GALA SAND MANA AXS CRV SNX RUNE ICP BCH XLM HBAR VET ALGO EOS THETA STX IMX PENDLE "
        "COMP ZEC DASH NEO KAVA APE CHZ ENS GMT LRC ONE ZIL").split()
SYMS = [s + "USDT" for s in SYMS]
MONTHS = [f"{y}-{m:02d}" for y in range(2022, 2027) for m in range(1, 13) if (2022, 11) <= (y, m) <= (2026, 10)]
UA = {"User-Agent": "Mozilla/5.0 (research-audit)"}
S = requests.Session(); S.headers.update(UA)

def head(url):
    for _ in range(3):
        try:
            r = S.head(url, timeout=30, allow_redirects=True)
            return r.status_code
        except Exception as e:
            err = type(e).__name__
    return err

def get_csv(url):
    r = S.get(url, timeout=120)
    if r.status_code != 200: return None, r.status_code
    z = zipfile.ZipFile(io.BytesIO(r.content)); n = z.namelist()[0]
    return list(csv.reader(io.TextIOWrapper(z.open(n), "utf-8"))), 200

def compress(ms):  # list of 'YYYY-MM' -> ranges text
    if not ms: return "-"
    idx = sorted(MONTHS.index(m) for m in ms); out = []; a = b = idx[0]
    for i in idx[1:]:
        if i == b + 1: b = i
        else: out.append((a, b)); a = b = i
    out.append((a, b)); return ",".join(MONTHS[a] if a == b else f"{MONTHS[a]}..{MONTHS[b]}" for a, b in out)

def coverage():
    tasks = [(s, m, k) for s in SYMS for m in MONTHS for k in ("klines/1m", "aggTrades")]
    def f(t):
        s, m, k = t; n = "1m" if k == "klines/1m" else "aggTrades"
        return t, head(f"{BASE}/monthly/{k}/{s}/{s}-{n}-{m}.zip")
    with ThreadPoolExecutor(24) as ex: res = dict(ex.map(f, tasks))
    print("\n== A1 monthly archive coverage (HEAD 200 = present), months 2022-11..2026-10 =", len(MONTHS))
    print("symbol | kl1m present/N missing | aggTrades present/N missing")
    bad = []
    for s in SYMS:
        row = []
        for k in ("klines/1m", "aggTrades"):
            ok = [m for m in MONTHS if res[(s, m, k)] == 200]; miss = [m for m in MONTHS if res[(s, m, k)] != 200]
            other = {res[(s, m, k)] for m in miss} - {404}
            row.append(f"{len(ok)}/{len(MONTHS)} miss {compress(miss)}" + (f" codes {other}" if other else ""))
        print(s, "|", " | ".join(row))
        if not any(res[(s, m, "klines/1m")] == 200 for m in MONTHS): bad.append(s)
    print("symbols with no 1m klines at all:", bad or "none")
    n200 = sum(v == 200 for v in res.values()); print(f"total files present {n200}/{len(res)}")

def validate_klines(s, m):
    rows, code = get_csv(f"{BASE}/monthly/klines/{s}/1m/{s}-1m-{m}.zip")
    if rows is None: return f"{s} {m} 1m: HTTP {code}"
    hdr = rows[0] if rows and not rows[0][0].isdigit() else None
    data = rows[1:] if hdr else rows
    ts = [int(r[0]) for r in data]; unit = "us" if ts[0] > 10**14 else "ms"; step = 60000 if unit == "ms" else 60_000_000
    gaps = sum(1 for a, b in zip(ts, ts[1:]) if b - a != step); dup = len(ts) - len(set(ts))
    y, mo = map(int, m.split("-")); nxt = date(y + (mo == 12), mo % 12 + 1, 1); exp = (nxt - date(y, mo, 1)).days * 1440
    tb = [float(r[9]) for r in data]; vol = [float(r[5]) for r in data]
    bad_tb = sum(1 for a, b in zip(tb, vol) if a < 0 or a > b * 1.0001)
    return (f"{s} {m} 1m: header={'yes' if hdr else 'NO'} cols={len(data[0])} unit={unit} rows={len(data)}/{exp} gaps={gaps} dup={dup} "
            f"taker_buy_col={'ok' if (hdr and len(hdr) > 9 and 'taker' in hdr[9]) else 'check '+str(hdr[9] if hdr and len(hdr)>9 else None)} "
            f"taker_buy_all_zero={all(x == 0 for x in tb)} taker>vol rows={bad_tb}")

def validate_agg(s, d):
    rows, code = get_csv(f"{BASE}/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip")
    if rows is None: return f"{s} {d} aggTrades: HTTP {code}"
    hdr = rows[0] if rows and not rows[0][0].isdigit() else None; data = rows[1:] if hdr else rows
    tt = [int(r[5]) for r in data]; mono = all(b >= a for a, b in zip(tt, tt[1:]))
    vals = {r[6].lower() for r in data}
    return f"{s} {d} aggTrades: header={hdr} rows={len(data)} is_buyer_maker values={sorted(vals)} time_monotonic={mono} unit={'us' if tt[0] > 10**14 else 'ms'}"

def quality():
    print("\n== A2 schema/quality samples (3 symbols x 3 periods)")
    for s in ("BTCUSDT", "SOLUSDT", "PENDLEUSDT"):
        for m in ("2022-11", "2024-06", "2026-10"):
            try: print(validate_klines(s, m))
            except Exception as e: print(s, m, "1m ERROR", type(e).__name__, e)
    for s in ("BTCUSDT", "SOLUSDT"):
        for d in ("2022-11-15", "2024-06-15", "2026-10-15"):
            try: print(validate_agg(s, d))
            except Exception as e: print(s, d, "aggTrades ERROR", type(e).__name__, e)

PROBES = [  # (name, method, url, note)
 ("FOMC calendar (federalreserve.gov)", "GET", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "official dates"),
 ("BLS CPI schedule", "GET", "https://www.bls.gov/schedule/news_release/cpi.htm", "official schedule (history limited)"),
 ("BLS public API v1 (no key)", "POST", "https://api.bls.gov/publicAPI/v1/timeseries/data/CUUR0000SA0", "latest revised values; no vintages"),
 ("FRED graph CSV (no key)", "GET", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", "current (revised) values"),
 ("ALFRED vintage CSV (no key)", "GET", "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=PAYEMS&vintage_date=2024-01-05", "first-release vintages"),
 ("FRED API vintagedates (needs key)", "GET", "https://api.stlouisfed.org/fred/series/vintagedates?series_id=CPIAUCSL&file_type=json", "expect 400 without key"),
 ("BEA release schedule", "GET", "https://www.bea.gov/news/schedule", "GDP/PCE dates"),
 ("Census retail sales", "GET", "https://www.census.gov/retail/index.html", "retail sales"),
 ("ISM PMI", "GET", "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/", "PMI; licensed/restricted"),
 ("Consensus: forexfactory", "GET", "https://www.forexfactory.com/calendar", "ToS-restricted; no verified bulk history"),
 ("Consensus: investing.com", "GET", "https://www.investing.com/economic-calendar/", "ToS-restricted"),
 ("Consensus: tradingeconomics", "GET", "https://tradingeconomics.com/calendar", "paid API for history"),
 ("Unscheduled news: GDELT DOC API", "GET", "https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin&mode=artlist&maxrecords=3&format=json", "free, timestamped (15-min), 2017+ via API windows"),
]

def probes():
    print("\n== N0 source reachability from this runner (status only; NO scraping, ToS not bypassed)")
    for name, meth, url, note in PROBES:
        try:
            r = S.post(url, json={"startyear": "2024", "endyear": "2024"}, timeout=40) if meth == "POST" else S.get(url, timeout=40)
            extra = ""
            if "alfred" in url: extra = " first-lines=" + repr(r.text[:80])
            if "gdelt" in url: extra = " body=" + repr(r.text[:80])
            print(f"{r.status_code} | {name} | bytes={len(r.content)} | {note}{extra}")
        except Exception as e: print(f"ERR | {name} | {type(e).__name__} | {note}")

if __name__ == "__main__":
    print("audit_a_n0 read-only; symbols", len(SYMS), "months", len(MONTHS))
    for f in (coverage, quality, probes):
        try: f()
        except Exception as e: print(f.__name__, "FAILED", type(e).__name__, e)
    print("\nNothing was written to the repository.")
