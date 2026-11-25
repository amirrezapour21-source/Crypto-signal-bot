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
    """A1 (corrected path): monthly 1m klines presence. aggTrades coverage/A2 quality already reported in the first run."""
    tasks = [(s, m) for s in SYMS for m in MONTHS]
    f = lambda t: (t, head(f"{BASE}/monthly/klines/{t[0]}/1m/{t[0]}-1m-{t[1]}.zip"))
    with ThreadPoolExecutor(24) as ex: res = dict(ex.map(f, tasks))
    print("\n== A1b monthly 1m klines coverage (HEAD 200 = present), months 2022-11..2026-10 =", len(MONTHS))
    for s in SYMS:
        ok = [m for m in MONTHS if res[(s, m)] == 200]; miss = [m for m in MONTHS if res[(s, m)] != 200]
        other = {res[(s, m)] for m in miss} - {404}
        print(f"{s} | {len(ok)}/{len(MONTHS)} miss {compress(miss)}" + (f" codes {other}" if other else ""))
    print(f"total 1m kline files present {sum(v == 200 for v in res.values())}/{len(res)}")

PROBES = [
 ("BLS CPI schedule (browser UA)", "GET", "https://www.bls.gov/schedule/news_release/cpi.htm"),
 ("FRED graph CSV (no key)", "GET", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"),
 ("ALFRED vintage CSV (no key)", "GET", "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=PAYEMS&vintage_date=2024-01-05"),
 ("ALFRED API-style (no key)", "GET", "https://api.stlouisfed.org/fred/series/observations?series_id=PAYEMS&realtime_start=2024-01-05&realtime_end=2024-01-05&file_type=json"),
 ("ISM report page", "GET", "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/"),
]

def probes():
    print("\n== N0b retry of failed/ambiguous source probes (status only; no scraping)")
    h = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36", "Accept": "text/html,text/csv,*/*"}
    for name, meth, url in PROBES:
        for a in range(2):
            try:
                r = requests.get(url, headers=h, timeout=90)
                print(f"{r.status_code} | {name} | bytes={len(r.content)} | head={r.text[:90]!r}"); break
            except Exception as e:
                if a: print(f"ERR | {name} | {type(e).__name__}")

if __name__ == "__main__":
    print("audit_a_n0 v2 read-only (A1b klines coverage with corrected path + N0b probe retries); symbols", len(SYMS))
    for f in (coverage, probes):
        try: f()
        except Exception as e: print(f.__name__, "FAILED", type(e).__name__, e)
    print("\nNothing was written to the repository.")
