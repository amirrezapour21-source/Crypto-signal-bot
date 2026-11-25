"""READ-ONLY full-archive integrity scan of monthly 1m kline ZIPs (Binance USDT-M, 2022-11..2026-10) for the 69-symbol scope.
Checks: ZIP CRC, CSV schema, timestamp unit, order/duplicates/gaps, row counts, taker-buy validity, OHLC sanity.
Missing files are reported separately: pre-listing (before the first present month) vs unexplained (inside/after the present range)."""
import calendar, io, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
sys.path.insert(0, ".")
from audit_a_n0 import SYMS, MONTHS, BASE, UA

COLS = ["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbqv", "ig"]
S = requests.Session(); S.headers.update(UA)

def get(url):
    for a in range(3):
        try:
            r = S.get(url, timeout=120)
            if r.status_code in (403, 404): return None, r.status_code
            if r.status_code == 200: return r.content, 200
        except requests.RequestException: pass
    return None, "ERR"

def check(sym, m, data):
    """-> (list of issue strings, rows). Pure function on the ZIP bytes."""
    z = zipfile.ZipFile(io.BytesIO(data)); bad = z.testzip()
    if bad: return [f"ZIP_CRC:{bad}"], 0
    if len(z.namelist()) != 1: return [f"ZIP_MEMBERS:{len(z.namelist())}"], 0
    raw = z.read(z.namelist()[0]); hdr = raw[:12].lower().startswith(b"open_time")
    df = pd.read_csv(io.BytesIO(raw)) if hdr else pd.read_csv(io.BytesIO(raw), header=None)
    df.columns = COLS if df.shape[1] == 12 else range(df.shape[1])
    if df.shape[1] != 12: return [f"COLS:{df.shape[1]}"], len(df)
    out = []; y, mo = map(int, m.split("-")); exp = calendar.monthrange(y, mo)[1] * 1440
    t = df["t"]
    if not (t.between(10**12, 10**14).all()): return ["TS_UNIT"], len(df)
    d = t.diff().dropna(); dup = int(t.duplicated().sum()); back = int((d < 0).sum()); gap = int((d[d > 60000] / 60000 - 1).sum())
    start = int(pd.Timestamp(f"{m}-01", tz="UTC").timestamp() * 1000)
    if len(df) != exp: out.append(f"ROWS:{len(df)}/{exp}")
    if dup: out.append(f"DUP:{dup}")
    if back: out.append(f"NOT_SORTED:{back}")
    if gap: out.append(f"GAP_MIN:{gap}")
    if int(t.iloc[0]) != start: out.append("FIRST_TS")
    if int(t.iloc[-1]) != start + (exp - 1) * 60000: out.append("LAST_TS")
    tb, v = df["tbv"], df["v"]
    nan = int(df[["o", "h", "l", "c", "v", "tbv"]].isna().sum().sum())
    if nan: out.append(f"NAN:{nan}")
    neg = int(((tb < 0) | (v < 0)).sum()); over = int((tb > v * (1 + 1e-6) + 1e-9).sum())
    if neg: out.append(f"NEG_VOL:{neg}")
    if over: out.append(f"TAKER_GT_VOL:{over}")
    if len(df) and (tb == 0).all() and (v > 0).any(): out.append("TAKER_ALL_ZERO")
    ohlc = int(((df["h"] < df["l"]) | (df["o"] > df["h"]) | (df["o"] < df["l"]) | (df["c"] > df["h"]) | (df["c"] < df["l"]) | (df[["o", "h", "l", "c"]] <= 0).any(axis=1)).sum())
    if ohlc: out.append(f"OHLC:{ohlc}")
    return out, len(df)

def one(job):
    s, m = job; data, code = get(f"{BASE}/monthly/klines/{s}/1m/{s}-1m-{m}.zip")
    if data is None: return s, m, code, None, 0
    try: iss, n = check(s, m, data); return s, m, 200, iss, n
    except Exception as e: return s, m, 200, [f"UNREADABLE:{type(e).__name__}"], 0

def main():
    k, n = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1)); syms = SYMS[k::n]
    print(f"shard {k}/{n}: {len(syms)} symbols: {' '.join(syms)}")
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, [(s, m) for s in syms for m in MONTHS]))
    ok = sum(1 for r in res if r[3] == []); rows = sum(r[4] for r in res)
    issues = [r for r in res if r[3]]; miss = [r for r in res if r[3] is None]
    print(f"files present {len(res) - len(miss)}, clean {ok}, with issues {len(issues)}, missing {len(miss)}, rows scanned {rows}")
    for s, m, c, iss, _ in issues: print(f"ISSUE {s} {m}: {' '.join(iss)}")
    pre = unexp = 0
    for s in syms:
        pres = [m for m in MONTHS if any(r[0] == s and r[1] == m and r[3] is not None for r in res)]
        if not pres: print(f"MISSING_ALL {s}"); continue
        lo, hi = MONTHS.index(pres[0]), MONTHS.index(pres[-1]); pm = [m for m in MONTHS if any(r[0] == s and r[1] == m and r[3] is None for r in res)]
        u = [m for m in pm if not (MONTHS.index(m) < lo)]; pre += len(pm) - len(u); unexp += len(u)
        if u or MONTHS.index(pres[-1]) < len(MONTHS) - 1:
            print(f"UNEXPLAINED_MISSING {s}: {u} (present {pres[0]}..{pres[-1]}; codes {sorted({str(r[2]) for r in res if r[0] == s and r[3] is None})})")
    print(f"missing split: pre-listing {pre}, unexplained {unexp}")
    print("Nothing was written to the repository.")

if __name__ == "__main__": main()
