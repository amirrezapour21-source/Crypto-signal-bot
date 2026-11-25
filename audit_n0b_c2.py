"""READ-ONLY. S1 reconstruct the earlier top-70 universe; S2 EOS/MKR/LRC archive gap facts; S3 FRED vintage test (only if FRED_API_KEY is
already in the environment; never printed); S4 official FOMC release-time scan (Fed press releases); S5 BEA/BLS probes;
S6 spot-vs-futures candle conventions + warm-up (signal logic only, no trade outcomes). Writes nothing to the repo."""
import csv, io, os, re, sys, time, zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import numpy as np, pandas as pd, requests
sys.path.insert(0, ".")

LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"; BASE = "https://data.binance.vision/data/futures/um"
SAMPLE_MONTHS = ["2022-12", "2023-09", "2024-06", "2025-03", "2026-01", "2026-09"]
EXCLUDE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "EUR", "BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD"}
# symbols that traded in the earlier backtest (ground-truth subset of the original 70; from its trades file)
KNOWN = ("1000FLOKI 1000PEPE ADA ARB AUCTION AXS BB BCH BLZ BNB BTC BZ CL CYBER DOGE ENA ETH FARTCOIN FRONT GUN HIFI HYPE JASMY KAITO LINK LTC "
         "MASK MATIC MUBARAK NEAR ORDI PEOPLE QNT SKHYNIX SOL SUI TRUMP UNI WIF WLD XAG XAU XRP ZEC").split()
S = requests.Session(); S.headers.update({"User-Agent": "Mozilla/5.0 (research-audit)"})

def get(url, **kw):
    for a in range(3):
        try:
            r = S.get(url, timeout=kw.pop("timeout", 60), **kw)
            if r.status_code in (403, 404): return r
            if r.status_code == 200: return r
        except requests.RequestException: pass
        time.sleep(1 + a)
    return None

def s3_list(prefix, delim=None):
    """-> (sub-prefixes, [(key, last_modified)]) with marker pagination."""
    subs, keys, marker = [], [], ""
    for _ in range(60):
        p = {"prefix": prefix}
        if delim: p["delimiter"] = delim
        if marker: p["marker"] = marker
        r = get(LIST, params=p)
        if r is None or r.status_code != 200: break
        root = ET.fromstring(r.content); ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        page = [(c.find(ns + "Key").text, c.find(ns + "LastModified").text) for c in root.iter(ns + "Contents")]
        keys += page; subs += [x.text for x in root.iter(ns + "Prefix") if x.text and x.text != prefix and x.text.endswith("/")]
        tr = root.find(ns + "IsTruncated")
        if tr is not None and tr.text == "true":
            nm = root.find(ns + "NextMarker"); marker = nm.text if nm is not None and nm.text else (page[-1][0] if page else "")
            if not marker: break
        else: break
    return subs, keys

def usable(s):
    b = s[:-4]
    return s.endswith("USDT") and "_" not in s and re.fullmatch(r"[A-Z0-9]+", s) and b and b not in EXCLUDE and not re.search(r"(UP|DOWN|BULL|BEAR)$", b)

def s1_universe():
    print("\n== S1 reconstruction of the earlier top-70 universe (same rule as the earlier backtest: max over 6 sample months of mean daily quote volume)")
    subs, _ = s3_list("data/futures/um/monthly/klines/", "/"); allsyms = sorted(x.split("/")[-2] for x in subs); syms = [s for s in allsyms if usable(s)]
    print("listing symbols", len(allsyms), "usable", len(syms))
    def one(j):
        s, m = j; r = get(f"{BASE}/monthly/klines/{s}/1d/{s}-1d-{m}.zip")
        if r is None or r.status_code != 200: return s, 0.0
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content)); rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), "utf-8"))); v = []
            for x in rows:
                try: v.append(float(x[7]))
                except (ValueError, IndexError): pass
            return s, float(np.mean(v)) if v else 0.0
        except Exception: return s, 0.0
    best = {}
    with ThreadPoolExecutor(24) as ex:
        for s, v in ex.map(one, [(s, m) for s in syms for m in SAMPLE_MONTHS]): best[s] = max(best.get(s, 0.0), v)
    top = [s for s, v in sorted(best.items(), key=lambda kv: -kv[1])[:70] if v > 0]
    from audit_a_n0 import SYMS
    kn = [k + "USDT" for k in KNOWN]; miss = [k for k in kn if k not in top]
    print(f"reconstructed {len(top)} symbols; known-traded subset contained: {len(kn) - len(miss)}/{len(kn)}; missing from reconstruction: {miss or 'none'}")
    print("reconstructed list:", " ".join(s[:-4] for s in top))
    print(f"overlap with the 69-symbol scope: {len(set(top) & set(SYMS))}; in reconstruction only: {' '.join(sorted(s[:-4] for s in set(top) - set(SYMS)))}")
    print("in 69-scope only:", " ".join(sorted(s[:-4] for s in set(SYMS) - set(top))))
    print("NOTE: this is a re-execution of the earlier selection rule on the archive as it is today, not the saved original list.")

def s2_gaps():
    print("\n== S2 archive facts for EOS/MKR/LRC (observations only; no cause assumed)")
    subs, _ = s3_list("data/futures/um/monthly/klines/", "/"); names = [x.split("/")[-2] for x in subs]
    for b in ("EOS", "MKR", "LRC"):
        rel = [n for n in names if n.startswith(b)]; print(f"-- {b}: symbols in archive starting with {b}: {rel}")
        s = b + "USDT"
        for kind, per in (("monthly/klines", "1m"), ("daily/klines", "1m"), ("monthly/aggTrades", None), ("daily/aggTrades", None)):
            pre = f"data/futures/um/{kind}/{s}/" + (f"{per}/" if per else ""); _, keys = s3_list(pre)
            names_ = [k for k, _ in keys if k.endswith(".zip")]
            tag = [re.search(r"(\d{4}-\d{2}(?:-\d{2})?)\.zip", k).group(1) for k in names_ if re.search(r"(\d{4}-\d{2}(?:-\d{2})?)\.zip", k)]
            by = {}
            for t in tag: by[t[:7]] = by.get(t[:7], 0) + 1
            last = max(keys, key=lambda kv: kv[0]) if keys else None
            print(f"   {kind}{'/' + per if per else ''}: files {len(tag)}, first {min(tag) if tag else '-'}, last {max(tag) if tag else '-'} (last modified {last[1] if last else '-'})")
            if "daily" in kind: print("     files per month, 2025-04..2026-10:", {m: by.get(m, 0) for m in sorted(by) if "2025-04" <= m <= "2026-10"})

def s3_fred():
    print("\n== S3 FRED vintage route")
    key = os.environ.get("FRED_API_KEY", "")
    if not key: print("no FRED_API_KEY in this environment: route NOT tested (no key requested or created)"); return
    def api(path, **p):
        try:
            r = S.get("https://api.stlouisfed.org/fred/" + path, params={**p, "api_key": key, "file_type": "json"}, timeout=60); return r.status_code, r.json()
        except Exception as e: return type(e).__name__, {}
    for sid, obs, vint in (("PAYEMS", "2024-01-01", "2024-02-02"), ("CPIAUCSL", "2023-12-01", "2024-01-11")):
        c1, a = api("series/observations", series_id=sid, observation_start=obs, observation_end=obs, realtime_start=vint, realtime_end=vint)
        c2, b = api("series/observations", series_id=sid, observation_start=obs, observation_end=obs)
        f = lambda d: (d.get("observations") or [{}])[0].get("value")
        print(f"{sid} {obs}: status {c1}/{c2}; value as known on {vint}: {f(a)}; latest revised value: {f(b)}")
    for rid, name in ((50, "Employment Situation"), (10, "CPI")):
        c, d = api("release/dates", release_id=rid, realtime_start="2022-11-01", realtime_end="2026-11-25", limit=1000, sort_order="asc")
        ds = [x["date"] for x in d.get("release_dates", [])]; print(f"{name}: status {c}, release dates {len(ds)}, first {ds[:1]}, last {ds[-1:]} (dates only, no time of day)")

def fomc_day(d):
    r = get(f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{d:%Y%m%d}a.htm", timeout=30)
    if r is None or r.status_code != 200: return d, None
    txt = re.sub(r"<[^>]+>", " ", r.text); txt = re.sub(r"\s+", " ", txt)
    m = re.search(r"[Ff]or (?:immediate )?release at (\d{1,2}:\d{2} ?[ap]\.?m\.? ?[A-Z]{2,3})", txt)
    t = re.search(r"<title>(.*?)</title>", r.text, re.S)
    return d, (m.group(1) if m else "NO_TIME_FOUND", (t.group(1).strip()[:60] if t else ""))

def s4_fomc():
    print("\n== S4 official FOMC statement release times (Fed press-release pages monetaryYYYYMMDDa.htm, Tue-Thu scan 2022-11-01..today)")
    days = [date(2022, 11, 1) + timedelta(i) for i in range((date.today() - date(2022, 11, 1)).days + 1)]; days = [d for d in days if d.weekday() in (1, 2, 3)]
    with ThreadPoolExecutor(6) as ex: res = [r for r in ex.map(fomc_day, days) if r[1]]
    print(f"pages found {len(res)} of {len(days)} days probed")
    cnt = {}
    for d, (t, ti) in res: print(f"  {d} | {t} | {ti}"); cnt[t] = cnt.get(t, 0) + 1
    print("release-time strings:", cnt)

def s5_probes():
    print("\n== S5 official release-time source probes")
    r = get("https://www.bea.gov/news/schedule")
    if r is not None and r.status_code == 200:
        t = re.sub(r"<[^>]+>", " ", r.text); tm = re.findall(r"\d{1,2}:\d{2} [AP]M", t); yr = sorted(set(re.findall(r"20\d\d", t)))
        print(f"BEA schedule: 200, time strings {len(tm)} (e.g. {tm[:3]}), years mentioned {yr[:1]}..{yr[-1:]}")
    else: print("BEA schedule:", getattr(r, "status_code", "ERR"))
    for u in ("https://www.bls.gov/schedule/news_release/empsit.htm", "https://www.bls.gov/bls/newsrels.htm", "https://www.bls.gov/schedule/2024/home.htm"):
        r = get(u); print("BLS", u.split("bls.gov")[1], getattr(r, "status_code", "ERR"))

def spot_df(rows):
    d = pd.DataFrame([(int(x[0]), float(x[1]), float(x[3]), float(x[4]), float(x[2]), float(x[5])) for x in rows], columns=["t", "open", "high", "low", "close", "volume"])
    d = d.drop_duplicates("t").sort_values("t").reset_index(drop=True); d["ts"] = pd.to_datetime(d["t"], unit="s", utc=True); return d.drop(columns="t")

def spot_long(sym, now, n=4380):
    import audit_c_parity as c
    end = int(pd.Timestamp(now).floor("4h").timestamp()); rows = []
    for _ in range(5):
        r = S.get("https://api.kucoin.com/api/v1/market/candles", params={"symbol": c.spot_sym(sym), "type": "4hour", "startAt": end - 1500 * 14400, "endAt": end}, timeout=30).json().get("data", [])
        if not r: break
        rows += r; end = min(int(x[0]) for x in r)
        if len(rows) >= n: break
        time.sleep(0.3)
    return spot_df(rows).iloc[-n:].reset_index(drop=True)

def s6_conventions(fetch_f=None, fetch_s=None):
    import audit_c_parity as c
    fetch_f = fetch_f or c.fetch_fut; fetch_s = fetch_s or spot_long
    print("\n== S6 spot (backtest source) vs futures (forward source): candle conventions and warm-up (signal logic only)")
    now = pd.Timestamp.now("UTC"); agg = dict(n=0, com=0, cl=[], op=[], vr=[], agree=[], flips=0, cmp=0, gaps=0)
    for sym in c.eng.SYMS:
        try:
            f = fetch_f(sym, now); s = fetch_s(sym, now)
            m = f.merge(s, on="ts", suffixes=("_f", "_s")); agg["n"] += len(f); agg["com"] += len(m)
            if not len(m): print(sym, "no common timestamps"); continue
            agg["cl"] += list((m.close_s / m.close_f - 1).abs() * 1e4); agg["op"] += list((m.open_s / m.open_f - 1).abs() * 1e4); agg["vr"] += list(m.volume_s / m.volume_f)
            vf = c.ind(*(f[k].tolist() for k in ("close", "high", "low", "volume")))[5]; vs = c.ind(*(s[k].tolist() for k in ("close", "high", "low", "volume")))[5]
            df_ = dict(zip(f.ts, [(v >= x) if x is not None else None for v, x in zip(f.volume, vf)])); ds_ = dict(zip(s.ts, [(v >= x) if x is not None else None for v, x in zip(s.volume, vs)]))
            for t in m.ts:
                if df_.get(t) is not None and ds_.get(t) is not None: agg["agree"].append(df_[t] == ds_[t])
            agg["gaps"] += int((s.ts.diff().dropna() != pd.Timedelta(hours=4)).sum())
            if len(s) > 1000:  # warm-up: spot with long history vs only the last 600 bars
                a, _ = c.sig_ind(*(s[k].tolist() for k in ("close", "high", "low", "volume"))); b, _ = c.sig_ind(*(s.iloc[-600:][k].tolist() for k in ("close", "high", "low", "volume")))
                off = len(s) - 600; A = {i - off for i in a if i - off >= 100}; B = {i for i in b if i >= 100}; agg["cmp"] += len(A | B); agg["flips"] += len(A ^ B)
        except Exception as e: print(sym, "ERR", type(e).__name__)
    p = lambda x, q: round(float(np.percentile(x, q)), 3) if len(x) else None
    print(f"bars futures {agg['n']}, common timestamps {agg['com']} ({100 * agg['com'] / max(agg['n'], 1):.1f}%), spot 4H gaps {agg['gaps']}")
    print(f"|close diff| bps: median {p(agg['cl'], 50)}, p95 {p(agg['cl'], 95)}; |open diff| bps: median {p(agg['op'], 50)}, p95 {p(agg['op'], 95)}")
    print(f"volume ratio spot/futures: p10 {p(agg['vr'], 10)}, median {p(agg['vr'], 50)}, p90 {p(agg['vr'], 90)} (units differ: base coin vs contracts)")
    print(f"volume >= median20 decision agreement spot vs futures: {100 * np.mean(agg['agree']):.1f}% of {len(agg['agree'])} bars" if agg["agree"] else "no agreement data")
    print(f"warm-up on spot (long history vs last 600 bars): signal decisions compared {agg['cmp']}, flips {agg['flips']}")

if __name__ == "__main__":
    print("audit_n0b_c2 read-only")
    for f in (s1_universe, s2_gaps, s3_fred, s4_fomc, s5_probes, s6_conventions):
        try: f()
        except Exception as e: print(f.__name__, "FAILED", type(e).__name__)
    print("\nNothing was written to the repository; no trade outcome fields were read.")
