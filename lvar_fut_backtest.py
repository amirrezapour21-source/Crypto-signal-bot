"""
FUTURES POSITIONING BACKTEST - Binance USDT-M perpetuals, daily bars, real open interest and real funding.
Hypotheses fixed IN ADVANCE, nothing tuned (read-only research). Data: the public download site data.binance.vision
(perpetual klines, funding history, daily open-interest files; the data test showed it is reachable from GitHub servers).

Universe: Binance USDT-M perpetuals listed on the download site, including contracts that were later delisted (less survivorship bias).
  The 70 symbols with the highest average daily dollar volume in six sample months are kept; a coin is tradable on a day only if its
  30-day average dollar volume is >= 20M USDT. Contract names with "_" (dated futures), stablecoins and index contracts are dropped.
Period: 2022-11-01 up to the end of the last complete month.

Hypothesis A - crowded-positioning reversal, both directions (eligible):
  SHORT after the day closes if: average daily funding of the last 3 days >= +0.09% per day (3x the normal 0.03%) AND the day is red
  AND it closes below the previous day's low. LONG is the mirror (funding <= -0.09%, green day closing above the previous day's high).
  Entry = next open. Stop = 2 x ATR14. Target = 2R. Time stop = close of the 7th day. Risk must be 2%-15% of price.
Hypothesis B - leveraged pump fade, SHORT only (eligible):
  Close >= +15% over 5 days AND open-interest value >= +15% over 5 days AND average daily funding of the last 3 days >= +0.03%,
  then a red day closing below the previous day's low. Entry = next open. Stop = 1% above the highest high of the last 4 days.
  Target = 2R. Time stop = close of the 10th day. Risk must be 5%-35% of price.
Controls (information only, not eligible): A0 = the A price trigger without the funding condition; B0 = the B price trigger without the
  open-interest and funding conditions. They show whether the positioning data adds anything over price alone.
Common: one position per coin, at most 8 open positions (the strongest signal wins a tie), SL before TP inside one bar, SL gaps fill at the open.
Costs: taker fee + slippage 0.20% round trip (stress 0.40%) PLUS the real funding paid or received for every settlement while the position
  was open (the whole exit day is counted). Long pays positive funding, short receives it.
Pass rule (fixed before the run, applied to A and to B separately): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
Two hypotheses are tested, so a pass is only a candidate for forward paper testing, not proof.
"""
import argparse
import csv
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import lvar_daily_backtest as B

DAY = B.DAY
BASE = "https://data.binance.vision/data/futures/um"
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research-backtest)"}
START_Y, START_M = 2022, 11
SAMPLE_MONTHS = ["2022-12", "2023-09", "2024-06", "2025-03", "2026-01", "2026-09"]
TOP_N = 70
MIN_VOL30 = 20_000_000
WORKERS = 24
ATR_N = 14
RR = 2.0
COST_RT = 0.0020
COST_STRESS = 0.0040
EXCLUDE_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "EUR", "BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD"}
FALLBACK = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
            "TRXUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
            "1000PEPEUSDT", "1000SHIBUSDT", "WIFUSDT", "FETUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "SEIUSDT", "TIAUSDT", "ORDIUSDT",
            "1000BONKUSDT", "1000FLOKIUSDT", "JUPUSDT", "WLDUSDT", "ENAUSDT", "TAOUSDT", "DYDXUSDT", "GALAUSDT", "SANDUSDT", "MANAUSDT",
            "AXSUSDT", "CRVUSDT", "SNXUSDT", "RUNEUSDT", "ICPUSDT", "BCHUSDT", "XLMUSDT", "HBARUSDT", "VETUSDT", "ALGOUSDT", "EOSUSDT",
            "THETAUSDT", "STXUSDT", "IMXUSDT", "PENDLEUSDT"]

HYP = {
    "A": {"sides": (-1, 1), "fth": 0.0009, "ret5": None, "oi5": None, "hold": 7, "stop": "atr"},
    "A0": {"sides": (-1, 1), "fth": None, "ret5": None, "oi5": None, "hold": 7, "stop": "atr"},
    "B": {"sides": (-1,), "fth": 0.0003, "ret5": 0.15, "oi5": 0.15, "hold": 10, "stop": "peak"},
    "B0": {"sides": (-1,), "fth": None, "ret5": 0.15, "oi5": None, "hold": 10, "stop": "peak"},
}

SESSION = requests.Session()
SESSION.headers.update(UA)
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS * 2))


# ---------------------------------------------------------------- download helpers
def norm_t(v):
    v = float(v)
    if v > 1e14:
        return v / 1e6
    return v / 1000.0 if v > 1e11 else v


def fetch(url, retries=3):
    """bytes of the file, or None when it does not exist / cannot be downloaded."""
    for a in range(retries):
        try:
            r = SESSION.get(url, timeout=40)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
            time.sleep(1 + a)
        except requests.RequestException:
            time.sleep(1 + a)
    return None


def read_zip_rows(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    with z.open(z.namelist()[0]) as fh:
        return list(csv.reader(io.TextIOWrapper(fh, "utf-8")))


def parse_klines(rows):
    """(day_start, open, high, low, close, quote_volume) from a daily-kline CSV (header optional)."""
    out = []
    for r in rows:
        try:
            out.append((int(norm_t(r[0]) // DAY * DAY), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])))
        except (ValueError, IndexError):
            continue
    return out


def parse_funding(rows):
    """(settlement_time, rate) from a funding CSV: calc_time, funding_interval_hours, last_funding_rate."""
    out = []
    for r in rows:
        try:
            out.append((norm_t(r[0]), float(r[2])))
        except (ValueError, IndexError):
            continue
    return out


def parse_metrics(rows):
    """(day_start, open_interest_value at the last 5-minute row of that day) from a daily metrics CSV."""
    best = {}
    for r in rows:
        try:
            ts = datetime.strptime(r[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            v = float(r[3])
        except (ValueError, IndexError):
            continue
        d = int(ts // DAY * DAY)
        if d not in best or ts > best[d][0]:
            best[d] = (ts, v)
    return [(d, v) for d, (ts, v) in best.items()]


def month_list(now=None):
    now = now or datetime.now(timezone.utc)
    y, m, out = START_Y, START_M, []
    while (y, m) < (now.year, now.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def usable_symbol(s):
    if not s.endswith("USDT") or "_" in s or not re.fullmatch(r"[A-Z0-9]+", s):
        return False
    base = s[:-4]
    return bool(base) and base not in EXCLUDE_BASES and not re.search(r"(UP|DOWN|BULL|BEAR)$", base)


def list_symbols():
    prefix = "data/futures/um/monthly/klines/"
    names, marker = [], ""
    try:
        for _ in range(30):
            params = {"delimiter": "/", "prefix": prefix}
            if marker:
                params["marker"] = marker
            r = SESSION.get(LIST_URL, params=params, timeout=40)
            if r.status_code != 200:
                return None
            root = ET.fromstring(r.content)
            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
            page = []
            for p in root.iter(ns + "Prefix"):
                txt = p.text or ""
                if txt.startswith(prefix) and txt != prefix and txt.endswith("/"):
                    page.append(txt)
            names += [x[len(prefix):-1] for x in page]
            trunc = root.find(ns + "IsTruncated")
            if trunc is not None and (trunc.text or "").lower() == "true":
                nm = root.find(ns + "NextMarker")
                marker = nm.text if nm is not None and nm.text else (page[-1] if page else "")
                if not marker:
                    break
            else:
                break
    except Exception:
        return None
    return sorted(set(names)) or None


def rank_symbols(symbols):
    jobs = [(s, m) for s in symbols for m in SAMPLE_MONTHS]

    def one(job):
        s, m = job
        c = fetch(f"{BASE}/monthly/klines/{s}/1d/{s}-1d-{m}.zip")
        if not c:
            return s, 0.0
        try:
            rows = parse_klines(read_zip_rows(c))
        except Exception:
            return s, 0.0
        return s, float(np.mean([r[5] for r in rows])) if rows else 0.0

    best = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        for s, v in ex.map(one, jobs):
            best[s] = max(best.get(s, 0.0), v)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    return [s for s, v in ranked[:TOP_N] if v > 0], ranked


def job(args):
    kind, sym, key, url = args
    c = fetch(url)
    if c is None:
        return kind, sym, key, None
    try:
        rows = read_zip_rows(c)
        parser = {"kl": parse_klines, "fu": parse_funding, "oi": parse_metrics}[kind]
        return kind, sym, key, parser(rows)
    except Exception:
        return kind, sym, key, None


def run_jobs(tasks, label):
    res, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for out in ex.map(job, tasks):
            res.append(out)
            done += 1
            if done % 10000 == 0:
                print(f"  {label}: {done}/{len(tasks)} files ({time.time() - t0:.0f}s)", flush=True)
    return res


# ---------------------------------------------------------------- data frame and signals
def build_frame(kl, fund, oi):
    df = pd.DataFrame(kl, columns=["time", "open", "high", "low", "close", "qv"])
    df = df.drop_duplicates("time").sort_values("time")
    df = df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.qv >= 0)].reset_index(drop=True)
    fd = {}
    for t, r in fund:
        d = int(t // DAY * DAY)
        fd[d] = fd.get(d, 0.0) + r
    od = {d: v for d, v in oi if v > 0}
    df["fund_day"] = df["time"].map(fd)
    df["oi_value"] = df["time"].map(od)
    return df


def indicators(df):
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out = {
        "atr": tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean(),
        "vol30": df["qv"].rolling(30, min_periods=30).mean(),
        "f3": df["fund_day"].rolling(3, min_periods=3).mean(),
        "ret5": c / c.shift(5) - 1,
        "oi5": df["oi_value"] / df["oi_value"].shift(5) - 1,
    }
    return {k: v.to_numpy(float) for k, v in out.items()}


def sim(o, h, l, c, k, side, entry, stop, target, hold):
    n = len(c)
    risk = side * (entry - stop)
    if risk <= 0:
        return None
    end_bar = k + hold - 1
    for m in range(k, min(n - 1, end_bar) + 1):
        if side == 1:
            if m > k and o[m] <= stop:
                return (o[m] - entry) / risk, m, "SL"
            if l[m] <= stop:
                return -1.0, m, "SL"
            if h[m] >= target:
                return (target - entry) / risk, m, "TP"
        else:
            if m > k and o[m] >= stop:
                return (entry - o[m]) / risk, m, "SL"
            if h[m] >= stop:
                return -1.0, m, "SL"
            if l[m] <= target:
                return (entry - target) / risk, m, "TP"
        if m == end_bar:
            return side * (c[m] - entry) / risk, m, "TIME"
    return None


def funding_sum(ft, fr, t_entry, t_exit_day):
    """sum of funding rates settled after the entry open up to the end of the exit day."""
    lo = int(np.searchsorted(ft, t_entry, side="right"))
    hi = int(np.searchsorted(ft, t_exit_day + DAY, side="right"))
    return float(fr[lo:hi].sum())


def make_candidates(sym, df, ft, fr, hyp):
    cfg = HYP[hyp]
    n = len(df)
    if n < 60:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy(np.int64)
    ind = indicators(df)
    out = []
    for i in range(35, n - 1):
        a, v30, f3 = ind["atr"][i], ind["vol30"][i], ind["f3"][i]
        if not (np.isfinite(a) and np.isfinite(v30)) or v30 < MIN_VOL30:
            continue
        if t[i] - t[i - 30] != 30 * DAY or t[i + 1] - t[i] != DAY:
            continue
        if cfg["fth"] is not None and not np.isfinite(f3):
            continue
        for side in cfg["sides"]:
            if side == -1:
                if not (c[i] < o[i] and c[i] < l[i - 1]):
                    continue
                if cfg["fth"] is not None and f3 < cfg["fth"]:
                    continue
            else:
                if not (c[i] > o[i] and c[i] > h[i - 1]):
                    continue
                if cfg["fth"] is not None and f3 > -cfg["fth"]:
                    continue
            if cfg["ret5"] is not None and not (np.isfinite(ind["ret5"][i]) and ind["ret5"][i] >= cfg["ret5"]):
                continue
            if cfg["oi5"] is not None and not (np.isfinite(ind["oi5"][i]) and ind["oi5"][i] >= cfg["oi5"]):
                continue
            k = i + 1
            entry = o[k]
            if cfg["stop"] == "atr":
                stop, lo_r, hi_r = entry - side * 2.0 * a, 0.02, 0.15
            else:
                stop, lo_r, hi_r = h[i - 3:i + 1].max() * 1.01, 0.05, 0.35
            risk = side * (entry - stop)
            risk_pct = risk / entry if entry > 0 else 0.0
            if risk <= 0 or not (lo_r <= risk_pct <= hi_r):
                continue
            target = entry + side * RR * risk
            if target <= 0:
                continue
            res = sim(o, h, l, c, k, side, entry, stop, target, cfg["hold"])
            if res is None:
                continue
            fsum = funding_sum(ft, fr, t[k], t[res[1]])
            r_net = res[0] - (COST_RT + side * fsum) / risk_pct
            r_stress = res[0] - (COST_STRESS + side * fsum) / risk_pct
            if cfg["ret5"] is not None:
                strength = float(ind["ret5"][i])
            elif cfg["fth"] is not None:
                strength = float(abs(f3))
            else:
                strength = float(abs(c[i] - o[i]) / a)
            out.append({"pair": sym, "side": "LONG" if side == 1 else "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[1]]),
                        "reason": res[2], "days": int(res[1] - k + 1), "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                        "r_net": float(r_net), "r_stress": float(r_stress), "strength": strength, "funding_sum": fsum, "hyp": hyp})
    return out


# ---------------------------------------------------------------- main run
def main():
    t_run = time.time()
    months = month_list()
    now = datetime.now(timezone.utc)
    end_t = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp())
    start_t = int(datetime(START_Y, START_M, 1, tzinfo=timezone.utc).timestamp())
    listed = list_symbols()
    source = "download-site listing" if listed else "built-in fallback list (listing not reachable)"
    symbols = [s for s in (listed or FALLBACK) if usable_symbol(s)]
    print(f"symbols from {source}: {len(symbols)}", flush=True)
    selected, ranked = rank_symbols(symbols)
    print(f"selected {len(selected)} symbols: {', '.join(selected)}", flush=True)
    L = ["FUTURES POSITIONING BACKTEST - Binance USDT-M perpetuals, daily, real open interest and real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source}",
         "costs: fee+slippage 0.20% round trip (stress 0.40%) plus REAL funding for every settlement while open; max 8 open positions",
         "Pass rule fixed before the run (A and B each): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Known problems: top-70 liquid contracts only, entries assume a fill at the day's open, daily bars hide intrabar order (SL is assumed first),",
         "funding of the whole exit day is counted, news is not in the data, two hypotheses were tested so a pass is only a candidate."]
    if len(selected) < 20:
        L.append("ABORTED: too few symbols could be downloaded; the download site may be blocked or changed.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_fut_result.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return
    tasks = []
    for s in selected:
        for m in months:
            tasks.append(("kl", s, m, f"{BASE}/monthly/klines/{s}/1d/{s}-1d-{m}.zip"))
            tasks.append(("fu", s, m, f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"))
    res = run_jobs(tasks, "klines+funding")
    kl, fu, oi = {s: [] for s in selected}, {s: [] for s in selected}, {s: [] for s in selected}
    cov = {"kl": [0, 0], "fu": [0, 0], "oi": [0, 0]}
    for kind, s, key, data in res:
        cov[kind][1] += 1
        if data:
            cov[kind][0] += 1
            (kl if kind == "kl" else fu)[s].extend(data)
    tasks = []
    for s in selected:
        if not kl[s]:
            continue
        first = max(min(r[0] for r in kl[s]), start_t)
        for d in range(first, end_t, DAY):
            ds = datetime.fromtimestamp(d, timezone.utc).strftime("%Y-%m-%d")
            tasks.append(("oi", s, d, f"{BASE}/daily/metrics/{s}/{s}-metrics-{ds}.zip"))
    res = run_jobs(tasks, "open-interest days")
    for kind, s, key, data in res:
        cov["oi"][1] += 1
        if data:
            cov["oi"][0] += 1
            oi[s].extend(data)
    L.append(f"files found/requested: klines {cov['kl'][0]}/{cov['kl'][1]} | funding {cov['fu'][0]}/{cov['fu'][1]} | open-interest days {cov['oi'][0]}/{cov['oi'][1]}")
    L.append("")
    cands = {h: [] for h in HYP}
    used = 0
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = build_frame(kl[s], fu[s], oi[s])
        df = df[(df["time"] >= start_t) & (df["time"] < end_t)].reset_index(drop=True)
        if len(df) < 120:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        for h in HYP:
            cands[h] += make_candidates(s, df, ft, fr, h)
    L.append(f"symbols with usable data: {used}")
    sets = [("HYPOTHESIS A - crowded funding reversal, long and short (eligible)", "A", True),
            ("CONTROL A0 - same price trigger without the funding condition (information only)", "A0", False),
            ("HYPOTHESIS B - leveraged pump fade: price + open interest + funding, short (eligible)", "B", True),
            ("CONTROL B0 - same price trigger without open interest and funding (information only)", "B0", False)]
    taken_all = []
    for title, h, eligible in sets:
        cap = B.portfolio(cands[h])
        L.append(f"signals before the position cap: {len(cands[h])} | taken: {len(cap)}")
        L += B.block(title, cap, eligible)
        if eligible:
            taken_all += cap
        if cap:
            fs = np.array([x["funding_sum"] * (1 if x["side"] == "LONG" else -1) for x in cap])
            L.append(f"  average real funding cost per trade: {100 * fs.mean():+.3f}% of position (negative = funding was income)")
            L.append("")
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_fut_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_fut_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["hyp", "pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "funding_sum", "strength"]
        w.writerow(cols)
        for x in taken_all:
            w.writerow([x[k] for k in cols])


# ---------------------------------------------------------------- self-tests (offline)
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def zipped(name, text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, text)
    return b.getvalue()


def synth_frame(n=200, seed=1, qv=5e7, fund=0.0, t0=1_650_000_000 // DAY * DAY):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    df = pd.DataFrame({"time": t0 + np.arange(n) * DAY, "open": o, "high": h, "low": l, "close": c, "qv": qv})
    df["fund_day"] = fund
    df["oi_value"] = 1e8
    return df


def crafted_short(after_low=50.0):
    """flat 100 bars, then day i=60: red, closes below the previous low, funding 0.12%/day on days 58-60; next bar drops."""
    n = 80
    t0 = 1_650_000_000 // DAY * DAY
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 102.5)
    l = np.full(n, 97.5)
    o[60], c[60], h[60], l[60] = 100.0, 96.0, 100.5, 95.5
    o[61], h[61], l[61], c[61] = 96.0, 96.5, after_low, after_low + 1
    for j in range(62, n):
        o[j], c[j], h[j], l[j] = after_low + 1, after_low + 1, after_low + 2, after_low
    df = pd.DataFrame({"time": t0 + np.arange(n) * DAY, "open": o, "high": h, "low": l, "close": c, "qv": 5e7})
    df["fund_day"] = 0.0
    df.loc[58:60, "fund_day"] = 0.0012
    df["oi_value"] = 1e8
    return df


def run_selftest():
    # parsers
    rows = [["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"],
            ["1667260800000", "1", "2", "0.5", "1.5", "10", "1667347199999", "15.5"], ["bad"]]
    check(parse_klines(rows) == [(1667260800, 1.0, 2.0, 0.5, 1.5, 15.5)], "klines parse, header and junk skipped")
    check(parse_klines([["1667260800000", "1", "2", "0.5", "1.5", "10", "x", "7"]])[0][0] == 1667260800, "klines without header")
    check(parse_funding([["calc_time", "funding_interval_hours", "last_funding_rate"], ["1667260800000", "8", "0.0001"]]) == [(1667260800.0, 0.0001)], "funding parse")
    m = parse_metrics([["create_time", "symbol", "a", "b"], ["2022-11-01 00:00:00", "X", "1", "100"], ["2022-11-01 23:55:00", "X", "1", "130"],
                       ["2022-11-02 00:05:00", "X", "1", "140"]])
    check(sorted(m) == [(1667260800, 130.0), (1667347200, 140.0)], f"metrics keep the last value of each day {m}")
    check(len(read_zip_rows(zipped("a.csv", "1,2\n3,4\n"))) == 2, "zip read")
    # symbols and months
    check(usable_symbol("BTCUSDT") and usable_symbol("1000PEPEUSDT"), "good symbols")
    check(not usable_symbol("BTCUSDT_230630") and not usable_symbol("USDCUSDT") and not usable_symbol("BTCDOMUSDT") and not usable_symbol("ETHUPUSDT"), "bad symbols dropped")
    mons = month_list(datetime(2023, 2, 15, tzinfo=timezone.utc))
    check(mons == ["2022-11", "2022-12", "2023-01"], f"months {mons}")
    # listing with pagination
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    pre = "data/futures/um/monthly/klines/"

    def page(names, trunc, nxt=""):
        cp = "".join(f"<CommonPrefixes><Prefix>{pre}{n}/</Prefix></CommonPrefixes>" for n in names)
        return (f'<ListBucketResult xmlns="{ns}"><Prefix>{pre}</Prefix><IsTruncated>{"true" if trunc else "false"}</IsTruncated>'
                f'{"<NextMarker>" + nxt + "</NextMarker>" if nxt else ""}{cp}</ListBucketResult>').encode()

    class Resp:
        def __init__(self, code, content=b"", text=""):
            self.status_code, self.content, self.text = code, content, text

    orig_get, orig_sleep = SESSION.get, time.sleep
    try:
        def fake_list(url, params=None, timeout=None):
            return Resp(200, page(["AAAUSDT", "BBBUSDT"], True, pre + "BBBUSDT/")) if not params.get("marker") else Resp(200, page(["CCCUSDT"], False))
        SESSION.get = fake_list
        check(list_symbols() == ["AAAUSDT", "BBBUSDT", "CCCUSDT"], "listing follows the next-page marker")
        SESSION.get = lambda url, params=None, timeout=None: Resp(403)
        check(list_symbols() is None, "blocked listing returns None")
        # fetch: retry then success, 404 -> None
        calls = {"n": 0}

        def flaky(url, params=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.ConnectionError("boom")
            return Resp(200, b"data")
        time.sleep = lambda x: None
        SESSION.get = flaky
        check(fetch("u") == b"data" and calls["n"] == 2, "fetch retries after an error")
        SESSION.get = lambda url, params=None, timeout=None: Resp(404)
        check(fetch("u") is None, "404 -> None")
    finally:
        SESSION.get, time.sleep = orig_get, orig_sleep
    # frame building
    df = build_frame([(1667260800, 1, 2, 0.5, 1.5, 10.0), (1667347200, 1, 2, 0.5, 1.6, 11.0)],
                     [(1667260800.0, 0.0001), (1667289600.0, 0.0002), (1667318400.0, 0.0003)], [(1667260800, 100.0), (1667347200, 0.0)])
    check(abs(df["fund_day"].iloc[0] - 0.0006) < 1e-12 and np.isnan(df["fund_day"].iloc[1]), "daily funding is the sum of settlements")
    check(df["oi_value"].iloc[0] == 100.0 and np.isnan(df["oi_value"].iloc[1]), "zero open interest is treated as missing")
    # simulation
    o = np.array([100.0, 100, 100]); h = np.array([101.0, 125, 101]); l = np.array([99.0, 99, 99]); c = np.array([100.0, 100, 100])
    check(sim(o, h, l, c, 0, 1, 100.0, 90.0, 120.0, 5) == (2.0, 1, "TP"), "long TP")
    o = np.array([100.0, 80, 80]); h = np.array([101.0, 85, 85]); l = np.array([99.0, 78, 78]); c = np.array([100.0, 80, 80])
    r = sim(o, h, l, c, 0, 1, 100.0, 90.0, 120.0, 5)
    check(r[2] == "SL" and abs(r[0] + 2.0) < 1e-9, f"long SL gap fills at the open {r}")
    o = np.array([100.0, 100]); h = np.array([140.0, 101]); l = np.array([80.0, 99]); c = np.array([100.0, 100])
    check(sim(o, h, l, c, 0, 1, 100.0, 90.0, 120.0, 5) == (-1.0, 0, "SL"), "SL and TP in one bar -> SL")
    o = np.array([100.0, 100, 100]); h = np.array([100.5, 100.5, 100.5]); l = np.array([99.5, 99.5, 99.5]); c = np.array([100.0, 100, 98.0])
    check(sim(o, h, l, c, 0, -1, 100.0, 110.0, 80.0, 3) == (0.2, 2, "TIME"), "short time stop at the close of the last day")
    check(sim(o[:2], h[:2], l[:2], c[:2], 0, -1, 100.0, 110.0, 80.0, 5) is None, "unresolved trades are dropped")
    # funding sum boundaries
    ft, fr = np.array([100.0, 200.0, 300.0, DAY + 100.0, 3 * DAY]), np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    check(funding_sum(ft, fr, 100.0, 0) == 6.0, "settlement at the entry time is excluded, whole exit day included")
    check(funding_sum(ft, fr, 100.0, DAY) == 14.0, "second day included")
    # signals on a crafted series
    df = crafted_short()
    ft, fr = np.array([float(x) for x in df["time"] + 3600]), np.full(len(df), 0.0004)
    got = make_candidates("X", df, ft, fr, "A")
    check(len(got) == 1, f"one A short signal, got {len(got)}")
    x = got[0]
    atr_i = indicators(df)["atr"][60]
    entry = df["open"].iloc[61]
    check(x["side"] == "SHORT" and x["entry_t"] == int(df["time"].iloc[61]), "entry is the next open")
    check(abs(x["risk_pct"] - 2 * atr_i / entry) < 1e-9, "stop is 2 ATR")
    check(x["reason"] == "TP" and abs(x["r_gross"] - 2.0) < 1e-9, f"TP reached {x['reason']} {x['r_gross']}")
    check(x["r_net"] < x["r_gross"] and x["r_stress"] < x["r_net"], "costs reduce R")
    expected_fund = funding_sum(ft, fr, int(df["time"].iloc[61]), int(df["time"].iloc[61]))
    check(abs(x["funding_sum"] - expected_fund) < 1e-12 and expected_fund > 0, "real funding is charged")
    low_f = crafted_short().assign(fund_day=0.0003)
    check(make_candidates("X", low_f, ft, fr, "A") == [] and len(make_candidates("X", low_f, ft, fr, "A0")) == 1, "funding condition separates A from A0")
    illiquid = crafted_short().assign(qv=1e6)
    check(make_candidates("X", illiquid, ft, fr, "A0") == [], "30-day volume filter")
    gap = crafted_short()
    gap.loc[40:, "time"] += DAY
    check(make_candidates("X", gap, ft, fr, "A0") == [], "a gap inside the 30-day window blocks the signal")
    # hypothesis B needs price, open interest and funding
    n = 80
    t0 = 1_650_000_000 // DAY * DAY
    o = np.full(n, 100.0); c = np.full(n, 100.0); h = np.full(n, 101.0); l = np.full(n, 99.0)
    for j, p in zip(range(55, 60), (103, 108, 118, 125, 130)):
        o[j], c[j], h[j], l[j] = p - 1, p, p + 1, p - 2
    o[60], c[60], h[60], l[60] = 130.0, 120.0, 131.0, 119.0
    o[61:], c[61:], h[61:], l[61:] = 120.0, 100.0, 121.0, 99.0
    dfb = pd.DataFrame({"time": t0 + np.arange(n) * DAY, "open": o, "high": h, "low": l, "close": c, "qv": 5e7, "fund_day": 0.0006})
    dfb["oi_value"] = np.where(np.arange(n) >= 55, 1.3e8, 1e8)
    dfb["oi_value"] = np.where(np.arange(n) >= 59, 1.5e8, dfb["oi_value"])
    ftb, frb = np.array([float(x) for x in dfb["time"] + 3600]), np.full(n, 0.0002)
    gb = make_candidates("X", dfb, ftb, frb, "B")
    check(len(gb) == 1 and gb[0]["side"] == "SHORT", f"B signal {len(gb)}")
    no_oi = dfb.assign(oi_value=1e8)
    check(make_candidates("X", no_oi, ftb, frb, "B") == [] and len(make_candidates("X", no_oi, ftb, frb, "B0")) == 1, "open-interest condition separates B from B0")
    # no look-ahead: signals found on a prefix equal those on the full series; future data cannot change them
    rng = np.random.default_rng(5)
    big = synth_frame(600, seed=9, qv=5e7)
    big["fund_day"] = np.where(rng.random(600) < 0.25, 0.0015, 0.0001)
    ftl, frl = np.array([float(x) for x in big["time"] + 3600]), np.where(big["fund_day"] > 0.001, 0.0005, 0.0001)
    full = make_candidates("R", big, ftl, frl, "A")
    check(len(full) > 5, f"random series produce signals {len(full)}")
    cut = 400
    part = make_candidates("R", big.iloc[:cut].reset_index(drop=True), ftl, frl, "A")
    fm = {(x["entry_t"], x["side"]): x for x in full}
    for x in part:
        y = fm.get((x["entry_t"], x["side"]))
        check(y is not None and abs(y["risk_pct"] - x["risk_pct"]) < 1e-12 and abs(y["strength"] - x["strength"]) < 1e-12, "prefix signal exists in the full run")
    future = big.copy()
    future.loc[cut:, ["open", "high", "low", "close"]] *= 3.0
    part2 = make_candidates("R", future.iloc[:cut].reset_index(drop=True), ftl, frl, "A")
    check([(x["entry_t"], x["side"]) for x in part] == [(x["entry_t"], x["side"]) for x in part2], "future prices do not change earlier signals")
    # portfolio integration
    cap = B.portfolio(full)
    check(0 < len(cap) <= len(full), "portfolio keeps a subset")
    text = "\n".join(B.block("test", cap, True))
    check("VERDICT" in text and "bootstrap" in text, "report block renders")
    print(f"ALL FUTURES SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
