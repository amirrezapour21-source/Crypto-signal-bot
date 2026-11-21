"""
LIMIT-ENTRY SCREEN for the live BRT-S SHORT rule - Binance USDT-M perpetuals, 4-hour bars, real funding. SENSITIVITY ONLY: no pass/fail rule, nothing is selected.
Question: the live bot enters at the next bar open (effectively a market order), so there is no time to review the signal. What does it cost to enter with a
sell limit instead? Setup, stop and 2R target logic are the unchanged live SHORT rule. Self-contained (helpers of lvar_daily/fut/box_backtest.py embedded unchanged).
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---- embedded helper modules (unchanged code extracted from lvar_daily_backtest.py, lvar_fut_backtest.py, lvar_box_backtest.py)
import sys
import types

SRC_LVAR_DAILY_BACKTEST = r'''
import csv
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

MAX_POS = 8

RISK_PER_TRADE = 0.005

BOOT = 2000

DAY = 86400

def portfolio(cands):
    ordered = sorted(cands, key=lambda x: (x["entry_t"], -x["strength"], x["pair"]))
    open_pos, taken = [], []
    for x in ordered:
        open_pos = [q for q in open_pos if q[0] >= x["entry_t"]]
        if len(open_pos) >= MAX_POS or any(q[1] == x["pair"] for q in open_pos):
            continue
        taken.append(x)
        open_pos.append((x["exit_t"], x["pair"]))
    return taken

def timeline(rows):
    t0 = min(x["entry_t"] for x in rows)
    t1 = max(x["exit_t"] for x in rows)
    cnt = np.zeros((t1 - t0) // DAY + 1, int)
    for x in rows:
        cnt[(x["entry_t"] - t0) // DAY:(x["exit_t"] - t0) // DAY + 1] += 1
    entry_days = len({x["entry_t"] for x in rows})
    return cnt, entry_days

def stats(rs):
    a = np.asarray(rs, float)
    if len(a) == 0:
        return None
    wins, losses = a[a > 0].sum(), -a[a < 0].sum()
    cum = np.cumsum(a)
    return {"n": len(a), "win%": 100 * float((a > 0).mean()), "avgR": float(a.mean()), "totalR": float(a.sum()),
            "PF": float(wins / losses) if losses > 0 else float("inf"), "maxDD": float((np.maximum.accumulate(cum) - cum).max())}

def fmt(label, rs):
    s = stats(rs)
    if s is None:
        return f"{label:38s} n=0"
    return f"{label:38s} n={s['n']:5d} win%={s['win%']:5.1f} avgR={s['avgR']:+.3f} totalR={s['totalR']:+8.1f} PF={s['PF']:.2f} maxDD={s['maxDD']:.1f}"

def month_of(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")

def year_of(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y")

def week_bootstrap(rows):
    wk = {}
    for x in rows:
        wk.setdefault(x["entry_t"] // (7 * DAY), []).append(x["r_net"])
    sums = np.array([sum(v) for v in wk.values()])
    cnts = np.array([len(v) for v in wk.values()])
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(sums), size=(BOOT, len(sums)))
    avg = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(np.percentile(avg, 2.5)), float(np.percentile(avg, 97.5)), float((avg > 0).mean()), len(sums)

def block(title, rows, eligible):
    L = ["=" * 60, title]
    if not rows:
        return L + ["NO TRADES", ""]
    rows = sorted(rows, key=lambda x: x["entry_t"])
    mid_t = rows[len(rows) // 2]["entry_t"]
    for k in ("r_gross", "r_net", "r_stress"):
        L.append(fmt(f"  {k}", [x[k] for x in rows]))
    a = [x["r_net"] for x in rows if x["entry_t"] < mid_t]
    b = [x["r_net"] for x in rows if x["entry_t"] >= mid_t]
    L += [fmt("  first half (net)", a), fmt("  second half (net)", b)]
    for side in ("LONG", "SHORT"):
        L.append(fmt(f"  {side} (net)", [x["r_net"] for x in rows if x["side"] == side]))
    for y in sorted({year_of(x["entry_t"]) for x in rows}):
        L.append(fmt(f"  year {y} (net)", [x["r_net"] for x in rows if year_of(x["entry_t"]) == y]))
    cnt, entry_days = timeline(rows)
    span = len(cnt)
    d = np.array([x["days"] for x in rows])
    L.append(f"  days open per trade: mean {d.mean():.1f} | median {int(np.median(d))} | 90th pct {int(np.percentile(d, 90))} | max {int(d.max())}")
    L.append(f"  trades per week: {len(rows) / (span / 7):.2f} | days with a new entry: {100 * entry_days / span:.0f}%")
    L.append(f"  open positions: mean {cnt.mean():.1f} | max {int(cnt.max())} | days with none open {100 * float((cnt == 0).mean()):.0f}% | days at cap {100 * float((cnt >= MAX_POS).mean()):.0f}%")
    exits = {}
    for x in rows:
        exits[x["reason"]] = exits.get(x["reason"], 0) + 1
    L.append("  exits: " + ", ".join(f"{k}={v}" for k, v in sorted(exits.items())))
    L.append(f"  median risk {100 * float(np.median([x['risk_pct'] for x in rows])):.1f}% of price")
    byx = sorted(rows, key=lambda x: x["exit_t"])
    cum, peak, peak_t, under = 0.0, 0.0, byx[0]["exit_t"], 0
    for x in byx:
        cum += x["r_net"]
        if cum >= peak:
            peak, peak_t = cum, x["exit_t"]
        under = max(under, (x["exit_t"] - peak_t) // DAY)
    L.append(f"  longest time below equity peak (net): {int(under)} days")
    mon = {}
    for x in byx:
        mon[month_of(x["exit_t"])] = mon.get(month_of(x["exit_t"]), 0.0) + x["r_net"]
    pos_m = sum(1 for v in mon.values() if v > 0)
    L.append(f"  profitable months (net, by exit): {pos_m} of {len(mon)} | worst month {min(mon.values()):+.1f}R | best month {max(mon.values()):+.1f}R")
    sn = stats([x["r_net"] for x in rows])
    L.append(f"  at {100 * RISK_PER_TRADE:.1f}% risk per trade (not compounded): total {sn['totalR'] * 100 * RISK_PER_TRADE:+.0f}% | max drawdown {sn['maxDD'] * 100 * RISK_PER_TRADE:.0f}%")
    byc = {}
    for x in rows:
        byc[x["pair"]] = byc.get(x["pair"], 0.0) + x["r_net"]
    top = sorted(byc.items(), key=lambda kv: -kv[1])
    top3 = {k for k, _ in top[:3]}
    rest = [x["r_net"] for x in rows if x["pair"] not in top3]
    L.append("  top 3 coins by net R: " + ", ".join(f"{k} {v:+.0f}" for k, v in top[:3]) + f" | total {sum(byc.values()):+.0f}")
    L.append(f"  avgR net without those 3 coins: {np.mean(rest):+.3f} (n={len(rest)})" if rest else "  without top 3 coins: no trades")
    wk = {}
    for x in rows:
        wk[x["entry_t"] // (7 * DAY)] = wk.get(x["entry_t"] // (7 * DAY), 0.0) + x["r_net"]
    wins = sorted((v for v in wk.values() if v > 0), reverse=True)
    if wins:
        L.append(f"  share of positive-week profit from the best 5 weeks: {100 * sum(wins[:5]) / sum(wins):.0f}%")
    lo, hi, pp, nw = week_bootstrap(rows)
    L.append(f"  weekly-block bootstrap of net avgR: 95% CI [{lo:+.3f}, {hi:+.3f}], P(avgR>0)={pp:.2f} ({nw} weeks)")
    sa, sb = stats(a), stats(b)
    ok = bool(sn and sa and sb and sn["avgR"] >= 0.10 and sn["PF"] >= 1.2 and sn["n"] >= 300 and sa["avgR"] > 0 and sb["avgR"] > 0)
    if eligible:
        L.append(f"  VERDICT (pre-registered rule): {'PASS - candidate for forward paper test' if ok else 'FAIL - rejected'}")
        if sn["n"] < 300:
            L.append("  note: fewer than 300 trades = insufficient sample")
    else:
        L.append("  (sensitivity only - not eligible for selection)")
    L.append("")
    return L
'''

SRC_LVAR_FUT_BACKTEST = r'''
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

START_Y, START_M = 2022, 11

SAMPLE_MONTHS = ["2022-12", "2023-09", "2024-06", "2025-03", "2026-01", "2026-09"]

TOP_N = 70

WORKERS = 24

EXCLUDE_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "EUR", "BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD"}

FALLBACK = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
            "TRXUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
            "1000PEPEUSDT", "1000SHIBUSDT", "WIFUSDT", "FETUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "SEIUSDT", "TIAUSDT", "ORDIUSDT",
            "1000BONKUSDT", "1000FLOKIUSDT", "JUPUSDT", "WLDUSDT", "ENAUSDT", "TAOUSDT", "DYDXUSDT", "GALAUSDT", "SANDUSDT", "MANAUSDT",
            "AXSUSDT", "CRVUSDT", "SNXUSDT", "RUNEUSDT", "ICPUSDT", "BCHUSDT", "XLMUSDT", "HBARUSDT", "VETUSDT", "ALGOUSDT", "EOSUSDT",
            "THETAUSDT", "STXUSDT", "IMXUSDT", "PENDLEUSDT"]

SESSION = requests.Session()

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
'''

SRC_LVAR_BOX_BACKTEST = r'''
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lvar_daily_backtest as B
import lvar_fut_backtest as F

DAY = B.DAY

TF = 4 * 3600

BPD = 6

ATR_N = 14

VOL_BARS = 180

MIN_VOL30 = 20_000_000

COST_RT = 0.0020

COST_STRESS = 0.0040

NON_CRYPTO = {"XAU", "XAG", "PAXG", "XAUT", "SOXL", "SOXS", "SPCX", "SKHYNIX", "SKHY", "SNDK", "MU", "KORU", "CL", "BZ", "NATGAS", "COPPER",
              "TSLA", "NVDA", "AAPL", "MSTR", "COIN", "HOOD", "CRCL", "QQQ", "SPY", "TSM", "INTC", "AMD", "SNXX", "GOOGL", "AMZN", "META",
              "MSFT", "PLTR", "EWY", "EWJ", "USO", "GLD", "SLV"}

def parse_klines4(rows):
    """(bar_open_time, open, high, low, close, quote_volume) from a 4h kline CSV (header optional)."""
    out = []
    for r in rows:
        try:
            out.append((int(F.norm_t(r[0])), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])))
        except (ValueError, IndexError):
            continue
    return out

def job4(args):
    kind, sym, url = args
    c = F.fetch(url)
    if c is None:
        return kind, sym, None
    try:
        rows = F.read_zip_rows(c)
        return kind, sym, (parse_klines4(rows) if kind == "k4" else F.parse_funding(rows))
    except Exception:
        return kind, sym, None

def build_frame4(kl):
    df = pd.DataFrame(kl, columns=["time", "open", "high", "low", "close", "qv"])
    df = df.drop_duplicates("time").sort_values("time")
    return df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.qv >= 0)].reset_index(drop=True)

def non_crypto(sym):
    return sym[:-4] in NON_CRYPTO

def indicators4(df):
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return {"atr": tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean().to_numpy(float),
            "vol": (df["qv"].rolling(VOL_BARS, min_periods=VOL_BARS).mean() * BPD).to_numpy(float)}

def funding_sum4(ft, fr, t_entry, t_exit_bar):
    """sum of funding settled after the entry open up to the end of the exit bar."""
    lo = int(np.searchsorted(ft, t_entry, side="right"))
    hi = int(np.searchsorted(ft, t_exit_bar + TF, side="right"))
    return float(fr[lo:hi].sum())
'''


def _load(name, source):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(source, name, "exec"), mod.__dict__)
    return mod


B = _load("lvar_daily_backtest", SRC_LVAR_DAILY_BACKTEST)
F = _load("lvar_fut_backtest", SRC_LVAR_FUT_BACKTEST)
X = _load("lvar_box_backtest", SRC_LVAR_BOX_BACKTEST)

TF = X.TF
DAY = X.DAY
PIV = 5
LOOKBACK = 180
TOL = 0.25
MIN_PIV = 3
ZONE_BELOW = 0.5
FRESH = 10
BREAK_CLEAN = 0.5
STOP_BUF = 0.25
RR = 2.0
HOLD = 30
RISK_LO, RISK_HI = 0.01, 0.12
MIN_VOL30 = X.MIN_VOL30
COST_MKT = 0.0020
COST_LIM = 0.0010
LIM_BARS = 3
LIM_B_ATR = 0.25
VARIANTS = ("MKT", "LIM_A", "LIM_B")


# ---------------------------------------------------------------- levels (same as lvar_sr_backtest.py)
def find_pivots(h, l):
    n = len(h)
    idx, px = [], []
    for p in range(PIV, n - PIV):
        left_h, right_h = h[p - PIV:p], h[p + 1:p + PIV + 1]
        left_l, right_l = l[p - PIV:p], l[p + 1:p + PIV + 1]
        if h[p] > left_h.max() and h[p] >= right_h.max():
            idx.append(p)
            px.append(h[p])
        if l[p] < left_l.min() and l[p] <= right_l.min():
            idx.append(p)
            px.append(l[p])
    return np.array(idx, int), np.array(px, float)


def levels_at(i, piv_idx, piv_px, a):
    lo = int(np.searchsorted(piv_idx, i - LOOKBACK, side="left"))
    hi = int(np.searchsorted(piv_idx, i - PIV, side="right"))
    if hi - lo < MIN_PIV:
        return []
    pr = piv_px[lo:hi]
    tol = TOL * a
    near = np.abs(pr[:, None] - pr[None, :]) <= tol
    cnt = near.sum(1)
    out = []
    for q in np.nonzero(cnt >= MIN_PIV)[0]:
        lvl, npv = float(np.median(pr[near[q]])), int(cnt[q])
        dup = [j for j, (x, _) in enumerate(out) if abs(x - lvl) <= tol]
        if not dup:
            out.append((lvl, npv))
        elif npv > out[dup[0]][1]:
            out[dup[0]] = (lvl, npv)
    return out


def limit_trade(o, h, l, c, t, i, p, stop, ft, fr, cost):
    """SHORT sell-limit at p placed after the close of signal bar i, valid for LIM_BARS bars (bars i+1 .. i+LIM_BARS).
    Fill when a bar's high reaches p (fill price p, never better). On the fill bar only the stop can hit (stop assumed first, target not evaluated);
    afterwards the usual rules. Returns the trade dict or None when not filled / invalid."""
    n = len(c)
    risk = stop - p
    risk_pct = risk / p if p > 0 else 0.0
    if risk <= 0 or not (RISK_LO <= risk_pct <= RISK_HI):
        return None
    target = p - RR * risk
    if target <= 0:
        return None
    m = None
    for j in range(i + 1, min(n - 1, i + LIM_BARS) + 1):
        if t[j] - t[j - 1] != TF:
            return None
        if h[j] >= p:
            m = j
            break
    if m is None:
        return None
    if h[m] >= stop:
        res = (-1.0, m, "SL")
    else:
        res = F.sim(o, h, l, c, m + 1, -1, p, stop, target, HOLD - 1) if m + 1 < n - 1 else None
    if res is None:
        return None
    fsum = X.funding_sum4(ft, fr, t[m], t[res[1]])
    return {"entry_t": int(t[m]), "exit_t": int(t[res[1]]), "reason": res[2], "days": float((res[1] - m + 1) * TF / DAY),
            "risk_pct": float(risk_pct), "r_gross": float(res[0]), "r_net": float(res[0] - (cost - fsum) / risk_pct),
            "r_stress": float(res[0] - (2 * cost - fsum) / risk_pct), "funding_sum": fsum}


def make_candidates(sym, df, ft, fr):
    """SHORT break-and-retest setups (rule of the live BRT-S bot, unchanged) with three ways to enter: MKT = next open (as now), LIM_A / LIM_B = sell limit."""
    n = len(df)
    if n < LOOKBACK + 10:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy(np.int64)
    ind = X.indicators4(df)
    atr, vol = ind["atr"], ind["vol"]
    piv_idx, piv_px = find_pivots(h, l)
    out = []
    for i in range(LOOKBACK, n - 2):
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(vol[i])) or vol[i] < MIN_VOL30:
            continue
        if not (c[i] < o[i]):
            continue
        if t[i] - t[i - FRESH - 1] != (FRESH + 1) * TF or t[i + 1] - t[i] != TF:
            continue
        best = None
        for lvl, npv in levels_at(i, piv_idx, piv_px, a):
            tol = TOL * a
            prev = c[i - FRESH:i]
            if not (c[i] < lvl and lvl - tol <= h[i] <= lvl + ZONE_BELOW * a):
                continue
            bad = np.nonzero(prev >= lvl)[0]
            if len(bad) == 0:
                continue
            kk = i - FRESH + int(bad[-1])
            if not (kk <= i - 2 and c[kk + 1:i].min() <= lvl - BREAK_CLEAN * a):
                continue
            dist = abs(h[i] - lvl)
            if best is None or dist < best[0]:
                best = (dist, lvl, npv)
        if best is None:
            continue
        _, lvl, npv = best
        stop = max(h[i], lvl) + STOP_BUF * a
        base = {"pair": sym, "side": "SHORT", "sig_t": int(t[i]), "strength": float(npv)}
        k = i + 1
        entry = o[k]
        risk = stop - entry
        risk_pct = risk / entry if entry > 0 else 0.0
        mk = None
        if risk > 0 and RISK_LO <= risk_pct <= RISK_HI and entry - RR * risk > 0:
            res = F.sim(o, h, l, c, k, -1, entry, stop, entry - RR * risk, HOLD)
            if res is not None:
                fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
                mk = {"entry_t": int(t[k]), "exit_t": int(t[res[1]]), "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY),
                      "risk_pct": float(risk_pct), "r_gross": float(res[0]), "r_net": float(res[0] - (COST_MKT - fsum) / risk_pct),
                      "r_stress": float(res[0] - (2 * COST_MKT - fsum) / risk_pct), "funding_sum": fsum}
        la = limit_trade(o, h, l, c, t, i, c[i], stop, ft, fr, COST_LIM)
        lb = limit_trade(o, h, l, c, t, i, c[i] + LIM_B_ATR * a, stop, ft, fr, COST_LIM)
        out.append({"base": base, "MKT": mk, "LIM_A": la, "LIM_B": lb})
    return out


def rows_of(cands, variant):
    return [dict(x["base"], **x[variant]) for x in cands if x[variant] is not None]


def summary_row(name, rows):
    if not rows:
        return f"{name:6s} n=0"
    rows = sorted(rows, key=lambda x: x["entry_t"])
    mid_t = rows[len(rows) // 2]["entry_t"]
    sn = B.stats([x["r_net"] for x in rows])
    sa = B.stats([x["r_net"] for x in rows if x["entry_t"] < mid_t])
    sb = B.stats([x["r_net"] for x in rows if x["entry_t"] >= mid_t])
    return (f"{name:6s} n={sn['n']:5d} avgR={sn['avgR']:+.3f} PF={sn['PF']:.2f} totalR={sn['totalR']:+7.1f} win%={sn['win%']:.1f} "
            f"halves={sa['avgR']:+.3f}/{sb['avgR']:+.3f}")


def build_report(cands):
    L = []
    taken = {v: B.portfolio(rows_of(cands, v)) for v in VARIANTS}
    n_sig = len(cands)
    L.append(f"SHORT setups of the live BRT-S rule: {n_sig} (before position cap)")
    for v in ("LIM_A", "LIM_B"):
        filled = [x for x in cands if x[v] is not None]
        miss = [x for x in cands if x[v] is None and x["MKT"] is not None]
        got = [x["MKT"]["r_net"] for x in cands if x[v] is not None and x["MKT"] is not None]
        L.append(f"{v}: filled {len(filled)} of {n_sig} setups ({100 * len(filled) / max(1, n_sig):.0f}%)")
        L.append(B.fmt(f"  market-entry result of the FILLED setups", got))
        L.append(B.fmt(f"  market-entry result of the NOT-filled setups", [x["MKT"]["r_net"] for x in miss]))
    L.append("")
    L.append("SUMMARY (net R, one position per coin, max 8 open; MKT cost 0.20%, limit cost 0.10% round trip, plus real funding)")
    for v in VARIANTS:
        L.append("  " + summary_row(v, taken[v]))
    L.append("")
    for v in VARIANTS:
        L += B.block(f"{v} - SHORT, entry mode {v} (sensitivity only)", taken[v], False)
    return L, taken


def main():
    t_run = time.time()
    months = F.month_list()
    dl_months = ["2022-10"] + months
    now = datetime.now(timezone.utc)
    end_t = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp())
    start_t = int(datetime(F.START_Y, F.START_M, 1, tzinfo=timezone.utc).timestamp())
    listed = F.list_symbols()
    source = "download-site listing" if listed else "built-in fallback list (listing not reachable)"
    symbols = [s for s in (listed or F.FALLBACK) if F.usable_symbol(s) and not X.non_crypto(s)]
    selected, _ = F.rank_symbols(symbols)
    print(f"selected {len(selected)} symbols", flush=True)
    L = ["LIMIT-ENTRY SCREEN for the live BRT-S SHORT rule - 4h bars, Binance USDT-M perpetuals, real funding",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source} (TradFi names removed)",
         "setup, stop, target (2R from the entry price, stop unchanged), hold 30 bars: identical to the live BRT-S SHORT rule",
         f"MKT = entry at the next open (as now, cost 0.20%). LIM_A = sell limit at the signal-bar close, LIM_B = sell limit at close + {LIM_B_ATR} ATR; both valid for {LIM_BARS} bars (12h), cost 0.10% (maker fee assumed).",
         "Limit fill: when a bar's high reaches the price, fill at that price (never better); on the fill bar only the stop can hit; unfilled = no trade.",
         "Purpose: what does it cost to enter by limit (so there is time to review the signal)? Sensitivity only, no pass/fail rule, nothing is selected from it."]
    if len(selected) < 20:
        L.append("ABORTED: too few symbols could be downloaded.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_limit_screen_result.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return
    tasks = []
    for s in selected:
        for m in dl_months:
            tasks.append(("k4", s, f"{F.BASE}/monthly/klines/{s}/4h/{s}-4h-{m}.zip"))
            tasks.append(("fu", s, f"{F.BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"))
    kl, fu = {s: [] for s in selected}, {s: [] for s in selected}
    cov = {"k4": [0, 0], "fu": [0, 0]}
    done = 0
    with ThreadPoolExecutor(F.WORKERS) as ex:
        for kind, s, data in ex.map(X.job4, tasks):
            cov[kind][1] += 1
            if data:
                cov[kind][0] += 1
                (kl if kind == "k4" else fu)[s].extend(data)
            done += 1
            if done % 1000 == 0:
                print(f"  downloaded {done}/{len(tasks)} files ({time.time() - t_run:.0f}s)", flush=True)
    L.append(f"files found/requested: 4h klines {cov['k4'][0]}/{cov['k4'][1]} | funding {cov['fu'][0]}/{cov['fu'][1]}")
    cands = []
    used = 0
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = X.build_frame4(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) < 600:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        cands += [x for x in make_candidates(s, df, ft, fr) if x["base"]["sig_t"] + TF >= start_t]
    L.append(f"symbols with usable data: {used}")
    L.append("")
    rep, taken = build_report(cands)
    L += rep
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_limit_screen_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------- self-tests (offline)
PASSED = 0
T0 = 1_650_000_000 // DAY * DAY


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def walk(seed, n=9000, vol=0.006, drift=0.0, qv=5e7):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(drift, vol, n)))
    o = np.concatenate([[100.0], c[:-1]])
    return pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": o, "high": np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.003, n))),
                         "low": np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.003, n))), "close": c, "qv": qv})


def cands_of(df):
    ft, fr = np.array([float(x) for x in df["time"] + 3600]), np.full(len(df), 0.0001)
    return make_candidates("R", df, ft, fr)


def run_selftest():
    total = 0
    for seed in (1, 2, 3):
        df = walk(seed)
        cs = cands_of(df)
        total += len(cs)
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        t = df["time"].to_numpy(np.int64)
        tidx = {int(v): j for j, v in enumerate(t)}
        for x in cs:
            i = tidx[x["base"]["sig_t"]]
            for v, price in (("LIM_A", c[i]), ("LIM_B", None)):
                r = x[v]
                if r is None:
                    continue
                check(r["entry_t"] > x["base"]["sig_t"] and r["entry_t"] <= x["base"]["sig_t"] + LIM_BARS * TF, "fill within the validity window, after the signal bar")
                m = tidx[r["entry_t"]]
                p = c[i] if v == "LIM_A" else c[i] + LIM_B_ATR * X.indicators4(df)["atr"][i]
                check(h[m] >= p - 1e-12, "fill bar high reaches the limit price")
                check(all(h[j] < p for j in range(i + 1, m)), "no earlier bar in the window reached the limit price")
                check(RISK_LO <= r["risk_pct"] <= RISK_HI, "risk inside 1%-12%")
                check(r["exit_t"] >= r["entry_t"], "exit not before entry")
                if r["reason"] == "SL" and r["exit_t"] == r["entry_t"]:
                    check(abs(r["r_gross"] + 1.0) < 1e-12, "fill-bar stop is exactly -1R")
            if x["MKT"] is not None:
                check(x["MKT"]["entry_t"] == x["base"]["sig_t"] + TF, "market entry is the next bar")
    check(total >= 40, f"enough synthetic setups ({total})")
    # crafted limit trade: price rallies into the limit, then falls to the target
    n = 60
    tt = T0 + np.arange(n) * TF
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)
    h[11] = 101.0
    l[12:] = 90.0
    h[12:] = 100.5
    c[12:] = 91.0
    ft, fr = np.array([float(tt[0])]), np.array([0.0])
    r = limit_trade(o, h, l, c, tt, 10, 100.8, 103.0, ft, fr, 0.001)
    check(r is not None and r["entry_t"] == int(tt[11]) and r["reason"] == "TP", "crafted: fill on bar 11, target hit later")
    check(abs(r["r_gross"] - 2.0) < 1e-9 and abs(r["risk_pct"] - 2.2 / 100.8) < 1e-12, "crafted: +2R and risk = (stop - limit) / limit")
    check(abs(r["r_net"] - (2.0 - 0.001 / (2.2 / 100.8))) < 1e-9, "crafted: net R uses the limit cost")
    check(limit_trade(o, h, l, c, tt, 10, 101.5, 103.0, ft, fr, 0.001) is None, "crafted: price never reaches the limit, no trade")
    h2 = h.copy()
    h2[11] = 103.5
    r2 = limit_trade(o, h2, l, c, tt, 10, 100.8, 103.0, ft, fr, 0.001)
    check(r2 is not None and r2["reason"] == "SL" and abs(r2["r_gross"] + 1.0) < 1e-12 and r2["exit_t"] == r2["entry_t"], "crafted: stop on the fill bar = -1R")
    allc = []
    for seed in (4, 5, 6):
        allc += [dict(x, base=dict(x["base"], pair=f"P{seed}")) for x in cands_of(walk(seed))]
    rep, taken = build_report(allc)
    text = "\n".join(rep)
    check("SUMMARY" in text and all(f"{v} - SHORT" in text for v in VARIANTS), "report renders")
    print(f"ALL LIMIT-SCREEN SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
