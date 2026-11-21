"""
MOMENTUM BREAKOUT SCREEN - Binance USDT-M perpetuals, 4-hour bars, real funding, LONG and SHORT. Rule and thresholds fixed IN ADVANCE, nothing tuned.
A SCREEN to reject a bad idea quickly, NOT proof of an edge; the forward paper test stays the final judge.
Self-contained (helpers of lvar_daily/fut/box_backtest.py embedded unchanged).

Rule: signal bar i closes beyond the extreme of the previous 30 bars (LONG: above the highest high, SHORT: below the lowest low), candle body at least 0.8 ATR14
in the signal direction. Entry = next bar open, stop = 2.0 ATR14 from entry, target 2R, time stop = close of the 30th bar, risk 1%-12% of price,
costs 0.20% round trip + real funding, 30-day dollar volume at least 20M, one position per coin, at most 8 open positions.
Variants (only two): BRK = no filter; BRK_REG = SHORT only if BTCUSDT 4h close is below its EMA50, LONG only if above (unknown rejected).
Pass rule (fixed before the run): net avgR at least +0.10 AND PF at least 1.2 AND n at least 300 AND both time halves net-positive.
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
N_BRK = 30
BODY = 0.8
STOP_ATR = 2.0
RR = 2.0
HOLD = 30
RISK_LO, RISK_HI = 0.01, 0.12
MIN_VOL30 = X.MIN_VOL30
COST_RT, COST_STRESS = X.COST_RT, X.COST_STRESS
REG_EMA = 50
REG_MIN_BARS = 200
BTC = "BTCUSDT"
VARIANTS = ("BRK", "BRK_REG")


def make_candidates(sym, df, ft, fr):
    """momentum breakout candidates: close beyond the extreme of the previous N_BRK bars with a strong body; entry at the next open."""
    n = len(df)
    if n < N_BRK + 20:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy(np.int64)
    ind = X.indicators4(df)
    atr, vol = ind["atr"], ind["vol"]
    out = []
    for i in range(max(N_BRK, 200), n - 2):
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(vol[i])) or vol[i] < MIN_VOL30:
            continue
        if t[i] - t[i - N_BRK] != N_BRK * TF or t[i + 1] - t[i] != TF:
            continue
        body = abs(c[i] - o[i])
        if body < BODY * a:
            continue
        side = 0
        if c[i] > o[i] and c[i] > h[i - N_BRK:i].max():
            side = 1
        elif c[i] < o[i] and c[i] < l[i - N_BRK:i].min():
            side = -1
        if side == 0:
            continue
        k = i + 1
        entry = o[k]
        risk = STOP_ATR * a
        risk_pct = risk / entry if entry > 0 else 0.0
        if not (RISK_LO <= risk_pct <= RISK_HI):
            continue
        stop = entry - side * risk
        target = entry + side * RR * risk
        if stop <= 0 or target <= 0:
            continue
        res = F.sim(o, h, l, c, k, side, entry, stop, target, HOLD)
        if res is None:
            continue
        fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
        out.append({"pair": sym, "side": "LONG" if side == 1 else "SHORT", "sig_t": int(t[i]), "entry_t": int(t[k]),
                    "exit_t": int(t[res[1]]), "reason": res[2], "days": float((res[1] - k + 1) * TF / DAY),
                    "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                    "r_net": float(res[0] - (COST_RT + side * fsum) / risk_pct),
                    "r_stress": float(res[0] - (COST_STRESS + side * fsum) / risk_pct), "strength": float(body / a),
                    "funding_sum": fsum, "body_atr": float(body / a)})
    return out


def btc_regime(btc_df):
    """{bar_time: (close, ema)}; the EMA of bar t uses closes up to and including bar t only."""
    c = btc_df["close"].astype(float)
    ema = c.ewm(span=REG_EMA, adjust=False).mean()
    t = btc_df["time"].to_numpy(np.int64)
    cv, ev = c.to_numpy(float), ema.to_numpy(float)
    return {int(t[j]): (float(cv[j]), float(ev[j])) for j in range(REG_MIN_BARS, len(t))}


def reg_ok(x, reg):
    v = reg.get(x["sig_t"])
    if v is None:
        return False
    return v[0] < v[1] if x["side"] == "SHORT" else v[0] > v[1]


def keep(variant, x, reg):
    return True if variant == "BRK" else reg_ok(x, reg)


def summary_row(name, rows):
    if not rows:
        return f"{name:8s} n=0"
    rows = sorted(rows, key=lambda x: x["entry_t"])
    mid_t = rows[len(rows) // 2]["entry_t"]
    sn = B.stats([x["r_net"] for x in rows])
    sa = B.stats([x["r_net"] for x in rows if x["entry_t"] < mid_t])
    sb = B.stats([x["r_net"] for x in rows if x["entry_t"] >= mid_t])
    ok = bool(sn and sa and sb and sn["avgR"] >= 0.10 and sn["PF"] >= 1.2 and sn["n"] >= 300 and sa["avgR"] > 0 and sb["avgR"] > 0)

    def side_avg(side):
        r = [x["r_net"] for x in rows if x["side"] == side]
        return B.stats(r)["avgR"] if r else float("nan")

    return (f"{name:8s} n={sn['n']:5d} avgR={sn['avgR']:+.3f} PF={sn['PF']:.2f} halves={sa['avgR']:+.3f}/{sb['avgR']:+.3f} "
            f"LONG={side_avg('LONG'):+.3f} SHORT={side_avg('SHORT'):+.3f} {'PASS' if ok else 'FAIL'}")


def build_report(cands, reg):
    L = []
    taken = {v: B.portfolio([x for x in cands if keep(v, x, reg)]) for v in VARIANTS}
    L.append("SUMMARY (net R after 0.20% costs + real funding, one position per coin, max 8 open)")
    for v in VARIANTS:
        L.append("  " + summary_row(v, taken[v]))
    L.append("")
    L.append(B.fmt("  all candidates (net, before cap)", [x["r_net"] for x in cands]))
    L.append(B.fmt("  rejected by REG (net, diagnostic only)", [x["r_net"] for x in cands if not reg_ok(x, reg)]))
    L.append("")
    for v in VARIANTS:
        L += B.block(f"{v} - momentum breakout, variant {v} (eligible)", taken[v], True)
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
    if BTC not in selected and BTC in symbols:
        selected = list(selected) + [BTC]
    print(f"selected {len(selected)} symbols", flush=True)
    L = ["MOMENTUM BREAKOUT SCREEN - 4h bars, Binance USDT-M perpetuals, real funding, long and short",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} selected out of {len(symbols)} from the {source} (TradFi names removed)",
         f"rule (fixed in advance): close beyond the {N_BRK}-bar extreme, body at least {BODY} ATR, entry next open, stop {STOP_ATR} ATR, target {RR}R, hold {HOLD} bars, risk 1%-12%",
         f"variants: BRK = no filter; BRK_REG = BTC 4h close vs EMA{REG_EMA} (short below, long above)",
         "Pass rule fixed before the run: net avgR at least +0.10 AND PF at least 1.2 AND n at least 300 AND both halves net-positive. Two variants only.",
         "This is a SCREEN, the forward paper test stays the final judge."]
    if not (len(selected) >= 20 and BTC in selected):
        L.append("ABORTED: too few symbols (or BTCUSDT) could be downloaded.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_brk_screen_result.txt", "w", encoding="utf-8") as fh:
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
    frames = {}
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = X.build_frame4(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) >= 600:
            frames[s] = df
    if BTC not in frames:
        L.append("ABORTED: BTCUSDT 4h data missing.")
        text = "\n".join(L) + "\n"
        print(text)
        with open("lvar_brk_screen_result.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return
    reg = btc_regime(frames[BTC])
    cands = []
    used = 0
    for s, df in frames.items():
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        cands += [x for x in make_candidates(s, df, ft, fr) if x["entry_t"] >= start_t]
    L.append(f"symbols with usable data: {used} | BTC regime bars: {len(reg)} | signals before the position cap: {len(cands)}")
    L.append("")
    rep, taken = build_report(cands, reg)
    L += rep
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_brk_screen_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_brk_screen_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "sig_t", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "body_atr", "funding_sum"]
        w.writerow(cols + ["reg_ok"] + [f"taken_{v}" for v in VARIANTS])
        ids = {v: {id(x) for x in taken[v]} for v in VARIANTS}
        for x in sorted(cands, key=lambda q: q["entry_t"]):
            w.writerow([x[k] for k in cols] + [int(reg_ok(x, reg))] + [int(id(x) in ids[v]) for v in VARIANTS])


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


def brute(df):
    """independent slow re-implementation of the signal conditions: returns {signal_bar_index: side} (before stop/risk filters)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = X.indicators4(df)["atr"]
    res = {}
    for i in range(max(N_BRK, 200), len(df) - 2):
        if not np.isfinite(atr[i]) or abs(c[i] - o[i]) < BODY * atr[i]:
            continue
        if c[i] > o[i] and all(c[i] > h[j] for j in range(i - N_BRK, i)):
            res[i] = "LONG"
        elif c[i] < o[i] and all(c[i] < l[j] for j in range(i - N_BRK, i)):
            res[i] = "SHORT"
    return res


def run_selftest():
    total = 0
    for seed in (1, 2, 3):
        df = walk(seed, n=6000)
        cs = cands_of(df)
        total += len(cs)
        tidx = {int(v): j for j, v in enumerate(df["time"].to_numpy(np.int64))}
        ref = brute(df)
        got = {tidx[x["sig_t"]]: x["side"] for x in cs}
        check(all(ref.get(i) == s for i, s in got.items()), f"every candidate satisfies the brute-force signal condition (seed {seed})")
        for x in cs:
            check(x["sig_t"] == x["entry_t"] - TF, "entry is the bar after the signal bar")
            check(abs(x["risk_pct"] - STOP_ATR * X.indicators4(df)["atr"][tidx[x["sig_t"]]] / df["open"].to_numpy(float)[tidx[x["entry_t"]]]) < 1e-9, "risk = 2 ATR of the signal bar over entry price")
            check(RISK_LO <= x["risk_pct"] <= RISK_HI, "risk inside 1%-12%")
        miss = [i for i in ref if i not in got]
        ind = X.indicators4(df)
        opn = df["open"].to_numpy(float)
        n_df = len(df)
        check(all((not (RISK_LO <= STOP_ATR * ind["atr"][i] / opn[i + 1] <= RISK_HI)) or ind["vol"][i] < MIN_VOL30 or not np.isfinite(ind["vol"][i]) or i + HOLD >= n_df - 1 for i in miss), f"every skipped brute-force signal is explained by the risk/volume/end-of-data filters (seed {seed})")
    check(total >= 30, f"enough synthetic signals ({total})")
    # no look-ahead: changing bars after the signal bar + 1 must not change the candidate list up to that point
    df = walk(7, n=4000)
    c1 = cands_of(df)
    df2 = df.copy()
    cut = 3000
    df2.loc[cut + 3:, ["open", "high", "low", "close"]] = df2.loc[cut + 3:, ["open", "high", "low", "close"]].to_numpy() * 0.4
    c2 = cands_of(df2)
    tcut = int(df["time"].iloc[cut])
    check([(x["sig_t"], x["side"], round(x["risk_pct"], 12)) for x in c1 if x["sig_t"] <= tcut] == [(x["sig_t"], x["side"], round(x["risk_pct"], 12)) for x in c2 if x["sig_t"] <= tcut], "signals do not depend on later bars")
    # regime
    b1 = walk(11, n=3000)
    b2 = b1.copy()
    b2.loc[2001:, ["open", "high", "low", "close"]] = b2.loc[2001:, ["open", "high", "low", "close"]].to_numpy() * 0.5
    r1, r2 = btc_regime(b1), btc_regime(b2)
    check(all(r1[int(t)] == r2[int(t)] for t in b1["time"].iloc[REG_MIN_BARS:2001]), "BTC regime of a bar does not depend on later bars")
    check(int(b1["time"].iloc[REG_MIN_BARS - 1]) not in r1 and int(b1["time"].iloc[REG_MIN_BARS]) in r1, "regime unknown before 200 bars")
    reg = {100: (90.0, 100.0), 200: (110.0, 100.0)}
    check(reg_ok({"side": "SHORT", "sig_t": 100}, reg) and not reg_ok({"side": "LONG", "sig_t": 100}, reg), "REG: bear BTC allows SHORT only")
    check(reg_ok({"side": "LONG", "sig_t": 200}, reg) and not reg_ok({"side": "SHORT", "sig_t": 200}, reg), "REG: bull BTC allows LONG only")
    check(not reg_ok({"side": "LONG", "sig_t": 999}, reg), "REG: unknown rejected")
    allc = []
    for seed in (4, 5, 6):
        allc += [dict(x, pair=f"P{seed}") for x in cands_of(walk(seed))]
    breg = btc_regime(walk(12, n=9000))
    rep, taken = build_report(allc, breg)
    text = "\n".join(rep)
    check("SUMMARY" in text and all(f"{v} - momentum breakout" in text for v in VARIANTS), "report renders")
    check(len(taken["BRK_REG"]) <= len(allc), "filtered variant is not larger than the candidate list")
    print(f"ALL BRK-SCREEN SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
