#!/usr/bin/env python3
"""
KuCoin USDT-M Futures - PAPER SIGNAL ENGINE v3 (audited rewrite of v2).
PAPER ONLY: public market data, no API keys, no private endpoints, no orders. Nothing here is an actual fill.

Strategy rules are UNCHANGED from v2:
  4H: EMA50 vs EMA200 trend + ADX14 >= 20.
  15m: EMA20 reclaim (close crosses EMA20 in the trend direction) + directional candle + volume >= SMA20 + RSI filter (LONG < 70, SHORT > 30).
  SL = 1.5 x ATR14(15m), TP = 2R, maximum holding = 30 bars. Same-candle SL/TP: SL wins.

Explicit conventions (v2 left these implicit):
  * A setup is evaluated once per closed 15m candle i, only for the newest closed candle, using 4H candles that closed at or before
    the close of candle i. Older candles are never re-evaluated after downtime.
  * Entry reference = OPEN of candle i+1 (time T). It is captured only if the engine sees that candle within ENTRY_MAX_DELAY_S after T.
    Otherwise the setup is logged MISSED_ENTRY. A later candle's open is never substituted.
  * The entry candle is bar 1 of the trade: SL/TP are checked on it (its moves happen after its open), and the time exit is at the
    close of bar 30. Gap rule: if a later candle OPENS beyond the stop, the exit is at that open (worse than the stop).
  * Only fully closed candles are used for signals and for SL/TP monitoring. Monitoring resumes from last_processed_ts, so downtime
    never skips candles and a candle is never counted twice. A missing candle stalls the position (logged) instead of guessing.
  * P&L: gross (price only) and net (gross - fees - slippage - funding). Fees, slippage and funding are configurable ASSUMPTIONS,
    applied to the entry notional; they are not measured fills. Funding: LONG pays a positive rate, SHORT receives it.
  * State and log use new v3 files; an existing v2 state/log is never read or modified. A corrupt or inconsistent state file stops
    the engine with an error and is left untouched (never silently reset).
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

BASE_URL = "https://api-futures.kucoin.com"
KLINE_URL = BASE_URL + "/api/v1/kline/query"

SYMBOLS = ["XBTUSDTM", "ETHUSDTM", "SOLUSDTM", "BNBUSDTM", "XRPUSDTM", "DOGEUSDTM", "ADAUSDTM", "LINKUSDTM", "AVAXUSDTM", "DOTUSDTM"]

STATE_FILE = Path(os.environ.get("PAPER_STATE_FILE", "futures_paper_state_v3.json"))
LOG_FILE = Path(os.environ.get("PAPER_LOG_FILE", "futures_paper_log_v3.csv"))
STATE_VERSION = 3

# ---- strategy (unchanged) ----
ADX_MIN = 20
SL_ATR = 1.5
TARGET_R = 2.0
MAX_HOLD_BARS = 30
MIN_CANDLES = 220
BAR_S = 900
H4_S = 14400
FETCH_15M = 260
FETCH_4H = 240

# ---- paper execution assumptions (configurable; NOT measured exchange fills) ----
FEE_RATE = 0.0006            # taker fee per side, fraction of notional
SLIPPAGE_RATE = 0.0004       # assumed adverse slippage per side
FUNDING_RATE = 0.0001        # assumed funding per interval (positive: LONG pays, SHORT receives)
FUNDING_INTERVAL_S = 8 * 3600
ENTRY_MAX_DELAY_S = 90       # the entry candle open must be observed within this many seconds of the candle start

# ---- API ----
CHUNK = 150                  # candles per request (the production v4 engine uses the same safe size)
RETRIES = 4
TIMEOUT = 25
MIN_REQ_GAP = 0.2
POLL_SECONDS = 20
POLL_PENDING_SECONDS = 5

LOG_FIELDS = ["event_id", "time_utc", "symbol", "side", "status", "setup_ts", "entry_ts", "entry_ref", "observed_latency_s", "stop",
              "target", "exit_ts", "exit_price", "exit_reason", "bars_held", "gross_pnl_pct", "gross_R", "fee_pct", "slippage_pct",
              "funding_pct", "net_pnl_pct", "net_R", "note", "error"]


class ApiError(Exception):
    pass


class StateError(Exception):
    pass


class LogError(Exception):
    pass


def utc_iso(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat()


def last_closed_ts(now, gran_s=BAR_S):
    """open time of the newest candle that is fully closed at `now`."""
    return int(now // gran_s) * gran_s - gran_s


# ================================================================ API
class KuCoinApi:
    """public kline endpoint with validation of the HTTP / API response, retries, back-off and request spacing."""

    def __init__(self, session=None, sleep=time.sleep, mono=time.monotonic):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "KuCoin-Futures-PaperEngine/3.0"
        self.sleep, self.mono, self._last = sleep, mono, None

    def _throttle(self):
        if self._last is not None:
            wait = MIN_REQ_GAP - (self.mono() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.mono()

    def _get(self, params):
        err = "unknown"
        for attempt in range(1, RETRIES + 1):
            self._throttle()
            delay = min(2 ** attempt, 20) + random.random()
            try:
                r = self.session.get(KLINE_URL, params=params, timeout=TIMEOUT)
                if r.status_code == 200:
                    payload = r.json()
                    code = str(payload.get("code")) if isinstance(payload, dict) else "?"
                    if code == "200000" and isinstance(payload.get("data"), list):
                        return payload["data"]
                    err = f"API code {code}: {str(payload)[:120]}"
                    if code.startswith("4") and code != "429000":
                        raise ApiError(err)                      # request error: retrying cannot help
                elif r.status_code == 429:
                    err = "HTTP 429 rate limited"
                    try:
                        delay = max(delay, float(r.headers.get("Retry-After", 0)))
                    except (TypeError, ValueError):
                        pass
                elif r.status_code >= 500:
                    err = f"HTTP {r.status_code}"
                else:
                    raise ApiError(f"HTTP {r.status_code}: {r.text[:120]}")
            except (requests.RequestException, ValueError) as e:
                err = repr(e)
            if attempt < RETRIES:
                self.sleep(delay)
        raise ApiError(f"FETCH_FAILED after {RETRIES} attempts: {err}")

    def klines(self, symbol, gran_min, start_s, end_s):
        """raw rows for [start_s, end_s] (seconds). Requests are split into <= CHUNK candles with a 1-candle overlap so that
        neither an inclusive nor an exclusive `to` boundary can open a gap."""
        gran_s, out, cs = gran_min * 60, [], int(start_s)
        while cs <= end_s:
            ce = min(int(end_s), cs + (CHUNK - 2) * gran_s)
            out += self._get({"symbol": symbol, "granularity": gran_min, "from": cs * 1000, "to": (ce + gran_s) * 1000})
            if ce >= end_s:
                break
            cs += (CHUNK - 2) * gran_s                           # next chunk starts at ce: 1-candle overlap
        return out


def parse_rows(rows, gran_s):
    """validated candles (ts_seconds, o, h, l, c, v) sorted by time + list of issues. Rows with bad numbers, impossible OHLC, a
    timestamp not aligned to the interval, or two conflicting rows for one timestamp are removed (the hole is then detected as a gap)."""
    seen, bad_ts, issues = {}, set(), []
    for r in rows or []:
        try:
            if len(r) < 6:
                raise ValueError("short row")
            t = int(float(r[0]))
            if t >= 10 ** 11:
                t //= 1000                                       # milliseconds -> seconds
            o, h, l, c, v = (float(x) for x in r[1:6])
        except (TypeError, ValueError):
            issues.append("BAD_ROW")
            continue
        if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0 or h < max(o, c) or l > min(o, c) or l > h:
            issues.append(f"BAD_OHLC:{t}")
            continue
        if t % gran_s:
            issues.append(f"MISALIGNED:{t}")
            continue
        row = (t, o, h, l, c, v)
        if t in seen and seen[t] != row:
            issues.append(f"CONFLICTING_DUPLICATE:{t}")
            bad_ts.add(t)
        seen[t] = row
    return [seen[t] for t in sorted(seen) if t not in bad_ts], issues


def closed_only(candles, gran_s, now):
    return [c for c in candles if c[0] + gran_s <= now]


def contiguous(candles, gran_s):
    return all(candles[i + 1][0] - candles[i][0] == gran_s for i in range(len(candles) - 1))


# ================================================================ indicators (Wilder / standard EMA)
def ema(values, period):
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if len(v) < period or not np.all(np.isfinite(v)):
        return out
    x = v[:period].mean()
    out[period - 1] = x
    a = 2.0 / (period + 1)
    for i in range(period, len(v)):
        x = a * v[i] + (1 - a) * x
        out[i] = x
    return out


def sma(values, period):
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if len(v) < period or not np.all(np.isfinite(v)):
        return out
    cs = np.cumsum(np.insert(v, 0, 0.0))
    out[period - 1:] = (cs[period:] - cs[:-period]) / period
    return out


def rsi(values, period=14):
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if len(v) <= period or not np.all(np.isfinite(v)):
        return out
    d = np.diff(v)
    gains, losses = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    ag, al = gains[:period].mean(), losses[:period].mean()

    def calc(g, l):
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + g / l)

    out[period] = calc(ag, al)
    for i in range(period + 1, len(v)):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        out[i] = calc(ag, al)
    return out


def atr_adx(high, low, close, period=14):
    """Wilder ATR and ADX. ATR[period] = mean of the first `period` true ranges (the first candle has no previous close and is skipped);
    ADX[2*period-1] = mean of the first `period` DX values, then Wilder smoothing."""
    h, l, c = (np.asarray(x, float) for x in (high, low, close))
    n = len(c)
    atr, adx = np.full(n, np.nan), np.full(n, np.nan)
    if n <= period or not (np.all(np.isfinite(h)) and np.all(np.isfinite(l)) and np.all(np.isfinite(c))):
        return atr, adx
    tr, pdm, mdm = np.empty(n), np.zeros(n), np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0

    def wilder_sum(x):
        out = np.full(n, np.nan)
        cur = x[1:period + 1].sum()
        out[period] = cur
        for j in range(period + 1, n):
            cur = cur - cur / period + x[j]
            out[j] = cur
        return out

    trs, ps, ms = wilder_sum(tr), wilder_sum(pdm), wilder_sum(mdm)
    dx = np.full(n, np.nan)
    for i in range(period, n):
        atr[i] = trs[i] / period
        if trs[i] > 0:
            pdi, mdi = 100 * ps[i] / trs[i], 100 * ms[i] / trs[i]
            dx[i] = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) else 0.0
    s, e = period, 2 * period - 1
    if n > e and np.all(np.isfinite(dx[s:e + 1])):
        cur = dx[s:e + 1].mean()
        adx[e] = cur
        for i in range(e + 1, n):
            if np.isfinite(dx[i]):
                cur = ((period - 1) * cur + dx[i]) / period
                adx[i] = cur
    return atr, adx


# ================================================================ signal (pure function of closed candles)
def evaluate_setup(h4, m15):
    """h4 / m15: validated candle lists (ts, o, h, l, c, v). Returns (setup | None, reason). Only 4H candles that closed at or before the
    close of the newest 15m candle are used. reason is "" for a normal result and a short code when data are unusable."""
    if len(m15) < MIN_CANDLES:
        return None, f"INSUFFICIENT_15M:{len(m15)}"
    as_of = m15[-1][0] + BAR_S
    h4 = [x for x in h4 if x[0] + H4_S <= as_of]
    if len(h4) < MIN_CANDLES:
        return None, f"INSUFFICIENT_4H:{len(h4)}"
    if not contiguous(m15[-MIN_CANDLES:], BAR_S):
        return None, "GAP_15M"
    if not contiguous(h4[-MIN_CANDLES:], H4_S):
        return None, "GAP_4H"
    c4 = [x[4] for x in h4]
    e50, e200 = ema(c4, 50), ema(c4, 200)
    _, adx4 = atr_adx([x[2] for x in h4], [x[3] for x in h4], c4)
    if not all(np.isfinite(a[-1]) for a in (e50, e200, adx4)):
        return None, "INDICATOR_INVALID_4H"
    if adx4[-1] < ADX_MIN or e50[-1] == e200[-1]:
        return None, ""
    trend = "LONG" if e50[-1] > e200[-1] else "SHORT"
    o = [x[1] for x in m15]
    c = [x[4] for x in m15]
    v = [x[5] for x in m15]
    e20, vsma, rs = ema(c, 20), sma(v, 20), rsi(c)
    atr15, _ = atr_adx([x[2] for x in m15], [x[3] for x in m15], c)
    i = len(m15) - 1
    if not all(np.isfinite(a[i]) for a in (e20, vsma, rs, atr15)) or not np.isfinite(e20[i - 1]) or atr15[i] <= 0:
        return None, "INDICATOR_INVALID_15M"
    long_ok = trend == "LONG" and c[i - 1] <= e20[i - 1] and c[i] > e20[i] and c[i] > o[i] and v[i] >= vsma[i] and rs[i] < 70
    short_ok = trend == "SHORT" and c[i - 1] >= e20[i - 1] and c[i] < e20[i] and c[i] < o[i] and v[i] >= vsma[i] and rs[i] > 30
    if not (long_ok or short_ok):
        return None, ""
    return {"side": "LONG" if long_ok else "SHORT", "setup_ts": int(m15[i][0]), "expected_entry_ts": int(m15[i][0]) + BAR_S,
            "atr": float(atr15[i])}, ""


# ================================================================ costs and P&L
def funding_count(entry_ts, end_ts):
    """funding settlements strictly after entry_ts up to and including end_ts."""
    return int(end_ts // FUNDING_INTERVAL_S - entry_ts // FUNDING_INTERVAL_S)


def compute_pnl(side, entry, stop, exit_price, entry_ts, exit_bar_ts):
    s = 1 if side == "LONG" else -1
    risk_abs = abs(entry - stop)
    risk_pct = risk_abs / entry
    gross_pct = s * (exit_price / entry - 1.0)
    fee_pct, slip_pct = 2 * FEE_RATE, 2 * SLIPPAGE_RATE
    fund_pct = s * FUNDING_RATE * funding_count(entry_ts, exit_bar_ts + BAR_S)
    net_pct = gross_pct - fee_pct - slip_pct - fund_pct
    return {"gross_pnl_pct": gross_pct, "gross_R": s * (exit_price - entry) / risk_abs, "fee_pct": fee_pct, "slippage_pct": slip_pct,
            "funding_pct": fund_pct, "net_pnl_pct": net_pct, "net_R": net_pct / risk_pct}


def check_candle(pos, c):
    """apply ONE closed candle to a position. Returns (reason, exit_price) or None. SL is checked before TP."""
    ts, o, h, l, cl = c[0], c[1], c[2], c[3], c[4]
    pos["bars_held"] += 1
    pos["last_processed_ts"] = ts
    first = ts == pos["entry_ts"]
    if pos["side"] == "LONG":
        if not first and o <= pos["stop"]:
            return "STOP", o
        if l <= pos["stop"]:
            return "STOP", pos["stop"]
        if h >= pos["target"]:
            return "TARGET", pos["target"]
    else:
        if not first and o >= pos["stop"]:
            return "STOP", o
        if h >= pos["stop"]:
            return "STOP", pos["stop"]
        if l <= pos["target"]:
            return "TARGET", pos["target"]
    if pos["bars_held"] >= MAX_HOLD_BARS:
        return "TIME", cl
    return None


# ================================================================ state
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def new_state():
    return {"version": STATE_VERSION, "pending": {}, "positions": {}, "last_scan": {}, "last_setup": {}}


def validate_state(obj, symbols=SYMBOLS):
    """returns (state, issues). Missing containers are added (non-destructive); anything inconsistent raises StateError."""
    if not isinstance(obj, dict):
        raise StateError("state is not a JSON object")
    if obj.get("version") != STATE_VERSION:
        raise StateError(f"state version {obj.get('version')!r} != {STATE_VERSION}")
    issues = []
    for k in ("pending", "positions", "last_scan", "last_setup"):
        if k not in obj:
            obj[k] = {}
            issues.append(f"MISSING_KEY_ADDED:{k}")
        elif not isinstance(obj[k], dict):
            raise StateError(f"'{k}' is not an object")
    for k, p in obj["pending"].items():
        if not (isinstance(p, dict) and p.get("symbol") == k and k in symbols and p.get("side") in ("LONG", "SHORT")
                and _is_int(p.get("setup_ts")) and _is_int(p.get("expected_entry_ts")) and _is_num(p.get("atr")) and p["atr"] > 0
                and p["expected_entry_ts"] == p["setup_ts"] + BAR_S):
            raise StateError(f"invalid pending entry for {k}")
    for k, p in obj["positions"].items():
        ok = (isinstance(p, dict) and p.get("symbol") == k and k in symbols and p.get("side") in ("LONG", "SHORT")
              and all(_is_int(p.get(x)) for x in ("setup_ts", "entry_ts", "bars_held", "last_processed_ts"))
              and all(_is_num(p.get(x)) and p[x] > 0 for x in ("entry", "stop", "target")))
        if ok:
            long_ = p["side"] == "LONG"
            ok = ((p["stop"] < p["entry"] < p["target"]) if long_ else (p["stop"] > p["entry"] > p["target"]))
            ok = ok and p["entry_ts"] % BAR_S == 0 and p["last_processed_ts"] % BAR_S == 0 and 0 <= p["bars_held"] < MAX_HOLD_BARS
            ok = ok and p["bars_held"] == (p["last_processed_ts"] - p["entry_ts"]) // BAR_S + 1
        if not ok:
            raise StateError(f"invalid or inconsistent position for {k}")
    for k in list(obj["pending"]):
        if k in obj["positions"]:
            del obj["pending"][k]
            issues.append(f"PENDING_DROPPED_DUPLICATE_OF_POSITION:{k}")
    for k in ("last_scan", "last_setup"):
        if not all(isinstance(a, str) and _is_int(b) for a, b in obj[k].items()):
            raise StateError(f"invalid '{k}' entries")
    return obj, issues


def load_state(path, symbols=SYMBOLS):
    path = Path(path)
    if not path.exists():
        return new_state(), []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StateError(f"state file unreadable or corrupt: {e}")
    return validate_state(obj, symbols)


def save_state(path, state):
    path = Path(path)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ================================================================ audit log
class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.ids = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                if rd.fieldnames != LOG_FIELDS:
                    raise LogError("log file header does not match the v3 schema; use a new PAPER_LOG_FILE")
                self.ids = {r["event_id"] for r in rd}

    def write(self, rec):
        """append one record unless its event_id was already written (restart safe). Returns True if written."""
        eid = rec["event_id"]
        if eid in self.ids:
            return False
        row = {k: rec.get(k, "") for k in LOG_FIELDS}
        row["time_utc"] = row["time_utc"] or utc_iso()
        new = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        self.ids.add(eid)
        return True


# ================================================================ engine
class Engine:
    def __init__(self, api, state, log, state_path=STATE_FILE, clock=time.time, symbols=SYMBOLS):
        self.api, self.state, self.log, self.state_path, self.clock, self.symbols = api, state, log, state_path, clock, list(symbols)
        self.dirty = False

    # ---- data ----
    def fetch(self, sym, gran_min, start_s, end_s, now):
        """closed, validated candles with start_s <= ts <= end_s (one extra candle is requested on each side so a single-candle
        range cannot return nothing)."""
        g = gran_min * 60
        rows, _ = parse_rows(self.api.klines(sym, gran_min, start_s - g, end_s + g), g)
        return [c for c in closed_only(rows, g, now) if start_s <= c[0] <= end_s]

    def _log(self, status, sym, key, **kw):
        rec = {"event_id": f"{sym}|{key}|{status}", "symbol": sym, "status": status}
        rec.update(kw)
        self.log.write(rec)

    # ---- scanning ----
    def scan(self, sym, now):
        latest = last_closed_ts(now)
        if self.state["last_scan"].get(sym) == latest:
            return
        m15 = self.fetch(sym, 15, latest - (FETCH_15M - 1) * BAR_S, latest, now)
        if not m15 or m15[-1][0] != latest:
            return                                              # newest closed candle not available yet: try again next cycle
        h4_last = ((latest + BAR_S) // H4_S) * H4_S - H4_S
        h4 = self.fetch(sym, 240, h4_last - (FETCH_4H - 1) * H4_S, h4_last, now)
        setup, reason = evaluate_setup(h4, m15)
        self.state["last_scan"][sym] = latest
        self.dirty = True
        if reason:
            self._log("DATA_ISSUE", sym, latest, setup_ts=latest, note=reason)
            return
        if not setup or self.state["last_setup"].get(sym) == setup["setup_ts"]:
            return
        setup.update(symbol=sym, detected_at=int(now))
        self.state["last_setup"][sym] = setup["setup_ts"]
        self.state["pending"][sym] = setup
        self._log("SETUP_QUEUED", sym, setup["setup_ts"], side=setup["side"], setup_ts=setup["setup_ts"], note="entry reference = next candle OPEN")
        self.process_pending(sym, now)

    # ---- entry ----
    def process_pending(self, sym, now):
        p = self.state["pending"].get(sym)
        if not p:
            return
        T = p["expected_entry_ts"]
        if now < T:
            return
        key = p["setup_ts"]
        if now - T > ENTRY_MAX_DELAY_S:
            self._log("MISSED_ENTRY", sym, key, side=p["side"], setup_ts=key, entry_ts=T,
                      note=f"entry candle open not observed within {ENTRY_MAX_DELAY_S}s of its start; no price substitution")
            del self.state["pending"][sym]
            self.dirty = True
            return
        rows, _ = parse_rows(self.api.klines(sym, 15, T - BAR_S, T + BAR_S), BAR_S)
        got = [r for r in rows if r[0] == T]
        if not got:
            return                                              # not visible yet: retry until the delay limit
        entry = got[0][1]                                       # OPEN of the entry candle only
        risk = SL_ATR * p["atr"]
        long_ = p["side"] == "LONG"
        stop, target = (entry - risk, entry + TARGET_R * risk) if long_ else (entry + risk, entry - TARGET_R * risk)
        if risk <= 0 or stop <= 0 or target <= 0:
            self._log("MISSED_ENTRY", sym, key, side=p["side"], setup_ts=key, entry_ts=T, note="invalid stop/target for this entry price")
            del self.state["pending"][sym]
            self.dirty = True
            return
        pos = {"symbol": sym, "side": p["side"], "setup_ts": key, "entry_ts": T, "entry": entry, "stop": stop, "target": target,
               "bars_held": 0, "last_processed_ts": T - BAR_S, "observed_latency_s": int(now - T)}
        self.state["positions"][sym] = pos
        del self.state["pending"][sym]
        self.dirty = True
        self._log("PAPER_OPEN", sym, key, side=p["side"], setup_ts=key, entry_ts=T, entry_ref=entry, observed_latency_s=int(now - T),
                  stop=stop, target=target, note="reference = candle OPEN (paper); not an actual fill; costs are assumptions")

    # ---- monitoring ----
    def process_position(self, sym, now):
        pos = self.state["positions"].get(sym)
        if not pos:
            return
        nxt, latest = pos["last_processed_ts"] + BAR_S, last_closed_ts(now)
        if nxt > latest:
            return
        by_ts = {c[0]: c for c in self.fetch(sym, 15, nxt, latest, now)}
        ts = nxt
        while ts <= latest:
            c = by_ts.get(ts)
            if c is None:                                       # missing candle: never guess, wait for it
                self._log("DATA_GAP", sym, f"{pos['setup_ts']}|{ts}", side=pos["side"], setup_ts=pos["setup_ts"], note=f"candle {ts} missing; monitoring paused")
                return
            res = check_candle(pos, c)
            self.dirty = True
            if res:
                reason, price = res
                pnl = compute_pnl(pos["side"], pos["entry"], pos["stop"], price, pos["entry_ts"], ts)
                self._log("PAPER_EXIT", sym, pos["setup_ts"], side=pos["side"], setup_ts=pos["setup_ts"], entry_ts=pos["entry_ts"],
                          entry_ref=pos["entry"], observed_latency_s=pos.get("observed_latency_s", ""), stop=pos["stop"], target=pos["target"],
                          exit_ts=ts, exit_price=price, exit_reason=reason, bars_held=pos["bars_held"],
                          note="paper exit; SL before TP in one candle; gap beyond the stop exits at the open; net P&L uses ASSUMED costs", **pnl)
                del self.state["positions"][sym]
                return
            ts += BAR_S

    # ---- cycle ----
    def run_cycle(self, now=None):
        now = self.clock() if now is None else now
        for sym in list(self.state["pending"]):
            self._safe(self.process_pending, sym, now)
        for sym in list(self.state["positions"]):
            self._safe(self.process_position, sym, now)
        for sym in self.symbols:
            if sym not in self.state["pending"] and sym not in self.state["positions"]:
                self._safe(self.scan, sym, now)
        if self.dirty:
            save_state(self.state_path, self.state)
            self.dirty = False

    def _safe(self, fn, sym, now):
        try:
            fn(sym, now)
        except ApiError as e:
            print(f"[API ERROR] {sym}: {e}", flush=True)
        except Exception as e:                                  # one symbol must not stop the others
            print(f"[ERROR] {sym}: {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    print("KUCOIN FUTURES PAPER ENGINE v3 - PUBLIC DATA ONLY, NO ORDERS, NOT ACTUAL FILLS", flush=True)
    try:
        state, issues = load_state(STATE_FILE)
        log = AuditLog(LOG_FILE)
    except (StateError, LogError) as e:
        print(f"STOPPED: {e}\nThe existing files were not changed. Fix or move them manually.", flush=True)
        sys.exit(2)
    eng = Engine(KuCoinApi(), state, log)
    for it in issues:
        log.write({"event_id": f"STATE|{it}|{int(time.time())}", "symbol": "", "status": "STATE_ISSUE", "note": it})
        eng.dirty = True
    while True:
        try:
            eng.run_cycle()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[CYCLE ERROR] {utc_iso()} {e!r}", flush=True)
        if args.once:
            break
        time.sleep(POLL_PENDING_SECONDS if state["pending"] else POLL_SECONDS)


# ================================================================ self-tests (offline: fake exchange, fake clock)
PASSED = 0
T0 = 1_700_000_000 // H4_S * H4_S


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def build_rows(n=4700, seed=11):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.003, n)))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    v = rng.lognormal(0, 0.5, n) * 1000
    return [[T0 + i * BAR_S, float(o[i]), float(h[i]), float(l[i]), float(c[i]), float(v[i])] for i in range(n)]


def mirror_rows(rows):
    K = 2 * max(r[2] for r in rows)
    return [[r[0], K - r[1], K - r[3], K - r[2], K - r[4], r[5]] for r in rows]


def agg4(rows15):
    out = {}
    for r in rows15:
        t = r[0] // H4_S * H4_S
        a = out.get(t)
        out[t] = [t, r[1], r[2], r[3], r[4], r[5]] if a is None else [t, a[1], max(a[2], r[2]), min(a[3], r[3]), r[4], a[5] + r[5]]
    return out


def prefix_eval(rows15, i, a4=None):
    a4 = a4 or agg4(rows15)
    m15 = [tuple(r) for r in rows15[max(0, i - FETCH_15M + 1):i + 1]]
    as_of = rows15[i][0] + BAR_S
    h4 = [tuple(a4[t]) for t in sorted(a4) if t + H4_S <= as_of][-FETCH_4H:]
    return evaluate_setup(h4, m15)


def find_setup(rows, side):
    a4 = agg4(rows)
    for i in range(3700, len(rows) - 5):
        st, _ = prefix_eval(rows, i, a4)
        if st and st["side"] == side:
            return i, st
    raise AssertionError(f"no {side} setup found in synthetic data")


class FakeApi:
    """public-API look-alike: candles up to `now`, forming candle shows only its open, inclusive from/to, ms timestamps."""

    def __init__(self, rows15, now_fn):
        self.rows, self.now, self.drop, self.hide_until, self.fail, self.calls = {r[0]: r for r in rows15}, now_fn, set(), {}, 0, 0

    def klines(self, symbol, gran_min, start_s, end_s):
        self.calls += 1
        if self.fail > 0:
            self.fail -= 1
            raise ApiError("simulated outage")
        now, g, out = self.now(), gran_min * 60, []
        src = self.rows if gran_min == 15 else agg4([self.rows[t] for t in sorted(self.rows) if t <= now])
        for t in range(int(start_s) // g * g, int(end_s) + 1, g):
            if t > now or t not in src or (gran_min, t) in self.drop or self.hide_until.get(t, 0) > now:
                continue
            r = src[t]
            if t + g > now:
                r = [t, r[1], r[1], r[1], r[1], 0.0]
            out.append([t * 1000, r[1], r[2], r[3], r[4], r[5], 0])
        return out


def scenario_rows(side, kind, tail=None):
    """real signal bar + designed future bars. Returns (rows, setup_index, setup, E, R)."""
    base = SCEN[side]
    rows = [list(r) for r in base["rows"][:base["i"] + 1]]
    s, st = base["i"], base["st"]
    E, R = rows[s][4], SL_ATR * st["atr"]
    sg = 1 if side == "LONG" else -1
    t = rows[s][0]

    def bar(k, o, h, l, c):
        rows.append([t + k * BAR_S, E + sg * o * R, E + sg * (h if sg > 0 else l) * R, E + sg * (l if sg > 0 else h) * R, E + sg * c * R, 500.0])

    quiet = (0.0, 0.3, -0.3, 0.0)
    if kind == "TP":
        bar(1, 0, .2, -.2, .1); bar(2, .1, 2.05, .05, 1.9)
    elif kind == "SL":
        bar(1, 0, .2, -.2, .1); bar(2, .1, .3, -1.05, -.9)
    elif kind == "COLLISION":
        bar(1, 0, .2, -.2, 0); bar(2, 0, 2.5, -1.2, 1.0)
    elif kind == "ENTRYBAR_SL":
        bar(1, 0, .2, -1.1, -.9)
    elif kind == "GAP":
        bar(1, 0, .2, -.2, 0); bar(2, -1.5, -1.4, -1.6, -1.5)
    elif kind == "TIME":
        for k in range(1, 40):
            bar(k, *quiet)
    elif kind == "LATE_TP":
        for k in range(1, 7):
            bar(k, *quiet)
        bar(7, 0, 2.2, -.1, 2.0)
        for k in range(8, 30):
            bar(k, *quiet)
    for k in range(len(rows) - s, len(rows) - s + 40):
        bar(k, *quiet)
    for r in rows[s + 1:]:                                       # keep candles valid: high >= max(o,c), low <= min(o,c)
        r[2], r[3] = max(r[2], r[1], r[4]), min(r[3], r[1], r[4])
    return rows, s, st, E, R


SCEN = {}


def make_engine(tmp, rows, symbols=("XBTUSDTM",), preload_state=None):
    clock = {"t": 0.0}
    api = FakeApi(rows, lambda: clock["t"])
    st = preload_state if preload_state is not None else new_state()
    eng = Engine(api, st, AuditLog(tmp / "log.csv"), state_path=tmp / "state.json", clock=lambda: clock["t"], symbols=symbols)
    return eng, api, clock


def drive(eng, clock, t_from, t_to, step=BAR_S, off=2):
    t = t_from
    while t <= t_to:
        clock["t"] = float(t + off)
        eng.run_cycle()
        t += step


def read_log(tmp):
    with open(tmp / "log.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_scenario(tmp_root, side, kind, n_cycles, symbols=("XBTUSDTM",)):
    import tempfile
    tmp = Path(tempfile.mkdtemp(dir=tmp_root))
    rows, s, st, E, R = scenario_rows(side, kind)
    eng, api, clock = make_engine(tmp, rows, symbols)
    B_ = rows[s][0] + BAR_S
    drive(eng, clock, B_, B_ + (n_cycles - 1) * BAR_S)
    return tmp, eng, rows, s, st, E, R, B_


def run_selftest():
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="paper_v3_"))
    # ---------- indicators ----------
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
              46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13]
    r = rsi(closes)
    exp = {14: 70.53, 15: 66.32, 16: 66.55, 17: 69.41, 18: 66.36, 19: 57.97, 20: 62.93, 21: 63.26, 22: 56.06, 23: 62.38, 24: 54.71, 25: 50.42,
           26: 39.99, 27: 41.46, 28: 41.87, 29: 45.46, 30: 37.30, 31: 33.08, 32: 37.77}
    check(all(abs(r[i] - v) < 0.1 for i, v in exp.items()) and np.isnan(r[13]), "RSI matches the published Wilder RSI(14) worked example (published values use rounded inputs, tolerance 0.1)")
    g_ = [max(closes[k] - closes[k - 1], 0.0) for k in range(1, len(closes))]
    l_ = [max(closes[k - 1] - closes[k], 0.0) for k in range(1, len(closes))]
    ag_, al_, ref_ = sum(g_[:14]) / 14, sum(l_[:14]) / 14, []
    ref_.append(100 - 100 / (1 + ag_ / al_))
    for k in range(14, len(g_)):
        ag_, al_ = (ag_ * 13 + g_[k]) / 14, (al_ * 13 + l_[k]) / 14
        ref_.append(100 - 100 / (1 + ag_ / al_))
    check(np.allclose(r[14:], ref_, rtol=1e-10), "RSI equals an independent exact Wilder loop")
    x = np.arange(1, 31, dtype=float)
    check(abs(sma(x, 5)[4] - 3.0) < 1e-12 and abs(sma(x, 5)[-1] - 28.0) < 1e-12 and np.isnan(sma(x, 5)[3]), "SMA")
    e = ema(x, 5)
    check(abs(e[4] - 3.0) < 1e-12 and abs(e[5] - (2 / 6 * 6 + 4 / 6 * 3.0)) < 1e-12 and np.isnan(e[3]), "EMA seeded with the SMA")
    check(np.all(np.isnan(ema([1, 2, np.nan, 4, 5, 6], 3))) and np.all(np.isnan(rsi([1.0] * 5))), "invalid or short input gives NaN, never a signal value")
    rng = np.random.default_rng(3)
    cc = 100 + np.cumsum(rng.normal(0, 1, 120))
    hh, ll = cc + np.abs(rng.normal(0, .5, 120)), cc - np.abs(rng.normal(0, .5, 120))
    atr, adx = atr_adx(hh, ll, cc)
    # independent textbook reference (list based Wilder, different code path)
    trr = [max(hh[i] - ll[i], abs(hh[i] - cc[i - 1]), abs(ll[i] - cc[i - 1])) for i in range(1, 120)]
    a = [sum(trr[:14]) / 14]
    for t_ in trr[14:]:
        a.append((a[-1] * 13 + t_) / 14)
    check(np.allclose(atr[14:], a, rtol=1e-10) and np.isnan(atr[13]), "ATR equals the Wilder recursion ATR = (13*prev + TR)/14, first value = mean of 14 TR")
    P, M = [0.0], [0.0]
    for i in range(1, 120):
        u, d = hh[i] - hh[i - 1], ll[i - 1] - ll[i]
        P.append(u if u > d and u > 0 else 0.0)
        M.append(d if d > u and d > 0 else 0.0)
    trl = [0.0] + trr
    sm = lambda arr: [sum(arr[1:15])] + [None] * 105
    st_, sp, sn = sum(trl[1:15]), sum(P[1:15]), sum(M[1:15])
    dxs = [100 * abs(sp - sn) / (sp + sn)]
    for i in range(15, 120):
        st_, sp, sn = st_ - st_ / 14 + trl[i], sp - sp / 14 + P[i], sn - sn / 14 + M[i]
        dxs.append(100 * abs(sp - sn) / (sp + sn))
    ref = [sum(dxs[:14]) / 14]
    for d in dxs[14:]:
        ref.append((ref[-1] * 13 + d) / 14)
    check(np.allclose(adx[27:], ref, rtol=1e-10) and np.isnan(adx[26]), "ADX equals the independent Wilder reference (first value at index 27)")
    up = np.cumsum(np.full(120, 1.0)) + rng.normal(0, .1, 120)
    _, adx_up = atr_adx(up + .5, up - .5, up)
    _, adx_flat = atr_adx(100 + np.sin(np.arange(120) * 1.3) + .5, 100 + np.sin(np.arange(120) * 1.3) - .5, 100 + np.sin(np.arange(120) * 1.3))
    check(adx_up[-1] > 40 and adx_flat[-1] < 20 and 0 <= adx_flat[-1], "ADX high in a trend, low in a range")
    # ---------- costs ----------
    p = compute_pnl("LONG", 100.0, 98.0, 104.0, 1000, 1000)
    check(abs(p["gross_R"] - 2.0) < 1e-12 and abs(p["gross_pnl_pct"] - 0.04) < 1e-12, "gross R and % for a LONG TP")
    check(abs(p["net_pnl_pct"] - (0.04 - 2 * FEE_RATE - 2 * SLIPPAGE_RATE - FUNDING_RATE * funding_count(1000, 1000 + BAR_S))) < 1e-12
          and abs(p["net_R"] - p["net_pnl_pct"] / 0.02) < 1e-12, "net = gross - fees - slippage - funding")
    ps = compute_pnl("SHORT", 100.0, 102.0, 96.0, 0, 3 * 3600)
    check(abs(ps["gross_R"] - 2.0) < 1e-12 and ps["funding_pct"] == -FUNDING_RATE * funding_count(0, 3 * 3600 + BAR_S), "SHORT R; SHORT receives positive funding")
    check(funding_count(100, 8 * 3600 - 1) == 0 and funding_count(100, 8 * 3600) == 1 and funding_count(8 * 3600, 16 * 3600) == 1, "funding settlement boundaries")
    # ---------- candle parsing / API validation ----------
    rows = [[1700000100000, 1, 2, .5, 1.5, 3, 0], [1700000100000, 1, 2, .5, 1.5, 3, 0], [1700001000000, 1, 2, .5, 1.5, 3, 0], ["x"],
            [1700001900000, 1, 2, 3, 1.5, 3, 0], [1700002800000, "nan", 2, .5, 1.5, 3, 0], [1699999200000, 5, 6, 4, 5.5, 1, 0],
            [1699999200000, 5, 6, 4, 5.6, 1, 0], [1700003700, 7, 8, 6, 7.5, 1, 0]]
    good, iss = parse_rows(rows, BAR_S)
    ts = [g[0] for g in good]
    check(1700000100 in ts and 1700001000 in ts and 1700003700 in ts and 1699999200 not in ts and 1700001900 not in ts and len(ts) == len(set(ts)) and ts == sorted(ts),
          f"bad OHLC / NaN / conflicting duplicate removed, identical duplicate merged, seconds and ms accepted {ts}")
    check(any(i.startswith("CONFLICTING") for i in iss) and any(i.startswith("BAD_OHLC") for i in iss) and "BAD_ROW" in iss, "issues reported")
    check(closed_only([(100, 1, 1, 1, 1, 1), (1000, 1, 1, 1, 1, 1)], 900, 1000) == [(100, 1, 1, 1, 1, 1)], "the forming candle is excluded")
    check(contiguous([(0,), (900,), (1800,)], 900) and not contiguous([(0,), (900,), (2700,)], 900), "contiguity check")

    class Resp:
        def __init__(self, code, payload=None, headers=None, text=""):
            self.status_code, self._p, self.headers, self.text = code, payload, headers or {}, text

        def json(self):
            if self._p is None:
                raise ValueError("no json")
            return self._p

    class Sess:
        def __init__(self, seq):
            self.seq, self.headers, self.n = list(seq), {}, 0

        def get(self, url, params=None, timeout=None):
            self.n += 1
            x = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
            if isinstance(x, Exception):
                raise x
            return x

    slept = []
    ok = Resp(200, {"code": "200000", "data": [[1, 2, 3, 4, 5, 6, 7]]})
    api = KuCoinApi(Sess([Resp(429, headers={"Retry-After": "3"}), Resp(503), requests.ConnectionError("x"), ok]), sleep=slept.append, mono=lambda: 0.0)
    check(api._get({}) == [[1, 2, 3, 4, 5, 6, 7]] and max(slept) >= 3, "retries 429 / 5xx / network errors, honours Retry-After")
    s2 = Sess([Resp(200, {"code": "400100", "msg": "bad"})])
    try:
        KuCoinApi(s2, sleep=slept.append, mono=lambda: 0.0)._get({})
        check(False, "non-retryable API error must raise")
    except ApiError:
        check(s2.n == 1, "a request error (4xxxxx) is not retried")
    s3 = Sess([Resp(200, {"code": "429000", "msg": "too many"}), ok])
    check(KuCoinApi(s3, sleep=slept.append, mono=lambda: 0.0)._get({}) and s3.n == 2, "API rate-limit code 429000 is retried")
    s4 = Sess([Resp(500)])
    try:
        KuCoinApi(s4, sleep=slept.append, mono=lambda: 0.0)._get({})
        check(False, "retry exhaustion must raise")
    except ApiError:
        check(s4.n == RETRIES, "gives up after RETRIES attempts")
    s5 = Sess([Resp(200, ["not a dict"])])
    try:
        KuCoinApi(s5, sleep=slept.append, mono=lambda: 0.0)._get({})
        check(False, "malformed payload must raise")
    except ApiError:
        check(True, "malformed payload rejected")
    for inclusive in (True, False):                                  # chunking never leaves a hole, whatever the `to` semantics
        calls = []

        class RangeSess:
            headers = {}

            def get(self, url, params=None, timeout=None):
                lo, hi = params["from"] // 1000, params["to"] // 1000
                calls.append((lo, hi))
                g = params["granularity"] * 60
                ts_ = [t for t in range(lo // g * g, hi + 1, g) if lo <= t and (t <= hi if inclusive else t < hi)]
                return Resp(200, {"code": "200000", "data": [[t * 1000, 1, 2, .5, 1.5, 1, 0] for t in ts_]})

        got = KuCoinApi(RangeSess(), sleep=lambda x: None, mono=lambda: 0.0).klines("X", 15, T0, T0 + 449 * BAR_S)
        gp, _ = parse_rows(got, BAR_S)
        check([g[0] for g in gp if g[0] <= T0 + 449 * BAR_S] == [T0 + i * BAR_S for i in range(450)] and all((hi - lo) // BAR_S + 1 <= CHUNK for lo, hi in calls),
              f"450 candles fetched without a gap in <= {CHUNK}-candle requests (inclusive={inclusive})")
    # ---------- synthetic market and signal tests ----------
    base = build_rows()
    for side, rws in (("LONG", base), ("SHORT", mirror_rows(base))):
        i, st = find_setup(rws, side)
        SCEN[side] = {"rows": rws, "i": i, "st": st}
    check(SCEN["LONG"]["i"] == SCEN["SHORT"]["i"], "the SHORT scenario is the exact mirror of the LONG scenario")
    i = SCEN["LONG"]["i"]
    a4 = agg4(base)
    st0, _ = prefix_eval(base, i, a4)
    far = [list(r) for r in base[:i + 1]] + [[base[i][0] + BAR_S, 1e6, 1e6, 1, 1, 1]]
    future4 = dict(a4)
    t4 = (base[i][0] + BAR_S) // H4_S * H4_S + H4_S
    future4[t4] = [t4, 1, 9e9, 1, 9e9, 1]
    st1, _ = prefix_eval(base, i, future4)
    check(st0 == st1, "a 4H candle that closes after the 15m signal candle cannot change the signal (no look-ahead)")
    m15 = [tuple(r) for r in base[i - FETCH_15M + 1:i + 1]]
    h4c = [tuple(a4[t]) for t in sorted(a4) if t + H4_S <= base[i][0] + BAR_S][-FETCH_4H:]
    check(evaluate_setup(h4c[:100], m15)[1].startswith("INSUFFICIENT_4H") and evaluate_setup(h4c, m15[:100])[1].startswith("INSUFFICIENT_15M"), "too little data gives no signal")
    gap4 = h4c[:150] + h4c[151:]
    check(evaluate_setup(gap4, m15)[1] == "GAP_4H", "a missing 4H candle blocks the signal")
    gap15 = m15[:200] + m15[201:]
    check(evaluate_setup(h4c, gap15)[1] == "GAP_15M", "a missing 15m candle blocks the signal")
    # ---------- full engine scenarios ----------
    def last(tmp):
        return [r for r in read_log(tmp) if r["status"] == "PAPER_EXIT"]

    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "LONG", "TP", 6)
    lg = read_log(tmp)
    check([r["status"] for r in lg] == ["SETUP_QUEUED", "PAPER_OPEN", "PAPER_EXIT"], f"LONG TP: log sequence {[r['status'] for r in lg]}")
    o_, x_ = lg[1], lg[2]
    check(abs(float(o_["entry_ref"]) - rows[s + 1][1]) < 1e-9 and int(o_["entry_ts"]) == B_ and int(o_["observed_latency_s"]) == 2, "entry = open of the next candle, latency recorded")
    check(abs(float(o_["stop"]) - (E - R)) < 1e-6 or abs(float(o_["stop"]) - (rows[s + 1][1] - R)) < 1e-6, "LONG stop = entry - 1.5 ATR")
    check(x_["exit_reason"] == "TARGET" and int(x_["bars_held"]) == 2 and int(x_["exit_ts"]) == B_ + BAR_S and abs(float(x_["gross_R"]) - 2.0) < 1e-9, "LONG TP on bar 2, +2R gross")
    check(float(x_["net_R"]) < float(x_["gross_R"]) and float(x_["net_pnl_pct"]) < float(x_["gross_pnl_pct"]), "net is below gross after costs")
    check(not eng.state["positions"] and not eng.state["pending"], "state is clean after the exit")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "SHORT", "SL", 6)
    x_ = last(tmp)[0]
    check(x_["side"] == "SHORT" and x_["exit_reason"] == "STOP" and abs(float(x_["gross_R"]) + 1.0) < 1e-9 and int(x_["bars_held"]) == 2, "SHORT SL = -1R on bar 2")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "LONG", "COLLISION", 6)
    x_ = last(tmp)[0]
    check(x_["exit_reason"] == "STOP" and abs(float(x_["gross_R"]) + 1.0) < 1e-9, "SL and TP in one candle -> SL (LONG)")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "SHORT", "COLLISION", 6)
    x_ = last(tmp)[0]
    check(x_["exit_reason"] == "STOP" and abs(float(x_["gross_R"]) + 1.0) < 1e-9, "SL and TP in one candle -> SL (SHORT)")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "LONG", "ENTRYBAR_SL", 4)
    x_ = last(tmp)[0]
    check(x_["exit_reason"] == "STOP" and int(x_["bars_held"]) == 1 and int(x_["exit_ts"]) == B_, "the entry candle is bar 1 and can stop the trade")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "LONG", "GAP", 5)
    x_ = last(tmp)[0]
    check(x_["exit_reason"] == "STOP" and abs(float(x_["gross_R"]) + 1.5) < 1e-6 and abs(float(x_["exit_price"]) - rows[s + 2][1]) < 1e-9, "gap beyond the stop exits at the open (-1.5R)")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "LONG", "TIME", 36)
    x_ = last(tmp)
    check(len(x_) == 1 and x_[0]["exit_reason"] == "TIME" and int(x_[0]["bars_held"]) == MAX_HOLD_BARS and int(x_[0]["exit_ts"]) == B_ + (MAX_HOLD_BARS - 1) * BAR_S
          and abs(float(x_[0]["exit_price"]) - rows[s + MAX_HOLD_BARS][4]) < 1e-9, "time exit at the close of bar 30 (entry candle = bar 1)")
    tmp, eng, rows, s, st, E, R, B_ = run_scenario(root, "SHORT", "TIME", 36)
    check(last(tmp)[0]["exit_reason"] == "TIME", "SHORT time exit")
    # delayed API response
    for delay, expect in ((30, "PAPER_OPEN"), (120, "MISSED_ENTRY")):
        import tempfile as _t
        tmp = Path(_t.mkdtemp(dir=root))
        rows, s, st, E, R = scenario_rows("LONG", "TP")
        eng, api, clock = make_engine(tmp, rows)
        B_ = rows[s][0] + BAR_S
        api.hide_until[B_] = B_ + delay
        for off in (2, 20, 35, 60, 95, 130):
            clock["t"] = float(B_ + off)
            eng.run_cycle()
        stt = [r["status"] for r in read_log(tmp)]
        check(expect in stt and (expect == "PAPER_OPEN") == ("MISSED_ENTRY" not in stt), f"delayed entry candle ({delay}s) -> {expect}: {stt}")
        if expect == "PAPER_OPEN":
            check(int(read_log(tmp)[1]["observed_latency_s"]) == 35, "latency recorded when the candle appears late")
        else:
            check(not eng.state["positions"] and not eng.state["pending"], "missed entry leaves no position")
    # engine started late: setup is found after the entry window -> MISSED_ENTRY, not a later open
    tmp = Path(_t.mkdtemp(dir=root))
    rows, s, st, E, R = scenario_rows("LONG", "TP")
    eng, api, clock = make_engine(tmp, rows)
    clock["t"] = float(rows[s][0] + BAR_S + 200)
    eng.pending_probe = None
    eng.state["last_scan"]["XBTUSDTM"] = 0
    eng.run_cycle(rows[s][0] + BAR_S + 5)
    check([r["status"] for r in read_log(tmp)][:1] != ["PAPER_OPEN"], "no position is opened from a stale setup")
    # missing candle while in a position, then it appears
    tmp0, eng0, rows, s, st, E, R, B_ = run_scenario(root, "LONG", "LATE_TP", 10)
    base_exit = last(tmp0)[0]
    tmp = Path(_t.mkdtemp(dir=root))
    eng, api, clock = make_engine(tmp, rows)
    api.drop.add((15, B_ + 3 * BAR_S))
    drive(eng, clock, B_, B_ + 9 * BAR_S)
    check(not last(tmp) and eng.state["positions"]["XBTUSDTM"]["bars_held"] == 3 and any(r["status"] == "DATA_GAP" for r in read_log(tmp)), "a missing candle pauses monitoring (no guessing)")
    api.drop.clear()
    drive(eng, clock, B_ + 10 * BAR_S, B_ + 10 * BAR_S)
    ex = last(tmp)
    check(len(ex) == 1 and ex[0]["exit_reason"] == base_exit["exit_reason"] and ex[0]["exit_ts"] == base_exit["exit_ts"] and ex[0]["gross_R"] == base_exit["gross_R"],
          "after the candle appears the result equals the no-gap run")
    # downtime catch-up: one late cycle processes every missed candle in order
    tmp = Path(_t.mkdtemp(dir=root))
    eng, api, clock = make_engine(tmp, rows)
    drive(eng, clock, B_, B_ + 2 * BAR_S)
    check(eng.state["positions"]["XBTUSDTM"]["bars_held"] == 2, "two bars processed before the outage")
    clock["t"] = float(B_ + 30 * BAR_S + 5)
    eng.run_cycle()
    ex = last(tmp)
    check(len(ex) == 1 and ex[0]["exit_ts"] == base_exit["exit_ts"] and ex[0]["exit_reason"] == "TARGET" and not eng.state["positions"], "after downtime the missed candles are processed once, in order")
    # restart recovery: a new engine from the saved files continues identically; duplicate cycles change nothing
    tmp = Path(_t.mkdtemp(dir=root))
    eng, api, clock = make_engine(tmp, rows)
    for k in range(10):
        clock["t"] = float(B_ + k * BAR_S + 2)
        eng.run_cycle()
        eng.run_cycle()                                           # duplicate processing at the same time
        stt, iss = load_state(tmp / "state.json")
        eng = Engine(api, stt, AuditLog(tmp / "log.csv"), state_path=tmp / "state.json", clock=eng.clock, symbols=("XBTUSDTM",))
    lg = read_log(tmp)
    check([r["status"] for r in lg] == ["SETUP_QUEUED", "PAPER_OPEN", "PAPER_EXIT"] and len({r["event_id"] for r in lg}) == len(lg), "restart + duplicate cycles: no duplicate records")
    check(lg[-1]["exit_ts"] == base_exit["exit_ts"] and lg[-1]["gross_R"] == base_exit["gross_R"], "restart recovery result equals the uninterrupted run")
    # crash between the log write and the state save
    tmp = Path(_t.mkdtemp(dir=root))
    eng, api, clock = make_engine(tmp, rows)
    drive(eng, clock, B_, B_ + 6 * BAR_S)
    stt, _ = load_state(tmp / "state.json")
    check("XBTUSDTM" in stt["positions"], "position still open before the exit candle")
    real_save = save_state
    globals()["save_state"] = lambda *a, **k: (_ for _ in ()).throw(OSError("crash before the state was saved"))
    try:
        clock["t"] = float(B_ + 7 * BAR_S + 2)
        try:
            eng.run_cycle()
        except OSError:
            pass
    finally:
        globals()["save_state"] = real_save
    check(len(last(tmp)) == 1, "exit was logged before the crash")
    stt, _ = load_state(tmp / "state.json")
    check("XBTUSDTM" in stt["positions"], "state file still shows the open position after the crash")
    eng = Engine(api, stt, AuditLog(tmp / "log.csv"), state_path=tmp / "state.json", clock=eng.clock, symbols=("XBTUSDTM",))
    eng.run_cycle(B_ + 7 * BAR_S + 4)
    check(len(last(tmp)) == 1 and not eng.state["positions"], "after the crash the exit is recomputed without a duplicate record")
    # API outage handled
    tmp = Path(_t.mkdtemp(dir=root))
    eng, api, clock = make_engine(tmp, rows)
    api.fail = 3
    clock["t"] = float(B_ + 2)
    eng.run_cycle()
    check(not eng.state["pending"] and not eng.state["positions"], "API failure creates no state")
    # ---------- state validation ----------
    sp_ = root / "bad.json"
    sp_.write_text("{not json", encoding="utf-8")
    before = sp_.read_bytes()
    try:
        load_state(sp_)
        check(False, "corrupt state must raise")
    except StateError:
        check(sp_.read_bytes() == before, "corrupt state file is left untouched")
    good_pos = {"symbol": "XBTUSDTM", "side": "LONG", "setup_ts": 900, "entry_ts": 1800, "entry": 100.0, "stop": 98.0, "target": 104.0, "bars_held": 2, "last_processed_ts": 2700}
    st_ok, iss = validate_state({"version": 3, "positions": {"XBTUSDTM": dict(good_pos)}, "pending": {"XBTUSDTM": {"symbol": "XBTUSDTM", "side": "LONG", "setup_ts": 900, "expected_entry_ts": 1800, "atr": 1.0}}})
    check(set(st_ok) >= {"last_scan", "last_setup"} and "XBTUSDTM" not in st_ok["pending"] and any("MISSING_KEY_ADDED" in x for x in iss) and any("DUPLICATE" in x for x in iss), "missing keys added, duplicate pending dropped, position kept")
    for mutate, label in ((lambda p: p.update(side="UP"), "bad side"), (lambda p: p.update(stop=101.0), "stop on the wrong side"), (lambda p: p.update(bars_held=5), "bars_held inconsistent"),
                          (lambda p: p.update(entry=float("nan")), "NaN price"), (lambda p: p.update(symbol="ETHUSDTM"), "key/symbol mismatch"), (lambda p: p.update(bars_held=30, last_processed_ts=1800 + 29 * BAR_S), "position should already be closed")):
        p = dict(good_pos)
        mutate(p)
        try:
            validate_state({"version": 3, "pending": {}, "positions": {"XBTUSDTM": p}, "last_scan": {}, "last_setup": {}})
            check(False, f"{label} must be rejected")
        except StateError:
            check(True, label)
    for bad in ([], {"version": 2}, {"version": 3, "pending": []}):
        try:
            validate_state(bad)
            check(False, "invalid state object must be rejected")
        except StateError:
            check(True, "invalid state object rejected")
    sp2 = root / "ok.json"
    save_state(sp2, st_ok)
    save_state(sp2, st_ok)
    check(load_state(sp2)[0]["positions"]["XBTUSDTM"]["entry"] == 100.0 and (root / "ok.json.bak").exists() and not (root / "ok.json.tmp").exists(), "atomic save with a backup")
    lp = root / "old.csv"
    lp.write_text("time_utc,symbol\n1,2\n", encoding="utf-8")
    try:
        AuditLog(lp)
        check(False, "foreign log schema must be refused")
    except LogError:
        check(True, "foreign log schema refused (no mixing with the v2 log)")
    print(f"ALL PAPER ENGINE v3 SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    main()
