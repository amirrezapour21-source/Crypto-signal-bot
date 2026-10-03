# ============================================================
# CANDIDATE 11 — FORWARD PAPER TRADING ENGINE (v2)
# FROZEN: 4H | SHORT ONLY | SL=1.25 ATR | TP=2R | HOLD=30
# Run every 4H (GitHub Actions cron). State is committed to the repo.
# ============================================================

import os
import sys
import json
import time
from collections import Counter

import requests
import pandas as pd
import numpy as np


BASE = "https://api-futures.kucoin.com"
STATE_FILE = os.environ.get(
    "PAPER_STATE_FILE",
    "candidate11_paper_state.json"
)
STATE_VERSION = 2

GRANULARITY = 240                # minutes -> 4H
BAR = pd.Timedelta(hours=4)
CHUNK = 150                      # candles per request (API returns only ~200 max)
OVERLAP = 2                      # candles shared between chunks (no gaps)
N_CHUNKS = 4                     # ~600 candles in total
MIN_CANDLES = 250                # warm-up for EMA200 / ATR

# FROZEN STRATEGY
SL_ATR = 1.25
TP_R = 2.0
HOLD = 30
MAX_BARS = HOLD + 1              # entry candle + 30 (same window as backtest)

# EXECUTION / REPORTING (not strategy parameters)
MAX_ENTRY_DELAY_MIN = 90         # later than this -> signal is MISSED, not traded
FEE_RT = 0.002                   # assumed round-trip fee+slippage (% of price)
TARGET_TRADES = 40
MAX_FAIL_FRAC = 0.2              # >20% symbols failing -> exit code 1
PAUSE = 0.15

SYMS = [
    "XBTUSDTM", "ETHUSDTM", "SOLUSDTM", "BNBUSDTM", "XRPUSDTM",
    "DOGEUSDTM", "ADAUSDTM", "LINKUSDTM", "AVAXUSDTM", "DOTUSDTM",
    "SUIUSDTM", "TRXUSDTM", "NEARUSDTM", "AAVEUSDTM", "OPUSDTM",
    "ARBUSDTM", "APTUSDTM", "ATOMUSDTM", "FILUSDTM", "LTCUSDTM",
    "BCHUSDTM", "ETCUSDTM", "UNIUSDTM", "INJUSDTM", "SEIUSDTM",
    "VETUSDTM", "HBARUSDTM", "ALGOUSDTM", "XLMUSDTM", "ICPUSDTM",
    "WIFUSDTM", "PEPEUSDTM", "FLOKIUSDTM",
]


def utcnow():
    return pd.Timestamp.now(tz="UTC")


# ============================================================
# STATE
# ============================================================

def new_state(now):
    return {
        "version": STATE_VERSION,
        "created": now.isoformat(),
        "last_run": None,
        "symbols": {},      # per-symbol: last processed candle
        "open": {},
        "closed": [],
        "signals": [],      # traded signals
        "missed": [],       # signals not traded (late / position open)
    }


def load_state(now):
    if not os.path.exists(STATE_FILE):
        return new_state(now), True

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception as e:
        # never silently reset: that would erase the forward record
        raise SystemExit(
            f"ABORT: STATE FILE UNREADABLE ({e!r}). "
            f"Fix or delete {STATE_FILE} manually."
        )

    if s.get("version") != STATE_VERSION:
        raise SystemExit(
            f"ABORT: STATE VERSION {s.get('version')} != {STATE_VERSION}"
        )

    for k, default in (
        ("symbols", {}), ("open", {}), ("closed", []),
        ("signals", []), ("missed", [])
    ):
        s.setdefault(k, default)

    return s, False


def save_state(state):
    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    os.replace(tmp, STATE_FILE)


# ============================================================
# API
# ============================================================

def check_ohlc(d):
    """Detects a wrong column order: high must be the max, low the min."""
    hi_ok = d["high"] >= d[["open", "close"]].max(axis=1) - d["close"].abs() * 1e-9
    lo_ok = d["low"] <= d[["open", "close"]].min(axis=1) + d["close"].abs() * 1e-9
    frac = float((hi_ok & lo_ok).mean())

    if frac < 0.99:
        raise ValueError(
            f"OHLC_INCONSISTENT ok={frac:.2f} (check column order)"
        )


def parse_klines(rows):
    # KuCoin FUTURES row order: time, open, high, low, close, volume, turnover
    if not rows:
        raise ValueError("EMPTY_DATA")

    if any(len(r) < 6 for r in rows):
        raise ValueError("BAD_ROW_LENGTH")

    d = pd.DataFrame(
        [r[:6] for r in rows],
        columns=["ts", "open", "high", "low", "close", "volume"]
    )

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna().copy()

    d["ts"] = pd.to_datetime(
        d["ts"].astype("int64"),
        unit="ms",
        utc=True
    )

    d = (
        d.drop_duplicates("ts")
         .sort_values("ts")
         .reset_index(drop=True)
    )

    check_ohlc(d)

    return d


def fetch_chunk(symbol, start_ms, end_ms):
    last_err = None

    for attempt in range(3):

        try:
            r = requests.get(
                f"{BASE}/api/v1/kline/query",
                params={
                    "symbol": symbol,
                    "granularity": GRANULARITY,
                    "from": start_ms,
                    "to": end_ms
                },
                timeout=20
            )
        except requests.RequestException as e:
            last_err = repr(e)
            time.sleep(1 + attempt)
            continue

        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"HTTP_{r.status_code}"
            time.sleep(2 * (attempt + 1))
            continue

        r.raise_for_status()
        j = r.json()

        if str(j.get("code")) != "200000":
            raise ValueError(f"API_CODE={j.get('code')}")

        return j.get("data") or []

    raise RuntimeError(f"REQUEST_FAILED: {last_err}")


def fetch_raw(symbol, now):
    # The endpoint returns only ~200 rows per call, so history is
    # collected in several overlapping time windows (newest first).
    bar_ms = GRANULARITY * 60 * 1000
    end_ms = int(now.timestamp() * 1000)

    rows = []

    for i in range(N_CHUNKS):
        e = end_ms - i * CHUNK * bar_ms
        s = e - (CHUNK + OVERLAP) * bar_ms
        rows.extend(fetch_chunk(symbol, s, e))
        time.sleep(PAUSE)

    return parse_klines(rows)


def check_contiguous(closed):
    if len(closed) < MIN_CANDLES:
        raise ValueError(f"TOO_FEW_CANDLES={len(closed)}")

    tail = closed.tail(MIN_CANDLES)

    if not (tail["ts"].diff().dropna() == BAR).all():
        raise ValueError("GAP_OR_WRONG_GRANULARITY")


# ============================================================
# INDICATORS (FROZEN — identical to the validated backtest)
# ============================================================

def prepare(d):
    d = d.copy()

    prev_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = tr.ewm(alpha=1 / 20, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()

    mean20 = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std()
    d["z"] = (d["close"] - mean20) / std20

    d["range20"] = (
        d["high"].rolling(20).max() - d["low"].rolling(20).min()
    )
    d["range_med"] = d["range20"].rolling(20).median()
    d["vol_med"] = d["volume"].rolling(20).median()

    return d


def explain(d, k):
    """First failed condition for candle k (or SIGNAL)."""
    c = d.iloc[k]
    p = d.iloc[k - 1]

    need = [
        c["atr"], c["ema200"], c["z"], p["z"],
        c["range20"], c["range_med"], c["vol_med"]
    ]

    if any(pd.isna(x) for x in need):
        return "WARMUP_NAN"

    if not (p["z"] >= 2 and c["z"] < 2):
        return "NO_Z_CROSS"

    if not c["close"] < c["ema200"]:
        return "ABOVE_EMA200"

    if not c["range20"] >= c["range_med"]:
        return "LOW_RANGE"

    if not c["volume"] >= c["vol_med"]:
        return "LOW_VOLUME"

    return "SIGNAL"


# ============================================================
# POSITIONS
# ============================================================

def seen(state, sym, sig_iso):
    for x in state["signals"] + state["missed"] + state["closed"]:
        if x.get("symbol") == sym and x.get("signal_ts") == sig_iso:
            return True

    p = state["open"].get(sym)
    return p is not None and p.get("signal_ts") == sig_iso


def handle_signal(state, sym, d, k, raw, now, run):
    sig_ts = d["ts"].iloc[k]
    sig_iso = sig_ts.isoformat()

    if seen(state, sym, sig_iso):
        return

    run["new_signals"] += 1

    entry_ts = sig_ts + BAR
    delay = (now - entry_ts).total_seconds() / 60.0

    if sym in state["open"]:
        reason = "OPEN_POSITION"
    elif delay > MAX_ENTRY_DELAY_MIN:
        reason = f"LATE_{int(delay)}MIN"
    else:
        reason = None

    if reason:
        state["missed"].append({
            "symbol": sym,
            "signal_ts": sig_iso,
            "reason": reason
        })
        run["missed"] += 1
        print("MISSED", sym, sig_iso, reason)
        return

    atr = float(d["atr"].iloc[k])

    if not np.isfinite(atr) or atr <= 0:
        return

    er = raw[raw["ts"] == entry_ts]

    if len(er):
        entry = float(er["open"].iloc[0])
        src = "entry_open"
    else:
        entry = float(d["close"].iloc[k])
        src = "signal_close"

    risk = SL_ATR * atr

    state["open"][sym] = {
        "symbol": sym,
        "signal_ts": sig_iso,
        "entry_ts": entry_ts.isoformat(),
        "entry": entry,
        "entry_src": src,
        "entry_delay_min": round(delay, 1),
        "detect_price": float(raw["close"].iloc[-1]),
        "atr": atr,
        "risk": risk,
        "risk_pct": risk / entry,
        "sl": entry + risk,
        "tp": entry - TP_R * risk,
        "bars": 0,
        "last_bar_ts": None
    }

    state["signals"].append({
        "symbol": sym,
        "signal_ts": sig_iso,
        "entry_ts": entry_ts.isoformat()
    })

    run["opened"] += 1

    print(
        "ENTRY", sym,
        "ENTRY=", round(entry, 8),
        "SL=", round(entry + risk, 8),
        "TP=", round(entry - TP_R * risk, 8),
        f"(delay {delay:.0f}min, {src})"
    )


def monitor_bar(state, sym, pos, row, run):
    ts = row["ts"]

    # idempotent: every candle is counted once
    if pos.get("last_bar_ts") and pd.Timestamp(pos["last_bar_ts"]) >= ts:
        return

    pos["last_bar_ts"] = ts.isoformat()
    pos["bars"] += 1

    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    # FROZEN: SL FIRST
    if high >= pos["sl"]:
        exit_price, gross_r, reason = pos["sl"], -1.0, "SL"

    elif low <= pos["tp"]:
        exit_price, gross_r, reason = pos["tp"], TP_R, "TP"

    elif pos["bars"] >= MAX_BARS:
        exit_price = close
        gross_r = (pos["entry"] - close) / pos["risk"]
        reason = "TIMEOUT"

    else:
        return

    state["closed"].append({
        **pos,
        "exit_ts": ts.isoformat(),
        "exit": exit_price,
        "reason": reason,
        "gross_r": float(gross_r)
    })

    del state["open"][sym]
    run["closed"] += 1

    print("CLOSED", sym, reason, "R=", round(gross_r, 4))


def process_symbol(state, sym, raw, now, run):
    closed = raw[raw["ts"] + BAR <= now].reset_index(drop=True)

    check_contiguous(closed)

    d = prepare(closed)
    n = len(d)

    run["reasons"][explain(d, n - 1)] += 1

    last = d.iloc[-1]

    if (
        pd.notna(last["z"])
        and last["z"] >= 2
        and last["close"] < last["ema200"]
    ):
        run["watch"].append(sym)

    if run["last_candle"] is None or d["ts"].iloc[-1] > run["last_candle"]:
        run["last_candle"] = d["ts"].iloc[-1]

    ss = state["symbols"].get(sym)

    if ss is None:
        # first sight: only the latest completed candle is evaluated
        ss = {"last_ts": d["ts"].iloc[-2].isoformat()}
        state["symbols"][sym] = ss

    last_ts = pd.Timestamp(ss["last_ts"])

    for k in range(1, n):

        ts = d["ts"].iloc[k]

        if ts <= last_ts:
            continue

        pos = state["open"].get(sym)

        if pos is not None and ts >= pd.Timestamp(pos["entry_ts"]):
            monitor_bar(state, sym, pos, d.iloc[k], run)

        if explain(d, k) == "SIGNAL":
            handle_signal(state, sym, d, k, raw, now, run)

        ss["last_ts"] = ts.isoformat()


# ============================================================
# REPORT
# ============================================================

def block(name, r):
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(eq)).min()

    print(
        f"{name:<22} Exp={r.mean():+.4f} PF={pf:.3f} "
        f"Total={r.sum():+.2f}R DD={dd:+.2f}R "
        f"Win={(r > 0).mean():.2%}"
    )


def report(state):
    tr = state["closed"]

    print()
    print("=" * 60)
    print("CUMULATIVE PAPER RESULTS")
    print("=" * 60)
    print(
        f"CLOSED={len(tr)}/{TARGET_TRADES} | "
        f"OPEN={len(state['open'])} | "
        f"TRADED={len(state['signals'])} | "
        f"MISSED={len(state['missed'])}"
    )

    if not tr:
        return

    g = np.array([x["gross_r"] for x in tr], dtype=float)
    rp = np.array([x["risk_pct"] for x in tr], dtype=float)

    block("GROSS", g)
    block(f"NET (fee {FEE_RT * 100:.2f}%)", g - FEE_RT / rp)

    reasons = Counter(x["reason"] for x in tr)
    print("EXITS:", dict(reasons))

    if len(tr) >= TARGET_TRADES:
        print(f"TARGET REACHED ({TARGET_TRADES}) -> READY FOR REVIEW")


def main():
    now = utcnow()
    state, fresh = load_state(now)

    print("=" * 60)
    print("CANDIDATE 11 — FORWARD PAPER TRADING (v2)")
    print("4H | SHORT ONLY | SL 1.25 ATR | TP 2R | MAX 31 BARS")
    print(f"RUN_TIME_UTC = {now.isoformat()}")
    print(
        f"STATE = {'NEW (baseline run)' if fresh else 'LOADED'} | "
        f"open={len(state['open'])} closed={len(state['closed'])}"
    )
    print("=" * 60)

    run = {
        "reasons": Counter(),
        "watch": [],
        "new_signals": 0,
        "opened": 0,
        "closed": 0,
        "missed": 0,
        "last_candle": None
    }

    ok = 0
    failed = {}

    for sym in SYMS:

        try:
            raw = fetch_raw(sym, now)
            process_symbol(state, sym, raw, now, run)
            ok += 1

        except Exception as e:
            failed[sym] = f"{type(e).__name__}: {e}"

        time.sleep(PAUSE)

    state["last_run"] = now.isoformat()
    save_state(state)

    print()
    print(f"SYMBOLS_OK={ok}/{len(SYMS)} FAILED={len(failed)}")

    for s, m in failed.items():
        print("FAIL", s, m)

    print("LAST CLOSED CANDLE =", run["last_candle"])
    print("CANDLE CHECK (first failed condition):", dict(run["reasons"]))
    print("WATCH (z>=2 & below EMA200):", run["watch"])
    print(
        f"THIS RUN: NEW_SIGNALS={run['new_signals']} "
        f"OPENED={run['opened']} CLOSED={run['closed']} "
        f"MISSED={run['missed']}"
    )

    for s, p in state["open"].items():
        print(
            "OPEN", s,
            "entry=", round(p["entry"], 8),
            "sl=", round(p["sl"], 8),
            "tp=", round(p["tp"], 8),
            "bars=", p["bars"]
        )

    report(state)

    print()
    print("DONE — Candidate 11 Paper Trading")

    if ok < (1 - MAX_FAIL_FRAC) * len(SYMS):
        sys.exit(1)


if __name__ == "__main__":
    main()
