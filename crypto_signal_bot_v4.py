# ============================================================
# CANDIDATE 11 — FORWARD PAPER TRADING ENGINE (v4)
# FROZEN: 4H | SHORT ONLY | ZSCORE20 +2 CROSS-DOWN
#         | BELOW EMA200 | RANGE20 >= RANGE MEDIAN20
#         | VOLUME >= VOLUME MEDIAN20
#         | ENTRY = NEXT CANDLE OPEN
#         | SL = 1.25 ATR20 | TP = 2R | HOLD = 30 BARS
# PAPER ONLY — NO REAL ORDERS
# ============================================================

import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests


BASE = "https://api-futures.kucoin.com"
KLINE_URL = BASE + "/api/v1/kline/query"

STATE_FILE = os.environ.get(
    "PAPER_STATE_FILE",
    "candidate11_paper_state.json"
)

STATE_VERSION = 3

# ------------------------------------------------------------
# FROZEN TIMEFRAME / EXECUTION
# ------------------------------------------------------------

GRANULARITY = 240
BAR = pd.Timedelta(hours=4)
BAR_SECONDS = 4 * 60 * 60

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

CHUNK = 150
N_CHUNKS = 4

# Real overlap between adjacent chunks.
OVERLAP = 2

MIN_CANDLES = 250

# ------------------------------------------------------------
# FROZEN INDICATORS
# ------------------------------------------------------------

ZSCORE_N = 20
EMA_N = 200
RANGE_N = 20
VOL_N = 20
ATR_N = 20

# ------------------------------------------------------------
# FROZEN TRADE PARAMETERS
# ------------------------------------------------------------

SL_ATR = 1.25
TP_R = 2.0
HOLD = 30

# IMPORTANT:
# Entry candle is the first monitored candle.
# Therefore HOLD=30 means exactly 30 monitored candles.
MAX_BARS = HOLD

# Signal candle closes -> next candle opens.
# If workflow arrives too late, signal is missed.
MAX_ENTRY_DELAY_MIN = 90

# Paper-report stress assumption only.
# NOT an actual exchange fee/funding calculation.
FEE_RT = 0.002

# ------------------------------------------------------------
# PAPER COLLECTION
# ------------------------------------------------------------

TARGET_TRADES = 40
MAX_FAIL_FRAC = 0.20

PAUSE = 0.15
REQUEST_TIMEOUT = 25
RETRIES = 4

# Keep diagnostic history bounded.
MAX_ERRORS = 100
MAX_MISSED = 500
MAX_SIGNALS = 500
MAX_CLOSED = 5000

# ------------------------------------------------------------
# SYMBOLS
# ------------------------------------------------------------

SYMS = [
    "XBTUSDTM",
    "ETHUSDTM",
    "SOLUSDTM",
    "BNBUSDTM",
    "XRPUSDTM",
    "DOGEUSDTM",
    "ADAUSDTM",
    "LINKUSDTM",
    "AVAXUSDTM",
    "DOTUSDTM",
    "SUIUSDTM",
    "TRXUSDTM",
    "NEARUSDTM",
    "AAVEUSDTM",
    "OPUSDTM",
    "ARBUSDTM",
    "APTUSDTM",
    "ATOMUSDTM",
    "FILUSDTM",
    "LTCUSDTM",
    "BCHUSDTM",
    "ETCUSDTM",
    "UNIUSDTM",
    "INJUSDTM",
    "SEIUSDTM",
    "VETUSDTM",
    "HBARUSDTM",
    "ALGOUSDTM",
    "XLMUSDTM",
    "ICPUSDTM",
    "WIFUSDTM",
    "PEPEUSDTM",
    "FLOKIUSDTM",
]


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "candidate11-paper-v4/1.0"
    }
)


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now():
    return pd.Timestamp.now(tz="UTC")


def iso(ts):
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


# ============================================================
# STATE
# ============================================================

def new_state():
    return {
        "state_version": STATE_VERSION,
        "strategy": "candidate11_short_only_frozen_v3",
        "created_at": utc_now().isoformat(),

        # Intentionally NOT updated every run.
        # This prevents 15-minute state churn.
        "last_run_at": None,

        "symbols": {
            s: {
                "last_ts": None,
                "baseline_done": False,
                "last_status": None,
            }
            for s in SYMS
        },

        "open": {},
        "closed": [],
        "signals": [],
        "missed": [],
        "errors": [],
    }


def atomic_save(state):
    directory = os.path.dirname(
        os.path.abspath(STATE_FILE)
    )

    os.makedirs(directory, exist_ok=True)

    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            indent=2,
            sort_keys=True
        )
        f.write("\n")

    os.replace(tmp, STATE_FILE)


def state_fingerprint(state):
    return json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_state():

    if not os.path.exists(STATE_FILE):
        return new_state(), True

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            state = json.load(f)

    except Exception as e:
        raise RuntimeError(
            f"STATE_READ_ERROR: {e}"
        )

    if not isinstance(state, dict):
        raise RuntimeError(
            "STATE_INVALID: root is not an object"
        )

    version = state.get("state_version")

    if version != STATE_VERSION:
        raise RuntimeError(
            "STATE_VERSION_MISMATCH: "
            f"found={version}, expected={STATE_VERSION}. "
            "State will NOT be auto-reset."
        )

    required = [
        "symbols",
        "open",
        "closed",
        "signals",
        "missed",
        "errors",
    ]

    for key in required:
        if key not in state:
            raise RuntimeError(
                f"STATE_INVALID: missing key '{key}'"
            )

    for sym in SYMS:
        state["symbols"].setdefault(
            sym,
            {
                "last_ts": None,
                "baseline_done": False,
                "last_status": None,
            }
        )

    state.setdefault("last_run_at", None)

    return state, False


# ============================================================
# KLINE PARSING
# ============================================================

def parse_klines(rows):

    if not rows:
        return pd.DataFrame()

    parsed = []

    for r in rows:

        if not isinstance(r, (list, tuple)):
            continue

        if len(r) < 6:
            continue

        try:
            # KuCoin Futures:
            # time, open, high, low, close, volume, turnover

            t = int(r[0])

            # Normalize seconds -> milliseconds.
            if t < 10_000_000_000:
                t *= 1000

            o = float(r[1])
            h = float(r[2])
            l = float(r[3])
            c = float(r[4])
            v = float(r[5])

        except (TypeError, ValueError):
            continue

        if not all(
            np.isfinite(x)
            for x in (o, h, l, c, v)
        ):
            continue

        if min(o, h, l, c) <= 0:
            continue

        if v < 0:
            continue

        if h < max(o, c):
            continue

        if l > min(o, c):
            continue

        if l > h:
            continue

        parsed.append(
            (
                t,
                o,
                h,
                l,
                c,
                v,
            )
        )

    if not parsed:
        return pd.DataFrame()

    df = pd.DataFrame(
        parsed,
        columns=[
            "ts_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df["ts"] = pd.to_datetime(
        df["ts_ms"],
        unit="ms",
        utc=True,
    )

    df = (
        df
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    return df[
        [
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


# ============================================================
# KUCOIN FETCH
# ============================================================

def fetch_chunk(symbol, start_ms, end_ms):

    params = {
        "symbol": symbol,
        "granularity": GRANULARITY,
        "from": int(start_ms),
        "to": int(end_ms),
    }

    last_error = None

    for attempt in range(1, RETRIES + 1):

        try:

            r = SESSION.get(
                KLINE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:

                payload = r.json()

                if (
                    isinstance(payload, dict)
                    and payload.get("code") == "200000"
                ):
                    return payload.get("data") or []

                last_error = RuntimeError(
                    "HTTP 200 but unexpected payload: "
                    + str(payload)[:180]
                )

            elif r.status_code in (
                429,
                500,
                502,
                503,
                504,
            ):

                last_error = RuntimeError(
                    f"HTTP {r.status_code}"
                )

            else:

                last_error = RuntimeError(
                    f"HTTP {r.status_code}: "
                    f"{r.text[:180]}"
                )

        except Exception as e:
            last_error = e

        if attempt < RETRIES:
            time.sleep(
                min(2 ** (attempt - 1), 6)
            )

    raise RuntimeError(
        f"FETCH_FAILED {symbol}: {last_error}"
    )


def fetch_raw(symbol, now):

    now_ms = int(
        pd.Timestamp(now).timestamp() * 1000
    )

    chunk_ms = (
        (CHUNK - 1)
        * BAR_SECONDS
        * 1000
    )

    rows = []

    newest_end = now_ms

    for _ in range(N_CHUNKS):

        start = (
            newest_end
            - chunk_ms
        )

        data = fetch_chunk(
            symbol,
            start,
            newest_end,
        )

        rows.extend(data)

        if data:

            parsed = parse_klines(data)

            if not parsed.empty:

                oldest = int(
                    parsed["ts"]
                    .iloc[0]
                    .timestamp()
                    * 1000
                )

                # IMPORTANT:
                # Move BACKWARD while preserving overlap.
                # Previous code used oldest - OVERLAP,
                # which creates a gap.
                #
                # oldest + OVERLAP means the next chunk
                # overlaps the previous chunk by OVERLAP bars.
                newest_end = (
                    oldest
                    + (
                        OVERLAP
                        * BAR_SECONDS
                        * 1000
                    )
                )

            else:

                newest_end = (
                    start
                    - (
                        OVERLAP
                        * BAR_SECONDS
                        * 1000
                    )
                )

        else:

            newest_end = (
                start
                - (
                    OVERLAP
                    * BAR_SECONDS
                    * 1000
                )
            )

        time.sleep(PAUSE)

    return parse_klines(rows)


# ============================================================
# COMPLETED CANDLES
# ============================================================

def completed_only(df, now):

    if df.empty:
        return df

    cutoff = pd.Timestamp(now).floor("4h")

    return (
        df[df["ts"] < cutoff]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# DATA QUALITY
# ============================================================

def check_contiguous(df):

    if len(df) < MIN_CANDLES:
        return (
            False,
            f"INSUFFICIENT_CANDLES:{len(df)}",
        )

    tail = df.tail(MIN_CANDLES)["ts"]

    diffs = tail.diff().dropna()

    if not (diffs == BAR).all():

        bad = diffs[diffs != BAR]

        sample = (
            str(bad.iloc[0])
            if len(bad)
            else "unknown"
        )

        return (
            False,
            f"GAP_OR_DUPLICATE:{sample}",
        )

    return True, "OK"


# ============================================================
# INDICATORS
# ============================================================

def prepare(df):

    d = df.copy()

    previous_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (
                d["high"]
                - previous_close
            ).abs(),
            (
                d["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Frozen Candidate 11 ATR.
    d["atr20"] = tr.ewm(
        alpha=1 / ATR_N,
        adjust=False,
    ).mean()

    # Frozen Candidate 11 Z-score.
    # pandas rolling std default ddof=1.
    mean20 = (
        d["close"]
        .rolling(ZSCORE_N)
        .mean()
    )

    std20 = (
        d["close"]
        .rolling(ZSCORE_N)
        .std()
    )

    d["z20"] = (
        d["close"] - mean20
    ) / std20.replace(
        0,
        np.nan,
    )

    # Frozen EMA200.
    d["ema200"] = (
        d["close"]
        .ewm(
            span=EMA_N,
            adjust=False,
        )
        .mean()
    )

    # Frozen RANGE20.
    d["range20"] = (
        d["high"]
        .rolling(RANGE_N)
        .max()
        -
        d["low"]
        .rolling(RANGE_N)
        .min()
    )

    # Median of RANGE20.
    d["range_median20"] = (
        d["range20"]
        .rolling(RANGE_N)
        .median()
    )

    # Median volume.
    d["vol_median20"] = (
        d["volume"]
        .rolling(VOL_N)
        .median()
    )

    return d


# ============================================================
# SIGNAL LOGIC
# ============================================================

def signal_reason(d, i):

    r = d.iloc[i]

    p = (
        d.iloc[i - 1]
        if i > 0
        else None
    )

    required = [
        "z20",
        "ema200",
        "range20",
        "range_median20",
        "vol_median20",
        "atr20",
    ]

    if any(
        not np.isfinite(float(r[x]))
        for x in required
    ):
        return "WARMUP_NAN"

    if (
        p is None
        or not np.isfinite(
            float(p["z20"])
        )
    ):
        return "NO_Z_CROSS"

    # --------------------------------------------------------
    # FROZEN SHORT TRIGGER
    # Previous Z >= +2
    # Current Z < +2
    # --------------------------------------------------------

    if not (
        float(p["z20"]) >= 2.0
        and
        float(r["z20"]) < 2.0
    ):
        return "NO_Z_CROSS"

    # --------------------------------------------------------
    # PRICE BELOW EMA200
    # --------------------------------------------------------

    if not (
        float(r["close"])
        < float(r["ema200"])
    ):
        return "ABOVE_EMA200"

    # --------------------------------------------------------
    # RANGE20 >= RANGE MEDIAN20
    # --------------------------------------------------------

    if not (
        float(r["range20"])
        >= float(r["range_median20"])
    ):
        return "LOW_RANGE"

    # --------------------------------------------------------
    # VOLUME >= VOLUME MEDIAN20
    # --------------------------------------------------------

    if not (
        float(r["volume"])
        >= float(r["vol_median20"])
    ):
        return "LOW_VOLUME"

    atr = float(r["atr20"])

    if not np.isfinite(atr) or atr <= 0:
        return "INVALID_ATR"

    return "SIGNAL"


# ============================================================
# DUPLICATE CONTROL
# ============================================================

def already_recorded(
    state,
    symbol,
    signal_ts,
):

    target = pd.Timestamp(
        signal_ts
    ).isoformat()

    for item in state["signals"]:

        if (
            item.get("symbol") == symbol
            and
            item.get("signal_ts") == target
        ):
            return True

    for item in state["missed"]:

        if (
            item.get("symbol") == symbol
            and
            item.get("signal_ts") == target
        ):
            return True

    for item in state["closed"]:

        if (
            item.get("symbol") == symbol
            and
            item.get("signal_ts") == target
        ):
            return True

    for item in state["open"].values():

        if (
            item.get("symbol") == symbol
            and
            item.get("signal_ts") == target
        ):
            return True

    return False


def add_missed(
    state,
    symbol,
    signal_ts,
    reason,
    extra=None,
):

    item = {
        "symbol": symbol,
        "signal_ts": iso(signal_ts),
        "entry_ts": iso(
            pd.Timestamp(signal_ts)
            + BAR
        ),
        "reason": reason,
    }

    if extra:
        item.update(extra)

    state["missed"].append(item)

    if len(state["missed"]) > MAX_MISSED:
        del state["missed"][
            :-MAX_MISSED
        ]


# ============================================================
# SIGNAL -> NEXT CANDLE OPEN
# ============================================================

def handle_signal(
    state,
    symbol,
    d,
    entry_source,
    i,
    now,
):

    signal_ts = d.iloc[i]["ts"]

    if already_recorded(
        state,
        symbol,
        signal_ts,
    ):
        return "DUPLICATE"

    entry_ts = (
        signal_ts + BAR
    )

    delay_min = (
        pd.Timestamp(now)
        - entry_ts
    ).total_seconds() / 60.0

    # Signal's next candle has not opened yet.
    if delay_min < 0:
        return "NOT_DUE"

    # NEVER use signal candle close as fallback entry.
    if delay_min > MAX_ENTRY_DELAY_MIN:

        add_missed(
            state,
            symbol,
            signal_ts,
            "ENTRY_DELAY_EXCEEDED",
            {
                "delay_min": round(
                    delay_min,
                    2,
                )
            },
        )

        return "MISSED"

    # Entry is exactly next candle OPEN.
    entry_row = entry_source[
        entry_source["ts"] == entry_ts
    ]

    if len(entry_row) != 1:

        add_missed(
            state,
            symbol,
            signal_ts,
            "ENTRY_CANDLE_UNAVAILABLE",
            {
                "delay_min": round(
                    delay_min,
                    2,
                )
            },
        )

        return "MISSED"

    row = entry_row.iloc[0]

    entry = float(row["open"])

    atr = float(
        d.iloc[i]["atr20"]
    )

    if (
        not np.isfinite(entry)
        or entry <= 0
        or not np.isfinite(atr)
        or atr <= 0
    ):

        add_missed(
            state,
            symbol,
            signal_ts,
            "INVALID_ENTRY_OR_ATR",
        )

        return "MISSED"

    risk = SL_ATR * atr

    if (
        not np.isfinite(risk)
        or risk <= 0
    ):

        add_missed(
            state,
            symbol,
            signal_ts,
            "INVALID_RISK",
        )

        return "MISSED"

    # One live position per symbol.
    if symbol in [
        p.get("symbol")
        for p in state["open"].values()
    ]:

        add_missed(
            state,
            symbol,
            signal_ts,
            "SYMBOL_ALREADY_OPEN",
        )

        return "MISSED"

    pos_id = (
        f"{symbol}|"
        f"{iso(signal_ts)}"
    )

    position = {

        "id": pos_id,

        "symbol": symbol,

        "side": "SHORT",

        "signal_ts": iso(
            signal_ts
        ),

        "entry_ts": iso(
            entry_ts
        ),

        "entry": entry,

        "atr": atr,

        "risk": risk,

        "risk_pct": risk / entry,

        "sl": entry + risk,

        "tp": entry - TP_R * risk,

        "bars": 0,

        "last_bar_ts": None,

        "source": "next_candle_open",

        "status": "OPEN",
    }

    state["open"][pos_id] = position

    state["signals"].append(
        {
            "id": pos_id,

            "symbol": symbol,

            "signal_ts": iso(
                signal_ts
            ),

            "entry_ts": iso(
                entry_ts
            ),

            "entry": entry,

            "atr": atr,

            "risk": risk,

            "sl": entry + risk,

            "tp": entry - TP_R * risk,

            "source": "next_candle_open",
        }
    )

    if len(state["signals"]) > MAX_SIGNALS:
        del state["signals"][
            :-MAX_SIGNALS
        ]

    return "OPENED"


# ============================================================
# POSITION MONITORING
# ============================================================

def monitor_bar(
    state,
    symbol,
    row,
):

    ts = row["ts"]

    candidates = [
        (pid, p)
        for pid, p
        in state["open"].items()
        if p.get("symbol") == symbol
    ]

    for pos_id, pos in candidates:

        entry_ts = pd.Timestamp(
            pos["entry_ts"]
        )

        if ts < entry_ts:
            continue

        # Idempotency.
        if (
            pos.get("last_bar_ts")
            == iso(ts)
        ):
            continue

        pos["last_bar_ts"] = iso(ts)

        pos["bars"] = (
            int(pos.get("bars", 0))
            + 1
        )

        hi = float(row["high"])
        lo = float(row["low"])
        close = float(row["close"])

        entry = float(pos["entry"])
        risk = float(pos["risk"])

        sl = float(pos["sl"])
        tp = float(pos["tp"])

        # FROZEN INTRABAR PRIORITY:
        # SL FIRST.
        hit_sl = hi >= sl
        hit_tp = lo <= tp

        if hit_sl:

            gross_r = -1.0
            reason = "SL"

        elif hit_tp:

            gross_r = TP_R
            reason = "TP"

        elif pos["bars"] >= MAX_BARS:

            # Exit at close of the 30th monitored candle.
            gross_r = (
                entry - close
            ) / risk

            reason = "TIMEOUT"

        else:

            continue

        risk_pct = risk / entry

        net_r = (
            gross_r
            - (
                FEE_RT
                / risk_pct
            )
        )

        if reason == "SL":
            exit_price = sl
        elif reason == "TP":
            exit_price = tp
        else:
            exit_price = close

        closed = dict(pos)

        closed.update(
            {
                "status": "CLOSED",

                "exit_ts": iso(ts),

                "exit_price": exit_price,

                "exit_reason": reason,

                "gross_r": gross_r,

                "net_r": net_r,

                "bars_held": pos["bars"],

                "fee_assumption_rt": FEE_RT,
            }
        )

        state["closed"].append(
            closed
        )

        if len(state["closed"]) > MAX_CLOSED:
            del state["closed"][
                :-MAX_CLOSED
            ]

        del state["open"][pos_id]

        return reason

    return None


# ============================================================
# SYMBOL PROCESSING
# ============================================================

def process_symbol(
    state,
    symbol,
    raw,
    now,
):

    completed = completed_only(
        raw,
        now,
    )

    if completed.empty:

        state["symbols"][symbol][
            "last_status"
        ] = "NO_COMPLETED_DATA"

        raise RuntimeError(
            "NO_COMPLETED_DATA"
        )

    ok, status = check_contiguous(
        completed
    )

    # IMPORTANT:
    # Data-quality failure is a real symbol failure.
    # Never silently return OK.
    if not ok:

        state["symbols"][symbol][
            "last_status"
        ] = status

        raise RuntimeError(status)

    d = prepare(completed)

    st = state["symbols"][symbol]

    # --------------------------------------------------------
    # FIRST RUN:
    # Do NOT replay historical signals.
    # Start from the latest completed candle.
    # --------------------------------------------------------

    if not st.get("baseline_done"):

        if len(d) < 2:

            st["last_status"] = (
                "BASELINE_WAIT"
            )

            return {
                "signals": 0,
                "opened": 0,
                "closed": 0,
                "missed": 0,
                "reasons": {},
            }

        st["last_ts"] = iso(
            d.iloc[-2]["ts"]
        )

        st["baseline_done"] = True

    last_ts = (
        pd.Timestamp(st["last_ts"])
        if st.get("last_ts")
        else None
    )

    if last_ts is not None:

        new_rows = d[
            d["ts"] > last_ts
        ].copy()

    else:

        new_rows = d.iloc[-1:].copy()

    stats = {
        "signals": 0,
        "opened": 0,
        "closed": 0,
        "missed": 0,
        "reasons": {},
    }

    for _, row in new_rows.iterrows():

        ts = row["ts"]

        idxs = d.index[
            d["ts"] == ts
        ]

        if len(idxs) != 1:
            continue

        idx = int(idxs[0])

        # ----------------------------------------------------
        # FIRST: monitor existing position
        # ----------------------------------------------------

        close_reason = monitor_bar(
            state,
            symbol,
            row,
        )

        if close_reason:
            stats["closed"] += 1

        # ----------------------------------------------------
        # SECOND: evaluate new signal
        # ----------------------------------------------------

        reason = signal_reason(
            d,
            idx,
        )

        stats["reasons"][reason] = (
            stats["reasons"].get(reason, 0)
            + 1
        )

        if reason == "SIGNAL":

            stats["signals"] += 1

            result = handle_signal(
                state,
                symbol,
                d,
                raw,
                idx,
                now,
            )

            if result == "OPENED":
                stats["opened"] += 1

            elif result == "MISSED":
                stats["missed"] += 1

        # ----------------------------------------------------
        # ADVANCE CURSOR
        # ----------------------------------------------------

        st["last_ts"] = iso(ts)

    st["last_status"] = "OK"

    return stats


# ============================================================
# REPORT
# ============================================================

def report(
    state,
    now,
    run_stats,
    ok_count,
    fail_count,
):

    closed = state["closed"]

    gross = [
        float(x["gross_r"])
        for x in closed
        if np.isfinite(
            float(
                x.get(
                    "gross_r",
                    np.nan,
                )
            )
        )
    ]

    net = [
        float(x["net_r"])
        for x in closed
        if np.isfinite(
            float(
                x.get(
                    "net_r",
                    np.nan,
                )
            )
        )
    ]

    def pf(values):

        gp = sum(
            x for x in values
            if x > 0
        )

        gl = -sum(
            x for x in values
            if x < 0
        )

        if gl <= 0:
            return float("inf")

        return gp / gl

    def max_dd(values):

        if not values:
            return 0.0

        eq = np.cumsum(values)

        peak = np.maximum.accumulate(
            eq
        )

        return float(
            np.min(eq - peak)
        )

    net_total = (
        float(sum(net))
        if net
        else 0.0
    )

    net_exp = (
        float(np.mean(net))
        if net
        else 0.0
    )

    net_pf = pf(net)

    wins = sum(
        x > 0
        for x in net
    )

    wr = (
        wins / len(net)
        if net
        else 0.0
    )

    by_reason = {}

    for x in closed:

        r = x.get(
            "exit_reason",
            "UNKNOWN",
        )

        by_reason[r] = (
            by_reason.get(r, 0)
            + 1
        )

    print("\n" + "=" * 72)

    print(
        "CANDIDATE 11 — "
        "FORWARD PAPER TRADING (v4)"
    )

    print(
        "4H | SHORT ONLY | "
        "NEXT OPEN | SL 1.25 ATR | "
        "TP 2R | HOLD 30"
    )

    print(
        f"RUN_TIME_UTC = {iso(now)}"
    )

    print(
        f"STATE_FILE = {STATE_FILE}"
    )

    print(
        f"STATE_VERSION = {STATE_VERSION}"
    )

    print(
        "PAPER ONLY — NO REAL ORDERS"
    )

    print("=" * 72)

    print(
        f"SYMBOLS_OK={ok_count}/{len(SYMS)} "
        f"FAILED={fail_count}"
    )

    print(
        f"THIS RUN: "
        f"NEW_SIGNALS="
        f"{run_stats['signals']} "
        f"OPENED="
        f"{run_stats['opened']} "
        f"CLOSED="
        f"{run_stats['closed']} "
        f"MISSED="
        f"{run_stats['missed']}"
    )

    print(
        f"CLOSED={len(closed)}/{TARGET_TRADES} "
        f"| OPEN={len(state['open'])} "
        f"| SIGNALS={len(state['signals'])} "
        f"| MISSED={len(state['missed'])}"
    )

    print(
        f"NET_EXP={net_exp:+.4f}R "
        f"| NET_PF={net_pf:.3f} "
        f"| NET_TOTAL={net_total:+.2f}R "
        f"| NET_DD={max_dd(net):+.2f}R "
        f"| WR={wr * 100:.2f}%"
    )

    print(
        f"EXIT_REASONS={by_reason}"
    )

    print(
        f"ASSUMED_ROUND_TRIP_COST="
        f"{FEE_RT * 100:.2f}% of price"
    )

    # --------------------------------------------------------
    # SIGNAL REASONS
    # --------------------------------------------------------

    reason_totals = {}

    for reason, count in run_stats[
        "reasons"
    ].items():

        reason_totals[reason] = (
            reason_totals.get(reason, 0)
            + count
        )

    if reason_totals:

        print(
            "THIS_RUN_SIGNAL_REASONS="
            f"{dict(sorted(reason_totals.items()))}"
        )

    if state["missed"]:

        print("RECENT_MISSED:")

        for x in state["missed"][-5:]:

            print(
                f"  {x['symbol']} "
                f"{x['signal_ts']} "
                f"-> {x['reason']}"
            )

    if len(closed) >= TARGET_TRADES:

        print(
            "PAPER_TARGET_REACHED = "
            f"YES ({len(closed)} closed trades)"
        )

    else:

        print(
            "PAPER_TARGET_REACHED = "
            f"NO ({TARGET_TRADES - len(closed)} "
            "more closed trades needed)"
        )

    attempts = (
        len(state["signals"])
        + len(state["missed"])
    )

    missed_fraction = (
        len(state["missed"])
        / attempts
        if attempts
        else 0.0
    )

    print(
        f"MISSED_FRACTION="
        f"{missed_fraction:.3f}"
    )

    if (
        attempts
        and
        missed_fraction > MAX_FAIL_FRAC
    ):

        print(
            "WARNING: "
            "MISSED_FRACTION_ABOVE_LIMIT"
        )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    state, is_new = load_state()

    now = utc_now()

    # Snapshot before processing.
    # This lets us save ONLY if the state really changed.
    before_fingerprint = state_fingerprint(
        state
    )

    print(
        "CANDIDATE 11 — "
        "FORWARD PAPER TRADING ENGINE (v4)"
    )

    print(
        "4H | SHORT ONLY | "
        "NEXT CANDLE OPEN | "
        "SL 1.25 ATR | TP 2R | HOLD 30"
    )

    print(
        f"RUN_TIME_UTC = {now.isoformat()}"
    )

    print(
        f"STATE = "
        f"{'NEW' if is_new else 'EXISTING'} "
        f"| open={len(state['open'])} "
        f"closed={len(state['closed'])}"
    )

    ok_count = 0
    fail_count = 0

    all_run_stats = {
        "signals": 0,
        "opened": 0,
        "closed": 0,
        "missed": 0,
        "reasons": {},
    }

    diagnostics = {}

    for symbol in SYMS:

        try:

            raw = fetch_raw(
                symbol,
                now,
            )

            completed = completed_only(
                raw,
                now,
            )

            diagnostics[symbol] = {
                "raw": len(raw),
                "completed": len(
                    completed
                ),
                "last_completed": (
                    iso(
                        completed[
                            "ts"
                        ].iloc[-1]
                    )
                    if not completed.empty
                    else None
                ),
            }

            if len(completed) < MIN_CANDLES:

                raise RuntimeError(
                    "INSUFFICIENT_COMPLETED_CANDLES:"
                    f"{len(completed)}"
                )

            stats = process_symbol(
                state,
                symbol,
                raw,
                now,
            )

            ok_count += 1

            for key in [
                "signals",
                "opened",
                "closed",
                "missed",
            ]:

                all_run_stats[key] += (
                    stats[key]
                )

            for reason, count in stats[
                "reasons"
            ].items():

                all_run_stats[
                    "reasons"
                ][reason] = (
                    all_run_stats[
                        "reasons"
                    ].get(reason, 0)
                    + count
                )

        except Exception as e:

            fail_count += 1

            diagnostics[symbol] = {
                "error": str(e)
            }

            state["errors"].append(
                {
                    "ts": now.isoformat(),
                    "symbol": symbol,
                    "error": str(e),
                }
            )

            if len(state["errors"]) > MAX_ERRORS:
                del state["errors"][
                    :-MAX_ERRORS
                ]

            state["symbols"][symbol][
                "last_status"
            ] = (
                "ERROR:"
                + str(e)[:160]
            )

            print(
                f"FETCH/PROCESS ERROR "
                f"{symbol}: "
                f"{str(e)[:180]}"
            )

    # --------------------------------------------------------
    # DATA QUALITY SUMMARY
    # --------------------------------------------------------

    valid_last = [
        x["last_completed"]
        for x in diagnostics.values()
        if x.get("last_completed")
    ]

    if valid_last:

        print(
            "LAST CLOSED CANDLE = "
            f"{max(valid_last)}"
        )

    # --------------------------------------------------------
    # FAILURE THRESHOLD
    # --------------------------------------------------------

    failure_fraction = (
        fail_count / len(SYMS)
        if SYMS
        else 1.0
    )

    if (
        failure_fraction
        > MAX_FAIL_FRAC
    ):

        print(
            "RUN_STATUS = "
            "DEGRADED / "
            "TOO_MANY_SYMBOL_FAILURES"
        )

        # Save diagnostic/error state only if changed.
        after_fingerprint = state_fingerprint(
            state
        )

        if (
            is_new
            or
            after_fingerprint
            != before_fingerprint
        ):
            atomic_save(state)
            print(
                "STATE_SAVE = CHANGED"
            )
        else:
            print(
                "STATE_SAVE = UNCHANGED"
            )

        sys.exit(2)

    # --------------------------------------------------------
    # STATE SAVE
    # --------------------------------------------------------

    after_fingerprint = state_fingerprint(
        state
    )

    if (
        is_new
        or
        after_fingerprint
        != before_fingerprint
    ):

        atomic_save(state)

        print(
            "STATE_SAVE = CHANGED"
        )

    else:

        print(
            "STATE_SAVE = UNCHANGED"
        )

    report(
        state,
        now,
        all_run_stats,
        ok_count,
        fail_count,
    )

    print(
        "RUN_STATUS = OK"
    )


if __name__ == "__main__":
    main()
