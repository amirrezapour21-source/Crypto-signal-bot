# ================================================================
# V4 — CANDIDATE 8
# 5-FOLD WALK-FORWARD VALIDATION
# VOLUME BREAKOUT + RETEST
#
# IMPORTANT:
# - Frozen strategy parameters
# - No parameter optimization
# - No extra filters
# - No overlap lock
# - Causal detection
# - Common timestamps
# - Same-candle SL-first
# - Entry candle NOT scanned for exits
# - Cost = 0.003R (FROZEN)
# ================================================================

import time
import json
import urllib.request
import numpy as np
import pandas as pd


# ================================================================
# FROZEN UNIVERSE
# ================================================================

SYMBOLS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT",
    "DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
    "LTC-USDT","BCH-USDT","ETC-USDT","ATOM-USDT","FIL-USDT",
    "NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT",
    "SEI-USDT","INJ-USDT","AAVE-USDT","UNI-USDT","XLM-USDT",
    "TRX-USDT","HBAR-USDT","ALGO-USDT","VET-USDT","ICP-USDT",
    "PEPE-USDT","SHIB-USDT","TON-USDT","MKR-USDT","WIF-USDT"
]


# ================================================================
# FROZEN PARAMETERS
# ================================================================

INTERVAL_SEC = 14400       # 4H
DAYS = 730
PAGE = 1500

DONCHIAN = 20
VOLUME_MEDIAN = 20
ATR_PERIOD = 20
EMA_PERIOD = 200

VOLUME_MULT = 1.5
RETEST_BARS = 5

SL_ATR = 1.25
TP_R = 2.0
HOLD = 30

# Frozen research cost
FEE = 0.001
SLIPPAGE = 0.002
COST = FEE + SLIPPAGE

# Five chronological WF folds
FOLDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00)
]


# ================================================================
# DATA FETCH
# ================================================================

def fetch_history(symbol):

    now = int(time.time())
    start = now - DAYS * 86400

    out = []
    cur = start

    while cur < now:

        end = min(
            cur + (PAGE - 1) * INTERVAL_SEC,
            now
        )

        url = (
            "https://api.kucoin.com/api/v1/market/candles"
            f"?symbol={symbol}"
            f"&type=4hour"
            f"&startAt={cur}"
            f"&endAt={end}"
        )

        success = False

        for attempt in range(3):

            try:

                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                with urllib.request.urlopen(
                    req,
                    timeout=30
                ) as r:

                    payload = json.loads(
                        r.read().decode()
                    )

                if payload.get("code") == "200000":

                    out.extend(payload["data"])
                    success = True
                    break

            except Exception:

                if attempt < 2:
                    time.sleep(1 + attempt)

        if not success:
            return pd.DataFrame()

        cur = end + INTERVAL_SEC
        time.sleep(0.10)

    if not out:
        return pd.DataFrame()

    d = pd.DataFrame(
        out,
        columns=[
            "time",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover"
        ]
    )

    d["time"] = pd.to_numeric(
        d["time"],
        errors="coerce"
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    d = (
        d
        .dropna(subset=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ])
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Remove incomplete current candle
    cutoff = (now // INTERVAL_SEC) * INTERVAL_SEC

    d = d[
        d["time"] < cutoff
    ].reset_index(drop=True)

    return d


# ================================================================
# INDICATORS
# ================================================================

def prepare(d):

    d = d.copy()

    # EMA200
    d["ema200"] = (
        d["close"]
        .ewm(
            span=EMA_PERIOD,
            adjust=False
        )
        .mean()
    )

    # True Range
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - d["close"].shift(1)).abs(),
            (d["low"] - d["close"].shift(1)).abs()
        ],
        axis=1
    ).max(axis=1)

    # Wilder ATR20
    d["atr"] = (
        tr
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    # Volume median
    d["volume_median"] = (
        d["volume"]
        .rolling(
            VOLUME_MEDIAN,
            min_periods=VOLUME_MEDIAN
        )
        .median()
    )

    # ATR median
    d["atr_median"] = (
        d["atr"]
        .rolling(
            ATR_PERIOD,
            min_periods=ATR_PERIOD
        )
        .median()
    )

    # Previous 20-candle Donchian range.
    # shift(1) guarantees causal calculation.
    d["high20"] = (
        d["high"]
        .shift(1)
        .rolling(
            DONCHIAN,
            min_periods=DONCHIAN
        )
        .max()
    )

    d["low20"] = (
        d["low"]
        .shift(1)
        .rolling(
            DONCHIAN,
            min_periods=DONCHIAN
        )
        .min()
    )

    return d


# ================================================================
# CAUSAL EVENT DETECTION
# ================================================================

def detect_events(d):

    events = []

    # EMA200 + rolling indicators need warmup
    START = 220

    for i in range(
        START,
        len(d) - RETEST_BARS
    ):

        r = d.iloc[i]

        if (
            pd.isna(r["high20"])
            or pd.isna(r["low20"])
            or pd.isna(r["atr"])
            or pd.isna(r["atr_median"])
            or pd.isna(r["volume_median"])
            or pd.isna(r["ema200"])
        ):
            continue

        # ========================================================
        # LONG BREAKOUT
        # ========================================================

        if (
            r["close"] > r["high20"]
            and r["volume"] >= VOLUME_MULT * r["volume_median"]
            and r["atr"] >= r["atr_median"]
            and r["close"] > r["ema200"]
        ):

            level = float(r["high20"])

            # First valid retest only
            for j in range(
                i + 1,
                min(
                    i + 1 + RETEST_BARS,
                    len(d)
                )
            ):

                x = d.iloc[j]

                if (
                    x["low"] <= level
                    and x["close"] > level
                ):

                    events.append({
                        "idx": j,
                        "time": int(x["time"]),
                        "side": "LONG"
                    })

                    break

        # ========================================================
        # SHORT BREAKOUT
        # ========================================================

        elif (
            r["close"] < r["low20"]
            and r["volume"] >= VOLUME_MULT * r["volume_median"]
            and r["atr"] >= r["atr_median"]
            and r["close"] < r["ema200"]
        ):

            level = float(r["low20"])

            # First valid retest only
            for j in range(
                i + 1,
                min(
                    i + 1 + RETEST_BARS,
                    len(d)
                )
            ):

                x = d.iloc[j]

                if (
                    x["high"] >= level
                    and x["close"] < level
                ):

                    events.append({
                        "idx": j,
                        "time": int(x["time"]),
                        "side": "SHORT"
                    })

                    break

    return events


# ================================================================
# TRADE SIMULATION
# ================================================================

def simulate_trade(d, event):

    i = event["idx"]
    side = event["side"]

    # Full HOLD window required
    if i + HOLD >= len(d):

        return {
            "status": "OPEN_AT_DATASET_END",
            "r": None
        }

    entry = float(
        d.iloc[i]["close"]
    )

    atr = float(
        d.iloc[i]["atr"]
    )

    if (
        not np.isfinite(entry)
        or not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    risk = SL_ATR * atr

    if side == "LONG":

        sl = entry - risk
        tp = entry + TP_R * risk

    else:

        sl = entry + risk
        tp = entry - TP_R * risk

    # ============================================================
    # IMPORTANT:
    # Entry candle is NOT scanned.
    # Exit scan starts at i+1.
    # ============================================================

    for j in range(
        i + 1,
        i + HOLD + 1
    ):

        x = d.iloc[j]

        if side == "LONG":

            # SL-first if both touched
            if x["low"] <= sl:
                return {
                    "status": "TRADED",
                    "r": -1.0 - COST
                }

            if x["high"] >= tp:
                return {
                    "status": "TRADED",
                    "r": TP_R - COST
                }

        else:

            # SL-first if both touched
            if x["high"] >= sl:
                return {
                    "status": "TRADED",
                    "r": -1.0 - COST
                }

            if x["low"] <= tp:
                return {
                    "status": "TRADED",
                    "r": TP_R - COST
                }

    # TIMEOUT
    return {
        "status": "TRADED",
        "r": -COST
    }


# ================================================================
# LOAD ALL DATA
# ================================================================

print("=" * 72)
print("V4 — CANDIDATE 8")
print("5-FOLD WALK-FORWARD VALIDATION")
print("VOLUME BREAKOUT + RETEST")
print("=" * 72)

datasets = {}

for symbol in SYMBOLS:

    d = fetch_history(symbol)

    if len(d) < 500:

        print(
            symbol,
            "-> INSUFFICIENT"
        )

        continue

    d = prepare(d)

    datasets[symbol] = d

    days = (
        d["time"].iloc[-1]
        - d["time"].iloc[0]
    ) / 86400

    print(
        symbol,
        "| candles =", len(d),
        "| days =", round(days, 1)
    )


print()
print(
    "VALID SYMBOLS =",
    len(datasets)
)


# ================================================================
# COMMON TIMESTAMP AUDIT
# ================================================================

common = None

for d in datasets.values():

    ts = set(
        d["time"].astype(np.int64)
    )

    if common is None:
        common = ts
    else:
        common &= ts

common = sorted(common)

if len(common) < 1000:
    raise RuntimeError(
        "COMMON TIMESTAMP DATA TOO SMALL"
    )

print(
    "COMMON TIMESTAMPS =",
    len(common)
)

print(
    "COMMON START =",
    pd.to_datetime(
        common[0],
        unit="s",
        utc=True
    )
)

print(
    "COMMON END =",
    pd.to_datetime(
        common[-1],
        unit="s",
        utc=True
    )
)


# ================================================================
# GAP AUDIT
# ================================================================

total_gaps = 0
gap_symbols = []

for symbol, d in datasets.items():

    t = d["time"].to_numpy()

    diffs = np.diff(t)

    gaps = int(
        np.sum(
            diffs != INTERVAL_SEC
        )
    )

    if gaps > 0:

        gap_symbols.append(
            (symbol, gaps)
        )

        total_gaps += gaps


print(
    "GAP SYMBOLS =",
    len(gap_symbols)
)

print(
    "TOTAL GAPS =",
    total_gaps
)


# ================================================================
# EVENT CACHE
# ================================================================

event_cache = {}

total_events = 0

for symbol, d in datasets.items():

    events = detect_events(d)

    event_cache[symbol] = events

    total_events += len(events)


print()
print(
    "TOTAL DETECTED EVENTS =",
    total_events
)


# ================================================================
# PORTFOLIO METRICS
# ================================================================

def calculate_metrics(arr):

    if len(arr) == 0:

        return {
            "n": 0,
            "wr": 0.0,
            "exp": 0.0,
            "pf": 0.0,
            "total": 0.0,
            "maxdd": 0.0
        }

    arr = np.asarray(
        arr,
        dtype=float
    )

    wins = arr[arr > 0]
    losses = arr[arr < 0]

    gross_profit = (
        wins.sum()
        if len(wins)
        else 0.0
    )

    gross_loss = (
        abs(losses.sum())
        if len(losses)
        else 0.0
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = np.cumsum(arr)

    running_peak = np.maximum.accumulate(
        np.insert(
            equity,
            0,
            0.0
        )
    )[1:]

    drawdown = (
        equity - running_peak
    )

    maxdd = (
        float(drawdown.min())
        if len(drawdown)
        else 0.0
    )

    return {
        "n": len(arr),
        "wr": float(
            np.mean(arr > 0)
        ),
        "exp": float(
            arr.mean()
        ),
        "pf": float(pf),
        "total": float(
            arr.sum()
        ),
        "maxdd": maxdd
    }


# ================================================================
# RUN ONE WALK-FORWARD FOLD
# ================================================================

def run_fold(
    fold_id,
    start_ratio,
    end_ratio
):

    start_pos = int(
        len(common) * start_ratio
    )

    end_pos = int(
        len(common) * end_ratio
    )

    fold_start = common[start_pos]
    fold_end = common[end_pos - 1]

    returns = []

    events = 0
    open_end = 0
    ambiguous = 0

    long_count = 0
    short_count = 0

    for symbol, d in datasets.items():

        for event in event_cache[symbol]:

            t = event["time"]

            if (
                t < fold_start
                or t > fold_end
            ):
                continue

            events += 1

            if event["side"] == "LONG":
                long_count += 1
            else:
                short_count += 1

            result = simulate_trade(
                d,
                event
            )

            if result is None:

                ambiguous += 1
                continue

            if (
                result["status"]
                == "OPEN_AT_DATASET_END"
            ):

                open_end += 1
                continue

            returns.append(
                result["r"]
            )

    metrics = calculate_metrics(
        returns
    )

    metrics.update({
        "fold": fold_id,
        "start": fold_start,
        "end": fold_end,
        "events": events,
        "open_end": open_end,
        "ambiguous": ambiguous,
        "long": long_count,
        "short": short_count
    })

    return metrics


# ================================================================
# RUN ALL 5 FOLDS
# ================================================================

fold_results = []

for k, (a, b) in enumerate(
    FOLDS,
    start=1
):

    result = run_fold(
        k,
        a,
        b
    )

    fold_results.append(
        result
    )

    print()
    print("=" * 72)
    print(
        f"FOLD {k} "
        f"({int(a*100)}%-{int(b*100)}%)"
    )
    print("=" * 72)

    print(
        "START =",
        pd.to_datetime(
            result["start"],
            unit="s",
            utc=True
        )
    )

    print(
        "END =",
        pd.to_datetime(
            result["end"],
            unit="s",
            utc=True
        )
    )

    print(
        "EVENTS =",
        result["events"]
    )

    print(
        "TRADED =",
        result["n"]
    )

    print(
        "OPEN_AT_DATASET_END =",
        result["open_end"]
    )

    print(
        "AMBIGUOUS =",
        result["ambiguous"]
    )

    print(
        "LONG =",
        result["long"],
        "| SHORT =",
        result["short"]
    )

    print(
        "WR =",
        round(result["wr"], 4)
    )

    print(
        "NetExp =",
        f'{result["exp"]:+.4f}R'
    )

    print(
        "NetPF =",
        round(result["pf"], 3)
    )

    print(
        "NetTotalR =",
        f'{result["total"]:+.2f}R'
    )

    print(
        "MaxDD =",
        f'{result["maxdd"]:+.2f}R'
    )


# ================================================================
# AGGREGATE WALK-FORWARD
# ================================================================

all_returns = []

for symbol, d in datasets.items():

    for event in event_cache[symbol]:

        t = event["time"]

        # Only events inside the five WF windows
        if t < common[int(len(common) * 0.50)]:
            continue

        result = simulate_trade(
            d,
            event
        )

        if result is None:
            continue

        if (
            result["status"]
            == "OPEN_AT_DATASET_END"
        ):
            continue

        all_returns.append(
            result["r"]
        )


aggregate = calculate_metrics(
    all_returns
)


# ================================================================
# WF FOLD QUALITY
# ================================================================

positive_folds = sum(
    1
    for x in fold_results
    if (
        x["exp"] > 0
        and x["pf"] > 1
    )
)


# ================================================================
# FINAL REPORT
# ================================================================

print()
print("=" * 72)
print("WALK-FORWARD AGGREGATE")
print("=" * 72)

print(
    "TRADES =",
    aggregate["n"]
)

print(
    "WR =",
    round(
        aggregate["wr"],
        4
    )
)

print(
    "NetExp =",
    f'{aggregate["exp"]:+.4f}R'
)

print(
    "NetPF =",
    round(
        aggregate["pf"],
        3
    )
)

print(
    "NetTotalR =",
    f'{aggregate["total"]:+.2f}R'
)

print(
    "MaxDD =",
    f'{aggregate["maxdd"]:+.2f}R'
)

print(
    "POSITIVE FOLDS =",
    f"{positive_folds}/5"
)


# ================================================================
# PREDECLARED WF GATE
#
# PASS:
# 1) Aggregate NetExp > 0
# 2) Aggregate NetPF > 1
# 3) At least 4/5 folds positive
#
# Otherwise FAIL.
# No parameter changes allowed.
# ================================================================

wf_pass = (
    aggregate["exp"] > 0
    and aggregate["pf"] > 1
    and positive_folds >= 4
)

print()
print("=" * 72)

if wf_pass:
    print(
        "WF STATUS = PASS"
    )
else:
    print(
        "WF STATUS = FAIL"
    )

print("=" * 72)


# ================================================================
# INTEGRITY AUDIT
# ================================================================

print()
print("=" * 72)
print("WALK-FORWARD INTEGRITY AUDIT")
print("=" * 72)

print(
    "VALID SYMBOLS =",
    len(datasets)
)

print(
    "COMMON TIMESTAMPS =",
    len(common)
)

print(
    "COMMON TIMESTAMP AUDIT =",
    "PASS" if total_gaps == 0 else "CHECK"
)

print(
    "CAUSAL SIGNAL DETECTION = TRUE"
)

print(
    "PREVIOUS 20 RANGE SHIFTED = TRUE"
)

print(
    "ENTRY CANDLE EXIT SCAN = FALSE"
)

print(
    "SAME CANDLE SL FIRST = TRUE"
)

print(
    "NO PARAMETER OPTIMIZATION = TRUE"
)

print(
    "NO EXTRA FILTERS = TRUE"
)

print(
    "NO OVERLAP LOCK = TRUE"
)

print(
    "SL_ATR =",
    SL_ATR
)

print(
    "TP_R =",
    TP_R
)

print(
    "HOLD =",
    HOLD
)

print(
    "RETEST_BARS =",
    RETEST_BARS
)

print(
    "VOLUME_MULT =",
    VOLUME_MULT
)

print(
    "TOTAL COST R =",
    COST
)

print(
    "WF FOLDS = 5"
)

print("=" * 72)
print("END")
print("=" * 72)
