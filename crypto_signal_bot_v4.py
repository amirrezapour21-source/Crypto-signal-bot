# ================================================================
# V4 — CANDIDATE 9 — MACD ZERO-LINE TREND CONTINUATION
# 365D DISCOVERY SCREEN
#
# NEW STRATEGY FAMILY
# MACD zero-line momentum + ADX regime + EMA200 HTF direction
#
# NO OPTIMIZATION
# NO EXTRA FILTERS
# NO OVERLAP LOCK
# FULLY CAUSAL
# ================================================================

import time
import json
import urllib.request
import numpy as np
import pandas as pd


# ================================================================
# UNIVERSE
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

INTERVAL = 14400
DAYS = 365
PAGE = 1500

EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 200

ADX_PERIOD = 14
ADX_MIN = 20.0

ATR_PERIOD = 20

# Volatility / activity confirmation
ATR_MEDIAN_PERIOD = 20
VOLUME_MEDIAN_PERIOD = 20

SL_ATR = 1.25
HOLD = 30

TP_LEVELS = [
    1.0,
    1.5,
    2.0,
    3.0
]

# Frozen trading cost
FEE = 0.001
SLIPPAGE = 0.002
COST = FEE + SLIPPAGE


# ================================================================
# FETCH
# ================================================================

def fetch_history(symbol):

    now = int(time.time())
    start = now - DAYS * 86400

    out = []
    cur = start

    while cur < now:

        end = min(
            cur + (PAGE - 1) * INTERVAL,
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

                    out.extend(
                        payload["data"]
                    )

                    success = True
                    break

            except Exception:

                if attempt < 2:
                    time.sleep(1 + attempt)

        if not success:
            return pd.DataFrame()

        cur = end + INTERVAL
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
        .dropna(
            subset=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Remove incomplete current candle
    cutoff = (
        now // INTERVAL
    ) * INTERVAL

    d = d[
        d["time"] < cutoff
    ].reset_index(drop=True)

    return d


# ================================================================
# INDICATORS
# ================================================================

def prepare(d):

    d = d.copy()

    # ------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------

    d["ema12"] = (
        d["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    d["ema26"] = (
        d["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    d["ema200"] = (
        d["close"]
        .ewm(
            span=EMA_TREND,
            adjust=False
        )
        .mean()
    )

    # ------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------

    d["macd"] = (
        d["ema12"] -
        d["ema26"]
    )

    d["macd_signal"] = (
        d["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    d["hist"] = (
        d["macd"] -
        d["macd_signal"]
    )

    # ------------------------------------------------------------
    # TRUE RANGE
    # ------------------------------------------------------------

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (
                d["high"] -
                d["close"].shift(1)
            ).abs(),
            (
                d["low"] -
                d["close"].shift(1)
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    # Wilder ATR
    d["atr"] = (
        tr
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    # ------------------------------------------------------------
    # ATR MEDIAN
    # ------------------------------------------------------------

    d["atr_median"] = (
        d["atr"]
        .rolling(
            ATR_MEDIAN_PERIOD,
            min_periods=ATR_MEDIAN_PERIOD
        )
        .median()
    )

    # ------------------------------------------------------------
    # VOLUME MEDIAN
    # ------------------------------------------------------------

    d["volume_median"] = (
        d["volume"]
        .rolling(
            VOLUME_MEDIAN_PERIOD,
            min_periods=VOLUME_MEDIAN_PERIOD
        )
        .median()
    )

    # ------------------------------------------------------------
    # ADX / DI
    # Wilder implementation
    # ------------------------------------------------------------

    up_move = (
        d["high"] -
        d["high"].shift(1)
    )

    down_move = (
        d["low"].shift(1) -
        d["low"]
    )

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0.0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=d.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=d.index
    )

    atr_adx = (
        tr
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        )
        .mean()
    )

    plus_smoothed = (
        plus_dm
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        )
        .mean()
    )

    minus_smoothed = (
        minus_dm
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        )
        .mean()
    )

    d["plus_di"] = (
        100 *
        plus_smoothed /
        atr_adx
    )

    d["minus_di"] = (
        100 *
        minus_smoothed /
        atr_adx
    )

    di_sum = (
        d["plus_di"] +
        d["minus_di"]
    )

    d["dx"] = (
        100 *
        (
            d["plus_di"] -
            d["minus_di"]
        ).abs() /
        di_sum.replace(
            0,
            np.nan
        )
    )

    d["adx"] = (
        d["dx"]
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        )
        .mean()
    )

    return d


# ================================================================
# SIGNAL DETECTION
# ================================================================

def detect_events(d):

    events = []

    # EMA200 + ADX require warm-up
    START = 250

    for i in range(
        START,
        len(d) - HOLD
    ):

        r = d.iloc[i]
        p = d.iloc[i - 1]

        required = [
            r["ema12"],
            r["ema26"],
            r["ema200"],
            r["hist"],
            p["hist"],
            r["atr"],
            r["atr_median"],
            r["volume_median"],
            r["adx"],
            r["plus_di"],
            r["minus_di"]
        ]

        if any(
            pd.isna(x)
            for x in required
        ):
            continue

        # ========================================================
        # LONG
        #
        # 1. Price above EMA200
        # 2. EMA12 > EMA26
        # 3. MACD histogram crosses from <=0 to >0
        # 4. ADX >= 20
        # 5. +DI > -DI
        # 6. ATR >= rolling ATR median
        # 7. Volume >= rolling volume median
        # ========================================================

        long_signal = (
            r["close"] > r["ema200"]
            and
            r["ema12"] > r["ema26"]
            and
            p["hist"] <= 0
            and
            r["hist"] > 0
            and
            r["adx"] >= ADX_MIN
            and
            r["plus_di"] > r["minus_di"]
            and
            r["atr"] >= r["atr_median"]
            and
            r["volume"] >= r["volume_median"]
        )

        if long_signal:

            events.append({
                "idx": i,
                "time": int(r["time"]),
                "side": "LONG"
            })

            continue

        # ========================================================
        # SHORT
        #
        # 1. Price below EMA200
        # 2. EMA12 < EMA26
        # 3. MACD histogram crosses from >=0 to <0
        # 4. ADX >= 20
        # 5. -DI > +DI
        # 6. ATR >= rolling ATR median
        # 7. Volume >= rolling volume median
        # ========================================================

        short_signal = (
            r["close"] < r["ema200"]
            and
            r["ema12"] < r["ema26"]
            and
            p["hist"] >= 0
            and
            r["hist"] < 0
            and
            r["adx"] >= ADX_MIN
            and
            r["minus_di"] > r["plus_di"]
            and
            r["atr"] >= r["atr_median"]
            and
            r["volume"] >= r["volume_median"]
        )

        if short_signal:

            events.append({
                "idx": i,
                "time": int(r["time"]),
                "side": "SHORT"
            })

    return events


# ================================================================
# TRADE SIMULATION
# ================================================================

def simulate_trade(
    d,
    event,
    tp_r
):

    i = event["idx"]
    side = event["side"]

    if i + HOLD >= len(d):
        return None

    entry = float(
        d.iloc[i]["close"]
    )

    atr = float(
        d.iloc[i]["atr"]
    )

    if (
        not np.isfinite(entry)
        or
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        return None

    risk = SL_ATR * atr

    if side == "LONG":

        sl = entry - risk
        tp = entry + tp_r * risk

    else:

        sl = entry + risk
        tp = entry - tp_r * risk

    # Entry candle is NOT scanned.
    for j in range(
        i + 1,
        i + HOLD + 1
    ):

        x = d.iloc[j]

        if side == "LONG":

            # Same-candle SL-first
            if x["low"] <= sl:

                return -1.0 - COST

            if x["high"] >= tp:

                return tp_r - COST

        else:

            # Same-candle SL-first
            if x["high"] >= sl:

                return -1.0 - COST

            if x["low"] <= tp:

                return tp_r - COST

    # TIMEOUT
    return -COST


# ================================================================
# METRICS
# ================================================================

def metrics(values):

    if not values:

        return {
            "n": 0,
            "wr": 0,
            "exp": 0,
            "pf": 0,
            "total": 0,
            "maxdd": 0
        }

    a = np.asarray(
        values,
        dtype=float
    )

    wins = a[a > 0]
    losses = a[a < 0]

    gp = (
        wins.sum()
        if len(wins)
        else 0.0
    )

    gl = (
        abs(losses.sum())
        if len(losses)
        else 0.0
    )

    pf = (
        gp / gl
        if gl > 0
        else np.inf
    )

    equity = np.cumsum(a)

    peak = np.maximum.accumulate(
        np.insert(
            equity,
            0,
            0.0
        )
    )[1:]

    dd = equity - peak

    return {
        "n": len(a),
        "wr": float(
            np.mean(a > 0)
        ),
        "exp": float(
            a.mean()
        ),
        "pf": float(pf),
        "total": float(
            a.sum()
        ),
        "maxdd": float(
            dd.min()
        )
    }


# ================================================================
# MAIN
# ================================================================

print("=" * 70)
print("V4 — CANDIDATE 9")
print("MACD ZERO-LINE TREND CONTINUATION")
print("365D DISCOVERY")
print("=" * 70)

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
        d["time"].iloc[-1] -
        d["time"].iloc[0]
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
# DETECT EVENTS
# ================================================================

all_events = []

for symbol, d in datasets.items():

    events = detect_events(d)

    for e in events:

        e = e.copy()
        e["symbol"] = symbol

        all_events.append(e)


print(
    "TOTAL EVENTS =",
    len(all_events)
)

print(
    "LONG EVENTS =",
    sum(
        e["side"] == "LONG"
        for e in all_events
    )
)

print(
    "SHORT EVENTS =",
    sum(
        e["side"] == "SHORT"
        for e in all_events
    )
)


# ================================================================
# EVALUATE ALL TP SCENARIOS
# ================================================================

for tp_r in TP_LEVELS:

    results = []

    for event in all_events:

        d = datasets[
            event["symbol"]
        ]

        r = simulate_trade(
            d,
            event,
            tp_r
        )

        if r is not None:
            results.append(r)

    m = metrics(results)

    print()
    print("=" * 70)
    print(
        f"TP = {tp_r}R"
    )
    print("=" * 70)

    print(
        "TRADES =",
        m["n"]
    )

    print(
        "WR =",
        round(
            m["wr"],
            4
        )
    )

    print(
        "NetExp =",
        f'{m["exp"]:+.4f}R'
    )

    print(
        "NetPF =",
        round(
            m["pf"],
            3
        )
    )

    print(
        "NetTotalR =",
        f'{m["total"]:+.2f}R'
    )

    print(
        "MaxDD =",
        f'{m["maxdd"]:+.2f}R'
    )


# ================================================================
# DIRECTION BREAKDOWN — TP2R
# INFORMATIONAL ONLY
# ================================================================

print()
print("=" * 70)
print("TP2R DIRECTION BREAKDOWN")
print("=" * 70)

for side in [
    "LONG",
    "SHORT"
]:

    values = []

    for event in all_events:

        if event["side"] != side:
            continue

        d = datasets[
            event["symbol"]
        ]

        r = simulate_trade(
            d,
            event,
            2.0
        )

        if r is not None:
            values.append(r)

    m = metrics(values)

    print()
    print(side)

    print(
        "TRADES =",
        m["n"]
    )

    print(
        "WR =",
        round(
            m["wr"],
            4
        )
    )

    print(
        "NetExp =",
        f'{m["exp"]:+.4f}R'
    )

    print(
        "NetPF =",
        round(
            m["pf"],
            3
        )
    )

    print(
        "NetTotalR =",
        f'{m["total"]:+.2f}R'
    )

    print(
        "MaxDD =",
        f'{m["maxdd"]:+.2f}R'
    )


# ================================================================
# DATA / CAUSAL AUDIT
# ================================================================

print()
print("=" * 70)
print("DISCOVERY INTEGRITY")
print("=" * 70)

print(
    "CAUSAL SIGNAL DETECTION = TRUE"
)

print(
    "MACD CROSS USES CURRENT + PREVIOUS CLOSED CANDLE = TRUE"
)

print(
    "EMA200 = CAUSAL"
)

print(
    "ADX = CAUSAL"
)

print(
    "ATR = CAUSAL"
)

print(
    "VOLUME MEDIAN = CAUSAL"
)

print(
    "ATR MEDIAN = CAUSAL"
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
    "TOTAL COST R =",
    COST
)

print("=" * 70)
print("END")
print("=" * 70)
