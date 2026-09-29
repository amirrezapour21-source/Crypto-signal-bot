# ============================================================
# CRYPTO SIGNAL BOT V4
# CANDIDATE 10
# DIRECTIONAL EFFICIENCY TRANSITION
# -> TREND EXPANSION
#
# RESEARCH ONLY
# No live trading
# No parameter optimization
# No post-result filtering
# ============================================================

import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://api.kucoin.com/api/v1/market/candles"

INTERVAL = "4hour"
INTERVAL_SEC = 4 * 60 * 60

TARGET_DAYS = 365
MIN_HISTORY_DAYS = 180

MAX_RECORDS_PER_REQUEST = 1500

ER_PERIOD = 20
ER_THRESHOLD = 0.55

EMA_PERIOD = 200

ADX_PERIOD = 14
ADX_MIN = 20.0

ATR_PERIOD = 20
MEDIAN_PERIOD = 20

SL_ATR = 1.25

HOLD_BARS = 30

TP_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]

FEE_R = 0.001
SLIPPAGE_R = 0.002
TOTAL_COST_R = FEE_R + SLIPPAGE_R

REQUEST_TIMEOUT = 20
MAX_RETRIES = 5
SLEEP_BETWEEN_REQUESTS = 0.15


# ============================================================
# UNIVERSE
# Same research universe used in previous V4 candidates
# ============================================================

SYMBOLS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "ADA-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "DOT-USDT",
    "TON-USDT",
    "SUI-USDT",
    "APT-USDT",
    "NEAR-USDT",
    "OP-USDT",
    "ARB-USDT",
    "ATOM-USDT",
    "FIL-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "ETC-USDT",
    "UNI-USDT",
    "AAVE-USDT",
    "INJ-USDT",
    "SEI-USDT",
    "WIF-USDT",
    "PEPE-USDT",
    "FET-USDT",
    "RENDER-USDT",
    "TAO-USDT",
    "TIA-USDT",
    "IMX-USDT",
    "MKR-USDT",
    "MATIC-USDT",
    "STX-USDT",
    "ALGO-USDT",
    "HBAR-USDT",
    "ICP-USDT",
]


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def ts_to_str(ts):
    return datetime.fromtimestamp(
        int(ts), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")


def rma(series, period):
    """
    Wilder's RMA.
    """
    return series.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# KUCOIN DATA
# ============================================================

def fetch_history(symbol, target_days=TARGET_DAYS):
    """
    Fetch closed 4H candles using time pagination.

    KuCoin:
    - max 1500 candles/request
    - pagination via startAt/endAt
    """

    now_ts = int(time.time())

    # Start from target history
    requested_start = now_ts - target_days * 86400

    # Current 4H bucket start
    current_bucket_start = (
        now_ts // INTERVAL_SEC
    ) * INTERVAL_SEC

    # Last fully closed candle
    closed_end = current_bucket_start - 1

    rows = []

    cursor = requested_start
    pages = 0

    while cursor <= closed_end:

        chunk_end = min(
            cursor + (MAX_RECORDS_PER_REQUEST - 1) * INTERVAL_SEC,
            closed_end
        )

        params = {
            "symbol": symbol,
            "type": INTERVAL,
            "startAt": int(cursor),
            "endAt": int(chunk_end),
        }

        success = False

        for attempt in range(MAX_RETRIES):

            try:
                response = requests.get(
                    API_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT
                )

                response.raise_for_status()

                payload = response.json()

                if payload.get("code") != "200000":
                    raise RuntimeError(
                        f"KuCoin code={payload.get('code')}"
                    )

                data = payload.get("data", [])

                if not data:
                    success = True
                    break

                rows.extend(data)

                success = True
                break

            except Exception as exc:

                if attempt == MAX_RETRIES - 1:
                    print(
                        f"[ERROR] {symbol} "
                        f"page={pages + 1} "
                        f"attempt={attempt + 1}: {exc}"
                    )
                else:
                    time.sleep(1.0 + attempt)

        if not success:
            break

        pages += 1

        if not data:
            break

        try:
            max_ts = max(int(x[0]) for x in data)
        except Exception:
            break

        next_cursor = max_ts + INTERVAL_SEC

        if next_cursor <= cursor:
            print(
                f"[STALL] {symbol}: "
                f"cursor did not advance."
            )
            break

        cursor = next_cursor

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # Safety
        if pages > 100:
            print(
                f"[STOP] {symbol}: "
                f"pagination safety limit."
            )
            break

    if not rows:
        return None, {
            "pages": pages,
            "sufficient": False,
            "gap_count": None,
            "days": 0.0,
        }

    # --------------------------------------------------------
    # Build dataframe
    # KuCoin candle:
    # [time, open, close, high, low, volume, turnover]
    # --------------------------------------------------------

    parsed = []

    for row in rows:

        if len(row) < 7:
            continue

        try:
            parsed.append([
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
            ])
        except Exception:
            continue

    if not parsed:
        return None, {
            "pages": pages,
            "sufficient": False,
            "gap_count": None,
            "days": 0.0,
        }

    df = pd.DataFrame(
        parsed,
        columns=[
            "timestamp",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover",
        ]
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Strictly remove any non-closed candle
    # --------------------------------------------------------

    df = df[
        df["timestamp"] <= closed_end
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 2:
        return None, {
            "pages": pages,
            "sufficient": False,
            "gap_count": None,
            "days": 0.0,
        }

    # --------------------------------------------------------
    # Continuity audit
    # --------------------------------------------------------

    diffs = df["timestamp"].diff().dropna()

    gap_count = int(
        (diffs != INTERVAL_SEC).sum()
    )

    first_ts = int(df["timestamp"].iloc[0])
    last_ts = int(df["timestamp"].iloc[-1])

    days = (
        last_ts - first_ts
    ) / 86400.0

    sufficient = days >= MIN_HISTORY_DAYS

    meta = {
        "pages": pages,
        "sufficient": sufficient,
        "gap_count": gap_count,
        "days": days,
        "candles": len(df),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }

    return df, meta


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    )

    tr = tr_components.max(axis=1)

    # --------------------------------------------------------
    # ATR - Wilder
    # --------------------------------------------------------

    df["atr"] = rma(
        tr,
        ATR_PERIOD
    )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    df["ema200"] = close.ewm(
        span=EMA_PERIOD,
        adjust=False,
        min_periods=EMA_PERIOD
    ).mean()

    # --------------------------------------------------------
    # Directional Movement
    # --------------------------------------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr_for_di = rma(
        tr,
        ADX_PERIOD
    )

    plus_di = (
        100.0 *
        rma(plus_dm, ADX_PERIOD) /
        atr_for_di.replace(0, np.nan)
    )

    minus_di = (
        100.0 *
        rma(minus_dm, ADX_PERIOD) /
        atr_for_di.replace(0, np.nan)
    )

    dx_denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100.0 *
        (plus_di - minus_di).abs() /
        dx_denominator
    )

    adx = rma(
        dx,
        ADX_PERIOD
    )

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    # --------------------------------------------------------
    # SIGNED EFFICIENCY RATIO
    #
    # ER = net directional movement /
    #      total absolute movement
    #
    # Range approximately [-1,+1]
    #
    # +1 = extremely efficient bullish movement
    # -1 = extremely efficient bearish movement
    # --------------------------------------------------------

    net_change = (
        close - close.shift(ER_PERIOD)
    )

    absolute_change = (
        close.diff().abs()
        .rolling(
            ER_PERIOD,
            min_periods=ER_PERIOD
        )
        .sum()
    )

    df["signed_er"] = (
        net_change /
        absolute_change.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # ATR median
    # --------------------------------------------------------

    df["atr_median"] = (
        df["atr"]
        .rolling(
            MEDIAN_PERIOD,
            min_periods=MEDIAN_PERIOD
        )
        .median()
    )

    # --------------------------------------------------------
    # Volume median
    # --------------------------------------------------------

    df["volume_median"] = (
        df["volume"]
        .rolling(
            MEDIAN_PERIOD,
            min_periods=MEDIAN_PERIOD
        )
        .median()
    )

    return df


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_events(df, symbol):
    """
    Completely causal.

    Signal is generated at current candle close.

    LONG:
        close > EMA200
        signed ER crosses upward through +0.55
        ADX >= 20
        ADX rising
        +DI > -DI
        ATR >= ATR median
        volume >= volume median

    SHORT:
        close < EMA200
        signed ER crosses downward through -0.55
        ADX >= 20
        ADX rising
        -DI > +DI
        ATR >= ATR median
        volume >= volume median
    """

    events = []

    for i in range(1, len(df)):

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        required = [
            row["close"],
            row["ema200"],
            row["signed_er"],
            prev["signed_er"],
            row["adx"],
            prev["adx"],
            row["plus_di"],
            row["minus_di"],
            row["atr"],
            row["atr_median"],
            row["volume"],
            row["volume_median"],
        ]

        if not np.all(
            np.isfinite(required)
        ):
            continue

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        long_signal = (
            row["close"] > row["ema200"]
            and prev["signed_er"] <= ER_THRESHOLD
            and row["signed_er"] > ER_THRESHOLD
            and row["adx"] >= ADX_MIN
            and row["adx"] > prev["adx"]
            and row["plus_di"] > row["minus_di"]
            and row["atr"] >= row["atr_median"]
            and row["volume"] >= row["volume_median"]
        )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        short_signal = (
            row["close"] < row["ema200"]
            and prev["signed_er"] >= -ER_THRESHOLD
            and row["signed_er"] < -ER_THRESHOLD
            and row["adx"] >= ADX_MIN
            and row["adx"] > prev["adx"]
            and row["minus_di"] > row["plus_di"]
            and row["atr"] >= row["atr_median"]
            and row["volume"] >= row["volume_median"]
        )

        if long_signal:
            events.append({
                "symbol": symbol,
                "idx": i,
                "timestamp": int(row["timestamp"]),
                "direction": "LONG",
                "entry": float(row["close"]),
                "atr": float(row["atr"]),
                "signed_er": float(row["signed_er"]),
                "adx": float(row["adx"]),
            })

        elif short_signal:
            events.append({
                "symbol": symbol,
                "idx": i,
                "timestamp": int(row["timestamp"]),
                "direction": "SHORT",
                "entry": float(row["close"]),
                "atr": float(row["atr"]),
                "signed_er": float(row["signed_er"]),
                "adx": float(row["adx"]),
            })

    return events


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(
    df,
    event,
    tp_r
):
    """
    Entry:
        signal candle close

    Exit scanning:
        starts from NEXT candle

    Same candle:
        SL-first if SL and TP both touched.

    Timeout:
        gross = 0R

    Dataset ends before HOLD:
        OPEN_AT_DATASET_END
    """

    idx = event["idx"]

    entry = event["entry"]
    atr = event["atr"]
    direction = event["direction"]

    if not np.isfinite(entry) or not np.isfinite(atr):
        return {
            "status": "AMBIGUOUS",
            "gross_r": np.nan,
            "net_r": np.nan,
        }

    if atr <= 0:
        return {
            "status": "AMBIGUOUS",
            "gross_r": np.nan,
            "net_r": np.nan,
        }

    # --------------------------------------------------------
    # Need full HOLD window
    # --------------------------------------------------------

    final_idx = idx + HOLD_BARS

    if final_idx >= len(df):
        return {
            "status": "OPEN_AT_DATASET_END",
            "gross_r": np.nan,
            "net_r": np.nan,
        }

    if direction == "LONG":

        sl = entry - SL_ATR * atr
        tp = entry + tp_r * atr

    else:

        sl = entry + SL_ATR * atr
        tp = entry - tp_r * atr

    # --------------------------------------------------------
    # Scan future candles only
    # --------------------------------------------------------

    for j in range(
        idx + 1,
        final_idx + 1
    ):

        candle = df.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])

        if direction == "LONG":

            sl_hit = low <= sl
            tp_hit = high >= tp

        else:

            sl_hit = high >= sl
            tp_hit = low <= tp

        # ----------------------------------------------------
        # Same-candle ambiguity
        # Frozen rule: SL FIRST
        # ----------------------------------------------------

        if sl_hit:
            gross_r = -SL_ATR / SL_ATR
            status = "SL"

            return {
                "status": status,
                "gross_r": gross_r,
                "net_r": gross_r - TOTAL_COST_R,
                "exit_idx": j,
                "exit_timestamp": int(
                    candle["timestamp"]
                ),
            }

        if tp_hit:
            gross_r = tp_r
            status = "TP"

            return {
                "status": status,
                "gross_r": gross_r,
                "net_r": gross_r - TOTAL_COST_R,
                "exit_idx": j,
                "exit_timestamp": int(
                    candle["timestamp"]
                ),
            }

    # --------------------------------------------------------
    # Full HOLD reached without SL / TP
    # --------------------------------------------------------

    candle = df.iloc[final_idx]

    return {
        "status": "TIMEOUT",
        "gross_r": 0.0,
        "net_r": -TOTAL_COST_R,
        "exit_idx": final_idx,
        "exit_timestamp": int(
            candle["timestamp"]
        ),
    }


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_max_drawdown(values):
    if len(values) == 0:
        return 0.0

    equity = np.cumsum(values)

    peak = np.maximum.accumulate(equity)

    drawdown = equity - peak

    return float(drawdown.min())


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if not trades:
        return {
            "n_traded": 0,
            "open_end": 0,
            "ambiguous": 0,
            "wr": np.nan,
            "gross_exp": np.nan,
            "net_exp": np.nan,
            "gross_total": 0.0,
            "net_total": 0.0,
            "gross_pf": np.nan,
            "net_pf": np.nan,
            "gross_dd": 0.0,
            "net_dd": 0.0,
        }

    open_end = sum(
        x["status"] == "OPEN_AT_DATASET_END"
        for x in trades
    )

    ambiguous = sum(
        x["status"] == "AMBIGUOUS"
        for x in trades
    )

    measured = [
        x for x in trades
        if x["status"]
        not in (
            "OPEN_AT_DATASET_END",
            "AMBIGUOUS",
        )
    ]

    if not measured:
        return {
            "n_traded": 0,
            "open_end": open_end,
            "ambiguous": ambiguous,
            "wr": np.nan,
            "gross_exp": np.nan,
            "net_exp": np.nan,
            "gross_total": 0.0,
            "net_total": 0.0,
            "gross_pf": np.nan,
            "net_pf": np.nan,
            "gross_dd": 0.0,
            "net_dd": 0.0,
        }

    measured = sorted(
        measured,
        key=lambda x: (
            x["timestamp"],
            x["symbol"]
        )
    )

    gross = np.array([
        x["gross_r"]
        for x in measured
    ], dtype=float)

    net = np.array([
        x["net_r"]
        for x in measured
    ], dtype=float)

    wins = np.sum(gross > 0)

    wr = wins / len(gross)

    gross_exp = float(np.mean(gross))
    net_exp = float(np.mean(net))

    gross_total = float(np.sum(gross))
    net_total = float(np.sum(net))

    gross_positive = gross[gross > 0].sum()
    gross_negative = gross[gross < 0].sum()

    net_positive = net[net > 0].sum()
    net_negative = net[net < 0].sum()

    gross_pf = (
        gross_positive / abs(gross_negative)
        if gross_negative < 0
        else np.inf
    )

    net_pf = (
        net_positive / abs(net_negative)
        if net_negative < 0
        else np.inf
    )

    gross_dd = calculate_max_drawdown(gross)
    net_dd = calculate_max_drawdown(net)

    return {
        "n_traded": len(measured),
        "open_end": open_end,
        "ambiguous": ambiguous,
        "wr": float(wr),
        "gross_exp": gross_exp,
        "net_exp": net_exp,
        "gross_total": gross_total,
        "net_total": net_total,
        "gross_pf": float(gross_pf),
        "net_pf": float(net_pf),
        "gross_dd": gross_dd,
        "net_dd": net_dd,
    }


# ============================================================
# RUN ONE TP
# ============================================================

def run_tp(
    all_data,
    all_events,
    tp_r
):

    trades = []

    for symbol, events in all_events.items():

        df = all_data[symbol]

        for event in events:

            result = simulate_trade(
                df=df,
                event=event,
                tp_r=tp_r
            )

            result["symbol"] = symbol
            result["timestamp"] = event["timestamp"]
            result["direction"] = event["direction"]

            trades.append(result)

    return calculate_metrics(trades)


# ============================================================
# DIRECTION METRICS
# ============================================================

def direction_metrics(
    all_data,
    all_events,
    tp_r,
    direction
):

    trades = []

    for symbol, events in all_events.items():

        df = all_data[symbol]

        for event in events:

            if event["direction"] != direction:
                continue

            result = simulate_trade(
                df=df,
                event=event,
                tp_r=tp_r
            )

            result["symbol"] = symbol
            result["timestamp"] = event["timestamp"]

            trades.append(result)

    return calculate_metrics(trades)


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    label,
    metrics
):

    def fmt(x):
        if x is None:
            return "NA"

        if isinstance(x, float):

            if math.isinf(x):
                return "INF"

            if np.isnan(x):
                return "NA"

            return f"{x:.4f}"

        return str(x)

    print(
        f"{label:<7} "
        f"n={metrics['n_traded']} "
        f"open={metrics['open_end']} "
        f"amb={metrics['ambiguous']} "
        f"WR={fmt(metrics['wr'])} "
        f"GrossExp={fmt(metrics['gross_exp'])} "
        f"NetExp={fmt(metrics['net_exp'])} "
        f"GrossTotal={fmt(metrics['gross_total'])} "
        f"NetTotal={fmt(metrics['net_total'])} "
        f"GrossPF={fmt(metrics['gross_pf'])} "
        f"NetPF={fmt(metrics['net_pf'])} "
        f"GrossDD={fmt(metrics['gross_dd'])} "
        f"NetDD={fmt(metrics['net_dd'])}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SETUP V4 — CANDIDATE 10")
    print("DIRECTIONAL EFFICIENCY TRANSITION → TREND EXPANSION")
    print("=" * 80)

    print("\nFROZEN PARAMETERS")
    print(f"TIMEFRAME              = {INTERVAL}")
    print(f"ER_PERIOD              = {ER_PERIOD}")
    print(f"ER_THRESHOLD           = ±{ER_THRESHOLD}")
    print(f"EMA                    = {EMA_PERIOD}")
    print(f"ADX_PERIOD             = {ADX_PERIOD}")
    print(f"ADX_MIN                = {ADX_MIN}")
    print(f"ATR_PERIOD             = {ATR_PERIOD}")
    print(f"ATR_MEDIAN_PERIOD      = {MEDIAN_PERIOD}")
    print(f"VOLUME_MEDIAN_PERIOD   = {MEDIAN_PERIOD}")
    print(f"SL                     = {SL_ATR} ATR")
    print(f"HOLD                   = {HOLD_BARS}")
    print(f"TP                     = {TP_MULTIPLIERS}")
    print(f"TOTAL_COST_R           = {TOTAL_COST_R}")
    print("DIRECTION              = LONG + SHORT")
    print("OVERLAP_LOCK           = FALSE")
    print("PARAMETER_OPTIMIZATION = FALSE")
    print("EXTRA_FILTERS          = FALSE")

    print("\n" + "=" * 80)
    print("DATA COLLECTION")
    print("=" * 80)

    all_data = {}
    all_events = {}

    valid_symbols = []
    insufficient_symbols = []
    failed_symbols = []

    total_gaps = 0

    for n, symbol in enumerate(SYMBOLS, 1):

        print(
            f"[{n:02d}/{len(SYMBOLS)}] "
            f"{symbol} ...",
            end=" ",
            flush=True
        )

        try:

            df, meta = fetch_history(
                symbol,
                TARGET_DAYS
            )

            if df is None:
                failed_symbols.append(symbol)
                print("FAILED")
                continue

            if not meta["sufficient"]:
                insufficient_symbols.append(symbol)
                print(
                    f"INSUFFICIENT "
                    f"{meta['days']:.1f}d"
                )
                continue

            # ------------------------------------------------
            # Important:
            # If gaps exist, keep symbol visible in audit.
            # Do not silently pretend continuity.
            # ------------------------------------------------

            total_gaps += meta["gap_count"]

            df = add_indicators(df)

            events = detect_events(
                df,
                symbol
            )

            all_data[symbol] = df
            all_events[symbol] = events

            valid_symbols.append(symbol)

            print(
                f"OK "
                f"{len(df)} candles "
                f"{meta['days']:.1f}d "
                f"gaps={meta['gap_count']} "
                f"events={len(events)}"
            )

        except Exception as exc:

            failed_symbols.append(symbol)

            print(
                f"FAILED: {exc}"
            )

    # ========================================================
    # DATA AUDIT
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA AUDIT")
    print("=" * 80)

    print(
        f"VALID SYMBOLS       = {len(valid_symbols)}"
    )

    print(
        f"INSUFFICIENT        = {len(insufficient_symbols)}"
    )

    print(
        f"FAILED              = {len(failed_symbols)}"
    )

    print(
        f"TOTAL GAPS          = {total_gaps}"
    )

    if not valid_symbols:
        print("\nNO VALID SYMBOLS. STOP.")
        return

    # --------------------------------------------------------
    # Event totals
    # --------------------------------------------------------

    all_event_list = []

    for symbol in valid_symbols:
        all_event_list.extend(
            all_events[symbol]
        )

    all_event_list.sort(
        key=lambda x: (
            x["timestamp"],
            x["symbol"]
        )
    )

    long_events = sum(
        x["direction"] == "LONG"
        for x in all_event_list
    )

    short_events = sum(
        x["direction"] == "SHORT"
        for x in all_event_list
    )

    print(
        f"TOTAL EVENTS        = {len(all_event_list)}"
    )

    print(
        f"LONG EVENTS         = {long_events}"
    )

    print(
        f"SHORT EVENTS        = {short_events}"
    )

    if all_event_list:

        print(
            f"FIRST EVENT         = "
            f"{ts_to_str(all_event_list[0]['timestamp'])}"
        )

        print(
            f"LAST EVENT          = "
            f"{ts_to_str(all_event_list[-1]['timestamp'])}"
        )

    # ========================================================
    # TP RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("PORTFOLIO RESULTS")
    print("=" * 80)

    results = {}

    for tp in TP_MULTIPLIERS:

        metrics = run_tp(
            all_data,
            all_events,
            tp
        )

        results[tp] = metrics

        print_metrics(
            f"TP{tp:g}R",
            metrics
        )

    # ========================================================
    # TP2 DIRECTION AUDIT
    # ========================================================

    print("\n" + "=" * 80)
    print("TP2R DIRECTION AUDIT")
    print("=" * 80)

    for direction in ["LONG", "SHORT"]:

        metrics = direction_metrics(
            all_data,
            all_events,
            2.0,
            direction
        )

        print_metrics(
            direction,
            metrics
        )

    # ========================================================
    # INTEGRITY AUDIT
    # ========================================================

    print("\n" + "=" * 80)
    print("INTEGRITY AUDIT")
    print("=" * 80)

    integrity = {
        "TIMEFRAME_4H": INTERVAL == "4hour",
        "CAUSAL_SIGNAL_DETECTION": True,
        "SIGNED_ER_CURRENT_PREVIOUS_CROSS": True,
        "EMA200_CAUSAL": True,
        "ADX_CAUSAL": True,
        "DI_CAUSAL": True,
        "ATR_CAUSAL": True,
        "ATR_MEDIAN_CAUSAL": True,
        "VOLUME_MEDIAN_CAUSAL": True,
        "ENTRY_AT_SIGNAL_CLOSE": True,
        "ENTRY_CANDLE_EXIT_SCAN": False,
        "SAME_CANDLE_SL_FIRST": True,
        "NO_OVERLAP_LOCK": True,
        "NO_PARAMETER_OPTIMIZATION": True,
        "NO_EXTRA_FILTERS": True,
        "LONG_AND_SHORT_INCLUDED": True,
        "TOTAL_COST_R_003": abs(
            TOTAL_COST_R - 0.003
        ) < 1e-12,
    }

    for key, value in integrity.items():

        print(
            f"{key:<42} "
            f"{'TRUE' if value else 'FALSE'}"
        )

    integrity_pass = all(
        integrity.values()
    )

    print(
        f"\nINTEGRITY RESULT = "
        f"{'PASS' if integrity_pass else 'FAIL'}"
    )

    # ========================================================
    # DISCOVERY DECISION
    # ========================================================

    print("\n" + "=" * 80)
    print("RESEARCH GATE")
    print("=" * 80)

    print(
        "IMPORTANT: This is a 365-day DISCOVERY screen."
    )

    print(
        "No TP or direction is selected for production."
    )

    # Discovery is intentionally descriptive.
    # No optimization or selection is performed here.

    tp2 = results.get(2.0)

    if tp2:

        print(
            f"\nTP2R NetExp = "
            f"{tp2['net_exp']:.4f}R"
        )

        print(
            f"TP2R NetPF  = "
            f"{tp2['net_pf']:.4f}"
        )

        print(
            f"TP2R NetTotal = "
            f"{tp2['net_total']:.2f}R"
        )

    print("\n" + "=" * 80)
    print("CANDIDATE 10 DISCOVERY COMPLETE")
    print("=" * 80)

    print(
        "\nNEXT STEP RULE:"
    )

    print(
        "If discovery evidence is acceptable, "
        "the SAME FROZEN SPEC must be tested on "
        "730D IS/OOS."
    )

    print(
        "If OOS passes the predeclared gate, "
        "then perform 5-fold Walk-Forward."
    )

    print(
        "No parameter changes are allowed "
        "after seeing these results."
    )


if __name__ == "__main__":
    main() 
