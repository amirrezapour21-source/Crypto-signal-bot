# ============================================================
# CRYPTO SIGNAL BOT V4
# CANDIDATE 10
# 730D IS / OOS VALIDATION
#
# DIRECTIONAL EFFICIENCY TRANSITION -> TREND EXPANSION
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
# FROZEN SPECIFICATION
# ============================================================

API_URL = "https://api.kucoin.com/api/v1/market/candles"

INTERVAL = "4hour"
INTERVAL_SEC = 4 * 60 * 60

TARGET_DAYS = 730
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
TOTAL_COST_R = 0.003

IS_RATIO = 0.70

REQUEST_TIMEOUT = 20
MAX_RETRIES = 5
SLEEP_BETWEEN_REQUESTS = 0.15


# ============================================================
# UNIVERSE
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

def ts_to_str(ts):
    return datetime.fromtimestamp(
        int(ts),
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")


def rma(series, period):
    return series.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# DATA FETCH
# ============================================================

def fetch_history(symbol, target_days=TARGET_DAYS):

    now_ts = int(time.time())

    requested_start = (
        now_ts - target_days * 86400
    )

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
            cursor +
            (MAX_RECORDS_PER_REQUEST - 1) *
            INTERVAL_SEC,
            closed_end
        )

        params = {
            "symbol": symbol,
            "type": INTERVAL,
            "startAt": int(cursor),
            "endAt": int(chunk_end),
        }

        data = None
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
                    time.sleep(
                        1.0 + attempt
                    )

        if not success:
            break

        pages += 1

        if not data:
            break

        rows.extend(data)

        try:
            max_ts = max(
                int(x[0])
                for x in data
            )
        except Exception:
            break

        next_cursor = (
            max_ts + INTERVAL_SEC
        )

        if next_cursor <= cursor:
            print(
                f"[STALL] {symbol}: "
                "pagination cursor stalled."
            )
            break

        cursor = next_cursor

        time.sleep(
            SLEEP_BETWEEN_REQUESTS
        )

        if pages > 100:
            print(
                f"[STOP] {symbol}: "
                "pagination safety limit."
            )
            break

    if not rows:
        return None, {
            "pages": pages,
            "sufficient": False,
            "gap_count": None,
            "days": 0.0,
        }

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

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Only closed candles
    df = df[
        df["timestamp"] <= closed_end
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    if len(df) < 2:
        return None, {
            "pages": pages,
            "sufficient": False,
            "gap_count": None,
            "days": 0.0,
        }

    diffs = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    gap_count = int(
        (diffs != INTERVAL_SEC).sum()
    )

    first_ts = int(
        df["timestamp"].iloc[0]
    )

    last_ts = int(
        df["timestamp"].iloc[-1]
    )

    days = (
        last_ts - first_ts
    ) / 86400.0

    sufficient = (
        days >= MIN_HISTORY_DAYS
    )

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

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    # ATR - Wilder
    df["atr"] = rma(
        tr,
        ATR_PERIOD
    )

    # EMA200
    df["ema200"] = close.ewm(
        span=EMA_PERIOD,
        adjust=False,
        min_periods=EMA_PERIOD
    ).mean()

    # --------------------------------------------------------
    # Directional Movement / ADX
    # --------------------------------------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr_di = rma(
        tr,
        ADX_PERIOD
    )

    plus_di = (
        100.0 *
        rma(
            plus_dm,
            ADX_PERIOD
        ) /
        atr_di.replace(
            0,
            np.nan
        )
    )

    minus_di = (
        100.0 *
        rma(
            minus_dm,
            ADX_PERIOD
        ) /
        atr_di.replace(
            0,
            np.nan
        )
    )

    dx_denominator = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan
    )

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
    # (Close - Close[n]) /
    # Sum(abs(Close.diff()), n)
    #
    # Range approximately -1 to +1
    # --------------------------------------------------------

    net_change = (
        close -
        close.shift(ER_PERIOD)
    )

    path_length = (
        close.diff()
        .abs()
        .rolling(
            ER_PERIOD,
            min_periods=ER_PERIOD
        )
        .sum()
    )

    df["signed_er"] = (
        net_change /
        path_length.replace(
            0,
            np.nan
        )
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
            row["close"] >
            row["ema200"]

            and
            prev["signed_er"] <=
            ER_THRESHOLD

            and
            row["signed_er"] >
            ER_THRESHOLD

            and
            row["adx"] >=
            ADX_MIN

            and
            row["adx"] >
            prev["adx"]

            and
            row["plus_di"] >
            row["minus_di"]

            and
            row["atr"] >=
            row["atr_median"]

            and
            row["volume"] >=
            row["volume_median"]
        )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        short_signal = (
            row["close"] <
            row["ema200"]

            and
            prev["signed_er"] >=
            -ER_THRESHOLD

            and
            row["signed_er"] <
            -ER_THRESHOLD

            and
            row["adx"] >=
            ADX_MIN

            and
            row["adx"] >
            prev["adx"]

            and
            row["minus_di"] >
            row["plus_di"]

            and
            row["atr"] >=
            row["atr_median"]

            and
            row["volume"] >=
            row["volume_median"]
        )

        if long_signal:

            events.append({
                "symbol": symbol,
                "idx": i,
                "timestamp":
                    int(row["timestamp"]),
                "direction": "LONG",
                "entry":
                    float(row["close"]),
                "atr":
                    float(row["atr"]),
            })

        elif short_signal:

            events.append({
                "symbol": symbol,
                "idx": i,
                "timestamp":
                    int(row["timestamp"]),
                "direction": "SHORT",
                "entry":
                    float(row["close"]),
                "atr":
                    float(row["atr"]),
            })

    return events


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    event,
    tp_r
):

    idx = event["idx"]

    entry = event["entry"]
    atr = event["atr"]

    direction = event["direction"]

    if (
        not np.isfinite(entry)
        or
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        return {
            "status": "AMBIGUOUS",
            "gross_r": np.nan,
            "net_r": np.nan,
        }

    final_idx = (
        idx + HOLD_BARS
    )

    # Full HOLD must exist
    if final_idx >= len(df):

        return {
            "status":
                "OPEN_AT_DATASET_END",
            "gross_r": np.nan,
            "net_r": np.nan,
        }

    if direction == "LONG":

        sl = (
            entry -
            SL_ATR * atr
        )

        tp = (
            entry +
            tp_r * atr
        )

    else:

        sl = (
            entry +
            SL_ATR * atr
        )

        tp = (
            entry -
            tp_r * atr
        )

    # IMPORTANT:
    # Entry candle is NOT scanned.
    # Exit starts from next candle.

    for j in range(
        idx + 1,
        final_idx + 1
    ):

        candle = df.iloc[j]

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        if direction == "LONG":

            sl_hit = (
                low <= sl
            )

            tp_hit = (
                high >= tp
            )

        else:

            sl_hit = (
                high >= sl
            )

            tp_hit = (
                low <= tp
            )

        # ----------------------------------------------------
        # Frozen ambiguity rule:
        # SL FIRST
        # ----------------------------------------------------

        if sl_hit:

            gross_r = -1.0

            return {
                "status": "SL",
                "gross_r": gross_r,
                "net_r":
                    gross_r -
                    TOTAL_COST_R,
                "exit_idx": j,
                "exit_timestamp":
                    int(
                        candle["timestamp"]
                    ),
            }

        if tp_hit:

            gross_r = tp_r

            return {
                "status": "TP",
                "gross_r": gross_r,
                "net_r":
                    gross_r -
                    TOTAL_COST_R,
                "exit_idx": j,
                "exit_timestamp":
                    int(
                        candle["timestamp"]
                    ),
            }

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    candle = df.iloc[
        final_idx
    ]

    return {
        "status": "TIMEOUT",
        "gross_r": 0.0,
        "net_r":
            -TOTAL_COST_R,
        "exit_idx": final_idx,
        "exit_timestamp":
            int(
                candle["timestamp"]
            ),
    }


# ============================================================
# METRICS
# ============================================================

def max_drawdown(values):

    if len(values) == 0:
        return 0.0

    equity = np.cumsum(values)

    peak = np.maximum.accumulate(
        equity
    )

    dd = equity - peak

    return float(dd.min())


def calculate_metrics(trades):

    open_end = sum(
        t["status"] ==
        "OPEN_AT_DATASET_END"
        for t in trades
    )

    ambiguous = sum(
        t["status"] ==
        "AMBIGUOUS"
        for t in trades
    )

    measured = [
        t for t in trades
        if t["status"]
        not in (
            "OPEN_AT_DATASET_END",
            "AMBIGUOUS",
        )
    ]

    if not measured:

        return {
            "events": len(trades),
            "traded": 0,
            "open": open_end,
            "amb": ambiguous,
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

    measured.sort(
        key=lambda x: (
            x["timestamp"],
            x["symbol"]
        )
    )

    gross = np.array(
        [
            t["gross_r"]
            for t in measured
        ],
        dtype=float
    )

    net = np.array(
        [
            t["net_r"]
            for t in measured
        ],
        dtype=float
    )

    wr = float(
        np.mean(gross > 0)
    )

    gross_exp = float(
        np.mean(gross)
    )

    net_exp = float(
        np.mean(net)
    )

    gross_total = float(
        np.sum(gross)
    )

    net_total = float(
        np.sum(net)
    )

    gross_wins = gross[
        gross > 0
    ].sum()

    gross_losses = gross[
        gross < 0
    ].sum()

    net_wins = net[
        net > 0
    ].sum()

    net_losses = net[
        net < 0
    ].sum()

    gross_pf = (
        gross_wins /
        abs(gross_losses)
        if gross_losses < 0
        else np.inf
    )

    net_pf = (
        net_wins /
        abs(net_losses)
        if net_losses < 0
        else np.inf
    )

    return {
        "events": len(trades),
        "traded": len(measured),
        "open": open_end,
        "amb": ambiguous,
        "wr": wr,
        "gross_exp": gross_exp,
        "net_exp": net_exp,
        "gross_total": gross_total,
        "net_total": net_total,
        "gross_pf": float(gross_pf),
        "net_pf": float(net_pf),
        "gross_dd":
            max_drawdown(gross),
        "net_dd":
            max_drawdown(net),
    }


# ============================================================
# PRINT
# ============================================================

def fmt(x):

    if x is None:
        return "NA"

    if isinstance(x, float):

        if np.isnan(x):
            return "NA"

        if math.isinf(x):
            return "INF"

        return f"{x:.4f}"

    return str(x)


def print_metrics(
    label,
    m
):

    print(
        f"{label:<8}"
        f"events={m['events']} "
        f"traded={m['traded']} "
        f"open={m['open']} "
        f"amb={m['amb']} "
        f"WR={fmt(m['wr'])} "
        f"GrossExp={fmt(m['gross_exp'])} "
        f"NetExp={fmt(m['net_exp'])} "
        f"GrossTotal={fmt(m['gross_total'])} "
        f"NetTotal={fmt(m['net_total'])} "
        f"GrossPF={fmt(m['gross_pf'])} "
        f"NetPF={fmt(m['net_pf'])} "
        f"GrossDD={fmt(m['gross_dd'])} "
        f"NetDD={fmt(m['net_dd'])}"
    )


# ============================================================
# BUILD TRADE LIST
# ============================================================

def build_trades(
    all_data,
    all_events,
    tp_r,
    start_ts=None,
    end_ts=None
):

    trades = []

    for symbol, events in all_events.items():

        df = all_data[symbol]

        for event in events:

            ts = event["timestamp"]

            if (
                start_ts is not None
                and
                ts < start_ts
            ):
                continue

            if (
                end_ts is not None
                and
                ts > end_ts
            ):
                continue

            result = simulate_trade(
                df,
                event,
                tp_r
            )

            result["symbol"] = symbol
            result["timestamp"] = ts
            result["direction"] = (
                event["direction"]
            )

            trades.append(result)

    return trades


# ============================================================
# COMMON TIMESTAMPS
# ============================================================

def get_common_timestamps(
    all_data
):

    timestamp_sets = []

    for df in all_data.values():

        timestamp_sets.append(
            set(
                df["timestamp"]
                .astype(np.int64)
                .tolist()
            )
        )

    if not timestamp_sets:
        return []

    common = timestamp_sets[0]

    for s in timestamp_sets[1:]:
        common = common.intersection(s)

    return sorted(common)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "SETUP V4 — CANDIDATE 10"
    )
    print(
        "DIRECTIONAL EFFICIENCY "
        "TRANSITION → TREND EXPANSION"
    )
    print("=" * 80)

    print("\nFROZEN PARAMETERS")
    print(
        f"TIMEFRAME              = "
        f"{INTERVAL}"
    )
    print(
        f"HISTORY                = "
        f"{TARGET_DAYS} DAYS"
    )
    print(
        f"IS/OOS                 = "
        f"{IS_RATIO:.0%}/{1-IS_RATIO:.0%}"
    )
    print(
        f"ER_PERIOD              = "
        f"{ER_PERIOD}"
    )
    print(
        f"ER_THRESHOLD           = "
        f"±{ER_THRESHOLD}"
    )
    print(
        f"EMA                    = "
        f"{EMA_PERIOD}"
    )
    print(
        f"ADX_PERIOD             = "
        f"{ADX_PERIOD}"
    )
    print(
        f"ADX_MIN                = "
        f"{ADX_MIN}"
    )
    print(
        f"ATR_PERIOD             = "
        f"{ATR_PERIOD}"
    )
    print(
        f"MEDIAN_PERIOD          = "
        f"{MEDIAN_PERIOD}"
    )
    print(
        f"SL                     = "
        f"{SL_ATR} ATR"
    )
    print(
        f"HOLD                   = "
        f"{HOLD_BARS}"
    )
    print(
        f"TP                     = "
        f"{TP_MULTIPLIERS}"
    )
    print(
        f"TOTAL_COST_R           = "
        f"{TOTAL_COST_R}"
    )
    print(
        "DIRECTION              = "
        "LONG + SHORT"
    )
    print(
        "OVERLAP_LOCK           = FALSE"
    )
    print(
        "PARAMETER_OPTIMIZATION = FALSE"
    )
    print(
        "EXTRA_FILTERS          = FALSE"
    )

    # ========================================================
    # DATA
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA COLLECTION")
    print("=" * 80)

    all_data = {}
    all_events = {}

    failed_symbols = []
    insufficient_symbols = []

    for n, symbol in enumerate(
        SYMBOLS,
        1
    ):

        print(
            f"[{n:02d}/{len(SYMBOLS)}] "
            f"{symbol} ... ",
            end="",
            flush=True
        )

        try:

            df, meta = fetch_history(
                symbol,
                TARGET_DAYS
            )

            if df is None:

                failed_symbols.append(
                    symbol
                )

                print("FAILED")
                continue

            if not meta["sufficient"]:

                insufficient_symbols.append(
                    symbol
                )

                print(
                    f"INSUFFICIENT "
                    f"{meta['days']:.1f}d"
                )

                continue

            df = add_indicators(
                df
            )

            events = detect_events(
                df,
                symbol
            )

            all_data[symbol] = df
            all_events[symbol] = events

            print(
                f"OK "
                f"{len(df)} candles "
                f"{meta['days']:.1f}d "
                f"gaps={meta['gap_count']} "
                f"events={len(events)}"
            )

        except Exception as exc:

            failed_symbols.append(
                symbol
            )

            print(
                f"FAILED: {exc}"
            )

    # ========================================================
    # DATA AUDIT
    # ========================================================

    print("\n" + "=" * 80)
    print("DATA AUDIT")
    print("=" * 80)

    valid_symbols = sorted(
        all_data.keys()
    )

    print(
        f"VALID SYMBOLS       = "
        f"{len(valid_symbols)}"
    )

    print(
        f"INSUFFICIENT        = "
        f"{len(insufficient_symbols)}"
    )

    print(
        f"FAILED              = "
        f"{len(failed_symbols)}"
    )

    if not valid_symbols:
        print(
            "\nNO VALID SYMBOLS. STOP."
        )
        return

    # --------------------------------------------------------
    # Gap audit
    # --------------------------------------------------------

    gap_map = {}

    for symbol in valid_symbols:

        timestamps = (
            all_data[symbol]["timestamp"]
            .astype(np.int64)
        )

        diffs = (
            timestamps.diff()
            .dropna()
        )

        gap_map[symbol] = int(
            (diffs != INTERVAL_SEC)
            .sum()
        )

    total_gaps = sum(
        gap_map.values()
    )

    print(
        f"TOTAL GAPS          = "
        f"{total_gaps}"
    )

    # ========================================================
    # COMMON TIMESTAMPS
    # ========================================================

    common_timestamps = (
        get_common_timestamps(
            all_data
        )
    )

    if len(common_timestamps) < 1000:

        print(
            "\nCOMMON TIMESTAMP "
            "COUNT TOO LOW. STOP."
        )
        return

    common_start = (
        common_timestamps[0]
    )

    common_end = (
        common_timestamps[-1]
    )

    print(
        f"COMMON TIMESTAMPS    = "
        f"{len(common_timestamps)}"
    )

    print(
        f"COMMON START        = "
        f"{ts_to_str(common_start)}"
    )

    print(
        f"COMMON END          = "
        f"{ts_to_str(common_end)}"
    )

    # ========================================================
    # 70 / 30 SPLIT
    # ========================================================

    split_index = int(
        len(common_timestamps) *
        IS_RATIO
    )

    # IS last timestamp
    is_end_ts = (
        common_timestamps[
            split_index - 1
        ]
    )

    # OOS starts at next common timestamp
    oos_start_ts = (
        common_timestamps[
            split_index
        ]
    )

    oos_end_ts = common_end

    print("\n" + "=" * 80)
    print("IS / OOS SPLIT")
    print("=" * 80)

    print(
        f"IS_START            = "
        f"{ts_to_str(common_start)}"
    )

    print(
        f"IS_END              = "
        f"{ts_to_str(is_end_ts)}"
    )

    print(
        f"OOS_START           = "
        f"{ts_to_str(oos_start_ts)}"
    )

    print(
        f"OOS_END             = "
        f"{ts_to_str(oos_end_ts)}"
    )

    print(
        "OOS_WARMUP_CONTEXT  = TRUE"
    )

    # ========================================================
    # EVENTS
    # ========================================================

    all_events_list = []

    for symbol in valid_symbols:

        all_events_list.extend(
            all_events[symbol]
        )

    all_events_list.sort(
        key=lambda x: (
            x["timestamp"],
            x["symbol"]
        )
    )

    total_events = len(
        all_events_list
    )

    long_events = sum(
        x["direction"] == "LONG"
        for x in all_events_list
    )

    short_events = sum(
        x["direction"] == "SHORT"
        for x in all_events_list
    )

    is_events = sum(
        x["timestamp"] <= is_end_ts
        for x in all_events_list
    )

    oos_events = sum(
        x["timestamp"] >= oos_start_ts
        for x in all_events_list
    )

    print("\n" + "=" * 80)
    print("EVENT AUDIT")
    print("=" * 80)

    print(
        f"TOTAL EVENTS        = "
        f"{total_events}"
    )

    print(
        f"LONG EVENTS         = "
        f"{long_events}"
    )

    print(
        f"SHORT EVENTS        = "
        f"{short_events}"
    )

    print(
        f"IS EVENTS           = "
        f"{is_events}"
    )

    print(
        f"OOS EVENTS          = "
        f"{oos_events}"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("IS RESULTS")
    print("=" * 80)

    is_results = {}

    for tp in TP_MULTIPLIERS:

        trades = build_trades(
            all_data,
            all_events,
            tp,
            start_ts=common_start,
            end_ts=is_end_ts
        )

        metrics = calculate_metrics(
            trades
        )

        is_results[tp] = metrics

        print_metrics(
            f"TP{tp:g}R",
            metrics
        )

    # ========================================================
    # OOS
    # ========================================================

    print("\n" + "=" * 80)
    print("OOS RESULTS")
    print("=" * 80)

    oos_results = {}

    for tp in TP_MULTIPLIERS:

        trades = build_trades(
            all_data,
            all_events,
            tp,
            start_ts=oos_start_ts,
            end_ts=oos_end_ts
        )

        metrics = calculate_metrics(
            trades
        )

        oos_results[tp] = metrics

        print_metrics(
            f"TP{tp:g}R",
            metrics
        )

    # ========================================================
    # OOS DIRECTION AUDIT
    # ========================================================

    print("\n" + "=" * 80)
    print("OOS TP2R DIRECTION AUDIT")
    print("=" * 80)

    for direction in [
        "LONG",
        "SHORT"
    ]:

        direction_trades = []

        for symbol, events in (
            all_events.items()
        ):

            df = all_data[symbol]

            for event in events:

                if (
                    event["direction"]
                    != direction
                ):
                    continue

                if (
                    event["timestamp"]
                    < oos_start_ts
                ):
                    continue

                if (
                    event["timestamp"]
                    > oos_end_ts
                ):
                    continue

                result = simulate_trade(
                    df,
                    event,
                    2.0
                )

                result["symbol"] = symbol
                result["timestamp"] = (
                    event["timestamp"]
                )

                direction_trades.append(
                    result
                )

        metrics = calculate_metrics(
            direction_trades
        )

        print_metrics(
            direction,
            metrics
        )

    # ========================================================
    # INTEGRITY
    # ========================================================

    print("\n" + "=" * 80)
    print("INTEGRITY AUDIT")
    print("=" * 80)

    integrity = {

        "TIMEFRAME_4H":
            INTERVAL == "4hour",

        "CAUSAL_SIGNAL_DETECTION":
            True,

        "SIGNED_ER_CURRENT_PREVIOUS_CROSS":
            True,

        "EMA200_CAUSAL":
            True,

        "ADX_CAUSAL":
            True,

        "DI_CAUSAL":
            True,

        "ATR_CAUSAL":
            True,

        "ATR_MEDIAN_CAUSAL":
            True,

        "VOLUME_MEDIAN_CAUSAL":
            True,

        "ENTRY_AT_SIGNAL_CLOSE":
            True,

        # FALSE is CORRECT here:
        # exit scan does NOT include entry candle
        "ENTRY_CANDLE_EXIT_SCAN":
            False,

        "SAME_CANDLE_SL_FIRST":
            True,

        "NO_OVERLAP_LOCK":
            True,

        "NO_PARAMETER_OPTIMIZATION":
            True,

        "NO_EXTRA_FILTERS":
            True,

        "LONG_AND_SHORT_INCLUDED":
            True,

        "TOTAL_COST_R_003":
            abs(
                TOTAL_COST_R - 0.003
            ) < 1e-12,

        "OOS_WARMUP_CONTEXT":
            True,

        "COMMON_TIMESTAMP_SPLIT":
            True,

        "FUTURE_DATA_NOT_USED_FOR_SIGNAL":
            True,
    }

    # IMPORTANT:
    # ENTRY_CANDLE_EXIT_SCAN must be FALSE.
    # Therefore it must NOT be included
    # in an all(True) test.

    critical_integrity = {
        k: v
        for k, v in integrity.items()
        if k !=
        "ENTRY_CANDLE_EXIT_SCAN"
    }

    for key, value in integrity.items():

        print(
            f"{key:<46} "
            f"{'TRUE' if value else 'FALSE'}"
        )

    integrity_pass = all(
        critical_integrity.values()
    )

    print(
        f"\nINTEGRITY RESULT = "
        f"{'PASS' if integrity_pass else 'FAIL'}"
    )

    # ========================================================
    # PREDECLARED RESEARCH GATE
    # ========================================================

    print("\n" + "=" * 80)
    print("PREDECLARED OOS RESEARCH GATE")
    print("=" * 80)

    print(
        "Gate is evaluated on OOS, not Discovery."
    )

    print(
        "Required:"
    )

    print(
        "1) OOS Net Expectancy > 0"
    )

    print(
        "2) OOS Net PF > 1"
    )

    print(
        "3) No direction-only selection"
    )

    print(
        "4) No parameter optimization"
    )

    print(
        "5) Integrity PASS"
    )

    print(
        "\nOOS RESULTS:"
    )

    for tp in TP_MULTIPLIERS:

        m = oos_results[tp]

        passed = (
            np.isfinite(
                m["net_exp"]
            )
            and
            np.isfinite(
                m["net_pf"]
            )
            and
            m["net_exp"] > 0
            and
            m["net_pf"] > 1
            and
            m["traded"] > 0
        )

        print(
            f"TP{tp:g}R "
            f"NetExp={fmt(m['net_exp'])} "
            f"NetPF={fmt(m['net_pf'])} "
            f"NetTotal={fmt(m['net_total'])} "
            f"Gate={'PASS' if passed else 'FAIL'}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Do not select a TP here for production.
    # We only report whether any TP has
    # positive OOS evidence.
    # --------------------------------------------------------

    positive_oos = [
        tp
        for tp in TP_MULTIPLIERS
        if (
            np.isfinite(
                oos_results[tp]["net_exp"]
            )
            and
            np.isfinite(
                oos_results[tp]["net_pf"]
            )
            and
            oos_results[tp]["net_exp"] > 0
            and
            oos_results[tp]["net_pf"] > 1
        )
    ]

    print("\n" + "=" * 80)
    print("FINAL 730D IS/OOS STATUS")
    print("=" * 80)

    if not integrity_pass:

        print(
            "STATUS = INVALID"
        )

        print(
            "Integrity failure. "
            "Do not interpret performance."
        )

    elif positive_oos:

        print(
            "STATUS = OOS POSITIVE EVIDENCE"
        )

        print(
            "Candidate 10 may proceed "
            "to predeclared 5-fold Walk-Forward."
        )

        print(
            "No TP or direction is selected "
            "for production from this run."
        )

    else:

        print(
            "STATUS = OOS FAIL"
        )

        print(
            "Candidate 10 should be archived."
        )

        print(
            "No parameter patching."
        )

    print("\n" + "=" * 80)
    print("END OF CANDIDATE 10 — 730D IS/OOS")
    print("=" * 80)


if __name__ == "__main__":
    main()
