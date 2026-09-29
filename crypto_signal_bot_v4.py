# ============================================================
# SETUP V4 — CANDIDATE 9
# MACD ZERO-LINE TREND CONTINUATION
# 730D IS / OOS VALIDATION
# ============================================================

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# FROZEN STRATEGY SPEC
# ============================================================

INTERVAL = "4hour"
DAYS = 730

EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 200

MACD_SIGNAL = 9
ADX_PERIOD = 14
ADX_MIN = 20

ATR_PERIOD = 20
MEDIAN_PERIOD = 20

SL_ATR = 1.25
HOLD_BARS = 30

TP_MULTS = [1.0, 1.5, 2.0, 3.0]

# COST = fee 0.10% + slippage 0.20%
TOTAL_COST_R = 0.003

MIN_HISTORY_DAYS = 180

BASE_URL = "https://api.kucoin.com/api/v1/market/candles"

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
    "SUI-USDT",
    "NEAR-USDT",
    "APT-USDT",
    "ARB-USDT",
    "OP-USDT",
    "ATOM-USDT",
    "FIL-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "ETC-USDT",
    "UNI-USDT",
    "AAVE-USDT",
    "INJ-USDT",
    "SEI-USDT",
    "TIA-USDT",
    "WIF-USDT",
    "PEPE-USDT",
    "FLOKI-USDT",
    "SHIB-USDT",
    "TRX-USDT",
    "TON-USDT",
    "MKR-USDT",
    "RUNE-USDT",
    "ICP-USDT",
    "XLM-USDT",
    "HBAR-USDT",
    "ALGO-USDT",
    "VET-USDT",
]

# ============================================================
# KUCOIN FETCH — PAGINATED 730D
# ============================================================

def fetch_history(symbol, days=DAYS):

    end_ts = int(time.time())
    start_ts = end_ts - days * 86400

    all_rows = []

    cursor_end = end_ts

    while cursor_end > start_ts:

        params = {
            "symbol": symbol,
            "type": INTERVAL,
            "startAt": start_ts,
            "endAt": cursor_end
        }

        try:
            r = requests.get(
                BASE_URL,
                params=params,
                timeout=30
            )

            if r.status_code != 200:
                print(f"{symbol} HTTP ERROR {r.status_code}")
                break

            js = r.json()

            if js.get("code") != "200000":
                print(f"{symbol} API ERROR: {js}")
                break

            rows = js.get("data", [])

            if not rows:
                break

            all_rows.extend(rows)

            oldest = min(int(x[0]) for x in rows)

            new_cursor = oldest - 1

            if new_cursor >= cursor_end:
                break

            cursor_end = new_cursor

            time.sleep(0.12)

        except Exception as e:
            print(f"{symbol} FETCH ERROR: {e}")
            break

    if not all_rows:
        return None

    # KuCoin:
    # [time, open, close, high, low, volume, turnover]

    df = pd.DataFrame(
        all_rows,
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

    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # KuCoin timestamps in this endpoint are seconds.
    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    for c in [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "turnover"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = (
        df.dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Remove currently incomplete 4H candle.
    now = pd.Timestamp.now(tz="UTC")

    df = df[df["time"] < now].copy()

    # Keep requested history window.
    cutoff = now - pd.Timedelta(days=days)

    df = df[df["time"] >= cutoff].copy()

    df = df.reset_index(drop=True)

    return df


# ============================================================
# WILDER RMA
# ============================================================

def rma(series, period):

    return series.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema12"] = close.ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema26"] = close.ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    df["ema200"] = close.ewm(
        span=EMA_TREND,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["macd"] = df["ema12"] - df["ema26"]

    df["macd_signal"] = df["macd"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    df["hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # TRUE RANGE
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # --------------------------------------------------------
    # ATR — WILDER
    # --------------------------------------------------------

    df["atr"] = rma(
        df["tr"],
        ATR_PERIOD
    )

    # --------------------------------------------------------
    # +DM / -DM
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

    plus_dm_rma = rma(
        plus_dm,
        ADX_PERIOD
    )

    minus_dm_rma = rma(
        minus_dm,
        ADX_PERIOD
    )

    atr_for_adx = rma(
        df["tr"],
        ADX_PERIOD
    )

    df["plus_di"] = (
        100 *
        plus_dm_rma /
        atr_for_adx.replace(0, np.nan)
    )

    df["minus_di"] = (
        100 *
        minus_dm_rma /
        atr_for_adx.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # DX / ADX
    # --------------------------------------------------------

    di_sum = (
        df["plus_di"] +
        df["minus_di"]
    )

    df["dx"] = (
        100 *
        (
            df["plus_di"] -
            df["minus_di"]
        ).abs() /
        di_sum.replace(0, np.nan)
    )

    df["adx"] = rma(
        df["dx"],
        ADX_PERIOD
    )

    # --------------------------------------------------------
    # ATR MEDIAN
    # --------------------------------------------------------

    df["atr_median"] = (
        df["atr"]
        .rolling(MEDIAN_PERIOD)
        .median()
    )

    # --------------------------------------------------------
    # VOLUME MEDIAN
    # --------------------------------------------------------

    df["volume_median"] = (
        df["volume"]
        .rolling(MEDIAN_PERIOD)
        .median()
    )

    return df


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_events(df):

    events = []

    # Start sufficiently late to ensure all indicators exist.
    start = max(
        EMA_TREND + 5,
        250
    )

    for i in range(start, len(df)):

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        values = [
            row["close"],
            row["ema200"],
            row["ema12"],
            row["ema26"],
            row["hist"],
            prev["hist"],
            row["adx"],
            row["plus_di"],
            row["minus_di"],
            row["atr"],
            row["atr_median"],
            row["volume"],
            row["volume_median"],
        ]

        if not np.all(np.isfinite(values)):
            continue

        # ====================================================
        # LONG
        # ====================================================

        long_signal = (
            row["close"] > row["ema200"]
            and
            row["ema12"] > row["ema26"]
            and
            prev["hist"] <= 0
            and
            row["hist"] > 0
            and
            row["adx"] >= ADX_MIN
            and
            row["plus_di"] > row["minus_di"]
            and
            row["atr"] >= row["atr_median"]
            and
            row["volume"] >= row["volume_median"]
        )

        # ====================================================
        # SHORT
        # ====================================================

        short_signal = (
            row["close"] < row["ema200"]
            and
            row["ema12"] < row["ema26"]
            and
            prev["hist"] >= 0
            and
            row["hist"] < 0
            and
            row["adx"] >= ADX_MIN
            and
            row["minus_di"] > row["plus_di"]
            and
            row["atr"] >= row["atr_median"]
            and
            row["volume"] >= row["volume_median"]
        )

        if long_signal:
            events.append({
                "idx": i,
                "time": row["time"],
                "direction": "LONG"
            })

        elif short_signal:
            events.append({
                "idx": i,
                "time": row["time"],
                "direction": "SHORT"
            })

    return events


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    event,
    tp_mult
):

    idx = event["idx"]
    direction = event["direction"]

    # Need full HOLD window.
    if idx + HOLD_BARS >= len(df):
        return {
            "status": "OPEN_AT_DATASET_END",
            "direction": direction,
            "entry_idx": idx
        }

    entry = float(df.iloc[idx]["close"])
    atr = float(df.iloc[idx]["atr"])

    if not np.isfinite(entry) or not np.isfinite(atr):
        return {
            "status": "AMBIGUOUS",
            "direction": direction,
            "entry_idx": idx
        }

    sl_distance = SL_ATR * atr
    tp_distance = tp_mult * sl_distance

    if direction == "LONG":

        sl = entry - sl_distance
        tp = entry + tp_distance

    else:

        sl = entry + sl_distance
        tp = entry - tp_distance

    # IMPORTANT:
    # Entry candle is NOT scanned for exits.
    # Scanning begins from idx + 1.
    for j in range(
        idx + 1,
        idx + HOLD_BARS + 1
    ):

        candle = df.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Same-candle SL-first.
            if hit_sl:
                gross_r = -1.0
                return {
                    "status": "SL",
                    "direction": direction,
                    "entry_idx": idx,
                    "exit_idx": j,
                    "gross_r": gross_r,
                    "net_r": gross_r - TOTAL_COST_R
                }

            if hit_tp:
                gross_r = tp_mult
                return {
                    "status": "TP",
                    "direction": direction,
                    "entry_idx": idx,
                    "exit_idx": j,
                    "gross_r": gross_r,
                    "net_r": gross_r - TOTAL_COST_R
                }

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Same-candle SL-first.
            if hit_sl:
                gross_r = -1.0
                return {
                    "status": "SL",
                    "direction": direction,
                    "entry_idx": idx,
                    "exit_idx": j,
                    "gross_r": gross_r,
                    "net_r": gross_r - TOTAL_COST_R
                }

            if hit_tp:
                gross_r = tp_mult
                return {
                    "status": "TP",
                    "direction": direction,
                    "entry_idx": idx,
                    "exit_idx": j,
                    "gross_r": gross_r,
                    "net_r": gross_r - TOTAL_COST_R
                }

    # Full HOLD completed without SL/TP.
    return {
        "status": "TIMEOUT",
        "direction": direction,
        "entry_idx": idx,
        "exit_idx": idx + HOLD_BARS,
        "gross_r": 0.0,
        "net_r": -TOTAL_COST_R
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    closed = [
        x for x in trades
        if x["status"] in [
            "SL",
            "TP",
            "TIMEOUT"
        ]
    ]

    open_end = [
        x for x in trades
        if x["status"] == "OPEN_AT_DATASET_END"
    ]

    ambiguous = [
        x for x in trades
        if x["status"] == "AMBIGUOUS"
    ]

    if not closed:
        return {
            "n_events": len(trades),
            "n_traded": 0,
            "open_end": len(open_end),
            "ambiguous": len(ambiguous)
        }

    gross = np.array(
        [x["gross_r"] for x in closed],
        dtype=float
    )

    net = np.array(
        [x["net_r"] for x in closed],
        dtype=float
    )

    wins = np.sum(gross > 0)

    wr = wins / len(closed)

    gross_exp = float(np.mean(gross))
    net_exp = float(np.mean(net))

    gross_total = float(np.sum(gross))
    net_total = float(np.sum(net))

    gross_profit = np.sum(
        gross[gross > 0]
    )

    gross_loss = abs(
        np.sum(gross[gross < 0])
    )

    net_profit = np.sum(
        net[net > 0]
    )

    net_loss = abs(
        np.sum(net[net < 0])
    )

    gross_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    net_pf = (
        net_profit / net_loss
        if net_loss > 0
        else np.inf
    )

    # --------------------------------------------------------
    # Max Drawdown
    # --------------------------------------------------------

    gross_curve = np.cumsum(gross)
    gross_peak = np.maximum.accumulate(
        np.insert(gross_curve, 0, 0)
    )[1:]

    gross_dd = (
        gross_curve -
        gross_peak
    )

    gross_max_dd = float(
        np.min(gross_dd)
    )

    net_curve = np.cumsum(net)
    net_peak = np.maximum.accumulate(
        np.insert(net_curve, 0, 0)
    )[1:]

    net_dd = (
        net_curve -
        net_peak
    )

    net_max_dd = float(
        np.min(net_dd)
    )

    return {
        "n_events": len(trades),
        "n_traded": len(closed),
        "open_end": len(open_end),
        "ambiguous": len(ambiguous),
        "wr": wr,
        "gross_exp": gross_exp,
        "net_exp": net_exp,
        "gross_total": gross_total,
        "net_total": net_total,
        "gross_pf": gross_pf,
        "net_pf": net_pf,
        "gross_max_dd": gross_max_dd,
        "net_max_dd": net_max_dd
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(label, metrics):

    print("")
    print("=" * 70)
    print(label)
    print("=" * 70)

    for k, v in metrics.items():

        if isinstance(v, float):

            if np.isinf(v):
                print(f"{k}: INF")

            else:
                print(
                    f"{k}: {v:.4f}"
                )

        else:
            print(
                f"{k}: {v}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 70)
    print("SETUP V4 — CANDIDATE 9")
    print("MACD ZERO-LINE TREND CONTINUATION")
    print("730D IS / OOS VALIDATION")
    print("=" * 70)

    datasets = {}

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    for symbol in SYMBOLS:

        print(
            f"\nFetching {symbol} ..."
        )

        df = fetch_history(
            symbol,
            DAYS
        )

        if df is None or len(df) < 1000:

            print(
                f"{symbol}: INSUFFICIENT_HISTORY"
            )
            continue

        actual_days = (
            (
                df["time"].iloc[-1] -
                df["time"].iloc[0]
            ).total_seconds()
            / 86400
        )

        if actual_days < MIN_HISTORY_DAYS:

            print(
                f"{symbol}: INSUFFICIENT_HISTORY "
                f"{actual_days:.1f}d"
            )
            continue

        # ----------------------------------------------------
        # Continuity
        # ----------------------------------------------------

        diffs = (
            df["time"]
            .diff()
            .dropna()
            .dt.total_seconds()
        )

        expected = 4 * 3600

        gap_count = int(
            np.sum(diffs != expected)
        )

        if gap_count > 0:

            print(
                f"{symbol}: GAP_COUNT={gap_count}"
            )

        df = add_indicators(df)

        datasets[symbol] = df

        print(
            f"{symbol}: "
            f"{len(df)} candles | "
            f"{actual_days:.1f}d | "
            f"gaps={gap_count}"
        )

    # --------------------------------------------------------
    # COMMON TIMESTAMPS
    # --------------------------------------------------------

    if not datasets:

        print("NO VALID DATA")
        return

    common_times = None

    for df in datasets.values():

        current = set(
            df["time"]
        )

        if common_times is None:
            common_times = current
        else:
            common_times &= current

    common_times = sorted(
        common_times
    )

    print("")
    print("=" * 70)
    print("DATA AUDIT")
    print("=" * 70)

    print(
        f"DATA_VALID_SYMBOLS: {len(datasets)}"
    )

    print(
        f"COMMON_TIMESTAMPS: {len(common_times)}"
    )

    if common_times:

        print(
            "COMMON_START:",
            common_times[0]
        )

        print(
            "COMMON_END:",
            common_times[-1]
        )

    # --------------------------------------------------------
    # ALIGN DATA TO COMMON TIMESTAMPS
    # --------------------------------------------------------

    aligned = {}

    for symbol, df in datasets.items():

        x = (
            df[
                df["time"].isin(common_times)
            ]
            .sort_values("time")
            .reset_index(drop=True)
        )

        aligned[symbol] = x

    common_n = len(common_times)

    # --------------------------------------------------------
    # 70 / 30 SPLIT
    # --------------------------------------------------------

    split_idx = int(
        common_n * 0.70
    )

    IS_END = common_times[
        split_idx - 1
    ]

    OOS_START = common_times[
        split_idx
    ]

    OOS_END = common_times[-1]

    print("")
    print("=" * 70)
    print("IS / OOS SPLIT")
    print("=" * 70)

    print(
        "IS_END:",
        IS_END
    )

    print(
        "OOS_START:",
        OOS_START
    )

    print(
        "OOS_END:",
        OOS_END
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    all_events = {}

    total_events = 0
    total_long = 0
    total_short = 0

    for symbol, df in aligned.items():

        events = detect_events(df)

        all_events[symbol] = events

        total_events += len(events)

        total_long += sum(
            e["direction"] == "LONG"
            for e in events
        )

        total_short += sum(
            e["direction"] == "SHORT"
            for e in events
        )

    print("")
    print("=" * 70)
    print("SIGNAL DETECTION")
    print("=" * 70)

    print(
        f"TOTAL_EVENTS: {total_events}"
    )

    print(
        f"LONG_EVENTS: {total_long}"
    )

    print(
        f"SHORT_EVENTS: {total_short}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("TP SCENARIOS")
    print("=" * 70)

    for tp_mult in TP_MULTS:

        is_trades = []
        oos_trades = []

        is_long = []
        is_short = []

        oos_long = []
        oos_short = []

        for symbol, events in all_events.items():

            df = aligned[symbol]

            for event in events:

                event_time = event["time"]

                trade = simulate_trade(
                    df,
                    event,
                    tp_mult
                )

                trade["symbol"] = symbol
                trade["time"] = event_time

                # --------------------------------------------
                # IS
                # --------------------------------------------

                if event_time <= IS_END:

                    is_trades.append(trade)

                    if trade["status"] in [
                        "SL",
                        "TP",
                        "TIMEOUT"
                    ]:

                        if event["direction"] == "LONG":
                            is_long.append(trade)
                        else:
                            is_short.append(trade)

                # --------------------------------------------
                # OOS
                # --------------------------------------------

                elif event_time >= OOS_START:

                    oos_trades.append(trade)

                    if trade["status"] in [
                        "SL",
                        "TP",
                        "TIMEOUT"
                    ]:

                        if event["direction"] == "LONG":
                            oos_long.append(trade)
                        else:
                            oos_short.append(trade)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        is_metrics = calculate_metrics(
            is_trades
        )

        oos_metrics = calculate_metrics(
            oos_trades
        )

        print_metrics(
            f"TP {tp_mult}R — IS",
            is_metrics
        )

        print_metrics(
            f"TP {tp_mult}R — OOS",
            oos_metrics
        )

        # ----------------------------------------------------
        # Direction breakdown
        # Informational only — NO selection.
        # ----------------------------------------------------

        if tp_mult == 2.0:

            print_metrics(
                "TP 2R — IS LONG",
                calculate_metrics(is_long)
            )

            print_metrics(
                "TP 2R — IS SHORT",
                calculate_metrics(is_short)
            )

            print_metrics(
                "TP 2R — OOS LONG",
                calculate_metrics(oos_long)
            )

            print_metrics(
                "TP 2R — OOS SHORT",
                calculate_metrics(oos_short)
            )

    # --------------------------------------------------------
    # INTEGRITY AUDIT
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("VALIDATION INTEGRITY AUDIT")
    print("=" * 70)

    print("CAUSAL_SIGNAL_DETECTION: TRUE")
    print("MACD_CROSS_USES_CURRENT_AND_PREVIOUS_HIST: TRUE")
    print("EMA200_CAUSAL: TRUE")
    print("ADX_CAUSAL: TRUE")
    print("ATR_CAUSAL: TRUE")
    print("ATR_MEDIAN_CAUSAL: TRUE")
    print("VOLUME_MEDIAN_CAUSAL: TRUE")
    print("ENTRY_AT_SIGNAL_CLOSE: TRUE")
    print("ENTRY_CANDLE_EXIT_SCAN: FALSE")
    print("SAME_CANDLE_SL_FIRST: TRUE")
    print("NO_OVERLAP_LOCK: TRUE")
    print("NO_PARAMETER_OPTIMIZATION: TRUE")
    print("NO_EXTRA_FILTERS: TRUE")
    print("LONG_AND_SHORT_INCLUDED: TRUE")
    print(
        f"TOTAL_COST_R: {TOTAL_COST_R:.4f}"
    )
    print("OOS_WARMUP_CONTEXT: TRUE")

    # --------------------------------------------------------
    # RESEARCH GATE
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("RESEARCH GATE")
    print("=" * 70)

    print(
        "Candidate 9 remains FROZEN."
    )

    print(
        "Do NOT select LONG-only or SHORT-only "
        "based on this run."
    )

    print(
        "Next decision depends on IS/OOS results."
    )

    print("")
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
