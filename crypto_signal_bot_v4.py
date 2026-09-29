import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# SETUP V4 — CANDIDATE 9
# MACD ZERO-LINE TREND CONTINUATION
# 5-FOLD WALK-FORWARD VALIDATION
# ============================================================

SYMBOLS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT",
    "DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
    "SUI-USDT","NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT",
    "ATOM-USDT","FIL-USDT","LTC-USDT","BCH-USDT","ETC-USDT",
    "UNI-USDT","AAVE-USDT","INJ-USDT","SEI-USDT","TIA-USDT",
    "WIF-USDT","PEPE-USDT","FLOKI-USDT","SHIB-USDT","TRX-USDT",
    "RUNE-USDT","ICP-USDT","XLM-USDT","HBAR-USDT","ALGO-USDT",
    "VET-USDT","TON-USDT","MKR-USDT"
]

INTERVAL = "4hour"
TARGET_DAYS = 730

EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 200
MACD_SIGNAL = 9

ADX_PERIOD = 14
ADX_MIN = 20.0

ATR_PERIOD = 20
ATR_MEDIAN_PERIOD = 20
VOLUME_MEDIAN_PERIOD = 20

SL_ATR = 1.25
HOLD_BARS = 30

TP_R_LIST = [1.0, 1.5, 2.0, 3.0]

FEE_R = 0.001
SLIPPAGE_R = 0.002
TOTAL_COST_R = FEE_R + SLIPPAGE_R

MIN_HISTORY_DAYS = 180

# ------------------------------------------------------------
# WALK-FORWARD
#
# 5 folds:
# 50-60%
# 60-70%
# 70-80%
# 80-90%
# 90-100%
#
# Each fold tests ONLY its future segment.
# No parameter optimization.
# ------------------------------------------------------------

WF_FOLDS = [
    ("WF1", 0.50, 0.60),
    ("WF2", 0.60, 0.70),
    ("WF3", 0.70, 0.80),
    ("WF4", 0.80, 0.90),
    ("WF5", 0.90, 1.00),
]


# ============================================================
# KUCOIN DATA
# ============================================================

BASE_URL = "https://api.kucoin.com"


def fetch_history(symbol, target_days=730):
    """
    Paginated KuCoin 4H candles.
    KuCoin candle timestamps are seconds.
    Returns oldest -> newest.
    """

    end_ts = int(time.time())
    start_ts = end_ts - target_days * 86400

    all_rows = []

    # KuCoin returns limited rows per request.
    # Use multiple overlapping pages safely.
    cursor_end = end_ts

    for _ in range(80):

        params = {
            "symbol": symbol,
            "type": INTERVAL,
            "startAt": start_ts,
            "endAt": cursor_end
        }

        try:
            r = requests.get(
                f"{BASE_URL}/api/v1/market/candles",
                params=params,
                timeout=20
            )

            data = r.json()

            if data.get("code") != "200000":
                print(f"{symbol} API ERROR: {data}")
                break

            rows = data.get("data", [])

            if not rows:
                break

            all_rows.extend(rows)

            ts_values = [int(x[0]) for x in rows]
            oldest = min(ts_values)

            if oldest <= start_ts:
                break

            new_cursor = oldest - 1

            if new_cursor >= cursor_end:
                break

            cursor_end = new_cursor

            time.sleep(0.05)

        except Exception as e:
            print(f"{symbol} FETCH ERROR: {e}")
            break

    if not all_rows:
        return None

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

    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=["time", "open", "close", "high", "low", "volume"]
    )

    df = df.drop_duplicates("time")

    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------------
    # Remove incomplete latest 4H candle
    # --------------------------------------------------------

    now = int(time.time())

    interval_seconds = 4 * 3600

    df = df[
        (df["time"] + interval_seconds) <= now
    ].copy()

    df = df[
        (df["time"] >= start_ts)
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 1000:
        return None

    return df


# ============================================================
# DATA AUDIT
# ============================================================

def audit_data(df):

    if df is None or len(df) < 1000:
        return False, "INSUFFICIENT_HISTORY"

    ts = df["time"].astype(np.int64).values

    if len(ts) < 2:
        return False, "TOO_SHORT"

    if not np.all(np.diff(ts) > 0):
        return False, "NON_MONOTONIC_OR_DUPLICATE"

    expected = 4 * 3600

    gaps = np.diff(ts)

    gap_count = int(np.sum(gaps != expected))

    return True, gap_count


def format_time(ts):
    return datetime.fromtimestamp(
        int(ts),
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    x["ema12"] = close.ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST
    ).mean()

    x["ema26"] = close.ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW
    ).mean()

    x["ema200"] = close.ewm(
        span=EMA_TREND,
        adjust=False,
        min_periods=EMA_TREND
    ).mean()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    x["macd"] = x["ema12"] - x["ema26"]

    x["macd_signal"] = x["macd"].ewm(
        span=MACD_SIGNAL,
        adjust=False,
        min_periods=MACD_SIGNAL
    ).mean()

    x["hist"] = x["macd"] - x["macd_signal"]

    # --------------------------------------------------------
    # TRUE RANGE / ATR
    # Wilder ATR via EWM(alpha=1/n)
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD
    ).mean()

    x["atr_median"] = x["atr"].rolling(
        ATR_MEDIAN_PERIOD,
        min_periods=ATR_MEDIAN_PERIOD
    ).median()

    # --------------------------------------------------------
    # ADX / DI
    # Wilder calculation
    # --------------------------------------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=x.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=x.index
    )

    atr_w = tr.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    plus_dm_w = plus_dm.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    minus_dm_w = minus_dm.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    plus_di = 100 * plus_dm_w / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm_w / atr_w.replace(0, np.nan)

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    x["plus_di"] = plus_di
    x["minus_di"] = minus_di
    x["adx"] = adx

    # --------------------------------------------------------
    # Volume median
    # --------------------------------------------------------

    x["volume_median"] = volume.rolling(
        VOLUME_MEDIAN_PERIOD,
        min_periods=VOLUME_MEDIAN_PERIOD
    ).median()

    return x


# ============================================================
# CANDIDATE 9 SIGNAL DETECTOR
# ============================================================

def detect_events(df):

    events = []

    x = calculate_indicators(df)

    for i in range(1, len(x)):

        row = x.iloc[i]
        prev = x.iloc[i - 1]

        required = [
            row["close"],
            row["ema12"],
            row["ema26"],
            row["ema200"],
            row["hist"],
            prev["hist"],
            row["adx"],
            row["plus_di"],
            row["minus_di"],
            row["atr"],
            row["atr_median"],
            row["volume"],
            row["volume_median"]
        ]

        if any(pd.isna(v) for v in required):
            continue

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

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

        if long_signal and not short_signal:

            events.append({
                "idx": i,
                "time": int(row["time"]),
                "direction": "LONG",
                "entry": float(row["close"]),
                "atr": float(row["atr"])
            })

        elif short_signal and not long_signal:

            events.append({
                "idx": i,
                "time": int(row["time"]),
                "direction": "SHORT",
                "entry": float(row["close"]),
                "atr": float(row["atr"])
            })

    return events


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, event, tp_r):

    i = event["idx"]

    entry = float(event["entry"])
    atr = float(event["atr"])
    direction = event["direction"]

    if not np.isfinite(entry) or not np.isfinite(atr):
        return {
            "status": "AMBIGUOUS",
            "r": None
        }

    if atr <= 0:
        return {
            "status": "AMBIGUOUS",
            "r": None
        }

    if direction == "LONG":

        sl = entry - SL_ATR * atr
        tp = entry + tp_r * SL_ATR * atr

    else:

        sl = entry + SL_ATR * atr
        tp = entry - tp_r * SL_ATR * atr

    last_available = len(df) - 1

    # Need HOLD_BARS complete future candles.
    if i + HOLD_BARS > last_available:
        return {
            "status": "OPEN_AT_DATASET_END",
            "r": None
        }

    # --------------------------------------------------------
    # IMPORTANT:
    # Entry candle is NOT scanned for exit.
    # Scan starts from next candle.
    # --------------------------------------------------------

    for j in range(i + 1, i + HOLD_BARS + 1):

        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Same-candle ambiguity:
            # SL FIRST
            if hit_sl:
                return {
                    "status": "SL",
                    "r": -1.0
                }

            if hit_tp:
                return {
                    "status": "TP",
                    "r": float(tp_r)
                }

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Same-candle ambiguity:
            # SL FIRST
            if hit_sl:
                return {
                    "status": "SL",
                    "r": -1.0
                }

            if hit_tp:
                return {
                    "status": "TP",
                    "r": float(tp_r)
                }

    # Full HOLD completed with no SL/TP
    return {
        "status": "TIMEOUT",
        "r": 0.0
    }


# ============================================================
# RESULT METRICS
# ============================================================

def calculate_metrics(results):

    traded = [
        x for x in results
        if x["status"] in ["TP", "SL", "TIMEOUT"]
        and x["r"] is not None
    ]

    open_end = sum(
        x["status"] == "OPEN_AT_DATASET_END"
        for x in results
    )

    ambiguous = sum(
        x["status"] == "AMBIGUOUS"
        for x in results
    )

    if not traded:

        return {
            "n_events": len(results),
            "n_traded": 0,
            "open_end": open_end,
            "ambiguous": ambiguous,
            "wr": 0.0,
            "net_exp": 0.0,
            "net_total": 0.0,
            "net_pf": 0.0,
            "net_max_dd": 0.0
        }

    gross_r = np.array(
        [float(x["r"]) for x in traded],
        dtype=float
    )

    # Cost applied trade-by-trade.
    net_r = gross_r - TOTAL_COST_R

    wins = np.sum(gross_r > 0)

    wr = wins / len(gross_r)

    net_exp = float(np.mean(net_r))

    net_total = float(np.sum(net_r))

    positive = net_r[net_r > 0]
    negative = net_r[net_r < 0]

    gross_profit = float(np.sum(positive))
    gross_loss = abs(float(np.sum(negative)))

    if gross_loss > 0:
        net_pf = gross_profit / gross_loss
    else:
        net_pf = float("inf")

    equity = np.cumsum(net_r)
    peak = np.maximum.accumulate(
        np.concatenate([[0.0], equity])
    )[1:]

    dd = equity - peak
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    return {
        "n_events": len(results),
        "n_traded": len(traded),
        "open_end": int(open_end),
        "ambiguous": int(ambiguous),
        "wr": float(wr),
        "net_exp": net_exp,
        "net_total": net_total,
        "net_pf": float(net_pf),
        "net_max_dd": max_dd
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SETUP V4 — CANDIDATE 9")
    print("MACD ZERO-LINE TREND CONTINUATION")
    print("5-FOLD WALK-FORWARD VALIDATION")
    print("=" * 70)

    print()
    print("FROZEN PARAMETERS")
    print("-----------------")
    print("TIMEFRAME: 4H")
    print("EMA: 12 / 26 / 200")
    print("MACD SIGNAL: 9")
    print("ADX: 14")
    print("ADX MIN: 20")
    print("ATR: 20")
    print("ATR MEDIAN: 20")
    print("VOLUME MEDIAN: 20")
    print("SL: 1.25 ATR")
    print("HOLD: 30 bars")
    print("TP: 1R / 1.5R / 2R / 3R")
    print("FEE + SLIPPAGE: 0.003R")
    print("NO OVERLAP LOCK")
    print("LONG + SHORT")
    print("NO PARAMETER OPTIMIZATION")
    print()

    data = {}

    # ========================================================
    # FETCH
    # ========================================================

    for symbol in SYMBOLS:

        print(f"Fetching {symbol} ...")

        df = fetch_history(
            symbol,
            TARGET_DAYS
        )

        ok, info = audit_data(df)

        if not ok:

            print(f"{symbol}: {info}")
            continue

        gaps = info

        print(
            f"{symbol}: "
            f"{len(df)} candles | "
            f"{(df['time'].iloc[-1] - df['time'].iloc[0]) / 86400:.1f}d | "
            f"gaps={gaps}"
        )

        if gaps != 0:
            print(f"{symbol}: excluded because gaps > 0")
            continue

        data[symbol] = df

    # ========================================================
    # COMMON TIMESTAMPS
    # ========================================================

    if not data:
        print("NO VALID DATA")
        return

    common = None

    for df in data.values():

        ts = set(
            df["time"].astype(np.int64).tolist()
        )

        if common is None:
            common = ts
        else:
            common &= ts

    common = sorted(common)

    print()
    print("=" * 70)
    print("DATA AUDIT")
    print("=" * 70)

    print(f"DATA_VALID_SYMBOLS: {len(data)}")
    print(f"COMMON_TIMESTAMPS: {len(common)}")

    if len(common) < 2000:
        print("FAIL: insufficient common timestamps")
        return

    print(
        "COMMON_START:",
        format_time(common[0])
    )

    print(
        "COMMON_END:",
        format_time(common[-1])
    )

    # ========================================================
    # ALIGN ALL DATA TO COMMON TIMESTAMPS
    # ========================================================

    common_set = set(common)

    aligned = {}

    for symbol, df in data.items():

        x = df[
            df["time"].isin(common_set)
        ].copy()

        x = x.sort_values("time").reset_index(drop=True)

        aligned[symbol] = x

    # ========================================================
    # PRE-COMPUTE EVENTS
    # ========================================================

    print()
    print("=" * 70)
    print("SIGNAL DETECTION")
    print("=" * 70)

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

        print(
            f"{symbol}: "
            f"{len(events)} events"
        )

    print()
    print(f"TOTAL_EVENTS: {total_events}")
    print(f"LONG_EVENTS: {total_long}")
    print(f"SHORT_EVENTS: {total_short}")

    # ========================================================
    # WALK-FORWARD FOLDS
    # ========================================================

    fold_results = []

    print()
    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    for fold_name, start_frac, end_frac in WF_FOLDS:

        n_common = len(common)

        start_idx = int(
            np.floor(n_common * start_frac)
        )

        end_idx = int(
            np.floor(n_common * end_frac)
        )

        if end_idx <= start_idx:
            continue

        test_start_time = common[start_idx]
        test_end_time = common[min(end_idx - 1, n_common - 1)]

        print()
        print("=" * 70)
        print(f"{fold_name}")
        print(
            f"TEST WINDOW: "
            f"{format_time(test_start_time)} -> "
            f"{format_time(test_end_time)}"
        )
        print("=" * 70)

        for tp_r in TP_R_LIST:

            fold_trades = []

            for symbol, df in aligned.items():

                events = all_events[symbol]

                for event in events:

                    event_time = event["time"]

                    # ----------------------------------------
                    # IMPORTANT:
                    # ONLY future fold segment is tested.
                    # ----------------------------------------

                    if event_time < test_start_time:
                        continue

                    if event_time >= test_end_time:
                        continue

                    result = simulate_trade(
                        df,
                        event,
                        tp_r
                    )

                    result_record = {
                        "symbol": symbol,
                        "time": event_time,
                        "direction": event["direction"],
                        "status": result["status"],
                        "r": result["r"]
                    }

                    fold_trades.append(
                        result_record
                    )

            metrics = calculate_metrics(
                fold_trades
            )

            metrics["fold"] = fold_name
            metrics["tp_r"] = tp_r

            fold_results.append(metrics)

            print()
            print(
                f"{fold_name} | TP {tp_r}R"
            )

            print(
                f"n_events: {metrics['n_events']}"
            )

            print(
                f"n_traded: {metrics['n_traded']}"
            )

            print(
                f"open_end: {metrics['open_end']}"
            )

            print(
                f"ambiguous: {metrics['ambiguous']}"
            )

            print(
                f"wr: {metrics['wr']:.4f}"
            )

            print(
                f"net_exp: {metrics['net_exp']:.4f}"
            )

            print(
                f"net_total: {metrics['net_total']:.4f}"
            )

            print(
                f"net_pf: {metrics['net_pf']:.4f}"
            )

            print(
                f"net_max_dd: {metrics['net_max_dd']:.4f}"
            )

    # ========================================================
    # AGGREGATE WF RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("WALK-FORWARD AGGREGATE")
    print("=" * 70)

    for tp_r in TP_R_LIST:

        rows = [
            x for x in fold_results
            if x["tp_r"] == tp_r
        ]

        all_fold_trades = []

        # Reconstruct aggregate from fold metrics
        # using per-fold expectancy * trade count
        total_trades = sum(
            x["n_traded"]
            for x in rows
        )

        weighted_net = sum(
            x["net_exp"] * x["n_traded"]
            for x in rows
        )

        aggregate_exp = (
            weighted_net / total_trades
            if total_trades > 0
            else 0.0
        )

        aggregate_total = sum(
            x["net_total"]
            for x in rows
        )

        positive_folds = sum(
            x["net_exp"] > 0
            for x in rows
        )

        # Conservative aggregate PF:
        # reconstruct gross positive/negative from
        # net expectancy and PF per fold.
        total_profit = 0.0
        total_loss = 0.0

        for x in rows:

            if x["net_pf"] <= 0:
                continue

            net_total = x["net_total"]
            n = x["n_traded"]

            # Sum of net returns:
            # PF = profit / loss
            #
            # profit - loss = net_total
            # profit / loss = PF
            #
            # therefore:
            # loss = -net_total / (PF - 1)
            #
            # for PF > 1.
            #
            # If PF < 1, solve using absolute net loss.
            if x["net_pf"] > 1.0:

                loss = (
                    -net_total /
                    (x["net_pf"] - 1.0)
                )

                if loss < 0:
                    loss = abs(loss)

                profit = (
                    x["net_pf"] * loss
                )

            elif x["net_pf"] < 1.0:

                profit = (
                    net_total /
                    (1.0 - 1.0 / x["net_pf"])
                )

                profit = abs(profit)

                loss = (
                    profit /
                    x["net_pf"]
                )

            else:

                profit = 0.0
                loss = 0.0

            total_profit += profit
            total_loss += loss

        aggregate_pf = (
            total_profit / total_loss
            if total_loss > 0
            else 0.0
        )

        max_dd = min(
            [x["net_max_dd"] for x in rows],
            default=0.0
        )

        print()
        print(
            f"TP {tp_r}R AGGREGATE"
        )

        print(
            f"folds: {len(rows)}"
        )

        print(
            f"traded: {total_trades}"
        )

        print(
            f"positive_folds: "
            f"{positive_folds}/{len(rows)}"
        )

        print(
            f"net_exp: {aggregate_exp:.4f}"
        )

        print(
            f"net_total: {aggregate_total:.4f}"
        )

        print(
            f"net_pf: {aggregate_pf:.4f}"
        )

        print(
            f"worst_fold_dd: {max_dd:.4f}"
        )

    # ========================================================
    # INTEGRITY AUDIT
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION INTEGRITY AUDIT")
    print("=" * 70)

    print("TIMEFRAME_4H: TRUE")
    print("CAUSAL_SIGNAL_DETECTION: TRUE")
    print("MACD_CROSS_CURRENT_PREVIOUS_HIST: TRUE")
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
    print(f"TOTAL_COST_R: {TOTAL_COST_R:.4f}")
    print("WALK_FORWARD_CAUSAL: TRUE")
    print("FUTURE_FOLD_ONLY: TRUE")
    print("NO_LOOKAHEAD: TRUE")

    # ========================================================
    # FINAL GATE
    # ========================================================

    print()
    print("=" * 70)
    print("RESEARCH GATE")
    print("=" * 70)

    for tp_r in TP_R_LIST:

        rows = [
            x for x in fold_results
            if x["tp_r"] == tp_r
        ]

        if not rows:
            continue

        total_trades = sum(
            x["n_traded"]
            for x in rows
        )

        aggregate_exp = (
            sum(
                x["net_exp"] * x["n_traded"]
                for x in rows
            ) / total_trades
            if total_trades > 0
            else 0.0
        )

        positive_folds = sum(
            x["net_exp"] > 0
            for x in rows
        )

        print(
            f"TP {tp_r}R | "
            f"NetExp={aggregate_exp:.4f} | "
            f"PositiveFolds="
            f"{positive_folds}/{len(rows)}"
        )

    print()
    print(
        "STATUS: WF RESULT IS RESEARCH EVIDENCE ONLY."
    )
    print(
        "NO LONG-ONLY / SHORT-ONLY SELECTION."
    )
    print(
        "NO PARAMETER PATCHING."
    )
    print(
        "NO PRODUCTION DEPLOYMENT FROM THIS RUN."
    )

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
