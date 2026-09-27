# ============================================================
# SETUP V4 — CANDIDATE 6
# VOLATILITY EXHAUSTION MEAN REVERSION
# ============================================================
#
# Research-only strategy.
#
# Frozen research specification:
#   TF              = 4H
#   BB_PERIOD       = 20
#   BB_STD          = 2.0
#   RSI_PERIOD      = 14
#   RSI_LONG        = 30
#   RSI_SHORT       = 70
#   ATR_PERIOD      = 20
#   ADX_PERIOD      = 14
#   ADX_MAX         = 30
#   SL_ATR          = 1.25
#   HOLD_BARS       = 30
#
# Entry:
#   LONG:
#       previous close < previous lower BB
#       current close >= current lower BB
#       previous RSI <= 30
#       current ADX < 30
#
#   SHORT:
#       previous close > previous upper BB
#       current close <= current upper BB
#       previous RSI >= 70
#       current ADX < 30
#
# Execution:
#   entry = signal candle close
#   SL = 1.25 ATR
#   TP = 1 / 1.5 / 2 / 3 R
#   same-candle SL-first
#   timeout = 0R
#   incomplete final HOLD window = OPEN_AT_DATASET_END
#
# Costs:
#   fee = 0.10%
#   slippage = 0.05%
#
# IMPORTANT:
#   No parameter optimization.
#   No post-hoc filters.
#   No look-ahead.
#   IS discovery only.
#
# ============================================================

import time
import math
import requests
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.kucoin.com"

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
    "LTC-USDT",
    "BCH-USDT",
    "UNI-USDT",
    "AAVE-USDT",
    "ATOM-USDT",
    "NEAR-USDT",
    "FIL-USDT",
    "ETC-USDT",
    "ICP-USDT",
    "APT-USDT",
    "ARB-USDT",
    "OP-USDT",
    "SUI-USDT",
    "SEI-USDT",
    "INJ-USDT",
    "TIA-USDT",
    "JUP-USDT",
    "PEPE-USDT",
    "SHIB-USDT",
    "TRX-USDT",
    "TON-USDT",
    "HBAR-USDT",
    "VET-USDT",
    "ALGO-USDT",
    "MATIC-USDT",
    "STX-USDT",
    "RUNE-USDT",
    "MKR-USDT",
]

# 4H
KLINE_TYPE = "4hour"

TARGET_DAYS = 730

# Strategy
BB_PERIOD = 20
BB_STD = 2.0

RSI_PERIOD = 14
RSI_LONG = 30.0
RSI_SHORT = 70.0

ATR_PERIOD = 20

ADX_PERIOD = 14
ADX_MAX = 30.0

SL_ATR = 1.25
HOLD_BARS = 30

TP_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]

# Costs
FEE_RATE = 0.0010
SLIPPAGE_RATE = 0.0005

# KuCoin limits
KUCOIN_MAX_CANDLES = 1500

REQUEST_SLEEP = 0.12
REQUEST_TIMEOUT = 20

# ============================================================
# HTTP
# ============================================================

session = requests.Session()


def kucoin_get(endpoint, params=None, retries=5):
    last_error = None

    for attempt in range(retries):
        try:
            r = session.get(
                BASE_URL + endpoint,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:
                payload = r.json()

                if payload.get("code") == "200000":
                    return payload.get("data", [])

                last_error = RuntimeError(
                    f"KuCoin API error: {payload}"
                )

            else:
                last_error = RuntimeError(
                    f"HTTP {r.status_code}: {r.text[:300]}"
                )

        except Exception as e:
            last_error = e

        time.sleep(1.0 + attempt)

    raise last_error


# ============================================================
# DATA LAYER
# ============================================================

def fetch_history(symbol, target_days=730):
    """
    Fetch approximately target_days of 4H candles.

    KuCoin response:
        [time, open, close, high, low, volume, turnover]

    Data returned:
        oldest -> newest
    """

    now = int(time.time())
    start_at = now - int(target_days * 86400)

    rows = []
    cursor_start = start_at

    max_iterations = 100

    for _ in range(max_iterations):

        data = kucoin_get(
            "/api/v1/market/candles",
            {
                "symbol": symbol,
                "type": KLINE_TYPE,
                "startAt": cursor_start,
                "endAt": now,
            },
        )

        if not data:
            break

        rows.extend(data)

        parsed_times = []

        for row in data:
            try:
                parsed_times.append(int(row[0]))
            except Exception:
                pass

        if not parsed_times:
            break

        oldest = min(parsed_times)

        # Prevent pagination stall
        if oldest <= cursor_start:
            break

        cursor_start = oldest - 4 * 3600

        # We already have enough data
        if oldest <= start_at:
            break

        if len(rows) >= 10000:
            break

        time.sleep(REQUEST_SLEEP)

    if not rows:
        return pd.DataFrame()

    records = []

    for row in rows:
        try:
            ts = int(row[0])

            records.append(
                {
                    "timestamp": ts,
                    "open": float(row[1]),
                    "close": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "volume": float(row[5]),
                    "turnover": float(row[6]),
                }
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove candles outside requested range
    df = df[df["timestamp"] >= start_at].copy()

    # --------------------------------------------------------
    # Remove incomplete current candle
    # --------------------------------------------------------

    current_4h_start = (int(time.time()) // (4 * 3600)) * (4 * 3600)

    df = df[df["timestamp"] < current_4h_start].copy()

    df = df.reset_index(drop=True)

    return df


# ============================================================
# DATA AUDIT
# ============================================================

def audit_data(df):
    if df.empty:
        return {
            "sufficient": False,
            "monotonic": False,
            "no_duplicates": False,
            "gap_count": None,
            "days": 0.0,
        }

    ts = df["timestamp"].astype(np.int64)

    monotonic = bool(ts.is_monotonic_increasing)
    no_duplicates = bool(ts.nunique() == len(ts))

    diffs = ts.diff().dropna()

    expected = 4 * 3600

    gap_count = int((diffs != expected).sum())

    days = 0.0

    if len(df) > 1:
        days = float(
            (ts.iloc[-1] - ts.iloc[0]) / 86400.0
        )

    sufficient = bool(days >= 180.0)

    return {
        "sufficient": sufficient,
        "monotonic": monotonic,
        "no_duplicates": no_duplicates,
        "gap_count": gap_count,
        "days": days,
    }


# ============================================================
# INDICATORS
# ============================================================

def true_range(df):

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (df["high"] - prev_close).abs()

    tr3 = (df["low"] - prev_close).abs()

    return pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)


def calculate_atr(df, period=20):

    tr = true_range(df)

    # Wilder-style ATR using EWM alpha=1/period
    atr = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def calculate_rsi(df, period=14):

    delta = df["close"].diff()

    gain = delta.clip(lower=0.0)

    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)

    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle persistent gains/losses
    rsi = rsi.where(
        avg_loss != 0,
        100.0,
    )

    rsi = rsi.where(
        avg_gain != 0,
        0.0,
    )

    both_zero = (
        (avg_gain == 0) &
        (avg_loss == 0)
    )

    rsi = rsi.where(
        ~both_zero,
        50.0,
    )

    return rsi


def calculate_adx(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    tr = true_range(df)

    atr = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100.0 *
        plus_dm.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr
    )

    minus_di = (
        100.0 *
        minus_dm.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr
    )

    denominator = (
        plus_di + minus_di
    ).replace(0.0, np.nan)

    dx = (
        100.0 *
        (plus_di - minus_di).abs()
        / denominator
    )

    adx = dx.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx


def add_indicators(df):

    df = df.copy()

    middle = (
        df["close"]
        .rolling(
            BB_PERIOD,
            min_periods=BB_PERIOD,
        )
        .mean()
    )

    std = (
        df["close"]
        .rolling(
            BB_PERIOD,
            min_periods=BB_PERIOD,
        )
        .std(ddof=0)
    )

    df["bb_mid"] = middle
    df["bb_upper"] = middle + BB_STD * std
    df["bb_lower"] = middle - BB_STD * std

    df["rsi"] = calculate_rsi(
        df,
        RSI_PERIOD,
    )

    df["atr"] = calculate_atr(
        df,
        ATR_PERIOD,
    )

    df["adx"] = calculate_adx(
        df,
        ADX_PERIOD,
    )

    return df


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_events(df):
    """
    Strictly causal.

    Event at i uses:
        previous candle indicators/close
        current candle close and current indicators

    No future candle is used.
    """

    events = []

    if len(df) < 100:
        return events

    for i in range(1, len(df)):

        prev = df.iloc[i - 1]
        cur = df.iloc[i]

        required_prev = [
            prev["close"],
            prev["bb_lower"],
            prev["bb_upper"],
            prev["rsi"],
        ]

        required_cur = [
            cur["close"],
            cur["bb_lower"],
            cur["bb_upper"],
            cur["rsi"],
            cur["atr"],
            cur["adx"],
        ]

        if not all(
            np.isfinite(x)
            for x in required_prev + required_cur
        ):
            continue

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        long_signal = (
            prev["close"] < prev["bb_lower"]
            and cur["close"] >= cur["bb_lower"]
            and prev["rsi"] <= RSI_LONG
            and cur["adx"] < ADX_MAX
        )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        short_signal = (
            prev["close"] > prev["bb_upper"]
            and cur["close"] <= cur["bb_upper"]
            and prev["rsi"] >= RSI_SHORT
            and cur["adx"] < ADX_MAX
        )

        if long_signal:
            events.append(
                {
                    "idx": i,
                    "timestamp": int(cur["timestamp"]),
                    "direction": "LONG",
                    "atr": float(cur["atr"]),
                    "entry": float(cur["close"]),
                    "rsi": float(prev["rsi"]),
                    "adx": float(cur["adx"]),
                    "reason": "BB_REENTRY_RSI_EXHAUSTION",
                }
            )

        elif short_signal:
            events.append(
                {
                    "idx": i,
                    "timestamp": int(cur["timestamp"]),
                    "direction": "SHORT",
                    "atr": float(cur["atr"]),
                    "entry": float(cur["close"]),
                    "rsi": float(prev["rsi"]),
                    "adx": float(cur["adx"]),
                    "reason": "BB_REENTRY_RSI_EXHAUSTION",
                }
            )

    return events


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    event,
    tp_multiple,
):
    """
    Entry at event candle close.

    Important:
    The entry candle itself is NOT used for TP/SL evaluation.
    Exit scanning starts from the next candle.

    This avoids artificially assuming that the trade could
    enter at the close and also have captured the same candle's
    high/low before entry.
    """

    idx = int(event["idx"])

    entry = float(event["entry"])
    atr = float(event["atr"])
    direction = event["direction"]

    if not np.isfinite(entry) or not np.isfinite(atr):
        return {
            "status": "INVALID",
        }

    if atr <= 0:
        return {
            "status": "INVALID",
        }

    risk = SL_ATR * atr

    if direction == "LONG":

        sl = entry - risk
        tp = entry + tp_multiple * risk

        # Apply adverse execution slippage to stop/TP
        sl_exec = sl * (1.0 - SLIPPAGE_RATE)
        tp_exec = tp * (1.0 - SLIPPAGE_RATE)

    else:

        sl = entry + risk
        tp = entry - tp_multiple * risk

        sl_exec = sl * (1.0 + SLIPPAGE_RATE)
        tp_exec = tp * (1.0 + SLIPPAGE_RATE)

    last_available = len(df) - 1

    full_exit_idx = idx + HOLD_BARS

    # --------------------------------------------------------
    # Not enough future candles
    # --------------------------------------------------------

    if full_exit_idx > last_available:

        return {
            "status": "OPEN_AT_DATASET_END",
            "entry_idx": idx,
            "entry_time": int(df.iloc[idx]["timestamp"]),
            "direction": direction,
        }

    # --------------------------------------------------------
    # Scan future candles
    # --------------------------------------------------------

    for j in range(idx + 1, full_exit_idx + 1):

        candle = df.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])

        hit_sl = False
        hit_tp = False

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

        # ----------------------------------------------------
        # Same-candle ambiguity:
        # SL FIRST
        # ----------------------------------------------------

        if hit_sl:

            gross_r = -1.0

            # Approximate cost as entry + exit transaction cost.
            net_r = (
                gross_r
                - 2.0 * FEE_RATE
                - 2.0 * SLIPPAGE_RATE
            )

            return {
                "status": "SL",
                "entry_idx": idx,
                "exit_idx": j,
                "entry_time": int(df.iloc[idx]["timestamp"]),
                "exit_time": int(candle["timestamp"]),
                "direction": direction,
                "gross_r": gross_r,
                "net_r": net_r,
                "bars_held": j - idx,
            }

        if hit_tp:

            gross_r = float(tp_multiple)

            net_r = (
                gross_r
                - 2.0 * FEE_RATE
                - 2.0 * SLIPPAGE_RATE
            )

            return {
                "status": "TP",
                "entry_idx": idx,
                "exit_idx": j,
                "entry_time": int(df.iloc[idx]["timestamp"]),
                "exit_time": int(candle["timestamp"]),
                "direction": direction,
                "gross_r": gross_r,
                "net_r": net_r,
                "bars_held": j - idx,
            }

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    exit_idx = full_exit_idx

    candle = df.iloc[exit_idx]

    return {
        "status": "TIMEOUT",
        "entry_idx": idx,
        "exit_idx": exit_idx,
        "entry_time": int(df.iloc[idx]["timestamp"]),
        "exit_time": int(candle["timestamp"]),
        "direction": direction,
        "gross_r": 0.0,
        "net_r": (
            -2.0 * FEE_RATE
            -2.0 * SLIPPAGE_RATE
        ),
        "bars_held": HOLD_BARS,
    }


# ============================================================
# MAX DRAWDOWN
# ============================================================

def calculate_max_dd(values):

    if not values:
        return 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for value in values:

        equity += float(value)

        if equity > peak:
            peak = equity

        dd = equity - peak

        if dd < max_dd:
            max_dd = dd

    return float(max_dd)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    traded = [
        t for t in trades
        if t["status"] in {
            "TP",
            "SL",
            "TIMEOUT",
        }
    ]

    open_end = [
        t for t in trades
        if t["status"] == "OPEN_AT_DATASET_END"
    ]

    ambiguous = [
        t for t in trades
        if t["status"] == "AMBIGUOUS"
    ]

    n = len(traded)

    if n == 0:

        return {
            "n_traded": 0,
            "open_end": len(open_end),
            "ambiguous": len(ambiguous),
            "wr": 0.0,
            "gross_exp": 0.0,
            "net_exp": 0.0,
            "gross_total": 0.0,
            "net_total": 0.0,
            "gross_pf": 0.0,
            "net_pf": 0.0,
            "gross_maxdd": 0.0,
            "net_maxdd": 0.0,
        }

    gross = [
        float(t["gross_r"])
        for t in traded
    ]

    net = [
        float(t["net_r"])
        for t in traded
    ]

    wins = [
        x for x in gross
        if x > 0
    ]

    gross_losses = [
        abs(x) for x in gross
        if x < 0
    ]

    net_wins = [
        x for x in net
        if x > 0
    ]

    net_losses = [
        abs(x) for x in net
        if x < 0
    ]

    gross_profit = sum(wins)
    gross_loss = sum(gross_losses)

    net_profit = sum(net_wins)
    net_loss = sum(net_losses)

    gross_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    net_pf = (
        net_profit / net_loss
        if net_loss > 0
        else float("inf")
    )

    return {
        "n_traded": n,
        "open_end": len(open_end),
        "ambiguous": len(ambiguous),

        "wr": len(wins) / n,

        "gross_exp": float(np.mean(gross)),
        "net_exp": float(np.mean(net)),

        "gross_total": float(sum(gross)),
        "net_total": float(sum(net)),

        "gross_pf": float(gross_pf),
        "net_pf": float(net_pf),

        "gross_maxdd": calculate_max_dd(gross),
        "net_maxdd": calculate_max_dd(net),
    }


# ============================================================
# SYMBOL TEST
# ============================================================

def run_symbol(symbol, df):

    audit = audit_data(df)

    if not audit["sufficient"]:
        return {
            "symbol": symbol,
            "status": "INSUFFICIENT_HISTORY",
            "audit": audit,
            "events": [],
            "results": {},
        }

    if not audit["monotonic"] or not audit["no_duplicates"]:
        return {
            "symbol": symbol,
            "status": "DATA_AUDIT_FAIL",
            "audit": audit,
            "events": [],
            "results": {},
        }

    if audit["gap_count"] > 0:
        return {
            "symbol": symbol,
            "status": "DATA_GAP",
            "audit": audit,
            "events": [],
            "results": {},
        }

    df = add_indicators(df)

    events = detect_events(df)

    results = {}

    for tp in TP_MULTIPLIERS:

        trades = []

        for event in events:

            trade = simulate_trade(
                df,
                event,
                tp,
            )

            if trade["status"] != "INVALID":
                trades.append(trade)

        results[tp] = calculate_metrics(trades)

    return {
        "symbol": symbol,
        "status": "OK",
        "audit": audit,
        "events": events,
        "results": results,
    }


# ============================================================
# PORTFOLIO AGGREGATION
# ============================================================

def aggregate_portfolio(symbol_results):

    portfolio = {}

    for tp in TP_MULTIPLIERS:

        all_trades = []

        for result in symbol_results:

            if result["status"] != "OK":
                continue

            symbol = result["symbol"]

            df_events = result["events"]

            # Reconstruct trade list from stored symbol data
            # through the stored metric values is insufficient,
            # so this aggregation is performed separately below.
            #
            # This block intentionally remains empty.
            _ = symbol
            _ = df_events

        portfolio[tp] = None

    return portfolio


# ============================================================
# PORTFOLIO RUN
# ============================================================

def run_all():

    print("=" * 72)
    print("SETUP V4 — CANDIDATE 6")
    print("VOLATILITY EXHAUSTION MEAN REVERSION")
    print("=" * 72)

    print("")
    print("FROZEN SPEC")
    print(f"TF              : {KLINE_TYPE}")
    print(f"BB               : {BB_PERIOD}, {BB_STD} std")
    print(f"RSI              : {RSI_PERIOD}")
    print(f"RSI LONG         : <= {RSI_LONG}")
    print(f"RSI SHORT        : >= {RSI_SHORT}")
    print(f"ATR              : {ATR_PERIOD}")
    print(f"ADX              : {ADX_PERIOD}")
    print(f"ADX MAX          : < {ADX_MAX}")
    print(f"SL               : {SL_ATR} ATR")
    print(f"HOLD             : {HOLD_BARS}")
    print(f"TP               : {TP_MULTIPLIERS}")
    print(f"FEE              : {FEE_RATE * 100:.2f}%")
    print(f"SLIPPAGE         : {SLIPPAGE_RATE * 100:.2f}%")
    print("")

    symbol_results = []

    total_loaded = 0
    total_sufficient = 0
    total_events = 0

    # --------------------------------------------------------
    # FETCH + TEST
    # --------------------------------------------------------

    for n, symbol in enumerate(SYMBOLS, 1):

        print(
            f"[{n:02d}/{len(SYMBOLS)}] "
            f"Loading {symbol} ..."
        )

        try:

            df = fetch_history(
                symbol,
                TARGET_DAYS,
            )

            if df.empty:

                print(
                    f"    -> EMPTY"
                )

                symbol_results.append(
                    {
                        "symbol": symbol,
                        "status": "EMPTY",
                        "audit": {},
                        "events": [],
                        "results": {},
                    }
                )

                continue

            total_loaded += 1

            result = run_symbol(
                symbol,
                df,
            )

            symbol_results.append(result)

            if result["status"] == "OK":

                total_sufficient += 1

                n_events = len(
                    result["events"]
                )

                total_events += n_events

                audit = result["audit"]

                print(
                    f"    candles={len(df)} "
                    f"days={audit['days']:.1f} "
                    f"gaps={audit['gap_count']} "
                    f"events={n_events}"
                )

            else:

                print(
                    f"    -> {result['status']}"
                )

        except Exception as e:

            print(
                f"    -> ERROR: {type(e).__name__}: {e}"
            )

            symbol_results.append(
                {
                    "symbol": symbol,
                    "status": "ERROR",
                    "audit": {},
                    "events": [],
                    "results": {},
                }
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # Re-simulate trades for portfolio metrics.
    #
    # This is intentionally deterministic and uses exactly
    # the same event list and simulator as symbol-level tests.
    # --------------------------------------------------------

    portfolio_metrics = {}

    for tp in TP_MULTIPLIERS:

        all_trades = []

        for result in symbol_results:

            if result["status"] != "OK":
                continue

            symbol = result["symbol"]

            # We need the original dataframe again only if
            # portfolio trade-level data is required.
            # To avoid duplicate API calls, store dataframes
            # in memory below.
            _ = symbol

        portfolio_metrics[tp] = {
            "trades": [],
        }

    # --------------------------------------------------------
    # Re-run using in-memory dataframes for exact portfolio
    # aggregation.
    # --------------------------------------------------------

    portfolio_trade_map = {
        tp: []
        for tp in TP_MULTIPLIERS
    }

    per_symbol_rows = []

    # Fetching again would be wasteful. Therefore build
    # deterministic portfolio metrics directly from the
    # symbol metrics using weighted trade counts/returns.
    #
    # For MaxDD we need chronological trades. We reconstruct
    # them below from fresh local data only if needed.
    #
    # Since the GitHub runner has enough time, perform a second
    # deterministic pass over the already-known symbols.
    #
    # This also gives us a complete portfolio chronology.

    print("")
    print("Building portfolio trade ledger ...")

    valid_symbols = [
        r["symbol"]
        for r in symbol_results
        if r["status"] == "OK"
    ]

    portfolio_ledger = {
        tp: []
        for tp in TP_MULTIPLIERS
    }

    for n, symbol in enumerate(valid_symbols, 1):

        print(
            f"  Portfolio pass "
            f"[{n:02d}/{len(valid_symbols)}] "
            f"{symbol}"
        )

        try:

            df = fetch_history(
                symbol,
                TARGET_DAYS,
            )

            if df.empty:
                continue

            df = add_indicators(df)

            events = detect_events(df)

            for tp in TP_MULTIPLIERS:

                for event in events:

                    trade = simulate_trade(
                        df,
                        event,
                        tp,
                    )

                    trade["symbol"] = symbol

                    portfolio_ledger[tp].append(
                        trade
                    )

        except Exception as e:

            print(
                f"    portfolio pass error: {e}"
            )

    # --------------------------------------------------------
    # Portfolio metrics
    # --------------------------------------------------------

    print("")
    print("=" * 72)
    print("PORTFOLIO RESULTS")
    print("=" * 72)

    for tp in TP_MULTIPLIERS:

        ledger = portfolio_ledger[tp]

        metrics = calculate_metrics(
            ledger
        )

        portfolio_metrics[tp] = metrics

        print("")
        print(f"TP = {tp}R")
        print(
            f"  n_traded       = {metrics['n_traded']}"
        )
        print(
            f"  open_end       = {metrics['open_end']}"
        )
        print(
            f"  ambiguous      = {metrics['ambiguous']}"
        )
        print(
            f"  WR             = {metrics['wr']:.4f}"
        )
        print(
            f"  Gross Exp      = {metrics['gross_exp']:+.4f}R"
        )
        print(
            f"  Net Exp        = {metrics['net_exp']:+.4f}R"
        )
        print(
            f"  Gross Total    = {metrics['gross_total']:+.2f}R"
        )
        print(
            f"  Net Total      = {metrics['net_total']:+.2f}R"
        )
        print(
            f"  Gross PF       = {metrics['gross_pf']:.3f}"
        )
        print(
            f"  Net PF         = {metrics['net_pf']:.3f}"
        )
        print(
            f"  Gross MaxDD    = {metrics['gross_maxdd']:+.2f}R"
        )
        print(
            f"  Net MaxDD      = {metrics['net_maxdd']:+.2f}R"
        )

    # --------------------------------------------------------
    # Per-symbol summary
    # --------------------------------------------------------

    print("")
    print("=" * 72)
    print("PER-SYMBOL RESULTS — TP 2R")
    print("=" * 72)

    tp_for_symbol = 2.0

    for result in symbol_results:

        if result["status"] != "OK":
            continue

        symbol = result["symbol"]

        m = result["results"].get(
            tp_for_symbol
        )

        if not m:
            continue

        print(
            f"{symbol:12s} "
            f"events={len(result['events']):4d} "
            f"traded={m['n_traded']:4d} "
            f"WR={m['wr']:.3f} "
            f"NetE={m['net_exp']:+.4f}R "
            f"NetPF={m['net_pf']:.3f} "
            f"NetTotal={m['net_total']:+.2f}R"
        )

    # --------------------------------------------------------
    # Direction split
    # --------------------------------------------------------

    print("")
    print("=" * 72)
    print("DIRECTION SPLIT — TP 2R")
    print("=" * 72)

    tp = 2.0

    direction_groups = {
        "LONG": [],
        "SHORT": [],
    }

    for trade in portfolio_ledger[tp]:

        if trade["status"] in {
            "TP",
            "SL",
            "TIMEOUT",
        }:

            direction_groups[
                trade["direction"]
            ].append(trade)

    for direction, trades in direction_groups.items():

        m = calculate_metrics(trades)

        print("")
        print(direction)

        print(
            f"  n_traded    = {m['n_traded']}"
        )

        print(
            f"  WR          = {m['wr']:.4f}"
        )

        print(
            f"  Gross Exp   = {m['gross_exp']:+.4f}R"
        )

        print(
            f"  Net Exp     = {m['net_exp']:+.4f}R"
        )

        print(
            f"  Net Total   = {m['net_total']:+.2f}R"
        )

        print(
            f"  Net PF      = {m['net_pf']:.3f}"
        )

        print(
            f"  Net MaxDD   = {m['net_maxdd']:+.2f}R"
        )

    # --------------------------------------------------------
    # Signal diagnostics
    # --------------------------------------------------------

    all_events = []

    for result in symbol_results:

        if result["status"] == "OK":

            for event in result["events"]:

                event_copy = dict(event)

                event_copy[
                    "symbol"
                ] = result["symbol"]

                all_events.append(
                    event_copy
                )

    long_events = [
        e for e in all_events
        if e["direction"] == "LONG"
    ]

    short_events = [
        e for e in all_events
        if e["direction"] == "SHORT"
    ]

    print("")
    print("=" * 72)
    print("SIGNAL DIAGNOSTICS")
    print("=" * 72)

    print(
        f"Total events       = {len(all_events)}"
    )

    print(
        f"Long events        = {len(long_events)}"
    )

    print(
        f"Short events       = {len(short_events)}"
    )

    if all_events:

        rsis = [
            e["rsi"]
            for e in all_events
            if np.isfinite(e["rsi"])
        ]

        adxs = [
            e["adx"]
            for e in all_events
            if np.isfinite(e["adx"])
        ]

        if rsis:
            print(
                f"Event RSI mean     = {np.mean(rsis):.2f}"
            )

        if adxs:
            print(
                f"Event ADX mean     = {np.mean(adxs):.2f}"
            )

    # --------------------------------------------------------
    # Research decision
    # --------------------------------------------------------

    print("")
    print("=" * 72)
    print("RESEARCH DECISION")
    print("=" * 72)

    # Do NOT optimize or cherry-pick.
    #
    # Primary discovery reference:
    #   TP 2R net expectancy
    #   TP 2R net PF
    #
    # A positive result is not automatically production-ready.
    # It only earns the right to proceed to OOS validation.

    primary = portfolio_metrics.get(
        2.0,
        {}
    )

    if (
        primary
        and primary.get("n_traded", 0) >= 100
        and primary.get("net_exp", -999) > 0
        and primary.get("net_pf", 0) > 1.0
    ):

        print(
            "STATUS = IS SURVIVES INITIAL DISCOVERY"
        )

        print(
            "NEXT   = RUN OOS + WALK-FORWARD VALIDATION"
        )

    else:

        print(
            "STATUS = CANDIDATE 6 DOES NOT CLEAR "
            "INITIAL IS DISCOVERY GATE"
        )

        print(
            "NEXT   = ARCHIVE CANDIDATE 6"
        )

    print("")
    print("=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_all()
