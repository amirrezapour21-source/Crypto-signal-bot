# ============================================================
# SETUP V4 — CANDIDATE 6
# VOLATILITY EXHAUSTION MEAN REVERSION
# ============================================================
# Stage:
# 730D DATA FIX + IS/OOS VALIDATION
#
# Frozen Strategy:
# 4H
# BB(20, 2.0)
# RSI(14)
# ATR(20)
# ADX(14)
# ADX < 30
# Long: previous close < previous lower BB
#       current close >= current lower BB
#       previous RSI <= 30
#       current ADX < 30
# Short: inverse
#
# Risk:
# SL = 1.25 ATR
# TP = 1 / 1.5 / 2 / 3 R
# HOLD = 30 candles
#
# Costs:
# Fee = 0.10% per side
# Slippage = 0.05% per side
# Total cost = 0.30% = 0.003R
#
# IMPORTANT:
# - No parameter optimization
# - No short-only selection
# - No additional filters
# - No overlap lock
# - Causal detection
# - Same-candle SL-first
# - Entry candle is NOT scanned for exits
# - Last incomplete HOLD => OPEN_AT_DATASET_END
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

BASE_URL = "https://api.kucoin.com"

INTERVAL = "4hour"
INTERVAL_MS = 4 * 60 * 60 * 1000

TARGET_DAYS = 730
TARGET_CANDLES = int(TARGET_DAYS * 24 / 4)

PAGE_LIMIT = 1500

BB_PERIOD = 20
BB_STD = 2.0

RSI_PERIOD = 14
ATR_PERIOD = 20
ADX_PERIOD = 14

ADX_MAX = 30.0
RSI_LONG_MAX = 30.0
RSI_SHORT_MIN = 70.0

SL_ATR = 1.25

HOLD_BARS = 30

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005

TOTAL_COST_R = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)

TP_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]

# Frozen valid universe for Candidate 6.
# TON / MATIC / MKR were rejected by current KuCoin API.
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
    "HBAR-USDT",
    "VET-USDT",
    "ALGO-USDT",
    "STX-USDT",
    "RUNE-USDT",
]


# ============================================================
# HELPERS
# ============================================================

def utc_str(ms):
    return datetime.fromtimestamp(
        ms / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# KUCOIN DATA
# ============================================================

def fetch_kucoin_page(symbol, start_ms, end_ms):
    url = f"{BASE_URL}/api/v1/market/candles"

    params = {
        "symbol": symbol,
        "type": INTERVAL,
        "startAt": int(start_ms / 1000),
        "endAt": int(end_ms / 1000),
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    payload = r.json()

    if payload.get("code") != "200000":
        raise RuntimeError(
            f"{symbol}: API error: {payload}"
        )

    return payload.get("data", [])


def fetch_history(symbol, target_days=730):
    """
    Robust chronological pagination.

    IMPORTANT:
    We intentionally fetch old -> new windows.
    This avoids the previous bug where repeatedly changing
    startAt against 'now' returned the same latest 1500 candles.
    """

    target_ms = target_days * 24 * 60 * 60 * 1000

    now_ms = int(time.time() * 1000)

    desired_start = now_ms - target_ms

    # Small safety margin around boundaries.
    desired_start -= 2 * INTERVAL_MS

    all_rows = []

    cursor = desired_start

    pages = 0

    while cursor < now_ms:
        window_end = min(
            cursor + (PAGE_LIMIT - 1) * INTERVAL_MS,
            now_ms
        )

        rows = fetch_kucoin_page(
            symbol,
            cursor,
            window_end
        )

        pages += 1

        if not rows:
            break

        all_rows.extend(rows)

        parsed_times = []

        for row in rows:
            try:
                parsed_times.append(int(row[0]) * 1000)
            except Exception:
                pass

        if not parsed_times:
            break

        newest = max(parsed_times)

        # Hard anti-stall protection.
        if newest < cursor:
            break

        next_cursor = newest + INTERVAL_MS

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        # If we have passed now, stop.
        if cursor >= now_ms:
            break

        time.sleep(0.08)

    if not all_rows:
        raise RuntimeError(f"{symbol}: no data")

    records = []

    for row in all_rows:
        if len(row) < 6:
            continue

        try:
            ts = int(row[0]) * 1000

            records.append({
                "timestamp": ts,
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
            })

        except Exception:
            continue

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(f"{symbol}: parsed dataframe empty")

    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove current incomplete candle.
    current_floor = (now_ms // INTERVAL_MS) * INTERVAL_MS

    df = df[df["timestamp"] < current_floor].copy()

    df = df.reset_index(drop=True)

    # Keep approximately target_days, but retain enough boundary
    # data to guarantee causal indicators.
    cutoff = now_ms - target_ms - 2 * INTERVAL_MS

    df = df[df["timestamp"] >= cutoff].copy()

    df = df.reset_index(drop=True)

    if len(df) < 180:
        raise RuntimeError(
            f"{symbol}: insufficient history: {len(df)} candles"
        )

    return df, pages


# ============================================================
# DATA AUDIT
# ============================================================

def audit_dataframe(df):
    ts = df["timestamp"].astype(np.int64).values

    monotonic = bool(np.all(np.diff(ts) > 0))

    duplicates = int(df["timestamp"].duplicated().sum())

    diffs = np.diff(ts)

    gap_count = int(
        np.sum(diffs != INTERVAL_MS)
    )

    gap_examples = []

    bad_idx = np.where(diffs != INTERVAL_MS)[0]

    for i in bad_idx[:5]:
        gap_examples.append({
            "from": utc_str(ts[i]),
            "to": utc_str(ts[i + 1]),
            "hours": round(
                (ts[i + 1] - ts[i]) / 3600000,
                2
            )
        })

    days = (
        (df["timestamp"].iloc[-1]
         - df["timestamp"].iloc[0])
        / 86400000
    )

    return {
        "candles": len(df),
        "days": days,
        "monotonic": monotonic,
        "duplicates": duplicates,
        "gap_count": gap_count,
        "gap_examples": gap_examples,
    }


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]

    # -------------------------
    # Bollinger Bands
    # -------------------------

    out["bb_mid"] = (
        close
        .rolling(BB_PERIOD)
        .mean()
    )

    out["bb_std"] = (
        close
        .rolling(BB_PERIOD)
        .std(ddof=0)
    )

    out["bb_upper"] = (
        out["bb_mid"]
        + BB_STD * out["bb_std"]
    )

    out["bb_lower"] = (
        out["bb_mid"]
        - BB_STD * out["bb_std"]
    )

    # -------------------------
    # RSI
    # -------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    out["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # Handle pure up/down edge cases.
    out.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi"
    ] = 100

    out.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "rsi"
    ] = 0

    # -------------------------
    # ATR
    # -------------------------

    prev_close = close.shift(1)

    tr1 = high - low

    tr2 = (high - prev_close).abs()

    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    out["atr"] = (
        tr
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    # -------------------------
    # ADX
    # -------------------------

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=out.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=out.index
    )

    atr_for_adx = (
        tr
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        )
        .mean()
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        ).mean()
        / atr_for_adx.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        ).mean()
        / atr_for_adx.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    out["adx"] = (
        dx
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        )
        .mean()
    )

    return out


# ============================================================
# CAUSAL SIGNAL DETECTION
# ============================================================

def detect_events(df, symbol):
    events = []

    # Start sufficiently late so all indicators are valid.
    start_idx = max(
        BB_PERIOD,
        RSI_PERIOD,
        ATR_PERIOD,
        ADX_PERIOD
    ) + 5

    for i in range(start_idx, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]

        values = [
            prev["close"],
            prev["bb_lower"],
            cur["close"],
            cur["bb_lower"],
            prev["rsi"],
            cur["adx"],
            cur["atr"],
        ]

        if not all(np.isfinite(v) for v in values):
            continue

        # -------------------------
        # LONG
        # -------------------------

        long_signal = (
            prev["close"] < prev["bb_lower"]
            and
            cur["close"] >= cur["bb_lower"]
            and
            prev["rsi"] <= RSI_LONG_MAX
            and
            cur["adx"] < ADX_MAX
        )

        # -------------------------
        # SHORT
        # -------------------------

        short_signal = (
            prev["close"] > prev["bb_upper"]
            and
            cur["close"] <= cur["bb_upper"]
            and
            prev["rsi"] >= RSI_SHORT_MIN
            and
            cur["adx"] < ADX_MAX
        )

        if long_signal:
            events.append({
                "symbol": symbol,
                "idx": i,
                "timestamp": int(cur["timestamp"]),
                "direction": "LONG",
                "entry": float(cur["close"]),
                "atr": float(cur["atr"]),
                "rsi": float(cur["rsi"]),
                "adx": float(cur["adx"]),
            })

        elif short_signal:
            events.append({
                "symbol": symbol,
                "idx": i,
                "timestamp": int(cur["timestamp"]),
                "direction": "SHORT",
                "entry": float(cur["close"]),
                "atr": float(cur["atr"]),
                "rsi": float(cur["rsi"]),
                "adx": float(cur["adx"]),
            })

    return events


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(df, event, tp_r):
    idx = event["idx"]

    entry = event["entry"]
    atr = event["atr"]
    direction = event["direction"]

    if not np.isfinite(entry) or not np.isfinite(atr):
        return {
            **event,
            "status": "AMBIGUOUS",
            "result_r": np.nan,
        }

    if atr <= 0:
        return {
            **event,
            "status": "AMBIGUOUS",
            "result_r": np.nan,
        }

    if direction == "LONG":
        sl = entry - SL_ATR * atr
        tp = entry + tp_r * SL_ATR * atr

    else:
        sl = entry + SL_ATR * atr
        tp = entry - tp_r * SL_ATR * atr

    first_exit_idx = idx + 1
    last_exit_idx = idx + HOLD_BARS

    # Not enough future data for complete HOLD.
    if last_exit_idx >= len(df):
        return {
            **event,
            "sl": sl,
            "tp": tp,
            "status": "OPEN_AT_DATASET_END",
            "result_r": np.nan,
            "exit_idx": len(df) - 1,
            "exit_timestamp": int(
                df["timestamp"].iloc[-1]
            ),
        }

    for j in range(
        first_exit_idx,
        last_exit_idx + 1
    ):
        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Frozen rule:
            # same candle => SL first.
            if hit_sl:
                return {
                    **event,
                    "sl": sl,
                    "tp": tp,
                    "status": "SL",
                    "result_r": -1.0,
                    "exit_idx": j,
                    "exit_timestamp": int(
                        candle["timestamp"]
                    ),
                }

            if hit_tp:
                return {
                    **event,
                    "sl": sl,
                    "tp": tp,
                    "status": "TP",
                    "result_r": float(tp_r),
                    "exit_idx": j,
                    "exit_timestamp": int(
                        candle["timestamp"]
                    ),
                }

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            if hit_sl:
                return {
                    **event,
                    "sl": sl,
                    "tp": tp,
                    "status": "SL",
                    "result_r": -1.0,
                    "exit_idx": j,
                    "exit_timestamp": int(
                        candle["timestamp"]
                    ),
                }

            if hit_tp:
                return {
                    **event,
                    "sl": sl,
                    "tp": tp,
                    "status": "TP",
                    "result_r": float(tp_r),
                    "exit_idx": j,
                    "exit_timestamp": int(
                        candle["timestamp"]
                    ),
                }

    # Full HOLD completed.
    exit_idx = last_exit_idx

    return {
        **event,
        "sl": sl,
        "tp": tp,
        "status": "TIMEOUT",
        "result_r": 0.0,
        "exit_idx": exit_idx,
        "exit_timestamp": int(
            df["timestamp"].iloc[exit_idx]
        ),
    }


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_metrics(trades):
    if not trades:
        return {
            "n_traded": 0,
            "open_at_end": 0,
            "ambiguous": 0,
            "wr": np.nan,
            "gross_exp": np.nan,
            "net_exp": np.nan,
            "gross_total": 0.0,
            "net_total": 0.0,
            "gross_pf": np.nan,
            "net_pf": np.nan,
            "gross_maxdd": 0.0,
            "net_maxdd": 0.0,
        }

    df = pd.DataFrame(trades)

    open_end = int(
        (df["status"] == "OPEN_AT_DATASET_END").sum()
    )

    ambiguous = int(
        (df["status"] == "AMBIGUOUS").sum()
    )

    closed = df[
        df["status"].isin(
            ["TP", "SL", "TIMEOUT"]
        )
    ].copy()

    if closed.empty:
        return {
            "n_traded": 0,
            "open_at_end": open_end,
            "ambiguous": ambiguous,
            "wr": np.nan,
            "gross_exp": np.nan,
            "net_exp": np.nan,
            "gross_total": 0.0,
            "net_total": 0.0,
            "gross_pf": np.nan,
            "net_pf": np.nan,
            "gross_maxdd": 0.0,
            "net_maxdd": 0.0,
        }

    # Deterministic chronological order.
    closed = closed.sort_values(
        ["exit_timestamp", "timestamp", "symbol"]
    ).reset_index(drop=True)

    gross = closed["result_r"].astype(float)

    net = gross - TOTAL_COST_R

    gross_total = float(gross.sum())
    net_total = float(net.sum())

    gross_exp = float(gross.mean())
    net_exp = float(net.mean())

    wins_gross = gross[gross > 0].sum()
    losses_gross = -gross[gross < 0].sum()

    wins_net = net[net > 0].sum()
    losses_net = -net[net < 0].sum()

    gross_pf = (
        float(wins_gross / losses_gross)
        if losses_gross > 0
        else np.inf
    )

    net_pf = (
        float(wins_net / losses_net)
        if losses_net > 0
        else np.inf
    )

    gross_curve = gross.cumsum()

    net_curve = net.cumsum()

    gross_dd = (
        gross_curve
        - gross_curve.cummax()
    )

    net_dd = (
        net_curve
        - net_curve.cummax()
    )

    gross_maxdd = float(gross_dd.min())
    net_maxdd = float(net_dd.min())

    wins = int((gross > 0).sum())

    wr = float(wins / len(closed))

    return {
        "n_traded": len(closed),
        "open_at_end": open_end,
        "ambiguous": ambiguous,
        "wr": wr,
        "gross_exp": gross_exp,
        "net_exp": net_exp,
        "gross_total": gross_total,
        "net_total": net_total,
        "gross_pf": gross_pf,
        "net_pf": net_pf,
        "gross_maxdd": gross_maxdd,
        "net_maxdd": net_maxdd,
    }


# ============================================================
# FORMAT
# ============================================================

def fmt(x, digits=4):
    if x is None or not np.isfinite(x):
        return "N/A"

    return f"{x:.{digits}f}"


def print_metrics(label, metrics):
    print(f"\n{label}")

    print(
        f"n_traded={metrics['n_traded']} | "
        f"open_end={metrics['open_at_end']} | "
        f"ambiguous={metrics['ambiguous']}"
    )

    print(
        f"WR={fmt(metrics['wr'])} | "
        f"GrossExp={fmt(metrics['gross_exp'])}R | "
        f"NetExp={fmt(metrics['net_exp'])}R"
    )

    print(
        f"GrossTotal={fmt(metrics['gross_total'], 2)}R | "
        f"NetTotal={fmt(metrics['net_total'], 2)}R"
    )

    print(
        f"GrossPF={fmt(metrics['gross_pf'], 3)} | "
        f"NetPF={fmt(metrics['net_pf'], 3)}"
    )

    print(
        f"GrossMaxDD={fmt(metrics['gross_maxdd'], 2)}R | "
        f"NetMaxDD={fmt(metrics['net_maxdd'], 2)}R"
    )


# ============================================================
# SPLIT
# ============================================================

def build_common_timestamps(data):
    sets = []

    for symbol, df in data.items():
        sets.append(
            set(
                df["timestamp"].astype(int).tolist()
            )
        )

    if not sets:
        return []

    common = set.intersection(*sets)

    return sorted(common)


def get_split(common_timestamps):
    if len(common_timestamps) < 100:
        raise RuntimeError(
            "Too few common timestamps for IS/OOS split."
        )

    split_idx = int(
        len(common_timestamps) * 0.70
    )

    split_idx = max(
        1,
        min(
            split_idx,
            len(common_timestamps) - 1
        )
    )

    is_end = common_timestamps[
        split_idx - 1
    ]

    oos_start = common_timestamps[
        split_idx
    ]

    oos_end = common_timestamps[-1]

    return is_end, oos_start, oos_end


# ============================================================
# VALIDATION
# ============================================================

def run_period(
    data,
    events_by_symbol,
    start_ts,
    end_ts,
    period_name
):
    all_trades = []

    for symbol in sorted(events_by_symbol.keys()):

        df = data[symbol]

        events = events_by_symbol[symbol]

        for event in events:

            ts = event["timestamp"]

            if ts < start_ts:
                continue

            if ts > end_ts:
                continue

            for tp_r in TP_MULTIPLIERS:

                trade = simulate_trade(
                    df,
                    event,
                    tp_r
                )

                trade["tp_r"] = tp_r
                trade["period"] = period_name

                all_trades.append(trade)

    results = {}

    for tp_r in TP_MULTIPLIERS:

        subset = [
            x for x in all_trades
            if x["tp_r"] == tp_r
        ]

        results[tp_r] = calculate_metrics(
            subset
        )

    return all_trades, results


# ============================================================
# DIRECTION METRICS
# ============================================================

def direction_metrics(trades, direction):
    subset = [
        x for x in trades
        if x["direction"] == direction
        and x["status"] in [
            "TP",
            "SL",
            "TIMEOUT"
        ]
    ]

    return calculate_metrics(subset)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "SETUP V4 — CANDIDATE 6 — "
        "VOLATILITY EXHAUSTION MEAN REVERSION"
    )
    print("730D DATA FIX + IS/OOS VALIDATION")
    print("=" * 78)

    print("\nFROZEN PARAMETERS")
    print(f"TIMEFRAME = {INTERVAL}")
    print(f"TARGET_DAYS = {TARGET_DAYS}")
    print(f"BB = {BB_PERIOD}, {BB_STD}σ")
    print(f"RSI = {RSI_PERIOD}")
    print(f"ATR = {ATR_PERIOD}")
    print(f"ADX = {ADX_PERIOD}")
    print(f"ADX_MAX = {ADX_MAX}")
    print(f"RSI_LONG <= {RSI_LONG_MAX}")
    print(f"RSI_SHORT >= {RSI_SHORT_MIN}")
    print(f"SL = {SL_ATR} ATR")
    print(f"HOLD = {HOLD_BARS}")
    print(f"TPs = {TP_MULTIPLIERS}")
    print(f"TOTAL_COST = {TOTAL_COST_R:.4f}R")
    print("NO PARAMETER OPTIMIZATION")
    print("NO SHORT-ONLY SELECTION")
    print("=" * 78)

    data = {}
    events_by_symbol = {}

    failed = []

    # ========================================================
    # DATA LOAD
    # ========================================================

    print("\n[1] FETCHING 730D HISTORY")

    for n, symbol in enumerate(
        SYMBOLS,
        start=1
    ):

        try:

            df, pages = fetch_history(
                symbol,
                TARGET_DAYS
            )

            audit = audit_dataframe(df)

            data[symbol] = df

            print(
                f"{n:02d}/{len(SYMBOLS)} "
                f"{symbol}: "
                f"{audit['candles']} candles | "
                f"{audit['days']:.1f}d | "
                f"pages={pages} | "
                f"gaps={audit['gap_count']}"
            )

            if not audit["monotonic"]:
                failed.append(
                    (symbol, "non_monotonic")
                )

            if audit["duplicates"] > 0:
                failed.append(
                    (symbol, "duplicates")
                )

            if audit["gap_count"] > 0:
                failed.append(
                    (symbol, "gaps")
                )

        except Exception as e:

            print(
                f"{n:02d}/{len(SYMBOLS)} "
                f"{symbol}: FAILED -> {e}"
            )

            failed.append(
                (symbol, str(e))
            )

    # ========================================================
    # DATA GATE
    # ========================================================

    print("\n[2] DATA GATE")

    usable = sorted(data.keys())

    print(
        f"CONFIGURED_SYMBOLS = {len(SYMBOLS)}"
    )

    print(
        f"USABLE_SYMBOLS = {len(usable)}"
    )

    if failed:

        print("\nFAILED SYMBOLS:")

        for symbol, reason in failed:
            print(
                f" - {symbol}: {reason}"
            )

    if len(usable) < 25:
        raise RuntimeError(
            "DATA GATE FAILED: fewer than 25 usable symbols."
        )

    # ========================================================
    # COMMON TIMESTAMP AUDIT
    # ========================================================

    common = build_common_timestamps(
        data
    )

    print(
        f"\nCOMMON_TIMESTAMPS = {len(common)}"
    )

    if len(common) < 1000:
        raise RuntimeError(
            "COMMON TIMESTAMP GATE FAILED."
        )

    print(
        f"COMMON_START = "
        f"{utc_str(common[0] * 1)}"
    )

    print(
        f"COMMON_END = "
        f"{utc_str(common[-1] * 1)}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    is_end, oos_start, oos_end = get_split(
        common
    )

    print("\n[3] TIME SPLIT")

    print(
        f"IS_END   = {utc_str(is_end)}"
    )

    print(
        f"OOS_START = {utc_str(oos_start)}"
    )

    print(
        f"OOS_END   = {utc_str(oos_end)}"
    )

    print(
        "SPLIT = 70% IS / 30% OOS"
    )

    print(
        "OOS uses historical warm-up context "
        "before OOS_START."
    )

    # ========================================================
    # INDICATORS + EVENTS
    # ========================================================

    print("\n[4] CAUSAL SIGNAL DETECTION")

    total_events = 0
    total_long = 0
    total_short = 0

    for symbol in usable:

        df = calculate_indicators(
            data[symbol]
        )

        data[symbol] = df

        events = detect_events(
            df,
            symbol
        )

        events_by_symbol[symbol] = events

        long_count = sum(
            x["direction"] == "LONG"
            for x in events
        )

        short_count = sum(
            x["direction"] == "SHORT"
            for x in events
        )

        total_events += len(events)
        total_long += long_count
        total_short += short_count

        print(
            f"{symbol}: "
            f"events={len(events)} | "
            f"LONG={long_count} | "
            f"SHORT={short_count}"
        )

    print("\nTOTAL EVENTS")
    print(f"TOTAL = {total_events}")
    print(f"LONG  = {total_long}")
    print(f"SHORT = {total_short}")

    # ========================================================
    # IS
    # ========================================================

    print("\n" + "=" * 78)
    print("[5] IN-SAMPLE — IS")
    print("=" * 78)

    is_trades, is_results = run_period(
        data,
        events_by_symbol,
        common[0],
        is_end,
        "IS"
    )

    for tp_r in TP_MULTIPLIERS:

        print_metrics(
            f"IS — TP {tp_r}R",
            is_results[tp_r]
        )

    # ========================================================
    # IS DIRECTION
    # ========================================================

    print("\nIS DIRECTION — TP 2R")

    is_tp2 = [
        x for x in is_trades
        if x["tp_r"] == 2.0
    ]

    print_metrics(
        "IS LONG",
        direction_metrics(
            is_tp2,
            "LONG"
        )
    )

    print_metrics(
        "IS SHORT",
        direction_metrics(
            is_tp2,
            "SHORT"
        )
    )

    # ========================================================
    # OOS
    # ========================================================

    print("\n" + "=" * 78)
    print("[6] OUT-OF-SAMPLE — OOS")
    print("=" * 78)

    oos_trades, oos_results = run_period(
        data,
        events_by_symbol,
        oos_start,
        oos_end,
        "OOS"
    )

    for tp_r in TP_MULTIPLIERS:

        print_metrics(
            f"OOS — TP {tp_r}R",
            oos_results[tp_r]
        )

    # ========================================================
    # OOS DIRECTION
    # ========================================================

    print("\nOOS DIRECTION — TP 2R")

    oos_tp2 = [
        x for x in oos_trades
        if x["tp_r"] == 2.0
    ]

    print_metrics(
        "OOS LONG",
        direction_metrics(
            oos_tp2,
            "LONG"
        )
    )

    print_metrics(
        "OOS SHORT",
        direction_metrics(
            oos_tp2,
            "SHORT"
        )
    )

    # ========================================================
    # FINAL INTEGRITY
    # ========================================================

    print("\n" + "=" * 78)
    print("[7] VALIDATION INTEGRITY")
    print("=" * 78)

    print(
        f"DATA_VALID_SYMBOLS = {len(usable)}"
    )

    print(
        f"COMMON_TIMESTAMPS = {len(common)}"
    )

    print(
        f"IS_END = {is_end}"
    )

    print(
        f"OOS_START = {oos_start}"
    )

    print(
        f"OOS_END = {oos_end}"
    )

    print(
        "OOS_WARMUP_CONTEXT = TRUE"
    )

    print(
        "CAUSAL_SIGNAL_DETECTION = TRUE"
    )

    print(
        "ENTRY_CANDLE_EXIT_SCAN = FALSE"
    )

    print(
        "SAME_CANDLE_SL_FIRST = TRUE"
    )

    print(
        "NO_PARAMETER_OPTIMIZATION = TRUE"
    )

    print(
        "NO_SHORT_ONLY_SELECTION = TRUE"
    )

    print(
        "NO_EXTRA_FILTERS = TRUE"
    )

    print(
        "NO_OVERLAP_LOCK = TRUE"
    )

    print(
        f"TOTAL_COST_R = {TOTAL_COST_R:.4f}"
    )

    if failed:
        print(
            "\nDATA AUDIT WARNING: "
            f"{len(failed)} symbols failed."
        )
    else:
        print(
            "\nDATA AUDIT = PASS"
        )

    # ========================================================
    # PRELIMINARY DECISION GATE
    # ========================================================

    oos_tp2 = oos_results[2.0]
    oos_tp3 = oos_results[3.0]

    print("\n" + "=" * 78)
    print("[8] RESEARCH GATE")
    print("=" * 78)

    print(
        "This gate is descriptive only."
    )

    print(
        "No strategy selection is performed automatically."
    )

    print(
        "\nTP2R OOS:"
    )

    print(
        f"NetExp = "
        f"{fmt(oos_tp2['net_exp'])}R"
    )

    print(
        f"NetPF = "
        f"{fmt(oos_tp2['net_pf'])}"
    )

    print(
        f"NetTotal = "
        f"{fmt(oos_tp2['net_total'], 2)}R"
    )

    print(
        f"NetMaxDD = "
        f"{fmt(oos_tp2['net_maxdd'], 2)}R"
    )

    print(
        "\nTP3R OOS:"
    )

    print(
        f"NetExp = "
        f"{fmt(oos_tp3['net_exp'])}R"
    )

    print(
        f"NetPF = "
        f"{fmt(oos_tp3['net_pf'])}"
    )

    print(
        f"NetTotal = "
        f"{fmt(oos_tp3['net_total'], 2)}R"
    )

    print(
        f"NetMaxDD = "
        f"{fmt(oos_tp3['net_maxdd'], 2)}R"
    )

    print("\nSTATUS:")

    if (
        np.isfinite(oos_tp2["net_exp"])
        and oos_tp2["net_exp"] > 0
        and oos_tp2["net_pf"] > 1
        and oos_tp2["net_total"] > 0
    ):
        print(
            "CANDIDATE 6 — OOS POSITIVE AT TP2R"
        )
        print(
            "NEXT GATE = WALK-FORWARD VALIDATION"
        )
    else:
        print(
            "CANDIDATE 6 — OOS DOES NOT SURVIVE TP2R GATE"
        )
        print(
            "NEXT ACTION = ARCHIVE"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
