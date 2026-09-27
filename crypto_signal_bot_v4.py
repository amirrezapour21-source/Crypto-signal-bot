"""
SETUP V4 — CANDIDATE 5
EMA TREND PULLBACK + RSI RE-ENTRY

Research only.
Independent candidate.
No V3 logic.
No Candidate 1/2/3/4 logic.

Frozen research parameters:
TIMEFRAME = 4H
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 20

Pullback:
LONG  -> close > EMA200, EMA50 > EMA200,
         previous close <= previous EMA50,
         current close > EMA50,
         RSI >= 50

SHORT -> close < EMA200, EMA50 < EMA200,
         previous close >= previous EMA50,
         current close < EMA50,
         RSI <= 50

Risk:
SL = 1 ATR
TP = 1 / 1.5 / 2 / 3 R
HOLD = 30 candles

Costs:
fee = 0.10%
slippage = 0.05%

Execution:
- causal
- entry at signal candle close
- same-candle SL first
- TIMEOUT = 0R
- incomplete forward window = OPEN_AT_DATASET_END
"""

import time
import math
import requests
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.kucoin.com"

TIMEFRAME = "4hour"

TARGET_DAYS = 730
MIN_DAYS = 180

EMA_FAST = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 20

SL_ATR = 1.0

TP_SCENARIOS = [1.0, 1.5, 2.0, 3.0]

HOLD_BARS = 30

FEE_RATE = 0.0010
SLIPPAGE_RATE = 0.0005

# Same research universe used in previous candidates
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
    "NEAR-USDT",
    "APT-USDT",
    "ARB-USDT",
    "OP-USDT",
    "SUI-USDT",
    "INJ-USDT",
    "TIA-USDT",
    "SEI-USDT",
    "FIL-USDT",
    "ATOM-USDT",
    "LTC-USDT",
    "ETC-USDT",
    "TRX-USDT",
    "ICP-USDT",
    "AAVE-USDT",
    "UNI-USDT",
    "RUNE-USDT",
    "GRT-USDT",
    "ALGO-USDT",
    "VET-USDT",
    "HBAR-USDT",
    "EGLD-USDT",
    "XLM-USDT",
    "THETA-USDT",
    "SAND-USDT",
    "MANA-USDT",
    "AXS-USDT",
    "CHZ-USDT",
]


# ============================================================
# KUCOIN DATA
# ============================================================

def fetch_history(symbol, target_days=TARGET_DAYS):

    end_at = int(time.time())
    start_at = end_at - target_days * 86400

    all_rows = []
    current_start = start_at

    page_seconds = 1500 * 4 * 3600

    while current_start < end_at:

        current_end = min(current_start + page_seconds, end_at)

        params = {
            "symbol": symbol,
            "type": TIMEFRAME,
            "startAt": current_start,
            "endAt": current_end,
        }

        try:
            r = requests.get(
                f"{BASE_URL}/api/v1/market/candles",
                params=params,
                timeout=20,
            )
            r.raise_for_status()
            payload = r.json()

        except Exception as e:
            print(f"{symbol} DATA ERROR: {e}")
            break

        if payload.get("code") != "200000":
            print(f"{symbol} KUCOIN ERROR: {payload}")
            break

        rows = payload.get("data", [])

        if not rows:
            break

        all_rows.extend(rows)

        newest = max(int(row[0]) for row in rows)

        next_start = newest + 4 * 3600

        if next_start <= current_start:
            break

        current_start = next_start

        time.sleep(0.08)

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
            "turnover",
        ],
    )

    # KuCoin timestamp = milliseconds
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        errors="coerce",
        utc=True,
    )

    # Safety correction:
    # KuCoin normally returns seconds for this endpoint.
    # If dates are obviously invalid, retry interpreting as ms.
    if df["time"].notna().any():
        max_year = df["time"].dt.year.max()

        if max_year > 2035:
            raw_time = pd.to_numeric(
                df["time"].astype("int64"),
                errors="coerce",
            )

            df["time"] = pd.to_datetime(
                raw_time,
                unit="ms",
                errors="coerce",
                utc=True,
            )

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    df = df.sort_values("time")
    df = df.drop_duplicates("time")

    # Remove currently incomplete candle
    now = pd.Timestamp.now(tz="UTC")

    df = df[
        df["time"] + pd.Timedelta(hours=4) <= now
    ].copy()

    if df.empty:
        return None

    # Exact target coverage window
    cutoff = now - pd.Timedelta(days=target_days)

    df = df[df["time"] >= cutoff].copy()

    df = df.reset_index(drop=True)

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema50"] = close.ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema200"] = close.ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        ATR_PERIOD
    ).mean()

    return df


# ============================================================
# CAUSAL SIGNAL DETECTOR
# ============================================================

def detect_events(df):

    events = []

    # Indicators at current candle are allowed.
    # Future candles are never accessed here.

    for i in range(1, len(df)):

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if not np.isfinite(row["ema50"]):
            continue

        if not np.isfinite(row["ema200"]):
            continue

        if not np.isfinite(row["rsi"]):
            continue

        if not np.isfinite(row["atr"]):
            continue

        if row["atr"] <= 0:
            continue

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        long_trend = (
            row["close"] > row["ema200"]
            and row["ema50"] > row["ema200"]
        )

        long_pullback_reclaim = (
            prev["close"] <= prev["ema50"]
            and row["close"] > row["ema50"]
        )

        long_rsi = row["rsi"] >= 50

        if (
            long_trend
            and long_pullback_reclaim
            and long_rsi
        ):
            events.append(
                {
                    "idx": i,
                    "time": row["time"],
                    "dir": "LONG",
                    "entry": float(row["close"]),
                    "atr": float(row["atr"]),
                }
            )

            continue

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        short_trend = (
            row["close"] < row["ema200"]
            and row["ema50"] < row["ema200"]
        )

        short_pullback_reclaim = (
            prev["close"] >= prev["ema50"]
            and row["close"] < row["ema50"]
        )

        short_rsi = row["rsi"] <= 50

        if (
            short_trend
            and short_pullback_reclaim
            and short_rsi
        ):
            events.append(
                {
                    "idx": i,
                    "time": row["time"],
                    "dir": "SHORT",
                    "entry": float(row["close"]),
                    "atr": float(row["atr"]),
                }
            )

    return events


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, event, tp_r):

    idx = event["idx"]
    direction = event["dir"]

    entry = float(event["entry"])
    atr = float(event["atr"])

    if not np.isfinite(entry) or not np.isfinite(atr):
        return {
            "status": "AMBIGUOUS",
            "gross_r": 0.0,
            "net_r": 0.0,
        }

    sl_distance = SL_ATR * atr
    tp_distance = tp_r * sl_distance

    if direction == "LONG":

        sl = entry - sl_distance
        tp = entry + tp_distance

    else:

        sl = entry + sl_distance
        tp = entry - tp_distance

    last_idx = idx + HOLD_BARS

    if last_idx >= len(df):

        return {
            "status": "OPEN_AT_DATASET_END",
            "gross_r": 0.0,
            "net_r": 0.0,
        }

    exit_status = "TIMEOUT"
    gross_r = 0.0
    exit_price = float(df.iloc[last_idx]["close"])

    bars_held = HOLD_BARS

    for j in range(idx + 1, last_idx + 1):

        candle = df.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Same candle: SL first
            if hit_sl:

                exit_status = "SL"
                exit_price = sl
                bars_held = j - idx
                gross_r = -1.0
                break

            if hit_tp:

                exit_status = "TP"
                exit_price = tp
                bars_held = j - idx
                gross_r = tp_r
                break

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Same candle: SL first
            if hit_sl:

                exit_status = "SL"
                exit_price = sl
                bars_held = j - idx
                gross_r = -1.0
                break

            if hit_tp:

                exit_status = "TP"
                exit_price = tp
                bars_held = j - idx
                gross_r = tp_r
                break

    # If neither SL nor TP was hit
    if exit_status == "TIMEOUT":

        if direction == "LONG":
            gross_r = (
                (exit_price - entry)
                / sl_distance
            )
        else:
            gross_r = (
                (entry - exit_price)
                / sl_distance
            )

    # --------------------------------------------------------
    # COST MODEL
    # --------------------------------------------------------

    # Approximate round-trip trading cost.
    # Fee + slippage on entry and exit.
    notional_cost = (
        2 * (FEE_RATE + SLIPPAGE_RATE)
    )

    risk_fraction = sl_distance / entry

    cost_r = (
        notional_cost / risk_fraction
        if risk_fraction > 0
        else 0.0
    )

    net_r = gross_r - cost_r

    return {
        "status": exit_status,
        "gross_r": float(gross_r),
        "net_r": float(net_r),
        "bars_held": int(bars_held),
        "entry": entry,
        "exit": float(exit_price),
    }


# ============================================================
# METRICS
# ============================================================

def profit_factor(values):

    gains = sum(x for x in values if x > 0)
    losses = abs(sum(x for x in values if x < 0))

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return gains / losses


def max_drawdown(values):

    if not values:
        return 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for r in values:

        equity += r

        if equity > peak:
            peak = equity

        dd = equity - peak

        if dd < max_dd:
            max_dd = dd

    return max_dd


def summarize(trades):

    traded = [
        t for t in trades
        if t["status"] != "OPEN_AT_DATASET_END"
        and t["status"] != "AMBIGUOUS"
    ]

    if not traded:
        return None

    gross = [t["gross_r"] for t in traded]
    net = [t["net_r"] for t in traded]

    wins = sum(
        1 for x in gross
        if x > 0
    )

    return {
        "n_traded": len(traded),
        "n_open_at_end": sum(
            t["status"] == "OPEN_AT_DATASET_END"
            for t in trades
        ),
        "n_ambiguous": sum(
            t["status"] == "AMBIGUOUS"
            for t in trades
        ),
        "WR": wins / len(traded),
        "GrossExp": np.mean(gross),
        "NetExp": np.mean(net),
        "GrossTotal": np.sum(gross),
        "NetTotal": np.sum(net),
        "GrossPF": profit_factor(gross),
        "NetPF": profit_factor(net),
        "GrossMaxDD": max_drawdown(gross),
        "NetMaxDD": max_drawdown(net),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SETUP V4 — CANDIDATE 5")
    print("EMA TREND PULLBACK + RSI RE-ENTRY")
    print("=" * 80)

    print(
        f"4H | EMA={EMA_FAST}/{EMA_SLOW} | "
        f"RSI={RSI_PERIOD} | ATR={ATR_PERIOD} | "
        f"SL={SL_ATR} ATR | HOLD={HOLD_BARS}"
    )

    print(
        f"TP={TP_SCENARIOS} | "
        f"Fee={FEE_RATE:.4f} | "
        f"Slippage={SLIPPAGE_RATE:.4f}"
    )

    print()

    data = {}

    sufficient = 0

    for symbol in SYMBOLS:

        df = fetch_history(
            symbol,
            TARGET_DAYS
        )

        if df is None or df.empty:

            print(
                f"{symbol:12s} DATA FAILED"
            )

            continue

        first = df["time"].iloc[0]
        last = df["time"].iloc[-1]

        days = (
            last - first
        ).total_seconds() / 86400

        if days >= MIN_DAYS:

            sufficient += 1

            df = calculate_indicators(df)

            data[symbol] = df

            print(
                f"{symbol:12s} "
                f"{len(df):5d} candles | "
                f"{days:6.1f} days | SUFFICIENT"
            )

        else:

            print(
                f"{symbol:12s} "
                f"{len(df):5d} candles | "
                f"{days:6.1f} days | "
                f"INSUFFICIENT"
            )

    print()
    print(
        f"DATA: {len(data)} loaded | "
        f"{sufficient} sufficient"
    )

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    all_events = {}

    total_events = 0

    for symbol, df in data.items():

        events = detect_events(df)

        all_events[symbol] = events

        total_events += len(events)

        long_count = sum(
            e["dir"] == "LONG"
            for e in events
        )

        short_count = sum(
            e["dir"] == "SHORT"
            for e in events
        )

        print(
            f"{symbol:12s} "
            f"events={len(events):4d} "
            f"LONG={long_count:4d} "
            f"SHORT={short_count:4d}"
        )

    print()
    print(
        f"TOTAL EVENTS: {total_events}"
    )

    # --------------------------------------------------------
    # SIMULATION
    # --------------------------------------------------------

    for tp_r in TP_SCENARIOS:

        portfolio_trades = []

        print()
        print("=" * 80)
        print(f"TP SCENARIO = {tp_r:.1f}R")
        print("=" * 80)

        for symbol, events in all_events.items():

            df = data[symbol]

            symbol_trades = []

            for event in events:

                trade = simulate_trade(
                    df,
                    event,
                    tp_r
                )

                trade["symbol"] = symbol
                trade["dir"] = event["dir"]
                trade["time"] = event["time"]

                symbol_trades.append(trade)
                portfolio_trades.append(trade)

            summary = summarize(
                symbol_trades
            )

            if summary:

                print(
                    f"{symbol:12s} "
                    f"N={summary['n_traded']:4d} "
                    f"WR={summary['WR']:.3f} "
                    f"GrossE={summary['GrossExp']:+.4f} "
                    f"NetE={summary['NetExp']:+.4f} "
                    f"NetPF={summary['NetPF']:.3f}"
                )

        portfolio = summarize(
            portfolio_trades
        )

        print()
        print("PORTFOLIO SUMMARY")

        if portfolio:

            print(
                f"n_traded      = "
                f"{portfolio['n_traded']}"
            )

            print(
                f"n_open_end    = "
                f"{portfolio['n_open_at_end']}"
            )

            print(
                f"n_ambiguous   = "
                f"{portfolio['n_ambiguous']}"
            )

            print(
                f"WR            = "
                f"{portfolio['WR']:.4f}"
            )

            print(
                f"Gross Exp     = "
                f"{portfolio['GrossExp']:+.4f}R"
            )

            print(
                f"Net Exp       = "
                f"{portfolio['NetExp']:+.4f}R"
            )

            print(
                f"Gross Total   = "
                f"{portfolio['GrossTotal']:+.2f}R"
            )

            print(
                f"Net Total     = "
                f"{portfolio['NetTotal']:+.2f}R"
            )

            print(
                f"Gross PF      = "
                f"{portfolio['GrossPF']:.3f}"
            )

            print(
                f"Net PF        = "
                f"{portfolio['NetPF']:.3f}"
            )

            print(
                f"Gross MaxDD   = "
                f"{portfolio['GrossMaxDD']:+.2f}R"
            )

            print(
                f"Net MaxDD     = "
                f"{portfolio['NetMaxDD']:+.2f}R"
            )

    print()
    print("=" * 80)
    print("CANDIDATE 5 IS TEST COMPLETE")
    print("=" * 80)
    print(
        "IMPORTANT: Do NOT optimize parameters from this run."
    )
    print(
        "If IS shows a credible net edge, run independent OOS."
    )


if __name__ == "__main__":
    main()
