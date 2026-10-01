import time
import requests
import pandas as pd
import numpy as np

# ============================================================
# CANDIDATE 11 — DIRECTION ROBUSTNESS
# FIXED WINDOW / LONG vs SHORT / CONCENTRATION BY DIRECTION
# ============================================================

BASE_URL = "https://api.kucoin.com/api/v1/market/candles"
INTERVAL = "4hour"

SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "DOT-USDT",
    "SUI-USDT", "TRX-USDT", "NEAR-USDT", "AAVE-USDT", "OP-USDT",
    "ARB-USDT", "APT-USDT", "ATOM-USDT", "FIL-USDT", "LTC-USDT",
    "BCH-USDT", "ETC-USDT", "UNI-USDT", "INJ-USDT", "SEI-USDT",
    "VET-USDT", "HBAR-USDT", "ALGO-USDT", "XLM-USDT", "ICP-USDT",
    "WIF-USDT", "PEPE-USDT", "FLOKI-USDT"
]

# ------------------------------------------------------------
# FROZEN HISTORICAL WINDOW
# ------------------------------------------------------------

END_TS = pd.Timestamp("2026-09-30 00:00:00", tz="UTC")
START_TS = pd.Timestamp("2024-09-30 04:00:00", tz="UTC")

STEP = pd.Timedelta(hours=4)
EXPECTED_CANDLES = 4380

# ------------------------------------------------------------
# FROZEN CANDIDATE 11 PARAMETERS
# ------------------------------------------------------------

ZSCORE_LEN = 20
ZSCORE_LONG = -2.0
ZSCORE_SHORT = 2.0

EMA_LEN = 200
RANGE_LEN = 20
ATR_LEN = 20

ATR_SL = 1.25
HOLD_BARS = 30

COST_R = 0.003

TP_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]


# ============================================================
# KUCOIN DATA
# ============================================================

def fetch_symbol(symbol):

    expected = pd.date_range(
        START_TS,
        END_TS,
        freq="4h",
        tz="UTC"
    )

    rows = []
    cursor = START_TS

    while cursor <= END_TS:

        end = min(
            END_TS,
            cursor + STEP * 1399
        )

        params = {
            "symbol": symbol,
            "type": INTERVAL,
            "startAt": int(cursor.timestamp()),
            "endAt": int(end.timestamp())
        }

        success = False

        for attempt in range(4):

            try:
                r = requests.get(
                    BASE_URL,
                    params=params,
                    timeout=20
                )

                data = r.json()

                if data.get("code") != "200000":
                    raise RuntimeError(
                        f"KuCoin error: {data.get('code')} "
                        f"{data.get('msg')}"
                    )

                batch = data.get("data", [])

                if batch:
                    rows.extend(batch)

                    max_ts = max(int(x[0]) for x in batch)

                    cursor = (
                        pd.Timestamp(max_ts, unit="s", tz="UTC")
                        + STEP
                    )

                else:
                    cursor = end + STEP

                success = True
                break

            except Exception as e:

                if attempt == 3:
                    raise RuntimeError(
                        f"{symbol}: {e}"
                    )

                time.sleep(1.0)

        if not success:
            raise RuntimeError(
                f"Failed fetching {symbol}"
            )

        time.sleep(0.08)

    if not rows:
        raise RuntimeError(
            f"{symbol}: empty data"
        )

    # KuCoin:
    # [time, open, close, high, low, volume, turnover]

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int),
        unit="s",
        utc=True
    )

    for col in [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "turnover"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df.drop_duplicates("timestamp")
          .sort_values("timestamp")
          .set_index("timestamp")
    )

    df = df.loc[
        (df.index >= START_TS) &
        (df.index <= END_TS)
    ]

    # Exact fixed-window audit
    if len(df) != EXPECTED_CANDLES:
        raise RuntimeError(
            f"{symbol}: expected {EXPECTED_CANDLES}, "
            f"got {len(df)}"
        )

    if not df.index.equals(expected):
        raise RuntimeError(
            f"{symbol}: timestamp grid mismatch"
        )

    return df


# ============================================================
# INDICATORS
# ============================================================

def prepare(df):

    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]

    # Z-SCORE 20
    mean20 = close.rolling(
        ZSCORE_LEN
    ).mean()

    std20 = close.rolling(
        ZSCORE_LEN
    ).std()

    out["zscore"] = (
        (close - mean20) / std20.replace(0, np.nan)
    )

    # EMA 200
    out["ema200"] = close.ewm(
        span=EMA_LEN,
        adjust=False,
        min_periods=EMA_LEN
    ).mean()

    # RANGE 20
    out["range20"] = (
        high.rolling(RANGE_LEN).max()
        -
        low.rolling(RANGE_LEN).min()
    )

    out["range_median"] = (
        out["range20"]
        .rolling(RANGE_LEN)
        .median()
    )

    # VOLUME MEDIAN
    out["volume_median"] = (
        out["volume"]
        .rolling(RANGE_LEN)
        .median()
    )

    # ATR 20 — Wilder
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    out["atr"] = tr.ewm(
        alpha=1 / ATR_LEN,
        adjust=False,
        min_periods=ATR_LEN
    ).mean()

    return out


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_events(df, symbol):

    events = []

    z = df["zscore"]

    for i in range(1, len(df)):

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if pd.isna(
            row["ema200"]
        ) or pd.isna(
            row["atr"]
        ) or pd.isna(
            row["range_median"]
        ) or pd.isna(
            row["volume_median"]
        ) or pd.isna(
            row["zscore"]
        ) or pd.isna(
            prev["zscore"]
        ):
            continue

        # -----------------------------
        # LONG
        # -----------------------------

        long_signal = (
            row["close"] > row["ema200"]
            and
            prev["zscore"] <= ZSCORE_LONG
            and
            row["zscore"] > ZSCORE_LONG
            and
            row["range20"] >= row["range_median"]
            and
            row["volume"] >= row["volume_median"]
        )

        if long_signal:

            events.append({
                "symbol": symbol,
                "timestamp": df.index[i],
                "index": i,
                "direction": "LONG",
                "entry": float(row["close"]),
                "atr": float(row["atr"])
            })

        # -----------------------------
        # SHORT
        # -----------------------------

        short_signal = (
            row["close"] < row["ema200"]
            and
            prev["zscore"] >= ZSCORE_SHORT
            and
            row["zscore"] < ZSCORE_SHORT
            and
            row["range20"] >= row["range_median"]
            and
            row["volume"] >= row["volume_median"]
        )

        if short_signal:

            events.append({
                "symbol": symbol,
                "timestamp": df.index[i],
                "index": i,
                "direction": "SHORT",
                "entry": float(row["close"]),
                "atr": float(row["atr"])
            })

    return events


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate(event, df, tp_mult):

    i = event["index"]

    entry = event["entry"]
    risk = event["atr"] * ATR_SL

    if risk <= 0 or not np.isfinite(risk):
        return None

    direction = event["direction"]

    if direction == "LONG":

        sl = entry - risk
        tp = entry + risk * tp_mult

    else:

        sl = entry + risk
        tp = entry - risk * tp_mult

    last_index = min(
        i + HOLD_BARS,
        len(df) - 1
    )

    # Entry candle is NOT scanned.
    start = i + 1

    if start > last_index:
        return None

    for j in range(start, last_index + 1):

        candle = df.iloc[j]

        h = candle["high"]
        l = candle["low"]

        # Same-candle SL-first
        if direction == "LONG":

            if l <= sl:
                return -1.0 - COST_R

            if h >= tp:
                return tp_mult - COST_R

        else:

            if h >= sl:
                return -1.0 - COST_R

            if l <= tp:
                return tp_mult - COST_R

    # Timeout = zero gross R, cost still applies
    return -COST_R


# ============================================================
# PERFORMANCE
# ============================================================

def metrics(values):

    if not values:
        return {
            "N": 0,
            "WR": 0.0,
            "EXP": 0.0,
            "TOTAL": 0.0,
            "PF": 0.0,
            "DD": 0.0
        }

    arr = np.array(values, dtype=float)

    wins = arr[arr > 0]

    losses = arr[arr < 0]

    total = arr.sum()

    expectancy = arr.mean()

    win_rate = (
        len(wins) / len(arr)
        if len(arr)
        else 0.0
    )

    gross_profit = wins.sum() if len(wins) else 0.0

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

    peak = np.maximum.accumulate(
        np.concatenate([[0.0], equity])
    )[1:]

    drawdown = equity - peak

    max_dd = (
        drawdown.min()
        if len(drawdown)
        else 0.0
    )

    return {
        "N": len(arr),
        "WR": win_rate,
        "EXP": expectancy,
        "TOTAL": total,
        "PF": pf,
        "DD": max_dd
    }


# ============================================================
# RUN DIRECTION TEST
# ============================================================

def run_test(all_events, data, direction, tp):

    results = []

    for event in all_events:

        if event["direction"] != direction:
            continue

        result = simulate(
            event,
            data[event["symbol"]],
            tp
        )

        if result is not None:
            results.append({
                "symbol": event["symbol"],
                "timestamp": event["timestamp"],
                "value": result
            })

    return results


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(label, values):

    m = metrics(values)

    pf_text = (
        f"{m['PF']:.3f}"
        if np.isfinite(m["PF"])
        else "INF"
    )

    print(
        f"{label}: "
        f"N={m['N']} "
        f"WR={m['WR']:.4f} "
        f"Exp={m['EXP']:.4f} "
        f"Total={m['TOTAL']:.2f}R "
        f"PF={pf_text} "
        f"DD={m['DD']:.2f}R"
    )


# ============================================================
# CONCENTRATION BY DIRECTION
# ============================================================

def concentration(results):

    if not results:
        print("NO RESULTS")
        return

    df = pd.DataFrame(results)

    portfolio = df["value"].sum()

    by_symbol = (
        df.groupby("symbol")["value"]
          .sum()
          .sort_values(ascending=False)
    )

    print(
        f"PORTFOLIO = {portfolio:.2f}R"
    )

    for k in [1, 3, 5]:

        removed = by_symbol.head(k).sum()

        remaining = portfolio - removed

        print(
            f"REMOVE TOP{k} = "
            f"{remaining:.2f}R"
        )

    print("TOP SYMBOLS:")

    for symbol, value in by_symbol.head(5).items():

        print(
            f"  {symbol}: "
            f"{value:.2f}R"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "CANDIDATE 11 — DIRECTION ROBUSTNESS"
    )
    print(
        "FIXED WINDOW / LONG vs SHORT"
    )
    print("=" * 60)

    print()
    print(
        f"WINDOW: "
        f"{START_TS} -> {END_TS}"
    )

    print(
        f"EXPECTED CANDLES / SYMBOL = "
        f"{EXPECTED_CANDLES}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data = {}

    for symbol in SYMBOLS:

        try:

            df = fetch_symbol(symbol)

            data[symbol] = prepare(df)

            print(
                f"{symbol} "
                f"{len(df)} PASS"
            )

        except Exception as e:

            print(
                f"{symbol} FAIL: {e}"
            )

    print()
    print(
        f"VALID SYMBOLS = {len(data)}"
    )

    if len(data) != len(SYMBOLS):

        print(
            "DATA AUDIT = FAIL"
        )

        raise SystemExit(
            "ABORTED: incomplete symbol universe"
        )

    # --------------------------------------------------------
    # COMMON TIMESTAMPS
    # --------------------------------------------------------

    common = set(
        data[SYMBOLS[0]].index
    )

    for symbol in SYMBOLS[1:]:

        common &= set(
            data[symbol].index
        )

    print(
        f"COMMON TIMESTAMPS = "
        f"{len(common)}"
    )

    if len(common) != EXPECTED_CANDLES:

        print(
            "DATA AUDIT = FAIL"
        )

        raise SystemExit(
            "ABORTED: common timestamp mismatch"
        )

    print(
        "DATA AUDIT = PASS"
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    all_events = []

    for symbol, df in data.items():

        events = detect_events(
            df,
            symbol
        )

        all_events.extend(events)

    all_events.sort(
        key=lambda x: x["timestamp"]
    )

    print()
    print("=" * 60)
    print("EVENT AUDIT")
    print("=" * 60)

    total_long = sum(
        e["direction"] == "LONG"
        for e in all_events
    )

    total_short = sum(
        e["direction"] == "SHORT"
        for e in all_events
    )

    print(
        f"TOTAL EVENTS = "
        f"{len(all_events)}"
    )

    print(
        f"LONG EVENTS = "
        f"{total_long}"
    )

    print(
        f"SHORT EVENTS = "
        f"{total_short}"
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    for tp in TP_MULTIPLIERS:

        print()
        print("=" * 60)
        print(
            f"TP{tp:g} — DIRECTION ROBUSTNESS"
        )
        print("=" * 60)

        long_results = run_test(
            all_events,
            data,
            "LONG",
            tp
        )

        short_results = run_test(
            all_events,
            data,
            "SHORT",
            tp
        )

        combined_results = (
            long_results +
            short_results
        )

        long_values = [
            x["value"]
            for x in long_results
        ]

        short_values = [
            x["value"]
            for x in short_results
        ]

        combined_values = [
            x["value"]
            for x in combined_results
        ]

        print()
        print_metrics(
            "LONG ",
            long_values
        )

        print_metrics(
            "SHORT",
            short_values
        )

        print_metrics(
            "BOTH ",
            combined_values
        )

        # ----------------------------------------------------
        # DIRECTION CONCENTRATION
        # ----------------------------------------------------

        print()
        print("LONG CONCENTRATION")

        concentration(
            long_results
        )

        print()
        print("SHORT CONCENTRATION")

        concentration(
            short_results
        )

    print()
    print("=" * 60)
    print(
        "DIRECTION ROBUSTNESS TEST COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
