import requests
import pandas as pd
import numpy as np
import time as time_module

# ============================================================
# SETUP V4 — CANDIDATE 4 — DONCHIAN VOLATILITY BREAKOUT
# Frozen: 4H | breakout=20 | ATR=20 | volume=20
# daily EMA20 bias | HOLD=30 | Long/Short
# ============================================================

SYMS = [
    "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT",
    "DOT-USDT", "NEAR-USDT", "APT-USDT", "ARB-USDT",
    "OP-USDT", "SUI-USDT", "INJ-USDT", "TIA-USDT",
    "SEI-USDT", "FIL-USDT", "ATOM-USDT", "LTC-USDT",
    "ETC-USDT", "TRX-USDT", "ICP-USDT", "AAVE-USDT",
    "UNI-USDT", "MKR-USDT", "RUNE-USDT", "FTM-USDT",
    "GRT-USDT", "ALGO-USDT", "VET-USDT", "HBAR-USDT",
    "EGLD-USDT", "XLM-USDT", "THETA-USDT", "SAND-USDT",
    "MANA-USDT", "AXS-USDT", "CHZ-USDT"
]

TARGET_DAYS = 730
MIN_DAYS = 180

BREAKOUT = 20
ATR_LEN = 20
VOL_LEN = 20
DAILY_EMA = 20

HOLD = 30

TP_SCENARIOS = [1.0, 1.5, 2.0, 3.0]

FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05

BASE_URL = "https://api.kucoin.com/api/v1/market/candles"

INTERVAL = "4hour"


# ============================================================
# DATA FETCH
# ============================================================

def fetch_history(symbol, target_days=TARGET_DAYS):

    end_at = int(time_module.time())
    start_at = end_at - target_days * 86400

    rows = []

    current_end = end_at

    while current_end > start_at:

        params = {
            "symbol": symbol,
            "type": INTERVAL,
            "startAt": start_at,
            "endAt": current_end
        }

        try:
            r = requests.get(
                BASE_URL,
                params=params,
                timeout=20
            )

            r.raise_for_status()

            payload = r.json()

            if payload.get("code") != "200000":
                break

            batch = payload.get("data", [])

            if not batch:
                break

            rows.extend(batch)

            # KuCoin candles are:
            # [time, open, close, high, low, volume, turnover]

            timestamps = []

            for row in batch:
                try:
                    timestamps.append(int(row[0]))
                except Exception:
                    continue

            if not timestamps:
                break

            oldest = min(timestamps)

            if oldest <= start_at:
                break

            # Move pagination window backward
            current_end = oldest - 1

            # Avoid API hammering
            time_module.sleep(0.15)

        except Exception as e:
            print(f"{symbol} fetch error: {e}")
            break

    if not rows:
        return None, 0

    df = pd.DataFrame(
        rows,
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

    # ========================================================
    # IMPORTANT TIMESTAMP FIX
    # KuCoin /market/candles returns UNIX timestamp in seconds
    # ========================================================

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(subset=["time"])

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    # Numeric coercion
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

    df = df.dropna(
        subset=[
            "open",
            "close",
            "high",
            "low",
            "volume"
        ]
    )

    # Remove duplicates
    df = df.drop_duplicates(
        subset=["time"]
    )

    # Chronological order
    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Drop incomplete current 4H candle
    # --------------------------------------------------------

    now = pd.Timestamp.now(tz="UTC")

    if len(df) > 0:
        last_time = df.iloc[-1]["time"]

        if last_time + pd.Timedelta(hours=4) > now:
            df = df.iloc[:-1].copy()

    if len(df) == 0:
        return None, 0

    days = (
        df["time"].iloc[-1] -
        df["time"].iloc[0]
    ).total_seconds() / 86400

    return df, days


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # True Range / ATR
    # --------------------------------------------------------

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] -
        prev_close
    ).abs()

    tr3 = (
        df["low"] -
        prev_close
    ).abs()

    df["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        df["TR"]
        .rolling(ATR_LEN)
        .mean()
    )

    # --------------------------------------------------------
    # Volume average
    # --------------------------------------------------------

    df["VOL_MA"] = (
        df["volume"]
        .rolling(VOL_LEN)
        .mean()
    )

    # --------------------------------------------------------
    # Daily EMA20 bias
    # --------------------------------------------------------

    daily = (
        df.set_index("time")
        .resample("1D")
        .agg({
            "close": "last"
        })
        .dropna()
    )

    daily["EMA20"] = (
        daily["close"]
        .ewm(
            span=DAILY_EMA,
            adjust=False
        )
        .mean()
    )

    daily["DAILY_BIAS"] = np.where(
        daily["close"] > daily["EMA20"],
        "LONG",
        "SHORT"
    )

    # Map daily bias back to each 4H candle
    daily_bias = daily["DAILY_BIAS"].reindex(
        df["time"].dt.floor("D"),
        method="ffill"
    )

    df["DAILY_BIAS"] = (
        daily_bias
        .to_numpy()
    )

    return df


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_events(df):

    events = []

    df = df.copy()

    # Previous 20-bar Donchian boundaries
    df["DONCHIAN_HIGH"] = (
        df["high"]
        .shift(1)
        .rolling(BREAKOUT)
        .max()
    )

    df["DONCHIAN_LOW"] = (
        df["low"]
        .shift(1)
        .rolling(BREAKOUT)
        .min()
    )

    for i in range(len(df)):

        # Need enough history
        if i < max(
            BREAKOUT,
            ATR_LEN,
            VOL_LEN
        ):
            continue

        atr = df.iloc[i]["ATR"]
        vol_ma = df.iloc[i]["VOL_MA"]

        if pd.isna(atr) or atr <= 0:
            continue

        if pd.isna(vol_ma) or vol_ma <= 0:
            continue

        close = df.iloc[i]["close"]
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        volume = df.iloc[i]["volume"]

        previous_high = df.iloc[i]["DONCHIAN_HIGH"]
        previous_low = df.iloc[i]["DONCHIAN_LOW"]

        daily_bias = df.iloc[i]["DAILY_BIAS"]

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if (
            close > previous_high
            and volume > vol_ma
            and daily_bias == "LONG"
        ):

            events.append({
                "idx": i,
                "time": df.iloc[i]["time"],
                "dir": "LONG",
                "entry": close,
                "atr": atr
            })

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        elif (
            close < previous_low
            and volume > vol_ma
            and daily_bias == "SHORT"
        ):

            events.append({
                "idx": i,
                "time": df.iloc[i]["time"],
                "dir": "SHORT",
                "entry": close,
                "atr": atr
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
    direction = event["dir"]

    entry = float(event["entry"])
    atr = float(event["atr"])

    if direction == "LONG":

        sl = entry - atr
        tp = entry + tp_r * atr

    else:

        sl = entry + atr
        tp = entry - tp_r * atr

    last_idx = min(
        idx + HOLD,
        len(df) - 1
    )

    exit_idx = None
    exit_price = None
    outcome = None

    for j in range(
        idx + 1,
        last_idx + 1
    ):

        candle = df.iloc[j]

        high = float(candle["high"])
        low = float(candle["low"])

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Same-candle SL-first
            if hit_sl:
                exit_idx = j
                exit_price = sl
                outcome = "SL"
                break

            if hit_tp:
                exit_idx = j
                exit_price = tp
                outcome = "TP"
                break

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Same-candle SL-first
            if hit_sl:
                exit_idx = j
                exit_price = sl
                outcome = "SL"
                break

            if hit_tp:
                exit_idx = j
                exit_price = tp
                outcome = "TP"
                break

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if exit_idx is None:

        if idx + HOLD >= len(df):
            return {
                "status": "OPEN_AT_DATASET_END"
            }

        exit_idx = idx + HOLD
        exit_price = float(
            df.iloc[exit_idx]["close"]
        )

        outcome = "TIMEOUT"

    # --------------------------------------------------------
    # Gross R
    # --------------------------------------------------------

    risk = atr

    if direction == "LONG":
        gross_r = (
            exit_price - entry
        ) / risk
    else:
        gross_r = (
            entry - exit_price
        ) / risk

    # --------------------------------------------------------
    # Fees + slippage
    # --------------------------------------------------------

    total_cost_pct = (
        FEE_PCT + SLIPPAGE_PCT
    ) / 100.0

    # Approximate round-trip percentage cost
    round_trip_cost = (
        2 * total_cost_pct
    )

    # Convert percentage cost to R
    cost_r = (
        entry * round_trip_cost
    ) / risk

    net_r = gross_r - cost_r

    return {
        "status": "TRADED",
        "entry_time": event["time"],
        "exit_time": df.iloc[exit_idx]["time"],
        "dir": direction,
        "outcome": outcome,
        "gross_r": gross_r,
        "net_r": net_r
    }


# ============================================================
# METRICS
# ============================================================

def calc_metrics(trades):

    if not trades:

        return {
            "n_traded": 0,
            "n_open_at_end": 0,
            "win_rate": 0,
            "gross_exp": 0,
            "net_exp": 0,
            "gross_total": 0,
            "net_total": 0,
            "gross_pf": 0,
            "net_pf": 0,
            "gross_max_dd": 0,
            "net_max_dd": 0
        }

    gross = np.array(
        [x["gross_r"] for x in trades],
        dtype=float
    )

    net = np.array(
        [x["net_r"] for x in trades],
        dtype=float
    )

    wins = np.sum(
        gross > 0
    )

    gross_positive = gross[
        gross > 0
    ].sum()

    gross_negative = abs(
        gross[gross < 0].sum()
    )

    net_positive = net[
        net > 0
    ].sum()

    net_negative = abs(
        net[net < 0].sum()
    )

    gross_pf = (
        gross_positive / gross_negative
        if gross_negative > 0
        else np.inf
    )

    net_pf = (
        net_positive / net_negative
        if net_negative > 0
        else np.inf
    )

    gross_curve = np.cumsum(gross)
    net_curve = np.cumsum(net)

    gross_peak = np.maximum.accumulate(
        gross_curve
    )

    net_peak = np.maximum.accumulate(
        net_curve
    )

    gross_dd = (
        gross_curve -
        gross_peak
    )

    net_dd = (
        net_curve -
        net_peak
    )

    return {
        "n_traded": len(trades),
        "n_open_at_end": 0,
        "win_rate": wins / len(trades),
        "gross_exp": gross.mean(),
        "net_exp": net.mean(),
        "gross_total": gross.sum(),
        "net_total": net.sum(),
        "gross_pf": gross_pf,
        "net_pf": net_pf,
        "gross_max_dd": gross_dd.min(),
        "net_max_dd": net_dd.min()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "SETUP V4 — CANDIDATE 4 — "
        "DONCHIAN VOLATILITY BREAKOUT"
    )

    print(
        "Frozen: 4H | breakout=20 | ATR=20 | "
        "volume=20 | daily EMA20 bias | "
        "HOLD=30 | Long/Short"
    )

    data = {}

    loaded = 0
    sufficient = 0

    # --------------------------------------------------------
    # Fetch all symbols
    # --------------------------------------------------------

    for symbol in SYMS:

        df, days = fetch_history(
            symbol,
            TARGET_DAYS
        )

        if df is None:
            continue

        loaded += 1

        data[symbol] = {
            "df": df,
            "days": days
        }

        if days >= MIN_DAYS:
            sufficient += 1

    print(
        f"DATA {loaded} loaded; "
        f"{sufficient} sufficient"
    )

    if sufficient == 0:

        print(
            "NO SUFFICIENT DATA"
        )

        return

    # --------------------------------------------------------
    # Filter usable symbols
    # --------------------------------------------------------

    usable = {
        symbol: item
        for symbol, item in data.items()
        if item["days"] >= MIN_DAYS
    }

    all_results = {}

    # --------------------------------------------------------
    # Run strategy
    # --------------------------------------------------------

    for symbol, item in usable.items():

        df = item["df"]

        df = add_indicators(df)

        events = detect_events(df)

        # Exclude final HOLD candles so every event
        # has complete forward data.
        max_event_idx = len(df) - HOLD - 1

        events = [
            e for e in events
            if e["idx"] <= max_event_idx
        ]

        print(
            f"{symbol}: "
            f"{len(df)} candles | "
            f"{item['days']:.0f}d | "
            f"{len(events)} events"
        )

        for tp_r in TP_SCENARIOS:

            trades = []

            for event in events:

                result = simulate_trade(
                    df,
                    event,
                    tp_r
                )

                if result["status"] == "TRADED":
                    trades.append(result)

            metrics = calc_metrics(
                trades
            )

            all_results.setdefault(
                tp_r,
                []
            ).extend(trades)

            print(
                f"  TP {tp_r}R | "
                f"traded={metrics['n_traded']} | "
                f"WR={metrics['win_rate']:.3f} | "
                f"GrossExp={metrics['gross_exp']:.4f}R | "
                f"NetExp={metrics['net_exp']:.4f}R"
            )

    # --------------------------------------------------------
    # Portfolio summary
    # --------------------------------------------------------

    print("")
    print("===== PORTFOLIO SUMMARY =====")

    for tp_r in TP_SCENARIOS:

        trades = all_results.get(
            tp_r,
            []
        )

        metrics = calc_metrics(
            trades
        )

        print("")
        print(
            f"TP {tp_r}R"
        )

        print(
            f"n_traded       = "
            f"{metrics['n_traded']}"
        )

        print(
            f"WR             = "
            f"{metrics['win_rate']:.4f}"
        )

        print(
            f"Gross Exp      = "
            f"{metrics['gross_exp']:.4f}R"
        )

        print(
            f"Net Exp        = "
            f"{metrics['net_exp']:.4f}R"
        )

        print(
            f"Gross Total    = "
            f"{metrics['gross_total']:.2f}R"
        )

        print(
            f"Net Total      = "
            f"{metrics['net_total']:.2f}R"
        )

        print(
            f"Gross PF       = "
            f"{metrics['gross_pf']:.3f}"
        )

        print(
            f"Net PF         = "
            f"{metrics['net_pf']:.3f}"
        )

        print(
            f"Gross MaxDD    = "
            f"{metrics['gross_max_dd']:.2f}R"
        )

        print(
            f"Net MaxDD      = "
            f"{metrics['net_max_dd']:.2f}R"
        )

    print("")
    print("===== TEST RUN COMPLETE =====")


if __name__ == "__main__":
    main()
