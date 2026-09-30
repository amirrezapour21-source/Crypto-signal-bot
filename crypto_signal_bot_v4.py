import time
import requests
import numpy as np
import pandas as pd

# ============================================================
# CANDIDATE 11 — ROBUSTNESS V3
# DATA PAGINATION FIX ONLY
# Strategy parameters remain FROZEN
# ============================================================

BASE = "https://api.kucoin.com/api/v1/market/candles"

SYMBOLS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT",
    "DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
    "SUI-USDT","TRX-USDT","NEAR-USDT","AAVE-USDT","OP-USDT",
    "ARB-USDT","APT-USDT","ATOM-USDT","FIL-USDT","LTC-USDT",
    "BCH-USDT","ETC-USDT","UNI-USDT","INJ-USDT","SEI-USDT",
    "VET-USDT","HBAR-USDT","ALGO-USDT","XLM-USDT","ICP-USDT",
    "WIF-USDT","PEPE-USDT","FLOKI-USDT"
]

TYPE = "4hour"

STEP = 4 * 60 * 60
ROWS = 1450

TARGET_DAYS = 730
TARGET_SECONDS = TARGET_DAYS * 86400

FEE = 0.001
SLIPPAGE = 0.002
BASE_COST = FEE + SLIPPAGE

HOLD = 30
SL_ATR = 1.25

TP_MULTS = [1.0, 1.5, 2.0, 3.0]

# Candidate 11 frozen parameters
ZS_LEN = 20
ZS_THRESHOLD = 2.0
EMA_LEN = 200
RANGE_LEN = 20

# ============================================================
# TIME
# ============================================================

def now_ts():
    return int(time.time())


# ============================================================
# KUCOIN FETCH
# ============================================================

def fetch_page(symbol, start_at, end_at):
    params = {
        "symbol": symbol,
        "type": TYPE,
        "startAt": int(start_at),
        "endAt": int(end_at)
    }

    for attempt in range(5):
        try:
            r = requests.get(
                BASE,
                params=params,
                timeout=20
            )

            r.raise_for_status()

            js = r.json()

            if js.get("code") != "200000":
                raise RuntimeError(
                    f"{symbol}: KuCoin error {js.get('code')}"
                )

            return js.get("data", [])

        except Exception as e:
            if attempt == 4:
                raise

            time.sleep(1.5 * (attempt + 1))

    return []


def fetch_history(symbol, days=TARGET_DAYS):
    """
    Robust forward pagination.

    Important:
    - KuCoin returns candles in reverse chronological order.
    - Each page is normalized.
    - Next cursor is derived from the ACTUAL maximum
      timestamp returned, not from an assumed page boundary.
    - Overlap is allowed.
    - Duplicates are removed after aggregation.
    """

    end_ts = now_ts()

    start_ts = end_ts - days * 86400

    chunks = []

    cursor = start_ts
    request_count = 0

    while cursor < end_ts:
        # 1450 candles per request.
        window_end = min(
            cursor + ROWS * STEP,
            end_ts
        )

        data = fetch_page(
            symbol,
            cursor,
            window_end
        )

        request_count += 1

        if not data:
            break

        rows = []

        for x in data:
            if len(x) < 7:
                continue

            rows.append([
                int(x[0]),
                float(x[1]),
                float(x[2]),
                float(x[3]),
                float(x[4]),
                float(x[5]),
                float(x[6])
            ])

        if not rows:
            break

        chunks.extend(rows)

        page_max = max(x[0] for x in rows)

        # Critical:
        # Continue from the actual last returned candle.
        # +1 avoids relying on endpoint inclusivity.
        next_cursor = page_max + 1

        if next_cursor <= cursor:
            raise RuntimeError(
                f"{symbol}: pagination stalled "
                f"cursor={cursor} next={next_cursor}"
            )

        cursor = next_cursor

        # Safety guard
        if request_count > 10:
            raise RuntimeError(
                f"{symbol}: pagination exceeded safety limit"
            )

        time.sleep(0.12)

    if not chunks:
        return pd.DataFrame()

    df = pd.DataFrame(
        chunks,
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

    # --------------------------------------------------------
    # FINAL NORMALIZATION
    # --------------------------------------------------------

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(subset=["time"])

    df["time"] = df["time"].astype(np.int64)

    df = df[
        (df["time"] >= start_ts) &
        (df["time"] <= end_ts)
    ]

    # Remove duplicates caused by possible page overlap.
    df = (
        df.drop_duplicates("time", keep="last")
          .sort_values("time")
          .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REMOVE INCOMPLETE CURRENT CANDLE
    # --------------------------------------------------------

    current_bucket = (now_ts() // STEP) * STEP

    df = df[
        df["time"] < current_bucket
    ].reset_index(drop=True)

    return df


# ============================================================
# DATA AUDIT
# ============================================================

def audit_data(df, symbol):
    if df.empty:
        return {
            "symbol": symbol,
            "valid": False,
            "candles": 0,
            "gaps": None,
            "duplicates": None,
            "monotonic": False,
            "first": None,
            "last": None
        }

    times = df["time"].astype(np.int64)

    diffs = times.diff().dropna()

    gap_count = int(
        (diffs != STEP).sum()
    )

    duplicate_count = int(
        times.duplicated().sum()
    )

    monotonic = bool(
        times.is_monotonic_increasing
    )

    candles = len(df)

    # Coverage based on actual first/last candle.
    coverage_days = (
        (times.iloc[-1] - times.iloc[0])
        / 86400
    )

    valid = (
        candles >= 4300
        and
        gap_count == 0
        and
        duplicate_count == 0
        and
        monotonic
        and
        coverage_days >= 725
    )

    return {
        "symbol": symbol,
        "valid": valid,
        "candles": candles,
        "gaps": gap_count,
        "duplicates": duplicate_count,
        "monotonic": monotonic,
        "coverage_days": round(coverage_days, 2),
        "first": int(times.iloc[0]),
        "last": int(times.iloc[-1])
    }


# ============================================================
# INDICATORS
# ============================================================

def atr_wilder(df, n=20):
    h = df["high"]
    l = df["low"]
    c = df["close"]

    prev = c.shift(1)

    tr = pd.concat([
        h - l,
        (h - prev).abs(),
        (l - prev).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n
    ).mean()


def build_indicators(df):
    df = df.copy()

    c = df["close"]

    mean = c.rolling(
        ZS_LEN
    ).mean()

    std = c.rolling(
        ZS_LEN
    ).std()

    df["zscore"] = (
        (c - mean) / std.replace(0, np.nan)
    )

    df["ema200"] = c.ewm(
        span=EMA_LEN,
        adjust=False
    ).mean()

    df["range20"] = (
        df["high"].rolling(RANGE_LEN).max()
        -
        df["low"].rolling(RANGE_LEN).min()
    )

    df["range20_median"] = (
        df["range20"]
        .rolling(RANGE_LEN)
        .median()
    )

    df["volume_median"] = (
        df["volume"]
        .rolling(RANGE_LEN)
        .median()
    )

    df["atr20"] = atr_wilder(
        df,
        20
    )

    return df


# ============================================================
# CANDIDATE 11 SIGNAL
# ============================================================

def detect_signals(df):
    """
    Candidate 11 — FROZEN

    LONG:
      close > EMA200
      previous z <= -2
      current z > -2
      range20 >= range20 median
      volume >= volume median

    SHORT:
      close < EMA200
      previous z >= +2
      current z < +2
      range20 >= range20 median
      volume >= volume median
    """

    events = []

    for i in range(1, len(df)):
        r = df.iloc[i]
        p = df.iloc[i - 1]

        if not np.isfinite(r["atr20"]):
            continue

        if not np.isfinite(r["zscore"]):
            continue

        if not np.isfinite(p["zscore"]):
            continue

        if not np.isfinite(r["ema200"]):
            continue

        if not np.isfinite(r["range20"]):
            continue

        if not np.isfinite(r["range20_median"]):
            continue

        if not np.isfinite(r["volume_median"]):
            continue

        long_signal = (
            r["close"] > r["ema200"]
            and
            p["zscore"] <= -ZS_THRESHOLD
            and
            r["zscore"] > -ZS_THRESHOLD
            and
            r["range20"] >= r["range20_median"]
            and
            r["volume"] >= r["volume_median"]
        )

        short_signal = (
            r["close"] < r["ema200"]
            and
            p["zscore"] >= ZS_THRESHOLD
            and
            r["zscore"] < ZS_THRESHOLD
            and
            r["range20"] >= r["range20_median"]
            and
            r["volume"] >= r["volume_median"]
        )

        if long_signal:
            events.append({
                "idx": i,
                "time": int(r["time"]),
                "direction": "LONG",
                "entry": float(r["close"]),
                "atr": float(r["atr20"])
            })

        elif short_signal:
            events.append({
                "idx": i,
                "time": int(r["time"]),
                "direction": "SHORT",
                "entry": float(r["close"]),
                "atr": float(r["atr20"])
            })

    return events


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(df, event, tp_mult):
    idx = event["idx"]
    direction = event["direction"]
    entry = event["entry"]
    atr = event["atr"]

    sl_distance = SL_ATR * atr
    tp_distance = tp_mult * sl_distance

    if direction == "LONG":
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        sl = entry + sl_distance
        tp = entry - tp_distance

    last_idx = min(
        idx + HOLD,
        len(df) - 1
    )

    # No future candle available.
    if idx + HOLD >= len(df):
        return {
            "status": "OPEN_AT_DATASET_END",
            "gross_r": None,
            "net_r": None,
            "time": event["time"],
            "direction": direction
        }

    # Entry candle is NOT scanned.
    for j in range(idx + 1, last_idx + 1):

        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Same-candle SL-first
            if hit_sl:
                gross_r = -1.0

                return {
                    "status": "SL",
                    "gross_r": gross_r,
                    "net_r": gross_r - BASE_COST,
                    "time": event["time"],
                    "direction": direction
                }

            if hit_tp:
                gross_r = tp_mult

                return {
                    "status": "TP",
                    "gross_r": gross_r,
                    "net_r": gross_r - BASE_COST,
                    "time": event["time"],
                    "direction": direction
                }

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Same-candle SL-first
            if hit_sl:
                gross_r = -1.0

                return {
                    "status": "SL",
                    "gross_r": gross_r,
                    "net_r": gross_r - BASE_COST,
                    "time": event["time"],
                    "direction": direction
                }

            if hit_tp:
                gross_r = tp_mult

                return {
                    "status": "TP",
                    "gross_r": gross_r,
                    "net_r": gross_r - BASE_COST,
                    "time": event["time"],
                    "direction": direction
                }

    # Full HOLD completed without SL/TP
    return {
        "status": "TIMEOUT",
        "gross_r": 0.0,
        "net_r": -BASE_COST,
        "time": event["time"],
        "direction": direction
    }


# ============================================================
# METRICS
# ============================================================

def calc_metrics(trades):
    if not trades:
        return None

    gross = np.array([
        x["gross_r"]
        for x in trades
    ])

    net = np.array([
        x["net_r"]
        for x in trades
    ])

    wins = (gross > 0).sum()

    gross_profit = gross[gross > 0].sum()
    gross_loss = abs(
        gross[gross < 0].sum()
    )

    net_profit = net[net > 0].sum()
    net_loss = abs(
        net[net < 0].sum()
    )

    def max_dd(values):
        eq = np.cumsum(values)
        peak = np.maximum.accumulate(
            np.r_[0, eq]
        )[1:]

        dd = eq - peak

        return float(dd.min())

    return {
        "n": len(trades),
        "wr": float(wins / len(trades)),
        "gross_exp": float(gross.mean()),
        "net_exp": float(net.mean()),
        "gross_total": float(gross.sum()),
        "net_total": float(net.sum()),
        "gross_pf": (
            gross_profit / gross_loss
            if gross_loss > 0 else np.inf
        ),
        "net_pf": (
            net_profit / net_loss
            if net_loss > 0 else np.inf
        ),
        "gross_dd": max_dd(gross),
        "net_dd": max_dd(net)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CANDIDATE 11 — ROBUSTNESS V3")
    print("DATA PAGINATION / CONTINUITY VALIDATION")
    print("=" * 70)

    all_data = {}
    audits = []

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    for n, symbol in enumerate(SYMBOLS, 1):

        print(
            f"\n[{n:02d}/{len(SYMBOLS)}] {symbol}"
        )

        try:
            df = fetch_history(
                symbol,
                TARGET_DAYS
            )

            audit = audit_data(
                df,
                symbol
            )

            audits.append(audit)

            print(
                f"  candles={audit['candles']} "
                f"coverage={audit.get('coverage_days')}d "
                f"gaps={audit['gaps']} "
                f"dupes={audit['duplicates']} "
                f"mono={audit['monotonic']}"
            )

            if audit["valid"]:
                all_data[symbol] = build_indicators(df)

            else:
                print(
                    "  STATUS=INVALID DATA"
                )

        except Exception as e:

            print(
                f"  ERROR={e}"
            )

    # --------------------------------------------------------
    # GLOBAL DATA AUDIT
    # --------------------------------------------------------

    valid_audits = [
        x for x in audits
        if x["valid"]
    ]

    total_gaps = sum(
        x["gaps"]
        for x in audits
        if x["gaps"] is not None
    )

    total_dupes = sum(
        x["duplicates"]
        for x in audits
        if x["duplicates"] is not None
    )

    print("\n" + "=" * 70)
    print("DATA AUDIT")
    print("=" * 70)

    print(
        f"VALID SYMBOLS = {len(valid_audits)}"
    )

    print(
        f"TOTAL GAPS = {total_gaps}"
    )

    print(
        f"TOTAL DUPLICATES = {total_dupes}"
    )

    # --------------------------------------------------------
    # HARD STOP
    # --------------------------------------------------------

    if (
        len(valid_audits) != len(SYMBOLS)
        or
        total_gaps != 0
        or
        total_dupes != 0
    ):

        print(
            "\nDATA AUDIT = FAIL"
        )

        print(
            "ROBUSTNESS TEST ABORTED."
        )

        print(
            "NO PERFORMANCE NUMBERS ARE VALID."
        )

        return

    print(
        "\nDATA AUDIT = PASS"
    )

    # --------------------------------------------------------
    # COMMON TIMESTAMPS
    # --------------------------------------------------------

    common = None

    for symbol, df in all_data.items():

        ts = set(
            df["time"].astype(np.int64)
        )

        if common is None:
            common = ts
        else:
            common &= ts

    common = sorted(common)

    print(
        f"COMMON TIMESTAMPS = {len(common)}"
    )

    if len(common) < 4300:

        print(
            "COMMON TIMESTAMP AUDIT = FAIL"
        )

        return

    print(
        "COMMON TIMESTAMP AUDIT = PASS"
    )

    # --------------------------------------------------------
    # RESTRICT ALL SYMBOLS TO COMMON TIMESTAMPS
    # --------------------------------------------------------

    for symbol in list(all_data):

        df = all_data[symbol]

        df = df[
            df["time"].isin(common)
        ].copy()

        df = (
            df.sort_values("time")
              .reset_index(drop=True)
        )

        all_data[symbol] = df

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    events_by_symbol = {}

    total_events = 0

    for symbol, df in all_data.items():

        events = detect_signals(df)

        events_by_symbol[symbol] = events

        total_events += len(events)

        print(
            f"{symbol}: EVENTS={len(events)}"
        )

    print(
        f"\nTOTAL EVENTS = {total_events}"
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUSTNESS PERFORMANCE")
    print("=" * 70)

    for tp in TP_MULTS:

        trades = []

        for symbol, df in all_data.items():

            for event in events_by_symbol[symbol]:

                result = simulate_trade(
                    df,
                    event,
                    tp
                )

                if result["status"] in [
                    "TP",
                    "SL",
                    "TIMEOUT"
                ]:

                    result["symbol"] = symbol

                    trades.append(result)

        m = calc_metrics(trades)

        if m is None:
            continue

        print(
            f"\nTP {tp}R"
        )

        print(
            f"N={m['n']}"
        )

        print(
            f"WR={m['wr']:.4f}"
        )

        print(
            f"GrossExp={m['gross_exp']:.4f}R"
        )

        print(
            f"NetExp={m['net_exp']:.4f}R"
        )

        print(
            f"GrossTotal={m['gross_total']:.2f}R"
        )

        print(
            f"NetTotal={m['net_total']:.2f}R"
        )

        print(
            f"GrossPF={m['gross_pf']:.3f}"
        )

        print(
            f"NetPF={m['net_pf']:.3f}"
        )

        print(
            f"GrossMaxDD={m['gross_dd']:.2f}R"
        )

        print(
            f"NetMaxDD={m['net_dd']:.2f}R"
        )

    # --------------------------------------------------------
    # COST STRESS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COST STRESS")
    print("=" * 70)

    global BASE_COST

    original_cost = BASE_COST

    for cost in [
        0.003,
        0.004,
        0.005
    ]:

        BASE_COST = cost

        print(
            f"\nTOTAL COST = {cost:.3f}R"
        )

        for tp in TP_MULTS:

            trades = []

            for symbol, df in all_data.items():

                for event in events_by_symbol[symbol]:

                    result = simulate_trade(
                        df,
                        event,
                        tp
                    )

                    if result["status"] in [
                        "TP",
                        "SL",
                        "TIMEOUT"
                    ]:

                        result["symbol"] = symbol

                        trades.append(result)

            m = calc_metrics(trades)

            if m:

                print(
                    f"TP{tp}: "
                    f"N={m['n']} "
                    f"NetExp={m['net_exp']:.4f} "
                    f"NetTotal={m['net_total']:.2f} "
                    f"PF={m['net_pf']:.3f} "
                    f"DD={m['net_dd']:.2f}"
                )

    BASE_COST = original_cost

    # --------------------------------------------------------
    # LONG / SHORT DIAGNOSTIC
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LONG / SHORT DIAGNOSTIC")
    print("=" * 70)

    for tp in TP_MULTS:

        for direction in [
            "LONG",
            "SHORT"
        ]:

            trades = []

            for symbol, df in all_data.items():

                for event in events_by_symbol[symbol]:

                    if event["direction"] != direction:
                        continue

                    result = simulate_trade(
                        df,
                        event,
                        tp
                    )

                    if result["status"] in [
                        "TP",
                        "SL",
                        "TIMEOUT"
                    ]:

                        result["symbol"] = symbol

                        trades.append(result)

            m = calc_metrics(trades)

            if m:

                print(
                    f"TP{tp} {direction}: "
                    f"N={m['n']} "
                    f"WR={m['wr']:.4f} "
                    f"NetExp={m['net_exp']:.4f} "
                    f"NetTotal={m['net_total']:.2f} "
                    f"PF={m['net_pf']:.3f} "
                    f"DD={m['net_dd']:.2f}"
                )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATA STATUS")
    print("=" * 70)

    print(
        "DATA PAGINATION = VERIFIED"
    )

    print(
        "DATA CONTINUITY = VERIFIED"
    )

    print(
        "DUPLICATES = 0"
    )

    print(
        "GAPS = 0"
    )

    print(
        "STRATEGY PARAMETERS = UNCHANGED"
    )

    print(
        "ROBUSTNESS V3 = COMPLETED"
    )


if __name__ == "__main__":
    main()
