# crypto_signal_bot_v4.py
# CANDIDATE 11 — SHORT-ONLY — 5-FOLD WALK-FORWARD
# Frozen strategy. No optimization. No extra filters.

import time
import requests
import numpy as np
import pandas as pd

BASE = "https://api.kucoin.com"
TYPE = "4hour"

SYMBOLS = [
    "BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT",
    "SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC",
    "BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP",
    "WIF","PEPE","FLOKI"
]

N_CANDLES = 4380
HOLD = 30
SL_ATR = 1.25
COST = 0.003

TP_LIST = [1.5, 2.0]

STEP_MS = 4 * 60 * 60 * 1000
CHUNK = 1500


def fetch_exact(symbol):
    url = f"{BASE}/api/v1/market/candles"

    end_ms = int(pd.Timestamp(
        "2026-09-30 00:00:00", tz="UTC"
    ).timestamp() * 1000)

    rows = []
    cur_end = end_ms

    while len(rows) < N_CANDLES:
        params = {
            "symbol": f"{symbol}-USDT",
            "type": TYPE,
            "endAt": cur_end // 1000,
            "startAt": max(0, (cur_end - CHUNK * STEP_MS) // 1000)
        }

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()

        if js.get("code") != "200000":
            raise RuntimeError(f"{symbol}: {js}")

        data = js.get("data", [])
        if not data:
            break

        rows.extend(data)

        ts = [int(x[0]) * 1000 for x in data]
        mn = min(ts)

        cur_end = mn - STEP_MS
        time.sleep(0.08)

    if not rows:
        raise RuntimeError(f"{symbol}: no data")

    df = pd.DataFrame(
        rows,
        columns=[
            "ts","open","close","high","low",
            "volume","turnover"
        ]
    )

    df["ts"] = pd.to_datetime(
        df["ts"].astype("int64"),
        unit="s",
        utc=True
    )

    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.drop_duplicates("ts")
          .sort_values("ts")
          .reset_index(drop=True)
    )

    if len(df) < N_CANDLES:
        raise RuntimeError(
            f"{symbol}: only {len(df)} candles"
        )

    # Exact latest backward window
    df = df.tail(N_CANDLES).copy()

    if len(df) != N_CANDLES:
        raise RuntimeError(
            f"{symbol}: final count={len(df)}"
        )

    diffs = df["ts"].diff().dropna().dt.total_seconds()

    if not (diffs == 4 * 3600).all():
        raise RuntimeError(
            f"{symbol}: timestamp gap detected"
        )

    return df


def indicators(df):
    x = df.copy()

    ma = x["close"].rolling(20).mean()
    sd = x["close"].rolling(20).std(ddof=0)

    x["z"] = (x["close"] - ma) / sd.replace(0, np.nan)

    x["ema200"] = x["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    x["range20"] = (
        x["high"].rolling(20).max()
        - x["low"].rolling(20).min()
    )

    x["range_med"] = x["range20"].rolling(20).median()

    x["vol_med"] = x["volume"].rolling(20).median()

    # Wilder ATR20
    prev_close = x["close"].shift(1)

    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    return x


def make_events(df, tp_r):
    x = indicators(df)
    events = []

    # Signal candle i, execution starts at i+1
    for i in range(20, len(x) - HOLD - 1):

        p = x.iloc[i - 1]
        c = x.iloc[i]

        # FROZEN CANDIDATE 11 — SHORT ONLY
        signal = (
            c["close"] < c["ema200"]
            and p["z"] >= 2.0
            and c["z"] < 2.0
            and c["range20"] >= c["range_med"]
            and c["volume"] >= c["vol_med"]
        )

        if not signal:
            continue

        entry_i = i + 1

        entry = x.iloc[entry_i]["close"]
        atr = x.iloc[entry_i]["atr"]

        if not np.isfinite(entry) or not np.isfinite(atr):
            continue

        if atr <= 0:
            continue

        sl = entry + SL_ATR * atr
        tp = entry - tp_r * SL_ATR * atr

        result = None
        exit_i = None

        last_i = min(
            entry_i + HOLD,
            len(x) - 1
        )

        # Start checking from ENTRY candle.
        # Same-candle SL-first.
        for j in range(entry_i, last_i + 1):

            hi = x.iloc[j]["high"]
            lo = x.iloc[j]["low"]

            if hi >= sl:
                result = -1.0
                exit_i = j
                break

            if lo <= tp:
                result = tp_r
                exit_i = j
                break

        if result is None:
            # Expiration at close of final holding candle
            exit_price = x.iloc[last_i]["close"]

            # Short R
            result = (entry - exit_price) / (
                SL_ATR * atr
            )

            result = float(
                np.clip(result, -1.0, tp_r)
            )

            exit_i = last_i

        net_r = result - COST

        events.append({
            "signal_ts": x.iloc[i]["ts"],
            "entry_ts": x.iloc[entry_i]["ts"],
            "exit_ts": x.iloc[exit_i]["ts"],
            "r": float(net_r)
        })

    return pd.DataFrame(events)


def metrics(events):
    if len(events) == 0:
        return {
            "N": 0,
            "Exp": np.nan,
            "PF": np.nan,
            "Total": 0.0,
            "DD": 0.0
        }

    r = events["r"].astype(float).to_numpy()

    wins = r[r > 0]
    losses = r[r < 0]

    exp = r.mean()

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    pf = (
        gross_win / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = np.cumsum(r)
    peak = np.maximum.accumulate(
        np.r_[0.0, equity]
    )[1:]

    dd = equity - peak
    max_dd = dd.min() if len(dd) else 0.0

    return {
        "N": len(r),
        "Exp": exp,
        "PF": pf,
        "Total": r.sum(),
        "DD": max_dd
    }


def print_metrics(label, m):
    pf = (
        f"{m['PF']:.3f}"
        if np.isfinite(m["PF"])
        else "INF"
    )

    print(
        f"{label:<10} "
        f"N={m['N']:4d} "
        f"Exp={m['Exp']:+.4f} "
        f"PF={pf:>7} "
        f"Total={m['Total']:+.2f}R "
        f"MaxDD={m['DD']:+.2f}R"
    )


def fold_events(all_events, start_ts, end_ts):
    """
    Only events fully contained inside the period.
    This prevents a trade from crossing an IS/OOS boundary.
    """

    e = all_events[
        (all_events["entry_ts"] >= start_ts) &
        (all_events["exit_ts"] < end_ts)
    ].copy()

    return e.sort_values("exit_ts").reset_index(drop=True)


def run():
    print("=" * 72)
    print("CANDIDATE 11 — SHORT-ONLY — 5-FOLD WALK-FORWARD")
    print("=" * 72)

    print("\nFetching data...")

    dfs = {}
    failed = []

    for s in SYMBOLS:
        try:
            df = fetch_exact(s)
            dfs[s] = df
            print(
                f"{s:<6} "
                f"{len(df)} candles  "
                f"{df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}"
            )
        except Exception as e:
            failed.append((s, str(e)))
            print(f"{s:<6} ERROR: {e}")

    if failed:
        print("\nABORT: some symbols failed.")
        for s, e in failed:
            print(" ", s, e)
        raise SystemExit(1)

    if len(dfs) != 33:
        print(
            f"\nABORT: VALID SYMBOLS={len(dfs)}, expected 33"
        )
        raise SystemExit(1)

    common = None

    for df in dfs.values():
        ts = set(df["ts"])
        common = ts if common is None else common & ts

    common = sorted(common)

    if len(common) != N_CANDLES:
        print(
            f"\nABORT: COMMON TIMESTAMPS={len(common)}, "
            f"expected {N_CANDLES}"
        )
        raise SystemExit(1)

    common = pd.DatetimeIndex(common)

    print("\nDATA AUDIT = PASS")
    print(f"VALID SYMBOLS = {len(dfs)}")
    print(f"COMMON TIMESTAMPS = {len(common)}")
    print(f"START = {common[0]}")
    print(f"END   = {common[-1]}")

    # ------------------------------------------------------------
    # Build events once, causally, for both TP variants.
    # ------------------------------------------------------------

    all_events = {}

    for tp in TP_LIST:
        chunks = []

        for s, df in dfs.items():
            z = df[
                df["ts"].isin(common)
            ].copy()

            ev = make_events(z, tp)

            if len(ev):
                ev["symbol"] = s
                chunks.append(ev)

        if chunks:
            e = pd.concat(
                chunks,
                ignore_index=True
            )
        else:
            e = pd.DataFrame(
                columns=[
                    "signal_ts",
                    "entry_ts",
                    "exit_ts",
                    "r",
                    "symbol"
                ]
            )

        e = e.sort_values(
            ["exit_ts", "entry_ts", "symbol"]
        ).reset_index(drop=True)

        all_events[tp] = e

        print(
            f"\nTP{tp:g}: TOTAL EVENTS = {len(e)}"
        )

    # ------------------------------------------------------------
    # 5 contiguous temporal folds.
    #
    # Fold k:
    # IS  = everything before OOS
    # OOS = fold k
    #
    # No event is allowed to cross either boundary.
    # ------------------------------------------------------------

    folds = np.array_split(
        np.arange(len(common)),
        5
    )

    for tp in TP_LIST:

        events = all_events[tp]

        print("\n")
        print("=" * 72)
        print(f"TP{tp:g}R — SHORT ONLY")
        print("=" * 72)

        oos_results = []

        for k, idx in enumerate(folds, 1):

            oos_start = common[idx[0]]

            if k < 5:
                oos_end = common[folds[k][0]]
            else:
                oos_end = common[-1] + pd.Timedelta(
                    hours=4
                )

            is_start = common[0]

            # IS ends exactly where OOS starts.
            is_end = oos_start

            is_ev = fold_events(
                events,
                is_start,
                is_end
            )

            oos_ev = fold_events(
                events,
                oos_start,
                oos_end
            )

            im = metrics(is_ev)
            om = metrics(oos_ev)

            print(f"\nFOLD {k}")
            print(
                f"IS  {is_start} -> "
                f"{is_end}"
            )
            print_metrics("IS", im)

            print(
                f"OOS {oos_start} -> "
                f"{oos_end}"
            )
            print_metrics("OOS", om)

            oos_results.append(om)

        # Aggregate OOS
        oos_frames = []

        for k, idx in enumerate(folds):
            oos_start = common[idx[0]]

            if k < 4:
                oos_end = common[folds[k + 1][0]]
            else:
                oos_end = common[-1] + pd.Timedelta(
                    hours=4
                )

            e = fold_events(
                events,
                oos_start,
                oos_end
            )

            if len(e):
                oos_frames.append(e)

        if oos_frames:
            aggregate = pd.concat(
                oos_frames,
                ignore_index=True
            ).sort_values(
                "exit_ts"
            )
        else:
            aggregate = pd.DataFrame(
                columns=events.columns
            )

        am = metrics(aggregate)

        positive_folds = sum(
            x["Exp"] > 0
            for x in oos_results
        )

        print("\n" + "-" * 72)
        print(f"TP{tp:g}R AGGREGATE OOS")
        print_metrics("OOS ALL", am)
        print(
            f"Positive OOS folds = "
            f"{positive_folds}/5"
        )

        # Simple validation gate — NOT a production decision.
        passed = (
            positive_folds == 5
            and am["N"] > 0
            and am["Exp"] > 0
            and am["PF"] > 1.0
            and am["Total"] > 0
        )

        print(
            "TEMPORAL VALIDATION = "
            + ("PASS" if passed else "FAIL")
        )

    print("\n" + "=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run()
