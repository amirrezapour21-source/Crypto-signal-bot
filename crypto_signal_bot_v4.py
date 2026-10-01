# crypto_signal_bot_v4.py
# CANDIDATE 11 — SHORT ONLY
# DATA FIX + 5-FOLD WALK-FORWARD
# No strategy modification.

import time
import requests
import numpy as np
import pandas as pd

BASE = "https://api.kucoin.com"
SYMBOLS = [
    "BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT",
    "SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC",
    "BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP",
    "WIF","PEPE","FLOKI"
]

N = 4380
STEP = 4 * 3600
HOLD = 30
SL_ATR = 1.25
COST = 0.003
TP_LIST = [1.5, 2.0]


def get_data(symbol):
    url = f"{BASE}/api/v1/market/candles"
    end = int(pd.Timestamp(
        "2026-09-30 00:00:00", tz="UTC"
    ).timestamp())

    all_rows = []

    # Large overlapping backward requests.
    # Overlap prevents pagination boundary loss.
    for _ in range(5):
        start = end - 1500 * STEP // 1000

        p = {
            "symbol": f"{symbol}-USDT",
            "type": "4hour",
            "startAt": start,
            "endAt": end
        }

        r = requests.get(url, params=p, timeout=30)
        r.raise_for_status()

        j = r.json()
        if j.get("code") != "200000":
            raise RuntimeError(str(j))

        rows = j.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)

        ts = min(int(x[0]) for x in rows)

        # 2-candle overlap
        end = ts - STEP * 2 // 1000

        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError("no candles returned")

    df = pd.DataFrame(
        all_rows,
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

    # Find an EXACT contiguous 4380-candle block.
    t = df["ts"].astype("int64").to_numpy() // 10**9
    diff = np.diff(t)

    good = diff == STEP

    run = 0
    start_idx = None

    for i, ok in enumerate(good):
        if ok:
            run += 1
        else:
            run = 0

        if run >= N - 1:
            start_idx = i - (N - 2)
            break

    if start_idx is None:
        raise RuntimeError(
            f"no contiguous {N}-candle block; "
            f"received={len(df)}"
        )

    out = df.iloc[
        start_idx:start_idx + N
    ].copy()

    if len(out) != N:
        raise RuntimeError(
            f"final count={len(out)}"
        )

    d = out["ts"].diff().dropna()

    if not (d == pd.Timedelta(hours=4)).all():
        raise RuntimeError(
            "contiguous-block validation failed"
        )

    return out.reset_index(drop=True)


def indicators(df):
    x = df.copy()

    m = x.close.rolling(20).mean()
    s = x.close.rolling(20).std(ddof=0)

    x["z"] = (x.close - m) / s.replace(0, np.nan)

    x["ema"] = x.close.ewm(
        span=200,
        adjust=False
    ).mean()

    x["range"] = (
        x.high.rolling(20).max()
        - x.low.rolling(20).min()
    )

    x["range_med"] = (
        x["range"].rolling(20).median()
    )

    x["vol_med"] = (
        x.volume.rolling(20).median()
    )

    pc = x.close.shift(1)

    tr = pd.concat([
        x.high - x.low,
        (x.high - pc).abs(),
        (x.low - pc).abs()
    ], axis=1).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    return x


def events(df, tp):
    x = indicators(df)
    out = []

    for i in range(20, len(x) - HOLD - 1):

        p = x.iloc[i - 1]
        c = x.iloc[i]

        # FROZEN CANDIDATE 11 — SHORT
        ok = (
            c.close < c.ema
            and p.z >= 2
            and c.z < 2
            and c["range"] >= c["range_med"]
            and c.volume >= c["vol_med"]
        )

        if not ok:
            continue

        ei = i + 1

        entry = x.iloc[ei].close
        atr = x.iloc[ei].atr

        if not np.isfinite(entry + atr) or atr <= 0:
            continue

        sl = entry + SL_ATR * atr
        target = entry - tp * SL_ATR * atr

        last = min(
            ei + HOLD,
            len(x) - 1
        )

        result = None
        exit_i = None

        for j in range(ei, last + 1):

            hi = x.iloc[j].high
            lo = x.iloc[j].low

            # SL first
            if hi >= sl:
                result = -1.0
                exit_i = j
                break

            if lo <= target:
                result = tp
                exit_i = j
                break

        if result is None:
            exit_price = x.iloc[last].close

            result = (
                entry - exit_price
            ) / (SL_ATR * atr)

            result = float(
                np.clip(result, -1, tp)
            )

            exit_i = last

        out.append({
            "entry_ts": x.iloc[ei].ts,
            "exit_ts": x.iloc[exit_i].ts,
            "r": float(result - COST)
        })

    return pd.DataFrame(out)


def metric(e):
    if len(e) == 0:
        return {
            "N": 0,
            "Exp": np.nan,
            "PF": np.nan,
            "Total": 0,
            "DD": 0
        }

    r = e.r.to_numpy(float)

    win = r[r > 0].sum()
    loss = abs(r[r < 0].sum())

    pf = win / loss if loss else np.inf

    eq = np.cumsum(r)
    peak = np.maximum.accumulate(
        np.r_[0, eq]
    )[1:]

    dd = eq - peak

    return {
        "N": len(r),
        "Exp": r.mean(),
        "PF": pf,
        "Total": r.sum(),
        "DD": dd.min()
    }


def show(name, m):
    pf = (
        f"{m['PF']:.3f}"
        if np.isfinite(m["PF"])
        else "INF"
    )

    print(
        f"{name:<8}"
        f"N={m['N']:4d} "
        f"Exp={m['Exp']:+.4f} "
        f"PF={pf:>7} "
        f"Total={m['Total']:+.2f}R "
        f"DD={m['DD']:+.2f}R"
    )


def inside(e, a, b):
    if len(e) == 0:
        return e

    return e[
        (e.entry_ts >= a) &
        (e.exit_ts < b)
    ].copy().sort_values("exit_ts")


def main():

    print("=" * 72)
    print("CANDIDATE 11 — SHORT-ONLY")
    print("DATA FIX + 5-FOLD WALK-FORWARD")
    print("=" * 72)

    data = {}

    for s in SYMBOLS:
        try:
            d = get_data(s)
            data[s] = d

            print(
                f"{s:<6} {len(d)} candles  "
                f"{d.ts.iloc[0]} -> {d.ts.iloc[-1]}"
            )

        except Exception as e:
            print(f"{s:<6} ERROR: {e}")

    if len(data) != len(SYMBOLS):
        print("\nABORT: data validation failed.")
        raise SystemExit(1)

    common = None

    for d in data.values():
        st = set(d.ts)

        common = (
            st if common is None
            else common & st
        )

    common = pd.DatetimeIndex(
        sorted(common)
    )

    if len(common) != N:
        print(
            f"\nABORT: COMMON={len(common)} "
            f"EXPECTED={N}"
        )
        raise SystemExit(1)

    print("\nDATA AUDIT = PASS")
    print(f"VALID SYMBOLS = {len(data)}")
    print(f"COMMON TIMESTAMPS = {len(common)}")
    print(f"START = {common[0]}")
    print(f"END   = {common[-1]}")

    all_events = {}

    for tp in TP_LIST:

        parts = []

        for s, d in data.items():

            d = d[
                d.ts.isin(common)
            ].copy()

            e = events(d, tp)

            if len(e):
                e["symbol"] = s
                parts.append(e)

        if parts:
            e = pd.concat(
                parts,
                ignore_index=True
            )
        else:
            e = pd.DataFrame(
                columns=[
                    "entry_ts",
                    "exit_ts",
                    "r",
                    "symbol"
                ]
            )

        e = e.sort_values(
            "exit_ts"
        ).reset_index(drop=True)

        all_events[tp] = e

        print(
            f"\nTP{tp:g}R TOTAL EVENTS = {len(e)}"
        )

    folds = np.array_split(
        np.arange(N), 5
    )

    for tp in TP_LIST:

        e = all_events[tp]

        print("\n" + "=" * 72)
        print(f"TP{tp:g}R — SHORT ONLY")
        print("=" * 72)

        oos_parts = []
        positive = 0

        for k, idx in enumerate(folds):

            oos_a = common[idx[0]]

            if k < 4:
                oos_b = common[folds[k + 1][0]]
            else:
                oos_b = common[-1] + pd.Timedelta(hours=4)

            is_a = common[0]
            is_b = oos_a

            ie = inside(e, is_a, is_b)
            oe = inside(e, oos_a, oos_b)

            im = metric(ie)
            om = metric(oe)

            print(f"\nFOLD {k + 1}")

            print(
                f"IS  {is_a} -> {is_b}"
            )
            show("IS", im)

            print(
                f"OOS {oos_a} -> {oos_b}"
            )
            show("OOS", om)

            if om["Exp"] > 0:
                positive += 1

            if len(oe):
                oos_parts.append(oe)

        if oos_parts:
            agg = pd.concat(
                oos_parts,
                ignore_index=True
            ).sort_values("exit_ts")
        else:
            agg = pd.DataFrame(
                columns=e.columns
            )

        am = metric(agg)

        print("\n" + "-" * 72)
        print(f"TP{tp:g}R AGGREGATE OOS")
        show("OOS ALL", am)
        print(
            f"Positive OOS folds = "
            f"{positive}/5"
        )

        passed = (
            positive == 5
            and am["Exp"] > 0
            and am["PF"] > 1
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
    main()
