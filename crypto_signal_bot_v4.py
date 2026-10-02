import requests
import pandas as pd
import numpy as np
import time

BASE = "https://api.kucoin.com/api/v1/market/candles"

SYMS = [
    "BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT",
    "SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC",
    "BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP",
    "WIF","PEPE","FLOKI"
]

N = 4380
STEP = 14400
HOLD = 30
ATR_M = 1.25
COSTS = [0.003, 0.004, 0.005]

START_DT = pd.Timestamp("2024-09-30T00:00:00Z")
END_DT = pd.Timestamp("2026-09-29T20:00:00Z")

START_TS = int(START_DT.timestamp())
END_TS = int(END_DT.timestamp())


def get_json(params):
    last = None

    for attempt in range(4):
        try:
            r = requests.get(
                BASE,
                params=params,
                timeout=12
            )

            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

            r.raise_for_status()
            return r.json().get("data", [])

        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"request failed: {last}")


def fetch(sym):
    rows = []

    CHUNK = 900
    pos = START_TS - 3 * STEP
    end_fetch = END_TS + 3 * STEP

    while pos <= end_fetch:

        chunk_end = min(
            pos + (CHUNK - 1) * STEP,
            end_fetch
        )

        rows.extend(
            get_json({
                "symbol": f"{sym}-USDT",
                "type": "4hour",
                "startAt": pos,
                "endAt": chunk_end
            })
        )

        if chunk_end >= end_fetch:
            break

        pos = chunk_end - STEP
        time.sleep(0.03)

    if not rows:
        raise ValueError(f"{sym}: no data")

    d = pd.DataFrame(
        rows,
        columns=[
            "ts","open","close","high",
            "low","vol","turn"
        ]
    )

    d["ts"] = pd.to_datetime(
        pd.to_numeric(d["ts"]),
        unit="s",
        utc=True
    )

    for c in ["open","close","high","low","vol"]:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    d = (
        d.drop_duplicates("ts")
         .sort_values("ts")
         .reset_index(drop=True)
    )

    d = d[
        (d["ts"] >= START_DT) &
        (d["ts"] <= END_DT)
    ].copy()

    if len(d) != N:
        raise ValueError(
            f"{sym}: candles={len(d)} expected={N}"
        )

    gaps = (
        d["ts"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    if not gaps.eq(STEP).all():
        raise ValueError(
            f"{sym}: timestamp gap"
        )

    return d.reset_index(drop=True)


def indicators(d):
    d = d.copy()

    c = d["close"]
    h = d["high"]
    l = d["low"]
    v = d["vol"]

    d["z"] = (
        c - c.rolling(20).mean()
    ) / c.rolling(20).std()

    d["ema"] = c.ewm(
        span=200,
        adjust=False
    ).mean()

    d["range20"] = (
        h.rolling(20).max()
        - l.rolling(20).min()
    )

    d["rmed"] = (
        d["range20"]
        .rolling(20)
        .median()
    )

    d["vmed"] = (
        v.rolling(20)
        .median()
    )

    tr = pd.concat(
        [
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    return d


def events(d, tp, symbol):
    out = []

    for i in range(21, len(d) - HOLD):

        if not (
            d["z"].iloc[i-1] >= 2
            and d["z"].iloc[i] < 2
        ):
            continue

        if d["close"].iloc[i] >= d["ema"].iloc[i]:
            continue

        if d["range20"].iloc[i] < d["rmed"].iloc[i]:
            continue

        if d["vol"].iloc[i] < d["vmed"].iloc[i]:
            continue

        entry = float(d["close"].iloc[i])
        atr = float(d["atr"].iloc[i])

        if not np.isfinite(atr) or atr <= 0:
            continue

        sl = entry + ATR_M * atr
        target = entry - tp * ATR_M * atr

        exit_i = i + HOLD - 1
        outcome = "HOLD"

        for j in range(i + 1, i + HOLD):

            hi = float(d["high"].iloc[j])
            lo = float(d["low"].iloc[j])

            if hi >= sl:
                outcome = "SL"
                exit_i = j
                break

            if lo <= target:
                outcome = "TP"
                exit_i = j
                break

        out.append([
            d["ts"].iloc[i],
            d["ts"].iloc[exit_i],
            symbol,
            outcome
        ])

    return pd.DataFrame(
        out,
        columns=[
            "entry",
            "exit",
            "symbol",
            "outcome"
        ]
    )


def add_r(e, cost, tp):
    x = e.copy()

    x["r"] = np.where(
        x["outcome"].eq("TP"),
        tp - cost,
        np.where(
            x["outcome"].eq("SL"),
            -1.0 - cost,
            -cost
        )
    )

    return x.sort_values(
        "entry"
    ).reset_index(drop=True)


def no_overlap(e):
    if e.empty:
        return e.copy()

    accepted = []

    for symbol, g in e.groupby("symbol"):

        g = g.sort_values(
            "entry"
        )

        last_exit = None

        for _, row in g.iterrows():

            if (
                last_exit is None
                or row["entry"] > last_exit
            ):
                accepted.append(row)
                last_exit = row["exit"]

    if not accepted:
        return e.iloc[0:0].copy()

    return (
        pd.DataFrame(accepted)
        .sort_values("entry")
        .reset_index(drop=True)
    )


def stats(e):
    if e.empty:
        return 0, np.nan, np.nan, 0.0, 0.0

    r = e["r"].to_numpy(float)

    gain = r[r > 0].sum()
    loss = -r[r < 0].sum()

    pf = (
        gain / loss
        if loss > 0
        else np.inf
    )

    equity = np.cumsum(r)

    dd = (
        equity
        - np.maximum.accumulate(equity)
    ).min()

    return (
        len(r),
        r.mean(),
        pf,
        r.sum(),
        dd
    )


def temporal(e, timestamps):
    folds = np.array_split(
        np.array(timestamps),
        5
    )

    results = []

    for k, fold in enumerate(folds):

        start = pd.Timestamp(fold[0])

        end = (
            pd.Timestamp(fold[-1])
            + pd.Timedelta(hours=4)
        )

        x = e[
            (e["entry"] >= start) &
            (e["entry"] < end) &
            (e["exit"] < end)
        ].copy()

        s = stats(x)

        results.append(x)

        print(
            f"FOLD {k+1} | "
            f"N={s[0]:4d} "
            f"Exp={s[1]:+.4f} "
            f"PF={s[2]:.3f} "
            f"Total={s[3]:+.2f}R "
            f"DD={s[4]:+.2f}R"
        )

    all_oos = pd.concat(
        results,
        ignore_index=True
    )

    s = stats(
        all_oos.sort_values("entry")
    )

    positive = sum(
        stats(x)[3] > 0
        for x in results
    )

    print("-" * 72)
    print(
        f"AGG OOS | "
        f"N={s[0]} "
        f"Exp={s[1]:+.4f} "
        f"PF={s[2]:.3f} "
        f"Total={s[3]:+.2f}R "
        f"DD={s[4]:+.2f}R"
    )

    print(
        f"POSITIVE FOLDS = {positive}/5"
    )

    return s, positive


def concentration(e):
    p = (
        e.groupby("symbol")["r"]
        .sum()
        .sort_values(ascending=False)
    )

    total = p.sum()

    print(
        f"PORTFOLIO = {total:+.2f}R"
    )

    for n in [1, 3, 5, 10]:

        remaining = (
            total
            - p.head(n).sum()
        )

        print(
            f"REMOVE TOP {n:2d} = "
            f"{remaining:+.2f}R"
        )

    print("\nSYMBOL CONTRIBUTION:")

    for sym, value in p.items():
        print(
            f"{sym:<6} {value:+.2f}R"
        )


def leave_one_out(e):
    total = stats(e)[3]

    vals = []

    for sym in sorted(
        e["symbol"].unique()
    ):

        x = e[
            e["symbol"] != sym
        ]

        vals.append(
            (
                sym,
                stats(x)[3]
            )
        )

    vals.sort(
        key=lambda z: z[1]
    )

    print("\nLEAVE-ONE-SYMBOL-OUT")

    print(
        f"FULL = {total:+.2f}R"
    )

    print(
        f"WORST REMOVAL:"
    )

    for sym, value in vals[:5]:
        print(
            f"remove {sym:<6} "
            f"=> {value:+.2f}R"
        )


print("=" * 72)
print("CANDIDATE 11 — SHORT ONLY")
print("FINAL ROBUSTNESS GATE")
print("=" * 72)

data = {}
bad = {}

for sym in SYMS:

    try:

        d = fetch(sym)
        data[sym] = indicators(d)

        print(
            f"{sym:<6} PASS {len(d)}",
            flush=True
        )

    except Exception as e:

        bad[sym] = str(e)

        print(
            f"{sym:<6} ERROR {e}",
            flush=True
        )

if bad:
    raise SystemExit(
        f"ABORT: {len(bad)} symbols failed"
    )


common = set(
    data[SYMS[0]]["ts"]
)

for sym in SYMS[1:]:
    common &= set(
        data[sym]["ts"]
    )

common = sorted(common)

if len(common) != N:
    raise SystemExit(
        f"ABORT COMMON={len(common)} "
        f"EXPECTED={N}"
    )

print("\nDATA AUDIT = PASS")
print(
    f"SYMBOLS={len(data)} "
    f"COMMON={len(common)}"
)
print(
    f"{common[0]} -> {common[-1]}"
)


for tp in [1.5, 2.0]:

    all_events = []

    for sym, d in data.items():

        x = events(
            d,
            tp,
            sym
        )

        if not x.empty:
            all_events.append(x)

    if not all_events:
        raise SystemExit(
            f"NO EVENTS TP{tp:g}"
        )

    raw = (
        pd.concat(
            all_events,
            ignore_index=True
        )
        .sort_values("entry")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 72)
    print(f"TP{tp:g}R — FINAL ROBUSTNESS")
    print(
        f"EVENTS = {len(raw)}"
    )
    print("=" * 72)

    for cost in COSTS:

        x = add_r(
            raw,
            cost,
            tp
        )

        s = stats(x)

        print(
            f"COST {cost:.3f} | "
            f"N={s[0]} "
            f"Exp={s[1]:+.4f} "
            f"PF={s[2]:.3f} "
            f"Total={s[3]:+.2f}R "
            f"DD={s[4]:+.2f}R"
        )

    base = add_r(
        raw,
        0.003,
        tp
    )

    print("\nNO-OVERLAP")

    no = no_overlap(base)

    s = stats(no)

    print(
        f"N={s[0]} "
        f"Exp={s[1]:+.4f} "
        f"PF={s[2]:.3f} "
        f"Total={s[3]:+.2f}R "
        f"DD={s[4]:+.2f}R"
    )

    print("\nCONCENTRATION")

    concentration(
        base
    )

    print("\nLEAVE-ONE-OUT")

    leave_one_out(
        base
    )

    print("\nTEMPORAL BASELINE")

    temporal(
        base,
        common
    )

    print("\nTEMPORAL NO-OVERLAP")

    temporal(
        no,
        common
    )


print("\n" + "=" * 72)
print("FINAL ROBUSTNESS GATE COMPLETE")
print("=" * 72)
