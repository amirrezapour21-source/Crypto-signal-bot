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
COST = 0.003

END_TS = int(pd.Timestamp("2026-09-29T20:00:00Z").timestamp())
START_TS = END_TS - (N - 1) * STEP
PAD = 3  # کندل اضافه دو طرف بازه برای جبران مرزهای startAt/endAt


def get_json(params):

    last = None

    for attempt in range(3):

        try:

            r = requests.get(
                BASE,
                params=params,
                timeout=10
            )

            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

            r.raise_for_status()
            return r.json().get("data", [])

        except requests.RequestException as e:
            last = e
            time.sleep(1)

    raise RuntimeError(f"request failed: {last}")


def fetch(sym):

    rows = []

    CHUNK = 900
    OVERLAP = 2
    pos = START_TS - PAD * STEP
    end_fetch = END_TS + PAD * STEP

    while pos <= end_fetch:

        chunk_end = min(
            pos + (CHUNK - 1) * STEP,
            end_fetch
        )

        data = get_json({
            "symbol": f"{sym}-USDT",
            "type": "4hour",
            "startAt": pos,
            "endAt": chunk_end
        })

        if data:
            rows.extend(data)

        # خروج از حلقه در آخرین قطعه
        if chunk_end >= end_fetch:
            break

        pos = chunk_end - OVERLAP * STEP + STEP

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
        (d["ts"] >= pd.Timestamp(
            "2024-09-29T12:00:00Z"
        )) &
        (d["ts"] <= pd.Timestamp(
            "2026-09-29T20:00:00Z"
        ))
    ].copy()

    ts = d["ts"].astype("int64") // 10**9
    diff = ts.diff()

    blocks = []
    start = 0

    for i in range(1, len(d)):
        if diff.iloc[i] != STEP:
            blocks.append((start, i))
            start = i

    blocks.append((start, len(d)))

    valid = [
        (a,b)
        for a,b in blocks
        if b-a >= N
    ]

    if not valid:
        raise ValueError(
            f"{sym}: no contiguous {N}-candle block; "
            f"received={len(d)}"
        )

    a,b = valid[-1]

    d = d.iloc[b-N:b].reset_index(drop=True)

    if len(d) != N:
        raise ValueError(
            f"{sym}: final={len(d)}"
        )

    if not (
        d["ts"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .eq(STEP)
        .all()
    ):
        raise ValueError(
            f"{sym}: final timestamp gap"
        )

    return d


def indicators(d):

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
            h-l,
            (h-c.shift()).abs(),
            (l-c.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    return d


def events(d, tp):

    out = []

    for i in range(
        21,
        len(d)-HOLD
    ):

        if not (
            d["z"].iloc[i-1] >= 2
            and d["z"].iloc[i] < 2
        ):
            continue

        if not (
            d["close"].iloc[i]
            < d["ema"].iloc[i]
        ):
            continue

        if not (
            d["range20"].iloc[i]
            >= d["rmed"].iloc[i]
        ):
            continue

        if not (
            d["vol"].iloc[i]
            >= d["vmed"].iloc[i]
        ):
            continue

        entry = float(
            d["close"].iloc[i]
        )

        atr = float(
            d["atr"].iloc[i]
        )

        if not np.isfinite(atr) or atr <= 0:
            continue

        sl = entry + ATR_M * atr
        target = entry - tp * ATR_M * atr

        result = -COST
        exit_i = i + HOLD - 1

        for j in range(i+1, i+HOLD):

            hi = float(d["high"].iloc[j])
            lo = float(d["low"].iloc[j])

            # SL-first
            if hi >= sl:
                result = -1 - COST
                exit_i = j
                break

            if lo <= target:
                result = tp - COST
                exit_i = j
                break

        out.append([
            d["ts"].iloc[i],
            d["ts"].iloc[exit_i],
            result
        ])

    return pd.DataFrame(
        out,
        columns=["entry","exit","r"]
    )


def stats(x):

    if len(x) == 0:
        return 0, np.nan, np.nan, 0, 0

    r = x["r"].to_numpy(float)

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


print("=" * 72)
print("CANDIDATE 11 — SHORT ONLY")
print("TIME-CHUNK KUCOIN FETCH + FIXED R ENGINE")
print("=" * 72)

data = {}
bad = {}

for s in SYMS:

    try:

        d = fetch(s)
        data[s] = indicators(d)

        print(
            f"{s:<6} {len(d)} candles | "
            f"{d['ts'].iloc[0]} -> "
            f"{d['ts'].iloc[-1]}",
            flush=True
        )

    except Exception as e:

        bad[s] = str(e)
        print(
            f"{s:<6} ERROR: {e}",
            flush=True
        )


if bad:

    raise SystemExit(
        f"ABORT: {len(bad)} symbols failed"
    )


common = set(
    data[SYMS[0]]["ts"]
)

for s in SYMS[1:]:

    common &= set(
        data[s]["ts"]
    )

common = sorted(common)


if len(common) != N:

    raise SystemExit(
        f"ABORT: COMMON TIMESTAMPS="
        f"{len(common)} EXPECTED={N}"
    )


print("\n" + "=" * 72)
print("DATA AUDIT = PASS")
print(f"VALID SYMBOLS = {len(data)}")
print(f"COMMON TIMESTAMPS = {len(common)}")
print(f"START = {common[0]}")
print(f"END   = {common[-1]}")
print("=" * 72)


folds = np.array_split(
    np.array(common),
    5
)


for tp in [1.5, 2.0]:

    all_events = []

    for s, d in data.items():

        e = events(d, tp)

        if len(e):

            e["symbol"] = s
            all_events.append(e)


    if not all_events:

        raise SystemExit(
            "ABORT: NO EVENTS"
        )


    ev = (
        pd.concat(
            all_events,
            ignore_index=True
        )
        .sort_values("entry")
        .reset_index(drop=True)
    )


    print("\n" + "=" * 72)
    print(f"TP{tp:g}R — SHORT ONLY")
    print(
        f"TOTAL EVENTS = {len(ev)}"
    )
    print("=" * 72)


    oos = []


    for k in range(5):

        start = pd.Timestamp(
            folds[k][0]
        )

        end = (
            pd.Timestamp(folds[k][-1])
            + pd.Timedelta(hours=4)
        )


        isx = ev[
            (ev["entry"] < start) &
            (ev["exit"] < start)
        ]

        ox = ev[
            (ev["entry"] >= start) &
            (ev["entry"] < end) &
            (ev["exit"] < end)
        ]


        a = stats(isx)
        b = stats(ox)


        print(
            f"FOLD {k+1} | "
            f"IS N={a[0]:4d} "
            f"Exp={a[1]:+.4f} "
            f"PF={a[2]:.3f} "
            f"Total={a[3]:+.2f}R | "
            f"OOS N={b[0]:4d} "
            f"Exp={b[1]:+.4f} "
            f"PF={b[2]:.3f} "
            f"Total={b[3]:+.2f}R "
            f"DD={b[4]:+.2f}R"
        )

        oos.append(ox)


    oos_all = pd.concat(
        oos,
        ignore_index=True
    )

    z = stats(oos_all)

    positive = sum(
        stats(x)[3] > 0
        for x in oos
    )


    print("-" * 72)

    print(
        f"TP{tp:g}R AGG OOS | "
        f"N={z[0]} "
        f"Exp={z[1]:+.4f} "
        f"PF={z[2]:.3f} "
        f"Total={z[3]:+.2f}R "
        f"DD={z[4]:+.2f}R"
    )

    print(
        f"Positive OOS folds = "
        f"{positive}/5"
    )

    print(
        "TEMPORAL VALIDATION =",
        "PASS"
        if (
            positive >= 4
            and z[1] > 0
            and z[2] > 1
        )
        else "FAIL"
    )


print("\n" + "=" * 72)
print("RUN COMPLETE")
print("=" * 72)
