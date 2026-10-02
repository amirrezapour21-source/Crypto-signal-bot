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

END_TS = int(pd.Timestamp("2026-09-29T20:00:00Z").timestamp())
START_TS = END_TS - (N - 1) * STEP
PAD = 3

START_DT = pd.Timestamp("2024-09-29T12:00:00Z")
END_DT = pd.Timestamp("2026-09-29T20:00:00Z")

BASE_COSTS = [0.003, 0.004, 0.005]


def get_json(params):
    last = None

    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=12)

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

        if chunk_end >= end_fetch:
            break

        pos = chunk_end - (OVERLAP - 1) * STEP

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

    diff = d["ts"].diff().dt.total_seconds()

    blocks = []
    start = 0

    for i in range(1, len(d)):
        if diff.iloc[i] != STEP:
            blocks.append((start, i))
            start = i

    blocks.append((start, len(d)))

    valid = [
        (a, b)
        for a, b in blocks
        if b - a >= N
    ]

    if not valid:
        longest = max(
            (b - a for a, b in blocks),
            default=0
        )

        raise ValueError(
            f"{sym}: no contiguous {N}; "
            f"received={len(d)} "
            f"blocks={len(blocks)} "
            f"longest={longest}"
        )

    a, b = valid[-1]

    d = d.iloc[b-N:b].reset_index(drop=True)

    if len(d) != N:
        raise ValueError(
            f"{sym}: final={len(d)}"
        )

    gaps = (
        d["ts"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    if not gaps.eq(STEP).all():
        raise ValueError(
            f"{sym}: final timestamp gap"
        )

    return d


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


def raw_events(d, tp):
    out = []

    for i in range(21, len(d) - HOLD):

        # Candidate 11 SHORT trigger
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

            # SL FIRST
            if hi >= sl:
                outcome = "SL"
                exit_i = j
                break

            if lo <= target:
                outcome = "TP"
                exit_i = j
                break

        out.append({
            "entry": d["ts"].iloc[i],
            "exit": d["ts"].iloc[exit_i],
            "symbol": None,
            "outcome": outcome,
            "tp": tp,
            "entry_i": i
        })

    return pd.DataFrame(out)


def apply_cost(e, cost):
    e = e.copy()

    e["r"] = np.where(
        e["outcome"].eq("TP"),
        e["tp"] - cost,
        np.where(
            e["outcome"].eq("SL"),
            -1.0 - cost,
            -cost
        )
    )

    return e


def remove_overlaps(e):
    if e.empty:
        return e.copy()

    parts = []

    for sym, x in e.groupby("symbol"):

        x = (
            x.sort_values("entry")
             .reset_index(drop=True)
        )

        accepted = []
        last_exit = None

        for _, row in x.iterrows():

            if (
                last_exit is None
                or row["entry"] > last_exit
            ):
                accepted.append(row)
                last_exit = row["exit"]

        if accepted:
            parts.append(
                pd.DataFrame(accepted)
            )

    if not parts:
        return e.iloc[0:0].copy()

    return pd.concat(
        parts,
        ignore_index=True
    )


def stats(x):
    if x.empty:
        return {
            "N": 0,
            "Exp": np.nan,
            "PF": np.nan,
            "Total": 0.0,
            "DD": 0.0
        }

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

    return {
        "N": len(r),
        "Exp": r.mean(),
        "PF": pf,
        "Total": r.sum(),
        "DD": dd
    }


def print_stats(label, x):
    s = stats(x)

    print(
        f"{label:<34} "
        f"N={s['N']:4d} "
        f"Exp={s['Exp']:+.4f} "
        f"PF={s['PF']:.3f} "
        f"Total={s['Total']:+.2f}R "
        f"DD={s['DD']:+.2f}R"
    )


def concentration(e):
    if e.empty:
        return

    p = (
        e.groupby("symbol")["r"]
         .sum()
         .sort_values(ascending=False)
    )

    total = p.sum()

    print("\n" + "-" * 72)
    print("CONCENTRATION")
    print("-" * 72)

    print(f"PORTFOLIO TOTAL = {total:+.2f}R")

    for n in [1, 3, 5]:

        removed = p.iloc[:n].sum()
        remaining = total - removed

        print(
            f"REMOVE TOP {n:<2d} = "
            f"{remaining:+.2f}R"
        )

    print("\nTOP SYMBOLS:")

    for sym, val in p.head(10).items():
        print(
            f"{sym:<6} {val:+.2f}R"
        )


def temporal(e):
    if e.empty:
        return

    common = sorted(
        set(
            pd.date_range(
                START_DT,
                END_DT,
                freq="4h",
                tz="UTC"
            )
        )
        & set(e["entry"])
    )

    if len(common) < 5:
        return

    folds = np.array_split(
        np.array(common),
        5
    )

    print("\n" + "-" * 72)
    print("TEMPORAL ROBUSTNESS")
    print("-" * 72)

    positive = 0
    all_oos = []

    for k in range(5):

        start = pd.Timestamp(folds[k][0])

        end = (
            pd.Timestamp(folds[k][-1])
            + pd.Timedelta(hours=4)
        )

        ox = e[
            (e["entry"] >= start) &
            (e["entry"] < end) &
            (e["exit"] < end)
        ]

        s = stats(ox)

        if s["Total"] > 0:
            positive += 1

        all_oos.append(ox)

        print(
            f"FOLD {k+1} | "
            f"OOS N={s['N']:4d} "
            f"Exp={s['Exp']:+.4f} "
            f"PF={s['PF']:.3f} "
            f"Total={s['Total']:+.2f}R "
            f"DD={s['DD']:+.2f}R"
        )

    if all_oos:
        oos = pd.concat(
            all_oos,
            ignore_index=True
        )

        s = stats(oos)

        print("-" * 72)

        print(
            f"AGG OOS | "
            f"N={s['N']} "
            f"Exp={s['Exp']:+.4f} "
            f"PF={s['PF']:.3f} "
            f"Total={s['Total']:+.2f}R "
            f"DD={s['DD']:+.2f}R"
        )

        print(
            f"POSITIVE FOLDS = "
            f"{positive}/5"
        )


print("=" * 72)
print("CANDIDATE 11 — SHORT ONLY")
print("ROBUSTNESS / STRESS TEST")
print("=" * 72)

# ---------------------------------------------------------
# 1. DATA FETCH — فقط یک بار
# ---------------------------------------------------------

data = {}
bad = {}

for s in SYMS:

    try:
        d = fetch(s)
        data[s] = indicators(d)

        print(
            f"{s:<6} {len(d)} | "
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


# ---------------------------------------------------------
# 2. GENERATE EVENTS — TP1.5 / TP2
# ---------------------------------------------------------

for tp in [1.5, 2.0]:

    all_events = []

    for sym, d in data.items():

        e = raw_events(d, tp)

        if not e.empty:
            e["symbol"] = sym
            all_events.append(e)

    if not all_events:
        raise SystemExit(
            f"ABORT: NO EVENTS TP{tp:g}"
        )

    ev = pd.concat(
        all_events,
        ignore_index=True
    )

    ev = ev.sort_values(
        "entry"
    ).reset_index(drop=True)

    print("\n" + "=" * 72)
    print(f"BASELINE TP{tp:g}R")
    print("=" * 72)

    # -----------------------------------------------------
    # 3. COST STRESS
    # -----------------------------------------------------

    for cost in BASE_COSTS:

        x = apply_cost(ev, cost)

        print_stats(
            f"COST {cost:.3f}",
            x
        )

    # -----------------------------------------------------
    # 4. NO-OVERLAP STRESS
    # -----------------------------------------------------

    x = apply_cost(
        ev,
        0.003
    )

    no_overlap = remove_overlaps(x)

    print("\nNO-OVERLAP:")

    print_stats(
        "NO OVERLAP",
        no_overlap
    )

    # -----------------------------------------------------
    # 5. CONCENTRATION — BASELINE COST
    # -----------------------------------------------------

    x = apply_cost(
        ev,
        0.003
    )

    concentration(x)

    # -----------------------------------------------------
    # 6. CONCENTRATION — NO OVERLAP
    # -----------------------------------------------------

    print("\nNO-OVERLAP CONCENTRATION:")

    concentration(no_overlap)

    # -----------------------------------------------------
    # 7. TEMPORAL ROBUSTNESS — BASELINE
    # -----------------------------------------------------

    print("\nBASELINE TEMPORAL:")

    temporal(x)

    # -----------------------------------------------------
    # 8. TEMPORAL ROBUSTNESS — NO OVERLAP
    # -----------------------------------------------------

    print("\nNO-OVERLAP TEMPORAL:")

    temporal(no_overlap)


print("\n" + "=" * 72)
print("ROBUSTNESS TEST COMPLETE")
print("=" * 72)
