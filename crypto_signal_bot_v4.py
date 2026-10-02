import requests, pandas as pd, numpy as np, time

SYMS = ["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT","SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC","BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP","WIF","PEPE","FLOKI"]

N = 4380
STEP = 14400
CHUNK = 900
OVERLAP = 2

START = pd.Timestamp("2024-09-29 12:00:00", tz="UTC")
END   = pd.Timestamp("2026-09-29 20:00:00", tz="UTC")

HOLDOUT_FRAC = 0.20
COSTS = [0.003, 0.004, 0.005]
TPS = [1.5, 2.0]

BASE = "https://api.kucoin.com/api/v1/market/candles"

def fetch(sym):
    rows = []
    end = END

    while end >= START:
        begin = max(START, end - pd.Timedelta(seconds=(CHUNK-1)*STEP))

        try:
            r = requests.get(
                BASE,
                params={
                    "symbol": f"{sym}-USDT",
                    "type": "4hour",
                    "startAt": int(begin.timestamp()),
                    "endAt": int(end.timestamp())
                },
                timeout=20
            )
            j = r.json()
            data = j.get("data", [])
        except Exception:
            data = []

        for x in data:
            if len(x) >= 7:
                rows.append(x)

        end = begin - pd.Timedelta(seconds=STEP)
        time.sleep(0.12)

    if not rows:
        return None

    d = pd.DataFrame(rows, columns=[
        "ts","open","close","high","low","volume","turnover"
    ])

    d["ts"] = pd.to_datetime(pd.to_numeric(d["ts"]), unit="s", utc=True)

    for c in ["open","close","high","low","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = (
        d.dropna()
         .drop_duplicates("ts")
         .sort_values("ts")
         .set_index("ts")
    )

    idx = pd.date_range(
        START,
        END,
        freq="4h",
        tz="UTC"
    )

    d = d.reindex(idx).dropna()

    if len(d) < N:
        return None

    d = d.iloc[-N:].copy()

    if len(d) != N:
        return None

    if not d.index.is_monotonic_increasing:
        return None

    if len(d.index.unique()) != N:
        return None

    gaps = d.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(hours=4)).all():
        return None

    return d

def prepare(d):
    d = d.copy()

    d["mean20"] = d["close"].rolling(20).mean()
    d["std20"] = d["close"].rolling(20).std()
    d["z"] = (d["close"] - d["mean20"]) / d["std20"]

    d["range20"] = (
        d["high"].rolling(20).max()
        - d["low"].rolling(20).min()
    )

    d["range_med"] = d["range20"].rolling(20).median()
    d["vol_med"] = d["volume"].rolling(20).median()

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs()
    ], axis=1).max(axis=1)

    d["atr20"] = tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    d["ema200"] = d["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    return d

def events(d, tp):
    out = []

    for i in range(200, len(d)-1):
        p = d.iloc[i-1]
        c = d.iloc[i]

        if not (
            np.isfinite(p["z"]) and
            np.isfinite(c["z"]) and
            np.isfinite(c["atr20"])
        ):
            continue

        # SHORT ONLY — Candidate 11 unchanged
        signal = (
            c["close"] < c["ema200"]
            and p["z"] >= 2
            and c["z"] < 2
            and c["range20"] >= c["range_med"]
            and c["volume"] >= c["vol_med"]
        )

        if not signal:
            continue

        entry_i = i + 1

        if entry_i >= len(d):
            continue

        entry = d.iloc[entry_i]
        ep = float(entry["open"])
        sl = ep + 1.25 * float(c["atr20"])
        risk = sl - ep
        target = ep - tp * risk

        result = None
        exit_ts = None

        last = min(len(d), entry_i + 31)

        for j in range(entry_i, last):
            x = d.iloc[j]

            # SHORT: SL first when both touch same candle
            if x["high"] >= sl:
                result = -1.0
                exit_ts = d.index[j]
                break

            if x["low"] <= target:
                result = tp
                exit_ts = d.index[j]
                break

        if result is None:
            result = 0.0
            exit_ts = d.index[last-1]

        out.append({
            "ts": d.index[entry_i],
            "exit": exit_ts,
            "r": result
        })

    return pd.DataFrame(out)

def stat(ev, cost=0.003):
    if ev.empty:
        return (0, 0, 0, 0, 0)

    x = ev.sort_values("ts").copy()
    r = x["r"] - cost

    total = float(r.sum())
    exp = float(r.mean())

    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()

    pf = wins / losses if losses > 0 else np.inf

    eq = r.cumsum()
    dd = float((eq - eq.cummax()).min())

    return len(r), exp, pf, total, dd

def no_overlap(ev):
    if ev.empty:
        return ev

    x = ev.sort_values(["sym","ts"]).copy()
    keep = []
    last_exit = {}

    for _, row in x.iterrows():
        s = row["sym"]

        if s not in last_exit or row["ts"] >= last_exit[s]:
            keep.append(True)
            last_exit[s] = row["exit"]
        else:
            keep.append(False)

    return x.loc[keep].sort_values("ts")

print("="*72)
print("CANDIDATE 11 — SHORT ONLY")
print("FINAL INTERNAL HOLDOUT VALIDATION")
print("="*72)

DATA = {}

for s in SYMS:
    d = fetch(s)

    if d is None:
        print(f"{s:<6} FAIL")
        continue

    DATA[s] = prepare(d)
    print(f"{s:<6} PASS {len(d)}")

print()
print("DATA AUDIT")

if len(DATA) != len(SYMS):
    raise SystemExit(
        f"ABORT: {len(DATA)}/{len(SYMS)} symbols passed"
    )

common = set(next(iter(DATA.values())).index)

for d in DATA.values():
    common &= set(d.index)

common = pd.DatetimeIndex(sorted(common))

if len(common) != N:
    raise SystemExit(
        f"ABORT: COMMON={len(common)} expected={N}"
    )

print(f"SYMBOLS={len(DATA)} COMMON={len(common)}")
print(f"{common[0]} -> {common[-1]}")

# ------------------------------------------------------------
# INTERNAL HOLDOUT: LAST 20%
# ------------------------------------------------------------

cut = int(N * (1 - HOLDOUT_FRAC))
holdout_ts = common[cut:]

HSTART = holdout_ts[0]
HEND = holdout_ts[-1]

print()
print("="*72)
print("HOLDOUT DEFINITION")
print("="*72)
print(f"TRAIN/DEV CANDLES : {cut}")
print(f"HOLDOUT CANDLES   : {len(holdout_ts)}")
print(f"HOLDOUT START     : {HSTART}")
print(f"HOLDOUT END       : {HEND}")
print("STRATEGY PARAMETERS: FROZEN")
print("="*72)

ALL = {}

for tp in TPS:
    parts = []

    for s, d in DATA.items():
        e = events(d, tp)

        if e.empty:
            continue

        e["sym"] = s

        # Only untouched tail timestamp block
        e = e[
            (e["ts"] >= HSTART) &
            (e["ts"] <= HEND)
        ]

        if not e.empty:
            parts.append(e)

    ALL[tp] = (
        pd.concat(parts, ignore_index=True)
        .sort_values("ts")
        if parts else
        pd.DataFrame(columns=["ts","exit","r","sym"])
    )

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

for tp in TPS:

    ev = ALL[tp]

    print()
    print("="*72)
    print(f"TP{tp}R — HOLDOUT")
    print("="*72)

    print(f"EVENTS = {len(ev)}")

    for cost in COSTS:
        n, ex, pf, total, dd = stat(ev, cost)
        print(
            f"COST {cost:.3f} | "
            f"N={n} Exp={ex:+.4f} "
            f"PF={pf:.3f} Total={total:+.2f}R DD={dd:+.2f}R"
        )

    # --------------------------------------------------------
    # NO OVERLAP
    # --------------------------------------------------------

    no = no_overlap(ev)

    n, ex, pf, total, dd = stat(no, 0.003)

    print()
    print("NO-OVERLAP")
    print(
        f"N={n} Exp={ex:+.4f} "
        f"PF={pf:.3f} Total={total:+.2f}R DD={dd:+.2f}R"
    )

    # --------------------------------------------------------
    # EXACT TEMPORAL BLOCKS
    # --------------------------------------------------------

    print()
    print("TEMPORAL HOLDOUT BLOCKS")

    folds = np.array_split(holdout_ts, 5)

    positive = 0
    aggregate = []

    for k, block in enumerate(folds, 1):

        a = block[0]
        b = block[-1]

        f = ev[
            (ev["ts"] >= a) &
            (ev["ts"] <= b)
        ].sort_values("ts")

        n, ex, pf, total, dd = stat(f, 0.003)

        if total > 0:
            positive += 1

        aggregate.append(f)

        print(
            f"FOLD {k} | "
            f"{a} -> {b} | "
            f"N={n:3d} "
            f"Exp={ex:+.4f} "
            f"PF={pf:.3f} "
            f"Total={total:+.2f}R "
            f"DD={dd:+.2f}R"
        )

    agg = (
        pd.concat(aggregate, ignore_index=True)
        if aggregate else
        pd.DataFrame(columns=ev.columns)
    )

    n, ex, pf, total, dd = stat(agg, 0.003)

    print("-"*72)
    print(
        f"AGG HOLDOUT | "
        f"N={n} Exp={ex:+.4f} "
        f"PF={pf:.3f} "
        f"Total={total:+.2f}R "
        f"DD={dd:+.2f}R"
    )

    print(f"POSITIVE FOLDS = {positive}/5")

    # --------------------------------------------------------
    # SYMBOL JACKKNIFE
    # --------------------------------------------------------

    print()
    print("SYMBOL JACKKNIFE")

    full = stat(ev, 0.003)[3]

    removals = []

    for s in SYMS:
        x = ev[ev["sym"] != s]
        t = stat(x, 0.003)[3]
        removals.append((s, t))

    removals.sort(key=lambda z: z[1])

    print(f"FULL = {full:+.2f}R")

    for s, t in removals[:5]:
        print(f"REMOVE {s:<6} => {t:+.2f}R")

    print()
    print("BEST REMOVALS")

    for s, t in removals[-5:][::-1]:
        print(f"REMOVE {s:<6} => {t:+.2f}R")

print()
print("="*72)
print("FINAL INTERNAL HOLDOUT VALIDATION COMPLETE")
print("="*72)
