import requests, pandas as pd, numpy as np, time

# ============================================================
# CANDIDATE 11 — SHORT ONLY
# FINAL INTERNAL HOLDOUT + STATISTICAL ROBUSTNESS
# ============================================================

SYMS = [
    "BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT",
    "SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC",
    "BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP",
    "WIF","PEPE","FLOKI"
]

N = 4380
STEP = 14400
PAGE = 1490

END = pd.Timestamp(
    "2026-09-29 20:00:00",
    tz="UTC"
)

COSTS = [0.003, 0.004, 0.005]
TPS = [1.5, 2.0]
HOLDOUT = 0.20

# Statistical robustness
BOOT = 20000
SEED = 20260927

URL = "https://api.kucoin.com/api/v1/market/candles"


# ============================================================
# DATA FETCH — FROZEN
# ============================================================

def get_page(sym, start_at, end_at):

    last = None

    for attempt in range(3):

        try:
            r = requests.get(
                URL,
                params={
                    "symbol": f"{sym}-USDT",
                    "type": "4hour",
                    "startAt": start_at,
                    "endAt": end_at
                },
                timeout=15
            )

            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

            r.raise_for_status()
            return r.json(), None

        except Exception as e:
            last = e
            time.sleep(1)

    return None, f"REQUEST_ERROR={last}"


def fetch(sym):

    rows = []

    cursor = int(END.timestamp()) + STEP

    min_ts = int(
        (END - pd.Timedelta(hours=4*(N+100))).timestamp()
    )

    while cursor >= min_ts:

        j, err = get_page(
            sym,
            cursor - PAGE * STEP,
            cursor
        )

        if err:
            return None, err

        if j.get("code") != "200000":
            return None, f"API_CODE={j.get('code')}"

        data = j.get("data", [])

        if not data:
            return None, "EMPTY_PAGE"

        rows.extend(data)

        ts = [
            int(x[0])
            for x in data
            if len(x) >= 7
        ]

        if not ts:
            return None, "BAD_PAGE"

        oldest = min(ts)

        if oldest >= cursor:
            return None, "NO_PROGRESS"

        cursor = oldest

        time.sleep(0.08)

    if not rows:
        return None, "NO_DATA"

    d = pd.DataFrame(
        rows,
        columns=[
            "ts","open","close","high","low",
            "volume","turnover"
        ]
    )

    d["ts"] = pd.to_datetime(
        pd.to_numeric(d["ts"]),
        unit="s",
        utc=True
    )

    for c in [
        "open","close","high","low","volume"
    ]:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    d = (
        d.dropna()
         .drop_duplicates("ts")
         .sort_values("ts")
         .set_index("ts")
    )

    d = d[d.index <= END]

    if len(d) < N:
        return None, f"TOTAL={len(d)}"

    idx = d.index

    gap = (
        idx.to_series()
        .diff()
        .ne(pd.Timedelta(hours=4))
    )

    group = gap.cumsum()

    blocks = (
        pd.DataFrame({
            "ts": idx,
            "g": group.values
        })
        .groupby("g")
        .agg(
            n=("ts","size"),
            start=("ts","min"),
            end=("ts","max")
        )
    )

    valid = blocks[
        blocks["n"] >= N
    ]

    if valid.empty:
        return None, (
            f"NO_CONTIGUOUS_{N}; "
            f"TOTAL={len(d)}; "
            f"LONGEST={int(blocks['n'].max())}"
        )

    b = valid.iloc[-1]

    out = d.loc[
        (d.index >= b["start"]) &
        (d.index <= b["end"])
    ].copy()

    if len(out) > N:
        out = out.iloc[-N:]

    if len(out) != N:
        return None, f"BLOCK={len(out)}"

    return out, "PASS"


# ============================================================
# INDICATORS — FROZEN
# ============================================================

def prepare(d):

    d = d.copy()

    d["mean20"] = d["close"].rolling(20).mean()

    d["std20"] = d["close"].rolling(20).std()

    d["z"] = (
        (d["close"] - d["mean20"]) /
        d["std20"]
    )

    d["range20"] = (
        d["high"].rolling(20).max() -
        d["low"].rolling(20).min()
    )

    d["range_med"] = (
        d["range20"].rolling(20).median()
    )

    d["vol_med"] = (
        d["volume"].rolling(20).median()
    )

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


# ============================================================
# EVENT ENGINE — FROZEN
# ============================================================

def events(d, tp, end_limit):

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

        signal = (
            c["close"] < c["ema200"]
            and p["z"] >= 2
            and c["z"] < 2
            and c["range20"] >= c["range_med"]
            and c["volume"] >= c["vol_med"]
        )

        if not signal:
            continue

        ei = i + 1

        if ei >= len(d):
            continue

        ep = float(d.iloc[ei]["open"])

        sl = ep + 1.25 * float(c["atr20"])

        risk = sl - ep

        target = ep - tp * risk

        result = None
        exit_ts = None

        last = min(
            len(d),
            ei + 31
        )

        for j in range(ei, last):

            x = d.iloc[j]

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

        if exit_ts > end_limit:
            continue

        out.append({
            "ts": d.index[ei],
            "exit": exit_ts,
            "r": result
        })

    return pd.DataFrame(out)


# ============================================================
# BASIC STATS
# ============================================================

def stat(ev, cost=.003):

    if ev.empty:
        return 0, 0, 0, 0, 0

    r = (
        ev.sort_values("ts")["r"]
        .to_numpy(dtype=float)
        - cost
    )

    total = float(r.sum())

    exp = float(r.mean())

    win = float(r[r > 0].sum())

    loss = float(-r[r < 0].sum())

    pf = (
        win / loss
        if loss > 0
        else np.inf
    )

    eq = np.cumsum(r)

    dd = float(
        (eq - np.maximum.accumulate(eq)).min()
    )

    return len(r), exp, pf, total, dd


# ============================================================
# NO OVERLAP
# ============================================================

def no_overlap(ev):

    if ev.empty:
        return ev

    x = ev.sort_values(
        ["sym", "ts"]
    ).copy()

    keep = []
    last = {}

    for _, r in x.iterrows():

        s = r["sym"]

        if (
            s not in last or
            r["ts"] >= last[s]
        ):
            keep.append(True)
            last[s] = r["exit"]
        else:
            keep.append(False)

    return x.loc[keep].sort_values("ts")


# ============================================================
# DATA
# ============================================================

print("=" * 72)
print("CANDIDATE 11 — SHORT ONLY")
print("FINAL INTERNAL HOLDOUT + STATISTICAL ROBUSTNESS")
print("=" * 72)

DATA = {}

for s in SYMS:

    d, msg = fetch(s)

    if d is None:

        print(
            f"{s:<6} FAIL | {msg}",
            flush=True
        )

    else:

        DATA[s] = prepare(d)

        print(
            f"{s:<6} PASS | "
            f"{len(d)} | "
            f"{d.index[0]} -> {d.index[-1]}",
            flush=True
        )


# ============================================================
# DATA AUDIT
# ============================================================

print()
print("=" * 72)
print("DATA AUDIT")
print("=" * 72)

if len(DATA) != len(SYMS):

    raise SystemExit(
        f"ABORT: {len(DATA)}/{len(SYMS)} symbols passed"
    )

common = set(
    next(iter(DATA.values())).index
)

for d in DATA.values():
    common &= set(d.index)

common = pd.DatetimeIndex(
    sorted(common)
)

if len(common) != N:

    raise SystemExit(
        f"ABORT: COMMON={len(common)} expected={N}"
    )

g = common.to_series().diff().dropna()

if not (
    g == pd.Timedelta(hours=4)
).all():

    raise SystemExit(
        "ABORT: COMMON GAP"
    )

print(f"SYMBOLS={len(DATA)}")
print(f"COMMON={len(common)}")
print(
    f"WINDOW={common[0]} -> {common[-1]}"
)
print("DATA AUDIT=PASS")


# ============================================================
# HOLDOUT
# ============================================================

cut = int(
    N * (1 - HOLDOUT)
)

hold = common[cut:]

HSTART = hold[0]
HEND = hold[-1]

print()
print("=" * 72)
print("HOLDOUT DEFINITION")
print("=" * 72)
print(f"TRAIN/DEV={cut}")
print(f"HOLDOUT={len(hold)}")
print(f"START={HSTART}")
print(f"END={HEND}")
print("PARAMETERS=FROZEN")
print("=" * 72)


# ============================================================
# BUILD EVENTS
# ============================================================

ALL = {}

for tp in TPS:

    parts = []

    for s, d in DATA.items():

        e = events(
            d,
            tp,
            HEND
        )

        if e.empty:
            continue

        e["sym"] = s

        e = e[
            (e["ts"] >= HSTART) &
            (e["ts"] <= HEND)
        ]

        if not e.empty:
            parts.append(e)

    ALL[tp] = (
        pd.concat(
            parts,
            ignore_index=True
        )
        .sort_values("ts")
        if parts
        else
        pd.DataFrame(
            columns=[
                "ts",
                "exit",
                "r",
                "sym"
            ]
        )
    )


# ============================================================
# HOLDOUT RESULTS
# ============================================================

for tp in TPS:

    ev = ALL[tp]

    print()
    print("=" * 72)
    print(f"TP{tp}R — HOLDOUT")
    print("=" * 72)

    print(
        f"EVENTS={len(ev)}"
    )

    for cost in COSTS:

        n, ex, pf, total, dd = stat(
            ev,
            cost
        )

        print(
            f"COST {cost:.3f} | "
            f"N={n} "
            f"Exp={ex:+.4f} "
            f"PF={pf:.3f} "
            f"Total={total:+.2f}R "
            f"DD={dd:+.2f}R"
        )

    no = no_overlap(ev)

    n, ex, pf, total, dd = stat(
        no,
        .003
    )

    print()
    print("NO-OVERLAP")

    print(
        f"N={n} "
        f"Exp={ex:+.4f} "
        f"PF={pf:.3f} "
        f"Total={total:+.2f}R "
        f"DD={dd:+.2f}R"
    )

    print()
    print("TEMPORAL HOLDOUT BLOCKS")

    folds = np.array_split(
        hold,
        5
    )

    positive = 0
    parts = []

    for k, b in enumerate(
        folds,
        1
    ):

        if len(b) == 0:
            continue

        a = b[0]
        z = b[-1]

        f = ev[
            (ev["ts"] >= a) &
            (ev["ts"] <= z)
        ]

        n, ex, pf, total, dd = stat(
            f,
            .003
        )

        if total > 0:
            positive += 1

        parts.append(f)

        print(
            f"FOLD {k} | "
            f"N={n} "
            f"Exp={ex:+.4f} "
            f"PF={pf:.3f} "
            f"Total={total:+.2f}R "
            f"DD={dd:+.2f}R"
        )

    if parts:

        agg = pd.concat(
            parts,
            ignore_index=True
        )

        n, ex, pf, total, dd = stat(
            agg,
            .003
        )

        print("-" * 72)

        print(
            f"AGG HOLDOUT | "
            f"N={n} "
            f"Exp={ex:+.4f} "
            f"PF={pf:.3f} "
            f"Total={total:+.2f}R "
            f"DD={dd:+.2f}R"
        )

    print(
        f"POSITIVE FOLDS={positive}/5"
    )

    print()
    print("SYMBOL JACKKNIFE")

    full = stat(
        ev,
        .003
    )[3]

    rem = []

    for s in SYMS:

        t = stat(
            ev[ev["sym"] != s],
            .003
        )[3]

        rem.append(
            (s, t)
        )

    rem.sort(
        key=lambda x: x[1]
    )

    print(
        f"FULL={full:+.2f}R"
    )

    for s, t in rem[:5]:

        print(
            f"REMOVE {s:<6} => {t:+.2f}R"
        )


# ============================================================
# STATISTICAL ROBUSTNESS
# ============================================================

def bootstrap_trade_level(r, B, rng):

    n = len(r)

    mean_boot = np.empty(B)
    total_boot = np.empty(B)
    pf_boot = np.empty(B)

    for i in range(B):

        s = rng.choice(
            r,
            size=n,
            replace=True
        )

        total_boot[i] = s.sum()
        mean_boot[i] = s.mean()

        win = s[s > 0].sum()
        loss = -s[s < 0].sum()

        pf_boot[i] = (
            win / loss
            if loss > 0
            else np.inf
        )

    return (
        mean_boot,
        total_boot,
        pf_boot
    )


def bootstrap_symbol_level(ev, B, rng):

    symbols = ev["sym"].unique()

    groups = []

    for s in symbols:

        r = (
            ev.loc[
                ev["sym"] == s,
                "r"
            ]
            .to_numpy(dtype=float)
            - 0.003
        )

        groups.append({
            "n": len(r),
            "total": r.sum(),
            "win": r[r > 0].sum(),
            "loss": -r[r < 0].sum()
        })

    m = len(groups)

    n_arr = np.array(
        [x["n"] for x in groups],
        dtype=float
    )

    total_arr = np.array(
        [x["total"] for x in groups],
        dtype=float
    )

    win_arr = np.array(
        [x["win"] for x in groups],
        dtype=float
    )

    loss_arr = np.array(
        [x["loss"] for x in groups],
        dtype=float
    )

    mean_boot = np.empty(B)
    total_boot = np.empty(B)
    pf_boot = np.empty(B)

    for i in range(B):

        pick = rng.integers(
            0,
            m,
            size=m
        )

        n = n_arr[pick].sum()

        total = total_arr[pick].sum()

        win = win_arr[pick].sum()

        loss = loss_arr[pick].sum()

        total_boot[i] = total

        mean_boot[i] = (
            total / n
            if n > 0
            else 0
        )

        pf_boot[i] = (
            win / loss
            if loss > 0
            else np.inf
        )

    return (
        mean_boot,
        total_boot,
        pf_boot
    )


def monte_carlo_dd(r, B, rng):

    n = len(r)

    dd = np.empty(B)

    for i in range(B):

        s = rng.permutation(r)

        eq = np.cumsum(s)

        peak = np.maximum.accumulate(
            np.r_[0.0, eq]
        )

        dd[i] = (
            eq - peak[1:]
        ).min()

    return dd


def print_ci(name, x):

    x = x[
        np.isfinite(x)
    ]

    q = np.percentile(
        x,
        [2.5, 50, 97.5]
    )

    print(
        f"{name} CI95% = "
        f"[{q[0]:+.4f}, "
        f"{q[2]:+.4f}] "
        f"median={q[1]:+.4f}"
    )

    return q


# ============================================================
# RUN STATISTICAL ROBUSTNESS
# ============================================================

print()
print()
print("#" * 72)
print("STATISTICAL ROBUSTNESS")
print("#" * 72)
print(
    f"BOOTSTRAPS={BOOT} | SEED={SEED}"
)

for tp in TPS:

    ev = ALL[tp].copy()

    if ev.empty:
        print(
            f"\nTP{tp}R | NO EVENTS"
        )
        continue

    # Net R at baseline cost
    r = (
        ev.sort_values("ts")["r"]
        .to_numpy(dtype=float)
        - 0.003
    )

    rng = np.random.default_rng(
        SEED + int(tp * 100)
    )

    # --------------------------------------------------------
    # Point estimate
    # --------------------------------------------------------

    n = len(r)

    total = float(r.sum())

    exp = float(r.mean())

    win = float(
        r[r > 0].sum()
    )

    loss = float(
        -r[r < 0].sum()
    )

    pf = (
        win / loss
        if loss > 0
        else np.inf
    )

    print()
    print("=" * 72)
    print(
        f"TP{tp}R — STATISTICAL ROBUSTNESS"
    )
    print("=" * 72)

    print(
        f"TRADES={n}"
    )

    print(
        f"POINT Exp={exp:+.4f} "
        f"PF={pf:.3f} "
        f"Total={total:+.2f}R"
    )

    # --------------------------------------------------------
    # Trade-level bootstrap
    # --------------------------------------------------------

    mean_b, total_b, pf_b = (
        bootstrap_trade_level(
            r,
            BOOT,
            rng
        )
    )

    print()
    print("TRADE-LEVEL BOOTSTRAP")

    mean_ci = print_ci(
        "MEAN R",
        mean_b
    )

    total_ci = print_ci(
        "TOTAL R",
        total_b
    )

    pf_ci = print_ci(
        "PF",
        pf_b
    )

    prob_total = float(
        np.mean(total_b > 0)
    )

    print(
        f"P(TOTAL R > 0)="
        f"{prob_total:.4f}"
    )

    # --------------------------------------------------------
    # Symbol-level bootstrap
    # --------------------------------------------------------

    (
        sm_mean,
        sm_total,
        sm_pf
    ) = bootstrap_symbol_level(
        ev,
        BOOT,
        rng
    )

    print()
    print("SYMBOL-LEVEL BOOTSTRAP")

    sm_mean_ci = print_ci(
        "MEAN R",
        sm_mean
    )

    sm_total_ci = print_ci(
        "TOTAL R",
        sm_total
    )

    sm_pf_ci = print_ci(
        "PF",
        sm_pf
    )

    sm_prob = float(
        np.mean(sm_total > 0)
    )

    print(
        f"P(TOTAL R > 0)="
        f"{sm_prob:.4f}"
    )

    # --------------------------------------------------------
    # Monte Carlo sequence DD
    # --------------------------------------------------------

    mc = monte_carlo_dd(
        r,
        BOOT,
        rng
    )

    qdd = np.percentile(
        mc,
        [5, 50, 95]
    )

    print()
    print("MONTE-CARLO SEQUENCE DD")

    print(
        f"DD 5/50/95% = "
        f"[{qdd[0]:+.2f}, "
        f"{qdd[1]:+.2f}, "
        f"{qdd[2]:+.2f}] R"
    )

    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    print()
    print("ROBUSTNESS FLAGS")

    print(
        "TRADE_MEAN_CI_POSITIVE=",
        bool(mean_ci[0] > 0)
    )

    print(
        "TRADE_TOTAL_CI_POSITIVE=",
        bool(total_ci[0] > 0)
    )

    print(
        "TRADE_PF_CI_ABOVE_1=",
        bool(pf_ci[0] > 1)
    )

    print(
        "SYMBOL_MEAN_CI_POSITIVE=",
        bool(sm_mean_ci[0] > 0)
    )

    print(
        "SYMBOL_TOTAL_CI_POSITIVE=",
        bool(sm_total_ci[0] > 0)
    )

    print(
        "SYMBOL_PF_CI_ABOVE_1=",
        bool(sm_pf_ci[0] > 1)
    )

    print(
        "TRADE_P_TOTAL_GT_0_GE_95=",
        bool(prob_total >= 0.95)
    )

    print(
        "SYMBOL_P_TOTAL_GT_0_GE_95=",
        bool(sm_prob >= 0.95)
    )

    print()
    print(
        "NOTE: Bootstrap/Monte-Carlo measures "
        "statistical uncertainty only; "
        "it is NOT independent OOS validation."
    )


print()
print("#" * 72)
print("CANDIDATE 11 — STATISTICAL ROBUSTNESS COMPLETE")
print("#" * 72)
