import requests, pandas as pd, numpy as np, time

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

COSTS = [0.003,0.004,0.005]
TPS = [1.5,2.0]
HOLDOUT = 0.20

URL = "https://api.kucoin.com/api/v1/market/candles"


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
        return None,"NO_DATA"

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
        return None,f"TOTAL={len(d)}"

    idx = d.index

    gap = (
        idx.to_series()
        .diff()
        .ne(pd.Timedelta(hours=4))
    )

    group = gap.cumsum()

    blocks = (
        pd.DataFrame({
            "ts":idx,
            "g":group.values
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

        return None,(
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
        return None,f"BLOCK={len(out)}"

    return out,"PASS"


def prepare(d):

    d=d.copy()

    d["mean20"]=d["close"].rolling(20).mean()
    d["std20"]=d["close"].rolling(20).std()

    d["z"]=(
        (d["close"]-d["mean20"]) /
        d["std20"]
    )

    d["range20"]=(
        d["high"].rolling(20).max() -
        d["low"].rolling(20).min()
    )

    d["range_med"]=(
        d["range20"].rolling(20).median()
    )

    d["vol_med"]=(
        d["volume"].rolling(20).median()
    )

    tr=pd.concat([
        d["high"]-d["low"],
        (d["high"]-d["close"].shift()).abs(),
        (d["low"]-d["close"].shift()).abs()
    ],axis=1).max(axis=1)

    d["atr20"]=tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    d["ema200"]=d["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    return d


def events(d,tp,end_limit):

    out=[]

    for i in range(200,len(d)-1):

        p=d.iloc[i-1]
        c=d.iloc[i]

        if not (
            np.isfinite(p["z"]) and
            np.isfinite(c["z"]) and
            np.isfinite(c["atr20"])
        ):
            continue

        signal=(
            c["close"]<c["ema200"]
            and p["z"]>=2
            and c["z"]<2
            and c["range20"]>=c["range_med"]
            and c["volume"]>=c["vol_med"]
        )

        if not signal:
            continue

        ei=i+1

        if ei>=len(d):
            continue

        ep=float(d.iloc[ei]["open"])

        sl=ep+1.25*float(c["atr20"])
        risk=sl-ep
        target=ep-tp*risk

        result=None
        exit_ts=None

        last=min(len(d),ei+31)

        for j in range(ei,last):

            x=d.iloc[j]

            if x["high"]>=sl:
                result=-1.0
                exit_ts=d.index[j]
                break

            if x["low"]<=target:
                result=tp
                exit_ts=d.index[j]
                break

        if result is None:
            result=0.0
            exit_ts=d.index[last-1]

        if exit_ts>end_limit:
            continue

        out.append({
            "ts":d.index[ei],
            "exit":exit_ts,
            "r":result,
            "risk_pct":risk/ep,
            "bars":(exit_ts-d.index[ei])/pd.Timedelta(hours=4)+1
        })

    return pd.DataFrame(out)


def stat(ev,cost=.003):

    if ev.empty:
        return 0,0,0,0,0

    r=(
        ev.sort_values("ts")["r"]
        -cost
    )

    total=float(r.sum())
    exp=float(r.mean())

    win=float(r[r>0].sum())
    loss=float(-r[r<0].sum())

    pf=win/loss if loss else np.inf

    eq=r.cumsum()

    dd=float(
        (eq-eq.cummax()).min()
    )

    return len(r),exp,pf,total,dd


def no_overlap(ev):

    if ev.empty:
        return ev

    x=ev.sort_values(
        ["sym","ts"]
    ).copy()

    keep=[]
    last={}

    for _,r in x.iterrows():

        s=r["sym"]

        if (
            s not in last or
            r["ts"]>=last[s]
        ):
            keep.append(True)
            last[s]=r["exit"]
        else:
            keep.append(False)

    return x.loc[keep].sort_values("ts")


# ============================================================
# DATA
# ============================================================

print("="*72)
print("CANDIDATE 11 — SHORT ONLY")
print("FINAL INTERNAL HOLDOUT VALIDATION")
print("="*72)

DATA={}

for s in SYMS:

    d,msg=fetch(s)

    if d is None:
        print(f"{s:<6} FAIL | {msg}", flush=True)
    else:
        DATA[s]=prepare(d)
        print(
            f"{s:<6} PASS | "
            f"{len(d)} | "
            f"{d.index[0]} -> {d.index[-1]}",
            flush=True
        )

print()
print("="*72)
print("DATA AUDIT")
print("="*72)

if len(DATA)!=len(SYMS):
    raise SystemExit(
        f"ABORT: {len(DATA)}/{len(SYMS)} symbols passed"
    )

common=set(next(iter(DATA.values())).index)

for d in DATA.values():
    common &= set(d.index)

common=pd.DatetimeIndex(sorted(common))

if len(common)!=N:
    raise SystemExit(
        f"ABORT: COMMON={len(common)} expected={N}"
    )

g=common.to_series().diff().dropna()

if not (g==pd.Timedelta(hours=4)).all():
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

cut=int(N*(1-HOLDOUT))

hold=common[cut:]

HSTART=hold[0]
HEND=hold[-1]

print()
print("="*72)
print("HOLDOUT DEFINITION")
print("="*72)
print(f"TRAIN/DEV={cut}")
print(f"HOLDOUT={len(hold)}")
print(f"START={HSTART}")
print(f"END={HEND}")
print("PARAMETERS=FROZEN")
print("="*72)


ALL={}

for tp in TPS:

    parts=[]

    for s,d in DATA.items():

        e=events(
            d,
            tp,
            HEND
        )

        if e.empty:
            continue

        e["sym"]=s

        e=e[
            (e["ts"]>=HSTART) &
            (e["ts"]<=HEND)
        ]

        if not e.empty:
            parts.append(e)

    ALL[tp]=(
        pd.concat(parts,ignore_index=True)
        .sort_values("ts")
        if parts
        else
        pd.DataFrame(
            columns=["ts","exit","r","sym","risk_pct","bars"]
        )
    )


# ============================================================
# RESULTS
# ============================================================

for tp in TPS:

    ev=ALL[tp]

    print()
    print("="*72)
    print(f"TP{tp}R — HOLDOUT")
    print("="*72)

    print(f"EVENTS={len(ev)}")

    for cost in COSTS:

        n,ex,pf,total,dd=stat(
            ev,cost
        )

        print(
            f"COST {cost:.3f} | "
            f"N={n} "
            f"Exp={ex:+.4f} "
            f"PF={pf:.3f} "
            f"Total={total:+.2f}R "
            f"DD={dd:+.2f}R"
        )

    no=no_overlap(ev)

    n,ex,pf,total,dd=stat(
        no,.003
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

    folds=np.array_split(hold,5)

    positive=0
    parts=[]

    for k,b in enumerate(folds,1):

        a=b[0]
        z=b[-1]

        f=ev[
            (ev["ts"]>=a) &
            (ev["ts"]<=z)
        ]

        n,ex,pf,total,dd=stat(
            f,.003
        )

        if total>0:
            positive+=1

        parts.append(f)

        print(
            f"FOLD {k} | "
            f"N={n} "
            f"Exp={ex:+.4f} "
            f"PF={pf:.3f} "
            f"Total={total:+.2f}R "
            f"DD={dd:+.2f}R"
        )

    agg=pd.concat(
        parts,
        ignore_index=True
    )

    n,ex,pf,total,dd=stat(
        agg,.003
    )

    print("-"*72)
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

    full=stat(ev,.003)[3]

    rem=[]

    for s in SYMS:

        t=stat(
            ev[ev["sym"]!=s],
            .003
        )[3]

        rem.append((s,t))

    rem.sort(key=lambda x:x[1])

    print(f"FULL={full:+.2f}R")

    for s,t in rem[:5]:
        print(
            f"REMOVE {s:<6} => {t:+.2f}R"
        )

print()
print("="*72)
print("FINAL INTERNAL HOLDOUT VALIDATION COMPLETE")
print("="*72)


# ============================================================
# REALISTIC COST TEST (percent-of-price fees + funding)
# ============================================================

RT_FEES = [0.0010, 0.0020, 0.0030]   # round-trip fee+slippage, % of price
FUND_8H = [0.0, 0.0001]              # funding paid per 8h by the short (0 = none)


def net_r(ev, rt, fund):
    return (
        ev["r"]
        - rt / ev["risk_pct"]
        - fund * (ev["bars"] / 2.0) / ev["risk_pct"]
    )


def stat_series(r, ts):
    r = pd.Series(r.values, index=ts.values).sort_index()
    if r.empty:
        return 0, 0, 0, 0, 0
    win = float(r[r > 0].sum())
    loss = float(-r[r < 0].sum())
    pf = win / loss if loss else np.inf
    eq = r.cumsum()
    dd = float((eq - eq.cummax()).min())
    return len(r), float(r.mean()), pf, float(r.sum()), dd


def cluster_p(ev, net, B=20000, seed=20261003):
    rng = np.random.default_rng(seed)
    g = (
        pd.DataFrame({"ts": ev["ts"].values, "net": net.values})
        .groupby("ts")["net"].sum().to_numpy(float)
    )
    x = rng.choice(g, (B, len(g)), replace=True).sum(1)
    return len(g), np.quantile(x, [.025, .5, .975]), (x > 0).mean()


print()
print("#"*72)
print("CANDIDATE 11 — REALISTIC COST TEST")
print("#"*72)

for tp in TPS:

    ev = ALL[tp]

    print()
    print("="*72)
    print(f"TP{tp}R")
    print("="*72)
    print(
        f"SL distance % of price: "
        f"median={ev['risk_pct'].median()*100:.2f}% "
        f"min={ev['risk_pct'].min()*100:.2f}% "
        f"max={ev['risk_pct'].max()*100:.2f}%"
    )
    print(f"mean bars held={ev['bars'].mean():.1f}")

    for fund in FUND_8H:
        for rt in RT_FEES:

            nr = net_r(ev, rt, fund)
            n, ex, pf, total, dd = stat_series(nr, ev["ts"])
            cn, ci, p = cluster_p(ev, nr)

            print(
                f"FEE {rt*100:.2f}% FUND/8h {fund*100:.3f}% | "
                f"N={n} Exp={ex:+.4f} PF={pf:.3f} "
                f"Total={total:+.2f}R DD={dd:+.2f}R | "
                f"CLUSTER CI95=[{ci[0]:+.1f},{ci[2]:+.1f}] "
                f"P(>0)={p:.3f}"
            )

print()
print("DONE — Candidate 11 realistic cost test")
