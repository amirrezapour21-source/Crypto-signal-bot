import time, requests, numpy as np, pandas as pd

BASE="https://api.kucoin.com"

SYMBOLS=[
"BTC","ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT",
"SUI","TRX","NEAR","AAVE","OP","ARB","APT","ATOM","FIL","LTC",
"BCH","ETC","UNI","INJ","SEI","VET","HBAR","ALGO","XLM","ICP",
"WIF","PEPE","FLOKI"
]

N=4380
STEP=14400
HOLD=30
SL=1.25
COST=.003
TPS=[1.5,2.0]


def fetch(symbol):

    url=f"{BASE}/api/v1/market/candles"

    end=int(pd.Timestamp(
        "2026-09-30 00:00:00",
        tz="UTC"
    ).timestamp())

    rows=[]

    # KuCoin pagination: move backward using the
    # actual minimum timestamp returned by each page.
    for _ in range(5):

        start=end-1500*STEP

        params={
            "symbol":f"{symbol}-USDT",
            "type":"4hour",
            "startAt":start,
            "endAt":end
        }

        r=requests.get(
            url,
            params=params,
            timeout=30
        )
        r.raise_for_status()

        j=r.json()

        if j.get("code")!="200000":
            raise RuntimeError(str(j))

        data=j.get("data",[])

        if not data:
            break

        rows.extend(data)

        mn=min(int(x[0]) for x in data)

        end=mn-STEP//1000

        time.sleep(.15)

    if not rows:
        raise RuntimeError("no data")

    df=pd.DataFrame(
        rows,
        columns=[
            "ts","open","close","high",
            "low","volume","turnover"
        ]
    )

    df["ts"]=pd.to_datetime(
        df["ts"].astype("int64"),
        unit="s",
        utc=True
    )

    for c in [
        "open","close","high","low","volume"
    ]:
        df[c]=pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df=(
        df.drop_duplicates("ts")
          .sort_values("ts")
          .reset_index(drop=True)
    )

    if len(df)<N:
        raise RuntimeError(
            f"only {len(df)} candles"
        )

    # IMPORTANT:
    # Do NOT reject individual symbols because of
    # pagination-boundary gaps here.
    # The common-timestamp audit below is authoritative.

    return df.tail(N).reset_index(drop=True)


def indicators(d):

    x=d.copy()

    mean=x.close.rolling(20).mean()
    std=x.close.rolling(20).std(ddof=0)

    x["z"]=(x.close-mean)/std.replace(0,np.nan)

    x["ema"]=x.close.ewm(
        span=200,
        adjust=False
    ).mean()

    x["rng"]=(
        x.high.rolling(20).max()
        -
        x.low.rolling(20).min()
    )

    x["rngm"]=x.rng.rolling(20).median()

    x["volm"]=x.volume.rolling(20).median()

    pc=x.close.shift(1)

    tr=pd.concat([
        x.high-x.low,
        (x.high-pc).abs(),
        (x.low-pc).abs()
    ],axis=1).max(axis=1)

    x["atr"]=tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    return x


def make_events(d,tp):

    x=indicators(d)
    out=[]

    for i in range(
        20,
        len(x)-HOLD-1
    ):

        p=x.iloc[i-1]
        c=x.iloc[i]

        # FROZEN CANDIDATE 11 — SHORT ONLY
        signal=(
            c.close<c.ema
            and p.z>=2
            and c.z<2
            and c.rng>=c.rngm
            and c.volume>=c.volm
        )

        if not signal:
            continue

        ei=i+1

        entry=x.iloc[ei].close
        atr=x.iloc[ei].atr

        if (
            not np.isfinite(entry)
            or not np.isfinite(atr)
            or atr<=0
        ):
            continue

        sl=entry+SL*atr
        tp=entry-tp*SL*atr

        last=min(
            ei+HOLD,
            len(x)-1
        )

        result=None
        exit_i=None

        for j in range(
            ei,
            last+1
        ):

            hi=x.iloc[j].high
            lo=x.iloc[j].low

            # Same-candle SL first
            if hi>=sl:
                result=-1.0
                exit_i=j
                break

            if lo<=tp:
                result=tp
                exit_i=j
                break

        if result is None:

            result=(
                entry-x.iloc[last].close
            )/(SL*atr)

            result=float(
                np.clip(
                    result,
                    -1,
                    tp
                )
            )

            exit_i=last

        out.append({
            "entry":x.iloc[ei].ts,
            "exit":x.iloc[exit_i].ts,
            "r":float(result-COST)
        })

    return pd.DataFrame(out)


def metric(e):

    if len(e)==0:
        return {
            "N":0,
            "Exp":np.nan,
            "PF":np.nan,
            "Total":0.0,
            "DD":0.0
        }

    r=e.r.to_numpy(float)

    wins=r[r>0].sum()
    losses=abs(r[r<0].sum())

    pf=(
        wins/losses
        if losses>0
        else np.inf
    )

    equity=np.cumsum(r)

    peak=np.maximum.accumulate(
        np.r_[0.0,equity]
    )[1:]

    dd=(equity-peak).min()

    return {
        "N":len(r),
        "Exp":r.mean(),
        "PF":pf,
        "Total":r.sum(),
        "DD":dd
    }


def show(name,m):

    pf=(
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


def main():

    print("="*72)
    print("CANDIDATE 11 — SHORT-ONLY — 5-FOLD")
    print("="*72)

    data={}

    for symbol in SYMBOLS:

        try:

            d=fetch(symbol)

            data[symbol]=d

            print(
                f"{symbol:<6} "
                f"{len(d)} candles  "
                f"{d.ts.iloc[0]} -> "
                f"{d.ts.iloc[-1]}"
            )

        except Exception as e:

            print(
                f"{symbol:<6} ERROR: {e}"
            )

    if len(data)!=33:

        print(
            f"\nABORT: VALID SYMBOLS="
            f"{len(data)}, EXPECTED=33"
        )

        raise SystemExit(1)

    # ----------------------------------------------------------
    # COMMON TIMESTAMP AUDIT
    # ----------------------------------------------------------

    common=None

    for d in data.values():

        s=set(d.ts)

        common=(
            s
            if common is None
            else common & s
        )

    common=pd.DatetimeIndex(
        sorted(common)
    )

    if len(common)!=N:

        print(
            f"\nABORT: COMMON TIMESTAMPS="
            f"{len(common)}, EXPECTED={N}"
        )

        raise SystemExit(1)

    # Verify the common window itself is contiguous.
    diff=common.to_series().diff().dropna()

    if not (
        diff==pd.Timedelta(hours=4)
    ).all():

        print(
            "\nABORT: COMMON WINDOW HAS GAP"
        )

        raise SystemExit(1)

    print("\nDATA AUDIT = PASS")
    print(
        f"VALID SYMBOLS = {len(data)}"
    )
    print(
        f"COMMON TIMESTAMPS = {len(common)}"
    )
    print(
        f"START = {common[0]}"
    )
    print(
        f"END   = {common[-1]}"
    )

    # ----------------------------------------------------------
    # EVENTS
    # ----------------------------------------------------------

    all_events={}

    for tp in TPS:

        parts=[]

        for symbol,d in data.items():

            d=d[
                d.ts.isin(common)
            ].copy()

            e=make_events(d,tp)

            if len(e):

                e["symbol"]=symbol
                parts.append(e)

        if parts:

            e=pd.concat(
                parts,
                ignore_index=True
            )

        else:

            e=pd.DataFrame(
                columns=[
                    "entry",
                    "exit",
                    "r",
                    "symbol"
                ]
            )

        e=e.sort_values(
            "exit"
        ).reset_index(drop=True)

        all_events[tp]=e

        print(
            f"\nTP{tp:g}R "
            f"TOTAL EVENTS = {len(e)}"
        )

    # ----------------------------------------------------------
    # 5 FOLD TEMPORAL VALIDATION
    # ----------------------------------------------------------

    folds=np.array_split(
        np.arange(N),
        5
    )

    for tp in TPS:

        e=all_events[tp]

        print("\n"+"="*72)
        print(
            f"TP{tp:g}R — SHORT ONLY"
        )
        print("="*72)

        oos_parts=[]
        positive=0

        for k,idx in enumerate(folds):

            oos_start=common[idx[0]]

            if k<4:

                oos_end=common[
                    folds[k+1][0]
                ]

            else:

                oos_end=(
                    common[-1]
                    +pd.Timedelta(hours=4)
                )

            is_start=common[0]
            is_end=oos_start

            is_events=e[
                (e.entry>=is_start)
                &
                (e.exit<is_end)
            ].copy()

            oos_events=e[
                (e.entry>=oos_start)
                &
                (e.exit<oos_end)
            ].copy()

            is_events=is_events.sort_values(
                "exit"
            )

            oos_events=oos_events.sort_values(
                "exit"
            )

            im=metric(is_events)
            om=metric(oos_events)

            print(
                f"\nFOLD {k+1}"
            )

            print(
                f"IS  {is_start} -> "
                f"{is_end}"
            )

            show("IS",im)

            print(
                f"OOS {oos_start} -> "
                f"{oos_end}"
            )

            show("OOS",om)

            if om["Exp"]>0:
                positive+=1

            if len(oos_events):
                oos_parts.append(
                    oos_events
                )

        if oos_parts:

            agg=pd.concat(
                oos_parts,
                ignore_index=True
            ).sort_values("exit")

        else:

            agg=pd.DataFrame(
                columns=e.columns
            )

        am=metric(agg)

        print("\n"+"-"*72)
        print(
            f"TP{tp:g}R "
            f"AGGREGATE OOS"
        )

        show("OOS ALL",am)

        print(
            f"Positive OOS folds = "
            f"{positive}/5"
        )

        passed=(
            positive==5
            and am["Exp"]>0
            and am["PF"]>1
            and am["Total"]>0
        )

        print(
            "TEMPORAL VALIDATION = "
            +
            (
                "PASS"
                if passed
                else "FAIL"
            )
        )

    print("\n"+"="*72)
    print("RUN COMPLETE")
    print("="*72)


if __name__=="__main__":
    main()
