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


def fetch(s):
    url=f"{BASE}/api/v1/market/candles"
    end=int(pd.Timestamp("2026-09-30 00:00:00",tz="UTC").timestamp())
    rows=[]

    for _ in range(6):
        start=end-1400*STEP

        p={
            "symbol":f"{s}-USDT",
            "type":"4hour",
            "startAt":start,
            "endAt":end
        }

        r=requests.get(url,params=p,timeout=30)
        r.raise_for_status()
        j=r.json()

        if j.get("code")!="200000":
            raise RuntimeError(str(j))

        d=j.get("data",[])
        if not d:
            break

        rows.extend(d)

        mn=min(int(x[0]) for x in d)
        end=mn-1
        time.sleep(.12)

        if len(rows)>=N+300:
            break

    if not rows:
        raise RuntimeError("no data")

    df=pd.DataFrame(rows,columns=[
        "ts","open","close","high","low","volume","turnover"
    ])

    df["ts"]=pd.to_datetime(
        df.ts.astype("int64"),unit="s",utc=True
    )

    for c in ["open","close","high","low","volume"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")

    df=df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    if len(df)<N:
        raise RuntimeError(f"received={len(df)}")

    ts=df.ts.astype("int64").to_numpy()//10**9
    good=np.diff(ts)==STEP

    run=0
    st=None

    for i,v in enumerate(good):
        run=run+1 if v else 0
        if run>=N-1:
            st=i-N+2
            break

    if st is None:
        raise RuntimeError(
            f"no contiguous {N}-candle block; received={len(df)}"
        )

    df=df.iloc[st:st+N].copy().reset_index(drop=True)

    if len(df)!=N:
        raise RuntimeError(f"final={len(df)}")

    if not (df.ts.diff().dropna()==pd.Timedelta(hours=4)).all():
        raise RuntimeError("gap after extraction")

    return df


def ind(d):
    x=d.copy()

    m=x.close.rolling(20).mean()
    sd=x.close.rolling(20).std(ddof=0)

    x["z"]=(x.close-m)/sd.replace(0,np.nan)
    x["ema"]=x.close.ewm(span=200,adjust=False).mean()

    x["rng"]=(x.high.rolling(20).max()-
              x.low.rolling(20).min())

    x["rngm"]=x.rng.rolling(20).median()
    x["volm"]=x.volume.rolling(20).median()

    pc=x.close.shift(1)

    tr=pd.concat([
        x.high-x.low,
        (x.high-pc).abs(),
        (x.low-pc).abs()
    ],axis=1).max(axis=1)

    x["atr"]=tr.ewm(alpha=1/20,adjust=False).mean()

    return x


def make_events(d,tp):
    x=ind(d)
    out=[]

    for i in range(20,len(x)-HOLD-1):

        p=x.iloc[i-1]
        c=x.iloc[i]

        ok=(
            c.close<c.ema and
            p.z>=2 and
            c.z<2 and
            c.rng>=c.rngm and
            c.volume>=c.volm
        )

        if not ok:
            continue

        ei=i+1
        entry=x.iloc[ei].close
        atr=x.iloc[ei].atr

        if not np.isfinite(entry) or not np.isfinite(atr) or atr<=0:
            continue

        sl=entry+SL*atr
        target=entry-tp*SL*atr
        last=min(ei+HOLD,len(x)-1)

        res=None
        ex=None

        for j in range(ei,last+1):

            hi=x.iloc[j].high
            lo=x.iloc[j].low

            if hi>=sl:
                res=-1
                ex=j
                break

            if lo<=target:
                res=tp
                ex=j
                break

        if res is None:
            res=(entry-x.iloc[last].close)/(SL*atr)
            res=float(np.clip(res,-1,tp))
            ex=last

        out.append({
            "entry":x.iloc[ei].ts,
            "exit":x.iloc[ex].ts,
            "r":float(res-COST)
        })

    return pd.DataFrame(out)


def metric(e):
    if len(e)==0:
        return dict(N=0,Exp=np.nan,PF=np.nan,Total=0,DD=0)

    r=e.r.to_numpy(float)
    w=r[r>0].sum()
    l=abs(r[r<0].sum())
    pf=w/l if l else np.inf

    eq=np.cumsum(r)
    peak=np.maximum.accumulate(np.r_[0,eq])[1:]
    dd=(eq-peak).min()

    return dict(
        N=len(r),
        Exp=r.mean(),
        PF=pf,
        Total=r.sum(),
        DD=dd
    )


def show(name,m):
    pf=f"{m['PF']:.3f}" if np.isfinite(m["PF"]) else "INF"
    print(
        f"{name:<8} N={m['N']:4d} "
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

    for s in SYMBOLS:
        try:
            d=fetch(s)
            data[s]=d
            print(
                f"{s:<6} {len(d)} "
                f"{d.ts.iloc[0]} -> {d.ts.iloc[-1]}"
            )
        except Exception as e:
            print(f"{s:<6} ERROR: {e}")

    if len(data)!=33:
        print(f"\nABORT: VALID SYMBOLS={len(data)}, EXPECTED=33")
        raise SystemExit(1)

    common=set(data["BTC"].ts)

    for d in data.values():
        common &= set(d.ts)

    common=pd.DatetimeIndex(sorted(common))

    if len(common)!=N:
        print(
            f"\nABORT: COMMON TIMESTAMPS={len(common)}, "
            f"EXPECTED={N}"
        )
        raise SystemExit(1)

    print("\nDATA AUDIT = PASS")
    print(f"VALID SYMBOLS = {len(data)}")
    print(f"COMMON TIMESTAMPS = {len(common)}")
    print(f"START = {common[0]}")
    print(f"END   = {common[-1]}")

    all_events={}

    for tp in TPS:

        parts=[]

        for s,d in data.items():

            d=d[d.ts.isin(common)].copy()
            e=make_events(d,tp)

            if len(e):
                e["symbol"]=s
                parts.append(e)

        e=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(
            columns=["entry","exit","r","symbol"]
        )

        e=e.sort_values("exit").reset_index(drop=True)
        all_events[tp]=e

        print(f"\nTP{tp:g}R TOTAL EVENTS = {len(e)}")

    folds=np.array_split(np.arange(N),5)

    for tp in TPS:

        e=all_events[tp]

        print("\n"+"="*72)
        print(f"TP{tp:g}R — SHORT ONLY")
        print("="*72)

        oos=[]
        positive=0

        for k,idx in enumerate(folds):

            oa=common[idx[0]]
            ob=(
                common[folds[k+1][0]]
                if k<4
                else common[-1]+pd.Timedelta(hours=4)
            )

            ia=common[0]
            ib=oa

            ie=e[
                (e.entry>=ia)&
                (e.exit<ib)
            ].copy()

            oe=e[
                (e.entry>=oa)&
                (e.exit<ob)
            ].copy()

            ie=ie.sort_values("exit")
            oe=oe.sort_values("exit")

            im=metric(ie)
            om=metric(oe)

            print(f"\nFOLD {k+1}")
            print(f"IS  {ia} -> {ib}")
            show("IS",im)

            print(f"OOS {oa} -> {ob}")
            show("OOS",om)

            if om["Exp"]>0:
                positive+=1

            if len(oe):
                oos.append(oe)

        agg=(
            pd.concat(oos,ignore_index=True)
            if oos else
            pd.DataFrame(columns=e.columns)
        )

        am=metric(agg)

        print("\n"+"-"*72)
        print(f"TP{tp:g}R AGGREGATE OOS")
        show("OOS ALL",am)
        print(f"Positive OOS folds = {positive}/5")

        passed=(
            positive==5 and
            am["Exp"]>0 and
            am["PF"]>1 and
            am["Total"]>0
        )

        print(
            "TEMPORAL VALIDATION = "
            +("PASS" if passed else "FAIL")
        )

    print("\n"+"="*72)
    print("RUN COMPLETE")
    print("="*72)


if __name__=="__main__":
    main()
