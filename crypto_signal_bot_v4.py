import time, requests, pandas as pd, numpy as np

URL="https://api.kucoin.com/api/v1/market/candles"

SYMS="""BTC ETH SOL BNB XRP DOGE ADA LINK AVAX DOT SUI TRX NEAR AAVE OP ARB APT ATOM FIL LTC BCH ETC UNI INJ SEI VET HBAR ALGO XLM ICP WIF PEPE FLOKI""".split()
SYMS=[x+"-USDT" for x in SYMS]

END=pd.Timestamp("2026-09-30 00:00",tz="UTC")
STEP=pd.Timedelta(hours=4)

N=4380
COST=.003
SL_ATR=1.25
HOLD=30
TPS=[1,1.5,2,3]


# ============================================================
# FETCH EXACTLY N CANDLES ENDING AT FIXED END
# ============================================================

def fetch(s):

    rows=[]
    end=END

    while len(rows)<N:

        start=end-STEP*1399

        ok=False

        for attempt in range(4):

            try:
                r=requests.get(
                    URL,
                    params={
                        "symbol":s,
                        "type":"4hour",
                        "startAt":int(start.timestamp()),
                        "endAt":int(end.timestamp())
                    },
                    timeout=20
                ).json()

                if r.get("code")!="200000":
                    raise Exception(r.get("msg"))

                a=r.get("data",[])

                if not a:
                    raise Exception("empty response")

                rows.extend(a)

                # Move backward from the oldest returned candle.
                mn=min(int(x[0]) for x in a)

                end=pd.Timestamp(
                    mn,
                    unit="s",
                    tz="UTC"
                )-STEP

                ok=True
                break

            except Exception as e:

                if attempt==3:
                    raise Exception(f"{s}: {e}")

                time.sleep(1)

        if not ok:
            raise Exception(f"{s}: fetch failed")

        time.sleep(.08)

    d=pd.DataFrame(
        rows,
        columns=[
            "ts","open","close","high",
            "low","vol","turn"
        ]
    )

    d.ts=pd.to_datetime(
        d.ts.astype(int),
        unit="s",
        utc=True
    )

    for c in [
        "open","close","high","low","vol"
    ]:
        d[c]=pd.to_numeric(d[c],errors="coerce")

    d=(
        d.drop_duplicates("ts")
         .sort_values("ts")
         .set_index("ts")
    )

    # Exact final window
    d=d.loc[d.index<=END].tail(N)

    if len(d)!=N:
        raise Exception(
            f"{s}: final candles={len(d)}"
        )

    return d


# ============================================================
# INDICATORS — CANDIDATE 11 FROZEN
# ============================================================

def prep(d):

    c,h,l,v=d.close,d.high,d.low,d.vol

    m=c.rolling(20).mean()
    sd=c.rolling(20).std()

    d["z"]=(c-m)/sd.replace(0,np.nan)

    d["ema"]=c.ewm(
        span=200,
        adjust=False,
        min_periods=200
    ).mean()

    d["rng"]=
        h.rolling(20).max()-l.rolling(20).min()

    d["rm"]=d["rng"].rolling(20).median()
    d["vm"]=v.rolling(20).median()

    pc=c.shift()

    tr=pd.concat([
        h-l,
        (h-pc).abs(),
        (l-pc).abs()
    ],axis=1).max(axis=1)

    d["atr"]=tr.ewm(
        alpha=1/20,
        adjust=False,
        min_periods=20
    ).mean()

    return d


# ============================================================
# SIGNALS
# ============================================================

def events(d,s):

    out=[]

    for i in range(1,len(d)):

        r=d.iloc[i]
        p=d.iloc[i-1]

        if pd.isna(
            r[["z","ema","rng","rm","vm","atr"]]
        ).any() or pd.isna(p.z):
            continue

        if (
            r.close>r.ema and
            p.z<=-2 and
            r.z>-2 and
            r.rng>=r.rm and
            r.vol>=r.vm
        ):
            out.append(
                (s,d.index[i],i,"LONG",
                 r.close,r.atr)
            )

        if (
            r.close<r.ema and
            p.z>=2 and
            r.z<2 and
            r.rng>=r.rm and
            r.vol>=r.vm
        ):
            out.append(
                (s,d.index[i],i,"SHORT",
                 r.close,r.atr)
            )

    return out


# ============================================================
# TRADE
# ============================================================

def trade(e,d,tp):

    s,t,i,side,entry,atr=e

    risk=atr*SL_ATR

    if side=="LONG":
        sl=entry-risk
        target=entry+risk*tp
    else:
        sl=entry+risk
        target=entry-risk*tp

    last=min(
        i+HOLD,
        len(d)-1
    )

    for j in range(i+1,last+1):

        h=d.high.iloc[j]
        l=d.low.iloc[j]

        # SL FIRST
        if side=="LONG":

            if l<=sl:
                return -1-COST

            if h>=target:
                return tp-COST

        else:

            if h>=sl:
                return -1-COST

            if l<=target:
                return tp-COST

    return -COST


# ============================================================
# REPORT
# ============================================================

def report(name,x):

    if not x:
        print(name,"N=0")
        return

    x=np.array(x)

    gp=x[x>0].sum()
    gl=-x[x<0].sum()

    pf=gp/gl if gl else 999

    print(
        f"{name}: "
        f"N={len(x)} "
        f"WR={(x>0).mean():.4f} "
        f"Exp={x.mean():.4f} "
        f"Total={x.sum():.2f}R "
        f"PF={pf:.3f}"
    )


def concentration(x):

    if not x:
        print("N=0")
        return

    q=(
        pd.DataFrame(x,columns=["symbol","r"])
        .groupby("symbol").r.sum()
        .sort_values(ascending=False)
    )

    total=q.sum()

    print(
        f"Portfolio={total:.2f}R "
        f"-Top1={total-q.head(1).sum():.2f}R "
        f"-Top3={total-q.head(3).sum():.2f}R "
        f"-Top5={total-q.head(5).sum():.2f}R"
    )

    print(
        "Top:",
        ", ".join(
            f"{k}={v:.2f}R"
            for k,v in q.head(5).items()
        )
    )


# ============================================================
# MAIN
# ============================================================

print("="*60)
print("CANDIDATE 11 — DIRECTION ROBUSTNESS V3")
print("FIXED END / BACKWARD EXACT WINDOW")
print("="*60)

data={}

for s in SYMS:

    try:

        d=fetch(s)

        data[s]=prep(d)

        print(
            s,
            len(d),
            d.index[0],
            "->",
            d.index[-1]
        )

    except Exception as e:

        print("FAIL",e)


if len(data)!=len(SYMS):

    raise SystemExit(
        "ABORTED: incomplete symbol universe"
    )


# ============================================================
# COMMON TIMESTAMPS
# ============================================================

common=set(data[SYMS[0]].index)

for s in SYMS[1:]:

    common &= set(data[s].index)

common=sorted(common)

print()
print("VALID SYMBOLS =",len(data))
print("COMMON TIMESTAMPS =",len(common))

if len(common)<N:

    raise SystemExit(
        f"ABORTED: common timestamps={len(common)}"
    )

common=pd.DatetimeIndex(common[-N:])

for s in SYMS:

    data[s]=data[s].loc[common]

print(
    "FINAL WINDOW =",
    common[0],
    "->",
    common[-1]
)

print("DATA AUDIT = PASS")


# ============================================================
# EVENTS
# ============================================================

E=[]

for s,d in data.items():
    E+=events(d,s)

print()
print("="*60)
print("EVENT AUDIT")
print("="*60)

print("TOTAL =",len(E))
print("LONG  =",sum(e[3]=="LONG" for e in E))
print("SHORT =",sum(e[3]=="SHORT" for e in E))


# ============================================================
# DIRECTION TEST
# ============================================================

for tp in TPS:

    print()
    print("="*60)
    print(f"TP{tp} DIRECTION")
    print("="*60)

    L=[]
    S=[]

    for e in E:

        r=trade(
            e,
            data[e[0]],
            tp
        )

        if e[3]=="LONG":
            L.append((e[0],r))
        else:
            S.append((e[0],r))

    report(
        "LONG ",
        [r for _,r in L]
    )

    report(
        "SHORT",
        [r for _,r in S]
    )

    report(
        "BOTH ",
        [r for _,r in L+S]
    )

    print("\nLONG CONCENTRATION")
    concentration(L)

    print("\nSHORT CONCENTRATION")
    concentration(S)


print()
print("="*60)
print("DIRECTION ROBUSTNESS V3 COMPLETED")
print("="*60)
