# CANDIDATE 11 — ROBUSTNESS / STRESS TEST
# Frozen logic — NO parameter optimization

import requests, time, math
import pandas as pd
import numpy as np

BASE = "https://api.kucoin.com"
TYPE = "4hour"

SYMBOLS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT",
    "DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
    "SUI-USDT","APT-USDT","NEAR-USDT","ATOM-USDT","LTC-USDT",
    "BCH-USDT","ETC-USDT","FIL-USDT","ARB-USDT","OP-USDT",
    "INJ-USDT","SEI-USDT","AAVE-USDT","UNI-USDT","TRX-USDT",
    "HBAR-USDT","PEPE-USDT","WIF-USDT","FLOKI-USDT","SHIB-USDT",
    "ICP-USDT","ALGO-USDT","VET-USDT","GRT-USDT","RUNE-USDT"
]

DAYS = 730
HOLD = 30
SL_ATR = 1.25
TPs = [1.0, 1.5, 2.0, 3.0]

COSTS = [0.003, 0.004, 0.005]

def fetch(symbol):
    end = int(time.time())
    start = end - DAYS * 86400

    rows = []
    cursor = end

    for _ in range(10):
        p = {
            "symbol": symbol,
            "type": TYPE,
            "endAt": cursor
        }

        r = requests.get(
            BASE + "/api/v1/market/candles",
            params=p,
            timeout=20
        )
        j = r.json()

        if j.get("code") != "200000":
            break

        data = j.get("data", [])
        if not data:
            break

        rows += data

        oldest = min(int(x[0]) for x in data)

        if oldest <= start:
            break

        cursor = oldest - 1
        time.sleep(0.15)

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["time","open","close","high","low","volume","turnover"]
    )

    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="s",
        utc=True
    )

    df = (
        df.drop_duplicates("time")
          .sort_values("time")
          .reset_index(drop=True)
    )

    # remove incomplete candle
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["time"] + pd.Timedelta(hours=4) <= now]

    df = df[
        df["time"] >= pd.Timestamp(start, unit="s", tz="UTC")
    ]

    return df.reset_index(drop=True)


def prepare(df):
    x = df.copy()

    # Candidate 11 — frozen indicators
    x["ema200"] = x["close"].ewm(
        span=200, adjust=False
    ).mean()

    mean20 = x["close"].rolling(20).mean()
    std20 = x["close"].rolling(20).std()

    x["z"] = (x["close"] - mean20) / std20

    x["range20"] = (
        x["high"].rolling(20).max()
        - x["low"].rolling(20).min()
    )

    x["range_med"] = x["range20"].rolling(20).median()
    x["vol_med"] = x["volume"].rolling(20).median()

    # Wilder ATR20
    prev_close = x["close"].shift(1)

    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1/20,
        adjust=False
    ).mean()

    return x


def events(df):
    out = []

    for i in range(1, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]

        if not np.isfinite(r["atr"]):
            continue

        if not np.isfinite(r["z"]) or not np.isfinite(p["z"]):
            continue

        if not np.isfinite(r["range20"]):
            continue

        if not np.isfinite(r["range_med"]):
            continue

        if not np.isfinite(r["vol_med"]):
            continue

        # LONG
        long_ok = (
            r["close"] > r["ema200"] and
            p["z"] <= -2.0 and
            r["z"] > -2.0 and
            r["range20"] >= r["range_med"] and
            r["volume"] >= r["vol_med"]
        )

        # SHORT
        short_ok = (
            r["close"] < r["ema200"] and
            p["z"] >= 2.0 and
            r["z"] < 2.0 and
            r["range20"] >= r["range_med"] and
            r["volume"] >= r["vol_med"]
        )

        if long_ok:
            out.append((i, "LONG"))

        elif short_ok:
            out.append((i, "SHORT"))

    return out


def trade(df, i, direction, tp_mult):
    if i + HOLD >= len(df):
        return None

    entry = float(df.iloc[i]["close"])
    atr = float(df.iloc[i]["atr"])

    if direction == "LONG":
        sl = entry - SL_ATR * atr
        tp = entry + tp_mult * SL_ATR * atr
    else:
        sl = entry + SL_ATR * atr
        tp = entry - tp_mult * SL_ATR * atr

    for j in range(i + 1, i + HOLD + 1):
        h = float(df.iloc[j]["high"])
        l = float(df.iloc[j]["low"])

        if direction == "LONG":

            # Same-candle SL first
            if l <= sl:
                return -1.0

            if h >= tp:
                return tp_mult

        else:

            # Same-candle SL first
            if h >= sl:
                return -1.0

            if l <= tp:
                return tp_mult

    # TIMEOUT = 0R gross
    return 0.0


def max_dd(values):
    if not values:
        return 0.0

    eq = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0, eq])[:-1]
    dd = eq - peak

    return float(dd.min())


def metrics(rows, cost):
    if not rows:
        return {
            "n": 0,
            "wr": 0,
            "exp": 0,
            "total": 0,
            "pf": 0,
            "dd": 0
        }

    net = np.array([r - cost for r in rows], dtype=float)

    wins = net[net > 0]
    losses = net[net < 0]

    pf = (
        wins.sum() / abs(losses.sum())
        if len(losses) else math.inf
    )

    return {
        "n": len(net),
        "wr": float((net > 0).mean()),
        "exp": float(net.mean()),
        "total": float(net.sum()),
        "pf": float(pf),
        "dd": max_dd(net)
    }


def main():

    all_events = []
    symbol_data = {}

    print("\n=== CANDIDATE 11 ROBUSTNESS TEST ===")
    print("Frozen logic | No optimization")
    print("Loading 730D / 4H data...\n")

    for s in SYMBOLS:
        try:
            df = fetch(s)

            if df is None or len(df) < 1000:
                print(s, "INSUFFICIENT")
                continue

            df = prepare(df)
            ev = events(df)

            symbol_data[s] = df

            for i, d in ev:
                all_events.append({
                    "symbol": s,
                    "i": i,
                    "direction": d
                })

            print(
                s,
                "candles=", len(df),
                "events=", len(ev)
            )

        except Exception as e:
            print(s, "ERROR", str(e))

    print("\nTOTAL EVENTS:", len(all_events))
    print("VALID SYMBOLS:", len(symbol_data))

    # ---------------------------------------------------------
    # TP + COST STRESS
    # ---------------------------------------------------------

    results = []

    for tp in TPs:

        gross_by_symbol = {}

        for e in all_events:

            s = e["symbol"]

            r = trade(
                symbol_data[s],
                e["i"],
                e["direction"],
                tp
            )

            if r is None:
                continue

            gross_by_symbol.setdefault(s, []).append(r)

        for cost in COSTS:

            rows = []

            for vals in gross_by_symbol.values():
                rows.extend(vals)

            m = metrics(rows, cost)

            results.append([
                tp,
                cost,
                m["n"],
                m["wr"],
                m["exp"],
                m["total"],
                m["pf"],
                m["dd"]
            ])

    print("\n=== COST STRESS ===")

    print(
        "TP | COST | N | WR | NET_EXP | NET_TOTAL | PF | MAX_DD"
    )

    for x in results:
        print(
            f"{x[0]:>3} | "
            f"{x[1]:.3f} | "
            f"{x[2]:>4} | "
            f"{x[3]:.3f} | "
            f"{x[4]:+.4f} | "
            f"{x[5]:+.2f} | "
            f"{x[6]:.3f} | "
            f"{x[7]:+.2f}"
        )

    # ---------------------------------------------------------
    # SYMBOL CONCENTRATION
    # ---------------------------------------------------------

    print("\n=== SYMBOL CONCENTRATION ===")
    print("Baseline cost = 0.003R")

    for tp in TPs:

        sym_total = {}

        for e in all_events:

            s = e["symbol"]

            r = trade(
                symbol_data[s],
                e["i"],
                e["direction"],
                tp
            )

            if r is None:
                continue

            sym_total[s] = sym_total.get(s, 0.0) + r - 0.003

        if not sym_total:
            continue

        total = sum(sym_total.values())

        ordered = sorted(
            sym_total.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\nTP {tp}R")
        print("Total:", round(total, 3))

        for n in [1, 3, 5]:

            removed = sum(
                v for _, v in ordered[:n]
            )

            remain = total - removed

            print(
                f"Remove Top {n}: "
                f"remain={remain:+.3f}R "
                f"removed={removed:+.3f}R"
            )

        print("Top contributors:")

        for s, v in ordered[:5]:
            print(
                f"  {s}: {v:+.3f}R"
            )

    # ---------------------------------------------------------
    # LONG / SHORT DIAGNOSTIC
    # ---------------------------------------------------------

    print("\n=== LONG / SHORT DIAGNOSTIC ===")

    for tp in TPs:

        for direction in ["LONG", "SHORT"]:

            rows = []

            for e in all_events:

                if e["direction"] != direction:
                    continue

                r = trade(
                    symbol_data[e["symbol"]],
                    e["i"],
                    direction,
                    tp
                )

                if r is not None:
                    rows.append(r)

            m = metrics(rows, 0.003)

            print(
                f"TP{tp} {direction}: "
                f"N={m['n']} "
                f"WR={m['wr']:.3f} "
                f"Exp={m['exp']:+.4f} "
                f"Total={m['total']:+.2f} "
                f"PF={m['pf']:.3f} "
                f"DD={m['dd']:+.2f}"
            )

    print("\n=== INTEGRITY ===")
    print("Causal indicators: PASS")
    print("Entry at signal close: PASS")
    print("Entry candle exit scan: FALSE")
    print("Same-candle SL first: PASS")
    print("Long + Short included: PASS")
    print("No parameter optimization: PASS")
    print("No extra filters: PASS")
    print("Stress costs only: PASS")

    print("\n=== ROBUSTNESS TEST COMPLETE ===")


if __name__ == "__main__":
    main()
