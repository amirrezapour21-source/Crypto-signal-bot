import time
import requests
import numpy as np
import pandas as pd

# ============================================================
# V4 — CANDIDATE 4 — DONCHIAN VOLATILITY BREAKOUT
# Frozen Spec — no optimization
# ============================================================

SYMS = [
    "ETH","SOL","BNB","XRP","DOGE","ADA","LINK","AVAX","DOT","NEAR",
    "APT","ARB","OP","SUI","INJ","TIA","SEI","FIL","ATOM","LTC",
    "ETC","TRX","ICP","AAVE","UNI","MKR","RUNE","FTM","GRT","ALGO",
    "VET","HBAR","EGLD","XLM","THETA","SAND","MANA","AXS","CHZ"
]

TARGET_DAYS = 730
MIN_DAYS = 180

BREAKOUT = 20
ATR_LEN = 20
VOL_LEN = 20
DAILY_EMA = 20

HOLD = 30
TP_SCENARIOS = [1.0, 1.5, 2.0, 3.0]

FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05

URL = "https://api.kucoin.com/api/v1/market/candles"


def fetch_symbol(sym):
    symbol = sym + "-USDT"
    rows = []
    end_at = int(time.time())

    target = TARGET_DAYS * 6
    seen = set()

    for _ in range(10):
        params = {
            "symbol": symbol,
            "type": "4hour",
            "endAt": end_at
        }

        try:
            r = requests.get(URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception:
            break

        if not data:
            break

        for x in data:
            ts = int(x[0])
            if ts not in seen:
                seen.add(ts)
                rows.append(x)

        oldest = min(int(x[0]) for x in data)

        if len(rows) >= target:
            break

        new_end = oldest - 1
        if new_end >= end_at:
            break

        end_at = new_end
        time.sleep(0.15)

    if len(rows) < 100:
        return None

    # KuCoin format:
    # [time, open, close, high, low, volume, turnover]
    df = pd.DataFrame(
        rows,
        columns=["time","open","close","high","low","volume","turnover"]
    )

    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # FIXED: unit changed from "s" to "ms" for KuCoin timestamp
    df["time"] = pd.to_datetime(
        df["time"], unit="ms", utc=True
    )

    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # Remove current incomplete 4H candle
    now = pd.Timestamp.now(tz="UTC")
    if len(df):
        last_end = df["time"].iloc[-1] + pd.Timedelta(hours=4)
        if now < last_end:
            df = df.iloc[:-1].copy()

    if len(df) < 100:
        return None

    days = (
        df["time"].iloc[-1] - df["time"].iloc[0]
    ).total_seconds() / 86400

    # ATR using True Range
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3], axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(
        ATR_LEN,
        min_periods=ATR_LEN
    ).mean()

    df["vol_ma"] = df["volume"].rolling(
        VOL_LEN,
        min_periods=VOL_LEN
    ).mean()

    return df, days


def daily_bias(df):
    d = df.set_index("time").resample("1D").agg({
        "close": "last"
    }).dropna()

    d["ema"] = d["close"].ewm(
        span=DAILY_EMA,
        adjust=False
    ).mean()

    return d


def build_bias(df):
    d = daily_bias(df)

    bias = {}

    for t, row in d.iterrows():
        bias[t] = (
            "LONG"
            if row["close"] > row["ema"]
            else "SHORT"
        )

    return bias


def get_bias(bias, ts):
    day = pd.Timestamp(ts).floor("D")
    return bias.get(day)


def detect_events(df):
    events = []

    if len(df) <= BREAKOUT + ATR_LEN + VOL_LEN:
        return events

    for i in range(
        max(BREAKOUT, ATR_LEN, VOL_LEN),
        len(df)
    ):
        close = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i])
        vol = float(df["volume"].iloc[i])
        vol_ma = float(df["vol_ma"].iloc[i])

        if not np.isfinite(atr) or atr <= 0:
            continue

        if not np.isfinite(vol_ma) or vol_ma <= 0:
            continue

        prior_high = float(
            df["high"].iloc[i-BREAKOUT:i].max()
        )

        prior_low = float(
            df["low"].iloc[i-BREAKOUT:i].min()
        )

        long_break = close > prior_high
        short_break = close < prior_low

        volume_ok = vol > vol_ma

        if not volume_ok:
            continue

        if long_break:
            events.append({
                "idx": i,
                "time": df["time"].iloc[i],
                "dir": "LONG",
                "entry": close,
                "atr": atr
            })

        elif short_break:
            events.append({
                "idx": i,
                "time": df["time"].iloc[i],
                "dir": "SHORT",
                "entry": close,
                "atr": atr
            })

    return events


def simulate(df, event, tp_mult):
    i = event["idx"]
    entry = event["entry"]
    atr = event["atr"]
    direction = event["dir"]

    if direction == "LONG":
        sl = entry - atr
        tp = entry + atr * tp_mult
    else:
        sl = entry + atr
        tp = entry - atr * tp_mult

    last = i + HOLD

    if last >= len(df):
        return {
            "status": "OPEN_AT_DATASET_END",
            "gross_r": None,
            "net_r": None,
            "ambiguous": False
        }

    cost_pct = (FEE_PCT + SLIPPAGE_PCT) / 100.0
    risk_pct = atr / entry

    if risk_pct <= 0:
        return {
            "status": "INVALID",
            "gross_r": None,
            "net_r": None,
            "ambiguous": False
        }

    cost_r = cost_pct / risk_pct

    for j in range(i + 1, i + HOLD + 1):

        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])

        if direction == "LONG":
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:
            hit_sl = high >= sl
            hit_tp = low <= tp

        if hit_sl and hit_tp:
            return {
                "status": "SL",
                "gross_r": -1.0,
                "net_r": -1.0 - cost_r,
                "ambiguous": True
            }

        if hit_sl:
            return {
                "status": "SL",
                "gross_r": -1.0,
                "net_r": -1.0 - cost_r,
                "ambiguous": False
            }

        if hit_tp:
            return {
                "status": "TP",
                "gross_r": tp_mult,
                "net_r": tp_mult - cost_r,
                "ambiguous": False
            }

    return {
        "status": "TIMEOUT",
        "gross_r": 0.0,
        "net_r": -cost_r,
        "ambiguous": False
    }


def max_drawdown(values):
    if not values:
        return 0.0

    eq = np.cumsum(values)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak

    return float(dd.min())


def main():

    print(
        "SETUP V4 — CANDIDATE 4 — DONCHIAN VOLATILITY BREAKOUT"
    )
    print(
        "Frozen: 4H | breakout=20 | ATR=20 | volume=20 | "
        "daily EMA20 bias | HOLD=30 | Long/Short"
    )

    data = {}

    for sym in SYMS:
        result = fetch_symbol(sym)

        if result is None:
            continue

        df, days = result

        data[sym] = {
            "df": df,
            "days": days
        }

    usable = {
        s: v["df"]
        for s, v in data.items()
        if v["days"] >= MIN_DAYS
    }

    print(
        f"DATA {len(data)} loaded; "
        f"{len(usable)} sufficient"
    )

    if not usable:
        print("NO SUFFICIENT DATA")
        return

    common = None

    for df in usable.values():
        ts = set(df["time"])
        common = ts if common is None else common & ts

    common = sorted(common)

    if not common:
        print("NO COMMON TIMESTAMPS")
        return

    print(
        f"COMMON_TIMESTAMPS {len(common)} "
        f"{common[0]} -> {common[-1]}"
    )

    biases = {
        s: build_bias(df)
        for s, df in usable.items()
    }

    all_events = []

    last_allowed = common[-1] - pd.Timedelta(
        hours=4 * HOLD
    )

    for sym, df in usable.items():

        events = detect_events(df)

        for e in events:

            if e["time"] not in common:
                continue

            if e["time"] > last_allowed:
                continue

            bias = get_bias(
                biases[sym],
                e["time"]
            )

            if bias != e["dir"]:
                continue

            all_events.append({
                "sym": sym,
                **e
            })

    all_events.sort(
        key=lambda x: (x["time"], x["sym"])
    )

    print(
        f"DETECTED_EVENTS {len(all_events)}"
    )

    for tp_mult in TP_SCENARIOS:

        results = []

        for e in all_events:

            df = usable[e["sym"]]

            r = simulate(
                df,
                e,
                tp_mult
            )

            if r["status"] == "INVALID":
                continue

            results.append({
                "sym": e["sym"],
                **r
            })

        traded = [
            x for x in results
            if x["status"] in
            ("TP", "SL", "TIMEOUT")
        ]

        tp = sum(
            x["status"] == "TP"
            for x in traded
        )

        sl = sum(
            x["status"] == "SL"
            for x in traded
        )

        timeout = sum(
            x["status"] == "TIMEOUT"
            for x in traded
        )

        ambiguous = sum(
            x["ambiguous"]
            for x in traded
        )

        gross = [
            x["gross_r"]
            for x in traded
        ]

        net = [
            x["net_r"]
            for x in traded
        ]

        gross_total = sum(gross)
        net_total = sum(net)

        gross_exp = (
            gross_total / len(gross)
            if gross else 0.0
        )

        net_exp = (
            net_total / len(net)
            if net else 0.0
        )

        gross_profit = sum(
            x for x in gross if x > 0
        )

        gross_loss = -sum(
            x for x in gross if x < 0
        )

        net_profit = sum(
            x for x in net if x > 0
        )

        net_loss = -sum(
            x for x in net if x < 0
        )

        gross_pf = (
            gross_profit / gross_loss
            if gross_loss > 0 else float("inf")
        )

        net_pf = (
            net_profit / net_loss
            if net_loss > 0 else float("inf")
        )

        wr = (
            tp / len(traded) * 100
            if traded else 0.0
        )

        dd = max_drawdown(net)

        print()
        print(
            f"TP {tp_mult:g} R | "
            f"TRADED {len(traded)} "
            f"TP {tp} SL {sl} TIMEOUT {timeout} "
            f"OPEN_AT_END 0 AMBIGUOUS {ambiguous}"
        )

        print(
            f"GROSS_EXP {gross_exp:.4f} "
            f"NET_EXP {net_exp:.4f} "
            f"GROSS_TOTAL {gross_total:.2f} "
            f"NET_TOTAL {net_total:.2f} "
            f"PF {gross_pf:.3f} "
            f"NET_PF {net_pf:.3f} "
            f"MAXDD {dd:.2f}"
        )

        print(f"WR {wr:.2f}%")

        print("PER_SYMBOL")

        by_symbol = {}

        for x in traded:
            sym = x["sym"]

            if sym not in by_symbol:
                by_symbol[sym] = {}

            status = x["status"]
            by_symbol[sym][status] = (
                by_symbol[sym].get(status, 0) + 1
            )

        for sym in sorted(by_symbol):
            print(sym, by_symbol[sym])

    print()
    print("DONE — send complete Test RUN output for review")


if __name__ == "__main__":
    main()
