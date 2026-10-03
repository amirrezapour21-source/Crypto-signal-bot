# ============================================================
# CANDIDATE 11 — FORWARD PAPER TRADING
# FROZEN: 4H | SHORT ONLY | SL=1.25 ATR | TP=2R | HOLD=30
# ============================================================

import os
import json
import time
import requests
import pandas as pd
import numpy as np


BASE = "https://api-futures.kucoin.com"
STATE_FILE = "candidate11_paper_state.json"

INTERVAL = 240
SL_ATR = 1.25
TP_R = 2.0
HOLD = 30

SYMS = [
    "XBTUSDTM",
    "ETHUSDTM",
    "SOLUSDTM",
    "BNBUSDTM",
    "XRPUSDTM",
    "DOGEUSDTM",
    "ADAUSDTM",
    "LINKUSDTM",
    "AVAXUSDTM",
    "DOTUSDTM",
    "SUIUSDTM",
    "TRXUSDTM",
    "NEARUSDTM",
    "AAVEUSDTM",
    "OPUSDTM",
    "ARBUSDTM",
    "APTUSDTM",
    "ATOMUSDTM",
    "FILUSDTM",
    "LTCUSDTM",
    "BCHUSDTM",
    "ETCUSDTM",
    "UNIUSDTM",
    "INJUSDTM",
    "SEIUSDTM",
    "VETUSDTM",
    "HBARUSDTM",
    "ALGOUSDTM",
    "XLMUSDTM",
    "ICPUSDTM",
    "WIFUSDTM",
    "PEPEUSDTM",
    "FLOKIUSDTM",
]


# ============================================================
# STATE
# ============================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "open": {},
            "closed": [],
            "signals": [],
            "last_scan": None
        }

    try:
        with open(STATE_FILE, "r") as f:
            s = json.load(f)

        for k in ["open", "closed", "signals"]:
            if k not in s:
                s[k] = []

        return s

    except Exception:
        return {
            "open": {},
            "closed": [],
            "signals": [],
            "last_scan": None
        }


def save_state(state):
    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)

    os.replace(tmp, STATE_FILE)


# ============================================================
# API
# ============================================================

def get_klines(symbol, n=260):
    now = int(time.time())
    start = now - n * INTERVAL * 60

    r = requests.get(
        f"{BASE}/api/v1/kline/query",
        params={
            "symbol": symbol,
            "granularity": INTERVAL,
            "from": start * 1000,
            "to": now * 1000
        },
        timeout=20
    )

    r.raise_for_status()

    raw = r.json().get("data", [])

    if not raw:
        return pd.DataFrame()

    d = pd.DataFrame(
        raw,
        columns=[
            "ts",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover"
        ]
    )

    d["ts"] = pd.to_numeric(
        d["ts"],
        errors="coerce"
    )

    for c in [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "turnover"
    ]:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    d["ts"] = pd.to_datetime(
        d["ts"],
        unit="ms",
        utc=True
    )

    d = (
        d.dropna()
         .drop_duplicates("ts")
         .sort_values("ts")
         .reset_index(drop=True)
    )

    return d


# ============================================================
# INDICATORS
# ============================================================

def prepare(d):
    d = d.copy()

    prev_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    # FROZEN ATR
    d["atr"] = tr.ewm(
        alpha=1 / 20,
        adjust=False
    ).mean()

    # FROZEN EMA
    d["ema200"] = d["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # FROZEN Z-SCORE
    mean20 = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std()

    d["z"] = (
        d["close"] - mean20
    ) / std20

    # FROZEN RANGE
    d["range20"] = (
        d["high"].rolling(20).max()
        -
        d["low"].rolling(20).min()
    )

    d["range_med"] = (
        d["range20"].rolling(20).median()
    )

    # FROZEN VOLUME
    d["vol_med"] = (
        d["volume"].rolling(20).median()
    )

    return d


# ============================================================
# TIME
# ============================================================

def completed_data(d):
    now = pd.Timestamp.now(tz="UTC")

    if d.empty:
        return d

    return d[
        d["ts"] + pd.Timedelta(hours=4) <= now
    ].copy().reset_index(drop=True)


# ============================================================
# SIGNAL
# ============================================================

def detect_signal(d):
    if len(d) < 220:
        return None

    p = d.iloc[-2]
    c = d.iloc[-1]

    vals = [
        c["atr"],
        c["ema200"],
        c["z"],
        c["range20"],
        c["range_med"],
        c["vol_med"]
    ]

    if any(pd.isna(x) for x in vals):
        return None

    valid = (
        c["close"] < c["ema200"]
        and
        p["z"] >= 2
        and
        c["z"] < 2
        and
        c["range20"] >= c["range_med"]
        and
        c["volume"] >= c["vol_med"]
    )

    if not valid:
        return None

    signal_ts = c["ts"]

    entry_ts = (
        signal_ts +
        pd.Timedelta(hours=4)
    )

    return {
        "signal_ts": signal_ts.isoformat(),
        "entry_ts": entry_ts.isoformat(),
        "atr": float(c["atr"]),
        "signal_close": float(c["close"])
    }


# ============================================================
# EXISTING SIGNAL CHECK
# ============================================================

def signal_exists(state, symbol, signal_ts):
    for x in state["signals"]:
        if (
            x.get("symbol") == symbol
            and
            x.get("signal_ts") == signal_ts
        ):
            return True

    for x in state["closed"]:
        if (
            x.get("symbol") == symbol
            and
            x.get("signal_ts") == signal_ts
        ):
            return True

    for x in state["open"].values():
        if (
            x.get("symbol") == symbol
            and
            x.get("signal_ts") == signal_ts
        ):
            return True

    return False


# ============================================================
# OPEN POSITION
# ============================================================

def open_position(state, symbol, signal, entry_row):
    entry = float(entry_row["open"])
    atr = float(signal["atr"])

    risk = SL_ATR * atr

    if risk <= 0:
        return

    state["open"][symbol] = {
        "symbol": symbol,
        "signal_ts": signal["signal_ts"],
        "entry_ts": entry_row["ts"].isoformat(),
        "entry": entry,
        "atr": atr,
        "risk": risk,
        "sl": entry + risk,
        "tp": entry - TP_R * risk,
        "bars": 0
    }

    state["signals"].append({
        "symbol": symbol,
        "signal_ts": signal["signal_ts"],
        "entry_ts": entry_row["ts"].isoformat()
    })

    print(
        "ENTRY",
        symbol,
        "ENTRY=", round(entry, 8),
        "SL=", round(entry + risk, 8),
        "TP=", round(entry - TP_R * risk, 8)
    )


# ============================================================
# MONITOR
# ============================================================

def monitor_position(state, symbol, d):
    if symbol not in state["open"]:
        return

    p = state["open"][symbol]

    entry_ts = pd.Timestamp(
        p["entry_ts"]
    )

    rows = d[
        d["ts"] > entry_ts
    ].copy()

    if rows.empty:
        return

    for _, row in rows.iterrows():

        p["bars"] += 1

        high = float(row["high"])
        low = float(row["low"])

        sl_hit = high >= p["sl"]
        tp_hit = low <= p["tp"]

        # FROZEN: SL FIRST
        if sl_hit:
            exit_price = p["sl"]
            gross_r = -1.0
            reason = "SL"

        elif tp_hit:
            exit_price = p["tp"]
            gross_r = TP_R
            reason = "TP"

        elif p["bars"] >= HOLD:
            exit_price = float(row["close"])

            gross_r = (
                p["entry"] - exit_price
            ) / p["risk"]

            reason = "TIMEOUT"

        else:
            continue

        state["closed"].append({
            **p,
            "exit_ts": row["ts"].isoformat(),
            "exit": exit_price,
            "reason": reason,
            "gross_r": gross_r
        })

        del state["open"][symbol]

        print(
            "CLOSED",
            symbol,
            reason,
            "R=",
            round(gross_r, 4)
        )

        break


# ============================================================
# REPORT
# ============================================================

def report(state):
    trades = state["closed"]

    if not trades:
        print()
        print("CLOSED=0")
        print("OPEN=", len(state["open"]))
        print("SIGNALS=", len(state["signals"]))
        return

    r = np.array(
        [
            float(x["gross_r"])
            for x in trades
        ],
        dtype=float
    )

    wins = r[r > 0]
    losses = r[r < 0]

    pf = (
        wins.sum() / abs(losses.sum())
        if len(losses)
        else float("inf")
    )

    equity = np.cumsum(r)

    drawdown = (
        equity -
        np.maximum.accumulate(equity)
    ).min()

    print()
    print("=" * 60)
    print("CANDIDATE 11 — PAPER TRADING")
    print("=" * 60)
    print("CLOSED =", len(r))
    print(
        "WINRATE =",
        round(float((r > 0).mean()), 4)
    )
    print(
        "EXP =",
        round(float(r.mean()), 4)
    )
    print(
        "PF =",
        round(float(pf), 3)
    )
    print(
        "TOTAL_R =",
        round(float(r.sum()), 3)
    )
    print(
        "MAX_DD =",
        round(float(drawdown), 3)
    )
    print("OPEN =", len(state["open"]))
    print("SIGNALS =", len(state["signals"]))
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CANDIDATE 11 — FORWARD PAPER TRADING")
    print("4H | SHORT ONLY | SL 1.25 ATR | TP 2R | HOLD 30")
    print("=" * 60)

    state = load_state()

    for symbol in SYMS:

        try:

            raw = get_klines(symbol)

            if raw.empty:
                print("NO_DATA", symbol)
                continue

            # Monitor existing position
            monitor_position(
                state,
                symbol,
                raw
            )

            closed = completed_data(raw)

            if closed.empty:
                continue

            # New signal
            if symbol not in state["open"]:

                signal = detect_signal(
                    closed
                )

                if signal is not None:

                    if not signal_exists(
                        state,
                        symbol,
                        signal["signal_ts"]
                    ):

                        entry_ts = pd.Timestamp(
                            signal["entry_ts"]
                        )

                        current_ts = pd.Timestamp.now(
                            tz="UTC"
                        )

                        # Signal must be processed
                        # during its actual next candle.
                        if (
                            entry_ts <= current_ts
                            and
                            current_ts <
                            entry_ts +
                            pd.Timedelta(hours=4)
                        ):

                            current = raw[
                                raw["ts"] == entry_ts
                            ]

                            if not current.empty:

                                open_position(
                                    state,
                                    symbol,
                                    signal,
                                    current.iloc[0]
                                )

        except Exception as e:

            print(
                "ERROR",
                symbol,
                repr(e)
            )

    state["last_scan"] = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    save_state(state)

    report(state)

    print()
    print("DONE — Candidate 11 Paper Trading")


if __name__ == "__main__":
    main()
