"""BRT-S TRAIL LONG (paper, hypothesis test). LONG break+retest entries of the BRT-S engine, taken ONLY while
BTC 4h close is above its EMA50 (bullish regime). Exit = trailing stop 3 ATR, max 60 bars (same as brt_trail_short.py).
Own state file brt_trail_long_state.json; Telegram label TRAIL-LONG. Other bots are not touched.
Backtest basis (weak): LONG + BTC>EMA50 + trailing = n=1661, net -0.023R, PF 0.97, halves -0.105/+0.059."""
import numpy as np
import pandas as pd

import brt_trail_short as T

S = T.S
E = T.E
E.STATE_FILE = "brt_trail_long_state.json"
T.TAG = "TRAIL-LONG"

_trail_new_state = E.new_state


def new_state_long():
    st = _trail_new_state()
    st["strategy"] = "brt_long_regime_trail_v1"
    return st


E.new_state = new_state_long

_orig_setup_at = S._orig_setup_at
_REG = {"map": None}


def regime_map():
    if _REG["map"] is not None:
        return _REG["map"]
    m = {}
    try:
        now = E.utc_now()
        raw = E.fetch_raw("XBTUSDTM", now)
        d = E.completed_only(raw, now)
        ema = d["close"].astype(float).ewm(span=50, adjust=False).mean()
        ok = (d["close"].astype(float) > ema).to_numpy()
        for ts, v, k in zip(d["ts"], ok, range(len(d))):
            if k >= 60:
                m[pd.Timestamp(ts)] = bool(v)
        print(f"REGIME_MAP = OK bars={len(m)} last_bullish={ok[-1]}")
    except Exception as e:
        print(f"REGIME_MAP = FAILED ({str(e)[:150]}) -> no LONG signals this run")
    _REG["map"] = m
    return m


def setup_at_long_regime(d, i):
    r = _orig_setup_at(d, i)
    if r is None or r[0] != 1:
        return None
    if regime_map().get(pd.Timestamp(d.iloc[i]["ts"])) is not True:
        return None
    return r


E.setup_at = setup_at_long_regime

if __name__ == "__main__":
    E.main()
