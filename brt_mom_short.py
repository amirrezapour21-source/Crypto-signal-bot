"""BRT-S MOM-SHORT (paper, hypothesis test). SHORT momentum breakdown, no BTC filter, no retest:
a 4h candle closes below the lowest low of the previous 30 candles with a red body of at least 0.8 ATR.
Entry = next candle open, initial stop = 2 ATR above entry (risk must be 1%-12% of price), exit = trailing stop 3 ATR, max 60 candles.
Own state file brt_mom_short_state.json, Telegram label MOM-SHORT. Other bots are not touched."""
import numpy as np
import pandas as pd

import brt_trail_short as T

E = T.E
E.STATE_FILE = "brt_mom_short_state.json"
T.TAG = "MOM-SHORT"
N_BRK, BODY_ATR, STOP_ATR = 30, 0.8, 2.0

_new_state = E.new_state


def new_state_mom():
    st = _new_state()
    st["strategy"] = "brt_mom_short_trail_v1"
    return st


E.new_state = new_state_mom


def setup_at_mom(d, i):
    if i < 200:
        return None
    a = float(d["atr"].iloc[i])
    if not (np.isfinite(a) and a > 0):
        return None
    o, c, hi = float(d["open"].iloc[i]), float(d["close"].iloc[i]), float(d["high"].iloc[i])
    if c >= o or (o - c) < BODY_ATR * a:
        return None
    if c >= float(d["low"].iloc[i - N_BRK:i].min()):
        return None
    return (-1, hi)


E.setup_at = setup_at_mom

_orig_handle = E.handle_signal


def handle_signal_mom(state, symbol, d, entry_source, i, now):
    n_out = len(E.OUTBOX)
    before = set(state["open"])
    lo, hi = E.RISK_LO, E.RISK_HI
    E.RISK_LO, E.RISK_HI = 0.0, 10.0
    try:
        res = _orig_handle(state, symbol, d, entry_source, i, now)
    finally:
        E.RISK_LO, E.RISK_HI = lo, hi
    if res != "OPENED":
        return res
    pos_id = (set(state["open"]) - before).pop()
    pos = state["open"][pos_id]
    entry, atr = float(pos["entry"]), float(pos["atr"])
    risk = STOP_ATR * atr
    if not (E.RISK_LO <= risk / entry <= E.RISK_HI):
        del state["open"][pos_id]
        state["signals"].pop()
        del E.OUTBOX[n_out:]
        E.add_missed(state, symbol, d.iloc[i]["ts"], "RISK_OUT_OF_RANGE", {"risk_pct": round(risk / entry, 5)})
        return "MISSED"
    for rec in (pos, state["signals"][-1]):
        rec["risk"] = risk
        rec["sl"] = entry + risk
        rec["tp"] = entry - 2.0 * risk
    pos["risk_pct"] = risk / entry
    del E.OUTBOX[n_out:]
    E.OUTBOX.append(msg_signal_mom(pos))
    E.OUTBOX.append(E.msg_entry(pos))
    return "OPENED"


def msg_signal_mom(pos):
    return T.msg_signal_trail(pos).replace("Break &amp; Retest (4H)", "Momentum Breakdown (4H)")


E.msg_signal = msg_signal_mom
E.handle_signal = handle_signal_mom

if __name__ == "__main__":
    E.main()
