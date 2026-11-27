"""Shared helpers for the new paper paths (MOM-LONG-P, MOM-SHORT-P, MOM-1H). Not run by itself.
Entry idea: momentum breakout/breakdown (close beyond the 30-candle extreme with a body of at least 0.8 ATR).
Stop = 2 ATR from entry (risk must be 1%-12% of price), exit = trailing stop 3 ATR, max 60 candles. Net R uses the 0.2% round-trip cost."""
import numpy as np
import pandas as pd

import brt_trail_short as T

E = T.E
N_BRK, BODY_ATR, STOP_ATR = 30, 0.8, 2.0
PB_WINDOW, PB_ZONE_ATR = 6, 0.25
TF = "4H"


def brk_long(d, i):
    """Broken level (highest high of the previous 30 candles) if candle i is a momentum breakout, else None."""
    if i < 200:
        return None
    a = float(d["atr"].iloc[i])
    if not (np.isfinite(a) and a > 0):
        return None
    o, c = float(d["open"].iloc[i]), float(d["close"].iloc[i])
    if c <= o or (c - o) < BODY_ATR * a:
        return None
    lvl = float(d["high"].iloc[i - N_BRK:i].max())
    return lvl if c > lvl else None


def brk_short(d, i):
    if i < 200:
        return None
    a = float(d["atr"].iloc[i])
    if not (np.isfinite(a) and a > 0):
        return None
    o, c = float(d["open"].iloc[i]), float(d["close"].iloc[i])
    if c >= o or (o - c) < BODY_ATR * a:
        return None
    lvl = float(d["low"].iloc[i - N_BRK:i].min())
    return lvl if c < lvl else None


def setup_both(d, i):
    """Plain momentum (both directions), used by the 1-hour path."""
    if brk_long(d, i) is not None:
        return (1, float(d["low"].iloc[i]))
    if brk_short(d, i) is not None:
        return (-1, float(d["high"].iloc[i]))
    return None


def _pullback_ok(d, k, lvl, a, side):
    lo, hi = float(d["low"].iloc[k]), float(d["high"].iloc[k])
    o, c = float(d["open"].iloc[k]), float(d["close"].iloc[k])
    if side == 1:
        return lo <= lvl + PB_ZONE_ATR * a and c > lvl and c > o
    return hi >= lvl - PB_ZONE_ATR * a and c < lvl and c < o


def setup_pullback(d, i, side):
    """Pullback entry: a breakout candle j happened 1..6 candles ago; candle i is the FIRST candle that comes back to the broken
    level (within 0.25 ATR), closes back beyond it in the trend direction with a candle of that colour; the level was never lost in between."""
    brk = brk_long if side == 1 else brk_short
    for j in range(i - PB_WINDOW, i):
        if j < 200:
            continue
        lvl = brk(d, j)
        if lvl is None:
            continue
        a = float(d["atr"].iloc[j])
        ok = True
        for k in range(j + 1, i):
            c = float(d["close"].iloc[k])
            if (side == 1 and c <= lvl) or (side == -1 and c >= lvl) or _pullback_ok(d, k, lvl, a, side):
                ok = False
                break
        if ok and _pullback_ok(d, i, lvl, a, side):
            return (1, float(d["low"].iloc[i])) if side == 1 else (-1, float(d["high"].iloc[i]))
        return None  # only the earliest breakout candle of the window counts
    return None


def configure(state_file, tag, strategy, setup_fn, kind, tf="4H"):
    global TF
    TF = tf
    E.STATE_FILE = state_file
    T.TAG = tag
    new_state0 = E.new_state

    def new_state_gen():
        st = new_state0()
        st["strategy"] = strategy
        return st

    E.new_state = new_state_gen
    E.setup_at = setup_fn
    orig_handle = E.handle_signal
    orig_close = E.msg_close

    def msg_signal_gen(pos):
        word = {"MOM": ("Momentum Breakout" if pos["side"] == "LONG" else "Momentum Breakdown"),
                "PB": "Momentum Pullback Entry"}[kind]
        return T.msg_signal_trail(pos).replace("Break &amp; Retest (4H)", f"{word} ({TF})")

    def msg_close_gen(c):
        m = orig_close(c)
        return m.replace("four-hour", "one-hour") if TF == "1H" else m

    def handle_signal_gen(state, symbol, d, entry_source, i, now):
        n_out = len(E.OUTBOX)
        before = set(state["open"])
        lo, hi = E.RISK_LO, E.RISK_HI
        E.RISK_LO, E.RISK_HI = 0.0, 10.0
        try:
            res = orig_handle(state, symbol, d, entry_source, i, now)
        finally:
            E.RISK_LO, E.RISK_HI = lo, hi
        if res != "OPENED":
            return res
        pos_id = (set(state["open"]) - before).pop()
        pos = state["open"][pos_id]
        sgn = 1 if pos["side"] == "LONG" else -1
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
            rec["sl"] = entry - sgn * risk
            rec["tp"] = entry + sgn * 2.0 * risk
        pos["risk_pct"] = risk / entry
        del E.OUTBOX[n_out:]
        E.OUTBOX.append(msg_signal_gen(pos))
        E.OUTBOX.append(E.msg_entry(pos))
        return "OPENED"

    E.msg_signal = msg_signal_gen
    E.msg_close = msg_close_gen
    E.handle_signal = handle_signal_gen


def use_one_hour():
    """Switch the engine from 4-hour to 1-hour candles (separate process only)."""
    E.GRANULARITY = 60
    E.BAR = pd.Timedelta(hours=1)
    E.BAR_SECONDS = 3600

    def completed_1h(df, now):
        if df.empty:
            return df
        cutoff = pd.Timestamp(now).floor("1h")
        return df[df["ts"] < cutoff].copy().reset_index(drop=True)

    E.completed_only = completed_1h
