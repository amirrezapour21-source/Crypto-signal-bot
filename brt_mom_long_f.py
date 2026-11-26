"""BRT-S MOM-LONG-F (paper). Same entry idea as MOM-LONG, plus 4 pre-registered filters (all must pass):
1) EXTENSION: close of the breakout candle is at most 3.0 ATR above the lowest low of the previous 30 candles (do not chase a finished rally).
2) DEMAND: breakout candle volume is at least 1.5 x the average volume of the previous 20 candles (real buying, not a thin move).
3) SUPPLY: no older swing high (candles 31..180 back) sits less than 2 ATR above the close (no resistance right overhead).
4) NEWS: entry candle must not start from 4h before to 8h after a scheduled macro event listed in NEWS_UTC.
Exit, stop, risk limits are identical to MOM-LONG. Own state file and label. Other bots are not touched."""
import numpy as np
import pandas as pd

import brt_mom_long as M

E = M.E
T = M.T
E.STATE_FILE = "brt_mom_long_f_state.json"
T.TAG = "MOM-LONG-F"
EXT_ATR, VOL_MULT, VOL_N, SUP_ATR = 3.0, 1.5, 20, 2.0
NEWS_BEFORE, NEWS_AFTER = pd.Timedelta(hours=4), pd.Timedelta(hours=8)
# Scheduled US macro events (UTC). Sources: bls.gov schedule, federalreserve.gov calendar.
NEWS_UTC = [
    "2026-12-04T13:30:00+00:00",  # US jobs report (NFP)
    "2026-12-09T19:00:00+00:00",  # FOMC decision
    "2026-12-10T13:30:00+00:00",  # US CPI
]

_new_state = E.new_state


def new_state_f():
    st = _new_state()
    st["strategy"] = "brt_mom_long_f_trail_v1"
    return st


E.new_state = new_state_f


def in_news_window(entry_ts):
    t = pd.Timestamp(entry_ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    for s in NEWS_UTC:
        ev = pd.Timestamp(s)
        if ev - NEWS_BEFORE <= t <= ev + NEWS_AFTER:
            return True
    return False


def setup_at_f(d, i):
    base = M.setup_at_mom(d, i)
    if base is None:
        return None
    if "volume" not in d.columns:
        return None
    a = float(d["atr"].iloc[i])
    c = float(d["close"].iloc[i])
    # 1) extension
    low30 = float(d["low"].iloc[i - 30:i].min())
    if (c - low30) > EXT_ATR * a:
        return None
    # 2) demand
    v = float(d["volume"].iloc[i])
    vavg = float(d["volume"].iloc[i - VOL_N:i].mean())
    if not (np.isfinite(v) and np.isfinite(vavg) and vavg > 0 and v >= VOL_MULT * vavg):
        return None
    # 3) supply
    h_old = float(d["high"].iloc[i - 180:i - 30].max())
    if c < h_old and (h_old - c) < SUP_ATR * a:
        return None
    # 4) news
    if in_news_window(pd.Timestamp(d["ts"].iloc[i]) + E.BAR):
        return None
    return base


E.setup_at = setup_at_f

if __name__ == "__main__":
    E.main()
