"""BRT-S MOM-T6 (paper). Same entries as MOM-LONG / MOM-SHORT (4h momentum breakout/breakdown, body >= 0.8 ATR, entry next open),
but with a FIXED take-profit: the trade closes as soon as price reaches +6% from entry. Stop = 2 ATR (fixed, no trailing), max 60 candles.
Usage: python brt_mom_t6.py LONG   or   python brt_mom_t6.py SHORT. Own state files. Other bots are not touched."""
import html
import sys

import pandas as pd

import brt_mom_gen as G

SIDE = (sys.argv[1] if len(sys.argv) > 1 else "LONG").upper()
if SIDE not in ("LONG", "SHORT"):
    raise SystemExit("side must be LONG or SHORT")
E, T = G.E, G.T
TP_PCT = 0.06
HOLD = 60
sgn_main = 1 if SIDE == "LONG" else -1
G.configure(f"brt_mom_{SIDE.lower()}_t6_state.json", f"MOM-{SIDE}-T6", f"brt_mom_{SIDE.lower()}_t6_v1",
            (lambda d, i: (1, float(d["low"].iloc[i])) if G.brk_long(d, i) is not None else None) if SIDE == "LONG"
            else (lambda d, i: (-1, float(d["high"].iloc[i])) if G.brk_short(d, i) is not None else None), "MOM")

_sig0, _entry0 = E.msg_signal, E.msg_entry


def _fix(txt):
    return (txt.replace("🎯 <b>Exit:</b> no fixed target. The stop follows price at 3 ATR and moves only in profit direction.",
                        "🎯 <b>Exit:</b> take-profit at +6% from entry. Fixed stop, no trailing.")
               .replace("Trailing 3 ATR, max 60 candles", "Take-profit +6%, max 60 candles")
               .replace("+ trailing stop", "+ 6% take-profit"))


E.msg_signal = lambda p: _fix(_sig0(p))
_trail0 = T.msg_signal_trail
T.msg_signal_trail = lambda p: _fix(_trail0(p))   # the signal message is built from this function
E.msg_entry = lambda p: _fix(_entry0(p))


def msg_close_t6(c):
    sym = html.escape(E.tg_symbol(c["symbol"]))
    entry = float(c["entry"])
    sgn = 1.0 if c["side"] == "LONG" else -1.0
    pct = sgn * (float(c["exit_price"]) - entry) / entry * 100
    r = float(c["gross_r"])
    if c["exit_reason"] == "TP":
        head, tail = f"🏆 <b>TAKE PROFIT +6% ({T.TAG}) — {sym}/USDT</b>", "🟢 <b>TRADE CLOSED</b>"
    elif c["exit_reason"] == "TIMEOUT":
        head, tail = f"⏱️ <b>TIMEOUT ({T.TAG}) — {sym}/USDT</b>", "after 60 four-hour candles"
    else:
        head, tail = f"🛑 <b>STOP LOSS HIT ({T.TAG}) — {sym}/USDT</b>", "🔴 <b>TRADE CLOSED</b>"
    return (f"{head}\n\n{E.side_icon(c['side'])} {c['side']}\n📍 Entry: <code>{E.fmt_price(entry)}</code>\n"
            f"🚪 Exit: <code>{E.fmt_price(c['exit_price'])}</code>\n\n📊 <b>Result: {pct:+.2f}% ({r:+.2f}R)</b>\n\n{tail}")


E.msg_close = msg_close_t6


def monitor_bar_t6(state, symbol, row):
    ts = row["ts"]
    for pos_id, pos in [(k, p) for k, p in state["open"].items() if p.get("symbol") == symbol]:
        if ts < pd.Timestamp(pos["entry_ts"]) or pos.get("last_bar_ts") == E.iso(ts):
            continue
        pos["last_bar_ts"] = E.iso(ts)
        pos["bars"] = int(pos.get("bars", 0)) + 1
        entry, risk, sl = float(pos["entry"]), float(pos["risk"]), float(pos["sl"])
        s = 1 if pos.get("side") == "LONG" else -1
        tp = entry * (1 + s * TP_PCT)
        o, hi, lo, close = (float(row[k]) for k in ("open", "high", "low", "close"))
        px = reason = None
        if s == 1:
            if pos["bars"] > 1 and o <= sl: px, reason = o, "SL"
            elif lo <= sl: px, reason = sl, "SL"            # stop first (conservative)
            elif pos["bars"] > 1 and o >= tp: px, reason = o, "TP"
            elif hi >= tp: px, reason = tp, "TP"
        else:
            if pos["bars"] > 1 and o >= sl: px, reason = o, "SL"
            elif hi >= sl: px, reason = sl, "SL"
            elif pos["bars"] > 1 and o <= tp: px, reason = o, "TP"
            elif lo <= tp: px, reason = tp, "TP"
        if reason is None and pos["bars"] >= HOLD:
            px, reason = close, "TIMEOUT"
        if reason is None:
            continue
        gross_r = s * (px - entry) / risk
        rp = risk / entry
        bars = int(pos["bars"])
        s3 = E.S3_FEE_PER_SIDE * (1.0 + px / entry) + E.S3_FUNDING_PER_8H * bars * 0.5
        closed = dict(pos)
        closed.update({"status": "CLOSED", "exit_ts": E.iso(ts), "exit_price": px, "exit_reason": reason,
                       "gross_r": gross_r, "net_r": gross_r - E.FEE_RT / rp, "net_r_s3": gross_r - s3 / rp,
                       "stop_gap": bool(reason == "SL" and pos["ba
