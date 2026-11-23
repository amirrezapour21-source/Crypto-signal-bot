"""BRT-S TRAIL (paper). Same SHORT entries as the live BRT-S bot, but NO fixed 2R target:
the stop trails 3 ATR behind the best price (max 60 bars). Own state file, own Telegram label.
Backtest basis: lvar_exit_TRAIL_result.txt (SHORT net avgR +0.122, PF 1.18, both halves positive).
LONG is off. The live bot (brt_short_only.py) is not touched."""
import html

import numpy as np
import pandas as pd

import brt_short_only as S

E = S.E
E.STATE_FILE = "brt_trail_state.json"
TRAIL_ATR = 3.0
TRAIL_HOLD = 60
TAG = "TRAIL"

_new_state = E.new_state


def new_state_trail():
    st = _new_state()
    st["strategy"] = "brt_short_trail_v1"
    return st


E.new_state = new_state_trail


def msg_signal_trail(pos):
    return "\n".join([
        f"📊 <b>{html.escape(E.tg_symbol(pos['symbol']))}/USDT — SIGNAL ({TAG})</b>", "",
        f"{E.side_icon(pos['side'])} <b>Direction:</b> {pos['side']}",
        "📈 <b>Strategy:</b> Break &amp; Retest (4H) + trailing stop", "",
        f"📍 <b>Entry:</b> <code>{E.fmt_price(pos['entry'])}</code>",
        f"🛑 <b>Initial Stop:</b> <code>{E.fmt_price(pos['sl'])}</code>", "",
        "🎯 <b>Exit:</b> no fixed target. The stop follows price at 3 ATR and moves only in profit direction.",
        "⚠️ <b>Risk Level:</b> PAPER TEST", "", "🏦 <b>Exchange:</b> KuCoin Futures",
    ])


def msg_entry_trail(pos):
    return (
        f"➕ <b>ENTRY EXECUTED ({TAG}) — {html.escape(E.tg_symbol(pos['symbol']))}/USDT</b>\n\n"
        f"📍 Entry: <code>{E.fmt_price(pos['entry'])}</code>\n"
        f"{E.side_icon(pos['side'])} <b>{pos['side']} ACTIVE</b>\n\n"
        f"🛑 Initial SL: <code>{E.fmt_price(pos['sl'])}</code>\n"
        "🎯 Trailing 3 ATR, max 60 candles"
    )


def msg_close_trail(c):
    sym = html.escape(E.tg_symbol(c["symbol"]))
    entry = float(c["entry"])
    sgn = 1.0 if c["side"] == "LONG" else -1.0
    pct = sgn * (float(c["exit_price"]) - entry) / entry * 100
    r = float(c["gross_r"])
    if c["exit_reason"] == "TIMEOUT":
        head = f"⏱️ <b>TIMEOUT ({TAG}) — {sym}/USDT</b>"
        tail = "after 60 four-hour candles"
    elif r >= 0:
        head = f"🏆 <b>TRAILING STOP, PROFIT ({TAG}) — {sym}/USDT</b>"
        tail = "🟢 <b>TRADE CLOSED</b>"
    else:
        head = f"🛑 <b>STOP LOSS HIT ({TAG}) — {sym}/USDT</b>"
        tail = "🔴 <b>TRADE CLOSED</b>"
    return (
        f"{head}\n\n{E.side_icon(c['side'])} {c['side']}\n"
        f"📍 Entry: <code>{E.fmt_price(entry)}</code>\n"
        f"🚪 Exit: <code>{E.fmt_price(c['exit_price'])}</code>\n\n"
        f"📊 <b>Result: {pct:+.2f}% ({r:+.2f}R)</b>\n\n{tail}"
    )


E.msg_signal = msg_signal_trail
E.msg_entry = msg_entry_trail
E.msg_close = msg_close_trail


def monitor_bar_trail(state, symbol, row):
    ts = row["ts"]
    for pos_id, pos in [(k, p) for k, p in state["open"].items() if p.get("symbol") == symbol]:
        if ts < pd.Timestamp(pos["entry_ts"]):
            continue
        if pos.get("last_bar_ts") == E.iso(ts):
            continue
        pos["last_bar_ts"] = E.iso(ts)
        pos["bars"] = int(pos.get("bars", 0)) + 1
        pos.setdefault("sl0", pos["sl"])
        entry, risk = float(pos["entry"]), float(pos["risk"])
        side_n = 1 if pos.get("side") == "LONG" else -1
        pos.setdefault("best", entry)
        o, hi, lo, close = (float(row[k]) for k in ("open", "high", "low", "close"))
        sl = float(pos["sl"])
        exit_price = reason = None
        # stop first (gap = open), same order as the backtest
        if side_n == 1:
            if pos["bars"] > 1 and o <= sl:
                exit_price = o
            elif lo <= sl:
                exit_price = sl
        else:
            if pos["bars"] > 1 and o >= sl:
                exit_price = o
            elif hi >= sl:
                exit_price = sl
        if exit_price is not None:
            reason = "SL"
        else:
            pos["best"] = max(float(pos["best"]), hi) if side_n == 1 else min(float(pos["best"]), lo)
            atr = float(row["atr"]) if "atr" in row else float("nan")
            if np.isfinite(atr) and atr > 0:
                new_sl = pos["best"] - side_n * TRAIL_ATR * atr
                pos["sl"] = max(sl, new_sl) if side_n == 1 else min(sl, new_sl)
            if pos["bars"] >= TRAIL_HOLD:
                exit_price, reason = close, "TIMEOUT"
        if reason is None:
            continue
        gross_r = side_n * (exit_price - entry) / risk
        risk_pct = risk / entry
        net_r = gross_r - E.FEE_RT / risk_pct
        bars_held = int(pos["bars"])
        s3 = E.S3_FEE_PER_SIDE * (1.0 + exit_price / entry) + E.S3_FUNDING_PER_8H * bars_held * 0.5
        closed = dict(pos)
        closed.update({
            "status": "CLOSED", "exit_ts": E.iso(ts), "exit_price": exit_price,
            "exit_reason": reason, "gross_r": gross_r, "net_r": net_r,
            "net_r_s3": gross_r - s3 / risk_pct,
            "stop_gap": bool(reason == "SL" and pos["bars"] > 1 and side_n * (o - sl) <= 0),
            "bars_held": bars_held, "fee_assumption_rt": E.FEE_RT,
        })
        state["closed"].append(closed)
        E.OUTBOX.append(E.msg_close(closed))
        if len(state["closed"]) > E.MAX_CLOSED:
            del state["closed"][:-E.MAX_CLOSED]
        del state["open"][pos_id]
        return reason
    return None


E.monitor_bar = monitor_bar_trail

if __name__ == "__main__":
    E.main()
