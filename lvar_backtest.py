"""
LVAR historical backtest (read-only research). Reuses the REAL detect_setup() of crypto_signal_bot.py with simulated time,
so the backtest cannot drift from the live rules. No parameter is tuned here; the output is a measurement only.
Writes lvar_backtest_result.txt and lvar_backtest_trades.csv.
Known biases (stated in the report): the universe is TODAY's liquid list (survivorship / look-ahead), entry = open of the
candle after the signal, flat fees only, no gap/slippage model.
"""
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import crypto_signal_bot as M

DAYS = int(os.getenv("BT_DAYS", "180"))
FEE_ROUND_TRIP = 0.002            # 0.1% taker per side
FEE_STRESS = 0.004
WARMUP_DAYS = 3
TIMEOUT_BARS = M.TIMEOUT_DAYS * 96
SIM = {"now": None, "pair": None, "i": None, "df15": None, "df4": None}


def fetch_history(pair: str, ctype: str, start: int, end: int) -> pd.DataFrame:
    step = {"15min": 900, "4hour": 14400}[ctype]
    rows, cur_end = [], end
    while cur_end > start:
        cur_start = max(start, cur_end - 1400 * step)
        payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/candles",
                                  {"symbol": pair, "type": ctype, "startAt": cur_start, "endAt": cur_end})
        if isinstance(payload, dict) and payload.get("code") == "200000" and isinstance(payload.get("data"), list):
            for it in payload["data"]:
                try:
                    rows.append({"time": int(it[0]), "open": float(it[1]), "close": float(it[2]),
                                 "high": float(it[3]), "low": float(it[4]), "volume": float(it[5])})
                except (TypeError, ValueError, IndexError):
                    continue
        cur_end = cur_start
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame(columns=["time", "open", "close", "high", "low", "volume"])
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df = df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.volume >= 0)]
    return df.reset_index(drop=True)


# ---- simulated market access used by the real detect_setup() ----
def sim_candles(pair, candle_type, limit, start_at=None):
    if candle_type == "15min":
        i = SIM["i"]
        return SIM["df15"].iloc[max(0, i - int(limit) + 1): i + 1].reset_index(drop=True)
    if candle_type == "4hour":
        d = SIM["df4"]
        now = int(SIM["now"].timestamp())
        d = d[d.time + 14400 <= now - 2]
        return d.tail(int(limit)).reset_index(drop=True) if len(d) else None
    return None


def sim_live_price(pair):
    return float(SIM["df15"].open.iloc[SIM["i"] + 1])     # price right after the signal candle closed


def simulate_trade(df15: pd.DataFrame, i: int, sig: dict):
    """Replays the bot's monitor rules. Returns trade dict or None (never filled / no data)."""
    d, entry, sl0, tp1, tp2 = sig["direction"], sig["entry"], sig["sl"], sig["tp1"], sig["tp2"]
    n = len(df15)
    e = i + 1
    risk = abs(entry - sl0)
    status, active_sl, result, exit_bar = "OPEN", sl0, None, None
    for k in range(e, n):
        row = df15.iloc[k]
        hi, lo = float(row.high), float(row.low)
        if k == e:                       # entry candle: only its CLOSE is certainly after the entry (same rule as the bot)
            hi = lo = float(row.close)
        sl_hit = lo <= active_sl if d == "LONG" else hi >= active_sl
        tp2_hit = hi >= tp2 if d == "LONG" else lo <= tp2
        tp1_hit = hi >= tp1 if d == "LONG" else lo <= tp1
        if sl_hit:
            result, exit_bar = ("BE" if status == "TP1_HIT" else "SL"), k
            break
        if tp2_hit:
            result, exit_bar = "TP2", k
            break
        if status == "OPEN" and tp1_hit:
            status, active_sl = "TP1_HIT", entry
        if k - e >= TIMEOUT_BARS:
            result, exit_bar = "TIMEOUT", k
            break
    if result is None:
        return None                                   # still open at the end of the data: excluded
    if result == "SL":
        r = -1.0
    elif result == "BE":
        r = 0.0
    elif result == "TP2":
        r = abs(tp2 - entry) / risk
    else:
        cl = float(df15.close.iloc[exit_bar])
        r = ((cl - entry) if d == "LONG" else (entry - cl)) / risk
    return {"result": result, "r_gross": r, "risk_pct": risk / entry, "bars": exit_bar - e,
            "entry": entry, "exit_time": int(df15.time.iloc[exit_bar]) + 900}


def run_pair(pair: str, symbol: str, rank: int, df15: pd.DataFrame, df4: pd.DataFrame) -> list:
    if len(df15) < 400 or len(df4) < 60:
        return []
    SIM.update(pair=pair, df15=df15, df4=df4)
    vol = df15["volume"]
    avg20 = vol.shift(1).rolling(M.VOLUME_AVG_BARS).mean()
    ok = (vol >= avg20 * M.VOLUME_CONFIRM_MULT) & (avg20 > 0)        # same gate as detect_setup (cheap pre-filter)
    t0 = int(df15.time.iloc[0]) + WARMUP_DAYS * 86400
    busy = {"LONG": -1, "SHORT": -1}
    out = []
    for i in np.flatnonzero(ok.to_numpy()):
        if i + 2 >= len(df15) or int(df15.time.iloc[i]) < t0:
            continue
        now = datetime.fromtimestamp(int(df15.time.iloc[i]) + 900 + 60, timezone.utc)
        if not (M.SESSION_START_UTC <= now.hour < M.SESSION_END_UTC):
            continue
        SIM.update(now=now, i=int(i))
        M.utc_now = lambda: SIM["now"]
        sig = M.detect_setup(pair, symbol)
        if not sig or i <= busy[sig["direction"]]:
            continue
        tr = simulate_trade(df15, int(i), sig)
        if tr is None:
            continue
        busy[sig["direction"]] = int(i) + 1 + tr["bars"]
        out.append({"pair": pair, "rank": rank, "mode": sig["mode"], "direction": sig["direction"],
                    "signal_time": int(df15.time.iloc[i]) + 900, **tr})
    return out


def apply_run_cap(trades: list) -> list:
    """MAX_SIGNALS_PER_RUN per scan, coins taken in turnover order (as in main())."""
    by_t = {}
    for t in trades:
        by_t.setdefault(t["signal_time"], []).append(t)
    kept = []
    for _, group in by_t.items():
        kept += sorted(group, key=lambda x: x["rank"])[: M.MAX_SIGNALS_PER_RUN]
    return sorted(kept, key=lambda x: x["signal_time"])


def stats(rs: list) -> dict:
    if not rs:
        return {"n": 0}
    a = np.array(rs)
    wins, losses = a[a > 0].sum(), -a[a < 0].sum()
    cum = np.cumsum(a)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return {"n": len(a), "win%": round(100 * float((a > 0).mean()), 1), "avgR": round(float(a.mean()), 3),
            "totalR": round(float(a.sum()), 2), "PF": round(float(wins / losses), 2) if losses > 0 else float("inf"),
            "maxDD_R": round(dd, 2)}


def fmt(label, st):
    if st["n"] == 0:
        return f"{label:34s} n=0"
    return f"{label:34s} n={st['n']:4d}  win%={st['win%']:5.1f}  avgR={st['avgR']:+.3f}  totalR={st['totalR']:+8.2f}  PF={st['PF']}  maxDD={st['maxDD_R']}"


def report(trades: list, meta: dict) -> str:
    for t in trades:
        t["r_net"] = t["r_gross"] - FEE_ROUND_TRIP / t["risk_pct"]
        t["r_stress"] = t["r_gross"] - FEE_STRESS / t["risk_pct"]
    L = ["LVAR BACKTEST (measurement only; no parameter was tuned on this data)",
         f"period: {meta['start']} -> {meta['end']} ({meta['days']} days), coins with data: {meta['coins']}",
         "Biases: universe = TODAY's liquid list (survivorship/look-ahead); entry = open of the candle after the signal;",
         "flat fees (0.2% round trip net, 0.4% stress); no gaps/slippage; SL assumed filled exactly at the stop.", ""]
    if not trades:
        return "\n".join(L + ["NO TRADES"])
    def block(title, rows):
        L.append(title)
        for key in ("r_gross", "r_net", "r_stress"):
            L.append(fmt(f"  {key}", stats([t[key] for t in rows])))
        L.append("")
    block("ALL", trades)
    for m in ("REVERSAL", "TREND"):
        for d in ("LONG", "SHORT"):
            rows = [t for t in trades if t["mode"] == m and t["direction"] == d]
            L.append(fmt(f"{m} {d} (r_net)", stats([t["r_net"] for t in rows])))
    L.append("")
    for m in ("REVERSAL", "TREND"):
        rows = [t for t in trades if t["mode"] == m]
        block(f"MODE {m}", rows) if rows else L.append(f"MODE {m}: n=0\n")
    mid = trades[len(trades) // 2]["signal_time"]
    for m in ("REVERSAL", "TREND"):
        a = [t["r_net"] for t in trades if t["mode"] == m and t["signal_time"] < mid]
        b = [t["r_net"] for t in trades if t["mode"] == m and t["signal_time"] >= mid]
        L.append(fmt(f"{m} first half (r_net)", stats(a)))
        L.append(fmt(f"{m} second half (r_net)", stats(b)))
    L.append("")
    months = {}
    for t in trades:
        months.setdefault(datetime.fromtimestamp(t["signal_time"], timezone.utc).strftime("%Y-%m"), []).append(t["r_net"])
    for k in sorted(months):
        L.append(fmt(f"month {k} (r_net)", stats(months[k])))
    L.append("")
    byc = {}
    for t in trades:
        byc[t["pair"]] = byc.get(t["pair"], 0.0) + t["r_net"]
    tot = sum(byc.values())
    top = sorted(byc.items(), key=lambda kv: -kv[1])[:3]
    L.append("top 3 coins by net R: " + ", ".join(f"{k} {v:+.1f}" for k, v in top) + f" | total {tot:+.1f}")
    res = {}
    for t in trades:
        res[t["result"]] = res.get(t["result"], 0) + 1
    L.append("exits: " + ", ".join(f"{k}={v}" for k, v in sorted(res.items())))
    L.append(f"median risk pct: {100 * float(np.median([t['risk_pct'] for t in trades])):.2f}")
    days = {datetime.fromtimestamp(t["signal_time"], timezone.utc).date() for t in trades}
    L.append(f"signal days: {len(days)} of {meta['days']}; trades/day avg {len(trades) / max(1, meta['days']):.2f}")
    return "\n".join(L)


# =====================================================================================================================
# STUDY 2 (new hypothesis, fixed in advance, NOT tuned): "VALUE-AREA RE-ENTRY ROTATION" (the classic 80% rule)
#   price CLOSES outside the previous day's value area, then two consecutive 15m closes back INSIDE it
#   -> trade the rotation to the opposite edge of the value area.
#   SHORT (from above VAH): bar i-2 closed above VAH, bars i-1 and i closed inside [VAL, VAH]; target = VAL.
#   LONG  (from below VAL): mirrored; target = VAH.
#   Stop beyond the excursion extreme (highest high / lowest low of the last 8 bars) +/- 0.4 ATR14.
#   Filters: RR >= 1.5, stop distance 0.5%..3%, session 07-20 UTC. Variant B adds the 4H trend gate (EMA20/EMA50 as in the bot).
# =====================================================================================================================
VA_LOOKBACK = 8
VA_MIN_RR = 1.5
VA_MAX_RISK = 0.03


def day_levels(df15: pd.DataFrame) -> dict:
    """UTC-day start -> (vah, val, poc) from the PREVIOUS completed day (same completeness rules as the bot)."""
    out = {}
    t = df15["time"].to_numpy()
    first_day = int(t[0]) // 86400 * 86400 + 86400
    last_day = int(t[-1]) // 86400 * 86400
    for day in range(first_day + 86400, last_day + 1, 86400):
        prev = df15[(df15.time >= day - 86400) & (df15.time < day)]
        if len(prev) < M.MIN_PREV_DAY_15M_BARS:
            continue
        if int(prev.time.min()) > day - 86400 + 3600 or int(prev.time.max()) < day - 3600 - 900:
            continue
        vp = M.calculate_volume_profile(prev)
        if vp["vah"] > vp["poc"] > vp["val"] > 0:
            out[day] = (vp["vah"], vp["val"], vp["poc"])
    return out


def h4_trend_series(df4: pd.DataFrame):
    f, sl_ = M.ema(df4["close"], M.H4_EMA_FAST), M.ema(df4["close"], M.H4_EMA_SLOW)
    return df4["time"].to_numpy() + 14400, f.to_numpy(), sl_.to_numpy(), df4["close"].to_numpy()


def h4_state(ser, t: int):
    closes_at, f, s_, c = ser
    k = int(np.searchsorted(closes_at, t - 2, side="right")) - 1
    if k < M.H4_EMA_SLOW + 5 or np.isnan(f[k]) or np.isnan(s_[k]):
        return None
    if f[k] > s_[k] and c[k] > s_[k]:
        return "up"
    if f[k] < s_[k] and c[k] < s_[k]:
        return "down"
    return None


def run_va_pair(pair: str, rank: int, df15: pd.DataFrame, df4: pd.DataFrame) -> list:
    if len(df15) < 400 or len(df4) < 60:
        return []
    lv = day_levels(df15)
    ser = h4_trend_series(df4)
    atr_s = M.atr(df15, 14).to_numpy()
    o, h, l, c = (df15[k].to_numpy() for k in ("open", "high", "low", "close"))
    t = df15["time"].to_numpy()
    t0 = int(t[0]) + WARMUP_DAYS * 86400
    busy = {"LONG": -1, "SHORT": -1}
    out = []
    for i in range(VA_LOOKBACK + 2, len(df15) - 2):
        if int(t[i]) < t0:
            continue
        day = int(t[i]) // 86400 * 86400
        if day not in lv:
            continue
        hour = ((int(t[i]) + 900 + 60) % 86400) // 3600
        if not (M.SESSION_START_UTC <= hour < M.SESSION_END_UTC):
            continue
        vah, val, poc = lv[day]
        a = atr_s[i]
        if not np.isfinite(a) or a <= 0:
            continue
        side = None
        if c[i - 2] > vah and val <= c[i - 1] <= vah and val <= c[i] <= vah:
            side = "SHORT"
        elif c[i - 2] < val and val <= c[i - 1] <= vah and val <= c[i] <= vah:
            side = "LONG"
        if side is None or i <= busy[side]:
            continue
        entry = float(o[i + 1])
        if side == "SHORT":
            sl = float(h[i - VA_LOOKBACK + 1:i + 1].max()) + M.STOP_ATR_BUFFER * a
            target, risk = val, sl - entry
            ok = target < entry < sl
        else:
            sl = float(l[i - VA_LOOKBACK + 1:i + 1].min()) - M.STOP_ATR_BUFFER * a
            target, risk = vah, entry - sl
            ok = sl < entry < target
        if not ok or risk <= 0 or not (M.MIN_SL_DISTANCE_PCT <= risk / entry <= VA_MAX_RISK):
            continue
        if abs(target - entry) / risk < VA_MIN_RR:
            continue
        sig = {"direction": side, "entry": entry, "sl": sl, "tp1": target, "tp2": target,
               "setup_low": 0.0, "setup_high": 0.0}
        tr = simulate_trade(df15, i, sig)
        if tr is None:
            continue
        busy[side] = i + 1 + tr["bars"]
        trend = h4_state(ser, int(t[i]) + 900)
        out.append({"pair": pair, "rank": rank, "mode": "VA_ROTATION", "direction": side,
                    "signal_time": int(t[i]) + 900, "trend": trend,
                    "with_trend": (side == "SHORT" and trend == "down") or (side == "LONG" and trend == "up"), **tr})
    return out


def va_report(trades: list, meta: dict) -> str:
    L = ["", "=" * 100, "STUDY 2: VALUE-AREA RE-ENTRY ROTATION (hypothesis fixed in advance; nothing tuned)",
         "Rule: close outside the previous-day VA, then 2 consecutive 15m closes back inside -> trade to the opposite VA edge.",
         "Stop beyond the 8-bar excursion extreme +/-0.4 ATR; RR>=1.5; risk 0.5-3%; session 07-20 UTC; entry = next candle open;",
         "entry candle exits use its close only; SL checked before TP; 10-day timeout. Same biases as STUDY 1.", ""]
    if not trades:
        return "\n".join(L + ["NO TRADES"])
    for t in trades:
        t["r_net"] = t["r_gross"] - FEE_ROUND_TRIP / t["risk_pct"]
    mid = trades[len(trades) // 2]["signal_time"]
    def line(label, rows):
        L.append(fmt(label, stats([x["r_net"] for x in rows])))
    line("A: all trades (r_net)", trades)
    L.append(fmt("A: all trades (r_gross)", stats([x["r_gross"] for x in trades])))
    line("A first half (r_net)", [x for x in trades if x["signal_time"] < mid])
    line("A second half (r_net)", [x for x in trades if x["signal_time"] >= mid])
    wt = [x for x in trades if x["with_trend"]]
    L.append("")
    line("B: only WITH the 4H trend (r_net)", wt)
    L.append(fmt("B: with trend (r_gross)", stats([x["r_gross"] for x in wt])))
    line("B first half (r_net)", [x for x in wt if x["signal_time"] < mid])
    line("B second half (r_net)", [x for x in wt if x["signal_time"] >= mid])
    line("C: AGAINST the 4H trend / no trend (r_net)", [x for x in trades if not x["with_trend"]])
    L.append("")
    for d in ("LONG", "SHORT"):
        line(f"A {d} (r_net)", [x for x in trades if x["direction"] == d])
    months = {}
    for x in trades:
        months.setdefault(datetime.fromtimestamp(x["signal_time"], timezone.utc).strftime("%Y-%m"), []).append(x["r_net"])
    for k in sorted(months):
        L.append(fmt(f"month {k} (A r_net)", stats(months[k])))
    byc = {}
    for x in trades:
        byc[x["pair"]] = byc.get(x["pair"], 0.0) + x["r_net"]
    top = sorted(byc.items(), key=lambda kv: -kv[1])[:3]
    L.append("top 3 coins by net R: " + ", ".join(f"{k} {v:+.1f}" for k, v in top) + f" | total {sum(byc.values()):+.1f}")
    res = {}
    for x in trades:
        res[x["result"]] = res.get(x["result"], 0) + 1
    L.append("exits: " + ", ".join(f"{k}={v}" for k, v in sorted(res.items())))
    L.append(f"median risk pct: {100 * float(np.median([x['risk_pct'] for x in trades])):.2f}")
    return "\n".join(L)


def main():
    end = int(time.time()) // 900 * 900
    start = end - (DAYS + WARMUP_DAYS) * 86400
    M.get_kucoin_candles = sim_candles
    M.get_live_price = sim_live_price
    coins = M.get_top_coins(M.SCAN_TOP_N_COINS)
    print("universe:", len(coins), [c["symbol"] for c in coins])
    trades, va_trades, have = [], [], 0
    for rank, c in enumerate(coins):
        df15 = fetch_history(c["pair"], "15min", start, end)
        df4 = fetch_history(c["pair"], "4hour", start - 40 * 86400, end)
        if len(df15) < 400:
            print("skip (no data):", c["pair"]); continue
        have += 1
        got = run_pair(c["pair"], c["symbol"], rank, df15, df4)
        va = run_va_pair(c["pair"], rank, df15, df4)
        print(f"{c['pair']}: bars15={len(df15)} study1={len(got)} study2={len(va)}", flush=True)
        trades += got
        va_trades += va
    trades = apply_run_cap(trades)
    va_trades = sorted(va_trades, key=lambda x: x["signal_time"])
    meta = {"start": datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
            "end": datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d"), "days": DAYS, "coins": have}
    text = "STUDY 1: THE LIVE BOT (reversal mode only; trend mode is disabled in the bot)\n" + report(trades, meta) + va_report(va_trades, meta)
    print(text)
    with open("lvar_backtest_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open("lvar_backtest_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "mode", "direction", "signal_time", "entry", "result", "r_gross", "r_net", "risk_pct", "bars"]
        w.writerow(cols)
        for t in trades + va_trades:
            t.setdefault("r_net", t["r_gross"] - FEE_ROUND_TRIP / t["risk_pct"])
            w.writerow([t.get(k) for k in cols])


if __name__ == "__main__":
    main()
