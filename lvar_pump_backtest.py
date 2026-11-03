"""
PUMP FADE - short after a blow-off top, daily bars, ALL KuCoin USDT coins - hypothesis fixed IN ADVANCE, nothing tuned (read-only research).
Universe: every USDT pair listed on KuCoin today (stablecoins and leveraged tokens excluded) with 24h turnover >= 50k USDT today and >= 200 daily bars.
  Missing daily candles (no trades) are forward-filled; a coin is dropped if more than 10% of its days are missing.
Pump (known at the close of day i): base = lowest close of days i-10..i-4; peak = highest close of days i-3..i.
  run-up = peak/base - 1 >= 80%.
Trigger (day i): red candle (close < open) AND close <= 85% of the peak close (gave back at least 15% from the top).
Entry = open of day i+1 (no lookahead). Stop = 1% above the highest HIGH of days i-3..i. Risk must be 5%-35% of the entry price.
Variant A: target 2R. Variant B (sensitivity only): target = 50% retracement of the whole run-up (must be below the entry).
Time stop: close of the 10th day (entry day = day 1). SL before TP inside one bar; an SL gap fills at the open; TP fills at the target.
One position per coin; a coin is idle for 5 days after an exit. Portfolio: at most 8 open positions; the biggest run-up wins a tie on the same day.
Liquidity (average daily turnover of days i-40..i-11, close x volume, KuCoin spot only): eligible set >= 500k USDT per day; wide set >= 100k.
Costs (small coins are expensive to trade): 0.30% round trip + funding 0.02% per 8h charged to every trade as an assumption; stress 0.60%.
Pass rule (fixed before the run, Variant A, eligible set only): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.
The wide set, the perpetual-listed subset and Variant B are sensitivity/information blocks only and are not eligible for selection.
Perpetual-listed subset = coins that have an active USDT-margined KuCoin futures contract TODAY (heuristic name match; a coin may have got
its contract after the pump, so this subset is only a rough guide to what could really have been shorted on futures).
"""
import csv
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import crypto_signal_bot as M
import lvar_daily_backtest as B

DAY = B.DAY
MIN_BARS = 200
MAX_MISSING = 0.10
MIN_TURNOVER_TODAY = 50_000
BASE_LO, BASE_HI = 10, 4            # base window = days i-10 .. i-4
PEAK_WIN = 4                        # peak window = days i-3 .. i
RUN_MIN = 0.80
DROP_MIN = 0.15
STOP_BUF = 0.01
MIN_RISK, MAX_RISK = 0.05, 0.35
RR = 2.0
RETRACE = 0.50
MAX_HOLD = 10
COOLDOWN = 5
LIQ_ELIG = 500_000
LIQ_ALL = 100_000
COST_RT = 0.0030
COST_STRESS = 0.0060
FUNDING_8H = 0.0002


def universe():
    payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/allTickers")
    data_block = payload.get("data") if isinstance(payload, dict) else None
    items = data_block.get("ticker", []) if isinstance(data_block, dict) else []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        raw = str(it.get("symbol") or "")
        if not raw.endswith("-USDT"):
            continue
        base = raw[:-5]
        try:
            vol = float(it.get("volValue") or 0)
        except (TypeError, ValueError):
            continue
        if not base or base in M.EXCLUDED_BASES or base.endswith(M.EXCLUDED_SUFFIXES) or not np.isfinite(vol) or vol < MIN_TURNOVER_TODAY:
            continue
        out.append((raw, vol))
    return sorted(out, key=lambda x: -x[1])


def perp_bases():
    """base currencies with an active USDT-margined KuCoin futures contract today (empty set if the call fails)."""
    payload = M.safe_get_json("https://api-futures.kucoin.com/api/v1/contracts/active")
    data = payload.get("data") if isinstance(payload, dict) else None
    out = set()
    for it in data if isinstance(data, list) else []:
        if not isinstance(it, dict) or str(it.get("status") or "Open").lower() != "open":
            continue
        if str(it.get("quoteCurrency") or "").upper() != "USDT" or it.get("isInverse") is True:
            continue
        b = str(it.get("baseCurrency") or "").upper()
        if b == "XBT":
            b = "BTC"
        for pre in ("1000000", "10000", "1000"):
            if b.startswith(pre) and len(b) > len(pre):
                b = b[len(pre):]
                break
        if b:
            out.add(b)
    return out


def fill_gaps(df):
    """forward-fill days without trades; returns (df, number_of_missing_days)."""
    t = df["time"].to_numpy()
    full = np.arange(t[0], t[-1] + DAY, DAY)
    miss = len(full) - len(t)
    if miss == 0:
        return df.reset_index(drop=True), 0
    d = df.set_index("time").reindex(full)
    d["close"] = d["close"].ffill()
    for k in ("open", "high", "low"):
        d[k] = d[k].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0.0)
    d.index.name = "time"
    return d.reset_index(), miss


def sim_short(o, h, l, c, k, entry, stop, target):
    n = len(c)
    risk = stop - entry
    if risk <= 0:
        return None
    end_bar = k + MAX_HOLD - 1
    for m in range(k, min(n - 1, end_bar) + 1):
        if m > k and o[m] >= stop:
            return (entry - o[m]) / risk, m, "SL"
        if h[m] >= stop:
            return -1.0, m, "SL"
        if l[m] <= target:
            return (entry - target) / risk, m, "TP"
        if m == end_bar:
            return (entry - c[m]) / risk, m, "TIME"
    return None


def finish(r, risk_pct, days):
    fund = FUNDING_8H * 3.0 * days
    return r, r - (COST_RT + fund) / risk_pct, r - (COST_STRESS + fund) / risk_pct


def candidates(pair, df, variant, perps=None):
    n = len(df)
    if n < 60:
        return []
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = df["time"].to_numpy()
    turn = c * v
    out, busy = [], -10 ** 9
    for i in range(41, n - 1):
        if i <= busy + COOLDOWN:
            continue
        base = c[i - BASE_LO:i - BASE_HI + 1].min()
        peak_c = c[i - PEAK_WIN + 1:i + 1].max()
        if base <= 0 or peak_c / base - 1 < RUN_MIN:
            continue
        if not (c[i] < o[i] and c[i] <= peak_c * (1 - DROP_MIN)):
            continue
        liq = float(turn[i - 40:i - 10].mean())
        if liq < LIQ_ALL:
            continue
        k = i + 1
        if t[k] - t[i] != DAY:
            continue
        entry = o[k]
        stop = h[i - PEAK_WIN + 1:i + 1].max() * (1 + STOP_BUF)
        risk = stop - entry
        risk_pct = risk / entry if entry > 0 else 0.0
        if stop <= entry or not (MIN_RISK <= risk_pct <= MAX_RISK):
            continue
        if variant == "A":
            target = entry - RR * risk
        else:
            target = base + RETRACE * (peak_c - base)
            if target >= entry:
                continue
        if target <= 0:
            continue
        res = sim_short(o, h, l, c, k, entry, stop, target)
        if res is None:
            continue
        days = res[1] - k + 1
        rg, rn, rs = finish(res[0], risk_pct, days)
        busy = res[1]
        out.append({"pair": pair, "side": "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[1]]), "reason": res[2],
                    "days": days, "risk_pct": risk_pct, "r_gross": rg, "r_net": rn, "r_stress": rs,
                    "strength": float(peak_c / base - 1), "liq": liq,
                    "perp": bool(perps) and pair.split("-")[0] in perps})
    return out


def tier_lines(rows):
    L = ["  by liquidity (net R, positions actually taken):"]
    for lo, hi, name in ((LIQ_ALL, LIQ_ELIG, "100k-500k"), (LIQ_ELIG, 2_000_000, "500k-2M"), (2_000_000, float("inf"), ">=2M")):
        L.append(fmt_rows(f"    turnover {name}", [x["r_net"] for x in rows if lo <= x["liq"] < hi]))
    return L


def fmt_rows(label, rs):
    return B.fmt(label, rs)


def report(sets, meta):
    L = ["PUMP FADE - short after a blow-off top, daily, all KuCoin USDT coins (hypothesis fixed in advance; nothing tuned)",
         f"period: {meta['start']} -> {meta['end']} | pairs listed today: {meta['listed']} | coins with usable data: {meta['coins']} | futures bases found: {meta.get('perps', 0)}",
         "costs: 0.30% round trip + funding 0.02%/8h assumed on every trade; stress 0.60%; max 8 open positions",
         "Pass rule fixed before the run (Variant A, eligible set): net avgR >= +0.10 AND PF >= 1.2 AND n >= 300 AND both halves net-positive.",
         "Known problems: only coins still listed today (dead coins missing), KuCoin spot instead of perpetual data, no borrow/perp availability check,",
         "funding and slippage are assumptions, small coins can gap and squeeze beyond what daily bars show, news is not in the data.", ""]
    for title, all_c, cap, eligible in sets:
        L.append(f"signals before the position cap: {len(all_c)} | taken: {len(cap)}")
        L += B.block(title, cap, eligible)
        if cap:
            L += tier_lines(cap) + [""]
    return "\n".join(L)


def main():
    end = int(time.time()) // DAY * DAY
    start = end - B.DAYS * DAY
    coins = universe()
    print("pairs listed today:", len(coins), flush=True)
    data, dropped = {}, 0
    for idx, (pair, _vol) in enumerate(coins):
        try:
            df = B.fetch_daily(pair, start, end)
        except Exception as exc:  # network hiccup on one pair must not kill the run
            print("fetch error", pair, exc, flush=True)
            dropped += 1
            continue
        if len(df) < MIN_BARS:
            dropped += 1
            continue
        df, miss = fill_gaps(df)
        if miss > MAX_MISSING * len(df):
            dropped += 1
            continue
        data[pair] = df
        if idx % 50 == 0:
            print(f"progress {idx}/{len(coins)} usable {len(data)}", flush=True)
    print("usable coins:", len(data), "dropped:", dropped, flush=True)
    perps = perp_bases()
    print("coins with an active futures contract today:", len(perps), flush=True)
    a_all, b_all = [], []
    for pair, df in data.items():
        a_all += candidates(pair, df, "A", perps)
        b_all += candidates(pair, df, "B", perps)
    a_elig = [x for x in a_all if x["liq"] >= LIQ_ELIG]
    b_elig = [x for x in b_all if x["liq"] >= LIQ_ELIG]
    a_perp = [x for x in a_elig if x["perp"]]
    sets = [("VARIANT A, eligible set (turnover >= 500k): the tested rule", a_elig, B.portfolio(a_elig), True),
            ("VARIANT A, eligible AND futures contract listed today: information only", a_perp, B.portfolio(a_perp), False),
            ("VARIANT A, wide set (turnover >= 100k): sensitivity", a_all, B.portfolio(a_all), False),
            ("VARIANT B (target = 50% retracement of the run), eligible set: sensitivity", b_elig, B.portfolio(b_elig), False)]
    meta = {"start": datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
            "end": datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d"), "listed": len(coins), "coins": len(data),
            "perps": len(perps)}
    text = report(sets, meta) if a_all else "NO SIGNALS"
    print(text)
    with open("lvar_pump_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open("lvar_pump_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "strength", "liq", "perp"]
        w.writerow(cols)
        for x in sets[0][2]:
            w.writerow([x[k] for k in cols])


if __name__ == "__main__":
    main()
