#!/usr/bin/env python3
"""
Backtest of the KuCoin paper-engine v3 strategy (rules frozen, nothing tuned).
Uses the engine's OWN functions (evaluate_setup, check_candle, compute_pnl) so the backtest cannot drift from the live logic.
Data = Binance USDT-M perpetual klines (data.binance.vision) as a proxy for the KuCoin symbols (KuCoin API is not reachable from CI history).

PRE-REGISTERED PASS RULE (set before the run, never changed afterwards):
  net avgR >= +0.10  AND  net PF >= 1.2  AND  n >= 300  AND  both time halves net-positive.
  net = engine cost assumptions (fee 0.06%/side, slippage 0.04%/side, funding 0.01%/8h). Stress = 0.40% round trip + funding.
Conventions: setup = newest closed 15m candle, 4H candles closed by then; entry = OPEN of the next 15m candle (bar 1);
SL before TP in one candle; gap beyond the stop exits at the open; time exit = close of bar 30; one position per symbol; no portfolio cap.
"""
import argparse
import bisect
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import kucoin_paper_engine_v3 as E

SYMBOLS = {s: ("BTCUSDT" if s == "XBTUSDTM" else s[:-2]) for s in E.SYMBOLS}
RESULT_TXT, TRADES_CSV = "paper_v3_result.txt", "paper_v3_trades.csv"
STRESS_RT = 0.0040
PRE_TOL = 0.003            # loose prefilter around the EMA20 cross; the exact decision is always evaluate_setup()
CONTROL_P = 0.004
START_Y, START_M = 2022, 11


# ------------------------------------------------------------------ data
def load_symbol(bsym, months, F):
    out = {}
    for iv, gran in (("15m", E.BAR_S), ("4h", E.H4_S)):
        rows = {}
        for ym in months:
            content = F.fetch(f"{F.BASE}/monthly/klines/{bsym}/{iv}/{bsym}-{iv}-{ym}.zip")
            if not content:
                continue
            for r in F.read_zip_rows(content):
                try:
                    t = int(F.norm_t(r[0]))
                    o, h, l, c, v = (float(x) for x in r[1:6])
                except (ValueError, IndexError):
                    continue
                if t % gran == 0 and min(o, h, l, c) > 0 and h >= max(o, c) and l <= min(o, c):
                    rows[t] = (t, o, h, l, c, v)
        out[iv] = [rows[t] for t in sorted(rows)]
    return out["15m"], out["4h"]


# ------------------------------------------------------------------ core
def simulate(m15, i, side, atr):
    """paper trade for a setup on candle i. Returns (trade_dict, exit_index) | (None, reason)."""
    n = len(m15)
    j = i + 1
    if j >= n or m15[j][0] != m15[i][0] + E.BAR_S:
        return None, "NO_ENTRY_CANDLE"
    entry = m15[j][1]
    risk = E.SL_ATR * atr
    stop, target = (entry - risk, entry + E.TARGET_R * risk) if side == "LONG" else (entry + risk, entry - E.TARGET_R * risk)
    if risk <= 0 or stop <= 0 or target <= 0:
        return None, "BAD_STOP"
    pos = {"side": side, "entry": entry, "stop": stop, "target": target, "entry_ts": m15[j][0], "bars_held": 0, "last_processed_ts": m15[j][0] - E.BAR_S}
    k = j
    while k < n:
        if m15[k][0] != pos["last_processed_ts"] + E.BAR_S:
            return None, "DATA_GAP"
        res = E.check_candle(pos, m15[k])
        if res:
            reason, price = res
            pnl = E.compute_pnl(side, entry, stop, price, pos["entry_ts"], m15[k][0])
            risk_pct = abs(entry - stop) / entry
            sg = 1 if side == "LONG" else -1
            stress_pct = pnl["gross_pnl_pct"] - STRESS_RT - pnl["funding_pct"]
            return {"side": side, "setup_ts": m15[i][0], "entry_ts": pos["entry_ts"], "exit_ts": m15[k][0], "reason": reason, "bars": pos["bars_held"],
                    "risk_pct": risk_pct, "gross_R": pnl["gross_R"], "net_R": pnl["net_R"], "stress_R": stress_pct / risk_pct, "sg": sg}, k
        k += 1
    return None, "OPEN_AT_END"


def run_symbol(sym, m15, h4, mode="strategy", seed=1):
    """sequential, one position per symbol. mode 'strategy' = the engine rules; 'control' = random side/time, same exits."""
    n = len(m15)
    trades, drops = [], {}
    if n < E.FETCH_15M + 5:
        return trades, drops
    c = np.array([x[4] for x in m15])
    o = np.array([x[1] for x in m15])
    e20 = E.ema(c, 20)
    atr15, _ = E.atr_adx([x[2] for x in m15], [x[3] for x in m15], c)
    h4_ts = [x[0] for x in h4]
    rnd = random.Random(f"{seed}-{sym}")
    i = E.FETCH_15M - 1
    while i < n - 1:
        side = atr = None
        if mode == "strategy":
            e, ep = e20[i], e20[i - 1]
            if np.isfinite(e) and np.isfinite(ep):
                cand = []
                if c[i] > o[i] and c[i - 1] <= ep * (1 + PRE_TOL) and c[i] > e * (1 - PRE_TOL):
                    cand.append("LONG")
                if c[i] < o[i] and c[i - 1] >= ep * (1 - PRE_TOL) and c[i] < e * (1 + PRE_TOL):
                    cand.append("SHORT")
                if cand:
                    as_of = m15[i][0] + E.BAR_S
                    kk = bisect.bisect_right(h4_ts, as_of - E.H4_S)
                    setup, reason = E.evaluate_setup(h4[max(0, kk - E.FETCH_4H):kk], m15[i - E.FETCH_15M + 1:i + 1])
                    if reason:
                        drops[reason] = drops.get(reason, 0) + 1
                    elif setup:
                        side, atr = setup["side"], setup["atr"]
        else:
            if rnd.random() < CONTROL_P and np.isfinite(atr15[i]) and atr15[i] > 0:
                side, atr = rnd.choice(["LONG", "SHORT"]), float(atr15[i])
        if side is None:
            i += 1
            continue
        tr, k = simulate(m15, i, side, atr)
        if tr is None:
            drops[k] = drops.get(k, 0) + 1
            i += 1
            continue
        tr["symbol"] = sym
        trades.append(tr)
        i = k                                                    # the engine may scan the exit candle itself for a new setup
    return trades, drops


# ------------------------------------------------------------------ statistics
def pf(rs):
    w = sum(r for r in rs if r > 0)
    l = -sum(r for r in rs if r < 0)
    return w / l if l > 0 else float("inf") if w > 0 else 0.0


def block_stats(title, trs):
    if not trs:
        return [f"{title}: n=0"]
    g = np.array([t["gross_R"] for t in trs])
    nr = np.array([t["net_R"] for t in trs])
    sr = np.array([t["stress_R"] for t in trs])
    wins = float((nr > 0).mean())
    return [f"{title}: n={len(trs)} grossAvgR={g.mean():+.3f} netAvgR={nr.mean():+.3f} netPF={pf(nr):.2f} stressAvgR={sr.mean():+.3f} netWin={wins:.1%}"]


def bootstrap_weekly(trs, n_boot=2000, seed=7):
    if len(trs) < 20:
        return None
    wk = {}
    for t in trs:
        wk.setdefault(t["entry_ts"] // (7 * 86400), []).append(t["net_R"])
    keys = list(wk)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [rng.choice(keys) for _ in keys]
        vals = [x for k in pick for x in wk[k]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def verdict(trs):
    if not trs:
        return "FAIL (no trades)", {}
    nr = [t["net_R"] for t in trs]
    ts = sorted(t["entry_ts"] for t in trs)
    mid = ts[len(ts) // 2]
    h1 = [t["net_R"] for t in trs if t["entry_ts"] < mid]
    h2 = [t["net_R"] for t in trs if t["entry_ts"] >= mid]
    m = {"n": len(nr), "avg": float(np.mean(nr)), "pf": pf(nr), "h1": float(np.mean(h1)) if h1 else 0.0, "h2": float(np.mean(h2)) if h2 else 0.0}
    ok = m["avg"] >= 0.10 and m["pf"] >= 1.2 and m["n"] >= 300 and m["h1"] > 0 and m["h2"] > 0
    return ("PASS" if ok else "FAIL"), m


def report(trs, drops, ctrl, meta):
    out = [f"PAPER ENGINE v3 STRATEGY BACKTEST  generated {datetime.now(timezone.utc).isoformat()}", meta, ""]
    out += block_stats("ALL", trs)
    for s in ("LONG", "SHORT"):
        out += block_stats(f"  {s}", [t for t in trs if t["side"] == s])
    ts = sorted(t["entry_ts"] for t in trs)
    if ts:
        mid = ts[len(ts) // 2]
        out += block_stats("  half1", [t for t in trs if t["entry_ts"] < mid])
        out += block_stats("  half2", [t for t in trs if t["entry_ts"] >= mid])
    out.append("per symbol:")
    for s in SYMBOLS:
        out += block_stats(f"  {s}", [t for t in trs if t["symbol"] == s])
    rs = {}
    for t in trs:
        rs[t["reason"]] = rs.get(t["reason"], 0) + 1
    out.append(f"exit reasons: {rs}")
    out.append(f"dropped setups/trades (data issues): {drops}")
    bs = bootstrap_weekly(trs)
    out.append(f"weekly-block bootstrap 95% CI of net avgR: {('[%+.3f, %+.3f]' % bs) if bs else 'n/a'}")
    out += ["", "CONTROL (random side and time, identical exits and costs):"] + block_stats("  control", ctrl)
    v, m = verdict(trs)
    out += ["", f"PRE-REGISTERED RULE: net avgR>=+0.10, PF>=1.2, n>=300, both halves>0  ->  {v}  {m}"]
    return "\n".join(out)


# ------------------------------------------------------------------ main
def main():
    import lvar_fut_backtest as F
    now = datetime.now(timezone.utc)
    months, y, m = [], START_Y, START_M
    while (y, m) < (now.year, now.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    t0 = time.time()
    data = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {s: ex.submit(load_symbol, b, months, F) for s, b in SYMBOLS.items()}
        for s, f in futs.items():
            data[s] = f.result()
            print(f"{s}: 15m={len(data[s][0])} 4h={len(data[s][1])}  {time.time() - t0:.0f}s", flush=True)
    trades, ctrl, drops = [], [], {}
    for s, (m15, h4) in data.items():
        tr, d = run_symbol(s, m15, h4, "strategy")
        trades += tr
        for k, v in d.items():
            drops[k] = drops.get(k, 0) + v
        ctrl += run_symbol(s, m15, h4, "control")[0]
        print(f"{s}: trades={len(tr)}  {time.time() - t0:.0f}s", flush=True)
    meta = f"data: Binance USDT-M 15m+4h, {months[0]}..{months[-1]}, symbols={list(SYMBOLS.values())}"
    txt = report(trades, drops, ctrl, meta)
    print(txt)
    open(RESULT_TXT, "w", encoding="utf-8").write(txt + "\n")
    if trades:
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["symbol", "side", "setup_ts", "entry_ts", "exit_ts", "reason", "bars", "risk_pct", "gross_R", "net_R", "stress_R"], extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(trades, key=lambda t: t["entry_ts"]))


# ------------------------------------------------------------------ self-test
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def to_tuples(rows):
    return [tuple(r) for r in rows]


def run_selftest():
    import tempfile
    from pathlib import Path
    base = E.build_rows()
    mir = E.mirror_rows(base)
    # 1) prefilter never hides a setup that the engine's own evaluate_setup would signal
    for name, rows in (("LONG", base), ("SHORT", mir)):
        m15 = to_tuples(rows)
        a4 = E.agg4(rows)
        h4 = [tuple(a4[t]) for t in sorted(a4)]
        exact = []
        for i in range(E.FETCH_15M - 1, len(rows) - 1):
            st, _ = E.prefix_eval(rows, i, a4)
            if st:
                exact.append((i, st["side"]))
        got, k = [], 0
        c = np.array([x[4] for x in m15]); o = np.array([x[1] for x in m15]); e20 = E.ema(c, 20)
        h4_ts = [x[0] for x in h4]
        for i in range(E.FETCH_15M - 1, len(rows) - 1):
            cand = (c[i] > o[i] and c[i - 1] <= e20[i - 1] * (1 + PRE_TOL) and c[i] > e20[i] * (1 - PRE_TOL)) or \
                   (c[i] < o[i] and c[i - 1] >= e20[i - 1] * (1 - PRE_TOL) and c[i] < e20[i] * (1 + PRE_TOL))
            if cand:
                kk = bisect.bisect_right(h4_ts, m15[i][0] + E.BAR_S - E.H4_S)
                st, _ = E.evaluate_setup(h4[max(0, kk - E.FETCH_4H):kk], m15[i - E.FETCH_15M + 1:i + 1])
                if st:
                    got.append((i, st["side"]))
        check(exact == got and len(exact) >= 1, f"{name}: prefiltered scan finds exactly the engine's setups ({len(exact)} setups)")
    # 2) backtest trades == a live engine driven cycle by cycle on the same data
    for name, rows in (("LONG", base), ("SHORT", mir)):
        m15, a4 = to_tuples(rows), E.agg4(rows)
        h4 = [tuple(a4[t]) for t in sorted(a4)]
        # restrict to the span where the engine run is cheap: last 900 candles
        start = len(rows) - 900
        sub = rows[:]
        bt, _ = run_symbol("XBTUSDTM", m15, h4, "strategy")
        bt = [t for t in bt if t["setup_ts"] >= rows[start][0]]
        tmp = Path(tempfile.mkdtemp())
        eng, api, clock = E.make_engine(tmp, sub)
        t = rows[start][0] + E.BAR_S
        while t <= rows[-1][0] + E.BAR_S:
            clock["t"] = float(t + 2)
            eng.run_cycle()
            t += E.BAR_S
        live = [r for r in E.read_log(tmp) if r["status"] == "PAPER_EXIT"]
        lv = [(int(r["entry_ts"]), int(r["exit_ts"]), r["exit_reason"], round(float(r["gross_R"]), 9), round(float(r["net_R"]), 9)) for r in live]
        bv = [(t_["entry_ts"], t_["exit_ts"], t_["reason"], round(t_["gross_R"], 9), round(t_["net_R"], 9)) for t_ in bt]
        # the live run starts flat at `start`; drop backtest trades that overlap a position opened before it
        bv = [x for x in bv if x[0] > rows[start][0] + E.BAR_S]
        lv = [x for x in lv if x[0] > rows[start][0] + E.BAR_S]
        check(bv == lv and len(lv) >= 1, f"{name}: backtest trades equal the live engine's trades ({len(lv)} trades)")
    # 3) no look-ahead: cutting the data cannot change trades that finished before the cut
    for rows in (base, mir):
        m15, a4 = to_tuples(rows), E.agg4(rows)
        h4 = [tuple(a4[t]) for t in sorted(a4)]
        full, _ = run_symbol("S", m15, h4)
        cut = len(rows) - 700
        t_cut = m15[cut][0]
        part, _ = run_symbol("S", m15[:cut + 1], [x for x in h4 if x[0] + E.H4_S <= t_cut + E.BAR_S])
        f_ = [(t["entry_ts"], t["exit_ts"], t["gross_R"]) for t in full if t["exit_ts"] < t_cut]
        p_ = [(t["entry_ts"], t["exit_ts"], t["gross_R"]) for t in part if t["exit_ts"] < t_cut]
        check(f_ == p_ and len(f_) >= 1, f"changing the future leaves earlier trades identical ({len(f_)} trades)")
    # 4) costs and stats
    tr = [{"symbol": "S", "side": "LONG", "entry_ts": i * 1000, "net_R": r, "gross_R": r, "stress_R": r, "reason": "x"} for i, r in enumerate([2.0, -1.0, 2.0, -1.0])]
    check(abs(pf([2.0, -1.0, 2.0, -1.0]) - 2.0) < 1e-12 and verdict(tr)[0] == "FAIL", "profit factor and the n>=300 requirement")
    big = [{"symbol": "S", "side": "LONG", "entry_ts": i * 1000, "net_R": 0.3 if i % 2 else -0.05, "gross_R": 0.3, "stress_R": 0.1, "reason": "x"} for i in range(400)]
    check(verdict(big)[0] == "PASS", "a clearly positive sample passes")
    bad_half = [dict(t, net_R=(-0.3 if t["entry_ts"] < 200000 else 0.5)) for t in big]
    check(verdict(bad_half)[0] == "FAIL", "one negative half fails even when the average is positive")
    # 5) null test on a driftless random walk: gross must be about zero, net negative
    rng = np.random.default_rng(5)
    allt, allc = [], []
    for k in range(4):
        n = 60000
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        o = np.concatenate([[100.0], c[:-1]])
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0008, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0008, n)))
        v = rng.lognormal(0, 0.5, n) * 1000
        rows = [[E.T0 + i * E.BAR_S, o[i], h[i], l[i], c[i], v[i]] for i in range(n)]
        a4 = E.agg4(rows)
        h4 = [tuple(a4[t]) for t in sorted(a4) if t + E.H4_S <= rows[-1][0] + E.BAR_S]
        allt += run_symbol(f"R{k}", to_tuples(rows), h4)[0]
        allc += run_symbol(f"R{k}", to_tuples(rows), h4, "control")[0]
    g = float(np.mean([t["gross_R"] for t in allt])) if allt else 0.0
    n_ = float(np.mean([t["net_R"] for t in allt])) if allt else 0.0
    gc = float(np.mean([t["gross_R"] for t in allc]))
    print(f"null (random walk): strategy n={len(allt)} gross={g:+.3f} net={n_:+.3f} | control n={len(allc)} gross={gc:+.3f}")
    check(len(allt) >= 100 and abs(g) < 0.2 and n_ < g and abs(gc) < 0.1, "random walk: gross about 0, net below gross (no hidden edge or bias)")
    print(f"ALL BACKTEST SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    run_selftest() if a.selftest else main()
