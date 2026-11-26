"""N2-E executable-edge study - DEVELOPMENT only (events 2022-11-01..2024-12-31). Read-only; prints only; nothing is written.
Holdout months and events are never loaded or analysed here. The SPEC text (with the hashed event calendar) is checked before any data is loaded."""
import hashlib, math, re, sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import n1_fomc_dev as n1

SPEC_SHA256 = "05954b52b57df2e28a0952b91ac9fa4958bc2a68f5af33571b7e8f9f56d8ad1c"
SPEC = r"""N2-E EXECUTABLE-EDGE STUDY - PRE-REGISTRATION (version 1)

QUESTION
After a realistic execution delay, fees, slippage, volatility and reversal risk, does the sign of the first 5-minute BTC reaction to scheduled US macro releases (CPI, Employment Situation = NFP) predict the following move by more than costs? Directional edge only; volatility increase alone is not an edge. Descriptive/research only: no deployment, no live trading, the four registered paper tracks are not involved.

1. EVENTS AND VERIFIED SOURCES
CPI and NFP: BLS yearly release calendar pages, fetched 2026-11-25, every row printed 08:30 AM Eastern Time: bls.gov/schedule/2022/home.htm (Nov-Dec 2022), bls.gov/schedule/2023/, bls.gov/schedule/2024/home.htm, bls.gov/schedule/2025, bls.gov/schedule/2026/home.htm.
FOMC statements: federalreserve.gov press-release pages (page-stated 2:00 p.m. ET), verified in N0b run #1 (commit 7b8bcbe).
Calendar anomalies: the BLS 2025 page lists no CPI for Oct-2025 data and no Employment Situation for Oct-2025 data (not published); the Sept-2025 CPI (2025-10-24) and the Sept-2025 Employment Situation (2025-11-20) are listed and included. Dates are the calendar dates printed by BLS; actual publication seconds are not verified; no consensus data.
t0 = release time converted from America/New_York to UTC with the tz database (DST handled).
EV_CPI: 2022-11-10 2022-12-13 2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12 2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11 2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18 2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14
EV_NFP: 2022-11-04 2022-12-02 2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08 2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06 2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16 2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02
EV_FOMC: 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28
Times: CPI and NFP 08:30 ET; FOMC 14:00 ET.

2. SAMPLES
Confirmatory family = CPI + NFP only. Development 2022-11-01..2024-12-31 (26 CPI + 26 NFP = 52 events). Holdout 2025-01-01..2026-10-31 (21 CPI + 21 NFP = 42 events): LOCKED, single use, not loaded by the development run.
FOMC (33 events) is EXPLORATORY ONLY: its first-reaction and +1h outcomes were already seen in N1, and the N1 holdout is consumed; FOMC is never part of a confirmatory test. No CPI/NFP outcome has been queried by any earlier analysis.
A CPI/NFP event is excluded if any other event in this calendar (any type) has t0 within [t0-60min, t0+240min] (contamination), if a 1m candle is missing in [t0-60min, t0+240min], or if fewer than 5 baseline days are valid; counts are reported.

3. DATA
Binance USDT-M monthly 1m klines (UTC): BTCUSDT primary, ETHUSDT secondary (descriptive). Fields used: open, high, low, close, volume, taker_buy_volume (rows with taker_buy > volume masked in imbalance sums and counted). Baseline for event time t0: the 10 preceding UTC weekdays that are not event dates of any type, same UTC clock time.

4. SIGNAL AND EXECUTION (only information available at decision time)
r5 = ln(close of the 1m candle starting t0+4min / open of the 1m candle starting t0). s = sign(r5).
sigma5 = median over the baseline days of the 5-minute range (max high - min low over t0..t0+4min, divided by the window open). sigma60 = same for 60 minutes. z5 = |r5| / sigma5.
A trade is taken iff z5 >= 1.0 (single fixed threshold, not optimised; other z5 buckets are descriptive only).
Entry: market order at the open of the 1m candle starting t0+D. D = 15 min (base; the current stack's observed dispatch delay is ~6 min typical, up to 35-59 min in rare windows), D = 30 min (stress delay). No fill at event-time prices.
Direction: continuation (H-E1): long if s>0, short if s<0. Reversal (H-E2): the opposite side.
Exit: at the close of the 1m candle starting t0+H-1 (horizon H = 60 min from t0 primary; 240 min secondary).
Risk: hard stop at 1.0 x sigma60 x entry price against the position, checked on every 1m candle from the entry candle to the exit candle; fill at the stop level, or at the candle open if the open is already beyond the stop. No take profit, one trade per event and direction.
Costs: round trip 30 bps (base) and 40 bps (stress), subtracted from each trade's simple return; they are meant to include fees and slippage.
Return per trade = dir * (exit/entry - 1) - cost.

5. CONFIRMATORY HYPOTHESES (BTC, H = 60, D = 15, cost 30 bps, events with z5 >= 1)
H-E1: mean net return of continuation trades > 0.
H-E2: mean net return of reversal trades > 0.
Test: resample events with replacement, 10000 draws, numpy default_rng seed 20261125; a hypothesis is supported if the 2.5th percentile of the bootstrap mean net return is > 0 (one-sided 97.5% bound; Bonferroni for two hypotheses).
Stages: (1) development run; (2) holdout gate: the holdout is run only for a hypothesis whose development mean net return (30 bps) is > 0; (3) single holdout run with the rules unchanged: confirmed only if the holdout 2.5th percentile > 0 and the development mean was > 0.
An "executable edge" statement additionally requires the holdout mean net return to be > 0 at 40 bps cost and at D = 30 min. Development and holdout are never pooled; no threshold, stop, horizon or delay is changed after outcomes are seen.

6. DESCRIPTIVE (no selection, no inference)
ETH; H = 240; D = 30; cost 40; CPI vs NFP split; z5 buckets (<0.5, 0.5-1, 1-2, >=2) over all events; agreement between the sign of the first-5-minute taker-buy imbalance and s; first-5-minute volume ratio vs baseline; FOMC exploratory results. None of these may be used to change confirmatory rules.

7. POWER AND LIMITATIONS
Active trades will be a subset of 52 development and 42 holdout events. With about 25 trades and a standard deviation of about 100 bps per trade, the standard error of the mean is about 20 bps, so only mean net edges of roughly 40 bps or more per trade (about 70 bps gross) are detectable; a non-significant result is not evidence of no edge. Actual n and standard deviation are reported with the development result before any holdout decision.
Other limits: no consensus or first-release data; BLS calendar dates, not verified publication seconds; other simultaneous releases not modelled; order-book depth and spreads at releases are not available, so the 30/40 bps costs are assumptions and may understate slippage at event time; stop fill assumptions are optimistic in fast markets.
"""
NY = ZoneInfo("America/New_York"); HM = {"CPI": (8, 30), "NFP": (8, 30), "FOMC": (14, 0)}
MON = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13) if (y, m) >= (2022, 10)]   # 2022-10 only feeds baselines
CFG = [(15, 60), (30, 60), (15, 240), (30, 240)]   # (entry delay D, horizon H) in minutes; primary = (15, 60)
SYMS = ("BTCUSDT", "ETHUSDT"); LAST_DEV = "2024-12-31"

def events():
    out = []
    for typ, ds in re.findall(r"^EV_(\w+): (.*)$", SPEC, re.M):
        h, m = HM[typ]
        for d in ds.split():
            y, mo, dd = map(int, d.split("-")); out.append((typ, d, int(datetime(y, mo, dd, h, m, tzinfo=NY).astimezone(timezone.utc).timestamp() * 1000)))
    return sorted(out, key=lambda e: e[2])

def trade(A, ie, ix, d, dist):
    """Market entry at open of candle ie, hard stop dist away, exit at close of candle ix. Returns (simple return, stopped)."""
    pe = A["o"][ie]; stop = pe - d * dist
    for j in range(ie, ix + 1):
        o, h, l = A["o"][j], A["h"][j], A["l"][j]
        if d > 0 and l <= stop: return min(stop, o) / pe - 1, 1
        if d < 0 and h >= stop: return -(max(stop, o) / pe - 1), 1
    return d * (A["c"][ix] / pe - 1), 0

def per_event(A, T0, t, evdays):
    i0 = (t - T0) // 60000
    if i0 - 60 < 0 or i0 + 241 > len(A["o"]) or any(np.isnan(A[k][i0 - 60:i0 + 241]).any() for k in ("o", "h", "l", "c", "v", "tbv")): return "missing"
    bt = [(x - T0) // 60000 for x in n1.base_times(t, evdays)]
    W = {k: [w for w in (n1.win(A, i, k) for i in bt) if w] for k in (5, 60)}
    if min(len(W[5]), len(W[60])) < 5: return "baseline"
    md = lambda k, f: float(np.median([w[f] for w in W[k]])); w5 = n1.win(A, i0, 5); r5 = math.log(w5["pk"] / w5["p0"]); s = int(np.sign(r5))
    ev = dict(r5=r5 * 1e4, s=s, z5=abs(r5) / md(5, "rng"), imb=w5["imb"], vr=w5["qv"] / md(5, "qv"), res={})
    for D, H in CFG:
        ie, ix = i0 + D, i0 + H - 1; dist = md(60, "rng") * A["o"][ie]
        for kind, dr in (("cont", s), ("rev", -s)): ev["res"][(D, H, kind)] = trade(A, ie, ix, dr, dist) if s else (0.0, 0)
    return ev

def line(label, tr):
    """tr = list of (gross return, stopped). Bootstrap over events."""
    if not tr: print(f"{label}: n 0"); return
    g = np.array([x[0] for x in tr]); s = f"{label}: n {len(g)}, mean gross {g.mean() * 1e4:+.1f} bps, sd {g.std(ddof=1) * 1e4 if len(g) > 1 else 0:.0f}, stopped {sum(x[1] for x in tr)}/{len(g)}"
    for c in (0.003, 0.004):
        lo, hi = n1.ci(g - c, np.mean) if len(g) > 1 else (np.nan, np.nan); s += f" | net@{int(c * 1e4)}: mean {(g - c).mean() * 1e4:+.1f} [{lo * 1e4:+.1f}, {hi * 1e4:+.1f}] win {np.mean(g - c > 0):.2f}"
    print(s)

def main():
    assert hashlib.sha256(SPEC.encode()).hexdigest() == SPEC_SHA256, "REGISTRATION MISMATCH: SPEC text changed"
    ev = events(); evdays = {pd.Timestamp(t, unit="ms", tz="UTC").strftime("%Y-%m-%d") for _, _, t in ev}; dev = [e for e in ev if e[1] <= LAST_DEV]
    ty = [e[0] for e in dev]; assert (ty.count("CPI"), ty.count("NFP"), ty.count("FOMC")) == (26, 26, 18) and max(MON) == "2024-12"
    print("SPEC sha256", SPEC_SHA256, "(verified)\nscript sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    print(f"DEVELOPMENT ONLY: {len(dev)} events 2022-11-02..2024-12-18; months loaded {MON[0]}..{MON[-1]}; holdout not loaded")
    R = {}; excl = {}
    for sym in SYMS:
        T0, A = n1.load(sym, MON)
        for typ, d, t in dev:
            if any(t - 3600000 <= t2 <= t + 14400000 for _, _, t2 in ev if t2 != t): r = "overlap"
            else: r = per_event(A, T0, t, evdays)
            if isinstance(r, str): excl[(sym, typ, r)] = excl.get((sym, typ, r), 0) + 1
            else: R[(sym, typ, d)] = r
    print("exclusions (symbol, type, reason): count:", excl or "none")
    P = (15, 60)
    def sel(sym, types, cfg=P, kind="cont", active=True):
        return [e["res"][cfg + (kind,)] for (s_, ty, d), e in sorted(R.items()) if s_ == sym and ty in types and e["s"] and (e["z5"] >= 1 or not active)]
    print("\n== CONFIRMATORY FAMILY (development): BTC, CPI+NFP, z5>=1, D=15, H=60; return per trade, bootstrap CI of the mean net return")
    gate = {}
    for kind, name in (("cont", "H-E1 continuation"), ("rev", "H-E2 reversal")):
        tr = sel("BTCUSDT", ("CPI", "NFP"), kind=kind); line(name, tr)
        g = np.array([x[0] for x in tr]) - 0.003
        gate[kind] = len(g) > 0 and g.mean() > 0
        lo = n1.ci(g, np.mean)[0] if len(g) > 1 else np.nan
        print(f"   development: 2.5th pct of mean net@30 = {lo * 1e4:+.1f} bps -> {'supported' if lo > 0 else 'not supported'}; HOLDOUT GATE (mean net@30 > 0): {'PASS' if gate[kind] else 'FAIL - holdout not to be run for this hypothesis'}")
    print("\nBTC event level (CPI+NFP): type | date | z5 | r5 bps | active | continuation gross bps (stopped)")
    for (s_, ty, d), e in sorted(R.items(), key=lambda x: x[0][2]):
        if s_ == "BTCUSDT" and ty in ("CPI", "NFP"):
            g, st = e["res"][P + ("cont",)]; print(ty, d, f"{e['z5']:.2f}", f"{e['r5']:+.1f}", "Y" if e["s"] and e["z5"] >= 1 else "n", f"{g * 1e4:+.1f}", st, sep=" | ")
    print("\n== DESCRIPTIVE (no selection): active trades, CPI+NFP, all configs (D, H) x direction, BTC")
    for cfg in CFG:
        for kind in ("cont", "rev"): line(f"BTC D{cfg[0]} H{cfg[1]} {kind}", sel("BTCUSDT", ("CPI", "NFP"), cfg, kind))
    for kind in ("cont", "rev"): line(f"ETH D15 H60 {kind}", sel("ETHUSDT", ("CPI", "NFP"), P, kind))
    for ty in ("CPI", "NFP"): line(f"BTC {ty} cont D15 H60", sel("BTCUSDT", (ty,)))
    print("z5 buckets (all events with a non-zero first reaction, BTC CPI+NFP, continuation, D15 H60; no threshold selection):")
    for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 99)):
        tr = [e["res"][P + ("cont",)] for (s_, ty, d), e in R.items() if s_ == "BTCUSDT" and ty in ("CPI", "NFP") and e["s"] and lo <= e["z5"] < hi]
        print(f"  z5 [{lo},{hi}): n {len(tr)}, mean gross {np.mean([x[0] for x in tr]) * 1e4 if tr else float('nan'):+.1f} bps")
    ag = [(e["res"][P + ("cont",)], np.sign(e["imb"]) == e["s"]) for (s_, ty, d), e in R.items() if s_ == "BTCUSDT" and ty in ("CPI", "NFP") and e["s"] and e["z5"] >= 1]
    for flag, name in ((True, "taker imbalance agrees with s"), (False, "taker imbalance disagrees")):
        t = [x[0][0] for x in ag if x[1] == flag]; print(f"  active, {name}: n {len(t)}, mean cont gross {np.mean(t) * 1e4 if t else float('nan'):+.1f} bps")
    vr = [e["vr"] for (s_, ty, d), e in R.items() if s_ == "BTCUSDT" and ty in ("CPI", "NFP")]; print(f"  first-5m volume ratio vs baseline, BTC CPI+NFP: median {np.median(vr) if vr else float('nan'):.2f}")
    print("\n== EXPLORATORY (FOMC outcomes already seen in N1; not confirmatory): BTC D15 H60, z5>=1")
    for kind in ("cont", "rev"): line(f"BTC FOMC {kind}", sel("BTCUSDT", ("FOMC",), kind=kind))
    print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
