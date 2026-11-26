"""H-F forced-flow reversal PILOT (Step 0a availability + Step 0b pilot). Pilot period 2022-11-01..2023-12-31 only. Read-only; prints only.
The SPEC text is checked against its hash before any data is loaded."""
import hashlib, io, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import n1_fomc_dev as n1, n2_exec_edge_dev as n2
from audit_a_n0 import BASE
from audit_a3_integrity import get

SPEC_SHA256 = "45a1d3a2a00d0958af68e120902a77e9a0f6ebb967488b069db3e13cadef1a43"
SPEC = r"""H-F FORCED-FLOW REVERSAL PILOT (Steps 0a and 0b) - PRE-REGISTRATION (version 1)

QUESTION
After a large 15-minute move on a liquid USDT-M perpetual, is the subsequent partial reversal larger when open interest fell during the move (forced flow) than when it did not (control), and large enough to matter against costs?

1. SAMPLE AND DATA
Pilot period only: event windows starting 2022-11-01 00:00 UTC, ending no later than 2023-12-31 19:00 UTC. No data after 2023-12-31 and none of the consumed holdouts (N1 2025-2026 events), the locked N2-E holdout or any development result of N2-N4 is used or consulted. Thresholds below are fixed here and come from no previous result.
Symbols BTCUSDT ETHUSDT SOLUSDT (Binance USDT-M). 1m klines (monthly archive 2022-10 .. 2023-12; 2022-10 is warm-up) and 5-minute open interest sum_open_interest from the public daily "metrics" files (BASE/daily/metrics/SYM/SYM-metrics-YYYY-MM-DD.zip, days 2022-11-01 .. 2023-12-31). A metrics timestamp is treated as the snapshot time; the same convention is applied at both ends of every window.

2. STEP 0a - AVAILABILITY GATE
For each symbol: days present, missing days, 5-minute slots found versus 288 x days, longest gap. Gate PASS if for every symbol at most 5% of the 5-minute slots are missing; otherwise the pilot is NOT run and the line stops.

3. STEP 0b - EVENTS (all times UTC)
Windows are the 15-minute intervals starting at :00, :15, :30, :45. For the window starting at s (1m candle starts s .. s+14 min), the window end is e = s + 15 min and r = ln(close of the 1m candle starting at s+14 min / open of the 1m candle starting at s). direction d = sign(r).
Volatility threshold: 4 x the median of |r| over the 672 windows (7 days) that END at or before s (the current window is excluded; at least 500 valid windows needed). An event is a window with |r| >= threshold.
Exclusions and spacing: events with a scheduled CPI, NFP or FOMC release (calendar of the N2-E registration, SPEC sha256 05954b52b57df2e28a0952b91ac9fa4958bc2a68f5af33571b7e8f9f56d8ad1c) between s - 120 min and e + 120 min are excluded; of the remaining events per symbol in time order, an event is dropped if its start is less than 60 min after the end of the last accepted event of that symbol (spacing uses size only, never open interest).
Open interest change: dOI = OI(e) / OI(s) - 1 using the 5-minute snapshots exactly at s and e (both required, positive). FORCED group: dOI <= -1.0%. CONTROL group: dOI >= 0. Events in between are counted and not analysed. Events with missing OI are counted and excluded.
Clusters: accepted events of all symbols, sorted by window end, form one cluster while consecutive window ends are at most 60 min apart; clusters are the resampling unit and cover BTC-coincident altcoin events. The share of ETH/SOL events in a cluster that also contains a BTC event is reported.

4. TRADES (reversal, against d; no stop)
Entry: open of the 1m candle starting at e + D minutes, D = 15 and D = 30. Exit: close of the 1m candle starting at entry time + H - 1 minutes (H minutes after entry), H = 60 and H = 240. gross = -d x (exit / entry - 1). A trade with a missing 1m candle at entry or exit is excluded and counted. Costs are not in gross; implications are shown at 30 bps (base) and 60 bps (stress), deducted once per round trip.

5. OUTPUT (fixed)
Counts at every filter step per symbol; for FORCED and CONTROL and each (D, H): n, mean gross, median, 10th/90th percentile, sd, share with gross > 0, mean net at 30 and 60 bps, cluster-bootstrap 95% interval of the mean gross; forced-minus-control difference with its cluster-bootstrap 95% interval; FORCED per symbol (D15, H60). Bootstrap: 10,000 draws, numpy default_rng seed 20261126, resampling clusters.

6. PREDECLARED STOP / CONTINUE (evaluated once, D = 15, H = 60, never changed after results)
CONTINUE only if ALL hold: (a) FORCED n >= 60; (b) FORCED mean gross >= 45 bps (1.5 x base cost); (c) FORCED mean gross > 0 in at least 2 of 3 symbols; (d) the 2.5th percentile of the forced-minus-control difference in mean gross > 0. Otherwise STOP: the line is closed without re-tuning thresholds. H = 240 and D = 30 are descriptive. A CONTINUE leads only to a separate preregistration (a later step), never to a strategy.

7. LIMITS
Open interest drop is a proxy for liquidation (voluntary closing also lowers it); no liquidation, spread or depth data; 5-minute snapshots; pilot results are exploratory; pooled clusters are not fully independent; costs are assumptions (stress is shown separately).
"""
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MON = [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13) if (y, m) >= (2022, 10)]
MS, Q, B = 60000, 900000, 10000
U = lambda *a: int(datetime(*a, tzinfo=timezone.utc).timestamp() * 1000)
START, END_LAST = U(2022, 11, 1), U(2023, 12, 31, 19)
DAYS = [(date(2022, 11, 1) + timedelta(d)).isoformat() for d in range(425)]       # 2022-11-01 .. 2023-12-31

def oi_day(a):
    sym, d = a
    data, _ = get(f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{d}.zip")
    if data is None: return None
    z = zipfile.ZipFile(io.BytesIO(data)); df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    if "create_time" not in df or "sum_open_interest" not in df: raise SystemExit(f"UNEXPECTED METRICS COLUMNS {list(df.columns)} ({sym} {d})")
    t = df["create_time"]
    ts = t.astype("int64").to_numpy() if pd.api.types.is_numeric_dtype(t) else ((pd.to_datetime(t, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).to_numpy(np.int64)
    return ts, df["sum_open_interest"].to_numpy(float)

def load_oi(sym):
    """Step 0a for one symbol -> ({ts: oi}, share of 5-minute slots missing)."""
    with ThreadPoolExecutor(12) as ex: res = list(ex.map(oi_day, [(sym, d) for d in DAYS]))
    ok = [r for r in res if r is not None]; oi = {}
    for ts, v in ok:
        for t, x in zip(ts, v):
            if START <= t < START + len(DAYS) * 86400000 and t % 300000 == 0 and x > 0: oi[int(t)] = float(x)
    k = np.array(sorted(oi)); gap = (np.diff(k).max() / MS) if len(k) > 1 else float("nan"); miss = 1 - len(oi) / (len(DAYS) * 288)
    print(f"  {sym}: days present {len(ok)}/{len(DAYS)} (missing {len(DAYS) - len(ok)}), valid 5m slots {len(oi)}/{len(DAYS) * 288}, missing share {miss:.3f}, longest gap {gap:.0f} min")
    return oi, miss

def events(sym, T0, A, rel, oi):
    n = len(A["o"]); s0 = int(-(T0 // MS) % 15); nw = (n - s0) // 15; i0 = s0 + 15 * np.arange(nw); st = T0 + i0 * MS
    r = np.log(A["c"][i0 + 14] / A["o"][i0]); thr = 4 * pd.Series(np.abs(r)).shift(1).rolling(672, min_periods=500).median().to_numpy()
    mask = ~np.isnan(r) & ~np.isnan(thr) & (np.abs(r) >= thr) & (st >= START) & (st + Q <= END_LAST)
    cnt = dict(size=0, news=0, spaced=0, nooi=0, mid=0, forced=0, control=0); out = []; last_end = -10**18
    for k in np.flatnonzero(mask):
        s = int(st[k]); e = s + Q; cnt["size"] += 1
        if any(s - 7200000 <= x <= e + 7200000 for x in rel): cnt["news"] += 1; continue
        if s < last_end + 3600000: cnt["spaced"] += 1; continue
        last_end = e; d = int(np.sign(r[k])); a, b = oi.get(s), oi.get(e)
        if a is None or b is None: g = "nooi"
        else: g = "forced" if b / a - 1 <= -0.01 else "control" if b / a - 1 >= 0 else "mid"
        cnt[g] += 1
        row = dict(sym=sym, e=e, grp=g, d=d)
        for D in (15, 30):
            for H in (60, 240):
                ie = int(i0[k]) + 15 + D; ix = ie + H - 1
                row[f"g{D}_{H}"] = -d * (A["c"][ix] / A["o"][ie] - 1) * 1e4 if ix < n and not np.isnan([A["o"][ie], A["c"][ix]]).any() else np.nan
        out.append(row)
    print(f"  {sym}: size-qualified {cnt['size']}, news-excluded {cnt['news']}, spacing-dropped {cnt['spaced']}, accepted {len(out)} -> OI missing {cnt['nooi']}, in-between {cnt['mid']}, FORCED {cnt['forced']}, CONTROL {cnt['control']}")
    return out

def main():
    assert hashlib.sha256(SPEC.encode()).hexdigest() == SPEC_SHA256, "REGISTRATION MISMATCH: SPEC text changed"
    print("SPEC sha256", SPEC_SHA256, "(verified)\nscript sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    print("PILOT ONLY: windows 2022-11-01..2023-12-31 19:00Z; klines months", MON[0], "..", MON[-1], "(no later data loaded)\n\nSTEP 0a - open-interest availability")
    OI = {s: load_oi(s) for s in SYMS}
    if any(m > 0.05 for _, m in OI.values()): print("\nAVAILABILITY GATE: FAIL (missing share > 5%) -> pilot NOT run; line stops."); return
    print("\nAVAILABILITY GATE: PASS\n\nSTEP 0b - event counts per filter step"); rel = [e[2] for e in n2.events()]; rows = []
    for s in SYMS: T0, A = n1.load(s, MON); rows += events(s, T0, A, rel, OI[s][0])
    df = pd.DataFrame(rows).sort_values("e").reset_index(drop=True); df["cid"] = (df.e.diff() > 3600000).cumsum().astype(int); C = int(df.cid.max()) + 1
    idx = np.random.default_rng(20261126).integers(0, C, (B, C)); print(f"accepted events {len(df)} in {C} clusters")
    def sn(v, m):
        k = m & ~np.isnan(v); return np.bincount(df.cid[k], v[k], minlength=C), np.bincount(df.cid[k], minlength=C).astype(float)
    def bm(S, N):
        with np.errstate(all="ignore"): return S[idx].sum(1) / N[idx].sum(1)
    P = lambda x: np.nanpercentile(x, [2.5, 97.5])
    G = {g: (df.grp == g).to_numpy() for g in ("forced", "control")}
    print("\nRESULTS: reversal gross return in bps (trade against the 15-min move; cost not included)")
    for D in (15, 30):
        for H in (60, 240):
            v = df[f"g{D}_{H}"].to_numpy(float)
            for g in ("forced", "control"):
                x = v[G[g] & ~np.isnan(v)]; S, N = sn(v, G[g]); lo, hi = P(bm(S, N))
                print(f"{g.upper():8s} D{D} H{H}: n {len(x)}, mean {x.mean():+.1f} [{lo:+.1f},{hi:+.1f}], median {np.median(x):+.1f}, p10/p90 {np.percentile(x, 10):+.1f}/{np.percentile(x, 90):+.1f}, sd {x.std():.0f}, win {np.mean(x > 0):.2f} | net@30 {x.mean() - 30:+.1f}, net@60 {x.mean() - 60:+.1f}")
            (Sf, Nf), (Sc, Nc) = sn(v, G["forced"]), sn(v, G["control"]); dm = np.nanmean(v[G["forced"]]) - np.nanmean(v[G["control"]]); lo, hi = P(bm(Sf, Nf) - bm(Sc, Nc))
            print(f"   forced-minus-control D{D} H{H}: {dm:+.1f} [{lo:+.1f},{hi:+.1f}]")
    v = df["g15_60"].to_numpy(float); sm = {}
    for s in SYMS:
        x = v[(df.sym == s).to_numpy() & G["forced"] & ~np.isnan(v)]; sm[s] = x.mean() if len(x) else np.nan
        print(f"FORCED {s} D15 H60: n {len(x)}, mean {sm[s]:+.1f}")
    btc = set(df.cid[df.sym == "BTCUSDT"]); alt = df[(df.sym != "BTCUSDT") & df.grp.isin(["forced", "control"])]
    print(f"ETH/SOL forced+control events in a cluster that also has a BTC event: {np.mean(alt.cid.isin(btc)) if len(alt) else float('nan'):.2f} (n {len(alt)})")
    x = v[G["forced"] & ~np.isnan(v)]; (Sf, Nf), (Sc, Nc) = sn(v, G["forced"]), sn(v, G["control"]); lo = P(bm(Sf, Nf) - bm(Sc, Nc))[0]
    c = [("(a) forced n >= 60", len(x) >= 60, len(x)), ("(b) forced mean gross >= 45 bps", len(x) > 0 and x.mean() >= 45, round(float(x.mean()), 1) if len(x) else None),
         ("(c) forced mean > 0 in >= 2 of 3 symbols", sum(m > 0 for m in sm.values()) >= 2, sum(m > 0 for m in sm.values())), ("(d) 2.5th pct of forced-minus-control > 0", lo > 0, round(float(lo), 1))]
    print("\nPREDECLARED DECISION (D15, H60):"); [print(" ", n, "->", "PASS" if ok else "FAIL", f"({val})") for n, ok, val in c]
    print("DECISION:", "CONTINUE (separate preregistration only)" if all(ok for _, ok, _ in c) else "STOP - line closed, no re-tuning")
    print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
