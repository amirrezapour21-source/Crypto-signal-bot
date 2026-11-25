"""N1 FOMC PILOT - DEVELOPMENT period only (2022-11-01..2024-12-31). Read-only; prints only; nothing is written.
Holdout months are never downloaded or analysed. Registration: the SPEC text below is hash-checked before any data is loaded."""
import hashlib, io, math, sys, zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from audit_a_n0 import BASE
from audit_a3_integrity import get, COLS

SPEC_SHA256 = "fc99e8b784488760478cead7a8bdb40a0295f76233a8c22734fe5a6fad99d5d1"
SPEC = r"""N1 FOMC PILOT - PRE-REGISTRATION (version 1)

Scope: descriptive event study of BTC and ETH USDT-M perpetual 1m candles around FOMC statement releases. No trading rule, entry rule, strategy or deployment is derived from it. The four registered paper tracks are not involved.

1. EVENTS (verified: Fed press-release page states 2:00 p.m. ET for each)
Development (18): 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
Holdout (15, LOCKED; dates only, no outcome queried): 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28
t0 = 14:00 America/New_York on that date converted to UTC with the tz database (DST handled).
Other scheduled categories (CPI, NFP, PCE, GDP, retail sales, PMI) and news are excluded until their dates and timestamp precision are verified.

2. DATA AND WINDOWS
Binance USDT-M monthly 1m klines (data.binance.vision), UTC. BTCUSDT primary, ETHUSDT secondary. No altcoin basket in this pilot.
P0 = open of the 1m candle starting at t0. P(+k) = close of the 1m candle starting at t0+k-1 minutes. Windows k = 1, 5, 15, 60, 240 minutes.
A missing minute inside a window excludes that event-window for that instrument (no interpolation).
Rows with taker_buy_volume > volume are masked out of the taker-imbalance sums (raw data unchanged) and counted.

3. METRICS (per event and window)
Log return; range = (max high - min low)/P0; quote volume; taker-buy imbalance = (2*taker_buy - volume)/volume.
Baseline = median of the same metric over the same-length window at the same UTC clock time on the 10 preceding UTC weekdays that are not FOMC event dates; at least 5 valid days required.
VR_range = range/baseline range; VR_vol = quote volume/baseline quote volume (log ratios reported); imbalance reported as difference to baseline.
Continuation c = sign(r5) * (P(+60) - P(+5))/P0 with r5 = P(+5)/P0 - 1. Reversal if c < 0; events with c = 0 are excluded.

4. HYPOTHESES (one primary endpoint each, BTC)
H1 volatility rises around FOMC statements. Primary: median log VR_range at k = 60 above 0. One-sided exact sign test (events with log VR_range > 0), alpha 0.05, plus 95% percentile bootstrap CI. Supported only if p < 0.05 and the CI lower bound > 0.
H2 the initial 5m move reverses at 1h more often than chance. Primary: reversal rate above 50%. One-sided exact binomial test, alpha 0.05, plus 95% bootstrap CI for the rate.
H3 (surprise sign vs direction): NOT TESTABLE - no verified first-release data and no consensus. No proxy is substituted.
All other windows, ETH and other metrics are descriptive only. No multiplicity adjustment (two primary endpoints, pilot).

5. UNIT AND INFERENCE
Each FOMC statement is one independent cluster (events are at least 6 weeks apart; no window overlap). Bootstrap: resample events with replacement, 10000 draws, numpy default_rng seed 20261125.

6. POWER (exact, one-sided alpha 0.05)
Development n = 18 needs at least 13 of 18 (72%); power if the true rate is 0.6 / 0.7 / 0.8 / 0.9 = 21% / 53% / 87% / 99%.
Holdout n = 15 needs at least 12 of 15 (80%); power = 9% / 30% / 65% / 94%.
The pilot can only detect large effects; a non-significant result is not evidence of no effect.

7. DEVELOPMENT / HOLDOUT
Development = 2022-11-01 to 2024-12-31. Holdout = 2025-01-01 to 2026-10-31 (end 2026-10 because integrity-scanned monthly files end at 2026-10); it is locked: not downloaded and not analysed in the development run.
The holdout is run once, only after designer approval, with sections 2-5 unchanged. It supports a pattern only with the same direction and p < 0.05 under the same test.

8. KNOWN LIMITATIONS
Small n; baseline weekdays may contain other macro releases; statement (14:00 ET) and press conference (14:30 ET) are not separated; time is the page-stated time, not verified to the second; no consensus; descriptive only.
"""
DEV = ("2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 "
       "2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18").split()
MON = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13) if (y, m) >= (2022, 10)]   # 2022-10..2024-12 (2022-10 only feeds baselines)
K = (1, 5, 15, 60, 240); SYMS = ("BTCUSDT", "ETHUSDT"); SEED = 20261125; B = 10000
NY = ZoneInfo("America/New_York")

def t0ms(day):
    y, m, d = map(int, day.split("-")); return int(datetime(y, m, d, 14, 0, tzinfo=NY).astimezone(timezone.utc).timestamp() * 1000)

def load(sym, mons=MON):
    parts = []
    for m in mons:
        data, code = get(f"{BASE}/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip")
        if data is None: sys.exit(f"missing {sym} {m} ({code})")
        z = zipfile.ZipFile(io.BytesIO(data)); raw = z.read(z.namelist()[0]); hdr = raw[:12].lower().startswith(b"open_time")
        df = pd.read_csv(io.BytesIO(raw)) if hdr else pd.read_csv(io.BytesIO(raw), header=None); df.columns = COLS
        parts.append(df[["t", "o", "h", "l", "c", "v", "qv", "tbv"]])
    d = pd.concat(parts).drop_duplicates("t").sort_values("t"); T0 = int(d.t.iloc[0]); n = int((d.t.iloc[-1] - T0) // 60000) + 1
    A = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c", "v", "qv", "tbv")}; ix = ((d.t - T0) // 60000).astype(int).to_numpy()
    for k in A: A[k][ix] = d[k].to_numpy(float)
    return T0, A

def win(A, i, k):
    if i < 0 or i + k > len(A["o"]): return None
    s = slice(i, i + k)
    if any(np.isnan(A[x][s]).any() for x in ("o", "h", "l", "c", "v", "qv", "tbv")): return None
    v, tb = A["v"][s], A["tbv"][s]; ok = tb <= v * (1 + 1e-6) + 1e-9; p0 = A["o"][i]; pk = A["c"][i + k - 1]; vs = v[ok].sum()
    return dict(p0=p0, pk=pk, ret=math.log(pk / p0), rng=(A["h"][s].max() - A["l"][s].min()) / p0, qv=A["qv"][s].sum(),
                imb=(2 * tb[ok].sum() - vs) / vs if vs > 0 else np.nan, masked=int((~ok).sum()))

def base_times(t, evdays):
    out, d = [], 1
    while len(out) < 10:
        x = t - d * 86400000; ts = pd.Timestamp(x, unit="ms", tz="UTC")
        if ts.weekday() < 5 and ts.strftime("%Y-%m-%d") not in evdays: out.append(x)
        d += 1
    return out

def pbin(k, n): return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n

def ci(x, f):
    x = np.asarray(x, float); idx = np.random.default_rng(SEED).integers(0, len(x), (B, len(x))); return np.percentile(f(x[idx], axis=1), [2.5, 97.5])

def analyse(data):
    ev = [t0ms(d) for d in DEV]; evdays = {pd.Timestamp(t, unit="ms", tz="UTC").strftime("%Y-%m-%d") for t in ev}; R = {}; masked = {}
    for sym, (T0, A) in data.items():
        rows = {k: [] for k in K}; extra = []; me = mb = 0
        for t in ev:
            i0 = (t - T0) // 60000; bt = [(x - T0) // 60000 for x in base_times(t, evdays)]; w5, w60 = win(A, i0, 5), win(A, i0, 60)
            for k in K:
                e = win(A, i0, k); b = [w for w in (win(A, i, k) for i in bt) if w]
                if k == 240:
                    me += e["masked"] if e else 0; mb += sum(w["masked"] for w in b)
                if not e or len(b) < 5: rows[k].append(None); continue
                md = lambda f: float(np.median([w[f] for w in b]))
                rows[k].append(dict(lvr=math.log(e["rng"] / md("rng")), lqv=math.log(e["qv"] / md("qv")), dimb=e["imb"] - md("imb"), ar=abs(e["ret"]) * 1e4, ret=e["ret"] * 1e4))
            r5 = w5["pk"] / w5["p0"] - 1 if w5 else None
            extra.append(dict(r5=r5 * 1e4, c=np.sign(r5) * (w60["pk"] - w5["pk"]) / w5["p0"] * 1e4) if (w5 and w60) else None)
        R[sym] = (rows, extra); masked[sym] = (me, mb)
    for sym, (rows, extra) in R.items():
        print(f"\n== {sym}: per-window, over events with valid data (log VR = log(event/baseline median); share>0 = events with VR>1)")
        print("k(min) | n | median logVR range [95% CI] | share>0 | median logVR volume | mean imbalance diff | median |ret| bps")
        for k in K:
            r = [x for x in rows[k] if x]
            if len(r) < 2: print(k, "| n", len(r)); continue
            l = [x["lvr"] for x in r]; lo, hi = ci(l, np.median)
            print(f"{k} | {len(r)} | {np.median(l):+.3f} [{lo:+.3f}, {hi:+.3f}] | {sum(v > 0 for v in l)}/{len(l)} | {np.median([x['lqv'] for x in r]):+.3f} | {np.mean([x['dimb'] for x in r]):+.4f} | {np.median([x['ar'] for x in r]):.1f}")
        print(f"taker>volume rows masked: event windows (k=240) {masked[sym][0]}, baseline windows (k=240) {masked[sym][1]}")
    rows, extra = R["BTCUSDT"]; l = [x["lvr"] for x in rows[60] if x]; n = len(l); pos = sum(v > 0 for v in l); lo, hi = ci(l, np.median)
    p1 = pbin(pos, n); print(f"\n== H1 (BTC, k=60, primary): n {n}, events with VR>1: {pos}, median logVR {np.median(l):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], one-sided exact p {p1:.4f} -> {'SUPPORTED' if p1 < .05 and lo > 0 else 'NOT supported'} (n small; see power in SPEC)")
    cs = [x["c"] for x in extra if x and x["c"] != 0]; n2 = len(cs); rev = sum(v < 0 for v in cs); lo2, hi2 = ci([v < 0 for v in cs], np.mean); p2 = pbin(rev, n2)
    print(f"== H2 (BTC, primary): n {n2}, reversals {rev}, rate {rev / n2:.3f}, 95% CI [{lo2:.3f}, {hi2:.3f}], one-sided exact p {p2:.4f} -> {'SUPPORTED' if p2 < .05 else 'NOT supported'}; mean c {np.mean(cs):+.1f} bps")
    print("== H3: NOT TESTABLE (no verified first-release data, no consensus); no proxy used")
    print("\nBTC per event: date | t0 UTC | r5 bps | c bps | ret60 bps | logVR60")
    for d, t, x, w in zip(DEV, ev, extra, rows[60]):
        print(d, pd.Timestamp(t, unit="ms", tz="UTC").strftime("%H:%M"), *(("n/a",) * 2 if not x else (f"{x['r5']:+.1f}", f"{x['c']:+.1f}")), *(("n/a",) * 2 if not w else (f"{w['ret']:+.1f}", f"{w['lvr']:+.3f}")), sep=" | ")

def main():
    assert hashlib.sha256(SPEC.encode()).hexdigest() == SPEC_SHA256, "REGISTRATION MISMATCH: SPEC text changed"
    assert all(d in SPEC for d in DEV) and len(DEV) == 18 and max(DEV) <= "2024-12-31" and max(MON) == "2024-12"
    print("SPEC sha256", SPEC_SHA256, "(verified)\nscript sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    print("DEVELOPMENT ONLY: events 2022-11-02..2024-12-18; months loaded", MON[0], "..", MON[-1], "; holdout not loaded")
    analyse({s: load(s) for s in SYMS}); print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
