"""N6 funding/carry FEASIBILITY (descriptive, read-only, prints only). BTC and ETH, long spot + short USDT-M perpetual, 1x notional hedge, no leverage on the spot leg.
Data: Binance public archive (funding-rate history, spot 1h klines, USDT-M 1h klines), months 2020-01..2026-10. No fitting, no thresholds tuned, no holdout.
Fixed conventions: positions are opened/closed at the OPEN of an hourly candle; a funding payment is captured only if it falls strictly inside (entry, exit) (conservative: boundary payments excluded).
Costs per one-way (both legs): 20 bps base (spot 10 taker + perp 5 taker + ~5 slippage), 30 bps stress; round trip = 2 x. Scenarios (fixed, not optimised): HOLD = one position per calendar year; ROLL = re-opened every month (turnover case);
FILTER7 = in position only when the mean of the previous 21 funding payments (7 days) is > 0, decided at each payment time with earlier data only.
Predeclared GO criterion (evaluated once): for BOTH symbols, HOLD net at STRESS cost is > 0 in at least 4 of the 6 full years 2020-2025 AND the mean HOLD net over those years is >= 600 bps/yr of notional (about T-bill 4.5% / 0.75 capital factor at 3x perp leverage) AND ROLL net at stress cost has a positive mean. Otherwise STOP."""
import io, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from audit_a_n0 import BASE
from audit_a3_integrity import get

SPOT = "https://data.binance.vision/data/spot"
SYMS = ["BTCUSDT", "ETHUSDT"]; MON = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13) if (y, m) <= (2026, 10)]
H, C1 = 3600000, (20.0, 30.0)
ms = lambda y, m=1, d=1: int(pd.Timestamp(year=y, month=m, day=d, tz="UTC").timestamp() * 1000)

def raw(url):
    data, _ = get(url)
    if data is None: return None
    z = zipfile.ZipFile(io.BytesIO(data)); b = z.read(z.namelist()[0]); return b, b[:12].lower().startswith((b"open_time", b"calc_time"))

def kl(url):
    r = raw(url)
    if r is None: return {}
    df = pd.read_csv(io.BytesIO(r[0]), header=0 if r[1] else None); t = df.iloc[:, 0].astype("int64").to_numpy(); t = np.where(t > 10**14, t // 1000, t)
    return dict(zip(t.tolist(), df.iloc[:, 1].astype(float).tolist()))

def fund(url):
    r = raw(url)
    if r is None: return []
    df = pd.read_csv(io.BytesIO(r[0]), header=0 if r[1] else None); rate = df["last_funding_rate"] if "last_funding_rate" in df else df.iloc[:, -1]
    return list(zip((df.iloc[:, 0].astype("int64") // H * H).tolist(), rate.astype(float).tolist()))   # payment time rounded down to the hour

def load(sym):
    jobs = [(k, m) for m in MON for k in ("s", "p", "f")]
    def one(j):
        k, m = j
        return (kl(f"{SPOT}/monthly/klines/{sym}/1h/{sym}-1h-{m}.zip") if k == "s" else kl(f"{BASE}/monthly/klines/{sym}/1h/{sym}-1h-{m}.zip") if k == "p"
                else fund(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip"))
    with ThreadPoolExecutor(12) as ex: res = list(ex.map(one, jobs))
    S, Pp, F = {}, {}, []
    for (k, m), x in zip(jobs, res):
        if k == "s": S.update(x)
        elif k == "p": Pp.update(x)
        else: F += x
    miss = [m for (k, m), x in zip(jobs, res) if not x]; F = sorted(set(F))
    print(f"  {sym}: spot hours {len(S)}, perp hours {len(Pp)}, funding payments {len(F)}, months with a missing/empty file {len(miss)} {miss[:6]}")
    return S, Pp, np.array([t for t, _ in F]), np.array([r for _, r in F])

def seg(S, Pp, pt, pr, a, b):
    """funding (bps) + hedge P&L (bps of notional) of long spot / short perp held from hour-open a to hour-open b; None if a price is missing."""
    if a not in S or b not in S or a not in Pp or b not in Pp: return None
    f = pr[(pt > a) & (pt < b)].sum() * 1e4; h = ((S[b] / S[a] - 1) - (Pp[b] / Pp[a] - 1)) * 1e4
    return f, h

def scenarios(S, Pp, pt, pr, last):
    out = {"HOLD": [], "ROLL": [], "FILTER7": []}
    for y in range(2020, 2027):
        a, b = ms(y), min(ms(y + 1), last)
        r = seg(S, Pp, pt, pr, a, b)
        if r: out["HOLD"].append((y, *r))
        for m in range(1, 13):
            a, b = ms(y, m), min(ms(y + (m == 12), m % 12 + 1), last)
            if a < last and b <= last and (r := seg(S, Pp, pt, pr, a, b)): out["ROLL"].append((y, *r))
    on = [i >= 21 and pr[i - 21:i].mean() > 0 for i in range(len(pt))]; i = 21
    while i < len(pt) - 1:
        if on[i]:
            j = i
            while j + 1 < len(pt) - 1 and on[j + 1]: j += 1
            a, b = int(pt[i]) + H, int(pt[j + 1]) + H
            if b <= last and (r := seg(S, Pp, pt, pr, a, b)): out["FILTER7"].append((pd.Timestamp(a, unit="ms", tz="UTC").year, *r))
            i = j + 1
        else: i += 1
    return out

def main():
    print("N6 FUNDING/CARRY FEASIBILITY (descriptive; no outcome-based tuning); data months", MON[0], "..", MON[-1])
    verdict = {}
    for sym in SYMS:
        S, Pp, pt, pr = load(sym); last = ms(2026, 11); y = pd.to_datetime(pt, unit="ms", utc=True).year
        print(f"\n== {sym} funding payments by year: n | sum bps (=bps of notional earned by a short) | share positive | median payment bps")
        for yy in range(2020, 2027):
            k = y == yy; print(f"  {yy}: {k.sum()} | {pr[k].sum() * 1e4:+.0f} | {np.mean(pr[k] > 0):.2f} | {np.median(pr[k]) * 1e4:+.2f}")
        w = np.convolve(pr, np.ones(90), "valid") * 1e4; c = [pr[i:i + 90].sum() for i in range(0, len(pr) - 89, 90)]
        print(f"  rolling 30-day funding sum (bps): min {w.min():+.0f}, p10 {np.percentile(w, 10):+.0f}, median {np.median(w):+.0f}, p90 {np.percentile(w, 90):+.0f}, max {w.max():+.0f}; share of 30-day windows negative {np.mean(w < 0):.2f}; lag-1 corr of consecutive 30-day sums {np.corrcoef(c[:-1], c[1:])[0, 1]:+.2f}")
        hrs = sorted(Pp); pa = np.array([Pp[h] for h in hrs]); n30 = 720
        rise = pa[n30:] / pa[:-n30] - 1; day = pa[24:] / pa[:-24] - 1
        print(f"  perp price risk (short leg): worst 30-day rise {rise.max() * 100:+.0f}%, share of 30-day windows with rise >= 33% (1/3 margin buffer at 3x) {np.mean(rise >= 0.33):.3f}, worst 1-day rise {day.max() * 100:+.1f}%")
        sc = scenarios(S, Pp, pt, pr, last); df = {k: pd.DataFrame(v, columns=["y", "f", "h"]) for k, v in sc.items()}
        print(f"  scenarios per year, bps of notional: segments | funding | hedge(basis) | net@base cost (20 bps one-way) | net@stress (30 bps one-way)")
        for k, d in df.items():
            if k == "FILTER7":
                n, yrs = len(d), (last - ms(2020)) / (365.25 * 86400000); fs, hs = d.f.sum(), d.h.sum()
                print(f"  FILTER7 (whole period, {yrs:.1f} yrs): {n} segments | funding {fs:+.0f} | hedge {hs:+.0f} | net@base {fs + hs - n * 2 * C1[0]:+.0f} ({(fs + hs - n * 2 * C1[0]) / yrs:+.0f}/yr) | net@stress {fs + hs - n * 2 * C1[1]:+.0f} ({(fs + hs - n * 2 * C1[1]) / yrs:+.0f}/yr)"); continue
            print(f"  {k}:")
            for yy, g in d.groupby("y"):
                n = len(g); print(f"    {yy}: {n} | {g.f.sum():+.0f} | {g.h.sum():+.0f} | {g.f.sum() + g.h.sum() - n * 2 * C1[0]:+.0f} | {g.f.sum() + g.h.sum() - n * 2 * C1[1]:+.0f}")
        hold = df["HOLD"]; hn = {yy: g.f.sum() + g.h.sum() - 2 * C1[1] for yy, g in hold.groupby("y") if yy <= 2025}; roll = df["ROLL"]
        rn = {yy: g.f.sum() + g.h.sum() - len(g) * 2 * C1[1] for yy, g in roll.groupby("y") if yy <= 2025}
        verdict[sym] = (sum(v > 0 for v in hn.values()) >= 4, np.mean(list(hn.values())) >= 600, np.mean(list(rn.values())) > 0, hn, rn)
    print("\nPREDECLARED GO CRITERION (stress cost):")
    for sym, (a, b, c, hn, rn) in verdict.items():
        print(f"  {sym}: HOLD net>0 in >=4 of 6 years {'PASS' if a else 'FAIL'} ({sum(v > 0 for v in hn.values())}/6); mean HOLD net >= 600 bps/yr {'PASS' if b else 'FAIL'} ({np.mean(list(hn.values())):+.0f}); ROLL mean net >0 {'PASS' if c else 'FAIL'} ({np.mean(list(rn.values())):+.0f})")
    print("RECOMMENDATION:", "GO - further research justified" if all(a and b and c for a, b, c, _, _ in verdict.values()) else "STOP - opportunity not promising under the predeclared criterion")
    print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
