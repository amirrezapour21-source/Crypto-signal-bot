"""N4-SD supply/demand zone study. DEVELOPMENT run (entries 2022-11-01..2024-12-31). Read-only; prints only. Holdout months are not loaded here.
The SPEC text is checked against its hash before any data is loaded."""
import hashlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import n1_fomc_dev as n1, n2_exec_edge_dev as n2

SPEC_SHA256 = "6fd4e31505fb1df59d4c90b48f61fdc1486e7eb49d38bc5db2b88ffa51f461b2"
SPEC = r"""N4-SD SUPPLY/DEMAND ZONE STUDY - PRE-REGISTRATION (version 1)

QUESTION
Does a simple, causal supply/demand-zone retest-rejection rule on 4h bars of liquid USDT-M perpetuals produce a positive mean net return per trade after a 15-minute execution delay, a 30 bps round-trip cost, stops and time exits?

1. DATA AND SAMPLE
Binance USDT-M 1m klines (data.binance.vision monthly files), symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT. 4h bars are built from 1m candles on UTC 4h boundaries (O first open, H max, L min, C last close). A bar with any missing 1m candle is invalid (no zone, no signal, ignored).
Development: trade entries 2022-11-01 .. 2024-12-31 (months 2022-10 .. 2024-12 loaded; 2022-10 is warm-up only).
Holdout: trade entries 2025-01-01 .. 2026-10-31 (single use, locked; opened only through the gate in section 6). No parameter is chosen from any outcome.

2. ZONES (causal)
ATR = mean of (H-L) over the last 14 bars including the current bar.
Pivot high at bar j: H_j > max(H of bars j-3..j-1) and H_j >= max(H of bars j+1..j+3). Pivot low symmetric with L. A pivot is known only at the close of bar j+3 and its zone is active from bar j+4.
Supply zone (short side): near edge = max(O_j, C_j), far edge = H_j. Demand zone (long side): near edge = min(O_j, C_j), far edge = L_j.
A zone is removed when (a) a bar closes beyond its far edge, (b) it is more than 30 bars old, or (c) it is touched (supply: bar H >= near edge; demand: bar L <= near edge). Each zone can therefore produce at most one signal, on its first touch.

3. SIGNAL AND ENTRY
Signal at the close of the touching bar if the bar closes back on the outside of the near edge (supply: C < near edge; demand: C > near edge) = rejection.
Entry: market order at the open of the 1m candle 15 minutes after that bar close (stress D = 30 min).
Skipped if: a position in the same symbol is still open; the stop distance (entry to stop) is below 1.0% of entry (economic rule: round-trip cost 0.30% must not exceed about one third of the risk); the stop is not on the adverse side of entry; any 1m candle in the trade window is missing; the trade would end after the loaded data.
News filter: no entry if any scheduled CPI, NFP or FOMC release (calendar of the N2-E registration, SPEC sha256 05954b52b57df2e28a0952b91ac9fa4958bc2a68f5af33571b7e8f9f56d8ad1c) lies within 120 minutes before to 240 minutes after the entry time. Positions are not closed around news.

4. EXITS AND COSTS
Stop: far edge of the zone moved 0.25 x ATR further away from the zone (below the demand low, above the supply high). R = stop distance. Target: 2 R from the actual entry price. Time exit: close of the 1m candle 12 bars (2880 minutes) after entry.
Checked on every 1m candle from the entry candle; if stop and target are both inside one candle the stop is taken; stop fill = stop price, or the candle open if it is already beyond the stop; target fill = target price.
Return per trade = direction x (exit/entry - 1) - cost; cost 30 bps round trip (base) or 40 bps (stress), deducted once. One trade per zone; fixed notional per trade.

5. HYPOTHESIS AND TEST
H-SD: mean net return per trade (30 bps, all symbols, long and short pooled) > 0.
Test: bootstrap of the mean with entry-day clusters (resample entry days with replacement), 10,000 draws, numpy default_rng seed 20261125; supported if the 2.5th percentile of the mean net return is > 0.
Development gate: the holdout is run only if development n >= 100 trades and the development mean net return (30 bps) > 0. Single holdout run, rules unchanged: confirmed only if the holdout 2.5th percentile > 0; an executable-edge statement additionally needs holdout mean net > 0 at 40 bps and at D = 30. Development and holdout are never pooled; nothing is changed after outcomes are seen.

6. DESCRIPTIVE (no selection)
Per symbol; long vs supply/demand side; exit type shares (stop/target/time); mean stop distance in bps and cost as a share of R; trades with a news release inside the holding period versus without; D = 30 and cost 40 bps.

7. LIMITS
Zones and costs are assumptions; no spread, depth or queue data; no stop slippage model beyond the open-gap rule; about 8 symbols x 26 months, trades are not fully independent across symbols (day clusters used); no consensus or first-release news data.
"""
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]
MON = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13) if (y, m) >= (2022, 10)]
LO, HI = 1667260800000, 1735689600000          # entries 2022-11-01 .. 2024-12-31 (UTC ms)
MINR, TGT, BUF, MAXB, MAXZ, B = 0.01, 2.0, 0.25, 12, 30, 10000

def trades(sym, T0, A, ev, delay):
    n = len(A["o"]); s0 = int(-(T0 // 60000) % 240); nb = (n - s0) // 240; sl = slice(s0, s0 + 240 * nb)
    R = {k: A[k][sl].reshape(nb, 240) for k in "ohlc"}; O, H, L, C = R["o"][:, 0], R["h"].max(1), R["l"].min(1), R["c"][:, -1]
    atr = pd.Series(H - L).rolling(14).mean().to_numpy(); Z, out, free = [], [], 0
    for k in range(nb):
        if np.isnan([O[k], H[k], L[k], C[k], atr[k]]).any(): continue
        keep = []
        for d, near, far, b in Z:                                   # d = +1 demand (long), -1 supply (short)
            ext = L[k] if d > 0 else H[k]
            if d * (far - C[k]) > 0 or k - b > MAXZ: continue        # closed beyond far edge / too old
            if d * (near - ext) < 0: keep.append((d, near, far, b)); continue   # not touched
            if d * (C[k] - near) > 0:                                # touched and rejected: signal
                ie = s0 + 240 * (k + 1) + delay; ix = ie + MAXB * 240 - 1
                if ix >= n or ie < free: continue
                t = T0 + ie * 60000
                if not LO <= t <= HI or any(t - 7200000 <= e <= t + 14400000 for e in ev): continue
                o, h, l, c = (A[x][ie:ix + 1] for x in "ohlc")
                if np.isnan([o, h, l, c]).any(): continue
                pe = o[0]; stop = far - d * BUF * atr[k]; dist = d * (pe - stop)
                if dist <= 0 or dist / pe < MINR: continue
                tg = pe + d * TGT * dist; hs, ht = (l <= stop, h >= tg) if d > 0 else (h >= stop, l <= tg)
                js, jt = (int(np.argmax(hs)) if hs.any() else 10**9), (int(np.argmax(ht)) if ht.any() else 10**9)
                if js <= jt and js < 10**9: px, j, kind = (min(stop, o[js]) if d > 0 else max(stop, o[js])), js, "stop"
                elif jt < 10**9: px, j, kind = tg, jt, "target"
                else: px, j, kind = c[-1], len(c) - 1, "time"
                free = ie + j + 1
                out.append(dict(sym=sym, d=d, t=t, day=t // 86400000, g=d * (px / pe - 1), r=dist / pe, kind=kind,
                                news=any(t < e <= t + (j + 1) * 60000 for e in ev)))
        Z = keep; j = k - 3
        if j >= 3 and not np.isnan([H[j - 3:j + 4], L[j - 3:j + 4]]).any():
            if H[j] > H[j - 3:j].max() and H[j] >= H[j + 1:j + 4].max(): Z.append((-1, max(O[j], C[j]), H[j], k))
            if L[j] < L[j - 3:j].min() and L[j] <= L[j + 1:j + 4].min(): Z.append((1, min(O[j], C[j]), L[j], k))
    return out

def cl(x, days):
    """mean and day-cluster bootstrap 95% interval (10,000 draws, seed 20261125)."""
    _, c = np.unique(days, return_inverse=True); S = np.bincount(c, x); N = np.bincount(c).astype(float)
    i = np.random.default_rng(20261125).integers(0, len(S), (B, len(S))); bm = S[i].sum(1) / N[i].sum(1)
    return x.mean(), *np.percentile(bm, [2.5, 97.5])

def line(label, df, cost):
    if len(df) < 2: print(f"{label}: n {len(df)}"); return
    m, lo, hi = cl(df.g.to_numpy() - cost, df.day.to_numpy())
    print(f"{label}: n {len(df)}, net@{int(cost * 1e4)} mean {m * 1e4:+.1f} bps [{lo * 1e4:+.1f}, {hi * 1e4:+.1f}], sd {df.g.std() * 1e4:.0f}, win {np.mean(df.g - cost > 0):.2f}, "
          f"gross {df.g.mean() * 1e4:+.1f}, stop/target/time {np.mean(df.kind == 'stop'):.2f}/{np.mean(df.kind == 'target'):.2f}/{np.mean(df.kind == 'time'):.2f}, mean R {df.r.mean() * 1e4:.0f} bps")

def main():
    assert hashlib.sha256(SPEC.encode()).hexdigest() == SPEC_SHA256, "REGISTRATION MISMATCH: SPEC text changed"
    print("SPEC sha256", SPEC_SHA256, "(verified)\nscript sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    print(f"DEVELOPMENT ONLY: entries 2022-11-01..2024-12-31; months loaded {MON[0]}..{MON[-1]}; holdout not loaded")
    ev = [e[2] for e in n2.events()]; D15, D30 = [], []
    for s in SYMS:
        T0, A = n1.load(s, MON); D15 += trades(s, T0, A, ev, 15); D30 += trades(s, T0, A, ev, 30)
    d15, d30 = pd.DataFrame(D15), pd.DataFrame(D30)
    print("\n== CONFIRMATORY (development): H-SD pooled, D15, cost 30 bps, day-cluster bootstrap")
    line("H-SD", d15, 0.003); m, lo, _ = cl(d15.g.to_numpy() - 0.003, d15.day.to_numpy())
    print(f"   2.5th pct of mean net@30 = {lo * 1e4:+.1f} bps -> {'supported' if lo > 0 else 'not supported'}; HOLDOUT GATE (n>=100 and mean net@30>0): {'PASS' if len(d15) >= 100 and m > 0 else 'FAIL - holdout not to be run'}")
    print("\n== DESCRIPTIVE")
    line("stress cost 40", d15, 0.004); line("stress D30 cost 30", d30, 0.003); line("stress D30 cost 40", d30, 0.004)
    for s in SYMS: line(s, d15[d15.sym == s], 0.003)
    for d, nm in ((1, "long (demand)"), (-1, "short (supply)")): line(nm, d15[d15.d == d], 0.003)
    for f, nm in ((True, "news inside holding period"), (False, "no news inside holding period")): line(nm, d15[d15.news == f], 0.003)
    print(f"cost as share of R: mean {0.003 / d15.r.mean():.2f} (30 bps / mean R)")
    print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
