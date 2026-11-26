"""N3 microstructure DISCOVERY scan - exploratory, read-only, prints only. Development months 2022-10..2024-12 only (2025+ never loaded).
Hourly bars built from Binance USDT-M 1m klines. Entry = open of the 1m candle 15 min after bar close, exit H minutes later (no stop); costs not deducted here:
an executable candidate needs |mean| > 30 bps (round-trip cost assumption) with the whole day-cluster bootstrap CI beyond 30 bps in the same direction."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import n1_fomc_dev as n1

MON = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13) if (y, m) >= (2022, 10)]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]
DELAY, HS, COST, B = 15, (60, 240), 30.0, 2000

def bars(sym):
    T0, A = n1.load(sym, MON); n = len(A["o"])
    ok = A["tbv"] <= A["v"] * (1 + 1e-6) + 1e-9; masked = int((~ok & ~np.isnan(A["tbv"])).sum()); A["tbv"] = np.where(ok, A["tbv"], np.nan)
    s0 = int(-(T0 // 60000) % 60); nb = (n - s0) // 60; sl = slice(s0, s0 + 60 * nb)
    R = {k: A[k][sl].reshape(nb, 60) for k in ("o", "h", "l", "c", "v", "qv", "tbv")}
    d = pd.DataFrame(dict(O=R["o"][:, 0], H=R["h"].max(1), L=R["l"].min(1), C=R["c"][:, -1], QV=R["qv"].sum(1), V=R["v"].sum(1), TB=R["tbv"].sum(1)))
    d["ts"] = T0 + (s0 + 60 * np.arange(nb)) * 60000; d = d.set_index("ts")
    d["imb"] = (2 * d.TB - d.V) / d.V; d["ret1"] = np.log(d.C / d.O)
    d["atr"] = ((d.H - d.L) / d.O).rolling(24).median().shift(1); d["prog"] = d.ret1.abs() / d.atr
    d["vr"] = d.QV / d.QV.rolling(168).median().shift(1); d["imb3"] = d.imb.rolling(3).mean()
    d["same3"] = (np.sign(d.imb).rolling(3).sum().abs() == 3); d["ret4"] = np.log(d.C / d.O.shift(3)); d["ret24"] = np.log(d.C / d.O.shift(23))
    hh, ll = d.H.rolling(24).max().shift(1), d.L.rolling(24).min().shift(1)
    d["up_rej"] = (d.H > hh) & (d.C < hh); d["dn_rej"] = (d.L < ll) & (d.C > ll); d["up_acc"] = d.C > hh; d["dn_acc"] = d.C < ll
    for H in HS:
        ie = s0 + 60 * (np.arange(nb) + 1) + DELAY; ix = ie + H - 1; f = np.full(nb, np.nan); g = ix < n
        f[g] = A["c"][ix[g]] / A["o"][ie[g]] - 1; d[f"f{H}"] = f * 1e4
    d["sym"] = sym; d["day"] = d.index // 86400000
    cov = f"{sym}: 1m rows {int((~np.isnan(A['c'])).sum())}/{n}, hourly bars valid {int(d.imb.notna().sum())}/{nb}, taker>vol masked {masked}"
    return d[d.index >= 1667260800000], cov   # signals from 2022-11-01 00:00Z

def stat(label, ds, df, H):
    f = df[f"f{H}"] * ds; m = f.notna() & (ds != 0); f, day = f[m].to_numpy(), df.day[m].to_numpy()
    if len(f) < 100: print(f"{label:34s} H{H:<3d} n {len(f)} (too few)"); return
    _, c = np.unique(day, return_inverse=True); S = np.bincount(c, f); N = np.bincount(c).astype(float)
    i = np.random.default_rng(20261126).integers(0, len(S), (B, len(S))); bm = S[i].sum(1) / N[i].sum(1); lo, hi = np.percentile(bm, [2.5, 97.5]); mu = f.mean()
    flag = "  <== EXEC-CANDIDATE" if (lo > COST or hi < -COST) else ""
    print(f"{label:34s} H{H:<3d} n {len(f):6d} days {len(S):3d} | cont mean {mu:+6.1f} [{lo:+6.1f},{hi:+6.1f}] bps, hit {np.mean(f > 0):.3f}, sd {f.std():.0f} | best-direction net@30 {abs(mu) - COST:+6.1f}{flag}")

def main():
    out = [bars(s) for s in SYMS]; print("DATA AVAILABILITY (klines only; spread, depth, order book, trade-level data NOT measurable -> costs are assumptions)")
    for _, c in out: print(" ", c)
    df = pd.concat([o[0] for o in out]); b = out[0][0]; df["btc24"] = df.index.map(b.ret24.to_dict()); df = df.reset_index(drop=True)
    df["rs"] = np.where(df.sym != "BTCUSDT", df.ret24 - df.btc24, np.nan); q = lambda x, p=80: np.nanpercentile(x, p)
    print(f"\nDISCOVERY (exploratory; {len(df)} symbol-hours, 8 symbols, 2022-11..2024-12; thresholds = pooled in-sample 80th pct, 33rd for prog)")
    print("signal 'ds' = side of the feature; 'cont' = trade in ds direction, reversal = negative of the mean. EXEC-CANDIDATE = day-cluster 95% CI beyond +-30 bps.")
    sg = np.sign
    ai, a3, rv = df.imb.abs(), df.imb3.abs(), df.ret4.abs()
    S = {
      "F1 imb1 top20%": (sg(df.imb), ai >= q(ai)),
      "F2 imb3 persistent top20%": (sg(df.imb3), (a3 >= q(a3)) & df.same3),
      "F3 absorption (vr top20, prog low33)": (sg(df.imb), (df.vr >= q(df.vr)) & (df.prog <= q(df.prog, 33))),
      "F4 sweep-reject (dir=swept side)": (np.where(df.up_rej & ~df.dn_rej, 1, np.where(df.dn_rej & ~df.up_rej, -1, 0)), None),
      "F5 sweep-accept (dir=broken side)": (np.where(df.up_acc, 1, np.where(df.dn_acc, -1, 0)), None),
      "F6 mom4 top20%": (sg(df.ret4), rv >= q(rv)),
      "F7 mom4 exhaustion (imb1 opposite)": (sg(df.ret4), (rv >= q(rv)) & (sg(df.imb) != sg(df.ret4))),
      "F8 rel strength vs BTC top20% (alts)": (sg(df.rs), df.rs.abs() >= q(df.rs.abs())),
      "F8b same, BTC 24h down": (sg(df.rs), (df.rs.abs() >= q(df.rs.abs())) & (df.btc24 < 0)),
    }
    for H in HS: stat("unconditional long", pd.Series(1.0, df.index), df, H)
    for k, (ds, m) in S.items():
        ds = pd.Series(np.asarray(ds, float), index=df.index)
        if m is not None: ds = ds.where(np.asarray(m), 0.0)
        for H in HS: stat(k, ds, df, H)
    print("\nNothing was written to the repository.")

if __name__ == "__main__": main()
