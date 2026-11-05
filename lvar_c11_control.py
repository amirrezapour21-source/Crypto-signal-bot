"""
POSITIVE CONTROL - does our backtest pipeline reproduce a strategy that is KNOWN to look good?
Candidate 11 (the frozen short strategy that is running as a paper test) is replayed through the SAME pipeline used for the box and
support/resistance tests: same Binance USDT-M 4h data, same liquid universe, same exit simulator (lvar_fut_backtest.sim), same costs and
the REAL funding. Nothing is tuned; the rules are copied from crypto_signal_bot_v4.py.

Candidate 11 rules (4h, SHORT only):
  Z20 = (close - mean20(close)) / std20(close) (sample std). Trigger: previous Z20 >= +2 and Z20 < +2 on the signal bar (cross down).
  Close < EMA200 (ewm span 200, adjust=False). RANGE20 (max high - min low of 20 bars) >= its 20-bar median. Volume >= 20-bar median volume.
  Entry = open of the next bar. SL = 1.25 x ATR20 (ewm alpha 1/20). TP = 2R. Time stop = close of the 30th bar.

How to read the result: if this pipeline gives a clearly positive net result for Candidate 11 (as its own earlier backtest did), the pipeline
and cost model are able to see an edge, and the rejections of the other strategies are real. If it gives a clearly negative result, the
pipeline (data, costs, exit rules) is harsher than the earlier C11 backtest and must be examined before trusting any rejection.
Pass rule used for the other tests is printed for reference only (net avgR >= +0.10, PF >= 1.2, n >= 300, both halves positive).
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import lvar_box_backtest as X
import lvar_daily_backtest as B
import lvar_fut_backtest as F

TF = X.TF
DAY = X.DAY
Z_N, RANGE_N, VOL_N, EMA_N, ATR_N = 20, 20, 20, 200, 20
SL_ATR, RR, HOLD = 1.25, 2.0, 30
RISK_LO, RISK_HI = 0.01, 0.12
MIN_VOL30 = X.MIN_VOL30
COST_RT, COST_STRESS, COST_C11 = 0.0020, 0.0040, 0.0030


def parse_k(rows):
    """(time, open, high, low, close, base_volume, quote_volume)."""
    out = []
    for r in rows:
        try:
            out.append((int(F.norm_t(r[0])), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7])))
        except (ValueError, IndexError):
            continue
    return out


def job(args):
    kind, sym, url = args
    c = F.fetch(url)
    if c is None:
        return kind, sym, None
    try:
        rows = F.read_zip_rows(c)
        return kind, sym, (parse_k(rows) if kind == "k4" else F.parse_funding(rows))
    except Exception:
        return kind, sym, None


def build_frame(kl):
    df = pd.DataFrame(kl, columns=["time", "open", "high", "low", "close", "volume", "qv"])
    df = df.drop_duplicates("time").sort_values("time")
    return df[(df.open > 0) & (df.close > 0) & (df.high >= df.low) & (df.qv >= 0) & (df.volume >= 0)].reset_index(drop=True)


def prepare(df):
    """the exact indicator formulas of crypto_signal_bot_v4.py"""
    d = df.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.ewm(alpha=1 / ATR_N, adjust=False).mean()
    mean20, std20 = d["close"].rolling(Z_N).mean(), d["close"].rolling(Z_N).std()
    d["z20"] = (d["close"] - mean20) / std20.replace(0, np.nan)
    d["ema200"] = d["close"].ewm(span=EMA_N, adjust=False).mean()
    d["range20"] = d["high"].rolling(RANGE_N).max() - d["low"].rolling(RANGE_N).min()
    d["range_median20"] = d["range20"].rolling(RANGE_N).median()
    d["vol_median20"] = d["volume"].rolling(VOL_N).median()
    d["dollar_vol"] = d["qv"].rolling(X.VOL_BARS, min_periods=X.VOL_BARS).mean() * X.BPD
    return d


def signal_mask(d):
    z, zp = d["z20"], d["z20"].shift(1)
    ok = (zp >= 2.0) & (z < 2.0) & (d["close"] < d["ema200"]) & (d["range20"] >= d["range_median20"]) & (d["volume"] >= d["vol_median20"])
    return (ok & d["atr20"].gt(0)).fillna(False).to_numpy(bool)


def make_candidates(sym, df, ft, fr, liquid_filter=True):
    n = len(df)
    if n < 400:
        return []
    d = prepare(df)
    sig = signal_mask(d)
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = d["time"].to_numpy(np.int64)
    atr, dv = d["atr20"].to_numpy(float), d["dollar_vol"].to_numpy(float)
    out = []
    for i in np.nonzero(sig)[0]:
        k = i + 1
        if k >= n - 1 or t[k] - t[i] != TF or t[i] - t[i - Z_N] != Z_N * TF:
            continue
        if liquid_filter and not (np.isfinite(dv[i]) and dv[i] >= MIN_VOL30):
            continue
        side = -1
        entry = o[k]
        risk = SL_ATR * atr[i]
        stop, target = entry + risk, entry - RR * risk
        risk_pct = risk / entry if entry > 0 else 0.0
        if risk <= 0 or not (RISK_LO <= risk_pct <= RISK_HI) or target <= 0:
            continue
        res = F.sim(o, h, l, c, k, side, entry, stop, target, HOLD)
        if res is None:
            continue
        fsum = X.funding_sum4(ft, fr, t[k], t[res[1]])
        out.append({"pair": sym, "side": "SHORT", "entry_t": int(t[k]), "exit_t": int(t[res[1]]), "reason": res[2],
                    "days": float((res[1] - k + 1) * TF / DAY), "risk_pct": float(risk_pct), "r_gross": float(res[0]),
                    "r_net": float(res[0] - (COST_RT + side * fsum) / risk_pct),
                    "r_stress": float(res[0] - (COST_STRESS + side * fsum) / risk_pct),
                    "r_c11": float(res[0] - (COST_C11 + side * fsum) / risk_pct),
                    "strength": float(d["z20"].iloc[i - 1]), "funding_sum": fsum, "touches": 0})
    return out


def main():
    t_run = time.time()
    months = F.month_list()
    dl_months = ["2022-10"] + months
    now = datetime.now(timezone.utc)
    end_t = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp())
    start_t = int(datetime(F.START_Y, F.START_M, 1, tzinfo=timezone.utc).timestamp())
    listed = F.list_symbols()
    symbols = [s for s in (listed or F.FALLBACK) if F.usable_symbol(s) and not X.non_crypto(s)]
    selected, _ = F.rank_symbols(symbols)
    L = ["POSITIVE CONTROL - Candidate 11 (short, 4h, z-score cross-down) replayed through the same pipeline as the other tests",
         f"run at: {now.strftime('%Y-%m-%d %H:%M UTC')} | period: {datetime.fromtimestamp(start_t, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(end_t, timezone.utc):%Y-%m-%d}",
         f"symbols: {len(selected)} liquid Binance USDT-M perpetuals (not the 33 KuCoin symbols of the paper test)",
         "costs: fee+slippage 0.20% round trip (stress 0.40%) plus REAL funding; the 0.30% line is the cost used in the paper-test report",
         "Frozen rules copied from crypto_signal_bot_v4.py; no parameter changed."]
    if len(selected) < 20:
        L.append("ABORTED: too few symbols could be downloaded.")
        text = "\n".join(L) + "\n"
        print(text)
        open("lvar_c11_result.txt", "w", encoding="utf-8").write(text)
        return
    tasks = []
    for s in selected:
        for m in dl_months:
            tasks.append(("k4", s, f"{F.BASE}/monthly/klines/{s}/4h/{s}-4h-{m}.zip"))
            tasks.append(("fu", s, f"{F.BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"))
    kl, fu = {s: [] for s in selected}, {s: [] for s in selected}
    with ThreadPoolExecutor(F.WORKERS) as ex:
        for kind, s, data in ex.map(job, tasks):
            if data:
                (kl if kind == "k4" else fu)[s].extend(data)
    cands, used = [], 0
    for s in selected:
        if not kl[s] or not fu[s]:
            continue
        df = build_frame(kl[s])
        df = df[df["time"] < end_t].reset_index(drop=True)
        if len(df) < 600:
            continue
        fs = sorted(fu[s])
        ft, fr = np.array([x[0] for x in fs], float), np.array([x[1] for x in fs], float)
        used += 1
        cands += [x for x in make_candidates(s, df, ft, fr) if x["entry_t"] >= start_t]
    L.append(f"symbols with usable data: {used}")
    L.append("")
    cap = B.portfolio(cands)
    L.append(f"signals before the position cap: {len(cands)} | taken (max 8 open, one per coin): {len(cap)}")
    L += B.block("CANDIDATE 11 through our pipeline", cap, False)
    if cands:
        for key, label in (("r_gross", "gross"), ("r_net", "net 0.20%+funding"), ("r_c11", "net 0.30%+funding (paper-test cost)")):
            v = np.array([x[key] for x in cands])
            L.append(f"  ALL signals (no cap) {label}: n={len(v)} avgR={v.mean():+.3f}")
    L.append("")
    L.append("Reference from the earlier C11 backtest (KuCoin, 33 symbols, its own period): TP2R n=580 avgR +0.1987; with 0.30% cost +0.1305.")
    L.append(f"total run time: {time.time() - t_run:.0f}s")
    text = "\n".join(L) + "\n"
    print(text)
    with open("lvar_c11_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open("lvar_c11_trades.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["pair", "side", "entry_t", "exit_t", "reason", "days", "risk_pct", "r_gross", "r_net", "r_c11", "funding_sum"]
        w.writerow(cols)
        for x in cap:
            w.writerow([x[k] for k in cols])


# ---------------------------------------------------------------- self-test
PASSED = 0
T0 = 1_650_000_000 // DAY * DAY


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def rand_frame(seed, n=5000):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(-0.0005, 0.012, n)))
    o = np.concatenate([[100.0], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.004, n)))
    return pd.DataFrame({"time": T0 + np.arange(n) * TF, "open": o, "high": hi, "low": lo, "close": c, "volume": r.lognormal(0, 0.5, n) * 100, "qv": 5e7})


def naive_signals(df):
    """plain-loop reading of the C11 rules (independent of the pandas vectorised mask)"""
    c, hi, lo, v = (df[k].tolist() for k in ("close", "high", "low", "volume"))
    n = len(c)
    alpha = 2.0 / (EMA_N + 1)
    ema = [c[0]]
    for x in c[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    z = [None] * n
    for i in range(Z_N - 1, n):
        w = c[i - Z_N + 1:i + 1]
        m = sum(w) / Z_N
        sd = (sum((y - m) ** 2 for y in w) / (Z_N - 1)) ** 0.5
        z[i] = (c[i] - m) / sd if sd > 0 else None
    rng = [None] * n
    for i in range(RANGE_N - 1, n):
        rng[i] = max(hi[i - RANGE_N + 1:i + 1]) - min(lo[i - RANGE_N + 1:i + 1])
    out = []
    for i in range(1, n):
        if z[i] is None or z[i - 1] is None or i < 2 * RANGE_N or rng[i] is None:
            continue
        rm = sorted(rng[i - RANGE_N + 1:i + 1])
        rmed = (rm[9] + rm[10]) / 2
        vm = sorted(v[i - VOL_N + 1:i + 1])
        vmed = (vm[9] + vm[10]) / 2
        if z[i - 1] >= 2.0 and z[i] < 2.0 and c[i] < ema[i] and rng[i] >= rmed and v[i] >= vmed:
            out.append(i)
    return out


def run_selftest():
    for sd in (1, 2, 3, 4):
        df = rand_frame(sd)
        d = prepare(df)
        got = list(np.nonzero(signal_mask(d))[0])
        ref = naive_signals(df)
        check(len(ref) >= 5 and got == ref, f"seed {sd}: pandas signals equal the naive plain-loop signals ({len(ref)})")
    df = rand_frame(9)
    ft, fr = np.array([float(x) for x in df["time"] + 3600]), np.full(len(df), 0.0001)
    cands = make_candidates("R", df, ft, fr)
    check(len(cands) >= 5 and all(x["side"] == "SHORT" for x in cands), f"short candidates produced ({len(cands)})")
    d = prepare(df)
    x = cands[0]
    i = int((df["time"] == x["entry_t"]).idxmax()) - 1
    atr, entry = d["atr20"].iloc[i], df["open"].iloc[i + 1]
    check(abs(x["risk_pct"] - SL_ATR * atr / entry) < 1e-12, "stop distance = 1.25 x ATR20 of the signal bar, entry = next open")
    check(all((q["reason"] != "TP" or abs(q["r_gross"] - 2.0) < 1e-9) and (q["reason"] != "SL" or q["r_gross"] <= -1.0 + 1e-9) for q in cands), "TP = +2R, SL <= -1R (gaps can be worse)")
    for q in cands:
        sg = -1
        check(abs(q["r_net"] - (q["r_gross"] - (COST_RT + sg * q["funding_sum"]) / q["risk_pct"])) < 1e-9 and q["r_c11"] < q["r_net"], "net R formula, 0.30% line below the 0.20% line")
    cut = 3500
    fut = df.copy()
    fut.loc[cut:, ["open", "high", "low", "close"]] *= 2.0
    tc = int(df["time"].iloc[cut])
    a1 = {(q["entry_t"]): q for q in cands if q["exit_t"] < tc}
    a2 = {(q["entry_t"]): q for q in make_candidates("R", fut, ft, fr) if q["exit_t"] < tc}
    check(len(a1) >= 3 and a1.keys() == a2.keys() and all(abs(a1[k]["r_gross"] - a2[k]["r_gross"]) < 1e-12 for k in a1), "future prices do not change earlier trades")
    check(make_candidates("R", df.assign(qv=1e5), ft, fr) == [], "liquidity filter removes illiquid coins")
    print(f"ALL C11 CONTROL SELF-TESTS PASSED ({PASSED} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        run_selftest()
    else:
        main()
