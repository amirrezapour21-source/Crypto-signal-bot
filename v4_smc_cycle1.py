#!/usr/bin/env python3
"""
SINGLE FILE: python v4_smc_cycle1.py selftest   -> offline synthetic self-test
             python v4_smc_cycle1.py            -> real Cycle 1 run
V4 SMC - CYCLE 1 (detection / counting / causality ONLY)
Frozen spec: Spec Freeze v1 + Patches 1-4.  No P&L, no R, no TP/exit tracking, no costs,
no OOS / Holdout.  Fully isolated from Candidate 11 (own files, own outputs, prefix v4_smc_).

Run:  python v4_smc_cycle1.py
Exit codes: 0 = validation PASS and causality PASS; 1 = data validation FAIL (stopped);
            2 = causality / provenance FAIL (stopped);  3 = fetch / config error.
"""
import os, sys, io, json, gzip, hashlib, time, math, random, types
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# FROZEN PARAMETERS (Spec Freeze v1 + Patches 1-4).  Never tuned.
# ----------------------------------------------------------------------------
BAR = 14400                       # 4H in seconds
P = dict(
    ATR_N=14, PIV=3, SMT_LB=20, SMT_PM=2, BOS_FIRST=4, BOS_LAST=15,
    DISP=0.8, FVG_MIN=0.10, SL_PAD=0.25, RISK_MIN=0.5, RISK_MAX=4.0,
    ORDER_LIFE=12, TT=0.10, EMBARGO=42, SPLIT_IS=60, SPLIT_OOS=80,
    WARM_DAYS=200, MIN_SYMS=20, EMA_FAST=50, EMA_SLOW=200,
    T3_MAX_GAP_BTC_ETH=6, T3_MISSING_FRAC=0.002, T2_MAX_GAPS=3,
)
SYMS = [
    "XBTUSDTM", "ETHUSDTM", "SOLUSDTM", "BNBUSDTM", "XRPUSDTM",
    "DOGEUSDTM", "ADAUSDTM", "LINKUSDTM", "AVAXUSDTM", "DOTUSDTM",
    "SUIUSDTM", "TRXUSDTM", "NEARUSDTM", "AAVEUSDTM", "OPUSDTM",
    "ARBUSDTM", "APTUSDTM", "ATOMUSDTM", "FILUSDTM", "LTCUSDTM",
    "BCHUSDTM", "ETCUSDTM", "UNIUSDTM", "INJUSDTM", "SEIUSDTM",
    "VETUSDTM", "HBARUSDTM", "ALGOUSDTM", "XLMUSDTM", "ICPUSDTM",
    "WIFUSDTM", "PEPEUSDTM", "FLOKIUSDTM",
]
BTC, ETH = "XBTUSDTM", "ETHUSDTM"

# implementation parameters (reported, not strategy rules)
FETCH_BARS = 6000
CHUNK, OVERLAP, PAUSE = 150, 2, 0.12
BASE = "https://api-futures.kucoin.com"
RANDOM_CUTS, POISON_CUTS, CUT_SEED = 12, 4, 20261013

# isolation: every output starts with v4_smc_, never Candidate 11 files
F_FROZEN = "v4_smc_frozen_data.csv.gz"
F_CFG = "v4_smc_cycle1_config.json"
F_REPORT = "v4_smc_cycle1_report.txt"
F_EVENTS = "v4_smc_cycle1_events.jsonl.gz"
FORBIDDEN = {"candidate11_paper_state.json", "crypto_signal_bot_v4.py"}


def safe_path(name):
    base = os.path.basename(name)
    assert base.startswith("v4_smc_") and base not in FORBIDDEN, f"ISOLATION_VIOLATION {name}"
    return name


def partner_of(sym):
    return ETH if sym == BTC else BTC


# ----------------------------------------------------------------------------
# FETCH (KuCoin Futures 4H klines, last-trade, closed candles only)
# ----------------------------------------------------------------------------
def fetch_chunk(symbol, start_ms, end_ms):
    import requests
    last = None
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/api/v1/kline/query",
                             params={"symbol": symbol, "granularity": 240,
                                     "from": start_ms, "to": end_ms}, timeout=20)
        except requests.RequestException as e:
            last = repr(e); time.sleep(1 + attempt); continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP_{r.status_code}"; time.sleep(2 * (attempt + 1)); continue
        r.raise_for_status()
        j = r.json()
        if str(j.get("code")) != "200000":
            raise ValueError(f"API_CODE={j.get('code')}")
        return j.get("data") or []
    raise RuntimeError(f"REQUEST_FAILED: {last}")


def fetch_symbol(symbol, now_s):
    bar_ms = BAR * 1000
    e0 = int(now_s * 1000)
    rows, empty = [], 0
    for i in range(int(math.ceil(FETCH_BARS / CHUNK))):
        e = e0 - i * CHUNK * bar_ms
        s = e - (CHUNK + OVERLAP) * bar_ms
        data = fetch_chunk(symbol, s, e)
        time.sleep(PAUSE)
        if not data:
            empty += 1
            if empty >= 2:
                break
            continue
        empty = 0
        rows.extend(data)
    return rows


def rows_to_df(sym, rows):
    out = []
    for x in rows:
        if len(x) < 7:
            continue
        try:
            out.append((sym, int(int(x[0]) // 1000), float(x[1]), float(x[2]), float(x[3]),
                        float(x[4]), float(x[5]), float(x[6])))
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(out, columns=["sym", "ts", "o", "h", "l", "c", "v", "t"])


def fetch_all(now_s):
    d1 = (int(now_s) // BAR) * BAR - BAR            # last CLOSED 4H candle open
    frames, failed = [], {}
    for k, s in enumerate(SYMS):
        try:
            rows = fetch_symbol(s, now_s)
            df = rows_to_df(s, rows)
            df = df[df["ts"] <= d1]
            df = df.drop_duplicates()
            frames.append(df)
            print(f"FETCH {k+1}/{len(SYMS)} {s} rows={len(df)}", flush=True)
        except Exception as e:                       # noqa
            failed[s] = repr(e)[:200]
            print(f"FETCH_FAIL {s} {failed[s]}", flush=True)
    if not frames:
        raise RuntimeError("NO_DATA_FETCHED")
    df = pd.concat(frames, ignore_index=True).sort_values(["sym", "ts"], kind="mergesort")
    return df.reset_index(drop=True), d1, failed


def csv_bytes(df):
    return df.to_csv(index=False, float_format="%.12g").encode()


def write_gz(path, data):
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
            g.write(data)


def read_gz(path):
    with gzip.open(path, "rb") as g:
        return g.read()


# ----------------------------------------------------------------------------
# RAW WORLD (one global UTC timeline)
# ----------------------------------------------------------------------------
def load_raw(df, d1):
    on = df[df["ts"] % BAR == 0]
    offgrid = int(len(df) - len(on))
    dup = on.duplicated(["sym", "ts"], keep=False)
    conflict = 0
    if dup.any():
        d = on[dup].drop_duplicates()
        conflict = int(d.duplicated(["sym", "ts"], keep=False).sum())
    on = on.drop_duplicates(["sym", "ts"], keep="first")
    on = on[on["ts"] <= d1]
    if on.empty:
        raise RuntimeError("EMPTY_AFTER_FILTER")
    g0 = int(on["ts"].min())
    n = (d1 - g0) // BAR + 1
    raw = types.SimpleNamespace(G=g0 + BAR * np.arange(n, dtype=np.int64), N=int(n), D1=int(d1),
                                syms=[], O={}, H={}, L={}, C={}, V={}, T={}, P={},
                                offgrid=offgrid, conflict=conflict)
    for s in SYMS:
        sub = on[on["sym"] == s]
        raw.syms.append(s)
        arrs = {k: np.full(n, np.nan) for k in "ohlcvt"}
        pres = np.zeros(n, bool)
        if len(sub):
            idx = ((sub["ts"].values - g0) // BAR).astype(np.int64)
            for k in "ohlcvt":
                arrs[k][idx] = sub[k].values
            pres[idx] = True
        raw.O[s], raw.H[s], raw.L[s], raw.C[s] = arrs["o"], arrs["h"], arrs["l"], arrs["c"]
        raw.V[s], raw.T[s], raw.P[s] = arrs["v"], arrs["t"], pres
    return raw


# ----------------------------------------------------------------------------
# TRACKED ARRAYS (provenance): every read records the latest candle index it depends on
# ----------------------------------------------------------------------------
class TA:
    __slots__ = ("a", "off", "trk")

    def __init__(self, a, trk, off=0):
        self.a, self.trk, self.off = a, trk, off

    def __getitem__(self, i):
        if isinstance(i, slice):
            assert i.start is not None and i.start >= 0 and i.stop is not None
            m = i.stop - 1 + self.off
        else:
            assert i >= 0
            m = i + self.off
        if m > self.trk[0]:
            self.trk[0] = m
        return self.a[i]


def pivots_hi(H, n):
    out = np.zeros(n, bool)
    k = P["PIV"]
    if n >= 2 * k + 1:
        w = np.lib.stride_tricks.sliding_window_view(H, 2 * k + 1)
        c = w[:, k]
        out[k:n - k] = (c > w[:, :k].max(axis=1)) & (c > w[:, k + 1:].max(axis=1))
    return out


def atr_wilder(H, L, C, valid):
    n = len(H)
    out = np.full(n, np.nan)
    k, trs, a, pc = 0, [], 0.0, 0.0
    N = P["ATR_N"]
    for i in range(n):
        if not valid[i]:
            k, trs = 0, []
            continue
        if k == 0:
            tr = H[i] - L[i]
        else:
            tr = max(H[i] - L[i], abs(H[i] - pc), abs(L[i] - pc))
        k += 1
        pc = C[i]
        if k <= N:
            trs.append(tr)
            if k == N:
                a = sum(trs) / N
                out[i] = a
        else:
            a = ((N - 1) * a + tr) / N
            out[i] = a
    return out


def ema_chain(dclose, complete, N):
    nd = len(dclose)
    out = np.full(nd, np.nan)
    al = 2.0 / (N + 1)
    run, e = [], None
    for k in range(nd):
        if not complete[k]:
            run, e = [], None
            continue
        c = dclose[k]
        run.append(c)
        if len(run) == N:
            e = sum(run) / N
            out[k] = e
        elif len(run) > N:
            e = al * c + (1 - al) * e
            out[k] = e
    return out


def clean_valid(O, H, L, C, V, pres):
    with np.errstate(invalid="ignore"):
        ok = (pres & np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & np.isfinite(C) & np.isfinite(V)
              & (V > 0) & (O > 0) & (H > 0) & (L > 0) & (C > 0)
              & (H >= np.maximum(O, C)) & (L <= np.minimum(O, C)) & (H >= L))
    return ok


def derive_sym(raw, sym, n, trk, day0):
    G = raw.G[:n]
    O, H, L, C = (raw.O[sym][:n].copy(), raw.H[sym][:n].copy(), raw.L[sym][:n].copy(), raw.C[sym][:n].copy())
    V, pres = raw.V[sym][:n], raw.P[sym][:n]
    valid = clean_valid(O, H, L, C, V, pres)
    for a in (O, H, L, C):
        a[~valid] = np.nan
    cinv = np.cumsum(~valid).astype(np.int64)
    atr = atr_wilder(H, L, C, valid)
    # --- daily (complete UTC days only: all 6 bars valid)
    day = (G // 86400 - day0).astype(np.int64)
    nd = int(day[-1]) + 1
    cv = np.bincount(day, weights=valid.astype(float), minlength=nd)
    complete = cv == 6
    dlast = ((day0 + np.arange(nd)) * 86400 + 72000 - raw.G[0]) // BAR
    dclose = np.full(nd, np.nan)
    ok = np.nonzero(complete)[0]
    dclose[ok] = C[dlast[ok].astype(np.int64)]
    rl = np.zeros(nd, np.int64)
    for k in range(nd):
        rl[k] = (rl[k - 1] + 1 if k > 0 else 1) if complete[k] else 0
    ema50 = ema_chain(dclose, complete, P["EMA_FAST"])
    ema200 = ema_chain(dclose, complete, P["EMA_SLOW"])
    prev_rl = np.concatenate([[0], rl[:-1]])
    pos = (G % 86400) // BAR
    cinv0 = np.concatenate([[0], cinv])
    idx = np.arange(n)
    before = cinv0[idx] - cinv0[np.maximum(0, idx - pos)]
    cond1 = prev_rl[day] >= P["WARM_DAYS"]
    elig = cond1 & (before == 0)
    elig_from = int(np.argmax(cond1)) if cond1.any() else None
    # --- pivot data-gap flag (direction independent)
    k = P["PIV"]
    pgap = np.zeros(n, bool)
    if n >= 2 * k + 1:
        lo_i = np.arange(k, n - k)
        inv_win = cinv0[lo_i + k + 1] - cinv0[lo_i - k]
        pgap[k:n - k] = valid[k:n - k] & (inv_win > 0)
    S = types.SimpleNamespace(sym=sym, valid=TA(valid, trk), cinv=TA(cinv, trk), atr=TA(atr, trk),
                              elig=TA(elig, trk), elig_from=elig_from, pgap=TA(pgap, trk, P["PIV"]),
                              dclose=dclose, ema50=ema50, ema200=ema200, dlast=dlast,
                              valid_raw=valid, elig_raw=elig, cinv_raw=cinv, d={})
    for sgn in (1, -1):
        if sgn == 1:
            Od, Hd, Ld, Cd = O, H, L, C
        else:
            Od, Hd, Ld, Cd = -O, -L, -H, -C
        S.d[sgn] = types.SimpleNamespace(
            O=TA(Od, trk), H=TA(Hd, trk), L=TA(Ld, trk), C=TA(Cd, trk),
            ph=TA(pivots_hi(Hd, n), trk, P["PIV"]),
            pl=TA(pivots_hi(-Ld, n), trk, P["PIV"]))
    return S


def derive(raw, n, trk):
    D = types.SimpleNamespace(G=raw.G[:n], n=n, trk=trk, day0=int(raw.G[0] // 86400), s={}, syms=[])
    for s in raw.syms:
        if raw.P[s][:n].any():
            D.s[s] = derive_sym(raw, s, n, trk, D.day0)
            D.syms.append(s)
    return D


def htf_state(S, D, b):
    dstar = (int(D.G[b]) + BAR) // 86400 - 1 - D.day0
    if dstar < 0:
        return None
    idx = int(S.dlast[dstar])
    if idx > D.trk[0]:
        D.trk[0] = idx
    c, e50, e200 = S.dclose[dstar], S.ema50[dstar], S.ema200[dstar]
    if not (np.isfinite(c) and np.isfinite(e50) and np.isfinite(e200)):
        return None
    return float(c), float(e50), float(e200)


# ----------------------------------------------------------------------------
# ORDER STATE MACHINE (SELL space; BUY = mirror through negated prices)
# ----------------------------------------------------------------------------
class OT:
    """One tracker ('P' = trade-through primary, 'T' = simple touch) for one order."""

    def __init__(self, mode, E, ob_hi, ob_lo, sl, atr):
        self.mode, self.E, self.ob_hi, self.ob_lo, self.sl, self.atr = mode, E, ob_hi, ob_lo, sl, atr
        self.state = "PENDING"

    def step(self, k, o, h, l, c):
        """k = 1..ORDER_LIFE candles after the BOS candle. Returns (state, details) or None."""
        if self.state != "PENDING":
            return None
        if o > self.ob_hi:                                   # 1) gap cancel
            self.state = "CANCELLED_GAP"
            return self.state, {}
        thr = self.E + P["TT"] * self.atr if self.mode == "P" else self.E
        if h >= thr:                                         # 2) fill (candle 12 eligible)
            hit = bool(h >= self.sl)
            self.state = "STOPPED_ON_FILL_CANDLE" if hit else "FILLED"
            return self.state, {"fill_dir_price": self.E, "sl_reached_on_fill_candle": hit}
        if c > self.ob_hi:                                   # 3) close beyond far edge
            self.state = "CANCELLED_FAR_EDGE"
            return self.state, {}
        if k >= P["ORDER_LIFE"]:                             # 4) expire after candle 12
            self.state = "EXPIRED"
            return self.state, {}
        return None


# ----------------------------------------------------------------------------
# DETECTION
# ----------------------------------------------------------------------------
def run_dir(D, S, B, sgn, dname, i0, upto, tis_idx, ev, viol):
    trk, G = D.trk, D.G
    sym = S.sym
    X, Y = S.d[sgn], B.d[sgn]
    O, H, L, C, PH, PL = X.O, X.H, X.L, X.C, X.ph, X.pl
    YH = Y.H
    val, cinv, atr, elig = S.valid, S.cinv, S.atr, S.elig
    bcinv = B.cinv
    if S.elig_from is None:
        return
    start = max(i0, S.elig_from)
    active, orders = None, []

    def inv(T, a, b):
        return int(T[b]) - (int(T[a - 1]) if a > 0 else 0)

    def emit(typ, b, **kw):
        r = {"ts": int(G[b]) + BAR, "sym": sym, "dir": dname, "type": typ}
        r.update(kw)
        r["bar"] = b
        r["max_in"] = int(trk[0])
        ev.append(r)

    for b in range(start, upto + 1):
        trk[0] = -1
        if not val[b]:
            for od in orders:
                for m in ("P", "T"):
                    t = od["tr"][m]
                    if t.state == "PENDING":
                        t.state = "ABORTED_GAP"
                        emit("ORDER_STATE", b, order=od["id"], variant=od["variant"], tracker=m, state="ABORTED_GAP")
            orders = [od for od in orders if any(t.state == "PENDING" for t in od["tr"].values())]
            if active is not None and b >= active["p"] + P["BOS_FIRST"]:
                emit("SMT_REJECT", b, reason="SMT_ABORTED_DATA_GAP", smt=active["id"])
                active = None
            if trk[0] > b:
                viol.append((sym, dname, b, int(trk[0])))
            continue

        # ---- 1) open orders: one candle step
        if orders:
            o_, h_, l_, c_ = O[b], H[b], L[b], C[b]
            for od in orders:
                k = b - od["bos"]
                for m in ("P", "T"):
                    r = od["tr"][m].step(k, o_, h_, l_, c_)
                    if r:
                        st, det = r
                        kw = {}
                        if "fill_dir_price" in det:
                            kw = {"fill_price": float(sgn * det["fill_dir_price"]),
                                  "sl_reached_on_fill_candle": det["sl_reached_on_fill_candle"]}
                        emit("ORDER_STATE", b, order=od["id"], variant=od["variant"], tracker=m, state=st, **kw)
            orders = [od for od in orders if any(t.state == "PENDING" for t in od["tr"].values())]

        # ---- 2) BOS for the active SMT (evaluated before a newer SMT replaces it)
        if active is not None and b >= active["p"] + P["BOS_FIRST"]:
            a = active
            c = float(C[b])
            if c < a["Lv"]:
                active = None
                body = float(O[b]) - c
                A14 = float(atr[b])
                if not np.isfinite(A14):
                    emit("BOS_REJECT", b, reason="ATR_UNDEFINED", smt=a["id"])
                elif body < P["DISP"] * A14:
                    emit("BOS_REJECT", b, reason="BOS_NO_DISPLACEMENT", smt=a["id"],
                         body=body / A14)
                else:
                    handle_bos(D, S, sgn, dname, b, a, A14, c, tis_idx, emit, orders)
            elif b >= a["p"] + P["BOS_LAST"]:
                emit("BOS_REJECT", b, reason="SMT_EXPIRED_NO_BOS", smt=a["id"])
                active = None

        # ---- 3) new pivot high confirmed at close of bar b (pivot bar p = b-3)
        p = b - P["PIV"]
        if p >= P["PIV"] and elig[b]:
            if sgn == 1 and S.pgap[p]:
                emit("PIVOT_GAP", b, reason="PIVOT_SKIPPED_DATA_GAP", pivot_ts=int(G[p]))
            if PH[p]:
                lo = p - P["SMT_LB"]
                if lo < 0:
                    pass
                else:
                    pid = f"{sym}|{dname}|{int(G[p])}"
                    if inv(cinv, lo, p + 2) > 0 or inv(bcinv, lo, p + 2) > 0:
                        emit("SMT_REJECT", b, reason="SMT_SKIPPED_DATA_GAP", smt=pid, detail="smt_window")
                    elif H[p] > H[lo:p].max():
                        href = float(YH[lo:p].max())
                        hwin = float(YH[p - P["SMT_PM"]:p + P["SMT_PM"] + 1].max())
                        if hwin <= href:
                            if active is not None:
                                emit("SMT_REPLACED", b, smt=active["id"], by=pid)
                                active = None
                            emit("SMT_CONFIRMED", b, smt=pid, pivot_ts=int(G[p]),
                                 pivot_price=float(sgn * H[p]),
                                 partner=B.sym, partner_ref=float(sgn * href), partner_win=float(sgn * hwin))
                            active = smt_reference(D, S, B, sgn, dname, b, p, pid, emit, inv)
        if trk[0] > b:
            viol.append((sym, dname, b, int(trk[0])))
        # keep orders created inside handle_bos (list mutated in place)


def smt_reference(D, S, B, sgn, dname, b, p, pid, emit, inv):
    """Reference pivot L, intactness and BOS-before-confirmation checks (all known at close of p+3)."""
    X = S.d[sgn]
    C, L, PL = X.C, X.L, X.pl
    j = None
    for jj in range(p - 4, P["PIV"] - 1, -1):                # confirmed before p: jj+3 <= p-1
        if PL[jj]:
            j = jj
            break
    if j is None:
        emit("SMT_REJECT", b, reason="NO_REFERENCE_PIVOT", smt=pid)
        return None
    if inv(S.cinv, j, p + 3) > 0:
        emit("SMT_REJECT", b, reason="SMT_SKIPPED_DATA_GAP", smt=pid, detail="reference_span")
        return None
    Lv = float(L[j])
    if not (float(C[j + 1:p + 1].min()) >= Lv):
        emit("SMT_REJECT", b, reason="NO_INTACT_REFERENCE", smt=pid, ref_ts=int(D.G[j]))
        return None
    if float(C[p + 1:p + 4].min()) < Lv:
        emit("SMT_REJECT", b, reason="BOS_BEFORE_SMT_CONFIRMATION", smt=pid, ref_ts=int(D.G[j]))
        return None
    return {"id": pid, "p": p, "j": j, "Lv": Lv}


def handle_bos(D, S, sgn, dname, b, a, A14, cbos, tis_idx, emit, orders):
    X = S.d[sgn]
    O, H, L, C = X.O, X.H, X.L, X.C
    p, pid = a["p"], a["id"]
    emb = bool(b >= tis_idx - P["EMBARGO"])
    emit("BOS", b, smt=pid, ref_price=float(sgn * a["Lv"]), close=float(sgn * cbos), atr14=A14,
         embargo=emb, ref_ts=int(D.G[a["j"]]))
    # ---- HTF flag at the BOS close
    h = htf_state(S, D, b)
    if h is None:
        emit("SETUP_REJECT", b, reason="HTF_UNAVAILABLE", smt=pid, embargo=emb)
        return
    dc, e50, e200 = h
    up, down = (dc > e50 and e50 > e200), (dc < e50 and e50 < e200)
    if (sgn == 1 and up) or (sgn == -1 and down):
        emit("SETUP_REJECT", b, reason="HTF_FORBIDDEN_UPTREND" if sgn == 1 else "HTF_FORBIDDEN_DOWNTREND",
             smt=pid, embargo=emb)
        return
    # ---- Order block: last opposite-colour candle in [p-2, b-1]
    Os, Cs = O[p - 2:b], C[p - 2:b]
    idx = np.nonzero(Cs > Os)[0]
    if len(idx) == 0:
        emit("SETUP_REJECT", b, reason="NO_OB", smt=pid, embargo=emb)
        return
    i = p - 2 + int(idx[-1])
    ob_lo, ob_hi = float(L[i]), float(H[i])
    # ---- FVG overlapping the OB (inside the leg OB..BOS, completed by the BOS candle)
    Hs, Ls = H[i:b + 1], L[i:b + 1]
    g_lo, g_hi = Hs[2:], Ls[:-2]
    size = g_hi - g_lo
    okf = (size >= P["FVG_MIN"] * A14) & (g_hi > ob_lo) & (g_lo < ob_hi)
    fvg_any = bool((size >= P["FVG_MIN"] * A14).any()) if len(size) else False
    fvg = bool(okf.any()) if len(size) else False
    E = ob_lo
    if not (E > cbos):
        emit("SETUP_REJECT", b, reason="INVALID_ENTRY_ALREADY_REACHED", smt=pid, embargo=emb,
             ob_index_ts=int(D.G[i]), entry=float(sgn * E), bos_close=float(sgn * cbos))
        return
    sl = float(H[p]) + P["SL_PAD"] * A14
    risk = (sl - E) / abs(E) * 100.0
    if not (P["RISK_MIN"] <= risk <= P["RISK_MAX"]):
        emit("SETUP_REJECT", b, reason="RISK_OUT_OF_RANGE", smt=pid, embargo=emb,
             risk_pct=float(risk), ob_index_ts=int(D.G[i]))
        return
    emit("SETUP_VALID", b, smt=pid, embargo=emb, fvg_overlap=fvg, fvg_any=fvg_any,
         ob_ts=int(D.G[i]), ob_low=float(min(sgn * ob_lo, sgn * ob_hi)), ob_high=float(max(sgn * ob_lo, sgn * ob_hi)),
         entry=float(sgn * E), sl=float(sgn * sl), risk_pct=float(risk), atr14=A14)
    for variant in ("A", "B"):
        if variant == "B" and not fvg:
            emit("VARIANT_REJECT", b, reason="NO_FVG_OVERLAP", smt=pid, variant="B", embargo=emb)
            continue
        oid = f"{pid}|{variant}"
        emit("ORDER_PLACED", b, order=oid, variant=variant, smt=pid, embargo=emb, entry=float(sgn * E),
             sl=float(sgn * sl), risk_pct=float(risk))
        orders.append({"id": oid, "variant": variant, "bos": b,
                       "tr": {m: OT(m, E, ob_hi, ob_lo, sl, A14) for m in ("P", "T")}})


def analyze(D, i0, upto, tis_idx, excluded):
    ev, viol = [], []
    upto = min(upto, D.n - 1)
    for sym in D.syms:
        if sym in excluded:
            continue
        S = D.s[sym]
        B = D.s.get(partner_of(sym))
        if S.elig_from is None or B is None:
            continue
        for sgn, dn in ((1, "SELL"), (-1, "BUY")):
            run_dir(D, S, B, sgn, dn, i0, upto, tis_idx, ev, viol)
    return ev, viol


# ----------------------------------------------------------------------------
# EVENT LOG HELPERS
# ----------------------------------------------------------------------------
def canon(ev, t=None):
    out = []
    for e in ev:
        if t is not None and e["ts"] > t:
            continue
        out.append(json.dumps({k: v for k, v in e.items() if k != "max_in"}, sort_keys=True))
    out.sort()
    return out


def order_states(ev, t):
    st = {}
    for e in ev:
        if e["ts"] > t:
            continue
        if e["type"] == "ORDER_PLACED":
            st[(e["order"], "P")] = "PENDING"
            st[(e["order"], "T")] = "PENDING"
        elif e["type"] == "ORDER_STATE":
            st[(e["order"], e["tracker"])] = e["state"]
    return st


FILLED_STATES = ("FILLED", "STOPPED_ON_FILL_CANDLE")


def pt_differences(ev):
    """Per-order comparison of tracker P (trade-through) vs tracker T (touch) from the event log.
    Reporting only: reads events, changes nothing. Returns (n_orders, list of differing orders)."""
    placed, st = {}, defaultdict(dict)
    for e in ev:
        if e["type"] == "ORDER_PLACED":
            placed[e["order"]] = e
        elif e["type"] == "ORDER_STATE":
            st[e["order"]][e["tracker"]] = e
    out = []
    for oid, o in placed.items():
        p, t = st[oid].get("P"), st[oid].get("T")
        ps, pb = (p["state"], p["bar"]) if p else ("PENDING", None)
        tsx, tb = (t["state"], t["bar"]) if t else ("PENDING", None)
        if (ps, pb) == (tsx, tb):
            continue
        out.append({"order": oid, "variant": o["variant"], "sym": o["sym"], "dir": o["dir"],
                    "placed_ts": o["ts"], "embargo": o.get("embargo", False),
                    "P_state": ps, "P_ts": p["ts"] if p else None, "P_candle": (pb - o["bar"]) if p else None,
                    "T_state": tsx, "T_ts": t["ts"] if t else None, "T_candle": (tb - o["bar"]) if t else None,
                    "state_diff": ps != tsx, "candle_diff": pb != tb})
    out.sort(key=lambda r: (r["placed_ts"], r["order"]))
    return len(placed), out


# ----------------------------------------------------------------------------
# VALIDATION (tiers 1-3, eligibility, D0 / D1 / T_IS_end)
# ----------------------------------------------------------------------------
def runs_of_false(valid_slice):
    inv = ~valid_slice
    if not inv.any():
        return []
    d = np.diff(np.concatenate([[0], inv.astype(int), [0]]))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return list(zip(st.tolist(), (en - st).tolist()))


def validate(raw):
    trk = [-1]
    D = derive(raw, raw.N, trk)
    V = {"fail": [], "info": {}, "per_symbol": {}}
    G, N = raw.G, raw.N
    V["info"].update(G0=int(G[0]), D1=int(G[-1]), N_bars=N, offgrid=raw.offgrid, conflicting_dups=raw.conflict)
    if raw.offgrid or raw.conflict:
        V["fail"].append(f"T3c off-grid={raw.offgrid} conflicting_duplicates={raw.conflict}")
    nodata = [s for s in raw.syms if s not in D.s]
    for s in (BTC, ETH):
        if s in nodata:
            V["fail"].append(f"T3a {s} has no data")
    if V["fail"] and (BTC in nodata or ETH in nodata):
        V["valid"] = False
        return V, D
    # --- D0 from RAW eligibility (before reliability exclusions)
    ecount = np.zeros(N, int)
    for s in D.syms:
        ecount += D.s[s].elig_raw
    both = D.s[BTC].elig_raw & D.s[ETH].elig_raw
    ok0 = np.nonzero((ecount >= P["MIN_SYMS"]) & both)[0]
    if len(ok0) == 0:
        V["fail"].append(f"T3d no bar with >= {P['MIN_SYMS']} eligible symbols incl. BTC+ETH "
                         f"(max eligible={int(ecount.max())})")
        V["info"]["max_eligible_symbols"] = int(ecount.max())
        V["valid"] = False
        return V, D
    i0 = int(ok0[0])
    i1 = N - 1
    tis = i0 + (P["SPLIT_IS"] * (i1 - i0)) // 100
    toos = i0 + (P["SPLIT_OOS"] * (i1 - i0)) // 100
    V["info"].update(D0_idx=i0, D0=int(G[i0]), T_IS_end_idx=int(tis), T_IS_end=int(G[tis]),
                     T_OOS_end_idx=int(toos), T_OOS_end=int(G[toos]), IS_bars=int(tis - i0),
                     IS_days=(tis - i0) / 6.0)
    # --- per-symbol gap analysis inside IS = [D0, T_IS_end)
    excluded = {}
    tot_exp = tot_miss = tot_exp_all = tot_miss_all = 0
    for s in SYMS:
        if s not in D.s:
            excluded[s] = "EXCLUDED_NO_DATA"
            V["per_symbol"][s] = {"status": "EXCLUDED_NO_DATA"}
            continue
        S = D.s[s]
        first = int(np.argmax(raw.P[s]))
        lo = max(first, i0)
        seg = S.valid_raw[lo:tis]
        runs = runs_of_false(seg) if len(seg) else []
        ngaps = len(runs)
        maxrun = max([r[1] for r in runs], default=0)
        miss = int((~seg).sum()) if len(seg) else 0
        exp = int(len(seg))
        full_runs = runs_of_false(S.valid_raw[first:])
        info = {"first_ts": int(G[first]), "elig_from_idx": S.elig_from,
                "elig_from_ts": int(G[S.elig_from]) if S.elig_from is not None else None,
                "IS_gaps": ngaps, "IS_max_gap_bars": maxrun, "IS_missing": miss, "IS_expected": exp,
                "full_history_gaps": len(full_runs), "full_history_max_gap": max([r[1] for r in full_runs], default=0)}
        if s in (BTC, ETH) and maxrun > P["T3_MAX_GAP_BTC_ETH"]:
            V["fail"].append(f"T3a {s} gap of {maxrun} bars inside IS (> {P['T3_MAX_GAP_BTC_ETH']})")
        tot_exp_all += exp
        tot_miss_all += miss
        if ngaps > P["T2_MAX_GAPS"]:
            excluded[s] = "EXCLUDED_UNRELIABLE_DATA"
            info["status"] = "EXCLUDED_UNRELIABLE_DATA"
        else:
            usable = int((S.elig_raw[i0:tis] & S.valid_raw[i0:tis]).sum())
            info["usable_bars"] = usable
            if usable == 0:
                excluded[s] = "EXCLUDED_NEVER_ELIGIBLE_IN_IS"
                info["status"] = "EXCLUDED_NEVER_ELIGIBLE_IN_IS"
            else:
                info["status"] = "INCLUDED"
                tot_exp += exp
                tot_miss += miss
        V["per_symbol"][s] = info
    frac = tot_miss / tot_exp if tot_exp else 0.0
    V["info"].update(missing_bars_IS=tot_miss, expected_bars_IS=tot_exp, missing_frac_IS=frac,
                     missing_frac_IS_all_symbols=(tot_miss_all / tot_exp_all if tot_exp_all else 0.0))
    if frac > P["T3_MISSING_FRAC"]:
        V["fail"].append(f"T3b missing bars {frac*100:.4f}% > {P['T3_MISSING_FRAC']*100:.2f}% of expected in IS")
    inc = [s for s in D.syms if s not in excluded]
    cnt = np.zeros(N, int)
    for s in inc:
        cnt += D.s[s].elig_raw
    mn = int(cnt[i0:tis].min()) if tis > i0 else 0
    V["info"]["min_eligible_included_in_IS"] = mn
    V["info"]["eligible_included_at_D0"] = int(cnt[i0])
    V["info"]["eligible_included_at_T_IS_end"] = int(cnt[tis - 1]) if tis > i0 else 0
    if mn < P["MIN_SYMS"]:
        V["fail"].append(f"T3d only {mn} eligible included symbols at the minimum inside IS (< {P['MIN_SYMS']})")
    usable_total = sum(V["per_symbol"][s].get("usable_bars", 0) for s in inc)
    V["info"]["usable_symbol_bars_IS"] = int(usable_total)
    V["info"]["usable_symbol_days_IS"] = usable_total / 6.0
    V["info"]["included_symbols"] = len(inc)
    V["excluded"] = excluded
    V["valid"] = not V["fail"]
    V["feasible"] = bool(V["valid"] and usable_total > 0 and tis > i0)
    return V, D


# ----------------------------------------------------------------------------
# CAUSALITY (truncation + provenance + poison)
# ----------------------------------------------------------------------------
def first_diffs(a, b, k=3):
    sa, sb = set(a), set(b)
    return {"only_in_full_filtered": sorted(sa - sb)[:k], "only_in_truncated": sorted(sb - sa)[:k]}


def causality(raw, V, base_ev, base_viol, log):
    info = V["info"]
    i0, tis, excluded = info["D0_idx"], info["T_IS_end_idx"], V["excluded"]
    upto = tis - 1
    rng = random.Random(CUT_SEED)
    cuts = set(rng.sample(range(i0 + 1, upto), min(RANDOM_CUTS, max(0, upto - i0 - 2))))
    by_type = defaultdict(list)
    for e in base_ev:
        key = e["type"] if e["type"] != "ORDER_STATE" else "ORDER_STATE_" + e["state"]
        by_type[key].append(e["bar"])
    ev_cuts = []
    for key in sorted(by_type):
        pick = rng.sample(by_type[key], min(2, len(by_type[key])))
        for eb in pick:
            for c in (eb - 1, eb, eb + 1):
                if i0 < c < upto:
                    cuts.add(c)
                    ev_cuts.append((key, c))
    res = {"cuts": 0, "mismatch": 0, "late_records": 0, "state_mismatch": 0, "prov_viol": 0,
           "poison_cuts": 0, "poison_mismatch": 0, "details": []}
    # provenance on the base run
    bad = [e for e in base_ev if e["max_in"] > e["bar"]]
    res["prov_viol"] += len(bad) + len(base_viol)
    log(f"TRUNCATION cuts={len(cuts)} (random={RANDOM_CUTS}, event-neighbour={len(set(c for _, c in ev_cuts))})")
    trk = [-1]
    for c in sorted(cuts):
        t = int(raw.G[c]) + BAR
        D2 = derive(raw, c + 1, trk)
        ev2, viol2 = analyze(D2, i0, c, tis, excluded)
        exp, got = canon(base_ev, t), canon(ev2)
        res["cuts"] += 1
        res["prov_viol"] += len(viol2) + sum(1 for e in ev2 if e["max_in"] > e["bar"])
        if any(e["ts"] > t for e in ev2):
            res["late_records"] += 1
        s_full, s_cut = order_states(base_ev, t), order_states(ev2, t)
        if s_full != s_cut:
            res["state_mismatch"] += 1
        if exp != got:
            res["mismatch"] += 1
            if len(res["details"]) < 5:
                res["details"].append({"cut_idx": c, "cut_ts": t, **first_diffs(exp, got)})
    # poison test: full-length arrays, every bar after the cut replaced with garbage
    prng = np.random.default_rng(CUT_SEED)
    pcuts = sorted(rng.sample(range(i0 + 1, upto), min(POISON_CUTS, max(0, upto - i0 - 2))))
    for c in pcuts:
        pr = types.SimpleNamespace(**{k: getattr(raw, k) for k in ("G", "N", "D1", "syms", "offgrid", "conflict")})
        for nm in ("O", "H", "L", "C", "V", "T", "P"):
            setattr(pr, nm, {s: getattr(raw, nm)[s].copy() for s in raw.syms})
        m = raw.N - (c + 1)
        for s in raw.syms:
            base = prng.uniform(1, 1e6, m)
            pr.O[s][c + 1:], pr.C[s][c + 1:] = base, base * prng.uniform(.9, 1.1, m)
            pr.H[s][c + 1:] = np.maximum(pr.O[s][c + 1:], pr.C[s][c + 1:]) * 1.2
            pr.L[s][c + 1:] = np.minimum(pr.O[s][c + 1:], pr.C[s][c + 1:]) * 0.8
            pr.V[s][c + 1:] = prng.uniform(1, 1e6, m)
            pr.P[s][c + 1:] = True
        D3 = derive(pr, pr.N, trk)
        ev3, viol3 = analyze(D3, i0, c, tis, excluded)
        t = int(raw.G[c]) + BAR
        res["poison_cuts"] += 1
        if canon(base_ev, t) != canon(ev3):
            res["poison_mismatch"] += 1
            if len(res["details"]) < 5:
                res["details"].append({"poison_cut_idx": c, **first_diffs(canon(base_ev, t), canon(ev3))})
        res["prov_viol"] += len(viol3)
    res["trunc_pass"] = res["mismatch"] == 0 and res["late_records"] == 0 and res["state_mismatch"] == 0
    res["poison_pass"] = res["poison_mismatch"] == 0
    res["prov_pass"] = res["prov_viol"] == 0
    res["pass"] = res["trunc_pass"] and res["poison_pass"] and res["prov_pass"]
    return res


# ----------------------------------------------------------------------------
# SUMMARY / REPORT
# ----------------------------------------------------------------------------
def ts_str(t):
    return pd.Timestamp(int(t), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")


def summarize(ev, upto_ts):
    per = defaultdict(Counter)
    reasons = Counter()
    for e in ev:
        k = (e["sym"], e["dir"])
        t = e["type"]
        emb = e.get("embargo", False)
        if t == "SMT_CONFIRMED":
            per[k]["SMT"] += 1
        elif t == "BOS":
            per[k]["BOS"] += 1
            per[k]["BOS_embargo"] += int(emb)
        elif t == "SETUP_VALID":
            per[k]["OB_valid"] += 1
            per[k]["FVG_ovl"] += int(e["fvg_overlap"])
        elif t == "ORDER_PLACED":
            per[k]["placed_" + e["variant"]] += 1
            per[k]["placed_" + e["variant"] + "_emb"] += int(emb)
        if t in ("SMT_REJECT", "BOS_REJECT", "SETUP_REJECT", "VARIANT_REJECT", "PIVOT_GAP", "SMT_REPLACED"):
            reasons[(t, e.get("reason", "-"))] += 1
    states = order_states(ev, upto_ts)
    ordinfo = {}
    for e in ev:
        if e["type"] == "ORDER_PLACED":
            ordinfo[e["order"]] = (e["variant"], e["sym"], e["dir"], e.get("embargo", False))
    st_cnt = defaultdict(Counter)
    for (oid, m), st in states.items():
        v, s, d, emb = ordinfo[oid]
        st_cnt[(v, m)][st] += 1
        if st in FILLED_STATES:
            per[(s, d)][f"fill_{v}_{m}"] += 1
    sl_fill = Counter()
    for e in ev:
        if e["type"] == "ORDER_STATE" and e["state"] in FILLED_STATES and e["sl_reached_on_fill_candle"]:
            sl_fill[(e["variant"], e["tracker"])] += 1
    return per, reasons, st_cnt, sl_fill


def build_report(V, ev, caus, files, meta):
    L = []
    w = L.append
    I = V["info"]
    w("=" * 78)
    w("V4 SMC CYCLE 1 - ACTUAL TEST RUN (detection / counting / causality only)")
    w("=" * 78)
    w(f"RUN_UTC={meta['run_utc']}  DATA={meta['data_source']}  FETCH_BARS_PARAM={FETCH_BARS}")
    w(f"DATA_SHA256={meta['sha']}")
    if meta.get("fetch_failed"):
        w(f"FETCH_FAILED_SYMBOLS={meta['fetch_failed']}")
    w("")
    w("1) DATA VALIDATION")
    w(f"  GRID_START={ts_str(I['G0'])}  D1(last closed 4H, frozen)={ts_str(I['D1'])}  N_BARS={I['N_bars']}")
    w(f"  T3c off_grid_timestamps={I['offgrid']} conflicting_duplicates={I['conflicting_dups']}")
    if "D0" in I:
        w(f"  D0={ts_str(I['D0'])} (idx {I['D0_idx']})  T_IS_end={ts_str(I['T_IS_end'])} (idx {I['T_IS_end_idx']})  "
          f"T_OOS_end={ts_str(I['T_OOS_end'])} (not used in Cycle 1)")
        w(f"  IS_WINDOW bars={I['IS_bars']} days={I['IS_days']:.1f}  EMBARGO={P['EMBARGO']} candles before T_IS_end")
        w(f"  T3b missing_bars_IS={I['missing_bars_IS']}/{I['expected_bars_IS']} = {I['missing_frac_IS']*100:.4f}% "
          f"(limit {P['T3_MISSING_FRAC']*100:.2f}%)  [all-symbols incl. excluded: {I['missing_frac_IS_all_symbols']*100:.4f}%]")
        w(f"  T3d eligible included symbols: at D0={I['eligible_included_at_D0']} min_in_IS={I['min_eligible_included_in_IS']} "
          f"at T_IS_end={I['eligible_included_at_T_IS_end']} (need >= {P['MIN_SYMS']}, BTC+ETH included)")
        w(f"  INCLUDED_SYMBOLS={I['included_symbols']}  USABLE_IS symbol-bars={I['usable_symbol_bars_IS']} "
          f"symbol-days={I['usable_symbol_days_IS']:.0f}")
        w("  PER SYMBOL (status | first bar | ELIGIBLE_FROM | usable_bars | IS gaps | max gap | full-history gaps/max):")
        for s in SYMS:
            d = V["per_symbol"].get(s, {})
            if d.get("status") == "EXCLUDED_NO_DATA":
                w(f"   {s:10s} EXCLUDED_NO_DATA")
                continue
            ef = ts_str(d["elig_from_ts"]) if d.get("elig_from_ts") else "never"
            w(f"   {s:10s} {d['status']:30s} first={ts_str(d['first_ts'])} elig_from={ef} "
              f"usable={d.get('usable_bars', 0)} IS_gaps={d['IS_gaps']} IS_maxgap={d['IS_max_gap_bars']} "
              f"hist_gaps={d['full_history_gaps']}/{d['full_history_max_gap']}")
    if V["fail"]:
        w("  VALIDATION FAIL REASONS:")
        for f in V["fail"]:
            w("   - " + f)
    w(f"  DATA_VALIDATION={'PASS' if V['valid'] else 'FAIL'}")
    w(f"  IS_FEASIBLE_AFTER_200D_WARMUP={'YES' if V.get('feasible') else 'NO'}")
    if not V["valid"]:
        w("")
        w("STOPPED: dataset validation failed. No detection was run. (Warm-up and minimum-symbol rules NOT relaxed.)")
        w("")
        w("FILES: " + ", ".join(files))
        return "\n".join(L)
    upto_ts = I["T_IS_end"]
    per, reasons, st_cnt, sl_fill = summarize(ev, upto_ts)
    w("")
    w("2) EVENT COUNTS (IS window only; embargo = BOS close within 42 candles before T_IS_end: counted, not used)")
    tot = Counter()
    w("  symbol dir  SMT BOS (emb) OBvalid FVGovl | placed A (emb) B | fills P:A P:B T:A T:B")
    for s in SYMS:
        for d in ("SELL", "BUY"):
            c = per.get((s, d))
            if not c:
                continue
            tot.update(c)
            w(f"  {s:10s} {d:4s} {c['SMT']:3d} {c['BOS']:3d} ({c['BOS_embargo']:2d}) {c['OB_valid']:5d} {c['FVG_ovl']:5d} | "
              f"{c['placed_A']:4d} ({c['placed_A_emb']:2d}) {c['placed_B']:3d} | "
              f"{c['fill_A_P']:3d} {c['fill_B_P']:3d} {c['fill_A_T']:3d} {c['fill_B_T']:3d}")
    for d in ("SELL", "BUY"):
        dt = Counter()
        for s in SYMS:
            dt.update(per.get((s, d), Counter()))
        w(f"  TOTAL {d:4s}: SMT={dt['SMT']} BOS={dt['BOS']} (embargo {dt['BOS_embargo']}) OBvalid={dt['OB_valid']} "
          f"FVGovl={dt['FVG_ovl']} placedA={dt['placed_A']} placedB={dt['placed_B']} "
          f"fillP A/B={dt['fill_A_P']}/{dt['fill_B_P']} fillT A/B={dt['fill_A_T']}/{dt['fill_B_T']}")
    w(f"  TOTAL ALL : SMT={tot['SMT']} BOS={tot['BOS']} (embargo {tot['BOS_embargo']}) OBvalid={tot['OB_valid']} "
      f"FVGovl={tot['FVG_ovl']} placedA={tot['placed_A']} (embargo {tot['placed_A_emb']}) placedB={tot['placed_B']} "
      f"(embargo {tot['placed_B_emb']})")
    nA = tot["placed_A"] - tot["placed_A_emb"]
    nB = tot["placed_B"] - tot["placed_B_emb"]
    w(f"  INFO (not a gate in Cycle 1): non-embargoed orders A={nA} B={nB}; Cycle-2 gate N>=100 refers to filled trades.")
    w("")
    w("3) REASON COUNTS")
    for (t, r), n in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
        w(f"  {t:15s} {r:34s} {n}")
    w("")
    w("4) ORDER STATES AS OF T_IS_end (primary P = trade-through, comparison T = touch)")
    for v in ("A", "B"):
        for m in ("P", "T"):
            c = st_cnt.get((v, m), Counter())
            fills = sum(c[s] for s in FILLED_STATES)
            w(f"  variant {v} tracker {m}: placed={sum(c.values())} FILLS={fills} "
              f"(FILLED={c['FILLED']} STOPPED_ON_FILL_CANDLE={c['STOPPED_ON_FILL_CANDLE']}) "
              f"CANCELLED_GAP={c['CANCELLED_GAP']} CANCELLED_FAR_EDGE={c['CANCELLED_FAR_EDGE']} "
              f"EXPIRED={c['EXPIRED']} ABORTED_GAP={c['ABORTED_GAP']} PENDING_AT_END={c['PENDING']} "
              f"SL_reached_on_fill_candle={sl_fill[(v, m)]}")
    w("")
    n_ord, pt = pt_differences(ev)
    w("4b) P vs T PER-ORDER DIFFERENCES (fill/terminal candle and final state; all orders compared)")
    w(f"  orders compared={n_ord}  differing={len(pt)}  final-state differences={sum(r['state_diff'] for r in pt)}  "
      f"same state but different candle/timestamp={sum((not r['state_diff']) and r['candle_diff'] for r in pt)}")
    for r in pt:
        pts = ts_str(r["P_ts"]) if r["P_ts"] else "-"
        tts = ts_str(r["T_ts"]) if r["T_ts"] else "-"
        w(f"   {r['order']} placed={ts_str(r['placed_ts'])} embargo={r['embargo']} | "
          f"P={r['P_state']}@candle{r['P_candle']} ({pts}) | T={r['T_state']}@candle{r['T_candle']} ({tts}) | "
          f"{'STATE_DIFF' if r['state_diff'] else 'timing_only'}")
    if not pt:
        w("   none")
    w("")
    w("5) CAUSALITY")
    if caus is None:
        w("  NOT RUN")
    else:
        w(f"  TRUNCATION: cuts={caus['cuts']} mismatches={caus['mismatch']} late_records={caus['late_records']} "
          f"order_state_mismatches={caus['state_mismatch']}  -> {'PASS' if caus['trunc_pass'] else 'FAIL'}")
        w(f"  POISON (future bars replaced by garbage): cuts={caus['poison_cuts']} mismatches={caus['poison_mismatch']} "
          f"-> {'PASS' if caus['poison_pass'] else 'FAIL'}")
        w(f"  PROVENANCE (latest input candle <= decision candle): violations={caus['prov_viol']} "
          f"-> {'PASS' if caus['prov_pass'] else 'FAIL'}")
        for d in caus["details"]:
            w("  MISMATCH_DETAIL " + json.dumps(d)[:1500])
        w(f"  CAUSALITY_OVERALL={'PASS' if caus['pass'] else 'FAIL'}")
    w("")
    w("6) IMPLEMENTATION NOTES (choices not fully fixed by the spec text; reported, not tuned)")
    for n in meta["notes"]:
        w("  - " + n)
    w("")
    w("FILES: " + ", ".join(files))
    return "\n".join(L)


NOTES = [
    "First close beyond L inside the BOS window decides the BOS; if its body < 0.8*ATR14 the result is BOS_NO_DISPLACEMENT and the SMT is consumed.",
    "A newer divergence-confirmed SMT replaces an older unconsumed one (event SMT_REPLACED) even if the newer one later fails the reference checks; the older SMT can still BOS on the bar where the newer is confirmed.",
    "Reference/intact/BOS-before checks are evaluated at SMT confirmation (close of p+3), SMT_CONFIRMED is logged first.",
    "Gaps in [p-20,p+2] (A or B) -> SMT_SKIPPED_DATA_GAP; gaps in [L pivot, p+3] -> SMT_SKIPPED_DATA_GAP (detail=reference_span); invalid bar during BOS window -> SMT_ABORTED_DATA_GAP; during order life -> ABORTED_GAP.",
    "Gate order after an accepted BOS: HTF flag -> OB -> FVG/variant -> INVALID_ENTRY_ALREADY_REACHED -> RISK_OUT_OF_RANGE. ATR14 for FVG minimum, SL and trade-through is the ATR at the BOS candle (frozen at order placement).",
    "Variant B uses the same entry as A (OB proximal edge); it is the subset of A with an FVG overlapping the OB (strict positive overlap).",
    "Touch tracker fills when high >= E (SELL) / low <= E (BUY); primary requires high >= E + 0.10*ATR14.",
    "Several orders of one symbol/direction can be pending simultaneously (no position logic in Cycle 1).",
    "D0 is taken from raw eligibility before the reliability exclusion; tier-3(d) is then evaluated as the minimum eligible included-symbol count over [D0, T_IS_end).",
    "Excluded-symbol set and D0/T_IS_end/D1 are dataset-level constants fixed from the frozen dataset and reused unchanged in every truncated run.",
    "Partner B needs valid data in the SMT window but is not required to be eligible itself.",
    "IS = bars [D0, T_IS_end); detection starts at max(D0, symbol ELIGIBLE_FROM); orders still pending at T_IS_end stay PENDING.",
    "The 30-minute live placement deadline is not simulated.",
]


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    for f in (F_FROZEN, F_CFG, F_REPORT, F_EVENTS):
        safe_path(f)
    out = io.StringIO()

    def log(s):
        print(s, flush=True)

    t_start = time.time()
    run_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")
    fetch_failed = {}
    if os.path.exists(F_FROZEN) and os.path.exists(F_CFG):
        cfg0 = json.load(open(F_CFG))
        data = read_gz(F_FROZEN)
        sha = hashlib.sha256(data).hexdigest()
        if sha != cfg0.get("sha256"):
            log(f"CONFIG_ERROR frozen data sha mismatch {sha} != {cfg0.get('sha256')}")
            return 3
        df = pd.read_csv(io.BytesIO(data))
        d1 = int(cfg0["D1"])
        src = "FROZEN_FILE"
        fetch_failed = cfg0.get("fetch_failed", {})
    else:
        cfg0 = None
        try:
            df, d1, fetch_failed = fetch_all(time.time())
        except Exception as e:                       # noqa
            log(f"FETCH_ERROR {e!r}")
            return 3
        data = csv_bytes(df)
        sha = hashlib.sha256(data).hexdigest()
        write_gz(F_FROZEN, data)
        src = "FETCHED_AND_FROZEN"
    log(f"DATA {src} rows={len(df)} D1={ts_str(d1)} sha256={sha}")
    raw = load_raw(df, d1)
    V, D = validate(raw)
    cfg = {"spec": "V4 SMC Spec Freeze v1 + Patches 1-4", "params": P, "D1": int(d1), "sha256": sha,
           "fetch_bars_param": FETCH_BARS, "symbols": SYMS, "fetch_failed": fetch_failed,
           "validation_pass": V["valid"], "info": V["info"], "excluded": V.get("excluded", {})}
    if cfg0 is not None:
        for k in ("D0", "T_IS_end", "T_IS_end_idx", "D0_idx"):
            if k in cfg0.get("info", {}) and cfg0["info"][k] != V["info"].get(k):
                log(f"CONFIG_ERROR {k} changed vs frozen config: {cfg0['info'][k]} -> {V['info'].get(k)}")
                return 3
    json.dump(cfg, open(F_CFG, "w"), indent=1, sort_keys=True)
    files = [F_FROZEN, F_CFG, F_REPORT]
    meta = {"run_utc": run_utc, "data_source": src, "sha": sha, "fetch_failed": fetch_failed, "notes": NOTES}
    if not V["valid"]:
        rep = build_report(V, [], None, files, meta)
        open(F_REPORT, "w").write(rep)
        print(rep)
        return 1
    I = V["info"]
    i0, tis = I["D0_idx"], I["T_IS_end_idx"]
    trk = [-1]
    t0 = time.time()
    ev, viol = analyze(D, i0, tis - 1, tis, V["excluded"])
    log(f"DETECTION events={len(ev)} viol={len(viol)} secs={time.time()-t0:.1f}")
    with gzip.open(F_EVENTS, "wt") as g:
        for e in sorted(ev, key=lambda e: (e["ts"], e["sym"], e["dir"], e["type"])):
            g.write(json.dumps(e, sort_keys=True) + "\n")
    files.append(F_EVENTS)
    t0 = time.time()
    caus = causality(raw, V, ev, viol, log)
    log(f"CAUSALITY secs={time.time()-t0:.1f} pass={caus['pass']}")
    rep = build_report(V, ev, caus, files, meta)
    open(F_REPORT, "w").write(rep)
    print(rep)
    log(f"TOTAL_SECS={time.time()-t_start:.0f}")
    return 0 if caus["pass"] else 2

# ----------------------------------------------------------------------------
# SELF-TEST (synthetic data only; run:  python v4_smc_cycle1.py selftest)
# ----------------------------------------------------------------------------
M = sys.modules[__name__]
FAILS, PASSES = [], 0


def check(name, cond, extra=""):
    global PASSES
    if cond:
        PASSES += 1
        print(f"PASS {name}")
    else:
        FAILS.append(name)
        print(f"FAIL {name} {extra}")


def mk_raw(series, n, g0=0):
    """series: {sym: (O,H,L,C) arrays len n}; missing bars marked by NaN in C."""
    G = g0 + M.BAR * np.arange(n, dtype=np.int64)
    raw = types.SimpleNamespace(G=G, N=n, D1=int(G[-1]), syms=list(series), offgrid=0, conflict=0,
                                O={}, H={}, L={}, C={}, V={}, T={}, P={})
    for s, (o, h, l, c) in series.items():
        o, h, l, c = (np.array(x, float) for x in (o, h, l, c))
        pres = np.isfinite(c)
        raw.O[s], raw.H[s], raw.L[s], raw.C[s] = o, h, l, c
        raw.V[s] = np.where(pres, 1000.0, np.nan)
        raw.T[s] = raw.V[s].copy()
        raw.P[s] = pres
    return raw


# ---------------------------------------------------------------- unit tests
def test_atr():
    rng = np.random.default_rng(1)
    n = 60
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1, n)
    l = np.minimum(o, c) - rng.uniform(0, 1, n)
    v = np.ones(n, bool)
    a = M.atr_wilder(h, l, c, v)
    tr = np.r_[h[0] - l[0], np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])]
    exp = np.full(n, np.nan)
    exp[13] = tr[:14].mean()
    for i in range(14, n):
        exp[i] = (13 * exp[i - 1] + tr[i]) / 14
    check("ATR14 Wilder (seed=mean first 14 TR, TR1=H-L)", np.allclose(a[13:], exp[13:]) and np.isnan(a[:13]).all())
    v2 = v.copy(); v2[30] = False
    a2 = M.atr_wilder(h, l, c, v2)
    check("ATR resets after a gap (no bridging)", np.isnan(a2[30:43]).all() and np.isfinite(a2[44]))


def test_pivots():
    H = np.array([1, 2, 3, 5, 3, 2, 1, 2, 5, 5, 1, 1, 1, 1], float)
    ph = M.pivots_hi(H, len(H))
    check("pivot high strict, 3 bars each side", ph[3] and not ph[8] and not ph[9] and ph.sum() == 1)
    H[5] = np.nan
    check("pivot with NaN neighbour is not a pivot", not M.pivots_hi(H, len(H))[3])


def test_ema():
    d = np.arange(1, 11, dtype=float)
    comp = np.ones(10, bool)
    e = M.ema_chain(d, comp, 5)
    al = 2 / 6
    exp = [np.nan] * 4 + [3.0]
    for k in range(5, 10):
        exp.append(al * d[k] + (1 - al) * exp[-1])
    check("Daily EMA seeded with SMA of first N closes", np.allclose(e[4:], exp[4:]) and np.isnan(e[:4]).all())
    comp2 = comp.copy(); comp2[7] = False
    e2 = M.ema_chain(d, comp2, 5)
    check("EMA restarts after incomplete day", np.isnan(e2[7:]).all())


def test_eligibility():
    n = 6 * 260
    c = np.full(n, 100.0)
    s = {"A": (c, c + 1, c - 1, c)}
    raw = mk_raw(s, n)
    D = M.derive(raw, n, [-1])
    ef = D.s["A"].elig_from
    check("ELIGIBLE_FROM = first bar after 200th complete day", ef == 200 * 6, f"got {ef}")
    cm = c.copy(); cm[6 * 100 + 2] = np.nan
    raw = mk_raw({"A": (cm, cm + 1, cm - 1, cm)}, n)
    S = M.derive(raw, n, [-1]).s["A"]
    # gap in day 100 -> need 200 new complete days: days 101..300 -> beyond series, never eligible
    check("gap resets warm-up (ineligible until 200 new days)", not S.elig_raw.any())
    cm = c.copy(); cm[6 * 10 + 2] = np.nan
    raw = mk_raw({"A": (cm, cm + 1, cm - 1, cm)}, n)
    S = M.derive(raw, n, [-1]).s["A"]
    check("after early gap eligibility starts at day 211 start", S.elig_from == 6 * 211, f"got {S.elig_from}")


def test_order_tracker():
    E, hi, sl, atr = 100.0, 105.0, 110.0, 4.0
    t = lambda m: M.OT(m, E, hi, 100.0, sl, atr)
    p, tc = t("P"), t("T")
    r1 = p.step(1, 99, 100.2, 98, 99.5); r2 = tc.step(1, 99, 100.2, 98, 99.5)
    check("touch fills, trade-through (0.10 ATR) does not", r1 is None and r2 and r2[0] == "FILLED")
    p, tc = t("P"), t("T")
    r = p.step(1, 106, 107, 99, 100)
    check("gap cancel (open > OB high) precedes fill", r and r[0] == "CANCELLED_GAP")
    p = t("P"); p.step(1, 99, 99.5, 98, 99)
    r = p.step(2, 99, 106, 98, 106)
    check("fill beats far-edge close on the same candle", r and r[0] == "FILLED")
    p = t("P")
    r = p.step(1, 99, 100.1, 98, 105.5)
    check("close beyond far edge cancels unfilled order", r and r[0] == "CANCELLED_FAR_EDGE")
    p = t("P")
    out = [p.step(k, 99, 100.3, 98, 99) for k in range(1, 12)]
    check("candle 11 does not expire", all(x is None for x in out) and p.state == "PENDING")
    p = t("P")
    for k in range(1, 12):
        p.step(k, 99, 100.3, 98, 99)
    r = p.step(12, 99, 100.5, 98, 99)
    check("fill on candle 12 allowed", r and r[0] == "FILLED")
    p = t("P")
    for k in range(1, 12):
        p.step(k, 99, 100.3, 98, 99)
    r = p.step(12, 99, 100.3, 98, 99)
    check("expired after candle 12 without fill", r and r[0] == "EXPIRED")
    p = t("P")
    r = p.step(1, 99, 111, 98, 100)
    check("SL reached on fill candle flagged, state STOPPED_ON_FILL_CANDLE",
          r and r[0] == "STOPPED_ON_FILL_CANDLE" and r[1]["sl_reached_on_fill_candle"])
    p = t("P"); p.step(1, 99, 100.5, 98, 99)
    check("tracker is terminal after fill (no TP/exit tracking)", p.step(2, 99, 120, 98, 99) is None)


# ---------------------------------------------------------- crafted scenario
def crafted(variant_b, n=6 * 230, base=1300):
    """Flat filler (no pivots) + scripted SELL setup on ETH, flat BTC partner."""
    o = np.full(n, 1000.0); h = np.full(n, 1005.0); l = np.full(n, 995.0); c = np.full(n, 1000.0)
    sc = [  # (o,h,l,c) relative bar r0 = j
        (1000, 1002, 980, 985),      # 0 j  pivot low 980
        (985, 1000, 984, 998),       # 1
        (998, 1020, 997, 1018),      # 2
        (1018, 1040, 1017, 1038),    # 3
        (1038, 1060, 1037, 1058),    # 4
        (1058, 1080, 1057, 1078),    # 5 p  pivot high 1080 (new 20-bar high)
        (1078, 1079, 1065, 1068),    # 6
        (1068, 1070, 1058, 1060),    # 7
        (1060, 1062, 1050, 1052),    # 8 p+3 confirmation
        (1052, 1066, 1051, 1065) if not variant_b else (1052, 1055, 1048, 1049),   # 9
        (1065, 1066, 1040, 1042),    # 10
        (1042, 1045, 1020, 1022),    # 11
        (1022, 1025, 1000, 1002),    # 12
        (1002, 1003, 950, 955),      # 13 BOS (close < 980)
    ]
    if variant_b:
        sc[10] = (1049, 1050, 1040, 1042)
    sc += [(955, 1000, 950, 998), (998, 1070, 990, 1068)] if variant_b else \
          [(955, 1040, 950, 1030), (1030, 1052.5, 1025, 1040), (1040, 1060, 1035, 1055)]
    for k, (a, b_, c_, d) in enumerate(sc):
        o[base + k], h[base + k], l[base + k], c[base + k] = a, b_, c_, d
    return n, base, {M.ETH: (o, h, l, c), M.BTC: (np.full(n, 1000.), np.full(n, 1005.), np.full(n, 995.), np.full(n, 1000.))}


def run_world(raw, upto=None, i0=1200):
    D = M.derive(raw, raw.N, [-1])
    upto = raw.N - 1 if upto is None else upto
    ev, viol = M.analyze(D, i0, upto, raw.N + 100, {})
    return D, ev, viol


def test_crafted():
    for vb in (False, True):
        n, base, s = crafted(vb)
        raw = mk_raw(s, n)
        D, ev, viol = run_world(raw)
        sell = [e for e in ev if e["sym"] == M.ETH and e["dir"] == "SELL"]
        types_ = [e["type"] for e in sell]
        tag = "B" if vb else "A"
        check(f"[{tag}] SMT_CONFIRMED at p+3", any(e["type"] == "SMT_CONFIRMED" and e["bar"] == base + 8 for e in sell), str(types_))
        check(f"[{tag}] BOS at r0+13 with displacement", any(e["type"] == "BOS" and e["bar"] == base + 13 for e in sell), str(types_))
        sv = [e for e in sell if e["type"] == "SETUP_VALID"]
        check(f"[{tag}] SETUP_VALID emitted", len(sv) == 1, str(types_))
        if sv:
            check(f"[{tag}] FVG overlap flag == {vb}", sv[0]["fvg_overlap"] == vb, str(sv[0]))
            check(f"[{tag}] entry = OB low", sv[0]["entry"] == (1057.0 if vb else 1051.0), str(sv[0]))
        placed = sorted(e["variant"] for e in sell if e["type"] == "ORDER_PLACED")
        check(f"[{tag}] variants placed {['A','B'] if vb else ['A']}", placed == (["A", "B"] if vb else ["A"]), str(placed))
        st = {(e["variant"], e["tracker"]): (e["bar"] - base, e["state"]) for e in sell if e["type"] == "ORDER_STATE"}
        if vb:
            check(f"[{tag}] both trackers fill on r0+15", st.get(("A", "P")) == (15, "FILLED") and st.get(("A", "T")) == (15, "FILLED"), str(st))
        else:
            check(f"[{tag}] touch fills r0+15, trade-through fills r0+16",
                  st.get(("A", "T")) == (15, "FILLED") and st.get(("A", "P")) == (16, "FILLED"), str(st))
        bad = [e for e in ev if e["type"] in ("BOS", "ORDER_PLACED", "ORDER_STATE") and (e["dir"] == "BUY" or e["sym"] == M.BTC)]
        check(f"[{tag}] no BOS/orders on BUY side or on BTC", not bad, str([(e['sym'], e['dir'], e['type']) for e in bad[:3]]))
        check(f"[{tag}] provenance: no violations", not viol and all(e["max_in"] <= e["bar"] for e in ev))
        # truncation at every bar around the setup
        ok = True
        for cut in range(base - 2, base + 20):
            D2 = M.derive(raw, cut + 1, [-1])
            e2, v2 = M.analyze(D2, 1200, cut, raw.N + 100, {})
            t = int(raw.G[cut]) + M.BAR
            if M.canon(ev, t) != M.canon(e2) or v2 or M.order_states(ev, t) != M.order_states(e2, t):
                ok = False
                print("  truncation mismatch at", cut)
        check(f"[{tag}] truncation identical for every cut around the setup", ok)


def test_buy_mirror():
    n, base, s = crafted(False)
    o, h, l, c = s[M.ETH]
    m = 2000.0
    # exact price mirror around 2000: BUY setup must give mirrored SELL events
    mir = {M.ETH: (m - o, m - l, m - h, m - c), M.BTC: tuple(m - x for x in (s[M.BTC][0], s[M.BTC][3], s[M.BTC][2], s[M.BTC][1]))}
    mir[M.BTC] = (m - s[M.BTC][0], m - s[M.BTC][2], m - s[M.BTC][1], m - s[M.BTC][3])
    raw = mk_raw(mir, n)
    D, ev, viol = run_world(raw)
    raw0 = mk_raw(s, n)
    _, ev0, _ = run_world(raw0)
    key = lambda e: (e["type"], e["bar"], e.get("variant"), e.get("tracker"), e.get("state"), e.get("reason"))
    a = sorted(key(e) for e in ev0 if e["dir"] == "SELL" and e["sym"] == M.ETH)
    b = sorted(key(e) for e in ev if e["dir"] == "BUY" and e["sym"] == M.ETH)
    check("BUY is the exact mirror of SELL (same event sequence on mirrored prices)", a == b and len(a) > 5, f"{len(a)} {len(b)}")


# ----------------------------------------------------------- random world
def synth(n_sym=26, days=420, seed=7):
    rng = np.random.default_rng(seed)
    n = days * 6
    common = np.cumsum(rng.normal(0, 0.012, n))
    series = {}
    for k, s in enumerate(M.SYMS[:n_sym]):
        ret = 0.7 * np.diff(common, prepend=0) + rng.normal(0, 0.014, n)
        lvl = 100 * np.exp(np.cumsum(ret) + 0.0)
        o = np.r_[lvl[0], lvl[:-1]]
        c = lvl
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
        series[s] = (o, h, l, c)
    # a few random missing bars in alts only
    for s in M.SYMS[2:5]:
        for i in rng.integers(300, n - 10, 2):
            for a in series[s]:
                a[i] = np.nan
    return n, series


def test_end_to_end():
    n, series = synth()
    raw = mk_raw(series, n)
    # symbols without data -> M.SYMS not in series are absent from raw.syms; add empties
    for s in M.SYMS:
        if s not in raw.P:
            raw.syms.append(s)
            for nm in "OHLCVT":
                getattr(raw, nm)[s] = np.full(n, np.nan)
            raw.P[s] = np.zeros(n, bool)
    V, D = M.validate(raw)
    check("validation runs on synthetic world and passes", V["valid"], str(V["fail"]))
    if not V["valid"]:
        return
    I = V["info"]
    check("D0 is first bar with >=20 eligible symbols incl BTC+ETH", I["D0_idx"] >= 200 * 6)
    ev, viol = M.analyze(D, I["D0_idx"], I["T_IS_end_idx"] - 1, I["T_IS_end_idx"], V["excluded"])
    cnt = {}
    for e in ev:
        cnt[e["type"]] = cnt.get(e["type"], 0) + 1
    print("  synthetic event counts:", json.dumps(cnt, sort_keys=True))
    check("synthetic run produces SMT and BOS events (test is not vacuous)", cnt.get("SMT_CONFIRMED", 0) > 5 and cnt.get("BOS", 0) > 0)
    res = M.causality(raw, V, ev, viol, print)
    print("  causality:", {k: v for k, v in res.items() if k != "details"})
    check("end-to-end truncation test: zero mismatches", res["trunc_pass"])
    check("end-to-end poison test: zero mismatches", res["poison_pass"])
    check("end-to-end provenance: zero violations", res["prov_pass"])


def test_negative_control():
    """The truncation test must be able to FAIL: inject a deliberate look-ahead and expect a mismatch."""
    n, base, s = crafted(False)
    raw = mk_raw(s, n)
    D = M.derive(raw, n, [-1])
    ev, _ = M.analyze(D, 1200, n - 1, n + 100, {})
    orig = M.OT.step

    def leaky(self, k, o, h, l, c):
        return orig(self, k, o, h, l, c)
    # leak: make the BOS check read the NEXT close via a patched tracker array
    S = D.s[M.ETH]
    X = S.d[1]
    real_c = X.C.a.copy()
    leak = real_c.copy()
    leak[:-1] = np.where(np.arange(len(leak) - 1) >= base + 10, real_c[1:], real_c[:-1])
    X.C.a[:] = leak
    ev_leak, _ = M.analyze(D, 1200, n - 1, n + 100, {})
    X.C.a[:] = real_c
    cut = base + 12
    D2 = M.derive(raw, cut + 1, [-1])
    e2, _ = M.analyze(D2, 1200, cut, n + 100, {})
    t = int(raw.G[cut]) + M.BAR
    check("negative control: a look-ahead leak is detected by the truncation comparison",
          M.canon(ev_leak, t) != M.canon(e2))


def test_buy_pivot_price_logging():
    n, base, s = crafted(False)
    _, ev0, _ = run_world(mk_raw(s, n))
    sell = [e for e in ev0 if e["type"] == "SMT_CONFIRMED" and e["dir"] == "SELL" and e["sym"] == M.ETH and e["bar"] == base + 8]
    check("SELL pivot_price logged = pivot high 1080", len(sell) == 1 and sell[0]["pivot_price"] == 1080.0, str(sell))
    o, h, l, c = s[M.ETH]
    m = 2000.0
    mir = {M.ETH: (m - o, m - l, m - h, m - c),
           M.BTC: (m - s[M.BTC][0], m - s[M.BTC][2], m - s[M.BTC][1], m - s[M.BTC][3])}
    _, ev1, _ = run_world(mk_raw(mir, n))
    buy = [e for e in ev1 if e["type"] == "SMT_CONFIRMED" and e["dir"] == "BUY" and e["sym"] == M.ETH and e["bar"] == base + 8]
    check("BUY pivot_price logged positive and = real pivot low (2000-1080=920)", len(buy) == 1 and buy[0]["pivot_price"] == 920.0, str(buy))
    allbuy = [e for e in ev1 if e["type"] == "SMT_CONFIRMED" and e["dir"] == "BUY"]
    check("no SMT_CONFIRMED has a non-positive pivot_price", all(e["pivot_price"] > 0 for e in ev0 + ev1 if e["type"] == "SMT_CONFIRMED") and len(allbuy) >= 1)


def test_pt_difference_report():
    # crafted scenario A: touch fills at r0+15, trade-through at r0+16 -> exactly one timing-only difference
    n, base, s = crafted(False)
    D, ev, _ = run_world(mk_raw(s, n))
    n_ord, pt = M.pt_differences(ev)
    check("P/T report: crafted A has 1 order, 1 timing-only difference", n_ord == 1 and len(pt) == 1 and not pt[0]["state_diff"]
          and pt[0]["candle_diff"] and (pt[0]["T_candle"], pt[0]["P_candle"]) == (2, 3), str(pt))
    check("P/T report: affected order id and placed timestamp are reported", pt and pt[0]["order"].endswith("|A") and pt[0]["placed_ts"] == int(D.G[base + 13]) + M.BAR, str(pt))
    # crafted scenario B: both trackers fill on the same candle -> no differences
    n, base, s = crafted(True)
    _, ev, _ = run_world(mk_raw(s, n))
    n_ord, pt = M.pt_differences(ev)
    check("P/T report: crafted B (same candle) has 0 differences", n_ord == 2 and pt == [], str(pt))
    # synthetic events: state difference and pending handling
    mk = lambda typ, **k: dict(type=typ, ts=k.pop("ts", 0), bar=k.pop("bar", 0), **k)
    evs = [mk("ORDER_PLACED", order="X|SELL|1|A", variant="A", sym="X", dir="SELL", ts=100, bar=10, embargo=False),
           mk("ORDER_STATE", order="X|SELL|1|A", tracker="T", state="FILLED", ts=200, bar=12),
           mk("ORDER_STATE", order="X|SELL|1|A", tracker="P", state="EXPIRED", ts=300, bar=22),
           mk("ORDER_PLACED", order="Y|BUY|1|A", variant="A", sym="Y", dir="BUY", ts=100, bar=10, embargo=False),
           mk("ORDER_STATE", order="Y|BUY|1|A", tracker="T", state="FILLED", ts=200, bar=12),
           mk("ORDER_STATE", order="Y|BUY|1|A", tracker="P", state="FILLED", ts=200, bar=12)]
    n_ord, pt = M.pt_differences(evs)
    check("P/T report: state difference detected, identical order not listed",
          n_ord == 2 and len(pt) == 1 and pt[0]["order"] == "X|SELL|1|A" and pt[0]["state_diff"], str(pt))


def run_selftest():
    for f in (test_atr, test_pivots, test_ema, test_eligibility, test_order_tracker, test_crafted,
              test_buy_mirror, test_buy_pivot_price_logging, test_pt_difference_report, test_end_to_end, test_negative_control):
        print(f"--- {f.__name__}")
        f()
    print(f"\nSELFTEST passes={PASSES} fails={len(FAILS)}")
    if FAILS:
        print("FAILED:", FAILS)
        return 1
    print("SELFTEST_OK (synthetic data only; says nothing about the strategy or real data)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(run_selftest())
    sys.exit(main())
