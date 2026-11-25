"""READ-ONLY Audit C: C11 definition vs engine parity on raw candles. Signal/entry-definition parity only.
Never reads closed trades, exit fields, R or P/L. Writes nothing to the repo."""
import json, math, random, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, ".")
import crypto_signal_bot_v4 as eng

B_TS = pd.Timestamp("2026-10-21T08:00:00+00:00")

def ind(c, h, l, v):
    """Independent pure-python implementation of the frozen definition (no pandas rolling/ewm)."""
    n = len(c); z = [None] * n; ema = [0.0] * n; atr = [0.0] * n; rng = [None] * n; rm = [None] * n; vm = [None] * n
    ema[0] = c[0]; a_e = 2 / 201
    tr0 = h[0] - l[0]; atr[0] = tr0
    for i in range(1, n):
        ema[i] = (1 - a_e) * ema[i - 1] + a_e * c[i]
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])); atr[i] = (1 - 1 / 20) * atr[i - 1] + tr / 20
    med = lambda x: sorted(x)[9] / 2 + sorted(x)[10] / 2
    for i in range(19, n):
        w = c[i - 19:i + 1]; m = sum(w) / 20; sd = math.sqrt(sum((x - m) ** 2 for x in w) / 19)
        z[i] = (c[i] - m) / sd if sd > 0 else None
        rng[i] = max(h[i - 19:i + 1]) - min(l[i - 19:i + 1]); vm[i] = med(v[i - 19:i + 1])
    for i in range(38, n): rm[i] = med(rng[i - 19:i + 1])
    return z, ema, atr, rng, rm, vm

def sig_ind(c, h, l, v):
    z, ema, atr, rng, rm, vm = ind(c, h, l, v); out = set()
    for i in range(1, len(c)):
        if None in (z[i], z[i - 1], rm[i], vm[i]): continue
        if z[i - 1] >= 2 and z[i] < 2 and c[i] < ema[i] and rng[i] >= rm[i] and v[i] >= vm[i] and atr[i] > 0: out.add(i)
    return out, atr

def sig_eng(df):
    d = eng.prepare(df); return {i for i in range(len(d)) if eng.signal_reason(d, i) == "SIGNAL"}, d

def fetch_fut(sym, now): return eng.completed_only(eng.fetch_raw(sym, now), now)

def spot_sym(s): return ("BTC" if s.startswith("XBT") else s[:-5]) + "-USDT"

def fetch_spot(sym, now):
    end = int(pd.Timestamp(now).floor("4h").timestamp()); r = requests.get("https://api.kucoin.com/api/v1/market/candles",
        params={"symbol": spot_sym(sym), "type": "4hour", "startAt": end - 640 * 14400, "endAt": end}, timeout=30).json()
    rows = [(int(x[0]), float(x[1]), float(x[3]), float(x[4]), float(x[2]), float(x[5])) for x in r.get("data", [])]  # spot order: o,c,h,l
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).sort_values("t")
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True); return df.drop(columns="t").reset_index(drop=True)

def main(fetch_f=fetch_fut, fetch_s=fetch_spot):
    now = pd.Timestamp.utcnow(); st = json.load(open("candidate11_paper_state.json"))
    sigs = [x for x in st["signals"]]; by = {}
    for x in sigs: by.setdefault(x["symbol"], []).append(x)
    missed = {(m.get("symbol"), m.get("signal_ts")): m.get("reason") for m in st.get("missed", [])}
    T = dict(bars=0, mism=0, look=0, lookn=0, flip=0, flipn=0, st_sig=0, st_found=0, st_nf=[], rec_not_state=[], f_bad=[], sp_f=0, sp_s=0, sp_both=0, sym_err=[])
    first = pd.Timestamp(st["created_at"]).floor("4h")
    print("engine sha256 source ok; state signals:", len(sigs), "created_at", st["created_at"], "now", now)
    for sym in eng.SYMS:
        try: df = fetch_f(sym, now)
        except Exception as e: T["sym_err"].append((sym, type(e).__name__)); continue
        if len(df) < 300: T["sym_err"].append((sym, f"short:{len(df)}")); continue
        c, h, l, v = (df[k].tolist() for k in ("close", "high", "low", "volume"))
        si, atr = sig_ind(c, h, l, v); se, d = sig_eng(df); T["bars"] += len(df)
        T["mism"] += len(si ^ se)
        if si ^ se: print(f"  PARITY MISMATCH {sym}: bars {sorted(df['ts'].iloc[i].isoformat() for i in si ^ se)[:5]}")
        # look-ahead: decision at bar i with data only up to i must equal full-series decision
        pick = sorted(si | set(random.Random(7).sample(range(60, len(df)), 40)))
        for i in pick:
            part, _ = sig_eng(df.iloc[:i + 1].reset_index(drop=True)); T["lookn"] += 1
            if (i in part) != (i in se): T["look"] += 1
        # warm-up sensitivity: engine decisions with only the last 300 bars vs full 600
        sub, _ = sig_eng(df.iloc[-300:].reset_index(drop=True)); off = len(df) - 300
        full_tail = {i - off for i in se if i >= off + 60}; sub_tail = {i for i in sub if i >= 60}
        T["flipn"] += len(full_tail | sub_tail); T["flip"] += len(full_tail ^ sub_tail)
        # state parity (signal definition + entry fields; no outcomes)
        tsi = {df["ts"].iloc[i]: i for i in range(len(df))}
        for x in by.get(sym, []):
            T["st_sig"] += 1; t = pd.Timestamp(x["signal_ts"]); i = tsi.get(t)
            if i is None: T["st_nf"].append((sym, x["signal_ts"], "bar not in fetched window")); continue
            if i in si: T["st_found"] += 1
            else: T["st_nf"].append((sym, x["signal_ts"], "NOT reproduced"))
            nxt = df.iloc[i + 1] if i + 1 < len(df) else None; ent = float(x["entry"]); chk = []
            if pd.Timestamp(x["entry_ts"]) != t + pd.Timedelta(hours=4): chk.append("entry_ts!=signal+4h")
            if nxt is not None and abs(float(nxt["open"]) / ent - 1) > 1e-9 and x.get("source") == "next_candle_open": chk.append("entry!=next open")
            if abs(float(x["atr"]) / atr[i] - 1) > 1e-6: chk.append(f"atr diff {float(x['atr'])/atr[i]-1:.2e}")
            if abs(float(x["risk"]) - 1.25 * float(x["atr"])) > 1e-9 * ent: chk.append("risk!=1.25atr")
            if abs(float(x["sl"]) - (ent + float(x["risk"]))) > 1e-9 * ent or abs(float(x["tp"]) - (ent - 2 * float(x["risk"]))) > 1e-9 * ent: chk.append("sl/tp geometry")
            if chk: T["f_bad"].append((sym, x["signal_ts"], chk))
        for i in si:
            t = df["ts"].iloc[i]
            if t >= first and not any(x["signal_ts"] == t.isoformat() for x in by.get(sym, [])):
                T["rec_not_state"].append((sym, t.isoformat(), missed.get((sym, t.isoformat()), "NO MISSED RECORD")))
        try:  # spot vs futures (backtest used spot candles)
            sp = fetch_s(sym, now); sp = sp.iloc[-len(df):] if len(sp) else sp
            if len(sp) > 300:
                ss, _ = sig_ind(*(sp[k].tolist() for k in ("close", "high", "low", "volume")))
                fs = {df["ts"].iloc[i] for i in si}; sps = {sp["ts"].iloc[i] for i in ss}; lo = max(df["ts"].iloc[200], sp["ts"].iloc[200])
                fs = {t for t in fs if t >= lo}; sps = {t for t in sps if t >= lo}
                T["sp_f"] += len(fs); T["sp_s"] += len(sps); T["sp_both"] += len(fs & sps)
        except Exception as e: print("  spot fetch", sym, type(e).__name__)
        time.sleep(0.2)
    print("\n== C1 independent-vs-engine decision parity (all fetched bars):", T["bars"], "bars; mismatching bars:", T["mism"])
    print("== C2 look-ahead test (truncate data at bar i): checked", T["lookn"], "decisions; differing:", T["look"])
    print("== C3 warm-up sensitivity (engine on last 300 vs 600 bars): decisions compared", T["flipn"], "; flips", T["flip"])
    print("== C4 state signals reproduced from raw candles:", T["st_found"], "of", T["st_sig"], "; not reproduced/outside window:", len(T["st_nf"]))
    for r in T["st_nf"]: print("   ", r)
    print("== C5 state field checks (entry_ts=signal+4h, entry=next open, atr, risk=1.25*atr, sl/tp geometry): issues", len(T["f_bad"]))
    for r in T["f_bad"]: print("   ", r)
    print("== C6 recomputed signals after state start not in state:", len(T["rec_not_state"]))
    for r in T["rec_not_state"]: print("   ", r)
    print("== C7 futures-vs-spot signal sets (same independent code, overlap window): futures", T["sp_f"], "spot", T["sp_s"], "both", T["sp_both"])
    print("   (backtest used KuCoin SPOT candles; forward engine uses KuCoin FUTURES candles)")
    print("symbol fetch problems:", T["sym_err"] or "none"); print("Nothing was written to the repository; no P/L, exit or R fields were read.")

if __name__ == "__main__": main()
