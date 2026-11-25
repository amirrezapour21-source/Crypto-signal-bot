"""READ-ONLY Audit C4: outcome-free decomposition of spot-vs-futures C11 signal differences by condition.
C11 conditions per 4H bar: Z = z20 crosses down through +2, E = close < EMA200, R = RANGE20 >= median RANGE20, V = volume >= median volume 20.
Same timestamps, own-series indicators (independent definition from audit_c_parity). No trades, P/L, R or exit data; nothing is written."""
import collections, sys, time
import pandas as pd
sys.path.insert(0, ".")
import audit_c_parity as c

N = ["Z", "E", "R", "V"]

def flags(df):
    cl, h, l, v = (df[k].tolist() for k in ("close", "high", "low", "volume"))
    z, ema, atr, rng, rm, vm = c.ind(cl, h, l, v); out = {}
    for i in range(1, len(df)):
        if None in (z[i], z[i - 1], rm[i], vm[i]): continue
        out[df["ts"].iloc[i]] = (z[i - 1] >= 2 and z[i] < 2, cl[i] < ema[i], rng[i] >= rm[i], v[i] >= vm[i])
    return out

def main(fetch_f=c.fetch_fut, fetch_s=c.fetch_spot, syms=None):
    now = pd.Timestamp.now("UTC"); T = collections.Counter(); combo = collections.Counter(); sw_f = collections.Counter(); sw_s = collections.Counter(); ns = 0
    for sym in syms or c.eng.SYMS:
        try:
            f, s = fetch_f(sym, now), fetch_s(sym, now); F, S = flags(f), flags(s); lo = max(f.ts.iloc[200], s.ts.iloc[200])
        except Exception as e: print(sym, "ERR", type(e).__name__); continue
        ns += 1; time.sleep(0.2)
        for t in sorted(set(F) & set(S)):
            if t < lo: continue
            a, b = F[t], S[t]; T["bars"] += 1
            for k, n in enumerate(N):
                T[n + "_f"] += a[k]; T[n + "_s"] += b[k]; T[n + "_both"] += a[k] and b[k]; T[n + "_diff"] += a[k] != b[k]
                sw_f[n] += all(a[:k] + (b[k],) + a[k + 1:]) != all(a); sw_s[n] += all(b[:k] + (a[k],) + b[k + 1:]) != all(b)
            fs, ss = all(a), all(b); T["f"] += fs; T["s"] += ss; T["both"] += fs and ss
            if fs != ss: combo[("futures-only" if fs else "spot-only", "+".join(n for k, n in enumerate(N) if a[k] != b[k]))] += 1
    print(f"== C4 symbols {ns}, aligned bars (same timestamps, index>=200 both sources) {T['bars']}")
    print("cond | futures true | spot true | both true | sources differ | futures signal flips if spot's cond used | spot signal flips if futures' cond used")
    for n in N: print(f"{n} | {T[n+'_f']} | {T[n+'_s']} | {T[n+'_both']} | {T[n+'_diff']} | {sw_f[n]} | {sw_s[n]}")
    u = T["f"] + T["s"] - T["both"]
    print(f"signal: futures {T['f']}, spot {T['s']}, both {T['both']}, futures-only {T['f']-T['both']}, spot-only {T['s']-T['both']}, Jaccard {T['both']/u:.2f}" if u else "no signals")
    print("signal-differing bars by set of conditions that differ between sources:")
    for (side, k), v in sorted(combo.items(), key=lambda x: -x[1]): print(f"  {side} | differing: {k} | {v}")
    tot = sum(combo.values()); print(f"  total {tot}; share with V among differing: {sum(v for (s_, k), v in combo.items() if 'V' in k)}/{tot}")
    print("Nothing was written to the repository; no trade outcome fields were read.")

if __name__ == "__main__": main()
