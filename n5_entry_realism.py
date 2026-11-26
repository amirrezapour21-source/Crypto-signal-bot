"""N5 entry-execution realism audit (outcome-free, read-only, prints only). Compares the registered entry-candle open with the price available at the
state-commit time of each signal. Reads only symbol, side, signal/entry timestamps and commit times from the state files; public KuCoin 1m klines.
PROXY only: not fills, spread or slippage. Convention: price at commit time T = close of the last COMPLETE 1m candle before T (no future data);
entry open = open of the 1m candle starting at entry_ts. adverse_bps > 0 = the price at T was WORSE for the trade side than the registered entry
(long: higher, short: lower), i.e. the registered entry is optimistic."""
import json, subprocess, time
from datetime import datetime, timezone
import numpy as np, requests

URL = "https://api-futures.kucoin.com/api/v1/kline/query"
TR = {"C11": ("candidate11_paper_state.json", -1), "U67-S": ("u67_short_paper_state.json", -1), "U67-L": ("u67_long_paper_state.json", 1)}
S = requests.Session(); P = lambda s: datetime.fromisoformat(s)
ms = lambda d: int(d.timestamp() * 1000)

def k1m(sym, t0, t1):
    """1m klines {start_ms: (o, c)} for [t0, t1] ms; None if the API fails."""
    for k in range(4):
        try:
            r = S.get(URL, params={"symbol": sym, "granularity": 1, "from": t0, "to": t1}, timeout=25)
            p = r.json()
            if r.status_code == 200 and p.get("code") == "200000":
                out = {}
                for x in p.get("data") or []:
                    t = int(x[0]); t = t * 1000 if t < 10**10 else t; out[t] = (float(x[1]), float(x[4]))
                return out
        except Exception: pass
        time.sleep(2 ** k)
    return None

def commit_time(path, sid):
    o = subprocess.run(["git", "log", "--reverse", "-S", sid, "--format=%cI", "--", path], capture_output=True, text=True).stdout.split()
    return P(o[0]).astimezone(timezone.utc) if o else None

def stats(x):
    x = np.array(x); q = np.percentile(x, [10, 25, 50, 75, 90])
    return f"mean {x.mean():+.1f}, median {q[2]:+.1f}, p10/p25/p75/p90 {q[0]:+.1f}/{q[1]:+.1f}/{q[3]:+.1f}/{q[4]:+.1f}, min {x.min():+.1f}, max {x.max():+.1f}"

def main():
    print("N5 ENTRY REALISM (proxy; commit-time price vs registered entry-candle open; no P/L, R, exits or stops read)")
    for tr, (f, d) in TR.items():
        st = json.load(open(f)); sg = list(st["signals"].values()) if isinstance(st["signals"], dict) else st["signals"]
        raw, adv, lag, bad, noc, mism = [], [], [], 0, 0, 0
        for s in sg:
            T = commit_time(f, s["id"])
            if T is None: noc += 1; continue
            te = ms(P(s["entry_ts"])); tc = ms(T) // 60000 * 60000 - 60000            # last complete 1m candle before commit
            ks = k1m(s["symbol"], te, tc + 60000)
            if not ks or te not in ks or tc not in ks: bad += 1; continue
            e, pc = ks[te][0], ks[tc][1]
            if abs(e / s["entry"] - 1) > 1e-6: mism += 1
            raw.append((pc / e - 1) * 1e4); adv.append(d * (pc / e - 1) * 1e4); lag.append((ms(T) - te) / 60000)
        print(f"\n{tr}: signals {len(sg)}, measured {len(adv)}, no commit found {noc}, kline missing {bad}, entry-open != state entry (1m open vs recorded) {mism}")
        if not adv: continue
        print("  commit lag after entry candle open (min):", stats(lag))
        print("  raw price change entry-open -> commit-time price (bps):", stats(raw))
        print("  adverse_bps (direction-adjusted, >0 = registered entry optimistic):", stats(adv))
        a = np.array(adv); print(f"  share adverse>0 {np.mean(a > 0):.2f}, >=10 bps {np.mean(a >= 10):.2f}, >=30 bps {np.mean(a >= 30):.2f}, ==0 {np.mean(a == 0):.2f}; |adverse| median {np.median(np.abs(a)):.1f} bps vs assumed 30 bps round-trip cost")
    print("\nLONG-M1: no signals (0). Nothing was written to the repository.")

if __name__ == "__main__": main()
