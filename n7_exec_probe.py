"""N7 execution-cost STEP 0 - one manual read-only validation run (not a collector). Public KuCoin Futures endpoints only; no credentials, nothing written.
Universe = symbols of the four registered tracks (read from their state files' 'symbols' key only; no outcome fields are read).
Liquid majors = XBTUSDTM, ETHUSDTM; three lower-liquidity contracts = spread-rank percentiles 50 / 80 / 95 of the track universe in this run (deterministic rule, no cherry-picking).
Impact = VWAP of a market order of the stated USDT notional walked through depth20 versus mid (includes half the spread). PROXY only: a snapshot, not a fill; no queue, latency or stress."""
import json, statistics, sys, time
import numpy as np, requests

B = "https://api-futures.kucoin.com"; S = requests.Session(); ST = dict(calls=0, retries=0, r429=0, failed=0)
FILES = ["candidate11_paper_state.json", "u67_short_paper_state.json", "u67_long_paper_state.json", "long_m1_paper_state.json"]
NOT = (1000, 5000, 20000); MAJ = ["XBTUSDTM", "ETHUSDTM"]; ASSUMED_RT = 30.0

def get(path, params=None):
    for k in range(4):
        t0 = time.time() * 1000
        try:
            r = S.get(B + path, params=params, timeout=20); t1 = time.time() * 1000; ST["calls"] += 1
            if r.status_code == 429:
                ST["r429"] += 1; ST["retries"] += 1
                try: w = float(r.headers.get("gw-ratelimit-reset", 2000)) / 1000
                except ValueError: w = 2.0
                time.sleep(min(max(w, 1.0), 10.0)); continue
            if r.status_code == 200:
                p = r.json()
                if p.get("code") == "200000": return p["data"], (t0 + t1) / 2, t1 - t0
            ST["retries"] += 1
        except (requests.RequestException, ValueError): ST["retries"] += 1
        time.sleep(2 ** k)
    ST["failed"] += 1; return None, None, None

def unit(ts):
    n = len(str(int(ts))); return {19: ("ns", 1e6), 16: ("us", 1e3), 13: ("ms", 1.0)}.get(n, ("?", None))

def walk(levels, target, mid, side):
    """levels [(price, usdt_notional)] best first; -> impact bps vs mid (buy +, sell +), or (None, covered_usdt) if depth is insufficient."""
    rem, cost, got = target, 0.0, 0.0
    for p, v in levels:
        t = min(v, rem); cost += p * t; got += t; rem -= t
        if rem <= 1e-9: break
    if rem > 1e-9: return None, got
    vw = cost / got; return ((vw / mid - 1) if side == "buy" else (1 - vw / mid)) * 1e4, got

def main():
    uni = sorted(set().union(*[set(json.load(open(f))["symbols"]) for f in FILES])); print(f"universe: {len(uni)} contracts from the 4 registered tracks")
    snaps = []
    for i in range(2):
        d, tm, lat = get("/api/v1/allTickers"); snaps.append((d, tm, lat)); time.sleep(2)
    if any(s[0] is None for s in snaps): print("allTickers FAILED"); return
    A = [{x["symbol"]: x for x in s[0]} for s in snaps]; print(f"allTickers: {len(A[0])} contracts returned, call latency {snaps[0][2]:.0f}/{snaps[1][2]:.0f} ms; universe present {sum(s in A[0] for s in uni)}/{len(uni)}")
    inv, spr, offs, units, mono = [], {}, [], {}, [0, 0]
    for s in uni:
        x = A[0].get(s)
        try: b, a, bs, as_ = float(x["bestBidPrice"]), float(x["bestAskPrice"]), float(x["bestBidSize"]), float(x["bestAskSize"])
        except (TypeError, ValueError, KeyError): inv.append(s); continue
        if not (b > 0 and a > b and bs > 0 and as_ > 0): inv.append(s); continue
        spr[s] = (a - b) / ((a + b) / 2) * 1e4
        try:
            u, div = unit(x["ts"]); units[u] = units.get(u, 0) + 1
            if div: offs.append(snaps[0][1] - float(x["ts"]) / div)
            y = A[1].get(s); mono[1] += 1; mono[0] += int(float(y["ts"]) >= float(x["ts"]) and int(y["sequence"]) >= int(x["sequence"]))
        except (TypeError, ValueError, KeyError): inv.append(s + "(ts/seq)")
    v = np.array(sorted(spr.values())); print(f"bid/ask validity: valid {len(spr)}/{len(uni)}, invalid {inv}")
    print(f"quoted spread bps (universe): min {v.min():.1f}, p25 {np.percentile(v, 25):.1f}, median {np.median(v):.1f}, p75 {np.percentile(v, 75):.1f}, p90 {np.percentile(v, 90):.1f}, max {v.max():.1f}")
    o = np.array(offs); print(f"ts units {units}; receive time minus ticker ts (ms): median {np.median(o):.0f}, p5 {np.percentile(o, 5):.0f}, p95 {np.percentile(o, 95):.0f}; two calls 2 s apart: ts and sequence non-decreasing for {mono[0]}/{mono[1]}")
    rk = sorted((s for s in spr if s not in MAJ), key=lambda s: spr[s]); pick = MAJ + [rk[round(p * (len(rk) - 1))] for p in (0.50, 0.80, 0.95)]
    print("\nselected:", pick, "(majors fixed; others = spread percentiles 50/80/95)\n")
    ok = True
    for s in pick:
        c, _, _ = get(f"/api/v1/contracts/{s}"); d1, t1, _ = get("/api/v1/level2/depth20", {"symbol": s}); time.sleep(1); d2, t2, _ = get("/api/v1/level2/depth20", {"symbol": s})
        if not (c and d1 and d2 and d1["bids"] and d1["asks"]): print(f"{s}: FAILED contract/depth fetch"); ok = False; continue
        m = float(c["multiplier"]); tk, mk = float(c.get("takerFeeRate", "nan")), float(c.get("makerFeeRate", "nan")); inv_c = bool(c.get("isInverse"))
        bid = [(float(p), float(q) * m * float(p)) for p, q in d1["bids"]]; ask = [(float(p), float(q) * m * float(p)) for p, q in d1["asks"]]
        mid = (bid[0][0] + ask[0][0]) / 2; u, div = unit(d1["ts"]); lag = t1 - float(d1["ts"]) / div if div else float("nan")
        cons = abs(bid[0][0] / float(A[0][s]["bestBidPrice"]) - 1) * 1e4; seq_ok = int(d2["sequence"]) >= int(d1["sequence"]) and float(d2["ts"]) >= float(d1["ts"])
        buy = [walk(ask, n, mid, "buy") for n in NOT]; sell = [walk(bid, n, mid, "sell") for n in NOT]
        print(f"{s}: mid {mid:.6g}, quoted spread {(ask[0][0] - bid[0][0]) / mid * 1e4:.1f} bps, levels bid/ask {len(bid)}/{len(ask)}, depth20 covers {sum(x[1] for x in bid):,.0f}/{sum(x[1] for x in ask):,.0f} USDT, ts unit {u}, recv-ts {lag:.0f} ms, depth seq/ts monotonic {seq_ok}, best bid vs ticker {cons:.1f} bps, inverse {inv_c}")
        print(f"   contract fee rates (public base rates, NOT the owner's tier): taker {tk * 1e4:.1f} bps, maker {mk * 1e4:.1f} bps; multiplier {m}")
        for n, (bi, bc), (si, sc) in zip(NOT, buy, sell):
            if bi is None or si is None: print(f"   {n:>6,} USDT: depth20 insufficient (covers buy {bc:,.0f} / sell {sc:,.0f})"); continue
            g = bi + si; print(f"   {n:>6,} USDT: impact buy {bi:.1f} / sell {si:.1f} bps -> gross market cost round trip {g:.1f} bps; illustrative incl. taker fees both sides {g + 2 * tk * 1e4:.1f} bps (assumption in studies: {ASSUMED_RT:.0f})")
        ok &= bool(seq_ok and len(bid) >= 10 and len(ask) >= 10 and not inv_c)
    print(f"\nAPI: calls {ST['calls']}, retries {ST['retries']}, 429 {ST['r429']}, failed {ST['failed']}")
    chk = [("all 100 quotes valid", len(spr) == len(uni)), ("ts unit single and recognised", len(units) == 1 and "?" not in units), ("receive-minus-ts median within +-5 s", abs(np.median(o)) < 5000),
           ("ts/sequence non-decreasing", mono[0] == mono[1]), ("depth/contract fetch and depth>=10 levels for all 5", ok), ("no failed calls", ST["failed"] == 0)]
    for n, r in chk: print(" ", n, "->", "PASS" if r else "FAIL")
    print("METHOD:", "RELIABLE enough to justify the bounded observation" if all(r for _, r in chk) else "NOT YET RELIABLE (see FAIL items)"); print("Nothing was written to the repository.")

if __name__ == "__main__": main()
