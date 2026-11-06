"""
C11 EXECUTION-REALISM + INTRABAR DIAGNOSTIC  (READ-ONLY, diagnostic only)
Does NOT write any file, does NOT touch candidate11_paper_state.json, does NOT commit, places NO orders.
Reads: the frozen records (ATOMUSDTM and BCHUSDTM, signal candle 2026-11-05T12:00Z) from the checked-out state file,
       and PUBLIC KuCoin Futures market data (1-minute and 4-hour klines, recent-trades endpoint).
Output: log only. Direction convention (SHORT): shift_R = (entry_frozen - price_at_processing) / frozen_risk.
        shift_R > 0  -> a real short placed at processing time would have received a WORSE price than the frozen open (frozen entry optimistic).
        shift_R < 0  -> a real short would have received a BETTER price (frozen entry conservative).
Data limits: KuCoin public API has no historical bid/ask and its trade-history endpoint returns only the latest trades,
        so 1-minute OHLCV is used as a PROXY and is labelled as such.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://api-futures.kucoin.com"
KLINE = BASE + "/api/v1/kline/query"
TRADES = BASE + "/api/v1/trade/history"
STATE = "candidate11_paper_state.json"
SYMBOLS = ["ATOMUSDTM", "BCHUSDTM"]
SIGNAL_TS = "2026-11-05T12:00:00+00:00"
ENTRY_TS = datetime(2026, 11, 5, 16, 0, tzinfo=timezone.utc)
END_TS = datetime(2026, 11, 6, 4, 0, tzinfo=timezone.utc)       # last candle checked closes here
DEFAULT_T = datetime(2026, 11, 5, 16, 35, 25, tzinfo=timezone.utc)   # ESTIMATE only (commit 16:36:31Z minus usual run length)
S = requests.Session()
S.headers.update({"User-Agent": "c11-exec-diag/1.0 (read-only)"})


def ms(dt):
    return int(dt.timestamp() * 1000)


def utc(msv):
    return datetime.fromtimestamp(msv / 1000, tz=timezone.utc)


def get(url, params, tries=4):
    last = None
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=25)
            if r.status_code == 200:
                p = r.json()
                if isinstance(p, dict) and p.get("code") == "200000":
                    return p.get("data") or []
                last = "payload " + str(p)[:150]
            else:
                last = f"HTTP {r.status_code} {r.text[:100]}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(2 ** k, 6))
    print("FETCH_FAILED", url, params.get("symbol"), last)
    return None


def klines(symbol, gran, start, end):
    out, cur = {}, start
    step = timedelta(minutes=gran * 400)
    while cur < end:
        nxt = min(end, cur + step)
        rows = get(KLINE, {"symbol": symbol, "granularity": gran, "from": ms(cur), "to": ms(nxt)})
        if rows is None:
            return None
        for r in rows:
            try:
                t = int(r[0])
                if t < 10_000_000_000:
                    t *= 1000
                out[t] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))   # o h l c v
            except (TypeError, ValueError, IndexError):
                continue
        cur = nxt
        time.sleep(0.2)
    return dict(sorted(out.items()))


def shift_r(entry, price, risk):
    """SHORT: positive = real fill worse than frozen entry."""
    return (entry - price) / risk


def touches(bars, sl, tp, t_from=None, t_to=None):
    """bars: {open_ms: (o,h,l,c,v)} 1-minute. Returns first-touch minute for SL (high>=sl) and TP (low<=tp), and whether the same minute hits both."""
    first_sl = first_tp = None
    both_same = []
    for t, (o, h, l, c, v) in bars.items():
        if t_from and utc(t) < t_from:
            continue
        if t_to and utc(t) >= t_to:
            continue
        hs, ht = h >= sl, l <= tp
        if hs and first_sl is None:
            first_sl = utc(t)
        if ht and first_tp is None:
            first_tp = utc(t)
        if hs and ht:
            both_same.append(utc(t))
    return first_sl, first_tp, both_same


def load_records():
    d = json.load(open(STATE, encoding="utf-8"))
    recs = {}
    for x in d["signals"]:
        if x["symbol"] in SYMBOLS and x["signal_ts"] == SIGNAL_TS:
            recs[x["symbol"]] = x
    where = {}
    for sym in SYMBOLS:
        pid = f"{sym}|{SIGNAL_TS}"
        where[sym] = "OPEN" if pid in d["open"] else ("CLOSED" if any(c["id"] == pid for c in d["closed"]) else "NOT_FOUND")
    return recs, where


def parse_T():
    raw = (os.environ.get("RUN_TIME_UTC") or "").strip()
    if not raw:
        return DEFAULT_T, "ESTIMATE (no RUN_TIME_UTC supplied)"
    try:
        t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc), "SUPPLIED from the Actions log (RUN_TIME_UTC)"
    except ValueError:
        print("Could not parse RUN_TIME_UTC:", raw)
        return DEFAULT_T, "ESTIMATE (RUN_TIME_UTC could not be parsed)"


def main():
    recs, where = load_records()
    T, tsrc = parse_T()
    print("C11 EXECUTION-REALISM + INTRABAR DIAGNOSTIC (read-only)")
    print(f"processing time T = {T.isoformat()}  [{tsrc}]")
    print(f"lag from entry open = {(T - ENTRY_TS).total_seconds() / 60:.2f} min")
    print("state file status now:", where)
    for sym in SYMBOLS:
        rec = recs.get(sym)
        print("=" * 70)
        print(sym)
        if not rec:
            print("  frozen signal record not found - skipped")
            continue
        entry, risk, sl, tp = float(rec["entry"]), float(rec["risk"]), float(rec["sl"]), float(rec["tp"])
        print(f"  frozen record: entry {entry}  risk {risk:.8g} ({100 * risk / entry:.3f}%)  SL {sl:.8g}  TP {tp:.8g}  entry_ts {rec['entry_ts']}")
        k4 = klines(sym, 240, ENTRY_TS - timedelta(hours=4), END_TS)
        k1 = klines(sym, 1, ENTRY_TS - timedelta(minutes=5), END_TS)
        if not k4 or not k1:
            print("  DATA UNAVAILABLE - nothing computed for this symbol")
            continue
        print("  4H candles (exchange):")
        for t, (o, h, l, c, v) in k4.items():
            if ENTRY_TS <= utc(t) < END_TS:
                tag = "SL-touch" if h >= sl else ""
                tag += (" TP-touch" if l <= tp else "")
                print(f"    {utc(t):%m-%d %H:%MZ}  O {o}  H {h}  L {l}  C {c}  {tag.strip() or 'no SL/TP touch'}")
        o16 = k4.get(ms(ENTRY_TS))
        if o16:
            print(f"  exchange 4H open at 16:00Z = {o16[0]}  vs frozen entry {entry}  -> {'MATCH' if abs(o16[0] - entry) <= 1e-9 * max(1, entry) else 'DIFFERENT'}")
        print("  1-minute proxy bars (O H L C): 15:59Z-16:02Z and T-6min..T+6min:")
        tmin = T.replace(second=0, microsecond=0)
        for t, (o, h, l, c, v) in k1.items():
            u = utc(t)
            if (datetime(2026, 11, 5, 15, 59, tzinfo=timezone.utc) <= u <= datetime(2026, 11, 5, 16, 2, tzinfo=timezone.utc)
                    or tmin - timedelta(minutes=6) <= u <= tmin + timedelta(minutes=6)):
                print(f"    {u:%H:%MZ}  {o}  {h}  {l}  {c}")
        tm = ms(T.replace(second=0, microsecond=0))
        bar = k1.get(tm)
        if bar:
            o, h, l, c, v = bar
            print(f"  PROXY price at T (1-min bar {utc(tm):%H:%MZ}; NOT a quote): open {o} high {h} low {l} close {c}")
            for nm, p in (("open", o), ("close", c), ("low", l), ("high", h)):
                print(f"    {nm:5s} {p}  diff {p - entry:+.8g} ({100 * (p / entry - 1):+.3f}%)  shift_R {shift_r(entry, p, risk):+.3f}   "
                      f"R-to-TP from here {(p - tp) / risk:.2f}  R-to-SL from here {(sl - p) / risk:.2f}")
        else:
            print("  1-minute bar for T not available")
        fs, ft, both = touches(k1, sl, tp)
        print(f"  1-min touches 16:00Z-04:00Z(11-06): first SL-touch {fs}  first TP-touch {ft}  minutes touching both: {both}")
        pre_s, pre_t, _ = touches(k1, sl, tp, ENTRY_TS, T.replace(second=0, microsecond=0))
        print(f"  touches during the pre-fill interval 16:00Z..T: SL {pre_s}  TP {pre_t}")
        c16 = [v for t, v in k4.items() if utc(t) == ENTRY_TS]
        if c16:
            print(f"  entry candle (16:00-20:00Z) path: high {c16[0][1]}  low {c16[0][2]}  -> SL distance left {(sl - c16[0][1]) / risk:+.2f}R, TP distance left {(c16[0][2] - tp) / risk:+.2f}R")
    print("=" * 70)
    print("RECENT-TRADES ENDPOINT (evidence of what historical tick data is available):")
    for sym in SYMBOLS:
        rows = get(TRADES, {"symbol": sym}) or []
        if rows:
            ts = [int(x.get("ts", 0)) for x in rows if isinstance(x, dict)]
            if ts:
                print(f"  {sym}: {len(ts)} trades returned, oldest {datetime.fromtimestamp(min(ts) / 1e9, tz=timezone.utc).isoformat()} newest {datetime.fromtimestamp(max(ts) / 1e9, tz=timezone.utc).isoformat()}")
        else:
            print(f"  {sym}: none/unavailable")
    print("Bid/ask history: not available from the public API. Nothing was written to the repository.")


if __name__ == "__main__":
    main()
