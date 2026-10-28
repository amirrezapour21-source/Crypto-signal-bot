# ============================================================
# U67 UNIVERSE SELECTION (ONE-SHOT, READ-ONLY, NO TRADING)
# Pure function of ONE KuCoin Futures snapshot (contracts/active)
# plus the engine's own candle-availability rule.
# It reads no trade outcome, runs no strategy, places no order.
# Output: u67_universe_snapshot.json
# ============================================================
import hashlib
import json
import os
import sys
import time

import pandas as pd

import crypto_signal_bot_v4 as c11  # reused read-only: fetch + candle rules

OUT_FILE = os.environ.get("U67_SNAPSHOT_FILE", "u67_universe_snapshot.json")
CONTRACTS_URL = c11.BASE + "/api/v1/contracts/active"
TARGET_N = 67

STABLE_BASES = {
    "USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDD", "USDE", "PYUSD",
    "BUSD", "USDP", "USD1", "EURC", "EURT", "USDS", "USDG", "RLUSD",
}

KEEP_FIELDS = [
    "symbol", "rootSymbol", "type", "baseCurrency", "quoteCurrency",
    "settleCurrency", "status", "isInverse", "isQuanto", "expireDate",
    "settleDate", "firstOpenDate", "turnoverOf24h", "volumeOf24h",
    "openInterest", "lastTradePrice",
]


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def get_contracts():
    last = None
    for attempt in range(1, 5):
        try:
            r = c11.SESSION.get(CONTRACTS_URL, timeout=c11.REQUEST_TIMEOUT)
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict) and payload.get("code") == "200000":
                    return r.text, payload.get("data") or []
                last = RuntimeError("unexpected payload: " + str(payload)[:180])
            else:
                last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last = e
        time.sleep(min(2 ** (attempt - 1), 6))
    raise RuntimeError(f"CONTRACTS_FETCH_FAILED: {last}")


def prefilter_reason(c):
    """Return None if the contract passes the static rules, else a reason."""
    if str(c.get("quoteCurrency")) != "USDT" or str(c.get("settleCurrency")) != "USDT":
        return "NOT_LINEAR_USDT"
    if c.get("isInverse") is not False:
        return "INVERSE"
    if str(c.get("status")) != "Open":
        return "STATUS_NOT_OPEN:" + str(c.get("status"))
    if c.get("expireDate") or c.get("settleDate"):
        return "HAS_EXPIRY_OR_SETTLE_DATE"
    if str(c.get("baseCurrency")).upper() in STABLE_BASES:
        return "STABLECOIN_BASE"
    if not str(c.get("symbol", "")).endswith("USDTM"):
        return "SYMBOL_NOT_USDTM"
    try:
        t = float(c.get("turnoverOf24h"))
    except Exception:
        return "NO_TURNOVER"
    if not (t > 0):
        return "NO_TURNOVER"
    return None


def candle_check(symbol):
    """Engine's own availability rule. Raises on persistent fetch failure."""
    last = None
    for attempt in range(1, 4):
        try:
            now = c11.utc_now()
            raw = c11.fetch_raw(symbol, now)
            completed = c11.completed_only(raw, now)
            ok, msg = c11.check_contiguous(completed)
            return {
                "completed": int(len(completed)),
                "contiguous_ok": bool(ok),
                "detail": msg,
                "eligible": bool(ok and len(completed) >= c11.MIN_CANDLES),
            }
        except Exception as e:
            last = e
            time.sleep(3)
    raise RuntimeError(f"CANDLE_CHECK_FAILED {symbol}: {last}")


def main():
    t0 = c11.utc_now()
    raw_text, contracts = get_contracts()
    t1 = c11.utc_now()

    existing = set(c11.SYMS)
    rows = []
    for c in contracts:
        row = {k: c.get(k) for k in KEEP_FIELDS}
        row["prefilter_reason"] = prefilter_reason(c)
        row["in_existing_33"] = c.get("symbol") in existing
        rows.append(row)

    passing = [
        r for r in rows
        if r["prefilter_reason"] is None and not r["in_existing_33"]
    ]
    passing.sort(key=lambda r: (-float(r["turnoverOf24h"]), r["symbol"]))

    selected = []
    examined = []
    for rank, r in enumerate(passing, start=1):
        if len(selected) >= TARGET_N:
            break
        chk = candle_check(r["symbol"])
        examined.append({
            "turnover_rank_among_prefiltered": rank,
            "symbol": r["symbol"],
            "turnoverOf24h": r["turnoverOf24h"],
            **chk,
        })
        print(f"{rank:3d} {r['symbol']:<14} turnover={r['turnoverOf24h']} "
              f"completed={chk['completed']} eligible={chk['eligible']}")
        if chk["eligible"]:
            selected.append(r["symbol"])

    shortfall = len(selected) < TARGET_N
    out = {
        "purpose": "U67 universe selection (single recorded snapshot)",
        "snapshot_request_start_utc": t0.isoformat(),
        "snapshot_request_end_utc": t1.isoformat(),
        "candle_checks_finished_utc": c11.utc_now().isoformat(),
        "contracts_url": CONTRACTS_URL,
        "raw_response_sha256": sha256_text(raw_text),
        "n_contracts_in_snapshot": len(contracts),
        "n_prefilter_pass_excluding_existing_33": len(passing),
        "rules": {
            "linear_usdt_perpetual": "quote=settle=USDT, isInverse False, status Open, no expireDate/settleDate, symbol endswith USDTM",
            "stablecoin_bases_excluded": sorted(STABLE_BASES),
            "exclude_existing_33": sorted(existing),
            "ranking": "turnoverOf24h descending, tie by symbol ascending",
            "candle_rule": "engine rule: completed 4H candles >= 250 and last 250 contiguous (check_contiguous), via the engine's own fetch_raw/completed_only",
            "target_n": TARGET_N,
        },
        "selected_symbols": selected,
        "selected_symbols_sha256": sha256_text("\n".join(selected)),
        "shortfall": shortfall,
        "examined": examined,
        "all_contracts": rows,
    }
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, OUT_FILE)
    print(f"SELECTED={len(selected)}/{TARGET_N} SHORTFALL={shortfall}")
    print("SELECTED_SYMBOLS_SHA256=" + out["selected_symbols_sha256"])


if __name__ == "__main__":
    main()
