# ============================================================
# U67 UNIVERSE RE-SELECTION (OPTION B) - ONE-SHOT, READ-ONLY
# Uses ONLY the already-recorded snapshot u67_universe_snapshot.json
# (no new contracts snapshot, no new turnover ranking).
# Applies the designer's exclusion list + evidence-based non-crypto
# exclusions, then candle-checks only candidates not yet checked.
# Reads no outcome data, runs no strategy, places no order.
# Output: u67_universe_final.json
# ============================================================
import hashlib
import json
import os
import sys

import select_u67_universe as s1  # candle_check (engine rule), hashing

SNAPSHOT_FILE = os.environ.get("U67_SNAPSHOT_FILE", "u67_universe_snapshot.json")
OUT_FILE = os.environ.get("U67_FINAL_FILE", "u67_universe_final.json")

EXPECTED_SNAPSHOT_SHA256 = (
    "3e9f1ba5903c3c73fe44f5c74e3f8aeddb2e046d7daa9cb43851de9e3930cc2a"
)
EXPECTED_RAW_RESPONSE_SHA256 = (
    "6176f7deb0c78c08eeca342a3dfdd185e84d6cee4e943e0c1dba559dbf9d0f02"
)
TARGET_N = 67
MAX_RANK_CONSIDERED = 100  # candidates beyond this rank are NOT classified

# Designer ruling (Q1, option B): fixed exclusion list.
DESIGNER_EXCLUDED = [
    "PAXGUSDTM", "XAUTUSDTM", "XAGUSDTM", "MUUSDTM", "SNDKUSDTM",
    "SKHYNIXUSDTM", "CRCLUSDTM", "SOXLUSDTM", "SOXSUSDTM", "SPCXUSDTM",
]

# Same policy for replacement candidates whose underlying is DEMONSTRABLY
# non-crypto. Evidence = KuCoin listing announcements (title/underlying text).
# Default for every other symbol: treated as crypto (burden of proof is on
# exclusion).
NON_CRYPTO_EVIDENCED = {
    "CLUSDTM": "KuCoin Futures new listing CLUSDT (crude oil perpetual)",
    "BZUSDTM": "KuCoin announcement 2026-03-31: BZUSDT & NATGASUSDT (Brent / natural gas)",
    "LITEUSDTM": "KuCoin announcement 2026-05-08: LITEUSDT Stock Index Perpetual",
    "TSLAUSDTM": "KuCoin TradFi stock contract (Tesla)",
}
# Additional evidenced non-crypto contracts (from KuCoin stock-index announcements read 2026-10-28).
NON_CRYPTO_EVIDENCED.update({
    "APPUSDTM": "2026-10-12 stock index announcement (Applovin)",
    "TMFUSDTM": "2026-10-12 stock index announcement (ICE US Treasury 20+ Year Bond Index)",
    "SKUUUSDTM": "2026-10-12 stock index announcement (2x Long SK Hynix ETF)",
    "MRNAUSDTM": "2026-10-12 stock index announcement (Moderna)",
    "SOFIUSDTM": "2026-10-12 stock index announcement (SoFi)",
    "AGPUUSDTM": "2026-09-24 stock index announcement (Axe Compute stock)",
    "CYPHUSDTM": "2026-09-24 stock index announcement (Cypherpunk Technologies stock)",
    "CSOPSKHYNIX2LUSDTM": "2026-09-24 stock index announcement (CSOP SK Hynix 2x product)",
    "CSOPSAMSUNG2LUSDTM": "2026-09-24 stock index announcement (CSOP Samsung 2x product)",
    "URNMUSDTM": "2026-10-21 stock index announcement (Sprott Uranium Miners ETF)",
    "GSUSDTM": "2026-10-21 stock index announcement (Goldman Sachs)",
    "MSTUUSDTM": "2026-10-21 stock index announcement (2x Long MSTR ETF)",
    "FWDIUSDTM": "2026-10-21 stock index announcement (Forward Industries)",
    "USDEXUSDTM": "2026-10-21 stock index announcement (StablecoinX Inc. stock)",
    "PYPLUSDTM": "2026-10-27 stock index announcement (PayPal)",
    "GEVUSDTM": "2026-10-27 stock index announcement (GE Vernova)",
    "NETUSDTM": "2026-10-27 stock index announcement (Cloudflare)",
    "PANWUSDTM": "2026-10-27 stock index announcement (Palo Alto Networks)",
    "TXNUSDTM": "2026-10-27 stock index announcement (Texas Instruments)",
    "KOUSDTM": "2026-08-19 stock index announcement (Coca-Cola)",
    "LYTEUSDTM": "2026-08-19 stock index announcement (Roundhill Photonics & Optics ETF)",
    "SAMSUNGEMUSDTM": "2026-08-19 stock index announcement (Samsung Electro-Mechanics)",
    "HANMIUSDTM": "2026-08-19 stock index announcement (HANMI Semiconductor)",
    "LRCXUSDTM": "2026-06-24 stock index announcement (Lam Research)",
    "KLACUSDTM": "2026-06-24 stock index announcement (KLA)",
    "SMCIUSDTM": "2026-06-24 stock index announcement (Super Micro Computer)",
    "CIENUSDTM": "2026-06-24 stock index announcement (Ciena)",
    "KORUUSDTM": "2026-06-24 stock index announcement (Direxion 3X South Korea ETF)",
    "SONYUSDTM": "2026-06-24 stock index announcement (Sony)",
    "BXUSDTM": "2026-06-08 stock index announcement (Blackstone)",
    "AMATUSDTM": "2026-06-08 stock index announcement (Applied Materials)",
    "CRWDUSDTM": "2026-06-08 stock index announcement (CrowdStrike)",
    "CRDOUSDTM": "2026-06-08 stock index announcement (Credo)",
    "IWMUSDTM": "2026-06-08 stock index announcement (iShares Russell 2000 ETF)",
    "AXTIUSDTM": "2026-06-08 stock index announcement (AXT Inc.)",
    "BEUSDTM": "2026-05-22 stock index announcement (Bloom Energy)",
    "APLDUSDTM": "2026-05-22 stock index announcement (Applied Digital)",
    "ASTSUSDTM": "2026-05-22 stock index announcement (AST SpaceMobile)",
    "VRTUSDTM": "2026-05-22 stock index announcement (Vertiv)",
})

# Listing-channel evidence that the replacement candidates in the
# considered range are crypto assets (informational; default is crypto).
CRYPTO_CHANNEL_EVIDENCE = {
    "POWERUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "CHIPUSDTM": "KuCoin new listing 'Pre-Market Trading' announcement",
    "GENIUSUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "AEONUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "FOLKSUSDTM": "KuCoin new listing 'Up to 50x Leverage' announcement",
    "TRUSTUSDTM": "KuCoin new listing 'Up to 50x Leverage' announcement",
    "BLENDUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "BILLUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "LABUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "PROSUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "EDGEUSDTM": "KuCoin new listing 'Pre-Market Trading' announcement",
    "SPXUSDTM": "SPX6900 token listed on KuCoin (crypto), not an equity index",
    "CTRUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "OUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
    "BASEDUSDTM": "KuCoin new listing 'Up to 20x Leverage' announcement",
}


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    raw = open(SNAPSHOT_FILE, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != EXPECTED_SNAPSHOT_SHA256:
        print(f"SNAPSHOT_HASH_MISMATCH got={got}")
        sys.exit(4)
    snap = json.loads(raw.decode("utf-8"))
    if snap["raw_response_sha256"] != EXPECTED_RAW_RESPONSE_SHA256:
        print("RAW_RESPONSE_HASH_MISMATCH")
        sys.exit(4)

    # Rebuild the SAME ranking from the stored rows (no new request).
    ranked = [
        r for r in snap["all_contracts"]
        if r["prefilter_reason"] is None and not r["in_existing_33"]
    ]
    ranked.sort(key=lambda r: (-float(r["turnoverOf24h"]), r["symbol"]))
    if len(ranked) != snap["n_prefilter_pass_excluding_existing_33"]:
        print("RANKING_COUNT_MISMATCH")
        sys.exit(4)

    prev = {e["symbol"]: e for e in snap["examined"]}
    # consistency: examined symbols must be exactly the first len(examined) ranks
    for i, e in enumerate(snap["examined"]):
        if ranked[i]["symbol"] != e["symbol"]:
            print("RANKING_ORDER_MISMATCH at", i + 1)
            sys.exit(4)

    designer = set(DESIGNER_EXCLUDED)
    log = []
    selected = []
    new_checks = 0
    stopped = None

    for rank, r in enumerate(ranked, start=1):
        if len(selected) >= TARGET_N:
            break
        sym = r["symbol"]
        if rank > MAX_RANK_CONSIDERED:
            stopped = f"RANK_LIMIT_{MAX_RANK_CONSIDERED}_REACHED_UNCLASSIFIED_BEYOND"
            break
        row = {"rank": rank, "symbol": sym, "turnoverOf24h": r["turnoverOf24h"]}
        if sym in designer:
            row.update(action="EXCLUDED", reason="DESIGNER_EXCLUDED_NON_CRYPTO_OR_UNVERIFIED")
        elif sym in NON_CRYPTO_EVIDENCED:
            row.update(action="EXCLUDED", reason="NON_CRYPTO_EVIDENCED",
                       evidence=NON_CRYPTO_EVIDENCED[sym])
        else:
            if sym in prev:
                chk = {k: prev[sym][k] for k in ("completed", "contiguous_ok", "detail", "eligible")}
                chk["source"] = "snapshot_run_2026-10-28T05:28Z"
            else:
                chk = s1.candle_check(sym)
                chk["source"] = "reselect_run"
                chk["checked_utc"] = s1.c11.utc_now().isoformat()
                new_checks += 1
            row["candle"] = chk
            if sym in CRYPTO_CHANNEL_EVIDENCE:
                row["crypto_evidence"] = CRYPTO_CHANNEL_EVIDENCE[sym]
            else:
                row["crypto_evidence"] = "default: no evidence of non-crypto underlying"
            if chk["eligible"]:
                row.update(action="SELECTED")
                selected.append(sym)
            else:
                row.update(action="REJECTED", reason="CANDLE_RULE:" + str(chk["detail"]))
        log.append(row)
        print(f"{rank:3d} {sym:<16} {row['action']:<9} {row.get('reason','')}")

    shortfall = len(selected) < TARGET_N
    out = {
        "purpose": "U67 final universe (option B), re-selected from the recorded snapshot",
        "snapshot_file": SNAPSHOT_FILE,
        "snapshot_file_sha256": EXPECTED_SNAPSHOT_SHA256,
        "snapshot_raw_response_sha256": EXPECTED_RAW_RESPONSE_SHA256,
        "snapshot_request_start_utc": snap["snapshot_request_start_utc"],
        "snapshot_request_end_utc": snap["snapshot_request_end_utc"],
        "reselect_finished_utc": s1.c11.utc_now().isoformat(),
        "new_candle_checks": new_checks,
        "designer_excluded": DESIGNER_EXCLUDED,
        "non_crypto_evidenced": NON_CRYPTO_EVIDENCED,
        "crypto_channel_evidence": CRYPTO_CHANNEL_EVIDENCE,
        "max_rank_considered": MAX_RANK_CONSIDERED,
        "rule": "static prefilter + ranking from recorded snapshot; designer exclusions; evidenced non-crypto exclusions; engine candle rule (>=250 completed, last 250 contiguous)",
        "selected_symbols": selected,
        "selected_symbols_sha256": sha256_text("\n".join(selected)),
        "shortfall": shortfall,
        "stopped": stopped,
        "decision_log": log,
    }
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, OUT_FILE)
    print(f"SELECTED={len(selected)}/{TARGET_N} SHORTFALL={shortfall} STOPPED={stopped}")
    print("FINAL_SYMBOLS_SHA256=" + out["selected_symbols_sha256"])
    if shortfall:
        sys.exit(3)


if __name__ == "__main__":
    main()
