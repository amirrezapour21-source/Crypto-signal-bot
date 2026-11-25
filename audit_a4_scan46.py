"""READ-ONLY Audit A4: classify, then integrity-scan the 46 symbols of the re-executed historical top-70 that are not in the 69-symbol scope.
Classification is by symbol name unless Binance exchangeInfo (if reachable) states the underlying type. No symbol is excluded."""
import os, re, sys
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, ".")
from audit_a_n0 import MONTHS, UA
from audit_a3_integrity import one

CLS = {"crypto": "AUCTION BB BLZ CYBER FARTCOIN FRONT GUN HIFI HIGH HYPE IO JASMY KAITO LISTA LSK MASK MATIC MUBARAK NIL NOT PARTI PEOPLE PERP PNUT QNT RARE RED RIVER TRB TRUMP TUT ZK",
       "commodity-like": "XAU XAG PAXG CL BZ", "equity/ETF-like": "MU SNDK SOXL KORU SKHYNIX", "uncertain": "AKE SKHY SPCX SNXX"}
LAB = {s + "USDT": k for k, v in CLS.items() for s in v.split()}
SYMS = sorted(LAB)

def api_types():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", headers=UA, timeout=30)
        if r.status_code != 200: return {}, r.status_code
        return {x["symbol"]: f'{x.get("underlyingType")}/{x.get("underlyingSubType")}' for x in r.json()["symbols"]}, 200
    except Exception as e: return {}, type(e).__name__

def main():
    assert len(SYMS) == 46, len(SYMS)
    k, n = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1)); syms = SYMS[k::n]
    api, code = api_types(); print(f"shard {k}/{n}: {len(syms)} symbols; exchangeInfo underlyingType lookup: {'ok' if api else 'unavailable (' + str(code) + ')'}")
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, [(s, m) for s in syms for m in MONTHS]))
    tot = dict(present=0, clean=0, issues=0, missing=0, rows=0, taker=0)
    for s in syms:
        r = [x for x in res if x[0] == s]; pres = [x[1] for x in r if x[3] is not None]; bad = [x for x in r if x[3]]
        lo = MONTHS.index(pres[0]) if pres else None
        unexp = [x[1] for x in r if x[3] is None and lo is not None and MONTHS.index(x[1]) > lo]
        pre = sum(1 for x in r if x[3] is None) - len(unexp)
        tk = sum(int(m) for x in bad for i in x[3] for m in re.findall(r"TAKER_GT_VOL:(\d+)", i))
        tot["present"] += len(pres); tot["clean"] += sum(1 for x in r if x[3] == []); tot["issues"] += len(bad); tot["missing"] += len(r) - len(pres)
        tot["rows"] += sum(x[4] for x in r); tot["taker"] += tk
        head = f"{s} [{LAB[s]}{'; api ' + api[s] if s in api else ''}] present {len(pres)}"
        print(head + (f" {pres[0]}..{pres[-1]}, pre-listing missing {pre}, unexplained missing {len(unexp)}" + (f" {unexp}" if unexp else "") + f", files with issues {len(bad)}" if pres else ", NO FILE IN 2022-11..2026-10 (all 404)"))
        for x in bad: print(f"  ISSUE {s} {x[1]}: {' '.join(x[3])}")
    print(f"shard totals: files present {tot['present']}, clean {tot['clean']}, with issues {tot['issues']}, missing {tot['missing']}, rows {tot['rows']}, taker>volume rows to mask {tot['taker']}")
    print("Nothing was written to the repository.")

if __name__ == "__main__": main()
