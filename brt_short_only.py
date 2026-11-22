import crypto_signal_bot_brt_short_v1 as E

# ---- LONG off (SHORT only) ----
_orig_setup_at = E.setup_at


def setup_at_short_only(d, i):
    r = _orig_setup_at(d, i)
    if r is None or r[0] != -1:
        return None
    return r


E.setup_at = setup_at_short_only

# ---- extra symbols (unknown on KuCoin = skipped automatically) ----
MORE_SYMS = [
    "GRAMUSDTM", "LDOUSDTM", "CRVUSDTM", "MKRUSDTM", "SANDUSDTM",
    "MANAUSDTM", "AXSUSDTM", "GALAUSDTM", "APEUSDTM", "DYDXUSDTM",
    "STXUSDTM", "IMXUSDTM", "FETUSDTM", "RENDERUSDTM", "GRTUSDTM",
    "SNXUSDTM", "COMPUSDTM", "1INCHUSDTM", "CHZUSDTM", "EOSUSDTM",
    "THETAUSDTM", "NEOUSDTM", "KAVAUSDTM", "ZILUSDTM", "ENJUSDTM",
    "BLURUSDTM", "PYTHUSDTM", "JTOUSDTM", "STRKUSDTM", "ONDOUSDTM",
    "WUSDTM", "ETHFIUSDTM", "AEVOUSDTM", "NOTUSDTM", "POPCATUSDTM",
    "BONKUSDTM", "SHIBUSDTM", "CAKEUSDTM", "JASMYUSDTM", "GMTUSDTM",
    "ROSEUSDTM", "QNTUSDTM", "EGLDUSDTM", "SUSHIUSDTM", "CFXUSDTM",
    "KASUSDTM", "TRBUSDTM", "ZROUSDTM", "IOUSDTM", "BOMEUSDTM",
    "MOODENGUSDTM", "GOATUSDTM", "VIRTUALUSDTM", "MEMEUSDTM",
    "SUSDTM", "AIXBTUSDTM",
]

for _s in MORE_SYMS:
    if _s not in E.SYMS:
        E.EXTRA_SYMS.append(_s)
        E.SYMS.append(_s)

if __name__ == "__main__":
    E.main()
