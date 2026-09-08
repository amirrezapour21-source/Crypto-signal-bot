"""Setup Family C — Cycle 2: In-Sample Edge Discovery"""
import requests, pandas as pd, time

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"
TEST_SYMBOLS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT","DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
    "NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT","INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
    "LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT","UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
    "ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT","THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
    "COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT","GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
    "1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT","ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
    "ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT","OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT",
]
LOOKBACK_RECENT = 120
MIN_PENETRATION_ATR = 0.10
MAX_FAILURE_CANDLES = 3
MAX_HOLD_CANDLES = 30
FEE_PCT = 0.10


def safe_get(url, params=None, retries=3):
    resp = None
    for a in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException:
            if a < retries-1: time.sleep(5)
            continue
        if resp.status_code == 200: return resp
        if resp.status_code == 429: time.sleep(10*(a+1)); continue
        return resp
    return resp


def fetch_kucoin_candles(symbol, end_at=None):
    params = {"symbol": symbol, "type": "4hour"}
    if end_at: params["endAt"] = int(end_at)
    resp = safe_get(KUCOIN_BASE, params=params)
    if resp is None or resp.status_code != 200: return None
    try: data = resp.json()
    except ValueError: return None
    if data.get("code") != "200000": return None
    raw = data.get("data", [])
    if not raw: return None
    rows = [{"time":int(r[0]),"open":float(r[1]),"close":float(r[2]),"high":float(r[3]),"low":float(r[4]),"volume":float(r[5])} for r in raw if len(r)>=6]
    if not rows: return None
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def get_ohlcv(symbol, total=250):
    dfs, end_at, remaining = [], None, total
    while remaining > 0:
        df = fetch_kucoin_candles(symbol, end_at)
        if df is None or df.empty: break
        dfs.append(df); remaining -= len(df); end_at = int(df["time"].min())-1
        time.sleep(0.15)
        if len(df) < 100: break
    if not dfs: return None
    full = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    full["dt"] = pd.to_datetime(full["time"], unit="s", utc=True)
    return full.tail(total).reset_index(drop=True)


def drop_unclosed(df):
    if df is None or df.empty: return df
    if int(time.time()) < int(df["time"].iloc[-1]) + 4*3600:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def find_swings(df, left=3, right=3):
    h, l = df["high"].values, df["low"].values
    sh, sl = [], []
    for i in range(left, len(df)-right):
        wh, wl = h[i-left:i+right+1], l[i-left:i+right+1]
        if h[i] == wh.max() and (wh==h[i]).sum()==1: sh.append({"index":i,"price":h[i]})
        if l[i] == wl.min() and (wl==l[i]).sum()==1: sl.append({"index":i,"price":l[i]})
    return sh, sl


def avg_range(df, lb=20): return (df["high"]-df["low"]).rolling(lb).mean()


def detect_failed_breakouts(df, sh, sl, ar, lookback=LOOKBACK_RECENT):
    events = []
    start = max(0, len(df)-lookback)
    last_r, last_s = None, None
    for i in range(start, len(df)):
        row = df.iloc[i]
        atr = ar.iloc[i]
        if pd.isna(atr) or atr == 0: continue
        pen = MIN_PENETRATION_ATR*atr
        rh = [s for s in sh if s["index"] < i]
        if rh:
            lvl = rh[-1]["price"]
            if row["high"] > lvl+pen and lvl != last_r:
                fi, ft = None, None
                if row["close"] < lvl: fi, ft = i, "same_candle"
                else:
                    for j in range(i+1, min(i+MAX_FAILURE_CANDLES+1, len(df))):
                        if df["close"].iloc[j] < lvl: fi, ft = j, "multi_bar"; break
                if fi is not None:
                    events.append({"direction":"bearish","level":lvl,"breakout_idx":i,"failure_idx":fi,"type":ft})
                    last_r = lvl
        rl = [s for s in sl if s["index"] < i]
        if rl:
            lvl = rl[-1]["price"]
            if row["low"] < lvl-pen and lvl != last_s:
                fi, ft = None, None
                if row["close"] > lvl: fi, ft = i, "same_candle"
                else:
                    for j in range(i+1, min(i+MAX_FAILURE_CANDLES+1, len(df))):
                        if df["close"].iloc[j] > lvl: fi, ft = j, "multi_bar"; break
                if fi is not None:
                    events.append({"direction":"bullish","level":lvl,"breakout_idx":i,"failure_idx":fi,"type":ft})
                    last_s = lvl
    return events


def track_excursion(df, direction, entry, atr, start_idx, max_hold=MAX_HOLD_CANDLES):
    end_idx = min(start_idx+1+max_hold, len(df))
    mfe, mae = 0, 0
    pos_l = [0.5,1.0,1.5,2.0,3.0]; neg_l = [0.5,0.75,1.0,1.25,1.5]
    rp = {lv:None for lv in pos_l}; rn = {lv:None for lv in neg_l}
    for step, i in enumerate(range(start_idx+1, end_idx), 1):
        row = df.iloc[i]
        if direction == "bullish":
            fav, adv = row["high"]-entry, entry-row["low"]
        else:
            fav, adv = entry-row["low"], row["high"]-entry
        mfe, mae = max(mfe,fav), max(mae,adv)
        fa, aa = fav/atr, adv/atr
        for lv in pos_l:
            if rp[lv] is None and fa >= lv: rp[lv] = step
        for lv in neg_l:
            if rn[lv] is None and aa >= lv: rn[lv] = step
    return {"mfe_atr":round(mfe/atr,3),"mae_atr":round(mae/atr,3),"rp":rp,"rn":rn}


def sim_fixed(df, direction, entry, sl_atr, tp_mult, atr, start_idx, max_hold=MAX_HOLD_CANDLES):
    if direction == "bullish":
        sl = entry-sl_atr*atr; tp = entry+(sl_atr*atr*tp_mult)
    else:
        sl = entry+sl_atr*atr; tp = entry-(sl_atr*atr*tp_mult)
    end_idx = min(start_idx+1+max_hold, len(df))
    for i in range(start_idx+1, end_idx):
        row = df.iloc[i]
        if direction == "bullish":
            sl_hit, tp_hit = row["low"]<=sl, row["high"]>=tp
        else:
            sl_hit, tp_hit = row["high"]>=sl, row["low"]<=tp
        if sl_hit: return -1.0
        if tp_hit: return tp_mult
    return None


def pctl(vals, p):
    if not vals: return None
    s = sorted(vals); k=(len(s)-1)*p; f=int(k); c=f+1 if f+1<len(s) else f
    return s[f] if f==c else s[f]+(s[c]-s[f])*(k-f)


def report_group(name, entries):
    N = len(entries)
    print(f"\n--- {name} | N={N} ---")
    if N == 0: return
    mfe = [e["exc"]["mfe_atr"] for e in entries]; mae = [e["exc"]["mae_atr"] for e in entries]
    print(f"MFE mean={sum(mfe)/N:.2f} median={pctl(mfe,0.5):.2f} P25={pctl(mfe,0.25):.2f} P75={pctl(mfe,0.75):.2f}")
    print(f"MAE mean={sum(mae)/N:.2f} median={pctl(mae,0.5):.2f} P25={pctl(mae,0.25):.2f} P75={pctl(mae,0.75):.2f}")
    print(f"P(MFE>=1ATR)={round(sum(1 for v in mfe if v>=1)/N*100,1)}% P(MFE>=2ATR)={round(sum(1 for v in mfe if v>=2)/N*100,1)}%")
    print(f"P(MAE<=1ATR)={round(sum(1 for v in mae if v<=1)/N*100,1)}% P(MAE<=1.5ATR)={round(sum(1 for v in mae if v<=1.5)/N*100,1)}%")
    for pl, nl in [(1.0,0.75),(1.0,1.0),(2.0,1.0),(2.0,1.25),(3.0,1.5)]:
        cp, cv = 0, 0
        for e in entries:
            rp_v, rn_v = e["exc"]["rp"].get(pl), e["exc"]["rn"].get(nl)
            if rp_v is None and rn_v is None: continue
            cv += 1
            if rp_v is not None and (rn_v is None or rp_v<=rn_v): cp += 1
        print(f"  P(+{pl} before -{nl}): {round(cp/cv*100,1) if cv else None}% (n={cv})")
    print("  Fixed SL=1.0 ATR trade matrix:")
    for tp in [1.0,1.5,2.0,3.0]:
        outs = [sim_fixed(e["df"], e["direction"], e["entry"], 1.0, tp, e["atr"], e["idx"]) for e in entries]
        outs = [o for o in outs if o is not None]
        if outs:
            wins=[o for o in outs if o>0]; losses=[o for o in outs if o<0]
            exp = sum(outs)/len(outs)
            pf = (sum(wins)/abs(sum(losses))) if losses and sum(losses)!=0 else None
            net_exp = exp - FEE_PCT/100
            print(f"    TP={tp}R N={len(outs)} WinRate={round(len(wins)/len(outs)*100,1)}% Exp={round(exp,3)} PF={round(pf,2) if pf else 'N/A'} NetExp={round(net_exp,3)}")


if __name__ == "__main__":
    print("SETUP FAMILY C — CYCLE 2: In-Sample Edge Discovery")
    print("="*70)
    entries = []
    for sym in TEST_SYMBOLS:
        df = get_ohlcv(sym, 250)
        if df is None or len(df) < 80: continue
        df = drop_unclosed(df)
        sh, sl = find_swings(df, 3, 3)
        ar = avg_range(df)
        events = detect_failed_breakouts(df, sh, sl, ar)
        for e in events:
            atr = ar.iloc[e["failure_idx"]]
            if pd.isna(atr) or atr == 0: continue
            entry_price = df["close"].iloc[e["failure_idx"]]
            exc = track_excursion(df, e["direction"], entry_price, atr, e["failure_idx"])
            entries.append({"symbol":sym,"direction":e["direction"],"entry":entry_price,"atr":atr,
                            "idx":e["failure_idx"],"df":df,"exc":exc,"type":e["type"]})
        time.sleep(0.3)

    N = len(entries)
    print(f"\nN={N} | Same-Candle={sum(1 for e in entries if e['type']=='same_candle')} | Multi-Bar={sum(1 for e in entries if e['type']=='multi_bar')}")
    symbols_used = sorted(set(e["symbol"] for e in entries))
    print(f"Symbols with events: {len(symbols_used)}")

    if N > 0:
        report_group("ALL", entries)
        report_group("SAME-CANDLE", [e for e in entries if e["type"]=="same_candle"])
        report_group("MULTI-BAR", [e for e in entries if e["type"]=="multi_bar"])

        by_sym = {}
        for e in entries: by_sym.setdefault(e["symbol"],[]).append(e)
        top = sorted(by_sym.items(), key=lambda x:-len(x[1]))[:8]
        print("\n--- Symbol Concentration (top 8) ---")
        for s, evs in top: print(f"  {s}: {len(evs)}")
    print("="*70)
