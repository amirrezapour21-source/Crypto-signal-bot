"""
DAILY DONCHIAN TREND-FOLLOWING ENSEMBLE - backtest (read-only research; nothing is tuned).
Rules are taken from the published design "Catching Crypto Trends" (Zarattini, Pagani, Barbon; Swiss Finance Institute rp2580):
an ensemble of Donchian-channel trend models with different lookbacks, long-only, volatility-based sizing, top liquid coins.
  model L: ON when the daily close breaks above the highest high of the previous L days; OFF when the close falls below the
           midpoint of the previous-L-day channel; otherwise keep the state. Decision at the close of day t, applied to day t+1.
  ensemble signal = mean of the model states (0..1); weights = signal / vol (30-day) normalised over the coins that exist
  that day, total exposure <= 100% (rest in cash); no leverage, no shorts.
Benchmarks over the SAME coins and dates: BTC buy&hold and equal-weight buy&hold (daily rebalanced).
Costs: 0.10% + 0.05% slippage per unit of turnover (stress: 0.30%).
Biases (stated): the universe is TODAY's liquid list (survivorship) - it favours long-only strategies AND the equal-weight
benchmark alike, so the comparison against the equal-weight benchmark is the fair one.
Output files keep the old names so the existing workflow commits them: lvar_backtest_result.txt, lvar_backtest_trades.csv.
"""
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import crypto_signal_bot as M

DAYS = int(os.getenv("BT_DAYS", "1400"))
LOOKBACKS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
VOL_WIN = 30
COST = 0.0015
COST_STRESS = 0.0030
MIN_HISTORY = 40
SKIP = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "USDD", "BUSD", "EURT", "XAUT", "PAXG"}


def fetch_daily(pair: str, start: int, end: int) -> pd.DataFrame:
    rows, cur_end = [], end
    while cur_end > start:
        cur_start = max(start, cur_end - 1400 * 86400)
        payload = M.safe_get_json(f"{M.KUCOIN_BASE}/api/v1/market/candles",
                                  {"symbol": pair, "type": "1day", "startAt": cur_start, "endAt": cur_end})
        if isinstance(payload, dict) and payload.get("code") == "200000" and isinstance(payload.get("data"), list):
            for it in payload["data"]:
                try:
                    rows.append({"time": int(it[0]), "open": float(it[1]), "close": float(it[2]),
                                 "high": float(it[3]), "low": float(it[4]), "volume": float(it[5])})
                except (TypeError, ValueError, IndexError):
                    continue
        cur_end = cur_start
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df = df[(df.close > 0) & (df.high >= df.low) & (df.volume >= 0)]
    df = df[df.time + 86400 <= int(time.time()) - 60]          # completed daily bars only
    df.index = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize()
    return df[["open", "high", "low", "close", "volume"]]


def model_state(high: np.ndarray, low: np.ndarray, close: np.ndarray, L: int) -> np.ndarray:
    """State at the CLOSE of each day (1 = long). Uses only data up to and including that day's close."""
    n = len(close)
    st = np.zeros(n)
    on = 0.0
    for t in range(L, n):
        hh = high[t - L:t].max()          # previous L days only
        ll = low[t - L:t].min()
        if on == 0.0 and close[t] > hh:
            on = 1.0
        elif on == 1.0 and close[t] < (hh + ll) / 2.0:
            on = 0.0
        st[t] = on
    return st


def ensemble_signal(df: pd.DataFrame) -> pd.Series:
    """Mean of the models that already have enough history (a model with L > available history is simply absent)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for L in LOOKBACKS:
        if n <= L + 1:
            continue
        s = model_state(h, l, c, L)
        acc[L:] += s[L:]
        cnt[L:] += 1
    sig = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return pd.Series(sig, index=df.index)


def portfolio(data: dict, cost: float):
    idx = sorted(set().union(*[d.index for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    ret = pd.DataFrame({k: d["close"].pct_change() for k, d in data.items()}).reindex(idx)
    sig = pd.DataFrame({k: ensemble_signal(d) for k, d in data.items()}).reindex(idx)
    vol = ret.rolling(VOL_WIN).std()
    avail = ret.notna() & vol.notna() & (vol > 0)
    invvol = (1.0 / vol).where(avail)
    norm = invvol.sum(axis=1)
    w_target = (sig * invvol).div(norm, axis=0).where(avail).fillna(0.0)       # decided at close t
    w = w_target.shift(1).fillna(0.0)                                          # applied to day t+1 returns
    gross = (w * ret.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - cost * turnover
    ew_w = avail.shift(1, fill_value=False).astype(float)
    ew = (ew_w * ret.fillna(0.0)).sum(axis=1) / ew_w.sum(axis=1).replace(0, np.nan)
    ew = ew.fillna(0.0)
    btc = ret["BTC-USDT"].fillna(0.0) if "BTC-USDT" in ret else ew
    return net, gross, ew, btc, w, turnover, avail


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return {"days": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.0
    dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0
    sh = float(r.mean() / r.std() * np.sqrt(365))
    return {"days": len(r), "total%": round(100 * float(eq.iloc[-1] - 1), 1), "CAGR%": round(100 * cagr, 1),
            "vol%": round(100 * float(r.std() * np.sqrt(365)), 1), "Sharpe": round(sh, 2), "maxDD%": round(100 * dd, 1),
            "Calmar": round(cagr / abs(dd), 2) if dd < 0 else None}


def line(label, m):
    if "total%" not in m:
        return f"{label:36s} n/a"
    return (f"{label:36s} total={m['total%']:+8.1f}%  CAGR={m['CAGR%']:+7.1f}%  vol={m['vol%']:5.1f}%  "
            f"Sharpe={m['Sharpe']:+5.2f}  maxDD={m['maxDD%']:6.1f}%  Calmar={m['Calmar']}")


def report(data, net, gross, ew, btc, w, turnover, stress, avail) -> str:
    start = net.index[np.argmax((avail.sum(axis=1) > 0).to_numpy())]
    sl = lambda s: s[s.index >= start + pd.Timedelta(days=VOL_WIN + 5)]
    net, gross, ew, btc, stress = map(sl, (net, gross, ew, btc, stress))
    mid = net.index[len(net) // 2]
    L = ["DAILY DONCHIAN TREND ENSEMBLE (rules fixed in advance from the published design; nothing tuned)",
         f"period: {net.index[0].date()} -> {net.index[-1].date()} ({len(net)} days), coins: {len(data)}",
         "Biases: universe = TODAY's liquid list (survivorship). Compare against the equal-weight benchmark (same bias).",
         f"Costs: {COST * 100:.2f}% per unit turnover (stress {COST_STRESS * 100:.2f}%). Long-only, no leverage, rest in cash.", ""]
    for label, s in (("STRATEGY net", net), ("STRATEGY gross", gross), ("STRATEGY stress costs", stress),
                     ("BENCH equal-weight B&H", ew), ("BENCH BTC B&H", btc)):
        L.append(line(label, metrics(s)))
    L.append("")
    for tag, f in (("first half", lambda s: s[s.index < mid]), ("second half", lambda s: s[s.index >= mid])):
        L.append(f"--- {tag} ---")
        for label, s in (("STRATEGY net", net), ("BENCH equal-weight", ew), ("BENCH BTC", btc)):
            L.append(line(label, metrics(f(s))))
    L.append("")
    L.append("--- by calendar year (net / equal-weight / BTC) ---")
    for y in sorted(set(net.index.year)):
        a = metrics(net[net.index.year == y]); b = metrics(ew[ew.index.year == y]); c = metrics(btc[btc.index.year == y])
        if "total%" in a and "total%" in b and "total%" in c:
            L.append(f"{y}: strategy {a['total%']:+7.1f}% (maxDD {a['maxDD%']:.1f}%) | equal-weight {b['total%']:+7.1f}% (maxDD {b['maxDD%']:.1f}%) | BTC {c['total%']:+7.1f}% (maxDD {c['maxDD%']:.1f}%)")
    exp = w.sum(axis=1)
    L.append("")
    L.append(f"average exposure {100 * float(exp.mean()):.1f}% of capital; days fully in cash {100 * float((exp < 0.01).mean()):.1f}%; "
             f"turnover/day {100 * float(turnover.mean()):.1f}% of capital")
    top = (w.mean() * 100).sort_values(ascending=False).head(5)
    L.append("largest average weights: " + ", ".join(f"{k.replace('-USDT', '')} {v:.1f}%" for k, v in top.items()))
    return "\n".join(L)


def main():
    end = int(time.time())
    start = end - DAYS * 86400
    coins = [c for c in M.get_top_coins(M.SCAN_TOP_N_COINS) if c["symbol"] not in SKIP]
    print("universe:", len(coins), [c["symbol"] for c in coins])
    data = {}
    for c in coins:
        df = fetch_daily(c["pair"], start, end)
        print(f"{c['pair']}: daily bars {len(df)}", flush=True)
        if len(df) >= MIN_HISTORY + VOL_WIN:
            data[c["pair"]] = df
    if len(data) < 5:
        text = "NOT ENOUGH DATA"
    else:
        net, gross, ew, btc, w, turnover, avail = portfolio(data, COST)
        stress = portfolio(data, COST_STRESS)[0]
        text = report(data, net, gross, ew, btc, w, turnover, stress, avail)
        pd.DataFrame({"net": net, "gross": gross, "equal_weight": ew, "btc": btc, "exposure": w.sum(axis=1)}).to_csv("lvar_backtest_trades.csv")
    print(text)
    with open("lvar_backtest_result.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    if not os.path.exists("lvar_backtest_trades.csv"):
        open("lvar_backtest_trades.csv", "w").write("no data\n")


if __name__ == "__main__":
    main()
