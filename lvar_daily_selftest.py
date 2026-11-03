"""Self-tests for lvar_daily_backtest.py on synthetic data (no network)."""
import numpy as np
import pandas as pd

import lvar_daily_backtest as B

DAY = B.DAY
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def synth(n=700, seed=1, drift=0.0015, vol=0.03, t0=1_600_000_000 // DAY * DAY):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    c = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100.0], c[:-1]])
    spread = np.abs(rng.normal(0, vol, n)) * c
    h = np.maximum(o, c) + spread * 0.5
    l = np.minimum(o, c) - spread * 0.5
    v = rng.uniform(800, 1200, n) * (1 + 0.5 * (np.abs(ret) > vol))
    return pd.DataFrame({"time": t0 + np.arange(n) * DAY, "open": o, "close": c, "high": h, "low": l, "volume": v})


def bars(rows):
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]  # o, h, l, c


def test_sim():
    # long: entry 100 at bar 0, stop 90 (risk 10), target 130
    o, h, l, c = bars([(100, 105, 95, 102), (102, 110, 96, 108), (108, 131, 100, 125)])
    r = B.sim_daily(o, h, l, c, 0, 1, 100.0, 90.0, 130.0)
    check(r == (3.0, 2, "TP"), f"TP long {r}")
    # SL and TP in the same bar -> SL first
    o, h, l, c = bars([(100, 140, 85, 100), (100, 101, 99, 100)])
    r = B.sim_daily(o, h, l, c, 0, 1, 100.0, 90.0, 130.0)
    check(r == (-1.0, 0, "SL"), f"same bar SL first {r}")
    # SL gap at the next open fills at the open (worse than -1R)
    o, h, l, c = bars([(100, 105, 95, 100), (80, 85, 78, 82)])
    r = B.sim_daily(o, h, l, c, 0, 1, 100.0, 90.0, 130.0)
    check(r[2] == "SL" and abs(r[0] - (-2.0)) < 1e-9 and r[1] == 1, f"gap SL {r}")
    # short mirror
    o, h, l, c = bars([(100, 104, 96, 98), (98, 99, 68, 70)])
    r = B.sim_daily(o, h, l, c, 0, -1, 100.0, 110.0, 70.0)
    check(r == (3.0, 1, "TP"), f"TP short {r}")
    # time stop after MAX_HOLD bars at the close of the last bar
    rows = [(100, 104, 96, 100)] * (B.MAX_HOLD + 3)
    o, h, l, c = bars(rows)
    c[B.MAX_HOLD - 1] = 105.0
    r = B.sim_daily(o, h, l, c, 0, 1, 100.0, 90.0, 130.0)
    check(r[2] == "TIME" and r[1] == B.MAX_HOLD - 1 and abs(r[0] - 0.5) < 1e-9, f"time stop {r}")
    # unresolved at the end of data -> None
    rows = [(100, 104, 96, 100)] * 5
    o, h, l, c = bars(rows)
    check(B.sim_daily(o, h, l, c, 0, 1, 100.0, 90.0, 130.0) is None, "unresolved must be dropped")


def test_finish():
    rg, rn, rs = B.finish(2.0, 0.05, 10)
    fund = B.FUNDING_8H * 3 * 10
    check(abs(rn - (2.0 - (B.COST_RT + fund) / 0.05)) < 1e-12, "net cost")
    check(abs(rs - (2.0 - (B.COST_STRESS + fund) / 0.05)) < 1e-12 and rs < rn < rg, "stress cost")


def test_regime():
    up = synth(400, seed=2, drift=0.004, vol=0.01)
    reg = B.btc_regime(up)
    check(reg[up["time"].iloc[-1]] == 1, "uptrend -> +1")
    down = synth(400, seed=3, drift=-0.004, vol=0.01)
    check(B.btc_regime(down)[down["time"].iloc[-1]] == -1, "downtrend -> -1")
    check(reg[up["time"].iloc[5]] == 0, "warm-up is neutral")


def test_no_lookahead():
    df = synth(700, seed=4, drift=0.002)
    reg = B.btc_regime(df)
    full = B.candidates("X-USDT", df, reg, True)
    cut = 450
    part = B.candidates("X-USDT", df.iloc[:cut].reset_index(drop=True), reg, True)
    key = lambda x: (x["entry_t"], x["side"])
    fm = {key(x): x for x in full}
    check(len(part) > 0, "prefix produced candidates")
    for x in part:
        y = fm.get(key(x))
        check(y is not None, "prefix candidate missing in full run")
        check(abs(y["risk_pct"] - x["risk_pct"]) < 1e-12 and abs(y["strength"] - x["strength"]) < 1e-12, "signal fields changed with future data")
    # changing the future must not change earlier signals
    df2 = df.copy()
    df2.loc[cut:, ["open", "high", "low", "close"]] *= 3.0
    part2 = B.candidates("X-USDT", df2.iloc[:cut].reset_index(drop=True), reg, True)
    check([key(x) for x in part] == [key(x) for x in part2], "future data leaked into earlier signals")


def test_entry_is_next_open():
    df = synth(700, seed=5, drift=0.002)
    reg = B.btc_regime(df)
    got = B.candidates("X-USDT", df, reg, True)
    check(len(got) > 0, "candidates exist")
    tset = set(df["time"].to_numpy())
    for x in got[:50]:
        check(x["entry_t"] in tset and x["exit_t"] >= x["entry_t"] and 1 <= x["days"] <= B.MAX_HOLD, "entry/exit sanity")
        check(B.MIN_RISK <= x["risk_pct"] <= B.MAX_RISK, "risk bounds")


def test_portfolio():
    t = 1_700_000_000 // DAY * DAY
    mk = lambda pair, s, ex, st: {"pair": pair, "entry_t": t + s * DAY, "exit_t": t + ex * DAY, "strength": st}
    cands = [mk(f"C{i}-USDT", 0, 5, float(i)) for i in range(12)]
    got = B.portfolio(cands)
    check(len(got) == B.MAX_POS, "cap respected")
    check({x["pair"] for x in got} == {f"C{i}-USDT" for i in range(12 - B.MAX_POS, 12)}, "strongest first")
    cands = [mk("A-USDT", 0, 3, 1.0), mk("A-USDT", 2, 4, 9.0), mk("A-USDT", 4, 6, 1.0)]
    got = B.portfolio(cands)
    check([x["entry_t"] for x in got] == [t, t + 4 * DAY], "same coin cannot overlap; re-entry after the exit day")
    cands = [mk("A-USDT", 0, 3, 1.0), mk("B-USDT", 3, 5, 1.0)]
    check(len(B.portfolio(cands)) == 2, "different coins independent")


def test_timeline_and_report():
    t = 1_700_000_000 // DAY * DAY
    rows = [{"pair": "A", "side": "LONG", "entry_t": t, "exit_t": t + 2 * DAY, "reason": "TP", "days": 3, "risk_pct": 0.05,
             "r_gross": 3.0, "r_net": 2.8, "r_stress": 2.6, "strength": 1.0},
            {"pair": "B", "side": "SHORT", "entry_t": t + DAY, "exit_t": t + DAY, "reason": "SL", "days": 1, "risk_pct": 0.05,
             "r_gross": -1.0, "r_net": -1.2, "r_stress": -1.4, "strength": 1.0}]
    cnt, ed = B.timeline(rows)
    check(list(cnt) == [1, 2, 1] and ed == 2, f"timeline {list(cnt)} {ed}")
    # full synthetic report runs end to end
    all_rows = []
    reg = B.btc_regime(synth(900, seed=10, drift=0.002))
    for s in range(6):
        all_rows += B.candidates(f"S{s}-USDT", synth(900, seed=20 + s, drift=0.002), reg, True)
    cap = B.portfolio(all_rows)
    text = B.report(all_rows, cap, B.portfolio(all_rows), {"start": "x", "end": "y", "coins": 6})
    check("VERDICT" in text and "days open per trade" in text and "bootstrap" in text, "report sections")


if __name__ == "__main__":
    for fn in (test_sim, test_finish, test_regime, test_no_lookahead, test_entry_is_next_open, test_portfolio, test_timeline_and_report):
        fn()
        print("ok", fn.__name__, flush=True)
    print(f"ALL SELF-TESTS PASSED ({PASSED} checks)")
