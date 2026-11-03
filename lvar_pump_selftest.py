"""Self-tests for lvar_pump_backtest.py on synthetic data (no network)."""
import numpy as np
import pandas as pd

import lvar_pump_backtest as P

DAY = P.DAY
T0 = 1_600_000_000 // DAY * DAY
PASSED = 0


def check(cond, msg):
    global PASSED
    if not cond:
        raise AssertionError(msg)
    PASSED += 1


def frame(closes, opens=None, vol=200000.0):
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * 1.02
    l = np.minimum(o, c) * 0.98
    return pd.DataFrame({"time": T0 + np.arange(len(c)) * DAY, "open": o, "close": c, "high": h, "low": l,
                         "volume": np.full(len(c), vol)})


def pump_series(after):
    """60 flat bars, 5 pump bars, one red reversal bar (index 65), then the bars in `after`."""
    flat = [1.0] * 60
    pump = [1.2, 1.5, 1.9, 2.3, 2.5]
    closes = flat + pump + [2.0] + list(after)
    opens = flat[:1] + flat[:-1] + [1.0] + pump[:-1] + [2.4] + [2.0] + list(after)[:-1]
    opens = opens[:len(closes)]
    return frame(closes, opens)


def test_signal_and_prices():
    df = pump_series([1.9, 1.7, 1.4, 1.1, 0.8, 0.7, 0.7])
    got = P.candidates("X-USDT", df, "A")
    check(len(got) == 1, f"exactly one signal, got {len(got)}")
    x = got[0]
    check(x["side"] == "SHORT" and x["entry_t"] == T0 + 66 * DAY, "entry is the open of the day after the trigger")
    peak_high = df["high"].iloc[62:66].max()
    entry = df["open"].iloc[66]
    risk = peak_high * 1.01 - entry
    check(abs(x["risk_pct"] - risk / entry) < 1e-9, "stop = 1% above the highest high of the last 4 days")
    check(abs(x["strength"] - (2.5 / 1.0 - 1)) < 1e-9, "run-up measured from the base close to the peak close")
    check(x["reason"] == "TP" and abs(x["r_gross"] - 2.0) < 1e-9, f"2R target reached {x['reason']} {x['r_gross']}")
    check(x["r_stress"] < x["r_net"] < x["r_gross"], "costs ordered")


def test_no_signal_cases():
    check(P.candidates("F-USDT", frame([1.0] * 200), "A") == [], "flat series has no signal")
    # small pump (+50%) is not enough
    small = frame([1.0] * 60 + [1.1, 1.2, 1.3, 1.5, 1.5] + [1.2] + [1.2] * 20, None)
    check(P.candidates("S-USDT", small, "A") == [], "run-up below 80% ignored")
    # illiquid coin is ignored (turnover below the wide threshold)
    check(P.candidates("L-USDT", pump_series([1.9, 1.7]).assign(volume=1.0), "A") == [], "illiquid ignored")
    # no red candle -> no trigger
    closes = [1.0] * 60 + [1.2, 1.5, 1.9, 2.3, 2.5, 2.0, 1.9]
    opens = closes[:1] + closes[:-1]
    opens[65] = 1.9  # day 65 closes above its open (green)
    check(P.candidates("G-USDT", frame(closes, opens), "A") == [], "green candle is not a trigger")


def test_stop_and_time():
    # price squeezes above the stop -> SL, -1R
    df = pump_series([2.2, 3.0, 3.0, 3.0, 3.0])
    got = P.candidates("X-USDT", df, "A")
    check(len(got) == 1 and got[0]["reason"] == "SL" and abs(got[0]["r_gross"] + 1.0) < 1e-9, f"SL {got}")
    # sideways for the whole holding period -> time stop on day 10
    df = pump_series([2.0] * 14)
    got = P.candidates("X-USDT", df, "A")
    check(len(got) == 1 and got[0]["reason"] == "TIME" and got[0]["days"] == P.MAX_HOLD, f"TIME {got}")


def test_sim_short_gap():
    o = np.array([2.0, 3.5, 3.6]); h = np.array([2.1, 3.6, 3.7]); l = np.array([1.9, 3.4, 3.5]); c = np.array([2.0, 3.5, 3.6])
    r = P.sim_short(o, h, l, c, 0, 2.0, 2.5, 1.0)
    check(r[2] == "SL" and r[1] == 1 and abs(r[0] - (2.0 - 3.5) / 0.5) < 1e-9, f"gap SL fills at the open {r}")
    o = np.array([2.0, 1.9]); h = np.array([2.6, 2.0]); l = np.array([1.5, 1.8]); c = np.array([2.0, 1.9])
    r = P.sim_short(o, h, l, c, 0, 2.0, 2.5, 1.0)
    check(r == (-1.0, 0, "SL"), f"SL and TP in one bar -> SL {r}")
    o = np.array([2.0] * 3); h = o + 0.1; l = o - 0.1; c = o.copy()
    check(P.sim_short(o, h, l, c, 0, 2.0, 2.5, 1.0) is None, "unresolved at the end of data is dropped")


def test_variant_b_and_cooldown():
    df = pump_series([1.9, 1.7, 1.4, 1.1, 0.8, 0.7, 0.7])
    got = P.candidates("X-USDT", df, "B")
    check(len(got) == 1, "variant B signal")
    entry = df["open"].iloc[66]
    target = 1.0 + 0.5 * (2.5 - 1.0)
    risk = df["high"].iloc[62:66].max() * 1.01 - entry
    check(abs(got[0]["r_gross"] - (entry - target) / risk) < 1e-9 and got[0]["reason"] == "TP", f"B target is the 50% retracement {got[0]}")
    # two consecutive trigger days produce only one trade (cooldown / busy)
    closes = [1.0] * 60 + [1.2, 1.5, 1.9, 2.3, 2.5, 2.0, 1.9] + [1.9] * 20
    opens = closes[:1] + closes[:-1]
    opens[65] = 2.4
    opens[66] = 2.05
    check(len(P.candidates("D-USDT", frame(closes, opens), "A")) <= 1, "one trade per episode")


def test_no_lookahead():
    rng = np.random.default_rng(7)
    closes = [1.0]
    for i in range(900):
        step = rng.normal(0, 0.05)
        if i % 120 == 100:
            step = 0.7
        if i % 120 in (101, 102):
            step = -0.12
        closes.append(max(0.05, closes[-1] * np.exp(step)))
    df = frame(closes, vol=5_000_000.0)
    full = P.candidates("R-USDT", df, "A")
    check(len(full) > 0, "random pumps produce signals")
    cut = 600
    part = P.candidates("R-USDT", df.iloc[:cut].reset_index(drop=True), "A")
    fm = {x["entry_t"]: x for x in full}
    for x in part:
        y = fm.get(x["entry_t"])
        check(y is not None and abs(y["risk_pct"] - x["risk_pct"]) < 1e-12 and abs(y["strength"] - x["strength"]) < 1e-12, "prefix signal must exist in the full run")
    df2 = df.copy()
    df2.loc[cut:, ["open", "high", "low", "close"]] *= 4.0
    part2 = P.candidates("R-USDT", df2.iloc[:cut].reset_index(drop=True), "A")
    check([x["entry_t"] for x in part] == [x["entry_t"] for x in part2], "future data must not change earlier signals")


def test_fill_gaps():
    df = frame([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    gap = df.drop(index=[2, 3]).reset_index(drop=True)
    out, miss = P.fill_gaps(gap)
    check(miss == 2 and len(out) == 6, "two missing days restored")
    check(out["volume"].iloc[2] == 0 and out["close"].iloc[2] == 1.1 and out["open"].iloc[2] == 1.1, "gap day is flat at the previous close with zero volume")
    out2, miss2 = P.fill_gaps(df)
    check(miss2 == 0 and len(out2) == 6, "no gaps untouched")


def test_perp_bases():
    fake = {"data": [
        {"baseCurrency": "XBT", "quoteCurrency": "USDT", "status": "Open", "isInverse": False},
        {"baseCurrency": "1000PEPE", "quoteCurrency": "USDT", "status": "Open", "isInverse": False},
        {"baseCurrency": "ETH", "quoteCurrency": "USDT", "status": "Open", "isInverse": False},
        {"baseCurrency": "XBT", "quoteCurrency": "USD", "status": "Open", "isInverse": True},
        {"baseCurrency": "DEAD", "quoteCurrency": "USDT", "status": "Paused", "isInverse": False},
        {"baseCurrency": "SOL", "quoteCurrency": "USDT", "isInverse": False}]}
    orig = P.M.safe_get_json
    try:
        P.M.safe_get_json = lambda url, params=None: fake
        got = P.perp_bases()
        check(got == {"BTC", "PEPE", "ETH", "SOL"}, f"futures bases {got}")
        P.M.safe_get_json = lambda url, params=None: None
        check(P.perp_bases() == set(), "failed call -> empty set")
    finally:
        P.M.safe_get_json = orig
    got = P.candidates("PEPE-USDT", pump_series([1.9, 1.7, 1.4, 1.1, 0.8, 0.7, 0.7]), "A", {"PEPE"})
    check(len(got) == 1 and got[0]["perp"] is True, "perp flag set")
    got = P.candidates("PEPE-USDT", pump_series([1.9, 1.7, 1.4, 1.1, 0.8, 0.7, 0.7]), "A", {"BTC"})
    check(got[0]["perp"] is False, "perp flag unset")
    got = P.candidates("PEPE-USDT", pump_series([1.9, 1.7, 1.4, 1.1, 0.8, 0.7, 0.7]), "A")
    check(got[0]["perp"] is False, "no perp list -> False")


def test_report_runs():
    rows = []
    for s in range(30):
        df = pump_series([1.9, 1.7, 1.4, 1.1, 0.8, 0.7, 0.7])
        df["time"] = df["time"] + s * 3 * DAY
        rows += P.candidates(f"C{s}-USDT", df, "A")
    cap = P.B.portfolio(rows)
    text = P.report([("VARIANT A test", rows, cap, True)], {"start": "x", "end": "y", "listed": 30, "coins": 30, "perps": 5})
    check("VERDICT" in text and "by liquidity" in text, "report sections present")


if __name__ == "__main__":
    for fn in (test_signal_and_prices, test_no_signal_cases, test_stop_and_time, test_sim_short_gap, test_variant_b_and_cooldown,
               test_no_lookahead, test_fill_gaps, test_perp_bases, test_report_runs):
        fn()
        print("ok", fn.__name__, flush=True)
    print(f"ALL PUMP SELF-TESTS PASSED ({PASSED} checks)")
