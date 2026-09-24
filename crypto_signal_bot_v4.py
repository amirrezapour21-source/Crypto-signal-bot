"""Unified Validation Engine — Layer C+D+E FINAL SELF-TEST"""

import requests
import pandas as pd
import time

BASE = "https://api.kucoin.com/api/v1/market/candles"

TARGET_DAYS = 730
ATR_LOOKBACK = 20
MAX_TRACK_BARS = 44

FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05


def fetch_page(sym, end_at=None):
    params = {
        "symbol": sym,
        "type": "4hour"
    }

    if end_at:
        params["endAt"] = int(end_at)

    r = requests.get(BASE, params=params, timeout=20)

    if r.status_code != 200:
        return None, r.status_code

    d = r.json()

    if d.get("code") != "200000" or not d.get("data"):
        return None, d.get("code")

    rows = [{
        "time": int(x[0]),
        "open": float(x[1]),
        "close": float(x[2]),
        "high": float(x[3]),
        "low": float(x[4]),
        "volume": float(x[5])
    } for x in d["data"]]

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .reset_index(drop=True),
        200
    )


def fetch_full_history(sym, target_days=TARGET_DAYS):
    target_candles = target_days * 6

    dfs = []
    end_at = None
    fetched = 0
    pages = 0

    prev_min_time = None
    consecutive_errors = 0
    stop_reason = "target_reached"

    while fetched < target_candles:

        df, status = fetch_page(sym, end_at)
        pages += 1

        if df is None or df.empty:
            consecutive_errors += 1

            if consecutive_errors >= 3:
                stop_reason = f"repeated_error(status={status})"
                break

            time.sleep(1)
            continue

        consecutive_errors = 0

        new_min_time = int(df["time"].min())

        if prev_min_time is not None and new_min_time >= prev_min_time:
            stop_reason = "no_progress_pagination_stalled"
            break

        dfs.append(df)
        fetched += len(df)
        prev_min_time = new_min_time

        end_at = new_min_time - 1

        time.sleep(0.1)

        if pages > 100:
            stop_reason = "page_safety_limit"
            break

    else:
        stop_reason = "target_reached"

    if not dfs:
        return None, {
            "stop_reason": "no_data_at_all"
        }

    full = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Remove unclosed current candle if necessary
    if int(time.time()) < int(full["time"].iloc[-1]) + 14400:
        full = full.iloc[:-1].reset_index(drop=True)

    full["range"] = full["high"] - full["low"]
    full["atr20"] = full["range"].rolling(ATR_LOOKBACK).mean()

    coverage_days = (
        full["time"].iloc[-1] - full["time"].iloc[0]
    ) / 86400

    return full, {
        "symbol": sym,
        "total_candles": len(full),
        "coverage_days": round(coverage_days, 1),
        "pages": pages,
        "stop_reason": stop_reason
    }


# ------------------------------------------------------------------
# DUMMY DETECTOR
# ------------------------------------------------------------------

def dummy_detector(df, every=50):
    return [
        {
            "idx": i,
            "dir": "bullish",
            "time": int(df["time"].iloc[i])
        }
        for i in range(
            ATR_LOOKBACK,
            len(df) - MAX_TRACK_BARS,
            every
        )
    ]


# ------------------------------------------------------------------
# DUMMY TRADE BUILDER
# ------------------------------------------------------------------

def dummy_build_trade(df, event):

    idx = event["idx"]
    direction = event["dir"]

    atr = df["atr20"].iloc[idx]

    if pd.isna(atr) or atr == 0:
        return None, "invalid_atr"

    entry = df["close"].iloc[idx]

    if direction == "bullish":
        sl = entry - 1.0 * atr
        tp = entry + 2.0 * atr
    else:
        sl = entry + 1.0 * atr
        tp = entry - 2.0 * atr

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": 2.0
    }, "ok"


# ------------------------------------------------------------------
# TRADE SIMULATION
# ------------------------------------------------------------------

def sim_trade(
    df,
    direction,
    trade,
    idx,
    max_hold=MAX_TRACK_BARS
):

    end = min(
        idx + 1 + max_hold,
        len(df)
    )

    for i in range(idx + 1, end):

        row = df.iloc[i]

        sl_hit = (
            row["low"] <= trade["sl"]
            if direction == "bullish"
            else row["high"] >= trade["sl"]
        )

        tp_hit = (
            row["high"] >= trade["tp"]
            if direction == "bullish"
            else row["low"] <= trade["tp"]
        )

        # Same-candle ambiguity:
        # SL takes priority according to frozen specification.
        if sl_hit and tp_hit:
            return -1.0, "SL", i - idx, True

        if sl_hit:
            return -1.0, "SL", i - idx, False

        if tp_hit:
            return trade["rr"], "TP", i - idx, False

    # IMPORTANT:
    # If the dataset ended before MAX_TRACK_BARS was completed,
    # this is genuinely open at dataset end.
    #
    # If MAX_TRACK_BARS was fully completed and no TP/SL occurred,
    # classify it as TIMEOUT instead.
    if end >= len(df):
        return None, "OPEN_AT_DATASET_END", None, False

    return 0.0, "TIMEOUT", max_hold, False


# ------------------------------------------------------------------
# UNIFIED ENGINE
# ------------------------------------------------------------------

def run_engine(
    df,
    symbol,
    detector_fn,
    trade_builder_fn
):

    events = detector_fn(df)

    # Deterministic chronological order
    events = sorted(
        events,
        key=lambda x: (x["time"], x["idx"])
    )

    audit = []
    last_exit_time = None

    for e in events:

        row = {
            "symbol": symbol,
            "idx": e["idx"],
            "time": pd.to_datetime(
                e["time"],
                unit="s"
            ),
            "dir": e["dir"]
        }

        # Sequential overlap lock
        if (
            last_exit_time is not None
            and e["time"] < last_exit_time
        ):
            row["final_status"] = "REJECTED@overlap"
            audit.append(row)
            continue

        trade, reason = trade_builder_fn(df, e)

        if trade is None:
            row["final_status"] = (
                f"REJECTED@trade_construction({reason})"
            )
            audit.append(row)
            continue

        gross_r, outcome, bars_held, ambiguous = sim_trade(
            df,
            e["dir"],
            trade,
            e["idx"]
        )

        # Open at dataset end is NOT a win/loss
        if outcome == "OPEN_AT_DATASET_END":

            row.update({
                "final_status": "OPEN_AT_DATASET_END",
                "entry": trade["entry"],
                "sl": trade["sl"],
                "tp": trade["tp"],
                "rr": trade["rr"],
                "outcome": outcome,
                "bars_held": None,
                "ambiguous": False
            })

            audit.append(row)
            continue

        # TIMEOUT is a closed evaluation state with 0R.
        # It is not a win.
        if outcome == "TIMEOUT":
            gross_r = 0.0

        risk_pct = (
            abs(trade["entry"] - trade["sl"])
            / trade["entry"]
            if trade["entry"]
            else None
        )

        cost_r = (
            (FEE_PCT + SLIPPAGE_PCT) / 100 / risk_pct
            if risk_pct
            else 0
        )

        net_r = gross_r - cost_r

        # Correct exit index:
        # bars_held = exit_idx - entry_idx
        exit_idx = (
            e["idx"] + bars_held
            if bars_held is not None
            else None
        )

        if exit_idx is not None:
            exit_idx = min(
                exit_idx,
                len(df) - 1
            )

            last_exit_time = df["time"].iloc[exit_idx]

        row.update({
            "final_status": "TRADED",
            "entry": trade["entry"],
            "sl": trade["sl"],
            "tp": trade["tp"],
            "rr": trade["rr"],
            "gross_r": gross_r,
            "net_r": round(net_r, 4),
            "outcome": outcome,
            "bars_held": bars_held,
            "ambiguous": ambiguous
        })

        audit.append(row)

    return audit


# ------------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------------

def compute_max_drawdown(r_series):

    cumulative = 0.0
    peak = 0.0
    maxdd = 0.0

    for r in r_series:

        cumulative += r
        peak = max(
            peak,
            cumulative
        )

        maxdd = min(
            maxdd,
            cumulative - peak
        )

    return maxdd


def evaluate(audit):

    status_counts = {}

    for a in audit:
        status = a["final_status"]

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    # Closed trades only
    traded = [
        a for a in audit
        if a["final_status"] == "TRADED"
    ]

    # Deterministic chronological sequence
    traded = sorted(
        traded,
        key=lambda x: (
            x["time"],
            x["idx"]
        )
    )

    n_traded = len(traded)

    n_open = status_counts.get(
        "OPEN_AT_DATASET_END",
        0
    )

    n_ambiguous = sum(
        1
        for t in traded
        if t["ambiguous"]
    )

    summary = {
        "n_traded": n_traded,
        "n_open_at_end": n_open,
        "n_ambiguous": n_ambiguous,
        "status_breakdown": status_counts
    }

    if n_traded == 0:
        return summary, []

    gross = [
        t["gross_r"]
        for t in traded
    ]

    net = [
        t["net_r"]
        for t in traded
    ]

    wins_gross = [
        g for g in gross
        if g > 0
    ]

    losses_gross = [
        g for g in gross
        if g < 0
    ]

    summary["win_rate_pct"] = round(
        len(wins_gross)
        / n_traded
        * 100,
        1
    )

    summary["gross_expectancy_R"] = round(
        sum(gross) / n_traded,
        3
    )

    summary["net_expectancy_R"] = round(
        sum(net) / n_traded,
        3
    )

    summary["gross_total_R"] = round(
        sum(gross),
        2
    )

    summary["net_total_R"] = round(
        sum(net),
        2
    )

    summary["profit_factor_gross"] = (
        round(
            sum(wins_gross)
            / abs(sum(losses_gross)),
            2
        )
        if losses_gross
        and sum(losses_gross) != 0
        else None
    )

    summary["max_drawdown_gross_R"] = round(
        compute_max_drawdown(gross),
        2
    )

    summary["max_drawdown_net_R"] = round(
        compute_max_drawdown(net),
        2
    )

    # Per-symbol summary
    by_symbol = {}

    for t in traded:
        by_symbol.setdefault(
            t["symbol"],
            []
        ).append(t["gross_r"])

    per_symbol = {
        symbol: {
            "n": len(rs),
            "gross_total_R": round(
                sum(rs),
                2
            )
        }
        for symbol, rs in by_symbol.items()
    }

    summary["per_symbol"] = per_symbol

    return summary, traded


# ------------------------------------------------------------------
# SELF-TEST
# ------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "UNIFIED VALIDATION ENGINE — "
        "Layer C/D/E FINAL Self-Test "
        "(Dummy plugins only)"
    )

    print("=" * 70)

    df, cov = fetch_full_history(
        "BTC-USDT"
    )

    if df is None:
        raise RuntimeError(
            f"Data fetch failed: {cov}"
        )

    print(f"Data: {cov}\n")

    audit = run_engine(
        df,
        "BTC-USDT",
        dummy_detector,
        dummy_build_trade
    )

    summary, traded = evaluate(audit)

    print("STATUS BREAKDOWN:")

    for k, v in summary[
        "status_breakdown"
    ].items():
        print(f"  {k}: {v}")

    print(
        f"\nn_traded={summary['n_traded']}"
        f" | n_open_at_end={summary['n_open_at_end']}"
        f" | n_ambiguous={summary['n_ambiguous']}"
    )

    if summary["n_traded"] > 0:

        print(
            f"WR={summary['win_rate_pct']}%"
        )

        print(
            f"Gross Exp="
            f"{summary['gross_expectancy_R']}R"
            f" | Net Exp="
            f"{summary['net_expectancy_R']}R"
        )

        print(
            f"Gross TotalR="
            f"{summary['gross_total_R']}"
            f" | Net TotalR="
            f"{summary['net_total_R']}"
        )

        print(
            f"Profit Factor (gross)="
            f"{summary['profit_factor_gross']}"
        )

        print(
            f"MaxDD gross="
            f"{summary['max_drawdown_gross_R']}R"
            f" | MaxDD net="
            f"{summary['max_drawdown_net_R']}R"
        )

        print(
            f"Per-symbol: "
            f"{summary['per_symbol']}"
        )
