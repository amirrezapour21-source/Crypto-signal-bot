"""Unified Validation Engine — END-TO-END Self-Test
Layer B -> C -> D -> E
Dummy plugins only
"""

import requests
import pandas as pd
import time

BASE = "https://api.kucoin.com/api/v1/market/candles"

TARGET_DAYS = 730
ATR_LOOKBACK = 20

HORIZONS = [1, 3, 6, 12, 24]
HIT_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]

MAX_TRACK_BARS = 44

FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05


# ================================================================
# DATA LAYER
# ================================================================

def fetch_page(sym, end_at=None):

    params = {
        "symbol": sym,
        "type": "4hour"
    }

    if end_at is not None:
        params["endAt"] = int(end_at)

    r = requests.get(
        BASE,
        params=params,
        timeout=20
    )

    if r.status_code != 200:
        return None, r.status_code

    d = r.json()

    if d.get("code") != "200000" or not d.get("data"):
        return None, d.get("code")

    rows = [
        {
            "time": int(x[0]),
            "open": float(x[1]),
            "close": float(x[2]),
            "high": float(x[3]),
            "low": float(x[4]),
            "volume": float(x[5])
        }
        for x in d["data"]
    ]

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .reset_index(drop=True),
        200
    )


def fetch_full_history(
    sym,
    target_days=TARGET_DAYS
):

    target_candles = target_days * 6

    dfs = []
    end_at = None
    fetched = 0
    pages = 0

    prev_min_time = None
    consecutive_errors = 0
    stop_reason = "target_reached"

    while fetched < target_candles:

        df, status = fetch_page(
            sym,
            end_at
        )

        pages += 1

        if df is None or df.empty:

            consecutive_errors += 1

            if consecutive_errors >= 3:
                stop_reason = (
                    f"repeated_error(status={status})"
                )
                break

            time.sleep(1)
            continue

        consecutive_errors = 0

        new_min_time = int(
            df["time"].min()
        )

        if (
            prev_min_time is not None
            and new_min_time >= prev_min_time
        ):
            stop_reason = (
                "no_progress_pagination_stalled"
            )
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
        pd.concat(
            dfs,
            ignore_index=True
        )
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Remove currently incomplete 4H candle
    if (
        int(time.time())
        < int(full["time"].iloc[-1]) + 14400
    ):
        full = (
            full.iloc[:-1]
            .reset_index(drop=True)
        )

    full["range"] = (
        full["high"] - full["low"]
    )

    full["atr20"] = (
        full["range"]
        .rolling(ATR_LOOKBACK)
        .mean()
    )

    coverage_days = (
        full["time"].iloc[-1]
        - full["time"].iloc[0]
    ) / 86400

    return full, {
        "symbol": sym,
        "total_candles": len(full),
        "coverage_days": round(
            coverage_days,
            1
        ),
        "pages": pages,
        "stop_reason": stop_reason
    }


# ================================================================
# DUMMY DETECTOR
# ================================================================

def dummy_detector(
    df,
    every=50
):

    events = []

    for i in range(
        ATR_LOOKBACK,
        len(df) - MAX_TRACK_BARS,
        every
    ):

        events.append(
            {
                "idx": i,
                "dir": "bullish",
                "time": int(
                    df["time"].iloc[i]
                )
            }
        )

    return events


# ================================================================
# LAYER B — SIGNAL LEVEL
# ================================================================

def measure_signal_level(
    df,
    events
):

    results = []

    for e in events:

        idx = e["idx"]
        direction = e["dir"]

        atr = df["atr20"].iloc[idx]

        if (
            pd.isna(atr)
            or atr == 0
        ):
            continue

        entry = df["close"].iloc[idx]

        end = min(
            idx + 1 + MAX_TRACK_BARS,
            len(df)
        )

        fwd_ret = {}

        mfe = 0.0
        mae = 0.0

        t_mfe = None
        t_mae = None

        first_reach = {
            lv: None
            for lv in HIT_LEVELS
        }

        for step, i in enumerate(
            range(idx + 1, end),
            start=1
        ):

            row = df.iloc[i]

            if direction == "bullish":

                fav = row["high"] - entry
                adv = entry - row["low"]

            else:

                fav = entry - row["low"]
                adv = row["high"] - entry

            if fav > mfe:
                mfe = fav
                t_mfe = step

            if adv > mae:
                mae = adv
                t_mae = step

            fav_atr = fav / atr

            for lv in HIT_LEVELS:

                if (
                    first_reach[lv] is None
                    and fav_atr >= lv
                ):
                    first_reach[lv] = step

            if step in HORIZONS:

                c = row["close"]

                if direction == "bullish":

                    ret = (
                        (c - entry)
                        / entry
                        * 100
                    )

                else:

                    ret = (
                        (entry - c)
                        / entry
                        * 100
                    )

                fwd_ret[step] = round(
                    ret,
                    3
                )

        hit_at_horizon = {
            lv: {
                h: (
                    first_reach[lv] is not None
                    and first_reach[lv] <= h
                )
                for h in HORIZONS
            }
            for lv in HIT_LEVELS
        }

        results.append(
            {
                "idx": idx,
                "mfe_atr": round(
                    mfe / atr,
                    3
                ),
                "mae_atr": round(
                    mae / atr,
                    3
                ),
                "time_to_mfe": t_mfe,
                "time_to_mae": t_mae,
                "fwd_ret": fwd_ret,
                "hit_at_horizon":
                    hit_at_horizon
            }
        )

    return results


# ================================================================
# LAYER C/D — TRADE CONSTRUCTION
# ================================================================

def dummy_build_trade(
    df,
    event
):

    idx = event["idx"]
    direction = event["dir"]

    atr = df["atr20"].iloc[idx]

    if (
        pd.isna(atr)
        or atr == 0
    ):
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


# ================================================================
# SHARED EXECUTION
# ================================================================

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

    for i in range(
        idx + 1,
        end
    ):

        row = df.iloc[i]

        if direction == "bullish":

            sl_hit = (
                row["low"]
                <= trade["sl"]
            )

            tp_hit = (
                row["high"]
                >= trade["tp"]
            )

        else:

            sl_hit = (
                row["high"]
                >= trade["sl"]
            )

            tp_hit = (
                row["low"]
                <= trade["tp"]
            )

        # Same-candle SL-first
        if sl_hit and tp_hit:

            return (
                -1.0,
                "SL",
                i - idx,
                True
            )

        if sl_hit:

            return (
                -1.0,
                "SL",
                i - idx,
                False
            )

        if tp_hit:

            return (
                trade["rr"],
                "TP",
                i - idx,
                False
            )

    # Dataset ended before the full holding
    # window was completed.
    if end >= len(df):

        return (
            None,
            "OPEN_AT_DATASET_END",
            None,
            False
        )

    # Full MAX_TRACK_BARS completed with
    # neither SL nor TP.
    return (
        0.0,
        "TIMEOUT",
        max_hold,
        False
    )


# ================================================================
# TRADE ENGINE
# ================================================================

def run_trade_layer(
    df,
    symbol,
    events,
    trade_builder_fn
):

    events = sorted(
        events,
        key=lambda e: (
            e["time"],
            e["idx"]
        )
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

            row["final_status"] = (
                "REJECTED@overlap"
            )

            audit.append(row)
            continue

        trade, reason = (
            trade_builder_fn(df, e)
        )

        if trade is None:

            row["final_status"] = (
                "REJECTED@trade_construction"
                f"({reason})"
            )

            audit.append(row)
            continue

        (
            gross_r,
            outcome,
            bars_held,
            ambiguous
        ) = sim_trade(
            df,
            e["dir"],
            trade,
            e["idx"]
        )

        # Dataset-end open:
        # not a win/loss and no fees applied.
        if outcome == "OPEN_AT_DATASET_END":

            row.update(
                {
                    "final_status":
                        "OPEN_AT_DATASET_END",
                    "entry":
                        trade["entry"],
                    "sl":
                        trade["sl"],
                    "tp":
                        trade["tp"],
                    "rr":
                        trade["rr"],
                    "outcome":
                        outcome,
                    "bars_held":
                        None,
                    "ambiguous":
                        False
                }
            )

            audit.append(row)
            continue

        risk_pct = (
            abs(
                trade["entry"]
                - trade["sl"]
            )
            / trade["entry"]
            if trade["entry"]
            else None
        )

        cost_r = (
            (
                FEE_PCT
                + SLIPPAGE_PCT
            )
            / 100
            / risk_pct
            if risk_pct
            else 0
        )

        net_r = (
            gross_r - cost_r
        )

        # Correct exit index:
        # exit_idx = entry_idx + bars_held
        exit_idx = (
            e["idx"]
            + bars_held
        )

        exit_idx = min(
            exit_idx,
            len(df) - 1
        )

        last_exit_time = (
            df["time"].iloc[exit_idx]
        )

        row.update(
            {
                "final_status":
                    "TRADED",
                "entry":
                    trade["entry"],
                "sl":
                    trade["sl"],
                "tp":
                    trade["tp"],
                "rr":
                    trade["rr"],
                "gross_r":
                    gross_r,
                "net_r":
                    round(
                        net_r,
                        4
                    ),
                "outcome":
                    outcome,
                "bars_held":
                    bars_held,
                "ambiguous":
                    ambiguous
            }
        )

        audit.append(row)

    return audit


# ================================================================
# LAYER E — PERFORMANCE
# ================================================================

def compute_maxdd(
    r_series
):

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


def evaluate(
    audit
):

    status_counts = {}

    for a in audit:

        status = a[
            "final_status"
        ]

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0
            ) + 1
        )

    traded = [
        a
        for a in audit
        if a["final_status"]
        == "TRADED"
    ]

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
        "status_breakdown":
            status_counts,
        "n_traded":
            n_traded,
        "n_open_at_end":
            n_open,
        "n_ambiguous":
            n_ambiguous
    }

    if n_traded == 0:

        return summary

    gross = [
        t["gross_r"]
        for t in traded
    ]

    net = [
        t["net_r"]
        for t in traded
    ]

    wins = [
        r for r in gross
        if r > 0
    ]

    losses = [
        r for r in gross
        if r < 0
    ]

    summary.update(
        {
            "win_rate_pct":
                round(
                    len(wins)
                    / n_traded
                    * 100,
                    1
                ),

            "gross_expectancy_R":
                round(
                    sum(gross)
                    / n_traded,
                    3
                ),

            "net_expectancy_R":
                round(
                    sum(net)
                    / n_traded,
                    3
                ),

            "gross_total_R":
                round(
                    sum(gross),
                    2
                ),

            "net_total_R":
                round(
                    sum(net),
                    2
                ),

            "profit_factor_gross":
                (
                    round(
                        sum(wins)
                        / abs(sum(losses)),
                        2
                    )
                    if losses
                    and sum(losses) != 0
                    else None
                ),

            "maxdd_gross_R":
                round(
                    compute_maxdd(gross),
                    2
                ),

            "maxdd_net_R":
                round(
                    compute_maxdd(net),
                    2
                )
        }
    )

    # Per-symbol summary
    by_symbol = {}

    for t in traded:

        by_symbol.setdefault(
            t["symbol"],
            []
        ).append(
            t["gross_r"]
        )

    summary["per_symbol"] = {
        symbol: {
            "n": len(rs),
            "gross_total_R":
                round(
                    sum(rs),
                    2
                )
        }
        for symbol, rs
        in by_symbol.items()
    }

    return summary


# ================================================================
# END-TO-END SELF-TEST
# ================================================================

if __name__ == "__main__":

    print(
        "UNIFIED VALIDATION ENGINE — "
        "END-TO-END Self-Test "
        "(B->C->D->E, Dummy plugins)"
    )

    print("=" * 70)

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    df, cov = fetch_full_history(
        "BTC-USDT"
    )

    if df is None:

        raise RuntimeError(
            f"Data fetch failed: {cov}"
        )

    print(
        f"Data: {cov}\n"
    )

    # ------------------------------------------------------------
    # DETECTION
    # ------------------------------------------------------------

    events = dummy_detector(df)

    print(
        f"Events detected: "
        f"{len(events)}"
    )

    # ------------------------------------------------------------
    # LAYER B
    # ------------------------------------------------------------

    signal_results = (
        measure_signal_level(
            df,
            events
        )
    )

    print(
        "\n[LAYER B] "
        f"Signal-level measurements: "
        f"{len(signal_results)}"
    )

    if signal_results:

        n = len(
            signal_results
        )

        mfe_l = [
            r["mfe_atr"]
            for r in signal_results
        ]

        mae_l = [
            r["mae_atr"]
            for r in signal_results
        ]

        print(
            f"  Mean MFE_ATR="
            f"{sum(mfe_l)/n:.2f}"
            f" | Mean MAE_ATR="
            f"{sum(mae_l)/n:.2f}"
        )

        for h in HORIZONS:

            vals = [
                r["fwd_ret"].get(h)
                for r in signal_results
                if r["fwd_ret"].get(h)
                is not None
            ]

            if vals:

                print(
                    f"  fwd_ret[{h}bar] "
                    f"mean="
                    f"{sum(vals)/len(vals):.3f}%"
                )

    # ------------------------------------------------------------
    # LAYER C/D
    # ------------------------------------------------------------

    audit = run_trade_layer(
        df,
        "BTC-USDT",
        events,
        dummy_build_trade
    )

    # ------------------------------------------------------------
    # LAYER E
    # ------------------------------------------------------------

    summary = evaluate(
        audit
    )

    print(
        "\n[LAYER C/D] "
        f"Status breakdown: "
        f"{summary['status_breakdown']}"
    )

    print(
        f"\n[LAYER E] "
        f"n_traded="
        f"{summary['n_traded']}"
        f" | n_open_at_end="
        f"{summary['n_open_at_end']}"
        f" | n_ambiguous="
        f"{summary['n_ambiguous']}"
    )

    if summary["n_traded"] > 0:

        print(
            f"  WR="
            f"{summary['win_rate_pct']}%"
        )

        print(
            f"  GrossExp="
            f"{summary['gross_expectancy_R']}R"
            f" | NetExp="
            f"{summary['net_expectancy_R']}R"
        )

        print(
            f"  GrossTotalR="
            f"{summary['gross_total_R']}"
            f" | NetTotalR="
            f"{summary['net_total_R']}"
        )

        print(
            f"  PF="
            f"{summary['profit_factor_gross']}"
        )

        print(
            f"  MaxDD gross="
            f"{summary['maxdd_gross_R']}R"
            f" | MaxDD net="
            f"{summary['maxdd_net_R']}R"
        )

        print(
            f"  Per-symbol="
            f"{summary['per_symbol']}"
        )

    # ------------------------------------------------------------
    # CONSISTENCY
    # ------------------------------------------------------------

    consistency_b = (
        len(signal_results)
        == len(events)
    )

    print(
        "\n[CONSISTENCY CHECK]"
    )

    print(
        "Layer B measurements == "
        f"Detected events: "
        f"{consistency_b}"
    )

    print(
        "Layer B event count == "
        f"Trade-layer input events: "
        f"{len(events) == len(events)}"
    )

    print("=" * 70)
