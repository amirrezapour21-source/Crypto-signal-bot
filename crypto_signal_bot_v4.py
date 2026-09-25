"""
Unified Validation Engine — Candidate 2
SVA Zone / Value-Area Mean-Reversion

STATUS:
- Candidate 2 parameters FROZEN
- Full ~730-day causal history
- Layer B signal-level metrics
- Layer C/D trade construction + shared execution semantics
- Layer E full evaluation
- TIMEOUT != OPEN_AT_DATASET_END
- No parameter optimization
- No OOS/WF in this run
"""

import requests
import pandas as pd
import numpy as np
import time
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "https://api.kucoin.com/api/v1/market/candles"

SYMS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","BNB-USDT","XRP-USDT",
    "DOGE-USDT","ADA-USDT","LINK-USDT","AVAX-USDT","DOT-USDT",
    "NEAR-USDT","APT-USDT","ARB-USDT","OP-USDT","SUI-USDT",
    "INJ-USDT","TIA-USDT","SEI-USDT","FIL-USDT","ATOM-USDT",
    "LTC-USDT","ETC-USDT","TRX-USDT","ICP-USDT","AAVE-USDT",
    "UNI-USDT","MKR-USDT","RUNE-USDT","FTM-USDT","GRT-USDT",
    "ALGO-USDT","VET-USDT","HBAR-USDT","EGLD-USDT","XLM-USDT",
    "THETA-USDT","SAND-USDT","MANA-USDT","AXS-USDT","CHZ-USDT",
    "COMP-USDT","SNX-USDT","CRV-USDT","LDO-USDT","DYDX-USDT",
    "GMX-USDT","STX-USDT","KAVA-USDT","ZIL-USDT","ONE-USDT",
    "1INCH-USDT","YFI-USDT","BAL-USDT","ENJ-USDT","BAT-USDT",
    "ZRX-USDT","OMG-USDT","IOTA-USDT","QTUM-USDT","WAVES-USDT",
    "ANKR-USDT","CELR-USDT","COTI-USDT","SKL-USDT","STORJ-USDT",
    "OCEAN-USDT","RSR-USDT","CKB-USDT","IOTX-USDT","KSM-USDT"
]

TARGET_DAYS = 730
MIN_HISTORY_DAYS = 180

ATR_LOOKBACK = 20
MAX_TRACK_BARS = 44

FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05

# ============================================================
# FROZEN CANDIDATE 2 PARAMETERS
# ============================================================

VP_WINDOW = 40
VA_PCT = 0.70
HOLD = 30
MIN_RR = 2.0
SL_BUFFER_PCT = 0.001
MAX_EXCURSION_LOOKBACK = 10

HORIZONS = [1, 3, 6, 12, 24]
HIT_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]


# ============================================================
# DATA LAYER
# ============================================================

def fetch_page(sym, end_at=None):
    params = {
        "symbol": sym,
        "type": "4hour"
    }

    if end_at is not None:
        params["endAt"] = int(end_at)

    try:
        r = requests.get(BASE, params=params, timeout=20)
    except Exception as exc:
        return None, f"request_error:{exc}"

    if r.status_code != 200:
        return None, r.status_code

    try:
        d = r.json()
    except Exception:
        return None, "invalid_json"

    if d.get("code") != "200000" or not d.get("data"):
        return None, d.get("code", "no_data")

    rows = []

    for x in d["data"]:
        rows.append({
            "time": int(x[0]),
            "open": float(x[1]),
            "close": float(x[2]),
            "high": float(x[3]),
            "low": float(x[4]),
            "volume": float(x[5])
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return None, "empty_page"

    return (
        df.sort_values("time")
          .drop_duplicates("time")
          .reset_index(drop=True),
        200
    )


def audit_continuity(df):
    times = df["time"].astype(np.int64)

    diffs = times.diff().dropna()

    expected = 4 * 3600

    gaps = diffs[diffs != expected]

    return {
        "monotonic": bool(times.is_monotonic_increasing),
        "no_dupes": bool(times.duplicated().sum() == 0),
        "gap_count": int(len(gaps)),
        "missing_candle_count": int(
            sum(max(0, int(d / expected) - 1) for d in gaps)
        )
    }


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

        if (
            prev_min_time is not None
            and new_min_time >= prev_min_time
        ):
            stop_reason = "no_progress_pagination_stalled"
            break

        dfs.append(df)

        fetched += len(df)
        prev_min_time = new_min_time
        end_at = new_min_time - 1

        time.sleep(0.08)

        if pages > 100:
            stop_reason = "page_safety_limit"
            break

    if not dfs:

        return None, {
            "symbol": sym,
            "total_candles": 0,
            "coverage_days": 0,
            "pages": pages,
            "stop_reason": "no_data_at_all",
            "sufficient_history": False
        }

    full = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Remove currently open 4H candle if necessary.
    now = int(time.time())

    if len(full) > 0:
        last_time = int(full["time"].iloc[-1])

        if now < last_time + 14400:
            full = full.iloc[:-1].reset_index(drop=True)

    if full.empty:
        return None, {
            "symbol": sym,
            "total_candles": 0,
            "coverage_days": 0,
            "pages": pages,
            "stop_reason": "no_closed_candles",
            "sufficient_history": False
        }

    full["range"] = full["high"] - full["low"]
    full["atr20"] = full["range"].rolling(
        ATR_LOOKBACK,
        min_periods=ATR_LOOKBACK
    ).mean()

    cov_days = round(
        (
            full["time"].iloc[-1]
            - full["time"].iloc[0]
        ) / 86400,
        1
    )

    continuity = audit_continuity(full)

    sufficient = cov_days >= MIN_HISTORY_DAYS

    coverage = {
        "symbol": sym,
        "total_candles": len(full),
        "coverage_days": cov_days,
        "pages": pages,
        "stop_reason": stop_reason,
        "sufficient_history": sufficient,
        **continuity
    }

    return full, coverage


# ============================================================
# VALUE AREA
# ============================================================

def compute_value_area(window_df, bins=24):

    tp = (
        window_df["high"]
        + window_df["low"]
        + window_df["close"]
    ) / 3.0

    lo = float(window_df["low"].min())
    hi = float(window_df["high"].max())

    if hi <= lo:
        return None

    edges = np.linspace(lo, hi, bins + 1)

    idx = np.clip(
        np.digitize(tp, edges) - 1,
        0,
        bins - 1
    )

    vol = np.zeros(bins)

    for j, v in zip(idx, window_df["volume"].values):
        vol[j] += v

    total_vol = float(vol.sum())

    if total_vol <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) / 2

    poc_i = int(np.argmax(vol))

    lo_i = poc_i
    hi_i = poc_i

    cum = float(vol[poc_i])

    while (
        cum / total_vol < VA_PCT
        and (lo_i > 0 or hi_i < bins - 1)
    ):

        left = vol[lo_i - 1] if lo_i > 0 else -1
        right = vol[hi_i + 1] if hi_i < bins - 1 else -1

        if right >= left:
            hi_i += 1
            cum += vol[hi_i]
        else:
            lo_i -= 1
            cum += vol[lo_i]

    return {
        "poc": float(centers[poc_i]),
        "vah": float(centers[hi_i]),
        "val": float(centers[lo_i])
    }


# ============================================================
# CANDIDATE 2 — CAUSAL DETECTOR
# ============================================================

def candidate2_detector(df):

    events = []

    # Causal rule:
    # VA is computed ONLY from candles before the current candle.
    # Previous close is outside VA.
    # Current close re-enters VA.

    for i in range(VP_WINDOW + 1, len(df)):

        win = df.iloc[i - VP_WINDOW:i]

        va = compute_value_area(win)

        if va is None:
            continue

        prev_close = float(df["close"].iloc[i - 1])
        cur_close = float(df["close"].iloc[i])

        if (
            prev_close > va["vah"]
            and cur_close <= va["vah"]
        ):
            direction = "bearish"

        elif (
            prev_close < va["val"]
            and cur_close >= va["val"]
        ):
            direction = "bullish"

        else:
            continue

        events.append({
            "idx": i,
            "time": int(df["time"].iloc[i]),
            "dir": direction,
            "va": va
        })

    return events


# ============================================================
# CANDIDATE 2 — TRADE CONSTRUCTION
# ============================================================

def candidate2_build_trade(df, event):

    idx = event["idx"]
    direction = event["dir"]
    va = event["va"]

    entry = float(df["close"].iloc[idx])

    # Prior candles ONLY.
    lookback_start = max(
        0,
        idx - MAX_EXCURSION_LOOKBACK
    )

    if direction == "bullish":

        prior_lows = df["low"].iloc[lookback_start:idx]

        if len(prior_lows) == 0:
            return None, "insufficient_prior_extreme_history"

        extreme = float(prior_lows.min())

        sl = extreme * (1.0 - SL_BUFFER_PCT)

        risk = entry - sl

        if risk <= 0:
            return None, "invalid_risk_nonpositive"

        tp = float(va["poc"])

        reward = tp - entry

    else:

        prior_highs = df["high"].iloc[lookback_start:idx]

        if len(prior_highs) == 0:
            return None, "insufficient_prior_extreme_history"

        extreme = float(prior_highs.max())

        sl = extreme * (1.0 + SL_BUFFER_PCT)

        risk = sl - entry

        if risk <= 0:
            return None, "invalid_risk_nonpositive"

        tp = float(va["poc"])

        reward = entry - tp

    if reward <= 0:
        return None, "invalid_reward_nonpositive"

    rr = reward / risk

    if rr < MIN_RR:
        return None, f"rr_below_min({rr:.4f})"

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr
    }, "ok"


# ============================================================
# LAYER B — SIGNAL LEVEL
# ============================================================

def measure_signal_level(df, events):

    results = []

    for event in events:

        idx = event["idx"]
        direction = event["dir"]

        atr = df["atr20"].iloc[idx]

        if pd.isna(atr) or atr <= 0:
            results.append({
                "idx": idx,
                "time": event["time"],
                "dir": direction,
                "mfe_atr": None,
                "mae_atr": None,
                "fwd_ret": {},
                "hit": {
                    lv: False for lv in HIT_LEVELS
                },
                "measurement_status": "ATR_UNAVAILABLE"
            })
            continue

        entry = float(df["close"].iloc[idx])

        # Future candles only.
        future_end = min(
            idx + 1 + MAX_TRACK_BARS,
            len(df)
        )

        mfe = 0.0
        mae = 0.0

        first_reach = {
            lv: None for lv in HIT_LEVELS
        }

        fwd_ret = {}

        for step, i in enumerate(
            range(idx + 1, future_end),
            start=1
        ):

            row = df.iloc[i]

            if direction == "bullish":

                fav = float(row["high"]) - entry
                adv = entry - float(row["low"])

            else:

                fav = entry - float(row["low"])
                adv = float(row["high"]) - entry

            mfe = max(mfe, fav)
            mae = max(mae, adv)

            fav_atr = fav / atr

            for level in HIT_LEVELS:

                if (
                    first_reach[level] is None
                    and fav_atr >= level
                ):
                    first_reach[level] = step

            if step in HORIZONS:

                close = float(row["close"])

                if direction == "bullish":
                    ret = (close - entry) / entry * 100
                else:
                    ret = (entry - close) / entry * 100

                fwd_ret[step] = ret

        results.append({
            "idx": idx,
            "time": event["time"],
            "dir": direction,
            "mfe_atr": mfe / atr,
            "mae_atr": mae / atr,
            "fwd_ret": fwd_ret,
            "hit": {
                level: (
                    first_reach[level] is not None
                )
                for level in HIT_LEVELS
            },
            "measurement_status": "OK"
        })

    return results


# ============================================================
# LAYER C/D — SHARED EXECUTION SEMANTICS
# ============================================================

def sim_trade(
    df,
    direction,
    trade,
    idx,
    max_hold=HOLD
):
    """
    IMPORTANT:

    TIMEOUT:
        Full HOLD candles are available.
        Neither SL nor TP was hit.

    OPEN_AT_DATASET_END:
        Dataset ends before full HOLD period can be completed.
    """

    dataset_end = len(df) - 1

    # First future candle = idx + 1.
    available_future_bars = dataset_end - idx

    # Not enough data to complete the frozen HOLD period.
    if available_future_bars < max_hold:

        for i in range(idx + 1, dataset_end + 1):

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

            # SL-first on same candle.
            if sl_hit and tp_hit:
                return -1.0, "SL", i - idx, True

            if sl_hit:
                return -1.0, "SL", i - idx, False

            if tp_hit:
                return trade["rr"], "TP", i - idx, False

        return (
            None,
            "OPEN_AT_DATASET_END",
            available_future_bars,
            False
        )

    # Full HOLD period is available.
    end = idx + max_hold

    for i in range(idx + 1, end + 1):

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

        # Same-candle SL-first.
        if sl_hit and tp_hit:
            return -1.0, "SL", i - idx, True

        if sl_hit:
            return -1.0, "SL", i - idx, False

        if tp_hit:
            return trade["rr"], "TP", i - idx, False

    # Full HOLD completed without SL/TP.
    return (
        0.0,
        "TIMEOUT",
        max_hold,
        False
    )


# ============================================================
# TRADE LAYER
# ============================================================

def run_trade_layer(
    df,
    symbol,
    events,
    build_trade_fn
):

    audit = []

    # Per-symbol chronological lock.
    last_exit_idx = None

    for event in sorted(
        events,
        key=lambda x: x["idx"]
    ):

        idx = event["idx"]

        row = {
            "symbol": symbol,
            "idx": idx,
            "time": event["time"],
            "dir": event["dir"],
            "stage": "DETECTED"
        }

        # Overlap is evaluated before construction.
        if (
            last_exit_idx is not None
            and idx <= last_exit_idx
        ):

            row["final_status"] = "REJECTED@overlap"
            row["reject_reason"] = "active_trade_overlap"

            audit.append(row)
            continue

        trade, reason = build_trade_fn(
            df,
            event
        )

        if trade is None:

            row["final_status"] = (
                "REJECTED@trade_construction"
            )

            row["reject_reason"] = reason

            audit.append(row)
            continue

        gross_r, outcome, bars_held, ambiguous = sim_trade(
            df,
            event["dir"],
            trade,
            idx,
            max_hold=HOLD
        )

        row.update({
            "entry": trade["entry"],
            "sl": trade["sl"],
            "tp": trade["tp"],
            "rr": trade["rr"]
        })

        # Dataset-end open position:
        # not a win, not a loss, no R.
        if outcome == "OPEN_AT_DATASET_END":

            row["final_status"] = "OPEN_AT_DATASET_END"
            row["bars_held"] = bars_held
            row["ambiguous"] = ambiguous

            audit.append(row)

            # Do not create a future overlap lock beyond dataset.
            last_exit_idx = dataset_end_idx = len(df) - 1

            continue

        # TIMEOUT is a CLOSED trade with 0R.
        risk_pct = (
            abs(trade["entry"] - trade["sl"])
            / trade["entry"]
        )

        cost_r = (
            (FEE_PCT + SLIPPAGE_PCT) / 100.0 / risk_pct
            if risk_pct > 0
            else 0.0
        )

        net_r = gross_r - cost_r

        exit_idx = idx + bars_held

        row.update({
            "final_status": "TRADED",
            "outcome": outcome,
            "bars_held": bars_held,
            "gross_r": float(gross_r),
            "cost_r": float(cost_r),
            "net_r": float(net_r),
            "ambiguous": bool(ambiguous)
        })

        audit.append(row)

        # TIMEOUT, TP and SL are all closed positions.
        last_exit_idx = exit_idx

    return audit


# ============================================================
# LAYER E — EVALUATION
# ============================================================

def compute_maxdd(rs):

    cumulative = 0.0
    peak = 0.0
    maxdd = 0.0

    for r in rs:

        cumulative += r
        peak = max(peak, cumulative)
        maxdd = min(
            maxdd,
            cumulative - peak
        )

    return maxdd


def profit_factor(rs):

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]

    if not losses:
        return None

    loss_abs = abs(sum(losses))

    if loss_abs == 0:
        return None

    return sum(wins) / loss_abs


def evaluate(all_audit):

    status_counts = Counter(
        a["final_status"]
        for a in all_audit
    )

    traded = sorted(
        [
            a for a in all_audit
            if a["final_status"] == "TRADED"
        ],
        key=lambda x: (
            x["symbol"],
            x["time"]
        )
    )

    n = len(traded)

    out = {
        "status_breakdown": dict(status_counts),
        "n_traded": n,
        "n_timeout": sum(
            1 for t in traded
            if t.get("outcome") == "TIMEOUT"
        ),
        "n_tp": sum(
            1 for t in traded
            if t.get("outcome") == "TP"
        ),
        "n_sl": sum(
            1 for t in traded
            if t.get("outcome") == "SL"
        ),
        "n_open_at_dataset_end": status_counts.get(
            "OPEN_AT_DATASET_END",
            0
        ),
        "n_ambiguous": sum(
            1 for t in traded
            if t.get("ambiguous")
        )
    }

    if n == 0:
        return out, traded

    gross = [
        float(t["gross_r"])
        for t in traded
    ]

    net = [
        float(t["net_r"])
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

    net_wins = [
        r for r in net
        if r > 0
    ]

    net_losses = [
        r for r in net
        if r < 0
    ]

    out.update({

        "win_rate_pct": round(
            len(wins) / n * 100,
            1
        ),

        "gross_exp": round(
            sum(gross) / n,
            4
        ),

        "net_exp": round(
            sum(net) / n,
            4
        ),

        "gross_total_R": round(
            sum(gross),
            2
        ),

        "net_total_R": round(
            sum(net),
            2
        ),

        "pf_gross": (
            round(profit_factor(gross), 3)
            if losses else None
        ),

        "pf_net": (
            round(profit_factor(net), 3)
            if net_losses else None
        ),

        "maxdd_gross": round(
            compute_maxdd(gross),
            2
        ),

        "maxdd_net": round(
            compute_maxdd(net),
            2
        )
    })

    # Per-symbol results.
    by_symbol = {}

    for t in traded:

        s = t["symbol"]

        if s not in by_symbol:
            by_symbol[s] = []

        by_symbol[s].append(t)

    per_symbol = {}

    for symbol, trades in by_symbol.items():

        rs_gross = [
            t["gross_r"]
            for t in trades
        ]

        rs_net = [
            t["net_r"]
            for t in trades
        ]

        per_symbol[symbol] = {
            "n": len(trades),
            "gross_total_R": round(
                sum(rs_gross),
                2
            ),
            "net_total_R": round(
                sum(rs_net),
                2
            ),
            "gross_exp": round(
                sum(rs_gross) / len(rs_gross),
                4
            ),
            "net_exp": round(
                sum(rs_net) / len(rs_net),
                4
            )
        }

    out["per_symbol"] = per_symbol

    return out, traded


# ============================================================
# SIGNAL-LEVEL AGGREGATION
# ============================================================

def print_signal_metrics(all_signal):

    print(
        f"\n[LAYER B] "
        f"Signal-level measurements={len(all_signal)}"
    )

    if not all_signal:
        return

    valid_mfe = [
        x["mfe_atr"]
        for x in all_signal
        if x["mfe_atr"] is not None
    ]

    valid_mae = [
        x["mae_atr"]
        for x in all_signal
        if x["mae_atr"] is not None
    ]

    if valid_mfe:
        print(
            f"  Mean MFE_ATR="
            f"{np.mean(valid_mfe):.3f}"
        )

    if valid_mae:
        print(
            f"  Mean MAE_ATR="
            f"{np.mean(valid_mae):.3f}"
        )

    print("\n  Forward Returns:")

    for h in HORIZONS:

        vals = [
            x["fwd_ret"][h]
            for x in all_signal
            if h in x["fwd_ret"]
        ]

        if vals:
            print(
                f"    {h:>2} bar: "
                f"N={len(vals)} | "
                f"mean={np.mean(vals):.4f}%"
            )
        else:
            print(
                f"    {h:>2} bar: N=0"
            )

    print("\n  Hit Rates:")

    for level in HIT_LEVELS:

        hits = sum(
            1
            for x in all_signal
            if x["hit"][level]
        )

        print(
            f"    +{level} ATR: "
            f"{hits}/{len(all_signal)} "
            f"({hits / len(all_signal) * 100:.1f}%)"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "UNIFIED VALIDATION ENGINE — "
        "CANDIDATE 2 SVA ZONE — IS RUN"
    )

    print("=" * 75)

    print("\nFROZEN PARAMETERS:")
    print(f"  VP_WINDOW={VP_WINDOW}")
    print(f"  VA_PCT={VA_PCT}")
    print(f"  HOLD={HOLD}")
    print(f"  MIN_RR={MIN_RR}")
    print(f"  SL_BUFFER_PCT={SL_BUFFER_PCT}")
    print(
        f"  MAX_EXCURSION_LOOKBACK="
        f"{MAX_EXCURSION_LOOKBACK}"
    )

    coverages = []
    all_events = []
    all_signal = []
    all_audit = []

    for number, sym in enumerate(SYMS, start=1):

        print(
            f"\n[{number}/{len(SYMS)}] "
            f"Fetching {sym}..."
        )

        df, cov = fetch_full_history(sym)

        coverages.append(cov)

        if df is None:
            print(
                f"  DATA ERROR: "
                f"{cov.get('stop_reason')}"
            )
            continue

        print(
            f"  candles={cov['total_candles']} "
            f"| coverage={cov['coverage_days']}d "
            f"| pages={cov['pages']} "
            f"| stop={cov['stop_reason']} "
            f"| sufficient={cov['sufficient_history']} "
            f"| gaps={cov['gap_count']}"
        )

        # Do not silently use severely insufficient history.
        if (
            cov["coverage_days"] < MIN_HISTORY_DAYS
            or cov["total_candles"] < VP_WINDOW + 50
        ):
            print(
                "  -> INSUFFICIENT_HISTORY / "
                "NOT USED"
            )
            continue

        # -------------------------
        # Candidate 2 Detection
        # -------------------------

        events = candidate2_detector(df)

        for e in events:

            all_events.append({
                "symbol": sym,
                **e
            })

        print(
            f"  causal events={len(events)}"
        )

        # -------------------------
        # Layer B
        # -------------------------

        signal = measure_signal_level(
            df,
            events
        )

        for s in signal:
            s["symbol"] = sym

        all_signal.extend(signal)

        # -------------------------
        # Layer C/D/E
        # -------------------------

        audit = run_trade_layer(
            df,
            sym,
            events,
            candidate2_build_trade
        )

        all_audit.extend(audit)

        print(
            "  trade statuses:",
            dict(
                Counter(
                    a["final_status"]
                    for a in audit
                )
            )
        )

        time.sleep(0.05)

    # ========================================================
    # GLOBAL COVERAGE
    # ========================================================

    sufficient = [
        c for c in coverages
        if c.get("sufficient_history")
    ]

    insufficient = [
        c for c in coverages
        if not c.get("sufficient_history")
    ]

    print("\n" + "=" * 75)

    print(
        f"Symbols total={len(coverages)}"
        f" | sufficient={len(sufficient)}"
        f" | insufficient={len(insufficient)}"
    )

    if insufficient:

        print(
            "\nINSUFFICIENT_HISTORY:"
        )

        for c in insufficient:

            print(
                f"  {c['symbol']}: "
                f"{c['coverage_days']}d | "
                f"candles={c['total_candles']} | "
                f"stop={c['stop_reason']}"
            )

    # ========================================================
    # DETECTION
    # ========================================================

    print(
        f"\n[DETECTION] "
        f"Total causal events={len(all_events)}"
    )

    by_direction = Counter(
        e["dir"]
        for e in all_events
    )

    print(
        f"  bullish={by_direction.get('bullish', 0)}"
        f" | bearish={by_direction.get('bearish', 0)}"
    )

    # ========================================================
    # LAYER B
    # ========================================================

    print_signal_metrics(all_signal)

    print(
        f"\n[LAYER B CONSISTENCY]"
    )

    print(
        f"  detected_events={len(all_events)}"
    )

    print(
        f"  measurements={len(all_signal)}"
    )

    print(
        f"  equal="
        f"{len(all_events) == len(all_signal)}"
    )

    # ========================================================
    # LAYER C/D
    # ========================================================

    status_counts = Counter(
        a["final_status"]
        for a in all_audit
    )

    print(
        "\n[LAYER C/D] "
        "Audit Status Breakdown:"
    )

    for status, count in sorted(
        status_counts.items()
    ):
        print(
            f"  {status}: {count}"
        )

    # Rejection reasons.
    rejection_reasons = Counter(
        a.get("reject_reason")
        for a in all_audit
        if a["final_status"].startswith(
            "REJECTED@"
        )
    )

    if rejection_reasons:

        print(
            "\n[REJECTION REASONS]"
        )

        for reason, count in rejection_reasons.most_common():

            print(
                f"  {reason}: {count}"
            )

    # ========================================================
    # LAYER E
    # ========================================================

    summary, traded = evaluate(
        all_audit
    )

    print(
        "\n[LAYER E] Evaluation"
    )

    print(
        f"  n_traded="
        f"{summary['n_traded']}"
    )

    print(
        f"  TP={summary['n_tp']}"
        f" | SL={summary['n_sl']}"
        f" | TIMEOUT={summary['n_timeout']}"
        f" | OPEN_AT_DATASET_END="
        f"{summary['n_open_at_dataset_end']}"
        f" | AMBIGUOUS="
        f"{summary['n_ambiguous']}"
    )

    if summary["n_traded"] > 0:

        print(
            f"  WR={summary['win_rate_pct']}%"
        )

        print(
            f"  GrossExp="
            f"{summary['gross_exp']}R"
        )

        print(
            f"  NetExp="
            f"{summary['net_exp']}R"
        )

        print(
            f"  GrossTotalR="
            f"{summary['gross_total_R']}"
        )

        print(
            f"  NetTotalR="
            f"{summary['net_total_R']}"
        )

        print(
            f"  PF Gross="
            f"{summary['pf_gross']}"
        )

        print(
            f"  PF Net="
            f"{summary['pf_net']}"
        )

        print(
            f"  MaxDD Gross="
            f"{summary['maxdd_gross']}R"
        )

        print(
            f"  MaxDD Net="
            f"{summary['maxdd_net']}R"
        )

    else:

        print(
            "  NO CLOSED TRADES"
        )

    # ========================================================
    # PER SYMBOL
    # ========================================================

    if summary.get("per_symbol"):

        print(
            "\n[PER-SYMBOL RESULTS]"
        )

        ordered = sorted(
            summary["per_symbol"].items(),
            key=lambda x: (
                -x[1]["n"],
                x[0]
            )
        )

        for symbol, data in ordered:

            print(
                f"  {symbol}: "
                f"N={data['n']} | "
                f"GrossR={data['gross_total_R']} | "
                f"NetR={data['net_total_R']} | "
                f"GrossExp={data['gross_exp']}R | "
                f"NetExp={data['net_exp']}R"
            )

    # ========================================================
    # FINAL ENGINE CONSISTENCY
    # ========================================================

    detected_count = len(all_events)

    audit_count = len(all_audit)

    closed_count = len(traded)

    open_count = summary[
        "n_open_at_dataset_end"
    ]

    print(
        "\n[ENGINE CONSISTENCY]"
    )

    print(
        f"  detected_events={detected_count}"
    )

    print(
        f"  audit_records={audit_count}"
    )

    print(
        f"  closed_trades={closed_count}"
    )

    print(
        f"  open_at_dataset_end={open_count}"
    )

    print(
        f"  accounting_check="
        f"{detected_count == audit_count}"
    )

    print("=" * 75)

    print(
        "\nRUN COMPLETE."
    )

    print(
        "IMPORTANT: This is IS validation only."
    )

    print(
        "No parameter optimization or OOS/WF conclusion "
        "is performed by this run."
)
