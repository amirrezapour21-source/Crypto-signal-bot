"""
Crypto Signal Bot V4 — Institutional Liquidity Reversal Strategy
--------------------------------------------------------------------
نسخه مستقل و جدید، طبق:
V4 Final Specification
V4 Final Developer Directive
V4 Breakout/Continuation Patch

⚠️ این فایل کاملاً جدا از crypto_signal_bot.py (نسخه V3) است.

PHASE 1:
Real OHLCV Data Layer — KuCoin

PHASE 2:
Structure Engine
Swing High / Swing Low
BOS با تأیید Close

PHASE 3:
Breakout / Continuation Engine

PATH A:
Compression → Displacement → Volume → Follow-through

PATH B:
Trend Continuation
Market Regime موافق جهت
+
Displacement
+
Volume
+
Follow-through

این نسخه برای تست چند نماد طراحی شده است.
فعلاً هیچ Telegram Signal ارسال نمی‌کند.
"""

import requests
import pandas as pd
import numpy as np
import time
import traceback


# ============================================================
# CONFIG
# ============================================================

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"

KUCOIN_INTERVALS = {
    "4h": "4hour",
    "15m": "15min",
    "1h": "1hour",
    "1d": "1day",
}

# ------------------------------------------------------------
# نمادهای تست
# ------------------------------------------------------------

TEST_SYMBOLS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "ADA-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "DOT-USDT",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_get(url, params=None, retries=3):
    """
    درخواست HTTP با Retry و مدیریت Rate Limit.
    """

    resp = None

    for attempt in range(retries):

        try:
            resp = requests.get(
                url,
                params=params,
                timeout=20
            )

        except requests.RequestException as e:

            print(
                f"⚠️ Request error "
                f"(attempt {attempt + 1}/{retries}): {e}",
                flush=True
            )

            time.sleep(5)
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code == 429:

            wait_time = 10 * (attempt + 1)

            print(
                f"⚠️ Rate limit 429 → waiting {wait_time}s",
                flush=True
            )

            time.sleep(wait_time)
            continue

        print(
            f"⚠️ HTTP {resp.status_code}: {resp.text[:300]}",
            flush=True
        )

        return resp

    return resp


# ============================================================
# PHASE 1 — REAL OHLCV DATA LAYER
# ============================================================

def fetch_kucoin_candles(
    symbol,
    interval_key,
    limit=100,
    end_at=None
):
    """
    دریافت کندل‌های واقعی KuCoin.

    KuCoin response format:
    [
        timestamp,
        open,
        close,
        high,
        low,
        volume,
        turnover
    ]
    """

    if interval_key not in KUCOIN_INTERVALS:
        raise ValueError(
            f"interval نامعتبر: {interval_key}"
        )

    params = {
        "symbol": symbol,
        "type": KUCOIN_INTERVALS[interval_key],
    }

    if end_at is not None:
        params["endAt"] = int(end_at)

    resp = safe_get(
        KUCOIN_BASE,
        params=params
    )

    if resp is None:
        return None

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    if data.get("code") != "200000":
        print(
            f"⚠️ KuCoin API error for {symbol}: "
            f"{data.get('msg', 'unknown')}",
            flush=True
        )
        return None

    raw = data.get("data", [])

    if not raw:
        return None

    rows = []

    for r in raw:

        if len(r) < 6:
            continue

        try:

            rows.append({
                "time": int(r[0]),
                "open": float(r[1]),
                "close": float(r[2]),
                "high": float(r[3]),
                "low": float(r[4]),
                "volume": float(r[5]),
            })

        except (ValueError, TypeError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)

    df = (
        df
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def get_ohlcv_v4(
    symbol,
    interval_key,
    total_candles=150
):
    """
    دریافت تعداد کندل مورد نیاز از KuCoin.

    چون API ممکن است محدودیت تعداد کندل داشته باشد،
    داده‌ها در چند درخواست دریافت می‌شوند.
    """

    all_dfs = []

    end_at = None

    remaining = total_candles

    while remaining > 0:

        request_limit = min(100, remaining)

        df = fetch_kucoin_candles(
            symbol=symbol,
            interval_key=interval_key,
            limit=request_limit,
            end_at=end_at
        )

        if df is None or df.empty:
            break

        all_dfs.append(df)

        remaining -= len(df)

        oldest_time = df["time"].min()

        end_at = int(oldest_time) - 1

        time.sleep(0.4)

        if len(df) < request_limit:
            break

    if not all_dfs:
        return None

    full_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    full_df = (
        full_df
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    full_df["dt"] = pd.to_datetime(
        full_df["time"],
        unit="s",
        utc=True
    )

    return (
        full_df
        .tail(total_candles)
        .reset_index(drop=True)
    )


def drop_unclosed_candle(df, interval_key):

    if df is None or df.empty:
        return df

    interval_seconds = {
        "4h": 4 * 3600,
        "15m": 15 * 60,
        "1h": 3600,
        "1d": 86400,
    }

    now_ts = int(time.time())

    last_candle_time = int(
        df["time"].iloc[-1]
    )

    duration = interval_seconds.get(
        interval_key,
        0
    )

    if now_ts < last_candle_time + duration:

        return (
            df
            .iloc[:-1]
            .reset_index(drop=True)
        )

    return df


# ============================================================
# PHASE 2 — STRUCTURE ENGINE
# ============================================================

def find_swings(
    df,
    left=3,
    right=3
):

    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []

    if len(df) < left + right + 1:
        return swing_highs, swing_lows

    for i in range(
        left,
        len(df) - right
    ):

        window_h = highs[
            i - left:
            i + right + 1
        ]

        window_l = lows[
            i - left:
            i + right + 1
        ]

        # Swing High
        if (
            highs[i] == window_h.max()
            and
            (window_h == highs[i]).sum() == 1
        ):

            swing_highs.append({
                "index": i,
                "price": float(highs[i]),
                "time": df["dt"].iloc[i],
            })

        # Swing Low
        if (
            lows[i] == window_l.min()
            and
            (window_l == lows[i]).sum() == 1
        ):

            swing_lows.append({
                "index": i,
                "price": float(lows[i]),
                "time": df["dt"].iloc[i],
            })

    return swing_highs, swing_lows


def classify_market_structure(
    df,
    left=3,
    right=3
):

    swing_highs, swing_lows = find_swings(
        df,
        left,
        right
    )

    if (
        len(swing_highs) < 2
        or
        len(swing_lows) < 2
    ):

        return {
            "regime": "range",
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
        }

    last_two_highs = swing_highs[-2:]
    last_two_lows = swing_lows[-2:]

    higher_high = (
        last_two_highs[1]["price"]
        >
        last_two_highs[0]["price"]
    )

    higher_low = (
        last_two_lows[1]["price"]
        >
        last_two_lows[0]["price"]
    )

    lower_high = (
        last_two_highs[1]["price"]
        <
        last_two_highs[0]["price"]
    )

    lower_low = (
        last_two_lows[1]["price"]
        <
        last_two_lows[0]["price"]
    )

    if higher_high and higher_low:

        regime = "up"

    elif lower_high and lower_low:

        regime = "down"

    else:

        regime = "range"

    return {
        "regime": regime,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
    }


def detect_bos(
    df,
    swing_highs,
    swing_lows,
    lookback_candles=30
):

    events = []

    recent_start = max(
        0,
        len(df) - lookback_candles
    )

    for i in range(
        recent_start,
        len(df)
    ):

        close_price = float(
            df["close"].iloc[i]
        )

        relevant_highs = [
            s
            for s in swing_highs
            if s["index"] < i
        ]

        relevant_lows = [
            s
            for s in swing_lows
            if s["index"] < i
        ]

        # Bullish BOS
        if relevant_highs:

            last_high = relevant_highs[-1]

            if close_price > last_high["price"]:

                events.append({
                    "type": "bullish_bos",
                    "index": i,
                    "time": df["dt"].iloc[i],
                    "close": close_price,
                    "broken_level": last_high["price"],
                    "broken_level_time": last_high["time"],
                })

        # Bearish BOS
        if relevant_lows:

            last_low = relevant_lows[-1]

            if close_price < last_low["price"]:

                events.append({
                    "type": "bearish_bos",
                    "index": i,
                    "time": df["dt"].iloc[i],
                    "close": close_price,
                    "broken_level": last_low["price"],
                    "broken_level_time": last_low["time"],
                })

    return events


# ============================================================
# PHASE 3 — BREAKOUT / CONTINUATION ENGINE
# ============================================================

def compute_avg_body(
    df,
    lookback=20
):

    body = (
        df["close"] -
        df["open"]
    ).abs()

    return body.rolling(
        window=lookback
    ).mean()


def compute_avg_volume(
    df,
    lookback=20
):

    return df["volume"].rolling(
        window=lookback
    ).mean()


def compute_avg_range(
    df,
    lookback=20
):

    return (
        df["high"] -
        df["low"]
    ).rolling(
        window=lookback
    ).mean()


def detect_compression(
    df,
    idx,
    lookback=20,
    threshold=0.75
):

    if idx < lookback * 2:

        return False, None

    recent_range = (
        df["high"] -
        df["low"]
    ).iloc[
        idx - lookback:
        idx
    ].mean()

    longer_range = (
        df["high"] -
        df["low"]
    ).iloc[
        idx - lookback * 2:
        idx
    ].mean()

    if longer_range == 0:

        return False, None

    ratio = (
        recent_range /
        longer_range
    )

    return (
        ratio < threshold,
        ratio
    )


def detect_displacement(
    df,
    idx,
    avg_body,
    avg_volume,
    direction,
    body_multiplier=1.2,
    volume_multiplier=1.3,
    close_position_pct=0.25
):

    if idx >= len(df):
        return False

    if (
        pd.isna(avg_body.iloc[idx])
        or
        pd.isna(avg_volume.iloc[idx])
        or
        avg_body.iloc[idx] == 0
        or
        avg_volume.iloc[idx] == 0
    ):

        return False

    row = df.iloc[idx]

    body_size = abs(
        row["close"] -
        row["open"]
    )

    candle_range = (
        row["high"] -
        row["low"]
    )

    if candle_range <= 0:
        return False

    # Body strength
    body_ok = (
        body_size >=
        avg_body.iloc[idx] *
        body_multiplier
    )

    # Volume strength
    volume_ok = (
        row["volume"] >=
        avg_volume.iloc[idx] *
        volume_multiplier
    )

    # Close position
    close_location = (
        row["close"] -
        row["low"]
    ) / candle_range

    if direction == "bullish":

        direction_ok = (
            row["close"] >
            row["open"]
        )

        close_ok = (
            close_location >=
            1 - close_position_pct
        )

    elif direction == "bearish":

        direction_ok = (
            row["close"] <
            row["open"]
        )

        close_ok = (
            close_location <=
            close_position_pct
        )

    else:

        return False

    return (
        body_ok
        and
        volume_ok
        and
        direction_ok
        and
        close_ok
    )


def detect_follow_through(
    df,
    idx,
    direction,
    lookahead=2,
    minimum_progress=0.001
):

    """
    بررسی ادامه حرکت بعد از Breakout.

    برای جلوگیری از Look-ahead در تولید واقعی،
    این تابع فقط در تست تاریخی استفاده شود.
    """

    end_idx = min(
        len(df),
        idx + lookahead + 1
    )

    if idx + 1 >= end_idx:
        return False

    breakout_close = float(
        df["close"].iloc[idx]
    )

    future = df.iloc[
        idx + 1:
        end_idx
    ]

    if future.empty:
        return False

    if direction == "bullish":

        max_future_close = future["close"].max()

        progress = (
            max_future_close -
            breakout_close
        ) / breakout_close

        return progress >= minimum_progress

    if direction == "bearish":

        min_future_close = future["close"].min()

        progress = (
            breakout_close -
            min_future_close
        ) / breakout_close

        return progress >= minimum_progress

    return False


def evaluate_breakout_setup(
    df,
    idx,
    regime,
    direction,
    avg_body,
    avg_volume
):

    """
    دو مسیر مستقل:

    PATH A:
    Compression
    +
    Displacement
    +
    Volume
    +
    Follow-through

    PATH B:
    Trend Continuation
    +
    Regime Alignment
    +
    Displacement
    +
    Volume
    +
    Follow-through

    Compression در PATH B اجباری نیست.
    """

    compression, compression_ratio = (
        detect_compression(
            df,
            idx
        )
    )

    displacement = detect_displacement(
        df,
        idx,
        avg_body,
        avg_volume,
        direction
    )

    follow_through = detect_follow_through(
        df,
        idx,
        direction
    )

    # --------------------------------------------------------
    # PATH A — Compression → Expansion
    # --------------------------------------------------------

    path_a = (
        compression
        and
        displacement
        and
        follow_through
    )

    # --------------------------------------------------------
    # PATH B — Trend Continuation
    # --------------------------------------------------------

    regime_alignment = (
        (
            direction == "bullish"
            and
            regime == "up"
        )
        or
        (
            direction == "bearish"
            and
            regime == "down"
        )
    )

    path_b = (
        regime_alignment
        and
        displacement
        and
        follow_through
    )

    if path_a:

        return {
            "valid": True,
            "path": "PATH_A_COMPRESSION_EXPANSION",
            "compression": compression,
            "compression_ratio": compression_ratio,
            "displacement": displacement,
            "follow_through": follow_through,
            "regime_alignment": regime_alignment,
        }

    if path_b:

        return {
            "valid": True,
            "path": "PATH_B_TREND_CONTINUATION",
            "compression": compression,
            "compression_ratio": compression_ratio,
            "displacement": displacement,
            "follow_through": follow_through,
            "regime_alignment": regime_alignment,
        }

    reasons = []

    if not compression:
        reasons.append(
            "no_compression"
        )

    if not displacement:
        reasons.append(
            "weak_displacement_or_volume"
        )

    if not follow_through:
        reasons.append(
            "no_follow_through"
        )

    if not regime_alignment:
        reasons.append(
            "regime_not_aligned"
        )

    return {
        "valid": False,
        "path": None,
        "compression": compression,
        "compression_ratio": compression_ratio,
        "displacement": displacement,
        "follow_through": follow_through,
        "regime_alignment": regime_alignment,
        "reasons": reasons,
    }


# ============================================================
# PHASE 3 — FIND ALL BREAKOUT EVENTS
# ============================================================

def scan_breakout_setups(
    df,
    structure,
    lookback=30
):

    avg_body = compute_avg_body(
        df
    )

    avg_volume = compute_avg_volume(
        df
    )

    events = detect_bos(
        df,
        structure["swing_highs"],
        structure["swing_lows"],
        lookback_candles=lookback
    )

    results = []

    for event in events:

        idx = event["index"]

        if event["type"] == "bullish_bos":

            direction = "bullish"

        elif event["type"] == "bearish_bos":

            direction = "bearish"

        else:

            continue

        evaluation = evaluate_breakout_setup(
            df=df,
            idx=idx,
            regime=structure["regime"],
            direction=direction,
            avg_body=avg_body,
            avg_volume=avg_volume
        )

        results.append({
            **event,
            **evaluation
        })

    return results


# ============================================================
# OUTPUT HELPERS
# ============================================================

def print_setup_result(
    symbol,
    setup
):

    timestamp = setup["time"]

    direction = setup.get(
        "type",
        ""
    )

    close = setup.get(
        "close",
        0
    )

    if setup["valid"]:

        print(
            f"\n{timestamp} | "
            f"{direction} | "
            f"close={close:.6f} | "
            f"✅ VALID SETUP",
            flush=True
        )

        print(
            f"  PATH: {setup['path']}",
            flush=True
        )

        print(
            f"  Compression: "
            f"{setup['compression']} "
            f"(ratio={setup['compression_ratio']})",
            flush=True
        )

        print(
            f"  Displacement: "
            f"{setup['displacement']}",
            flush=True
        )

        print(
            f"  Follow-through: "
            f"{setup['follow_through']}",
            flush=True
        )

        print(
            f"  Regime Alignment: "
            f"{setup['regime_alignment']}",
            flush=True
        )

    else:

        reasons = ", ".join(
            setup.get(
                "reasons",
                []
            )
        )

        print(
            f"\n{timestamp} | "
            f"{direction} | "
            f"close={close:.6f} | "
            f"❌ REJECTED",
            flush=True
        )

        print(
            f"  Compression: "
            f"{setup['compression']} "
            f"(ratio={setup['compression_ratio']})",
            flush=True
        )

        print(
            f"  Displacement: "
            f"{setup['displacement']}",
            flush=True
        )

        print(
            f"  Follow-through: "
            f"{setup['follow_through']}",
            flush=True
        )

        print(
            f"  Regime Alignment: "
            f"{setup['regime_alignment']}",
            flush=True
        )

        print(
            f"  دلایل رد شدن: {reasons}",
            flush=True
        )


# ============================================================
# SYMBOL SCANNER
# ============================================================

def scan_symbol(
    symbol,
    candles_4h=150
):

    print(
        "\n"
        + "=" * 68,
        flush=True
    )

    print(
        f"SYMBOL: {symbol}",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    # --------------------------------------------------------
    # Download 4H data
    # --------------------------------------------------------

    df = get_ohlcv_v4(
        symbol=symbol,
        interval_key="4h",
        total_candles=candles_4h
    )

    if df is None or df.empty:

        print(
            "❌ داده 4H دریافت نشد",
            flush=True
        )

        return {
            "symbol": symbol,
            "status": "NO_DATA",
            "valid_setups": [],
        }

    df = drop_unclosed_candle(
        df,
        "4h"
    )

    if df is None or len(df) < 60:

        print(
            f"❌ کندل کافی نیست: "
            f"{0 if df is None else len(df)}",
            flush=True
        )

        return {
            "symbol": symbol,
            "status": "INSUFFICIENT_DATA",
            "valid_setups": [],
        }

    print(
        f"✅ {len(df)} کندل بسته‌شده دریافت شد",
        flush=True
    )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    structure = classify_market_structure(
        df
    )

    print(
        f"\nMarket Regime (4H): "
        f"{structure['regime'].upper()}",
        flush=True
    )

    print(
        f"Swing Highs: "
        f"{len(structure['swing_highs'])}",
        flush=True
    )

    print(
        f"Swing Lows: "
        f"{len(structure['swing_lows'])}",
        flush=True
    )

    # --------------------------------------------------------
    # BOS
    # --------------------------------------------------------

    bos_events = detect_bos(
        df,
        structure["swing_highs"],
        structure["swing_lows"],
        lookback_candles=30
    )

    print(
        f"تعداد رویداد BOS اخیر: "
        f"{len(bos_events)}",
        flush=True
    )

    # --------------------------------------------------------
    # Breakout Scanner
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 68,
        flush=True
    )

    print(
        "بررسی Breakout / Continuation Setups:",
        flush=True
    )

    print(
        "-" * 68,
        flush=True
    )

    setups = scan_breakout_setups(
        df,
        structure,
        lookback=30
    )

    valid_setups = []

    for setup in setups:

        print_setup_result(
            symbol,
            setup
        )

        if setup["valid"]:

            valid_setups.append(
                setup
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 68,
        flush=True
    )

    print(
        f"SUMMARY — {symbol}",
        flush=True
    )

    print(
        "-" * 68,
        flush=True
    )

    print(
        f"Regime: "
        f"{structure['regime'].upper()}",
        flush=True
    )

    print(
        f"BOS Events: "
        f"{len(bos_events)}",
        flush=True
    )

    print(
        f"Valid Setups: "
        f"{len(valid_setups)}",
        flush=True
    )

    if valid_setups:

        print(
            "\n🔥 VALID SETUPS:",
            flush=True
        )

        for setup in valid_setups:

            print(
                f"  {setup['time']} | "
                f"{setup['type']} | "
                f"{setup['path']} | "
                f"close={setup['close']:.6f}",
                flush=True
            )

    else:

        print(
            "\nهیچ Setup معتبر پیدا نشد.",
            flush=True
        )

    return {
        "symbol": symbol,
        "status": "OK",
        "regime": structure["regime"],
        "bos_count": len(bos_events),
        "valid_setups": valid_setups,
        "data": df,
    }


# ============================================================
# MULTI-SYMBOL SCANNER
# ============================================================

def run_multi_symbol_scan():

    print(
        "\n"
        + "=" * 68,
        flush=True
    )

    print(
        "CRYPTO SIGNAL BOT V4",
        flush=True
    )

    print(
        "MULTI-SYMBOL PHASE 1–3 TEST",
        flush=True
    )

    print(
        "KuCoin Real OHLCV + Structure + "
        "Breakout/Continuation",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    print(
        f"\nتعداد نمادهای تست: "
        f"{len(TEST_SYMBOLS)}",
        flush=True
    )

    all_results = []

    for symbol in TEST_SYMBOLS:

        try:

            result = scan_symbol(
                symbol=symbol,
                candles_4h=150
            )

            all_results.append(
                result
            )

        except Exception as e:

            print(
                f"\n❌ ERROR — {symbol}",
                flush=True
            )

            print(
                str(e),
                flush=True
            )

            traceback.print_exc()

            all_results.append({
                "symbol": symbol,
                "status": "ERROR",
                "valid_setups": [],
            })

        # جلوگیری از فشار روی API
        time.sleep(1)

    # ========================================================
    # FINAL GLOBAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 68,
        flush=True
    )

    print(
        "FINAL MULTI-SYMBOL SUMMARY",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    total_symbols = len(
        all_results
    )

    successful_symbols = sum(
        1
        for r in all_results
        if r.get("status") == "OK"
    )

    failed_symbols = (
        total_symbols -
        successful_symbols
    )

    total_valid = sum(
        len(
            r.get(
                "valid_setups",
                []
            )
        )
        for r in all_results
    )

    print(
        f"Total Symbols: "
        f"{total_symbols}",
        flush=True
    )

    print(
        f"Successful: "
        f"{successful_symbols}",
        flush=True
    )

    print(
        f"Failed: "
        f"{failed_symbols}",
        flush=True
    )

    print(
        f"Total Valid Setups: "
        f"{total_valid}",
        flush=True
    )

    print(
        "\n"
        + "-" * 68,
        flush=True
    )

    print(
        "SYMBOL RESULTS",
        flush=True
    )

    print(
        "-" * 68,
        flush=True
    )

    for result in all_results:

        symbol = result["symbol"]

        status = result.get(
            "status",
            "UNKNOWN"
        )

        regime = result.get(
            "regime",
            "-"
        )

        bos_count = result.get(
            "bos_count",
            0
        )

        valid_count = len(
            result.get(
                "valid_setups",
                []
            )
        )

        print(
            f"{symbol:<12} | "
            f"status={status:<18} | "
            f"regime={regime:<6} | "
            f"BOS={bos_count:<3} | "
            f"VALID={valid_count}",
            flush=True
        )

    # ========================================================
    # BEST CANDIDATES
    # ========================================================

    valid_candidates = []

    for result in all_results:

        for setup in result.get(
            "valid_setups",
            []
        ):

            valid_candidates.append({
                "symbol": result["symbol"],
                "time": setup["time"],
                "type": setup["type"],
                "path": setup["path"],
                "close": setup["close"],
            })

    print(
        "\n"
        + "=" * 68,
        flush=True
    )

    print(
        "VALID CANDIDATES FOR NEXT PHASE",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    if not valid_candidates:

        print(
            "هیچ کاندید معتبری در Phase 3 پیدا نشد.",
            flush=True
        )

    else:

        for i, candidate in enumerate(
            valid_candidates,
            start=1
        ):

            print(
                f"{i}. "
                f"{candidate['symbol']} | "
                f"{candidate['type']} | "
                f"{candidate['path']} | "
                f"close={candidate['close']:.6f} | "
                f"time={candidate['time']}",
                flush=True
            )

    print(
        "\n"
        + "=" * 68,
        flush=True
    )

    print(
        "PHASE 1–3 TEST FINISHED",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    return all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_multi_symbol_scan()
