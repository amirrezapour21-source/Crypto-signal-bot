N4-SD SUPPLY/DEMAND ZONE STUDY - PRE-REGISTRATION (version 1)

QUESTION
Does a simple, causal supply/demand-zone retest-rejection rule on 4h bars of liquid USDT-M perpetuals produce a positive mean net return per trade after a 15-minute execution delay, a 30 bps round-trip cost, stops and time exits?

1. DATA AND SAMPLE
Binance USDT-M 1m klines (data.binance.vision monthly files), symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT. 4h bars are built from 1m candles on UTC 4h boundaries (O first open, H max, L min, C last close). A bar with any missing 1m candle is invalid (no zone, no signal, ignored).
Development: trade entries 2022-11-01 .. 2024-12-31 (months 2022-10 .. 2024-12 loaded; 2022-10 is warm-up only).
Holdout: trade entries 2025-01-01 .. 2026-10-31 (single use, locked; opened only through the gate in section 6). No parameter is chosen from any outcome.

2. ZONES (causal)
ATR = mean of (H-L) over the last 14 bars including the current bar.
Pivot high at bar j: H_j > max(H of bars j-3..j-1) and H_j >= max(H of bars j+1..j+3). Pivot low symmetric with L. A pivot is known only at the close of bar j+3 and its zone is active from bar j+4.
Supply zone (short side): near edge = max(O_j, C_j), far edge = H_j. Demand zone (long side): near edge = min(O_j, C_j), far edge = L_j.
A zone is removed when (a) a bar closes beyond its far edge, (b) it is more than 30 bars old, or (c) it is touched (supply: bar H >= near edge; demand: bar L <= near edge). Each zone can therefore produce at most one signal, on its first touch.

3. SIGNAL AND ENTRY
Signal at the close of the touching bar if the bar closes back on the outside of the near edge (supply: C < near edge; demand: C > near edge) = rejection.
Entry: market order at the open of the 1m candle 15 minutes after that bar close (stress D = 30 min).
Skipped if: a position in the same symbol is still open; the stop distance (entry to stop) is below 1.0% of entry (economic rule: round-trip cost 0.30% must not exceed about one third of the risk); the stop is not on the adverse side of entry; any 1m candle in the trade window is missing; the trade would end after the loaded data.
News filter: no entry if any scheduled CPI, NFP or FOMC release (calendar of the N2-E registration, SPEC sha256 05954b52b57df2e28a0952b91ac9fa4958bc2a68f5af33571b7e8f9f56d8ad1c) lies within 120 minutes before to 240 minutes after the entry time. Positions are not closed around news.

4. EXITS AND COSTS
Stop: far edge of the zone moved 0.25 x ATR further away from the zone (below the demand low, above the supply high). R = stop distance. Target: 2 R from the actual entry price. Time exit: close of the 1m candle 12 bars (2880 minutes) after entry.
Checked on every 1m candle from the entry candle; if stop and target are both inside one candle the stop is taken; stop fill = stop price, or the candle open if it is already beyond the stop; target fill = target price.
Return per trade = direction x (exit/entry - 1) - cost; cost 30 bps round trip (base) or 40 bps (stress), deducted once. One trade per zone; fixed notional per trade.

5. HYPOTHESIS AND TEST
H-SD: mean net return per trade (30 bps, all symbols, long and short pooled) > 0.
Test: bootstrap of the mean with entry-day clusters (resample entry days with replacement), 10,000 draws, numpy default_rng seed 20261125; supported if the 2.5th percentile of the mean net return is > 0.
Development gate: the holdout is run only if development n >= 100 trades and the development mean net return (30 bps) > 0. Single holdout run, rules unchanged: confirmed only if the holdout 2.5th percentile > 0; an executable-edge statement additionally needs holdout mean net > 0 at 40 bps and at D = 30. Development and holdout are never pooled; nothing is changed after outcomes are seen.

6. DESCRIPTIVE (no selection)
Per symbol; long vs supply/demand side; exit type shares (stop/target/time); mean stop distance in bps and cost as a share of R; trades with a news release inside the holding period versus without; D = 30 and cost 40 bps.

7. LIMITS
Zones and costs are assumptions; no spread, depth or queue data; no stop slippage model beyond the open-gap rule; about 8 symbols x 26 months, trades are not fully independent across symbols (day clusters used); no consensus or first-release news data.
