"""BRT-S MOM-LONG-P (paper). Momentum breakout, but the entry waits for a pullback: after a 4h breakout candle (close above the 30-candle high,
green body >= 0.8 ATR) the signal comes only on the FIRST candle within the next 6 candles that returns to the broken level (within 0.25 ATR)
and closes back above it as a green candle, with no close below the level in between. No pullback = no trade.
Entry = next candle open, stop = 2 ATR below entry, trailing stop 3 ATR, max 60 candles. Own state file. Other bots are not touched."""
import brt_mom_gen as G

G.configure("brt_mom_long_p_state.json", "MOM-LONG-P", "brt_mom_long_p_trail_v1",
            lambda d, i: G.setup_pullback(d, i, 1), "PB")

if __name__ == "__main__":
    G.E.main()
