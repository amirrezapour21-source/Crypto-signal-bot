"""BRT-S MOM-SHORT-P (paper). Mirror of MOM-LONG-P: after a 4h breakdown candle (close below the 30-candle low, red body >= 0.8 ATR)
the signal comes only on the FIRST candle within the next 6 candles that returns to the broken level (within 0.25 ATR)
and closes back below it as a red candle, with no close above the level in between. No pullback = no trade.
Entry = next candle open, stop = 2 ATR above entry, trailing stop 3 ATR, max 60 candles. Own state file. Other bots are not touched."""
import brt_mom_gen as G

G.configure("brt_mom_short_p_state.json", "MOM-SHORT-P", "brt_mom_short_p_trail_v1",
            lambda d, i: G.setup_pullback(d, i, -1), "PB")

if __name__ == "__main__":
    G.E.main()
