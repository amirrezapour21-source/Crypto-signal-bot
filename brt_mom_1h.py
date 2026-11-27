"""BRT-S MOM-1H (paper). Same momentum rule as MOM-LONG/MOM-SHORT but on 1-hour candles, LONG and SHORT:
close beyond the 30-candle extreme with a body >= 0.8 ATR, entry = next 1h open, stop = 2 ATR (risk must be 1%-12% of price),
trailing stop 3 ATR, max 60 one-hour candles. Needs an hourly dispatch. Own state file and workflow. Other bots are not touched."""
import brt_mom_gen as G

G.use_one_hour()
G.configure("brt_mom_1h_state.json", "MOM-1H", "brt_mom_1h_trail_v1", G.setup_both, "MOM", tf="1H")

if __name__ == "__main__":
    G.E.main()
