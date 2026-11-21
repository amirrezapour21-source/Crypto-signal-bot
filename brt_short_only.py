"""BRT-S, SHORT only. Runs the unchanged engine crypto_signal_bot_brt_short_v1.py with every LONG setup switched off.
Decision 2026-11-21: LONG was net-negative in every backtest slice (4 years, 70 coins, both halves). SHORT logic, state file, messages: unchanged."""
import crypto_signal_bot_brt_short_v1 as E

_orig_setup_at = E.setup_at


def setup_at_short_only(d, i):
    r = _orig_setup_at(d, i)
    if r is None or r[0] != -1:
        return None
    return r


E.setup_at = setup_at_short_only

if __name__ == "__main__":
    E.main()
