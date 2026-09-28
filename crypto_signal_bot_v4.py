CANDIDATE 6 — FINAL VALIDATION STEP

Do NOT change the frozen strategy, parameters, filters, costs, entry/SL/TP logic, or data layer.

Run ONLY a lightweight 5-fold Walk-Forward validation:

- 4H
- same 35 valid symbols
- same 730-day dataset
- same frozen Candidate 6 rules
- primary target: TP = 2R
- SL = 1.25 ATR
- HOLD = 30
- same 0.003R total cost
- no overlap lock
- no short-only selection
- no parameter optimization
- no extra filters

Folds:
1) IS 0–50%, OOS 50–60%
2) IS 0–60%, OOS 60–70%
3) IS 0–70%, OOS 70–80%
4) IS 0–80%, OOS 80–90%
5) IS 0–90%, OOS 90–100%

For each OOS fold report ONLY:
events, traded, open_at_fold_end, WR, NetExp, NetPF, NetTotalR, NetMaxDD.

Then report aggregate across all 5 OOS folds.

IMPORTANT:
Trades must NOT use candles beyond that fold's OOS end.
No parameter fitting is allowed; IS is context only.

FINAL GATE:
PASS only if:
1) aggregate NetExp > 0
2) aggregate NetPF > 1
3) at least 4 of 5 folds have positive NetExp

If PASS → NEXT STEP = COST STRESS + SYMBOL ROBUSTNESS.
If FAIL → ARCHIVE Candidate 6. DO NOT optimize or patch it.

Keep output concise.
