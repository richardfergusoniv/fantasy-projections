# RB carry vacancy (2026-08-14)

## Diagnosis (`scripts/diag_rb_carry_vacancy.py`)

Against the post-`NAMED_RUSH_COVERAGE` / replacement 2026 board vs matched Sleeper:

| Component (RB1 / depth_rank=1) | Result |
|---|---|
| Season EV (`pred_season` − Sleeper season) | **−19.3** mean (under) |
| Rate vs Sleeper/18 | +2.3 carries/g (over; Sleeper uses gp=18) |
| Pre-reconcile season − Sleeper | −43.8 (fill closes gap) |
| Mean `team_rushing_volume_scale` | 1.16 (still UP-fills toward coverage) |
| Ceiling pins / RB SHARE &lt; 0.70 tripwires | **0** |

Interpretation: the harmful lead-back bias that blocked carry α was **rush reconcile fill into thin rooms** (Jacobs-style 25.0 caps). That path was already fixed by `NAMED_RUSH_COVERAGE=0.814` plus replacement-level committee rows. Remaining “rate overshoot vs Sleeper/18” is largely a games-framing difference, not a live ceiling bug. Season-EV lead backs sit **below** Sleeper on average.

Vacancy leverage at α=1.0 is large on high-turnover rooms (e.g. v_net=0.30 → scale 1.43) but reconcile still cannot push named volume past `min(anchor, max(raw, anchor×coverage))`, and rookies already net `(1−α)`.

## Decision

**Shipped** `INCUMBENT_VACANCY_ALPHA["carry"] = 1.0` (measured value).

No additional Gate B rate shrink: evidence did not support a new rate correction once coverage + replacement removed the inflate path. Hierarchical rush L2 (C2) further budgets RB group carries before reconcile.

## Validation

- Unit tests (runtime / rush / pass / depth / rookies / team grain) expected green.
- CAPPED / RB SHARE tripwire spirit unchanged; board should remain silent on those.
