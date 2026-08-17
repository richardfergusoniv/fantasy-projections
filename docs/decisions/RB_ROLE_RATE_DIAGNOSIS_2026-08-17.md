# RB Role-Rate Diagnosis — 2026-08-17

## Decision

Do not add an RB-receiving correction, and do not replace the full-season
role-rate exposure with the availability model's projected games.

The earlier single-fold artifact diagnosis mixed per-appearance production
with the model's per-eligible-week role-rate contract. Re-measuring on the
actual model basis changes the conclusion: deep RBs are already overprojected
at the **rate** level, while the availability model is far too low to serve as
their season exposure. The tier-1 receiving miss is real but not RB-specific
enough to support an RB-only constant.

## Method

- Rebuilt the 2016-2026 nflverse database and all derived OL/OC tables.
- Verified the rebuilt eligible drafted-rookie cohorts exactly match the
  pre-deletion counts (QB 104, RB 189, WR 289, TE 121).
- Used strictly-forward veteran folds 2021→2022 to train 2022→2023,
  expanding through 2024→2025.
- Used the shipped composed receiving path
  (`backtest._predict_reframed_receiving`) and the matching
  `{stat}_per_elig` actual.
- Calibration is `sum(actual role rate) / sum(predicted role rate)`;
  1.00 is calibrated, below 1.00 is overprediction.

## Tier-1 receiving is not uniquely an RB problem

| Holdout | RB | WR | TE |
|---|---:|---:|---:|
| 2023 | 1.067 (n=34) | 1.032 (n=76) | 1.056 (n=39) |
| 2024 | 1.063 (n=35) | 1.039 (n=82) | 1.005 (n=38) |
| 2025 | 1.149 (n=27) | 0.991 (n=30) | 1.192 (n=29) |
| pooled | **1.090 (n=96)** | **1.027 (n=188)** | **1.082 (n=106)** |

RB receiving is consistently low across the three valid folds, but TE has a
similar pooled miss and a larger 2025 miss. This is not evidence for an
RB-specific multiplier.

Within RB, the calibration changes direction by depth tier:

| Depth tier | Receiving-yards calibration | Rushing-yards calibration |
|---|---:|---:|
| 1 | 1.090 | 1.022 |
| 2 | 0.910 | 0.922 |
| 3 | 1.147 | 1.069 |
| 4 | 0.603 | 0.602 |
| 5 | 0.539 | 0.565 |

An RB-wide upward correction would improve tier 1 while worsening tiers 2,
4, and 5. A tier-1-only correction would recreate the retired post-model
depth multiplier on a thin slice rather than improve the fitted model.

## Deep RBs: exposure is not the safe fix

The 2025 tier-4 RB cohort (n=38) is overpredicted at the role-rate level:

| Quantity | Receiving yards | Rushing yards |
|---|---:|---:|
| role-rate calibration | 0.603 | 0.602 |
| actual mean eligible weeks | 15.11 | 15.11 |
| actual mean appearance games | 5.26 | 5.26 |
| availability-model projected games | 1.68 | 1.68 |
| calibration using flat 17-week exposure | 0.592 | 0.570 |
| calibration using projected-games exposure | 5.482 | 4.743 |

The flat 17-week board does overstate deep-player season exposure, but the
availability model overcorrects dramatically. More importantly, the 0.60
role-rate calibration shows that exposure is not the whole error: the rate
model itself assigns too much average role to this zero-inflated tier.

The earlier artifact-only conclusion that the rate was conservative came
from dividing by games appeared. That is a conditional-on-playing rate and
is not the quantity this project now predicts.

## What would justify revisiting

Revisit only with a model that handles the zero mass inside the fitted role
rate (or with materially more forward folds). Do not add a hurdle merely for
the point estimate without a held-out MAE win; that approach has already
measured neutral twice in the rookie path. Any future exposure change must
land between the demonstrably high 17-week assumption and the demonstrably
low 1.68-game availability output, and must improve both calibration and MAE
out of sample.
