# Leakage-safe 2024 -> 2025 fantasy evaluation

## Result

The model was trained only on the 2021->2022, 2022->2023, and 2023->2024
transitions, then scored against 2025 half-PPR season points and positional
finish. The evaluation contains all 755 players on the earliest 2025 regular-
season roster snapshot with status ACT, DEV, RES, INA, or EXE. Position and
team are frozen from that snapshot. The population is 107 QB, 185 RB, 298 WR,
and 165 TE, including 166 preseason-defined rookies, 166 zero-game outcomes,
and 215 zero-point outcomes.

The model has a complete forecast for 661 players. The other 94 are veterans
without a 2024 feature row; they remain in the end-to-end population with zero
model points rather than disappearing through an inner join. All 166 rookies
have a forecast from draft/Week-1-roster inputs and <=2024 rookie bucket data.

### All-eligible results

| Pos | Model Spearman | Carry-forward | Availability baseline | Model points MAE | Best baseline MAE | Top-tier hits |
|---|---:|---:|---:|---:|---:|---:|
| QB | 0.792 | 0.651 | 0.667 | 41.72 | 44.55 | 6/12 |
| RB | 0.729 | 0.579 | 0.585 | 33.27 | 36.00 | 15/24 |
| WR | 0.769 | 0.687 | 0.684 | 23.11 | 26.59 | 22/36 |
| TE | 0.837 | 0.688 | 0.699 | 16.55 | 20.73 | 5/12 |

The model improves rank correlation and absolute season-point error at all four
positions. Tier identification is less decisive: QB top-12 precision/recall is
0.50 versus carry-forward's 0.583; RB improves to 0.625; WR ties both baselines
at 0.611; TE improves to 0.417 from 0.25. Ties at the cutoff are included, so a
top-K set can expand beyond K on another dataset.

VORP uses the method-specific kth score as replacement: QB13, RB25, WR37, and
TE13 by default. Model VORP MAE is 47.98/35.74/25.72/39.28 for QB/RB/WR/TE.
The model beats both baselines for RB and WR. Availability-adjusted QB is
slightly better (44.93), and both TE baselines are better (31.45 and 32.18).
VORP here centers each method on its own replacement estimate, so that result
does not by itself identify a causal calibration defect. Separately, the
model's TE13 replacement estimate is only 92.99 points versus the actual
133.00, evidence that TE replacement-level calibration warrants monitoring
despite strong TE ordering.

## Leakage and population controls

- Forecast history is sliced to `season <= 2024` before any fitted transform.
- The 2025 rookie outcome columns are explicitly erased before rookie
  prediction. Historical rookie cohorts use the same contracted Week-1 roster
  definition, drafted-first deduplication, and Week-1 UDFAs.
- Actual 2025 component totals are joined only after forecasts are frozen.
  Multi-position outcomes are grouped by player and ranked under the frozen
  preseason position.
- Actual points are rebuilt from component totals using 0.04 passing yards,
  4 passing TD, -2 interception, 0.1 rushing/receiving yard, 6 rushing/receiving
  TD, and 0.5 reception. Sleeper data is not used.
- Spearman ranks use average ties, including the large zero-outcome tail.
  Top-tier sets include every player tied at the kth score. Replacement value
  uses the configurable kth order statistic.

## Production parity and limits

The evaluation applies the shipped veteran depth-rate ladder, predicted
availability, exposure-weighted receiving composition, mutually exclusive QB
volume allocation, team passing/rushing anchors, player stat constraints, and
canonical season-total exposure. The ladder is not applied to rookies.

Historical curated roles, target-year coordinator context, and the production
elite residual correction are unavailable on a strictly preseason-consistent
2025 path and are therefore omitted. The eligible universe is Week 1, not all
August camp participants; players cut before Week 1 cannot be evaluated. Both
`all_eligible` and `forecast_covered` scopes are emitted so coverage cannot be
mistaken for performance.

## Artifacts

- `output/fantasy_evaluation_2025.csv`: player-level population, predictions,
  actual points, average positional ranks, exposure, coverage, and baselines.
- `output/fantasy_evaluation_summary_2025.csv`: position/method/scope metrics.
- `output/fantasy_evaluation_summary_2025.json`: metrics plus population,
  training-pair, tier, replacement-rank, and limitation metadata.

Run with:

```text
python -m src.projection.fantasy_evaluation
```

Tier and replacement ranks are configurable with `--tier-ranks` and
`--replacement-ranks`. The full test suite passes: 57 tests.
