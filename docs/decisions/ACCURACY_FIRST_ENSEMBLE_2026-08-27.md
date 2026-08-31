# Accuracy-first 2026 ensemble

**Date:** 2026-08-27
**Verdict:** `promote_accuracy_ensemble`
**Scope:** separate top-120 ADP point board; native v1 and v3 intervals unchanged

## Decision

The accuracy-first bake-off improves the untouched 2025 draft-relevant
holdout and therefore publishes a separate 2026 best-estimate point board at
`output/accuracy_first_2026/fantasy_points_2026.csv`.

| Top-120 ADP holdout | MAE | Spearman |
|---|---:|---:|
| Shipped v1/v2 incumbent | 58.72 | .504 |
| Selected accuracy ensemble | **53.14** | **.602** |

The paired bootstrap also clears zero: candidate-minus-incumbent MAE delta
95% interval **[-9.81, -1.66]** points; Spearman delta **[+.026, +.179]**.

## What was selected

- QB and TE retain the shipped v1/v2 weights.
- RB uses 0.10 v1, 0.30 v2, and 0.60 ADP-implied points.
- WR uses 0.55 v2 and 0.45 ADP-implied points.
- Only players inside the frozen top-120 ADP population receive the new
  weights. Everyone else remains on the incumbent forecast.

v3 receives **zero point weight at every position**. The full WR arm tied the
market-without-v3 arm only because its fitted v3 coefficient was zero. The
tie-break correctly selects the simpler arm. Thus v3 adds no measured marginal
point accuracy here and remains the calibrated p10/p50/p90 and finish-
probability overlay.

## Evaluation contract

- Historical v3 p50 values use the exact 1,000-draw production
  joint-bootstrap path and reproduce the accepted calibration metrics to
  machine precision.
- ADP-to-points curves are monotonic and position-specific. Every target uses
  only earlier seasons for calibration.
- Candidate weights fit on 2024; 2025 is the untouched selection holdout.
  The accepted design is then refit on 2024-2025 for 2026.
- Historical ECR is unavailable, so 2026 ECR is diagnostic only and receives
  no fitted weight.

## Artifacts and reproduction

- `output/accuracy_first_2026/report.json`: metrics, weights, coverage,
  bootstrap intervals, exact-path parity, and the v3 marginal verdict.
- `output/accuracy_first_2026/evaluation_players.parquet`: player-level 2024
  and 2025 inputs and candidate predictions.
- `output/accuracy_first_2026/ensemble_weights.json`: selected/refit weights
  with source hashes.
- `output/accuracy_first_2026/freeze_manifest.json`: hashes for the frozen
  board, reports, model inputs, consensus snapshots, and v3 fold checkpoints.

Reproduce with:

```bash
python scripts/evaluate_accuracy_first_ensemble.py
```

Parity-checked fold checkpoints are reused only when their hashes, draw count,
and calibration hash match.
