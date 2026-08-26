# V3 Probabilistic Pipeline

**Status:** Implemented parallel to v1 (2026-08-25)

## What shipped

- Evaluation foundation: [`src/projection/evaluation/`](../src/projection/evaluation/), [`scripts/run_rolling_backtest.py`](../scripts/run_rolling_backtest.py), [`scripts/report_calibration.py`](../scripts/report_calibration.py)
- Learned reconcile weights: [`scripts/fit_reconcile_alpha.py`](../scripts/fit_reconcile_alpha.py) → `models/reconcile_calibration.json`
- Conditional intervals: [`scripts/fit_interval_models.py`](../scripts/fit_interval_models.py) → `models/interval_models/`
- Interim Monte Carlo: [`src/projection/inference/simulate.py`](../src/projection/inference/simulate.py), [`scripts/run_v3_simulation.py`](../scripts/run_v3_simulation.py)
- Generative modules: [`src/projection/models/`](../src/projection/models/), [`src/projection/inference/reconcile.py`](../src/projection/inference/reconcile.py)
- Weekly features: [`src/projection/data/features_weekly.py`](../src/projection/data/features_weekly.py)
- Draft percentiles: [`src/draft_assistant/prepare.py`](../src/draft_assistant/prepare.py) merges `output/model_v3/simulation_summary_<season>.csv`
- Means backtest: [`scripts/backtest_v3_means.py`](../scripts/backtest_v3_means.py) → `output/model_v3/means_backtest.json`
- Promotion gate: [`scripts/v3_promotion_gate.py`](../scripts/v3_promotion_gate.py)

## Point-engine decision (2026-08-25)

**v3 does not replace the point engine yet.**

| Verdict | Meaning |
|---|---|
| `hold_v1_default` | Missing sim/calibration artifacts |
| `simulation_ready` | Percentile UI overlay OK; ranks/VORP stay v1 (+ optional v2 blend) |
| `promote_v3_means` | Generative means beat v1 and blend on rolling fantasy MAE/Spearman |

Hard gate artifact: `output/model_v3/means_backtest.json` from
`scripts/backtest_v3_means.py`. Soft/hard verdicts from
`scripts/v3_promotion_gate.py`.

Flagged draft cutover (default off):

```bash
python -m src.draft_assistant.prepare --season 2026 --v3-means
# or experiment without gate:
python -m src.draft_assistant.prepare --season 2026 --force-v3-means
```

## Production default

v1 `compose_board` + optional v1/v2 draft ensemble remain the production point
engine until `v3_promotion_gate.py` reports `promote_v3_means`. Treat
`simulation_ready` as UI-only.

## Run order

```bash
python scripts/run_rolling_backtest.py
python scripts/report_calibration.py
python scripts/fit_reconcile_alpha.py
python scripts/fit_interval_models.py
python scripts/run_v3_simulation.py --season 2026 --draws 1000
python scripts/backtest_v3_means.py --draws 200
python scripts/v3_promotion_gate.py
python -m src.draft_assistant.prepare --season 2026
```
