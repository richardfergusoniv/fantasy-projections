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

### Measured result (2026-08-26, 200 draws, 3 rolling folds)

| Fold | v1 MAE / rho | blend MAE / rho | v3 interim | v3 generative |
|---|---|---|---|---|
| 2023 | 29.602 / .7564 | **28.721 / .7730** | 30.610 / .7283 | 29.798 / .7591 |
| 2024 | 28.029 / .7821 | 28.093 / .7780 | 29.466 / .7593 | **28.021 / .7876** |
| 2025 | 27.714 / .7583 | **26.025 / .7727** | 29.678 / .7007 | 28.939 / .7543 |

The generative arm is competitive: it wins 2024 on both metrics, is level on
2023 (better rho, +0.2 MAE), and loses 2025. It does not clear the gate,
which requires beating **both** v1 and the blend on **every** fold. The blend
is the harder incumbent and the one the draft board actually ships.

2025 by position, v1 vs generative: QB 39.207/.7800 vs 40.296/.7696;
RB 35.252/.6906 vs 38.635/.6921; WR 24.682/.7801 vs 25.332/.**7830**;
TE 17.285/.8348 vs **17.217**/.8328. RB is the weakest cell.

### Correction to the earlier reading (2026-08-25)

The first recorded backtest put the generative arm at **47.21 MAE / .611 rho**
on 2025, with QB Spearman of **-0.037**, and that was taken as evidence
against the architecture. It was not. It measured four defects in the
composer, none of which had a test:

1. QBs were selected by position on a per-(player, stat) board, so each
   emitted one passing line **per stat**.
2. Every QB in a room drew the whole team's attempts.
3. Draws were made per-game and summed into `fantasy_pts_season`, making a
   receiver's season **one game**.
4. Conversion rates were league constants, so QBs were near-indistinguishable.

A fifth predates the composer: `allocate_opportunities` keys rooms by
position, so WR/TE/RB each received a full team's targets — a **3x**
over-allocation. The per-game scale masked it (3/17 reads as
under-projection); fixing the scale exposed it, and the interim rematch came
back at MAE 77 before it was found.

Fixing these moved the generative arm from 47.21/.611 to 28.939/.754 on 2025.
The verdict is unchanged — v3 stays off the point engine — but the reason is
now "it does not beat the shipped incumbent on every fold", not "generative
modelling loses".

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
