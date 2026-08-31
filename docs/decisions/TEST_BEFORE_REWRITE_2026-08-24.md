# Test-before-rewrite go/no-go

**Generated:** 2026-08-25T01:20:24.191474+00:00

**Verdict:** `do_not_rewrite`

v1/v2 post-process blend shows multi-season ADP residual edge that beats carry-forward and improves 2025 holdout MAE/Spearman. Ship the ensemble as a draft-assistant post-process; do not rewrite compose_board or LightGBM. Revisit targeted levers only if residual remains after the blend is in production use.

## Shipped follow-up (2026-08-24)

Implemented without touching the projection engine:

- Canonical weights: [`src/draft_assistant/ensemble_weights.json`](../../src/draft_assistant/ensemble_weights.json) (fit 2023–2024, holdout 2025).
- `python -m src.draft_assistant.prepare` applies the blend by default when `output/model_v2/fantasy_points_<season>.csv` exists; `--no-ensemble` keeps a pure v1 board.
- Board meta sets `model_id` to `v1_v2_ensemble` when the blend is applied.
- Team / totals views remain native v1 (`output/projections_*.csv` unchanged).

## Signals

```json
{
  "seasons_edge_beats_carry": {
    "v1": [
      2024,
      2025
    ],
    "v2": [
      2023,
      2024,
      2025
    ],
    "blend": [
      2023,
      2024,
      2025
    ]
  },
  "multi_season_beat_carry": {
    "v1": [
      2024,
      2025
    ],
    "v2": [
      2023,
      2024,
      2025
    ],
    "blend": [
      2023,
      2024,
      2025
    ]
  },
  "actionable_summary_raw": {
    "v1": {
      "seasons_with_points_edge": [
        2023,
        2024,
        2025
      ],
      "multi_season_edge": true
    },
    "v2": {
      "seasons_with_points_edge": [
        2023,
        2024,
        2025
      ],
      "multi_season_edge": true
    },
    "blend": {
      "seasons_with_points_edge": [
        2023,
        2024,
        2025
      ],
      "multi_season_edge": true
    },
    "carry_forward": {
      "seasons_with_points_edge": [
        2023,
        2024,
        2025
      ],
      "multi_season_edge": true
    }
  },
  "blend_helps_holdout_mae": true,
  "holdout_mae": {
    "v1": 33.9222999396039,
    "v2": 32.54179611945449,
    "blend": 31.700424617449237
  },
  "holdout_spearman": {
    "v1": 0.7420652554737535,
    "v2": 0.7321912371962188,
    "blend": 0.762923899859361
  },
  "rolling_dispersion": {
    "spearman_min": 0.7631427902512102,
    "spearman_max": 0.7893216215671945,
    "spearman_range": 0.026178831315984308,
    "mae_min": 27.71401815333269,
    "mae_max": 29.601874030498795,
    "mae_range": 1.8878558771661034,
    "n_folds": 3
  },
  "untestable_levers": [
    {
      "lever": "curated_depth_chart",
      "why": "src/depth_chart/starters_<season>.csv is hand-researched for 2026 only"
    },
    {
      "lever": "status_overrides_ir_pup",
      "why": "status_overrides_<season>.csv is 2026-only"
    },
    {
      "lever": "elite_residual_correction",
      "why": "models/corrections.joblib spans the target season on the ship path"
    },
    {
      "lever": "prediction_intervals",
      "why": "interval_residuals.csv is fit across the target season"
    },
    {
      "lever": "roster_vacancy_boosts",
      "why": "vacancy alphas live in roster_moves upstream of the leakage-safe harness"
    }
  ]
}
```

## Rules applied

- Require ADP residual edge to beat carry-forward (not merely nonzero)
- No full rewrite if blend captures multi-season edge and helps holdout
- Targeted changes only if residual persists after blend
- Full generative/weekly/props rewrite only after explicit product decision
