# Weekly v2 port provenance

Team-first weekly projection code was consolidated from the sibling repository
[`fantasy-projections-2`](../../fantasy-projections-2) into
`src/projection/weekly/` on 2026-08-30. Each ported module carries an inline
provenance comment pointing here.

## Source → destination mapping

| Source (fantasy-projections-2) | Destination (this repo) |
|---|---|
| `src/projections/pipeline/accounting.py` | `src/projection/weekly/pipeline/accounting.py` |
| `src/projections/pipeline/availability.py` | `src/projection/weekly/pipeline/availability.py` |
| `src/projections/pipeline/veteran_projector.py` | `src/projection/weekly/pipeline/veteran_projector.py` |
| `src/projections/pipeline/rookie_projector.py` | `src/projection/weekly/pipeline/rookie_projector.py` |
| `src/projections/pipeline/season_projector.py` | `src/projection/weekly/pipeline/season_projector.py` |
| `src/projections/pipeline/__init__.py` | `src/projection/weekly/pipeline/__init__.py` |
| `src/projections/models/base.py` | `src/projection/weekly/models/base.py` |
| `src/projections/models/registry.py` | `src/projection/weekly/models/registry.py` |
| `src/projections/models/volume.py` | `src/projection/weekly/models/volume.py` |
| `src/projections/models/efficiency.py` | `src/projection/weekly/models/efficiency.py` |
| `src/projections/models/team_totals.py` | `src/projection/weekly/models/team_totals.py` |
| `src/projections/models/calibration.py` | `src/projection/weekly/models/calibration.py` |
| `src/projections/models/rookie.py` | `src/projection/weekly/models/rookie.py` |
| `src/projections/models/__init__.py` | `src/projection/weekly/models/__init__.py` |
| `src/projections/features/injuries.py` | `src/projection/weekly/features/injuries.py` |
| `src/projections/features/depth.py` | `src/projection/weekly/features/depth.py` |
| `src/projections/features/team_context.py` | `src/projection/weekly/features/team_context.py` |
| `src/projections/features/leakage.py` | `src/projection/weekly/features/leakage.py` |
| `src/projections/features/rolling.py` | `src/projection/weekly/features/rolling.py` |
| `src/projections/features/effective_depth.py` | `src/projection/weekly/features/effective_depth.py` |
| `src/projections/features/contracts.py` | `src/projection/weekly/features/contracts.py` |
| `src/projections/features/sleeper.py` | `src/projection/weekly/features/sleeper.py` |
| `src/projections/features/xfp.py` | `src/projection/weekly/features/xfp.py` |
| `src/projections/features/advanced_public.py` | `src/projection/weekly/features/advanced_public.py` |
| `src/projections/features/panel.py` | `src/projection/weekly/features/panel.py` |
| `src/projections/features/rookie_college.py` | `src/projection/weekly/features/rookie_college.py` |
| `src/projections/features/__init__.py` | `src/projection/weekly/features/__init__.py` |
| `src/projections/config/scoring.py` | `src/projection/weekly/config/scoring.py` |
| `src/projections/config/paths.py` | `src/projection/weekly/config/paths.py` |
| `src/projections/config/__init__.py` | `src/projection/weekly/config/__init__.py` |
| `src/projections/data/teams.py` | `src/projection/weekly/data/teams.py` |
| `src/projections/data/ids.py` | `src/projection/weekly/data/ids.py` |
| `src/projections/data/nflverse_loader.py` | `src/projection/weekly/data/nflverse_loader.py` |
| `src/projections/data/espn_injuries.py` | `src/projection/weekly/data/espn_injuries.py` |
| `src/projections/data/sleeper.py` | `src/projection/weekly/data/sleeper.py` |
| `src/projections/data/cfbd_loader.py` | `src/projection/weekly/data/cfbd_loader.py` |
| `src/projections/data/__init__.py` | `src/projection/weekly/data/__init__.py` |
| `src/projections/scoring/fantasy_points.py` | `src/projection/weekly/scoring/fantasy_points.py` |
| `src/projections/scoring/__init__.py` | `src/projection/weekly/scoring/__init__.py` |

## Import path changes

- `projections.*` → `src.projection.weekly.*`
- Default artifact dirs: `output/weekly_v2/` and `output/weekly_v2/models/` (override with
  `WEEKLY_V2_OUTPUTS_DIR` / `WEEKLY_V2_MODELS_DIR`).

## Leakage-safe patterns preserved

- `features/leakage.py`: `filter_as_of` excludes the target `(season, week)` from history.
- `features/rolling.py`: rolling windows use `shift(1)` before rolling aggregates.
- `pipeline/accounting.py`: strips same-week box-score actuals before projecting.
- `pipeline/availability.py`: `estimate_projected_games` uses only seasons `< target_season`.
- `data/features_weekly.py` (existing v3 path): `targets_share_roll3` / `carries_share_roll3`
  remain shift-lagged separately; not modified by this port.

## Port tooling

- One-shot copy script: `scripts/port_weekly_v2.py`
- Parity fixture: `tests/fixtures/weekly_v2_parity.json`
- Parity tests: `tests/test_weekly_v2_parity.py`

## Known blockers / follow-ups

1. **`nflreadpy`**: `data/nflverse_loader.py` still expects `nflreadpy` (v2 dependency).
   This repo's main ingest path uses `nfl-data-py`. Install `nflreadpy` separately or
   add a bridge adapter before running live panel builds from this package.
2. **Trained model artifacts**: `predict_volume`, `predict_efficiency`, `predict_team_totals`,
   and `predict_rookie_fp_pg` expect joblib models under `output/weekly_v2/models/`.
   Copy or retrain from v2 `models/*.joblib` before end-to-end weekly projection.
3. **CFBD API key**: college/rookie features in `data/cfbd_loader.py` require
   `CFBD_API_KEY` when building rookie panels at runtime.
