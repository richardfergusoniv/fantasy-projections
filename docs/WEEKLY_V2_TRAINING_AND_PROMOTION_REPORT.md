# Weekly v2 training and promotion report

Generated: 2026-08-31 (local, final pass)

Downstream (separately evaluated; does not rewrite this promotion result):
[WEEKLY_JOINT_USAGE_DRAWS_REPORT.md](WEEKLY_JOINT_USAGE_DRAWS_REPORT.md) and
[WEEKLY_V2_VOLUME_TUNING_REPORT.md](WEEKLY_V2_VOLUME_TUNING_REPORT.md).

## Completion audit (requirement → evidence)

| Requirement | Status | Evidence |
|---|---|---|
| Inventory artifacts | **Done** | `scripts/weekly_v2_inventory.py` → `output/weekly_v2/weekly_v2_inventory.json` |
| Port training/evaluation entrypoints | **Done** | `scripts/weekly_v2_{train,evaluate,fit_calibration,project}.py` |
| Train 2026 candidate (2016–2025) | **Done** | `output/weekly_v2/models/season=2026/` + manifest v2 |
| Strict historical evaluation | **Done, failed promotion** | `output/weekly_v2/preseason_backtest.json` — dispersion gate |
| Fix S1 false-green | **Done** | Real inference path + `tests/app/test_weekly_v2_false_green.py` |
| Honest promotion gates | **Done** | `auto_publish_allowed=false`; automatic production blocked (`test_automatic_trained_promotion_does_not_swap_pointer`) |
| League-specific scoring (six leagues) | **Done** | `output/weekly_v2/six_league_scoring_shadow.json` + shadow_sync `league_scoring_shadow` |
| Shadow validation (six Sleeper leagues) | **Done (read-only)** | `output/live_shadow/sleeper_sync_report.json` (prior pass) + scoring shadow script |
| Weekly stat-draw partitions | **Partial** | `stat_draw_partition.json` persisted; conservation/K-DST gates open |
| Five go/no-go decisions | **Done** | Executive table below |
| Preserve fixture/fallback | **Done** | Fixture path unchanged; trained requires full provenance chain |
| `auto_publish_allowed=true` | **Not met** | Evaluation promotion still failing |

## Executive go / no-go decisions

| Decision | Result | Evidence |
|---|---|---|
| Trained artifact classification | **GO** (with caveats) | Manifest v2 validates; nine required joblibs load under `output/weekly_v2/models/season=2026/`; `weekly_v2_readiness(2026,1).state == trained` |
| Manual trained shadow publication | **NO-GO** | Historical evaluation promotion gate failed (2023–2024 dispersion 0.689/0.688 vs 0.70 min); shadow sync can exercise trained path manually (`automatic=False`) but auto-publish remains blocked |
| Automatic weekly publication | **NO-GO** | `auto_publish_allowed: false` — evaluation promotion failed; production artifact gate blocks automatic runs |
| Use trained results for start/sit decisions | **NO-GO** | Same evaluation failure; draw partitions / league-scored weekly distributions not fully validated |
| Public-internet deployment | **NO-GO** | Unchanged external blockers (PostgreSQL, Docker, email, OpenAI, deployment) |

## 1. Models: retrained, not migrated

Sibling `fantasy-projections-2` joblib artifacts were **not** copied. Pickle namespace (`projections.*` vs `src.projection.weekly.*`) makes direct migration unsafe.

**Action taken:** Retrained in-repo from verified sibling panel copy:

- Panel: `data/processed/player_week_panel.parquet` (SHA256 `ce5505850376e4e135fb0cc7fd7c1c11773e2532ba5d56a63cfb78858c3ed7ff`, 49,803,893 bytes)
- Training seasons: 2016–2025 (target 2026 excluded)
- Candidate directory: `output/weekly_v2/models/season=2026/`
- `model_version`: `weekly_v2_2026_1a8d6f0598fa6ccc` (retrained with sibling `tuning_selection.json` two-stage volume)

## 2. Critical false-green defect and fix

**Defect (S1):** `weekly_v2_bridge` labelled runs `trained` from nine joblib files + schema manifest while `weekly_run.py` always called `_weekly_rows()` (hash-scaled preseason means).

**Fix:**

1. `src/app/projections/weekly_inference.py` — executes `project_week_with_rookies()` and persists provenance-linked parquet output.
2. `src/app/projections/weekly_manifest.py` — manifest v2 with SHA256 verification and safe path validation.
3. `src/app/projections/weekly_v2_bridge.py` — `trained` requires manifest + loadable models + output provenance hash linkage; `auto_publish_allowed` additionally requires passing evaluation promotion.
4. `src/app/projections/weekly_run.py` — `STATE_TRAINED` path calls real inference; fallback retains explicit `preseason_bundle_scaled` derivation.
5. `src/app/releases/gates.py` — `validate_inference_provenance` blocks trained label with hash-scaled rows; artifact gate respects `auto_publish_allowed`.
6. `tests/app/test_weekly_v2_false_green.py` — regression proving dummy joblibs alone cannot publish; automatic production block when eval fails
7. `src/app/projections/weekly_draws.py` — stat-draw partition manifests with SHA256 linkage
8. Fixed trained promotion partition gate ordering in `weekly_run.py` (validate after register)

## 3. Evaluation vs baselines

Strict leave-one-season-out preseason evaluation (`scripts/weekly_v2_evaluate.py`, 2022–2025):

| Gate | Result |
|---|---|
| Raw promotion | **FAILED** — dispersion below 0.70 all seasons |
| Calibrated promotion (post `weekly_v2_fit_calibration.py`) | **FAILED** — 2023 (0.689) and 2024 (0.688) dispersion below 0.70 policy minimum after WR/TE slope cap adjustment (1.52) |

Artifact: `output/weekly_v2/preseason_backtest.json`

Representative 2022 metrics: MAE 4.01, rank_corr 0.45, dispersion 0.51, interval coverage 0.73, coverage 100%.

**Remediation:** Complete calibration loop (OOF → fit → retrain with `calibration.json` in season dir → re-evaluate). Consider tuning volume architecture via `weekly_v2_tune_preseason.py`.

## 4. League scoring / draw validation

Weekly inference emits component stats in `mean_json` (attempts, yards, TDs, receptions).
The league scoring layer applies each compiled Sleeper contract via `score_stat_draw`.

**Completed this pass:**
- `src/app/projections/weekly_stat_draw.py` — weekly model columns → draw stat keys
- `src/app/projections/weekly_league_scoring.py` — per-league scoring of weekly component rows
- `scripts/weekly_v2_six_league_scoring.py` — scores trained week-1 output against all six live shadow contracts
- `tests/test_weekly_league_scoring.py` — PPR vs half-PPR, PPFD draw-level, inference ≠ scaling, six-league shadow validation

**Six-league shadow scoring (2026-08-31):**
- Artifact: `output/weekly_v2/six_league_scoring_shadow.json`
- Six distinct contract hashes; cross-league spread > 0.25 on sample players; all contracts publishable
- PPFD leagues score linear components only — first-down stats are not yet modeled in weekly-v2 output, so PPFD contribution is understated until a documented conditional rate model exists

**Not completed:** Conservation gates across teammates, K/DST weekly models integrated into readiness. Stat-draw partitions are now persisted (`stat_draw_partition.json`) with SHA256 verification at promotion time.

## 5. Commands run

| Command | Outcome |
|---|---|
| `uv sync --all-extras --dev` | OK |
| `uv run pytest tests/app/test_weekly_v2_false_green.py …` | 16 passed (after test isolation fix) |
| `uv run python scripts/weekly_v2_inventory.py` | OK → `output/weekly_v2/weekly_v2_inventory.json` |
| `uv run python scripts/weekly_v2_train.py --train-start 2016 --train-end 2025 --target-season 2026 --skip-cfbd` | OK (models + manifest) |
| `uv run python scripts/weekly_v2_evaluate.py --start 2022 --end 2025` | Exit 2 (promotion failed) |
| `uv run python scripts/weekly_v2_fit_calibration.py` | OK |
| `uv run python scripts/weekly_v2_project.py --season 2026 --week 1` | OK (933 players) |
| `run_weekly_inference(2026, 1)` | OK — output SHA256 `2810503962ee1f8f…` |
| `uv run python scripts/weekly_v2_six_league_scoring.py` | OK — six distinct contracts, cross-league spread validated |
| `uv run pytest tests/test_weekly_panel_leakage.py tests/test_weekly_league_scoring.py` | 6 passed |
| `uv run python scripts/audit_blueprint_mvp.py` | **49/49 passed** |
| Node/web e2e | **Blocked** (Node/npm not exercised) |

## 6. Artifact locations and hashes

| Artifact | Path / hash |
|---|---|
| Manifest | `output/weekly_v2/models/season=2026/manifest.json` |
| Team totals | `60830ad323dadba4192b555bd5e421d474cd3496ab3b8db0505e518e410e109c` |
| Volume QB | `c4aa1cf5a04b62b6ac87139b2d0945d9fe5ad666d77510cd38be167e8c2b84c2` |
| Week-1 output | `output/weekly_v2/season=2026/week=01/weekly_projections.parquet` (`2810503962ee1f8f…`) |
| Output provenance | `output/weekly_v2/season=2026/week=01/output_provenance.json` |
| Evaluation | `output/weekly_v2/preseason_backtest.json` |
| Stat-draw partition | `output/weekly_v2/season=2026/week=01/stat_draw_partition.json` |

## 7. Remaining steps

### Model / data
- Re-run evaluation after calibration integrated into season candidate
- **Volume tuning pass (2026-08-31 / 2026-09-01):** Valid **no-go**. Harness repaired; nested grid complete under `output/weekly_v2/experiments/volume_tune_20260831_v2/` — no candidate passes frozen dispersion gates (closest: `legacy_direct` 2023=0.6915 / 2024=0.7026). See [WEEKLY_V2_VOLUME_TUNING_REPORT.md](WEEKLY_V2_VOLUME_TUNING_REPORT.md). Existing 2026 candidate unchanged; auto-publish remains blocked.
- Optional CFBD college features (`CFBD_API_KEY`) for rookie model quality
- Field-level leakage audit with poisoned-future tests on real panel — **done** (`tests/test_weekly_panel_leakage.py`)

### Operational
- Shadow promotion failure-injection / rollback exercise for trained candidate
- Six-league draw-level scoring shadow comparison — **done** (`output/weekly_v2/six_league_scoring_shadow.json`)
- Wire draw partitions for weekly grain (not season-long artifacts)

### External
- PostgreSQL, Docker, email, OpenAI, public deployment (unchanged)
- Enable recurring production schedule only via explicit operator flag after all gates pass

## 8. Files changed (summary)

**Training / evaluation:** `scripts/weekly_v2_*.py`, `src/projection/weekly/evaluate/*`, `src/projection/weekly/models/selection.py`, `src/projection/weekly/config/paths.py`, `pyproject.toml` (nflreadpy)

**Inference / application:** `src/app/projections/weekly_inference.py`, `weekly_manifest.py`, `weekly_v2_bridge.py`, `weekly_run.py`, `src/app/releases/gates.py`, `src/app/api/v1/operations.py`, `src/app/league/sleeper/shadow_sync.py`

**Tests:** `tests/app/test_weekly_v2_false_green.py`, `tests/app/test_weekly_runs.py`

**Documentation:** `docs/WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md` (this file)

## Operator action for automatic publication

**Do not enable.** After calibration re-evaluation passes, manually review `preseason_backtest.json`, run shadow promotion, then set the documented production schedule flag — not done automatically by this task.
