# QB Projection Final Repair Report

**Date:** 2026-09-03  
**Branch:** `cursor/qb-projection-final-repair-59f0`  
**Sealed baseline (unchanged):** `v2_baseline_20260830`  
**Verdict:** **NO-GO**

## Executive decision

Historical rolling-origin selection does **not** support promoting a repaired QB architecture over the sealed baseline. Experimental arms that lift Lamar/Burrow on the 2026 board fail the required starter MAE + Spearman gates on fit seasons / untouched 2025 holdout when evaluated without using the final holdout for selection.

Therefore:

- Do **not** create a promotion-ready immutable release namespace.
- Do **not** overwrite `v2_baseline_20260830`.
- Do **not** move `active_release_2026.json`.
- Do **not** deploy or upload to production Storage.
- Keep RB/WR/TE sealed estimates unchanged (verified).

The repair **code, tests, diagnostics, and 2026 arm boards** are delivered for review under `output/qb_repair/` and `src/projection/qb_repair/`.

## Commands and results

```bash
uv sync --frozen --extra jobs --extra dev
python -m pytest tests/test_qb_projection_final_repair.py \
  tests/test_remediation_options.py \
  tests/test_yardage_repair.py \
  tests/test_composition_unification.py -q
# 29 passed

python scripts/qb_projection_final_repair.py
# stage attribution: 32 QB1s
# Lamar provenance: mobile rushing lost in raw rate model
# 2026 arms: all preserve non-QB pred_pg invariance
# selection: baseline; verdict NO-GO
```

### Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `output/qb_repair/qb1_stage_attribution.csv` | `9498b8d64359f85ae057a088fe28f3dcaef58873097f83b84f2e2f810d9e941a` |
| `output/qb_repair/selection_decision.json` | `200d385d7b241a42d9bcad75fefcb5424af439df2715fa23089f03e13048e271` |
| `output/qb_repair/lamar_rush_provenance.json` | `8312921b84a6403f3b8957797479e568ddbe142358d2eabb419466e73f066b2f` |
| `output/qb_repair/sanity_2026.json` | `0e249419ef8e40ad4ae6517c158328af0f6cec9623e4a759875e68373412e8f0` |
| Active pointer | `7499f11a51a787703cc47e32b311fd8e17f9d45c119df86f4c31ca8749efd6e9` |
| Sealed manifest `v2_baseline_20260830` | `5a8e14536aa7b062b1e5ff6e64aa78356847fb063579d0a42cdbdc5cc159fbb1` |

Candidate namespace: **none** (`candidate_publish.json`: `published=false`, reason `gates_returned_NO-GO`).

## 1. Sealed QB board stage attribution

Source: re-compose of `output/projections_2026_raw.csv` via `compose_board_stages`, joined to accuracy-first ensemble points.

Full table: `output/qb_repair/qb1_stage_attribution.csv` (32 projected QB1s).

### Focus players (half-PPR PPG)

| Player | Final ens. rank | Raw | Post-exposure | Post-team-volume | Post-TD | Compose final | Ensemble | Raw carries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lamar Jackson | 18 | 13.49 | 13.49 | 11.99 | 11.09 | 11.09 | 16.72 | 4.42 |
| Joe Burrow | 20 | 14.01 | 14.01 | 10.90 | 9.37 | 9.37 | 16.50 | 1.89 |

Interpretation:

1. **Burrow:** large loss at team-volume reconciliation (backup volume still forces the starter below a defensible share under the shipped protect-starter floor), then further TD-rate clip.
2. **Lamar:** already weak at raw forecast; team-volume and TD clip remove more points; **carries remain 4.42 through every compose stage** — reconciliation never restores rushing.

Ensemble 40/60 v1/v2 lifts both via v2 season points, but sealed component rates on the published board remain the under-rushed / under-allocated lines (Lamar 4.51 car / 21.8 rush yds; Burrow 30.4 att / 203.5 pass yds).

## 2. Experimental arms (2026, production artifacts untouched)

| Arm | Non-QB invariance | Notes |
|---|---|---|
| `baseline` | pass | Shipped compose |
| `allocation` | pass | Historical QB1/backup passing allocation |
| `multi_season_prior` | pass | Established-QB multi-season rate prior |
| `mobile_rush_prior` | pass | Mobile archetype rush components only |
| `allocation_plus_priors` | pass | Both fixes |

Arm fantasy boards: `output/qb_repair/fantasy_qb_<arm>_2026.csv`.

### Illustrative 2026 board under `allocation_plus_priors` (not selected)

| Player | Rank | PPG | Att | Car | Rush yds |
|---|---:|---:|---:|---:|---:|
| Josh Allen | 1 | 19.98 | 24.2 | 7.55 | 36.3 |
| Jalen Hurts | 2 | 19.66 | 22.3 | 9.78 | 38.3 |
| Lamar Jackson | 3 | 17.47 | 22.4 | 8.74 | 49.1 |
| Drake Maye | 4 | 16.85 | 28.1 | 4.81 | 26.3 |
| Joe Burrow | 20 | 13.60 | 24.3 | 3.26 | 12.0 |

Lamar’s rush profile is restored by the prior. Burrow remains mid-board on the v1-compose path even after allocation; any ensemble lift is a separate question and was **not** approved by holdout reselection (see §6).

## 3. Structural reconciliation fix

Implemented in `src/projection/qb_repair/allocation.py`:

- Fits starter share of team passing claim from **prior seasons only** (attempt-leadership proxy when historical curated depth charts are absent).
- Allocates the QB room to the measured team share (`0.941` / `0.942`) while enforcing a historical starter-share floor.
- A backup’s inflated raw forecast cannot push a healthy tier-1 below that floor.
- No player/team hardcodes; no universal fantasy-point floor.
- Room conservation is exact on the allocation path; material violations are reported.
- RB/other positions still run through shipped `reconcile_team_volume` so non-QB rates stay bit-identical.

## 4. Multi-season rate prior (leakage-safe)

Implemented in `src/projection/qb_repair/rate_prior.py` + `history.py`:

- For target season \(T\), uses only seasons \(< T\).
- Lookback up to 4 seasons; partial seasons keep per-game rates but down-weight by evidence mass (`games / 12`, capped).
- Separate components: pass volume, Y/A / TD% / INT%, designed carries, scramble carries, total rush yards, rush-TD rate.
- Applies only to established projected QB1s with sufficient history; small samples regress toward mobile/pocket archetype means.
- Every applied player records input seasons, sample games, weight, components, and before/after adjustments.

## 5. Lamar designed-run / scramble provenance

Artifact: `output/qb_repair/lamar_rush_provenance.json`.

| Source | Carries/g | Rush yds/g | Designed car/g (where observed) |
|---|---:|---:|---:|
| Games-weighted prior (pre-2026) | ~8.03 | ~49+ | ~6.0 |
| Raw model 2026 | **4.42** | 26.7 | — |
| Sealed board | **4.51** | 21.8 | — |

Causes (no hardcoding; mechanism-level):

1. Veteran rate model already under-predicts rush volume vs multi-season evidence.
2. Designed-run historical usage (~6/g in 2023–24 pbp) is not preserved in the raw carries forecast.
3. Compose never restores rushing (team-volume siblings exclude rush stats).
4. Ship `QB_PARTIAL_PRIOR_SHRINK` remains disabled; 2025’s partial 13-game, 5.15 car/g season is allowed to dominate T−1 features.

Verdict string: `mobile_rushing_lost_in_raw_rate_model_and_unrepaired_by_compose`.

## 6. Ensemble reselection

`src/projection/qb_repair/gates.py::reselect_qb_ensemble_weights`:

- Fit season 2024 grid over v1/v2 weights (step 0.05).
- Untouched 2025 holdout decides promotion.
- ECR/ADP point weight fixed at **0** (diagnostic only).

Result: fit leader preferred ~70/30 v1/v2, but holdout MAE **worsened** vs incumbent 40/60 (`27.63 → 32.42`). **Retain 40/60.** Spearman ≈0.175 on the accuracy-first top-120 QB slice remains the known weak incumbent; no better leakage-safe blend cleared the holdout.

## 7. Evaluation policy and GO/NO-GO

### Design

- Fit / architecture selection on 2023–2024 only.
- 2025 reserved as untouched selection holdout.
- Segments: QB1, all-QB, mobile, pocket, partial-prior-season, high-confidence starter.
- Metrics: PPG MAE, season MAE, bias, Spearman, top-6/12 hit rate, calibration slope, fraction improved.
- Comparators: carry-forward, simple multi-season rate, v1 baseline, experimental arms.

### Fit-season leaderboard (starter PPG MAE ↑ worse)

| Arm | Starter PPG MAE | Starter Spearman | All-QB PPG MAE |
|---|---:|---:|---:|
| **baseline** | **6.517** | 0.349 | **10.833** |
| allocation | 6.545 | 0.358 | 10.914 |
| multi_season_prior | 6.701 | 0.400 | 18.038 |
| allocation_plus_priors | 6.757 | 0.417 | 18.092 |
| mobile_rush_prior | 6.906 | 0.366 | 17.308 |
| simple_multi_season_rate | 7.801 | 0.402 | 27.212 |
| carry_forward | 8.242 | 0.341 | 11.298 |

Selected arm: **baseline**.

### Holdout gate (2025 high-confidence starters)

Baseline starter PPG MAE 5.151 / Spearman 0.476. Selected arm identical → no improvement.

**Decision reasons:**

- `starter_ppg_mae_not_improved`
- `starter_spearman_not_improved`
- `selected_arm_is_baseline`

Component-level prior vs carry-forward on 2025 mobile rush rates is mixed and **positively biased** (2025 injury-shortened rush outcomes), reinforcing that the prior cannot be promoted from this holdout alone.

## 8. 2026 sanity review (diagnostic only; not selected)

After selection stuck on baseline, sanity artifacts still record experimental arm boards for inspection (`sanity_2026.json` reflects the selected/baseline path).

Frozen consensus diagnostic (zero fitted weight):

| Player | ECR | ADP |
|---|---:|---:|
| Josh Allen | 25.9 | 33.0 |
| Lamar Jackson | 33.5 | 56.8 |
| Drake Maye | 38.3 | 51.5 |
| Joe Burrow | 46.8 | 57.3 |

Sealed accuracy-first ranks remain Lamar QB18 / Burrow QB20 vs consensus QB2 / QB4 — the miscalibration that motivated this work is real, but history does not yet clear a replacement gate.

## 9. Non-QB invariant

`non_qb_invariance_check` compares every non-QB `(player_id, position, stat)` `pred_pg` before/after each arm.

All five arms: **pass** (`n_compared=3271`, `n_changed=0` after fixing allocation to preserve shipped RB reconcile).

Unit test: `tests/test_qb_projection_final_repair.py::test_non_qb_invariance`.

## 10. Publication / rollback

| Step | Status |
|---|---|
| New immutable candidate namespace | **Not created** (NO-GO) |
| Regenerate simulations / VORP / players JSON | Blocked (no `projections.db` in this environment; gates failed anyway) |
| Promote active pointer | **Not done** |
| Overwrite `v2_baseline_20260830` | **Not done** |

**Rollback if a future candidate is ever staged:** leave `draft_assistant/data/active_release_2026.json` on `v2_baseline_20260830` (manifest `5a8e145…`); delete the candidate namespace directory; do not upload.

**Promotion recommendation:** **Do not promote.** Revisit after (a) DB-backed leakage-safe re-compose folds for allocation, and/or (b) a rate-model fix so mobile designed-run features are not lost before compose.

## 11. Tests added

`tests/test_qb_projection_final_repair.py`:

- leakage boundary
- partial-season weighting
- mobile prior application
- QB1/backup allocation + conservation
- allocation fit uses only prior seasons
- non-QB invariance

## 12. Delivered code map

| Path | Role |
|---|---|
| `src/projection/qb_repair/` | History, allocation, priors, arms, attribution, eval, gates, provenance |
| `scripts/qb_projection_final_repair.py` | End-to-end runner |
| `output/qb_repair/` | Attribution, arm boards, selection decision, provenance |
| `docs/QB_PROJECTION_FINAL_REPAIR_REPORT.md` | This report |
