# QB Upstream Feature Construction + Joint Allocation Report

**Date:** 2026-09-03  
**Branch:** `cursor/qb-upstream-feature-allocation-59f0`  
**Sealed baseline (unchanged):** `v2_baseline_20260830`  
**Active pointer (unchanged):** `draft_assistant/data/active_release_2026.json`  
**Verdict:** **NO-GO**

## Executive decision

Upstream repairs (expanded QB rush representation, games-weighted multi-season pooling, joint QB-room allocation) are implemented behind defaults that leave production compose bit-identical. Chronological evaluation does **not** clear promotion gates:

- Rate pooling improves carries/g MAE on fit + holdout point estimates, but the 2025 holdout bootstrap 95% CI still includes zero.
- Patching the frozen `QB_carries` inference vector with pooled priors **worsens** fold MAE (bias shrinks, absolute error rises).
- Full season-points / PPG / rank / top-12 gates cannot be claimed without historical raw-board retrain (blocked here without `projections.db`).
- 2026 sanity (Burrow attempts ↑, Lamar reconstructed patched carries ↑) is diagnostic only and was **not** used for selection.

Therefore: no candidate namespace, no sealed overwrite, no active-pointer move.

## Reproduce

```bash
uv sync --frozen --extra jobs --extra dev

# Focused regression + production-invariance tests
uv run pytest tests/test_qb_upstream_features_allocation.py \
  tests/test_composition_unification.py -q
# 14 passed

# Rolling-origin + lineage + 2026 joint allocation eval
uv run python scripts/qb_upstream_feature_allocation_eval.py
# verdict NO-GO
```

### Artifact paths

| Artifact | Path |
|---|---|
| Selection decision | `output/qb_upstream/selection_decision.json` |
| Lamar stage lineage | `output/qb_upstream/lamar_feature_lineage.json` |
| Rate-pooling folds | `output/qb_upstream/rate_pooling_folds.json` |
| Model-patch folds | `output/qb_upstream/model_feature_patch_folds.json` |
| Joint allocation 2026 | `output/qb_upstream/joint_allocation_2026.json` |
| This report | `docs/QB_UPSTREAM_FEATURE_ALLOCATION_REPORT.md` |

### Integrity (production untouched)

| Artifact | SHA-256 |
|---|---|
| Active pointer | `7499f11a51a787703cc47e32b311fd8e17f9d45c119df86f4c31ca8749efd6e9` |
| Sealed manifest `v2_baseline_20260830` | `5a8e14536aa7b062b1e5ff6e64aa78356847fb063579d0a42cdbdc5cc159fbb1` |

## 1. Lamar Jackson feature lineage (stage-by-stage)

Source: `output/qb_upstream/lamar_feature_lineage.json` (player_id `00-0034796`).

### 1 — Raw historical source (2022–2025)

| Season | Games | Carries | Carries/g | Designed | Scramble | Designed run rate |
|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 12 | 112 | 9.33 | 86 | 26 | 0.196 |
| 2023 | 16 | 148 | 9.25 | 91 | 66 | 0.136 |
| 2024 | 17 | 139 | 8.18 | 107 | 46 | 0.153 |
| 2025 | 13 | 67 | 5.15 | 37 | 30 | 0.091 |

Games-weighted carries/g (2022–2025): **8.03**.

### 2 — Season feature construction

- `qb_designed_run_rate` is computed (`designed_rush_attempts / team_plays_active`) and listed in `FEATURE_COLS` and `BLEND_FEATURES`.
- **QB rows are excluded from blend** (`features.py`: `blendable = position in {RB,WR,TE}`). Evidence comment cites pass-volume ablation; consequence: a depressed 2025 designed-run rate is **not** pooled with 2022–2024 in the legacy column the sealed model sees.
- Additive expansion (`qb_scramble_per_dropback`, YPC splits, RZ/GL designed rates, `*_pooled`, archetype priors) is attached for experimental use and is **not** in the sealed contract.

### 3 — Transition-pair / training row

`QB_carries.joblib` predicts label `carries_per_elig` from role features including `qb_designed_run_rate` and `prior_role_rate_3y`.

### 4 — Serialized model feature contract

| Check | Result |
|---|---|
| `qb_designed_run_rate` in sealed features | **True** (not missing at runtime) |
| Gain rank | **21** (gain ≈ 42) |
| `prior_role_rate_3y` gain | ≈ 4349 (rank 1) |
| `prior_rushing_yards_pg` / `prior_carries_pg` | gain ≫ designed-run |

**Verdict:** present but low-gain / overwhelmed by prior role and prior rush features; QB blend exclusion leaves injured-season designed rate unpooled.

### 5 — 2026 inference row

| Quantity | Value |
|---|---:|
| Unpatched `prior_carries_pg` (from 2025) | 5.15 |
| Games-weighted 2022–2025 pool | 8.03 |
| Patched `prior_carries_pg` (experimental) | 8.03 |

### 6 — Raw model prediction (sealed 2026 raw board)

Carries **4.42**/g, rush yards **26.7**/g. Reconstructing with unpatched vs patched inference: **3.30 → 7.27** carries/g (frozen model weights; vector-only patch).

### 7 — Team-volume reconciliation

Rush cells keep `team_volume_scale = 1.0`. Reconcile does not restore rushing.

### 8 — Composition

Sealed compose leaves carries ≈ 4.5 through finalization.

### 9 — Ensemble output

Accuracy-first board: ~16.7 PPG / ~4.51 car/g (ensemble lifts points via v2 season blend; component rush rates stay weak).

## 2. Historical fold results

### Candidate A — games-weighted multi-season carries/g pool vs T−1 carry-forward

| Season | n | CF MAE | Pooled MAE | Δ MAE | Bootstrap Δ 95% CI | Spearman CF | Spearman pooled |
|---:|---:|---:|---:|---:|---|---:|---:|
| 2023 (fit) | 32 | 0.957 | 0.870 | −0.087 | [−0.282, 0.110] | 0.728 | 0.776 |
| 2024 (fit) | 34 | 0.885 | 0.882 | −0.003 | [−0.173, 0.173] | 0.743 | 0.733 |
| 2025 (hold) | 32 | 1.060 | 0.968 | −0.092 | [−0.300, 0.092] | 0.819 | 0.845 |

Point estimates improve; holdout bootstrap CI **includes zero** → fail.

### Candidate B — sealed `QB_carries` + pooled inference patch

| Season | n | Baseline MAE | Patched MAE | Δ MAE | Bootstrap Δ 95% CI | Mean \|Δ pred\| |
|---:|---:|---:|---:|---:|---|---:|
| 2023 | 32 | 0.949 | 0.951 | +0.002 | [−0.096, 0.093] | 0.169 |
| 2024 | 36 | 0.924 | 0.994 | +0.070 | [−0.046, 0.187] | 0.316 |
| 2025 | 32 | 0.771 | 0.812 | +0.041 | [−0.115, 0.220] | 0.241 |

Patch moves predictions (not a no-op) but **does not improve** aggregate MAE → fail.

### Candidate C — joint QB-room allocation (2026 compose flag)

| Check | Result |
|---|---|
| Non-QB `pred_pg` invariance | **pass** (3271 compared, 0 changed) |
| Burrow attempts (baseline → joint) | 17.97 → **23.96** |
| Team attempts conserved | enforced in `reconcile_qb_joint_room` |
| Default `qb_joint_room_allocation` | **False** (shipped path unchanged) |

### Gates checklist (supplement; none weakened)

| Gate | Status |
|---|---|
| Season points MAE | unavailable (no historical raw retrain) |
| PPG MAE | unavailable |
| Rank correlation | unavailable |
| Top-12 recall / ordering | unavailable |
| Rushing-attempt calibration (rate pool proxy) | pointwise OK; bootstrap CI fail |
| Passing-volume conservation | pass (joint path) |
| Stability across seasons | fail (model patch regresses) |
| Non-QB projection invariance | pass |

## 3. 2026 sanity table (diagnostic only — not used for selection)

Joint-allocation compose on `projections_2026_raw.csv` (experimental flag on):

| Player | Rank | PPG | Attempts | Carries |
|---|---:|---:|---:|---:|
| Josh Allen | 1 | 18.26 | 23.96 | 6.09 |
| Patrick Mahomes | 9 | 14.75 | — | — |
| Jalen Hurts | 10 | 14.73 | 22.33 | 4.31 |
| Lamar Jackson | 16 | 13.25 | 22.08 | 4.42 |
| Jayden Daniels | 21 | 12.93 | 23.68 | 6.44 |
| Joe Burrow | 25 | 11.93 | 23.96 | 1.89 |

Lamar carries remain ~4.42 under joint allocation alone (rush is not a team-pass-volume cell). Raising Lamar requires the rush-feature / model path, which failed chronological gates.

## 4. Architecture delivered (defaults off)

| Module | Role |
|---|---|
| `src/projection/qb_rush_features.py` | Designed/scramble/YPC/RZ/GL splits; multi-season pooled + archetype priors; inference patch helper |
| `src/projection/data_prep.py` | `player_season_qb_rush_splits()` |
| `src/projection/features.py` | Additive expansion + pooling; legacy `qb_designed_run_rate` unchanged |
| `src/projection/qb_joint_allocation.py` | Starter-first / backup-residual room; conserves team attempts & games |
| `src/projection/composition.py` | `CompositionContext.qb_joint_room_allocation=False` by default |
| `scripts/qb_upstream_feature_allocation_eval.py` | Lineage + folds + bootstrap + decision |
| `tests/test_qb_upstream_features_allocation.py` | Unit + production-flag invariance |

## 5. Selected configuration

**None (NO-GO).**

Do not publish a candidate release namespace. Do not modify `v2_baseline_20260830` or `active_release_2026.json`.
