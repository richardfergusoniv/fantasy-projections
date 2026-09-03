# QB Active-Start × Archetype Experiment Report

**Date:** 2026-09-03  
**Branch:** `cursor/qb-upstream-feature-allocation-59f0` (draft PR #8)  
**Sealed baseline (unchanged):** `v2_baseline_20260830`  
**Active pointer (unchanged)**  
**Experiment verdict:** **GO** (vs conflated-rate baseline; predeclared gates)  
**Production promotion:** **NO**

## Decision

Chronological rolling-origin evaluation against a **conflated per-game carry-forward baseline** clears the predeclared non-inferiority and cohort gates. The selected experimental configuration is:

`active_start_rates + archetype_conditional_rush_priors + joint_qb_room_v2`

This does **not** authorize sealing, pointer movement, or production publish. Existing compose defaults remain off; RB/WR/TE outputs are byte-identical under the candidate board path (`n_changed=0`).

2026 diagnostics are reported separately and were **not** used for selection.

## Predeclared thresholds (frozen before results)

See `output/qb_active_archetype/predeclared_thresholds.json`.

| Gate | Threshold |
|---|---|
| Overall points MAE non-inferiority | ≤ +2% vs baseline |
| Primary cohort (dual-threat ∪ returning-injury) fit folds improved | ≥ 2 |
| Holdout primary cohort MAE | must improve |
| Holdout primary paired bootstrap 95% CI | upper bound < 0 |
| Holdout top-12 MAE non-inferiority | ≤ +3% |
| Holdout Spearman drop | ≤ 0.02 |
| Use 2026 for selection | **false** |

## Reproduce

```bash
uv sync --frozen --extra jobs --extra dev
uv run pytest tests/test_qb_active_archetype.py tests/test_composition_unification.py -q
uv run python scripts/qb_active_archetype_eval.py
```

Artifacts: `output/qb_active_archetype/`

## H1 — Active-game rate vs availability

Missed games must not dilute per-active-start opportunity.

### Burrow decomposition

| Season | Active starts | Att / active | Att / conflated game | Note |
|---:|---:|---:|---:|---|
| 2022 | 16 | 37.9 | 37.9 | Full |
| 2023 | 10 | 36.5 | 36.5 | Short — rate intact |
| 2024 | 17 | 38.4 | 38.4 | Full |
| 2025 | 8 | ~34.8 | ~34.8 | Short — rate intact |

Forward expected active starts into 2026 (shrunk empirical): **~13.06**.  
Pooled active attempt rate: **~37.3**.

**Diagnosis:** Low sealed Burrow output was **not** a collapsed active-start attempt rate. Short 2023/2025 seasons cut availability; team-volume backup floors then squeezed the conflated board. Candidate separates rate × expected starts, then allocates backups as residual only.

### Lamar decomposition

| Season | Active starts | Carries / active | Designed / active | Scrambles / active |
|---:|---:|---:|---:|---:|
| 2022 | 12 | 9.33 | ~7.2 | ~2.2 |
| 2023 | 16 | 9.25 | 5.69 | 4.13 |
| 2024 | 17 | 8.18 | 6.29 | 2.71 |
| 2025 | 13 | 5.15 | ~2.85 | ~2.31 |

2025 depresses rush rates; games-weighted archetype prior still labels him **designed_runner** and pools toward ~8.1 carries / active.

## H2 — Archetype-conditional rush priors

Classification uses only seasons `< target` (no names, no future labels):

- designed_runner / mobile_scrambler / pocket_passer / insufficient_history

Hierarchical priors shrink player multi-season active rates toward **same-archetype** peer means. Pocket passers do not inherit dual-threat carries.

Lamar @ 2026: `designed_runner`; designed carries / active prior ≈ **6.04**; total carries / active ≈ **8.14**.

## Joint QB-room v2

- Team pass claim from anchors first  
- Starter season volume = `max(active_rate × expected_starts, historical starter share)`  
- Backups = residual only; conserve by cutting backups first (never uniform-scale the starter)  
- No player-specific overrides  

## Historical fold results

| Season | Role | n | Overall Δ MAE | Primary Δ MAE | Primary bootstrap CI | Top-12 Δ MAE | Spearman base→cand |
|---:|---|---:|---:|---:|---|---:|---|
| 2023 | fit | 42 | −5.07 | −19.45 | (fit; not binding) | −27.17 | 0.47→0.56 |
| 2024 | fit | 43 | −4.07 | −8.58 | (fit) | −7.05 | 0.65→0.65 |
| 2025 | holdout | 37 | −13.59 | −23.47 | **[−47.0, −0.76]** | −20.03 | 0.51→0.53 |

Holdout dual-threat Δ MAE ≈ −19.6; returning-injury ≈ −30.7; pocket ≈ −11.3.

All predeclared gates: **pass**.

## 2026 diagnostic table (not used for selection)

| Player | Baseline rank / PPG / att / car | Candidate rank / PPG / att / car |
|---|---|---|
| Josh Allen | 3 / 16.36 / ~27.3 / 6.09 | 1 / 19.60 / 31.10 / 7.00 |
| Lamar Jackson | 20 / 11.09 / 22.3 / **4.42** | 5 / 17.15 / 26.38 / **8.14** |
| Jayden Daniels | 16 / 11.55 | 3 / 17.85 / 27.47 / 8.59 |
| Jalen Hurts | 12 / 12.73 | 2 / 19.05 / 27.59 / 9.06 |
| Joe Burrow | 29 / 9.37 / **17.97** | 21 / 13.62 / **30.07** / 3.21 |
| Patrick Mahomes | 15 / 12.35 | 12 / 15.04 / 33.09 / 4.15 |
| QB12 (cand) | — | Mahomes @ 12 |

### Lamar rush stages (candidate path)

| Stage | Carries | Rush yards | Rush TDs | Designed / active prior | Scrambles / DB |
|---|---:|---:|---:|---:|---:|
| Raw sealed model | 4.42 | 26.7 | 0.11 | — | — |
| Baseline composed | 4.42 | 26.7 | 0.11 | — | — |
| Archetype prior | 8.14 | 48.2 | 0.30 | 6.04 | 0.135 |
| Candidate composed | 8.14 | 48.2 | 0.30 | — | — |

### Burrow volume attribution (candidate)

- Active-start attempt rate ≈ **37.3** (healthy)  
- Expected active starts ≈ **13.1** (availability, not rate)  
- Board attempts after joint v2 ≈ **30.1** (= season claim / 17 exposure)  
- Residual low fantasy rank still reflects TD rate / supporting cast vs peers — not a 18-attempt squeeze  

## Non-QB invariance

`pass: true`, `n_compared: 3271`, `n_changed: 0`, `max_abs_delta: 0.0`

## What was not done

- No sealed release overwrite  
- No active pointer change  
- No production candidate namespace publish  
- PR #8 remains draft (CI green ≠ ready-to-merge for promotion)

## Next step if productionizing later

A full historical raw-board retrain under this architecture (with team anchors regenerated) before any seal. This report does **not** start that work.
