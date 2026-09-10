# H3 — End-to-End QB Pipeline Experiment

**Date:** 2026-09-03  
**Branch:** `cursor/qb-upstream-feature-allocation-59f0` (draft PR #8)  
**Namespace:** `src/projection/qb_h3/`, `output/qb_h3/`, `scripts/qb_h3_*.py`  
**H1/H2 thresholds:** frozen (not retuned on 2025)  
**Verdict:** **NO-GO**

## Terminology

| Label | Meaning |
|---|---|
| **2025** | Latest chronological OOS fold (informed subsequent decisions; **not** a pristine holdout) |
| **2026** | Prospective holdout (diagnostics only; never used for GO/NO-GO) |
| Comparator | Sealed `fantasy_evaluation_*.csv::model_points_end_to_end` |

No release candidate, pointer move, production-default change, or app integration.

---

## Step 1 — Explain the 2023 +5.11 regression (before trusting H3)

Frozen post-hoc candidate vs sealed e2e on 2023: **mean AE Δ = +5.11** (sum ≈ +215).  
Artifacts: `output/qb_h3/step1_2023_decomposition/`.

### Ten largest negative contributors (candidate worse)

| Player | Actual | Sealed | Frozen cand | AE Δ | Exp starts | Act starts | Archetype (frozen) | Dominant flags |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Jameis Winston | low | low | inflated | **+137** | starter-like | few | pocket | backup→starter rates, conservation |
| Kyler Murray | mid | mid | over | **+116** | high | low | pocket\* | starts, archetype, volume |
| C.J. Beathard | low | low | inflated | **+105** | high | low | insuff. | backup rates, starts |
| Mike White | low | low | inflated | **+95** | high | low | pocket | backup, team/role, starts |
| Ryan Tannehill | mid | mid | over | **+94** | high | mid | pocket | starts, pass/rush volume |
| Jalen Hurts | 367 | 345 | 256 | **+88** | under | 17 | pocket\* | archetype + under-starts |
| Cooper Rush | low | low | inflated | **+75** | — | — | insuff. | backup conservation |
| Kyle Allen | low | low | inflated | **+74** | — | — | pocket | backup, role/team |
| Justin Herbert | mid | mid | over | **+51** | high | mid | pocket | starts, pass volume |
| Sam Darnold | low | low | inflated | **+48** | — | — | pocket | backup conservation |

\*Frozen classifier labeled dual-threats `pocket_passer` when designed/scramble were null (before carries≥5.5). **6** such mislabels; Lamar/Hurts frozen=`pocket_passer`.

### Ten largest positive contributors (candidate better)

Daniel Jones (−120), Kenny Pickett (−114), Gardner Minshew (−102), Joe Burrow (−92), Zach Wilson (−88), plus five additional improvers in `top_contributors.json`.

### How much of the regression comes from each factor?

Shares are of **positive** AE Δ mass (flags overlap; not a partition):

| Factor | Share of positive AE Δ | Finding |
|---|---:|---|
| Passing-volume changes | ~0.89 | Active rates × starts without team reconcile |
| Large divergence from sealed stack | ~0.65 | Never entered feature→reconcile→compose |
| Incorrect expected starts | ~0.59 | Over-predicted starts inflate season totals |
| Backup / starter–backup conservation | ~0.51 | Depth≥2 got starter active rates |
| Availability overprediction | ~0.47 | Same mechanism as starts error |
| Rushing-volume changes | ~0.44 | Under/over rush opportunity vs actual |
| Returning-injury cohort | ~0.29 | |
| Rookie / insufficient history | ~0.21 | |
| Incorrect archetype assignment | ~0.20 | Null designed→pocket bug |
| Role or team change | ~0.20 | |
| Partial-game / efficiency residual | ~0.10 | |
| Touchdown composition | ~0.04 | Secondary residual |
| **Double application of availability** | **~0** | Not the 2023 failure mode (once via expected starts) |
| **Ensemble dilution** | **N/A** | 2023 sealed fold is v1 compose, not accuracy-first blend |

### Pipeline stage values (sealed CSV)

Per player in `players.json::pipeline_stages`:

- **raw** → `model_forecast_points`
- **compose / e2e** → `model_points_end_to_end`
- **reconcile** → embedded in forecast (no separate CSV checkpoint)
- **ensemble** → N/A on 2023 fold
- **frozen candidate** → post-hoc rate×starts only (outside sealed path)

**Do not hide this failure in an aggregate average.** Even after H3 archetype fix, 2023 overall still regresses (below).

---

## Steps 2–3 — Reformulated targets + composition contract

### Targets (experimental)

| Block | Modeled quantities |
|---|---|
| **A. Availability** | Expected starts, expected active games, partial/early-exit rate |
| **B. Passing (active)** | Attempts / completions / pass yards / pass TD / INT **per active start** |
| **C. Rushing (active)** | Carries / rush yards / rush TD per active; designed + scramble priors by archetype |
| **D. Efficiency** | YPA, pass TD rate, INT rate, designed YPC, scramble YPA, rush TD rate |

Active rates are trained only on **active starts** (not scheduled games or injury misses). Missed starts live in availability.

### Composition contract

```
active_start_opportunity × expected_active_starts = expected_season_opportunity
```

Implemented in `src/projection/qb_h3/composition_contract.py`:

- Availability applied **exactly once** (`assert_availability_applied_once`, `detect_double_availability`)
- Starter conditional volume preserved; backups fill residual missed starts
- Starter + backup conserved to team claim
- Fantasy **points per active start** and **expected season points** are separate outputs

Unit proof: `output/qb_h3/composition_contract_proof.json` + `tests/test_qb_h3.py` (6 passed).

H3 archetype fix (control-flow only; **thresholds unchanged**): null designed/scramble no longer forces `pocket_passer` before the carries≥5.5 dual-threat check. Lamar/Hurts @2023 → `mobile_scrambler` under H3 vs `pocket_passer` frozen.

---

## Step 4 — Real sealed pipeline status

**Required path:** feature construction → trained target models → team reconciliation → fantasy-point composition → ensemble → `model_points_end_to_end`.

| Requirement | Status |
|---|---|
| Inject H3 targets into leakage-safe sealed retrain | **Blocked** — `data/projections.db` absent |
| Compare intermediate/post-hoc to sealed final | Forbidden; H3 reports experimental full FP vs sealed e2e points only |
| Ensemble weights | **Unchanged** (primary report) |
| Nested chronological weight selection | **Not run** (would be reported separately; blocked without sealed injection) |
| ECR/ADP for ranking repair | **Not added** |

Honest comparison available here: H3 experimental e2e (cache weekly rates → avail×opp×eff → FP) vs sealed CSV e2e points. That is **not** a sealed-path injection bakeoff.

---

## Step 5 — Historical evaluation vs sealed e2e

Seasons with trustworthy eval CSVs: **2023, 2024, 2025** (no 2022 `fantasy_evaluation_*.csv` in this environment).  
Train / archetype / availability: seasons **&lt; target** only.  
Gates: frozen H1/H2 (`GateThresholds`).  
Δ MAE = H3 − sealed (negative = H3 better).

### Overall

| Season | Label | n | Sealed MAE | H3 MAE | Δ MAE | Spearman sealed→H3 | Double-avail viol. |
|---:|---|---:|---:|---:|---:|---|---:|
| 2023 | fit | 42 | 80.97 | 85.03 | **+4.06** | 0.534→0.562 | 0 |
| 2024 | fit | 43 | 71.75 | 69.26 | −2.49 | 0.749→0.660 | 0 |
| 2025 | latest chronological OOS | 37 | 66.82 | 66.43 | −0.39 | 0.559→0.539 | 0 |

### Primary cohort (dual-threat ∪ returning-injury)

| Season | n | Δ MAE | Bootstrap CI (Δ MAE) |
|---:|---:|---:|---|
| 2023 | 14 | **+9.98** | [−21.1, +41.2] |
| 2024 | 21 | −7.39 | [−26.3, +11.5] |
| 2025 | 20 | −9.62 | [−34.1, **+13.3**] (CI includes 0) |

Primary improves on **1/2** fit folds (need ≥2). Latest-fold primary improves pointwise but bootstrap CI does not exclude zero; Spearman drops beyond 0.02 tol.

### Other cohorts (selected)

| Season | Dual-threat Δ | Pocket Δ | Rookie/insuff. Δ | Returning Δ | Top-12 Δ |
|---:|---:|---:|---:|---:|---:|
| 2023 | **+24.5** | −2.4 | +8.7 | +8.7 | −2.1 |
| 2024 | −7.4 | −5.0 | **+14.1** | −4.0 | **+18.1** |
| 2025 | −15.8 | **+7.8** | +1.6 | −9.4 | −16.5 |

**2024 top-12 regression and 2023 dual-threat / primary regressions show one cohort’s gains do not conceal material losses elsewhere.**

Per-player contributions: `output/qb_h3/fold_{season}_rows.json`.  
Starts / attempts-per-active / carries-per-active MAE reported in fold cohort blocks.  
Team-volume conservation: **unit contract proven**; full-board conservation needs sealed `team_reconcile` (DB).

---

## Step 6 — 2026 diagnostics (after historical eval; not for selection)

Trained through permitted **2025** cutoff (`seasons < 2026`). Artifact: `output/qb_h3/sanity_2026.json`.

| Player | Exp starts | Pts/active | Exp season pts | Avail-adj PPG | Att/active | Season att | Des. car/act | Scrambles/act | Car/act | Season car | Pass TD | Rush TD | Sealed final | H3 final | Diff | Sealed rk | H3 rk | Arch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Josh Allen | 16.3 | 23.2 | 378 | 22.2 | 32.4 | 526 | 5.0 | 2.0 | 7.0 | 114 | 28.8 | 11.4 | 357 | 378 | +20 | 1 | 1 | designed_runner |
| Lamar Jackson | 14.3 | 21.8 | 309 | 18.2 | 27.9 | 396 | 6.0 | 2.1 | 8.1 | 115 | 25.6 | 4.2 | 284 | 309 | +25 | 18 | 4 | designed_runner |
| Jayden Daniels | 12.4 | 20.2 | 244 | 14.3 | 29.8 | 360 | 6.2 | 2.4 | 8.6 | 104 | 17.3 | 5.1 | 294 | 244 | −50 | 14 | 14 | designed_runner |
| Jalen Hurts | 15.4 | 21.9 | 335 | 19.7 | 29.7 | 454 | 7.7 | 1.4 | 9.1 | 138 | 21.7 | 11.7 | 309 | 335 | +26 | 8 | 2 | designed_runner |
| Joe Burrow | 13.1 | 20.0 | 260 | 15.3 | 37.3 | 484 | 1.4 | 1.8 | 3.2 | 42 | 28.0 | 1.8 | 281 | 260 | −21 | 20 | 10 | pocket_passer |
| Patrick Mahomes | 15.7 | 20.0 | 312 | 18.3 | 37.5 | 585 | 2.0 | 2.1 | 4.1 | 65 | 28.7 | 2.7 | 333 | 312 | −21 | 4 | 3 | mobile_scrambler |

Full starter table: `output/qb_h3/sanity_2026.json`.  
**QB12 boundary (experimental):** Drake Maye ≈ 254.6 season pts (experimental rank 12; sealed rank 2).

---

## Step 7 — Decision

# **NO-GO**

### Failing stage

1. **`sealed_leakage_safe_refit_blocked_without_projections_db`** — cannot run true feature→train→reconcile→compose→ensemble injection.
2. **Experimental H3 e2e vs sealed e2e points** still fails frozen gates.

### Failing cohorts / gates

- 2023 overall MAE Δ **+4.06** (material single-season regression; not hidden in average)
- Primary cohort improved on **1/2** fit folds (need 2)
- Latest OOS (2025) primary bootstrap CI **includes zero**
- Latest OOS Spearman drop beyond tol
- Full-board team-volume conservation **not** proven on sealed path
- Zero non-QB changes: satisfied for this experiment (no production board rewrite)

### Explicitly **not** started

- H4
- Release packaging / candidate bundle
- Production default or pointer changes
- Application integration

### Reproduce

```bash
python scripts/qb_h3_step1_2023_decomposition.py
python -m pytest tests/test_qb_h3.py -q
python scripts/qb_h3_end_to_end_eval.py   # exit 2 = NO-GO
```

---

## Production safety

| Item | Status |
|---|---|
| `v2_baseline_20260830` | Untouched |
| Active release pointer | Untouched |
| H1/H2 frozen thresholds | Untouched |
| PR #8 | Remains **draft** |
| Release candidate | **Not created** |
