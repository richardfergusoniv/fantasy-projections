# H3 evaluation infrastructure repair

**Date:** 2026-09-03  
**Branch:** `cursor/qb-upstream-feature-allocation-59f0` (draft PR #8)  
**H3 model specification:** frozen (thresholds, pooling weights, availability coefficients, gates, ensemble weights, cohort definitions unchanged)  
**Verdict:** **NO-GO FOR H3**

2025 is the latest chronological OOS fold (not a pristine holdout). 2026 is the prospective holdout (diagnostics only).

No release candidate, pointer move, production-default change, or application work. H4 was **not** started.

---

## Why the prior H3 run was not a definitive architectural rejection

1. Full sealed-path team reconciliation could not run.
2. The expected-start allocator elevated backups (Winston, Beathard, White, Rush, Kyle Allen, Darnold).
3. Null designed-run history mapped six QBs to `pocket_passer`.
4. `data/projections.db` is a zero-byte / missing Cloud placeholder.
5. Full-board conservation was therefore not evaluated.

This pass repairs that **evaluation infrastructure** and reruns the same frozen H3 architecture.

---

## Task 1 — Portable reconciliation contract

Minimum inputs used (public-derived, already in the workspace; **not** the complete DB or `player_week_panel.parquet`):

| Source | Role |
|---|---|
| `data/raw/weekly_qb_repair_cache/qb_weekly.parquet` | Prior-season team pass attempts and QB carries |
| `output/qb_active_archetype/active_season_rates.parquet` + weekly | Prior starts / active rates |
| `output/fantasy_evaluation_{2023,2024,2025}.csv` | Preseason team + depth-chart role at cutoff; labels only |

Builder: `scripts/qb_h3_build_portable_fixture.py`  
Fixture: `output/qb_h3/infra/portable_qb_reconcile_fixture.parquet` (312 rows)  
Manifest: `output/qb_h3/infra/portable_qb_reconcile_manifest.json`

- Schema version: `qb_h3_reconcile_contract_v1`
- Prediction-side columns use only seasons **&lt; prediction_season**
- Actual starts / outcomes are `actual_*` / sealed label columns
- Leakage audit: **ok**
- Deterministic content hash recorded in the manifest
- Committed as a versioned evaluation fixture (small; no secrets, no Sleeper league data, no DB)

**Fail-fast:** `src/projection/qb_h3/projections_db.py` raises `ProjectionsDbUnusable` on a missing or zero-byte DB. The evaluator prints that error and **refuses to skip reconciliation** — it requires the portable fixture. This run:

```
ERROR: data/projections.db is missing or zero bytes (placeholder). ...
reconciliation_source = portable_fixture
reconciliation_ran = true
conservation_violations = []
```

---

## Task 2 — Archetype missingness (thresholds unchanged)

Pocket passer now requires **observed** low designed **and** low scramble.

- Null / insufficient designed-run history → `insufficient_history` (or dual-threat via carries ≥ 5.5)
- Never pocket on null designed
- Lamar / Hurts @2023 classified from prior carries → `mobile_scrambler` (not pocket)

Six previously misclassified 2023 rows (regression-tested): Allen, Hurts, Lamar, Fields, Murray, Watson — none are pocket.

---

## Task 3 — Expected-start and role allocation

Frozen `expected_availability` coefficients are unchanged. A **role layer** then applies preseason depth:

- Productivity while active does **not** imply starter status
- Backups cannot inherit starter volume from a strong per-start rate
- QB1 starts come from the frozen availability estimate
- Backup starts = residual scheduled games + explicit QB1 partial-game exposure
- Room expected starts conserve **17** scheduled games
- Destination-team preseason role is taken at the cutoff
- Rookie / package roles use distinct residual weights (not the starter league prior)

2023 targeted checks (all passing):

| Player | Role | Allocated starts | Notes |
|---|---|---:|---|
| Winston | backup | 0.61 | was starter-like before |
| Beathard | backup | 0.30 | |
| White | backup | 3.97 | |
| Rush | backup | 3.49 | |
| Kyle Allen | backup | 0.74 | |
| Darnold | backup | 4.69 | |
| Murray | package (depth 5 behind Dobbs) | 1.71 | no longer ~14 starter starts |
| Tannehill | starter | 14.12 | frozen avail; backups did not cut his rate |
| Herbert | starter | 16.15 | |
| Hurts | starter | 12.29 | Mariota cannot reduce his active rate |

---

## Task 4 — Frozen H3 sequence (this environment)

```
features (weekly active rates, seasons < T)
→ active-start models (frozen pooling)
→ expected starts (frozen availability coeffs)
→ starter/backup allocation (new role layer)
→ team reconciliation (exact QB pass + QB-rush conservation on portable fixture)
→ composition (half-PPR)
→ sealed ensemble weights unchanged (identity passthrough)
→ experimental final points vs sealed model_points_end_to_end
```

- Team passing and QB-rushing conservation: **exact** (0 violations)
- Availability applied once: **0** double-avail violations
- Reconciliation: **not skipped**
- Future role leakage: blocked by fixture audit + OOS asserts
- Non-QB projection changes: **0**
- Ensemble weights: unchanged; no nested reweight; no ECR/ADP

`projections.db` was not opened. This is portable-fixture reconciliation, not a DB-backed sealed-model retrain.

---

## Task 5 — Corrected verdict vs sealed `model_points_end_to_end`

Δ MAE = repaired H3 − sealed (negative = H3 better).

| Season | Label | n | Sealed MAE | H3 MAE | Δ MAE | Spearman sealed→H3 |
|---:|---|---:|---:|---:|---:|---|
| 2023 | fit | 42 | 80.97 | 72.96 | **−8.00** | 0.534→0.653 |
| 2024 | fit | 43 | 71.75 | 60.07 | **−11.67** | 0.749→0.772 |
| 2025 | latest chronological OOS | 37 | 66.82 | 67.88 | **+1.06** | 0.559→0.548 |

2025 +1.06 is inside the frozen +2% overall non-inferiority band (tol ≈ 1.34).

### Primary cohort (dual-threat ∪ returning-injury)

| Season | n | Δ MAE | Bootstrap 95% CI |
|---:|---:|---:|---|
| 2023 | 14 | −9.95 | [−32.0, +10.8] |
| 2024 | 21 | −11.92 | [−32.1, +8.2] |
| 2025 | 20 | −3.45 | [−22.6, **+16.6**] |

Primary improves on **both** fit folds and pointwise on 2025. The 2025 paired CI **includes zero**.

### Other cohorts (do not hide inside the average)

| Season | Dual-threat | Pocket | Insuff. hist. | Returning | Top-12 | Depth-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | +7.31 | — (null designed → insuff.) | −10.56 | −15.14 | −5.81 | −11.96 |
| 2024 | −5.02 | −6.45 | −19.80 | −13.09 | **+14.65** | −7.77 |
| 2025 | −9.43 | −8.15 | **+19.23** | −4.07 | −15.60 | +3.27 |

2024 top-12 and 2025 insufficient-history remain material cohort regressions.

### How infrastructure repair changed the prior 2023 +4.06

| | Overall Δ MAE vs sealed |
|---|---:|
| Prior H3 (no role / no reconcile) | **+4.06** |
| Repaired H3 | **−8.00** |
| Infra change | **−12.06** |

Largest 2023 AE improvements from the allocator + reconcile (prior AE Δ → repaired AE Δ):

| Player | Prior H3 pts | Repaired | Actual | Prior AE Δ | Repaired AE Δ |
|---|---:|---:|---:|---:|---:|
| Jameis Winston | 150 | 21 | 12 | +135 | +6 |
| C.J. Beathard | 134 | 36 | 21 | +93 | −5 |
| Kyler Murray | 291 | 79 | 144 | +122 | +40 |
| Kyle Allen | 106 | 34 | −1 | +71 | −1 |
| Mike White | 127 | 58 | 4 | +93 | +24 |
| Sam Darnold | 152 | 91 | 25 | +47 | −14 |
| Cooper Rush | 107 | 47 | 3 | +64 | +4 |

Starter-side residuals that infrastructure does **not** fix (frozen availability): Tannehill (+87), Herbert (+53), Hurts still under-started vs 17 games (+66 vs sealed). Taysom Hill / Zach Wilson worsened when residual starts were taken away (package/backup).

Full player table: `output/qb_h3/infra/step5_2023_infra_delta.json`.

---

## Frozen gates

| Gate | Result |
|---|---|
| No material single-season regression like 2023 +5.11 | pass (2023 now −8.00) |
| Overall non-inferiority (≤ +2%) | pass |
| Primary improves on ≥2 fit folds | pass |
| Latest-OOS primary pointwise improve | pass |
| Latest-OOS primary bootstrap CI upper &lt; 0 | **FAIL** ([−22.6, +16.6]) |
| Latest-OOS Spearman drop ≤ 0.02 | pass |
| Latest-OOS top-12 non-inferiority | pass |
| Team-volume conservation | pass (exact) |
| Availability once | pass |
| Zero non-QB changes | pass |

# **NO-GO FOR H3**

### Failing stage (legitimate model result after infra repair)

**Latest chronological OOS (2025) primary-cohort paired uncertainty.** Point estimate improves (−3.45 MAE) but the frozen gate requires the bootstrap CI to exclude zero on the helpful side.

### Failing / concerning cohorts

- 2025 **insufficient_history** Δ MAE **+19.23** (rookies / missing designed splits)
- 2025 depth-1 +3.27
- 2024 top-12 +14.65 (fit fold; not a latest-OOS gate, but material)

### H4 hypothesis (identified only — **not started**)

A distinct leakage-safe prior for **insufficient-history / rookie** volume and starts, plus designed-split coverage before 2023 so pocket vs insufficient is identified from observed rush features rather than missingness. That is a new architecture (H4), not a threshold retune of frozen H3.

---

## 2026 diagnostics (not used for selection)

See `output/qb_h3/sanity_2026.json` (Allen, Lamar, Daniels, Hurts, Burrow, Mahomes, QB12). Prospective holdout only.

---

## Reproduce

```bash
python scripts/qb_h3_build_portable_fixture.py
python -m pytest tests/test_qb_h3.py tests/test_qb_h3_infra.py -q
python scripts/qb_h3_end_to_end_eval.py   # exit 2 = NO-GO FOR H3
```

## Production safety

| Item | Status |
|---|---|
| `v2_baseline_20260830` | Untouched |
| Active pointer `5a8e1453…` | Untouched |
| Frozen H1/H2 thresholds | Untouched |
| PR #8 | Remains **draft** |
| Release / pointer / app | Not touched |
