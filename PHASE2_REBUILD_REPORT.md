# Phase 2 Rebuild Report: Pooled Multi-Season OL Attribution

This rebuild implements the recommendation from `PHASE2_STABILITY_INVESTIGATION.md`: replace the five independent per-season ridge fits (`ol_coefficients`, `src/ol_model/pipeline.py`/`fit.py`) with one pooled ridge regression per sub-model across all 2021-2025 plays, with season fixed effects instead of season-specific player coefficients, a higher fixed ridge alpha for stability, and an explicit lineup-churn confidence flag. `PHASE2_REPORT.md` and `PHASE2_STABILITY_INVESTIGATION.md` are unchanged - this is additive. The old per-season pipeline and `ol_coefficients` table still exist for comparison but are **not** the path Phase 4 should read from; use `ol_coefficients_pooled`, `ol_season_effects_pooled`, and `ol_team_season_churn` instead.

## What changed

- **Pooling**: one ridge fit per sub-model (pass protection, run blocking) across all 2021-2025 plays instead of 5 independent per-season fits. Each lineman gets ONE coefficient (their overall contribution) instead of up to 5 independent noisy per-season estimates.
- **Season fixed effects**: a one-hot season dummy (2021-2025) replaces the intercept, letting the model express real year-to-year shifts (scheme, aging, roster context) separately from the player-level baseline, without forcing one player's coefficient to be identical across seasons or splitting their signal into 5 noisy pieces.
- **Higher fixed alpha**: RidgeCV's cross-validated alpha optimizes predictive fit, not coefficient stability. The final fit uses alpha=**31623** for pass protection and alpha=**31623** for run blocking - both exactly 10x the RidgeCV pick on the pooled data (see Alpha section below).
- **Lineup-churn confidence flag**: a new `ol_team_season_churn` table plus a per-player `confidence_flag` carried onto `ol_coefficients_pooled` (see below).

## Sample sizes

| Sub-model | Pooled N (all 5 seasons) | RidgeCV alpha (predictive-fit optimum) | Alpha used (10x) |
|---|---|---|---|
| pass_protection | 99881 | 3162.3 | 31622.8 |
| run_blocking | 68991 | 3162.3 | 31622.8 |

## Alpha choice

Per `PHASE2_STABILITY_INVESTIGATION.md`'s alpha-sensitivity finding (split-half stability improves up to ~10x the RidgeCV-selected alpha and flattens beyond that), the final fit for each sub-model uses `RidgeCV`'s pooled-data pick times `ALPHA_STABILITY_MULT = 10` (see `src/ol_model/pooled_fit.py`). This was re-checked directly on the pooled data before implementation (games-split split-half, 5 seasons pooled) rather than assumed:

| Sub-model | 1x (CV) | 5x | 10x (used) | 30x | 100x |
|---|---|---|---|---|---|
| pass_protection | 0.463 | 0.553 | 0.569 | 0.582 | 0.590 |
| run_blocking | 0.394 | 0.501 | 0.517 | 0.524 | 0.522 |

Both curves flatten well before 100x, and run_blocking's gain from 30x->100x is essentially zero (even a slight dip). 10x captures most of the available gain without over-shrinking every coefficient toward zero.

## Stability results: pooled vs. old per-season fits

| Sub-model | Old split-half (single-season, PHASE2_STABILITY_INVESTIGATION.md) | New pooled split-half |
|---|---|---|
| pass_protection | 0.330 | 0.589 |
| run_blocking | 0.291 | 0.501 |

Old year-over-year coefficient correlation (5 independent per-season fits): **0.144** average (range 0.025-0.218). Old single-season split-half reliability ceiling: **0.310**. New pooled-fit split-half reliability: **0.545** average.

**Honest read**: pooling is a real, meaningful improvement - split-half reliability roughly doubled versus the old single-season ceiling (0.31 -> 0.55), which is the correct comparison since year-over-year correlation in the old model was always going to sit below its own same-season noise floor. The pooled fit no longer throws away 4/5 of a multi-season player's data when estimating their coefficient, and pooling plus the higher alpha both push in the same direction. It is still not a 'stable individual trait' correlation in the >0.6-0.7 sense noted in the investigation - the structural identifiability ceiling from low-churn team-seasons (see below) is architectural, not a sample-size problem, and pooling seasons does not fix it: a player who was part of a fixed 5-man line for most of a season is exactly as collinear with his 4 linemates in the pooled fit as in the per-season one, for the plays in that block. What pooling and the higher alpha fix is the *estimation-noise* share of the instability (cause #1 in the investigation, the majority contributor); they do not and cannot fix the *identifiability* share (cause #3).

## Lineup-churn confidence flag

`ol_team_season_churn` has one row per (season, team), 160 rows total across 2021-2025 (32 teams x 5 seasons, minus any team-seasons with no resolved 5-man-line plays). Each row has `n_plays`, `n_distinct_lineups`, `top_lineup_frac` (share of that team's snaps run by its single most common 5-man combination), and `confidence_flag` = `unit_level` if `top_lineup_frac >= 0.90` else `individual` - same 90% threshold the investigation used when it found only 2/32 teams cleared it in 2023.

- **4 / 160** team-seasons (2.5%) are flagged `unit_level` (low churn - a fixed starting five with little rotation, individual credit within that block is not statistically identified).
- **156 / 160** are `individual` (enough rotation/injury churn for the model to actually separate players' coefficients).

This is carried onto `ol_coefficients_pooled` as a per-player `confidence_flag`: a player is flagged `unit_level` if they logged >=50 plays for ANY team-season that itself is `unit_level` (their coefficient in that window is partly an arbitrary split of a shared unit effect, and since the pooled model gives them one coefficient across all their plays, that unit-level noise contaminates their overall number, not just one season of it). `worst_top_lineup_frac` and `n_team_seasons` are also carried for anyone who wants finer-grained judgment than the binary flag.

Player-level result: **40 / 593** player-submodel rows (20 distinct players) are flagged `unit_level`. Given only ~2-6 teams per season clear the 90% threshold, most linemen who spent a meaningful stretch on one team end up touched by at least one low-churn team-season somewhere in 2021-2025 - this flag should be read as 'treat with more caution', not as a rare edge case.

## Schema

- **`ol_coefficients_pooled`**: gsis_id, coef, submodel, display_name, position, worst_top_lineup_frac, confidence_flag, n_team_seasons. One row per (gsis_id, submodel) - no `season` column, since the player-level coefficient is now pooled across all seasons they appeared in. `worst_top_lineup_frac`/`n_team_seasons`/`confidence_flag` are NaN/`individual` for players who never crossed the 50-play relevance threshold for any team-season (rare - means very few resolved plays overall).
- **`ol_season_effects_pooled`**: season, coef, submodel. The season fixed-effect term - add a player's `coef` from `ol_coefficients_pooled` to the relevant season's `coef` here to get that player-season's fitted baseline (same additive structure as the design matrix used for fitting).
- **`ol_team_season_churn`**: season, team, n_plays, n_distinct_lineups, top_lineup_frac, confidence_flag. Team-season-level churn/identifiability metric, independent of sub-model (describes who was on the field, not the outcome model).

## Caveats and judgment calls

- **Computational cost**: pooling means one design matrix per sub-model across ~100k (pass-pro) / ~69k (run-block) plays and ~580-590 lineman columns each - RidgeCV (cv=5) plus the final fixed-alpha fit ran in well under a minute per sub-model on this machine. Not a practical concern at this data volume, but would need revisiting if the window grows to many more seasons.
- **Players seen in only one season** are not obviously worse off under pooling - they still get all their plays' worth of signal, same as before, just now sharing a design matrix (and its season dummies) with everyone else rather than a season-specific one. No evidence found that single-season players got *harder* to distinguish from noise; the season fixed effects absorb the year-level shift so a one-season player's coefficient is still estimated from just their own plays relative to that season's baseline, structurally similar to before.
- **Season fixed effects vs. player x season interactions**: this rebuild does NOT fit player x season interaction terms (that would reintroduce the original per-season noise problem for any player without a lot of snaps in every season). A player's true year-over-year change (aging, injury, scheme fit) is not recoverable from this model - it is deliberately smoothed into a single across-season coefficient. If Phase 4 needs within-player trajectory, that's a different, harder model, not a small extension of this one.
- **The 50-play relevance threshold and 90% churn threshold** are both carried over from the investigation's already-validated choices, not re-derived here; they are reasonable but arbitrary round numbers, not fit to any objective function.
- **Ridge coefficients remain on a relative, not absolute, scale** - same caveat as the original Phase 2 report; season fixed effects have their own separate scale and should not be interpreted as directly comparable to the player coefficients.
