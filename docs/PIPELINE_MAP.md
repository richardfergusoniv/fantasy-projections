# The Projection Pipeline

How raw NFL play-by-play becomes a calibrated draft board: the data, the models, the reconciliation that makes a team's players add up, the Monte Carlo layer that turns point estimates into distributions, and the acceptance gates that decide whether any of it is allowed to ship.

**Updated:** 2026-08-29

**Stage flow:** Raw nflverse data → Feature engineering → Point models → Resolve roster + depth → Compose + reconcile → Selected board (ensemble) → Simulate (profile draws) → Acceptance gate → Draft board (VORP + tiers) → Seal release bundle → Promote pointer → Browser surfaces

> **Where the workflow stands today.** There are two boards in the repository and they are not the same board. The **browser** no longer reads the loose `players_2026.json`; it resolves `active_release_2026.json` → a sealed, hash-verified bundle under `draft_assistant/data/releases/<namespace>/` (§8a). The active bundle is `phase1_rehearsal_20260829`: the **accuracy-first** ensemble board at 10000 draws, with every draw-derived overlay populated on all 778 players. The loose legacy files still on disk are the older **v1/v2** export whose overlay was withheld by `selected_board_hash_mismatch` — they are the pointer-absent bootstrap path, not what the site serves. See §10 for both states side by side.

---

## 1. Raw data sources

Everything nflverse publishes is pulled through `src/ingest/sources.py`, cached, and loaded into local SQLite. Depth-chart eligibility comes from a hand-curated file the pipeline never edits itself.

### nflverse → SQLite

| Source | Call | Table | Used for |
|---|---|---|---|
| Play-by-play | `import_pbp_data` | `pbp` | Attempts, rush/red-zone/air-yards, EPA, opponent strength |
| Weekly box scores | `import_weekly_data` | `weekly` | Per-player-week stats; pbp fallback before the official file publishes |
| Snap counts | `import_snap_counts` | `snap_counts` | Offensive snap %, keyed by `pfr_player_id` |
| Depth charts | `import_depth_charts` | `depth_charts` | Preseason charts 2016–2026 (two schema eras, harmonised) |
| Weekly rosters | `import_weekly_rosters` | `weekly_rosters` | ACT/INA/DEV/RES status — eligibility and role-zero logic |
| Seasonal rosters | `import_seasonal_rosters` | `seasonal_rosters` | Age, position, rookie season |
| Schedules | `import_schedules` | `schedules` | Opponent list for strength-of-schedule priors |
| Injuries | `import_injuries` | `injuries` | Weekly report → `injury_durability_rate` |
| Draft picks | `import_draft_picks` | `draft_picks` | Draft capital — rookie path only |
| Participation / NGS / FTN / PFR / combine | various | — | Loaded; partially consumed |

### Hand-curated — `src/depth_chart/`

| File | Purpose |
|---|---|
| `starters_2026.csv` | Eligibility / room membership / formation role, never auto-edited |
| `live_depth_2026.csv` | Derived chart after applying injury events |
| `status_overrides_2026.csv` | Dated IR / PUP / suspension games caps |

---

## 2. Feature engineering

Raw tables aggregate to one row per player-season (2016–2025) in `features.py`, built on `data_prep.py`, `ol_quality.py` and `depth_history.py`.

**Usage shares** (`features.py` · `FEATURE_COLS`) — volume as a share of team opportunity rather than raw counts, so it survives a role change:
- `carry_share`, `target_share`
- `rz_carry_share`, `rz_target_share`
- `rz_carry_monopoly`, `rz_target_monopoly`
- `air_yards_share`, `adot`
- `peak_receiving_yards_share`

**Team & scheme context** (`data_prep.py` · `OC_METRICS`) — prior-season opponent and coordinator signal, decoupled from the player's own production:
- `opp_def_pass_epa_prior`, `opp_def_rush_epa_prior`
- `pass_oe`, `pass_oe_neutral`, `neutral_sec_per_play`, `play_action_rate`
- `personnel_11_rate` / `12_rate` / `21_rate` / `other_rate`
- `qb_designed_run_rate`

**Quality & health** (`ol_quality.py`, `data_prep.py`) — slow-moving traits:
- `snap_pct`
- `ol_pass_protection_score`, `ol_run_blocking_score`, `ol_confidence_low_churn`
- `injury_durability_rate` = (missed games + 0.4×flagged-but-played) / team games
- `age`, `career_year`

**Role stability & depth** (`transitions.py`, `depth_history.py`):
- `prior_{stat}_pg` for every target stat
- `prior_role_rate`, `prior_role_rate_3y` for stable stats
- `target_depth_rank` (availability feature), `DEPTH_TIER_COLUMN` (volume feature)
- Two-season blend `w = min(1, games / 8)`, RB/WR/TE only

### Where two of those feature families come from

`OC_METRICS` and the `ol_*` columns are not aggregations — each is the output of its own upstream fitting stage, and both are re-used as plain features by everything in §3.

**Coordinator tendencies — `src/coordinator/`.** `tendencies.py` computes one profile per (season, team) from pbp/FTN/participation: `neutral_sec_per_play` (median snap-to-snap seconds inside win probability 0.2–0.8, quarters 1–3, so garbage-time snap-spam cannot distort pace), `pass_oe` (win probability 0.05–0.95) and `pass_oe_neutral` (0.2–0.8), `play_action_rate` (NULL before 2022 — FTN does not exist yet, a real gap rather than an imputed zero), and the `personnel_*_rate` shares with `n_personnel_plays` carried as the denominator so a thin week reads NaN, not 0.

`oc_profiles.py` re-keys those profiles to the coordinator who actually called the plays, and handles the case with no play-by-play to key on: a **first-year OC in a new seat** inherits a blend of the team's own prior-season profile and the incoming OC's most recent prior stop. Weights live in one place (`inheritance.INHERITANCE_WEIGHTS`) and are LOSO grid-fit, not asserted — **0.60 team / 0.40 OC for both internal promotions and outside hires**, which beat the original judgment-call 70/30-and-30/70 split and the team-only baseline. A first-time play-caller with no prior stop has no (b) to blend and the row is flagged team-inertia-only. Returning coordinators use their observed profile directly.

**OL attribution — `src/ol_model/`.** Two ridge sub-models (pass protection, run blocking) attribute play outcomes to individual linemen, producing `ol_pass_protection_score` and `ol_run_blocking_score`. The shipped fit is **pooled across 2021–2025** (`pooled_fit.py`): one lineman-indicator column per `gsis_id` plus season fixed effects, so a player who appears in five seasons gets one coefficient instead of five noisy per-season ones. `RidgeCV` picks the prediction-optimal alpha and the final fit uses **10× that** — deliberately, because CV optimises held-out prediction while this model is being read as *attribution*, and split-half coefficient stability keeps improving up to roughly 10× before flattening.

`churn.py` supplies the honesty column. A team that runs the same five linemen for most of a season makes their indicator columns collinear, so ridge cannot separate individual credit inside that block. Team-seasons at or above **0.90** top-lineup snap share are marked unit-level, and that propagates to `ol_confidence_low_churn` — the score in those rows is a shared unit effect, not an individual measurement.

**Weekly features** (`data/features_weekly.py`) — a separate player-week frame for the weekly track. `targets_share_roll3` / `carries_share_roll3` are lagged one week *before* rolling (`shift(1).rolling(3)`), so a target week's feature cannot contain that week's own outcome. The as-of audit (`scripts/audit_weekly_features.py`) is green on this.

---

## 3. Point models

Six approaches, each matched to how much data and how much smoothness its target has. All fit on season-N→N+1 transition pairs, 2021–2025.

### Role rates & shares — LightGBM, 24 cells

One `LGBMRegressor` per (position, stat), heavily regularised because each cell holds only a few hundred rows:

```
n_estimators=100, max_depth=3, num_leaves=8, min_child_samples=10,
subsample=0.8, colsample_bytree=0.8, reg_alpha=reg_lambda=0.1
```

Target is a *per-eligible-week* rate (season total ÷ rostered non-reserve weeks, zeros included), not per-appearance. Receiving stats are reframed to predict a **share of team volume**, composed against the team model downstream. Each model persists its own feature list, so cells can diverge.
— `train.py :: fit_one()`

### Team totals — RidgeCV, 4 anchors

Team-season pass attempts, passing yards, carries, rushing yards. Smooth, autocorrelated targets on ~100 rows (32 teams × 3–4 transitions), where regularised linear beats boosting.

```
RidgeCV(alphas = logspace(-2, 3, 20))   # leave-one-out CV
```
— `train.py :: fit_team_total()` → `models/team_*.joblib`

### Availability — LightGBM, 4 cells

Games played next season, one model per position, from usage history plus the *target* season's preseason chart. Turns a per-game rate into a season total.
— `train.py :: fit_availability()` → `models/{POS}_games.joblib`

### Rookies — rule-based, not ML

Rookies have none of the trailing-history features everything else depends on, so this is a two-parameter fit plus an explicit vacancy adjustment:

```
rate = intercept + slope * log(effective_pick)          # OLS, per position/stat
rate_adj = rate * clip(vacated_share / bucket_vacated_share, VACATED_CLIP)
```

Scaling above 1× only applies to rookies the curated chart lists as starter or committee, and rookie claims are netted against what veteran models already took from the same opening.
— `rookies.py`

### Elite shrinkage — OLS, WR/TE receiving yards

Single-parameter post-hoc correction fit on cross-fitted leave-one-transition-out residuals, applied only above a per-position knot (WR 60, TE 50 ypg):

```
residual ≈ beta * max(0, observed_ypg - knot)           # through the origin
```

Ships only past three gates: ≥15 rows above the knot, positive beta, cross-season consistency ≥ 2.0 SE. Capped at 8.0 ypg.
— `corrections.py`

### Conditional intervals — QuantileRegressor, 46 models

Per-cell q10/q90 regressions on `pred`, `depth_tier`, `experience_bucket`, `volume_bucket`, fit on strictly-forward cross-fitted residuals — so a band widens with volume and role rather than being flat per cell.

```
pred_low/high = pred_pg + q10/q90(resid | pred, depth, experience, volume)
```

The predictor returns **one row per input row**, index-aligned. A frame missing those features falls back to the flat empirical band rather than borrowing one.
— `evaluation/interval_models.py`, `veterans.py`

### QB context — a retrain track, not a shipped model

`qb_context.py` defines `QB_CONTEXT_FEATURES` (`qb_changed`, `qb_prior_epa_per_dropback`, `qb_prior_cpoe`, the change-versus-previous deltas, `qb_rookie_or_unknown`, `qb_low_sample`) as a **separate contract** from `ALL_FEATURES` / `ROLE_FEATURES` / the availability features, with no target-season outcome allowed in any column. `train.py` picks these up only where the columns are present (`context_cols = [c for c in QB_CONTEXT_FEATURES if c in data.columns]`), so the E2 retrain is opt-in on the frame rather than on a flag.

Whether it ships is decided by `evaluation/qb_context_gate.py` against a frozen baseline over folds 2023/2024/2025. It is fail-closed: a missing fold, a missing evidence manifest, or an unmatched artifact hash yields `hold_qb_context` rather than a partial pass, and the best available verdict is `qb_context_review_ready` — a *review* state, not an activation.

---

## 3a. Resolving who plays where — before anything is composed

A rate model answers "how productive per week", not "on which roster, behind whom, for how many games". Two leaf modules answer that second question, and both are deliberately import-isolated (neither may import `predict`) so the resolution order cannot be quietly rearranged.

**Roster and vacancy — `roster_moves.py`.** Resolves the target-season roster map, then handles the two ways a player's opportunity changes without his own history changing:

- **Incumbent vacancy** — a returning player inherits part of what left his room, damped by `INCUMBENT_VACANCY_ALPHA` and bounded by `INCUMBENT_VACANCY_NET_CLIP` / `INCUMBENT_VACANCY_SCALE_CAP`. Only `BOOST_ELIGIBLE_ROLES` (`starter`, `committee`) are eligible, so a fifth receiver does not absorb a departed WR1's targets.
- **Team changers** — reassigned to the new team and re-based against *its* vacated share, clipped by `TEAM_CHANGE_SHARE_CLIP`.

Vacancy damping is provenance-tracked in `contracts.py` rather than tuned freely: RB carry alpha sits at **0.0 (neutral)** because the evidence that had re-enabled it at 1.0 turned out to be Sleeper-comparison rows, and Sleeper agreement is not accuracy (§7b). It was disabled on Sleeper evidence and re-enabled on Sleeper evidence, never scored on realised points — so it is off until it is measured.

**Depth gating and the Gate-B rate ladder — `depth_gating.py`, `depth_rates.py`.** The curated chart, the live chart (post-injury-event) and the dated status overrides load here, and a conditional-rate multiplier keyed on `(position, nfl_depth_rank)` discounts a projection for depth:

| Position | Rank 1 | 2 | 3 | 4 | 5 | Deeper | Off chart |
|---|---|---|---|---|---|---|---|
| QB | 1.00 | 0.77 | — | — | — | 0.84 | 1.00 |
| RB | 1.00 | 0.98 | 0.73 | — | — | 0.70 | 0.86 |
| WR | 1.00 | 1.00 | 0.97 | 1.00 | 0.86 | 0.94 | 0.79 |
| TE | 1.00 | 0.90 | 0.83 | — | — | 1.00 | 0.77 |

The application rule lives in `depth_rates.py` and **only** there. That consolidation was a fix, not tidying: the ladder had been applied by three different rules in three places — `depth_gating` applied it only when a curated chart existed (true for 2026 alone, so the shipped path discounted exactly one season), `fantasy_evaluation` applied it unconditionally (so every leakage-safe fold measured something the shipped path did not do), and `backtest.py` never applied it at all — which meant `interval_residuals.csv` and the elite-shrinkage coefficients in `corrections.joblib` were fit on *undiscounted* predictions and then consumed by a path that ships discounted ones. The curated-chart guard was a leftover from before Gate B re-keyed the multiplier onto `nfl_depth_rank`, which `depth_history.py` reconstructs from nflverse for every season.

---

## 4. Composing one coherent board

Every stat above is predicted independently. This stage stops a team's players collectively over- or under-claiming their own team's predicted volume — without silently redistributing playing time between teammates.

```
              Team anchor · RidgeCV
        (QB owns 0.941 · RB owns 0.810)
                      |
        reframed stats: share x team total
                      v
   Player A        Player B        Player C      (independent pred_pg)
        \              |              /
         \             |             /
          v            v            v
              reconcile_team_volume
     team sum(players) -> pulled toward anchor, alpha = 0.75 fitted
                      |
                      v
  concentration -> TD-rate clip -> child<=parent -> season totals -> identities
```

1. **Attach team anchors** — every player row receives its team's predicted pass attempts, passing yards, carries and rushing yards.
2. **Compose reframed receiving stats** — predicted share of team passing yards, normalised against the team's summed shares and multiplied by the team total. Rookie shares enter the same denominator, so an incoming rookie squeezes veterans.
3. **Reconcile team volume — α = 0.75** — each team's summed player volume is pulled toward its anchor by an exponent **fitted by rolling origin**, not asserted. Pooled player-level season MAE picked 0.75 (64.185) over 1.0 (64.204), 0.5 (64.311) and 0.25 (64.644). `TEAM_RECONCILE_ALPHA = 0.5` survives only as the unfitted fallback. QB rooms protect the tier-1 starter; RB rooms scale together.
4. **Position rooms claim their measured share** — QB takes **0.941** of team attempts and **0.942** of passing yards; RB takes **0.810** of carries. A structural 1.000 was tried and reverted: paired per-player scoring over 2023–2025 (n=312) put it at −0.478 QB mean absolute error, 95% CI [−0.86, −0.10]. The remainder is scrambles and sweeps.
5. **Concentration, TD rates, identities** — within-room concentration calibrated with exact volume conservation; pass-TD-rate and rush-TD-per-carry clipped to historical bands; `completions ≤ attempts` and `receptions ≤ targets` enforced.
6. **Season totals and team identities** — `pred_season = pred_pg × projected_games`, then a symmetric geometric split restores team-level season identities (receiving yards = passing yards, receptions = completions).

---

## 4a. The selected board — post-process ensemble

`compose_board` output is no longer published directly as the mean. A per-position blend runs *after* composition and decides which number the site ranks on. Composition itself is untouched, so team identities and reconciliation still hold on the v1 frame underneath.

| Layer | `model_id` | What it blends | Where |
|---|---|---|---|
| v1 rate forecast | `v1_rate_forecast` | composed board alone | §3–4 |
| v1/v2 draft ensemble | `v1_v2_ensemble` | `v1_pred`, `v2_pred` per position | `src/draft_assistant/ensemble_weights.json` |
| **accuracy-first ensemble** | `accuracy_first_ensemble` | `v1_pred`, `v2_pred`, **`adp_points`** | `output/accuracy_first_2026/ensemble_weights.json` |
| v3 means cutover | `v3_means` | v3 simulation p50 as the mean | flagged, not applied |

The accuracy-first arm (`accuracy_first_2026_v1`, verdict `promote_accuracy_ensemble`) is fit on 2024, selected on an untouched 2025 holdout, refit on 2024–2025, against top-120 ADP half-PPR point MAE with non-regressing Spearman. It lets the **market into the forecast** for the two positions where the models lost:

| Position | Arm | v1 | v2 | ADP |
|---|---|---|---|---|
| QB | incumbent | 0.40 | 0.60 | — |
| RB | market_no_v3 | 0.10 | 0.30 | **0.60** |
| WR | market_no_v3 | 0.00 | 0.55 | **0.45** |
| TE | incumbent | 0.90 | 0.10 | — |

`v3_p50` was a candidate in every arm and lost to `market_no_v3` at RB/WR — the simulation median is not part of the published mean. Selection is hash-pinned (`artifact_hash`, plus source hashes for the v3 calibration, the incumbent weights, and the exact-p50 evaluation frames).

**The board hash is the contract.** `simulate` records `selected_board_hash` / `selected_board_model_id` on the simulation manifest; `prepare` recomputes it from the board it is about to export and refuses any draw-derived overlay on mismatch. That guard is what is firing today (§10).

---

## 5. The simulation layer

The composed board is a point estimate. This stage turns it into a distribution — 10000 draws per season under the decision-stable compromise — by representing both *sampling* noise and the larger uncertainty that the projection itself is wrong.

```
FITTED UNCERTAINTY (leakage-safe, OOF folds 2023-2025)
  team residual      Dirichlet         beta-binomial      conversion
  covariance         concentration     games              sigmas
      |                  |                  |                 |
      v                  v                  v                 v
 team volume  ->  opportunity share  ->  player volume  ->  conversions
 anchor+MVN       simplex, exposure-      x drawn games      Poisson/binom/
 residual         weighted                                   lognormal
                                                                 |
                                                                 v
                                              season stat line -> fantasy points
                                                       x 10000 draws
                                                                 |
                       joint bootstrap correction  <--------------+
              whole donor residual vector, re-centred on generative p50
                                   |
                                   v
                    p10 · p25 · p50 · p75 · p90

  point estimate, VORP and tiers never pass through this layer
```

### Draw budget and profiles — `config/simulation.json`

Draw counts are configuration, not a constant. `--simulation-profile` selects one; `random_seed` and per-artifact `deterministic_seed` make a run reproducible.

| Profile | Draws | Use |
|---|---|---|
| `dev` | 1000 | fast local iteration (still the argparse default) |
| `publish` | 10000 | namespaced sealed bundle (`--artifact-namespace`); does not flip the active pointer |
| `release_candidate` | 10000 | namespaced, non-public RC publish (requires `--artifact-namespace` and `--rollout-label`) |

The same file carries the stability tolerances, decision thresholds (`p_finish_top12/24`, `p_vorp_positive` at 0.5) and diagnostic reference draw counts that §9's evaluation tracks consume, so gate thresholds and the runs they judge read from one source.

Draws are written **partitioned** under `output/model_v3/simulations/season=<season>/run_id=<id>/` (40 partitions on the current run) with per-partition hashes on the manifest, alongside the flat parquet and the recentered pair. The current 2026 run: 10000 draws, `joint_bootstrap`, ~7183 s.

**WR residual scale.** The joint-bootstrap band under-dispersed at WR specifically; `wr_calibration_version = v1_wr_residual_scale` applies `wr_residual_scale = 1.7`, hash-pinned on the manifest.

### Generative draw — `inference/reconcile.py`

One team volume draw feeds a player's whole line, so his stats move together. Volume is allocated on the simplex from an **exposure-weighted** season prior (`pred_season`), so a player projected for ten games cannot claim a full share. WR, TE and RB compete for *one* pool of team targets; QB and RB carry pools are separate and each takes only its measured share.

Conversion rates are the ones implied by that player's own board predictions — completions/attempts, yards/completion, TDs/attempt — clipped to plausible bands, so two QBs on equal volume do not produce identical lines.

### Projection uncertainty — `models/uncertainty.py`

Sampling noise alone is far too tight. Three components, all fitted leakage-safely from rolling out-of-fold rows through 2025:

```
team environment   correlated MVN residual on (pass attempts, carries)
opportunity share  Dirichlet concentration, method of moments per pool
availability       beta-binomial games, by position x role x fragility
```

Concentrations land at **1.0** for QB attempts, **4.7** for RB carries, **29.7** for receiving targets — a top-heavy QB room genuinely carries more allocation risk than a receiving corps. Availability is fit across 39 cells.
— `scripts/fit_v3_uncertainty.py`

### Joint bootstrap — the selected distribution mode

The generative layer still under-disperses. Instead of adding per-stat noise independently — which destroys within-player correlation (+0.62 to +0.88 between a player's own stats) and understates summed spread by 31% — this resamples a whole donor player-season's fantasy-point residual, bucketed by position and role, then **re-centres the finite resample** so the published p50 stays exactly on the generative median.

Distribution only: it changes the band, never the point estimate the board ranks on. 1952 donors, hash-verified against the manifest that selected them.

---

## 6. The acceptance gate

Nothing distributional reaches the board on the strength of existing. A candidate is scored against realised fantasy points on the *production* simulator, rolling-origin, and must clear every gate — then three artifacts must hash-match before the overlay attaches.

| Arm | Coverage | Width | p50 MAE | ρ | Verdict |
|---|---|---|---|---|---|
| baseline — no uncertainty | 0.5455 | 61.2 | 33.21 | .7625 | reference |
| generative + projection uncertainty | 0.7343 | 90.9 | 33.66 | .7647 | short of band |
| **joint bootstrap** | **0.7552** | 85.2 | 33.66 | .7620 | **selected** |

| Gate | Value |
|---|---|
| Aggregate coverage within [0.75, 0.85] | 0.7552 |
| Every fold within [0.72, 0.88] | 2 folds |
| No position below 0.70 at n ≥ 50 | QB .793 · RB .767 · WR .747 · TE .734 |
| Interval score improves on baseline | pinball |
| No material p50 MAE regression | +0.454 ≤ 0.5 |
| No material rank regression | −0.0005 ≥ −0.01 |

**Fails closed by construction.** A `hold` verdict — or a manifest with no verdict at all — zeroes the uncertainty parameters and runs the plain baseline, so a rejected candidate cannot ship simply because it was fitted. The promotion gate then re-checks the whole chain independently: calibration basis, selected mode, uncertainty `artifact_hash`, joint-donor `sha256`, and the simulation's own `projection_run_id`. Any mismatch and the percentile columns are refused rather than published stale.
— `scripts/calibrate_v3_distribution.py`, `scripts/v3_promotion_gate.py`

---

## 7. Scoring, VORP and tiers

Deterministic arithmetic on the composed board — no ML from here on.

### Fantasy points — half-PPR, 4-pt passing TD

`fantasy_pts = Σ pred[stat] × weight[stat]`

| Stat | Weight | | Stat | Weight |
|---|---|---|---|---|
| Passing yards | 1 / 25 | | Rushing yards | 1 / 10 |
| Passing TD | 4 | | Rushing TD | 6 |
| Interception | −2 | | Receiving yards | 1 / 10 |
| Reception | 0.5 | | Receiving TD | 6 |

Fumbles lost and two-point conversions are not modelled upstream, so they are absent by construction rather than silently zeroed.
— `fantasy_points.py`

### VORP — `draft_assistant/vorp.py`

```
replacement_rank = roster_math(position) * (17 / mean_projected_games)
vorp = vorp_input_pts - replacement_pts        # signed, not floored at zero
```

Starters QB1/RB2/WR3/TE1 plus weighted flex demand; the availability factor deepens replacement for positions whose starters miss more games. Flooring VORP at zero was tried and rejected — it collapsed most of the board to ties. TE alone gets a shape correction toward a fitted historical surplus curve.

### Tiers & overlay — `draft_assistant/tiers.py`, `prepare.py`

Tiers break where the gap to the next player exceeds an absolute or relative threshold, computed separately for the whole board, within position, and the RB/WR/TE flex pool.

The published board carries `fantasy_pts_p10/p25/p50/p75/p90` and `volatility_flag` from the calibrated v3 overlay (gated by `simulation_ready`). When the finish-probability gate clears, additive fields `p_finish_top6/12/24/36/48` attach from **recentered** draws anchored on the accuracy-first selected forecast. When the simulated-VORP gate clears, `sim_vorp_p10/p50/p90`, `p_vorp_positive`, `expected_pos_rank`, and `median_pos_rank` attach as a separate uncertainty overlay. Deterministic `vorp`, ranks, and tiers remain authoritative.

#### Rank tie policies (intentional distinction)

Rank-derived simulation fields do **not** share one tie convention. The exported board documents this in `meta.draft_value_simulation.rank_tie_policies`.

| Field group | Tie policy | Pandas `rank` method | Meaning on ties |
|---|---|---|---|
| `p_finish_top6` … `p_finish_top48` | `first_occurrence` | `first` | One player gets each rank; stable row order breaks ties |
| `expected_pos_rank`, `median_pos_rank` | `minimum_competition_rank` | `min` | Tied fantasy points share the same rank; next rank skips (e.g. 1, 2, 2, 4) |

Top-N cutoff semantics are the same for both: a player qualifies when `positional_rank_draw <= N` within position for that draw. Do not assume finish probabilities and simulated rank moments were computed with identical tie handling.

Simulated VORP (`sim_vorp_*`) does not use positional rank; it is `recentered_draw_points - fixed_replacement_points[position]`.

### Recentered distribution — `inference/recenter.py`

```
recentered_draw = selected_points + (v3_draw - v3_p50)
```

Floored at zero with iterative median correction so recentered p50 matches the displayed board. Provenance requires both `canonical_projection_run_id` and `selected_board_hash` in `simulation_manifest_<season>.json`.

---

## 7a. Consumer surfaces

The board feeds five static pages under `draft_assistant/`. They do **not** read a loose export any more: every page loads `js/release_loader.js` first and resolves one sealed release namespace for the whole page load (§8a).

- **Draft board** (`index.html`) — ranks, tiers, VORP, snake tracking, roster builder. Stat displays are reconciled to the *blended* points, so a displayed stat line and the displayed total come from the same board.
- **Teams / Totals** (`teams/`, `totals/`) — same blended displays at team grain, from `team_stats_<season>.json` (`src/team_stats/prepare.py`, which carries the 32-team division/conference metadata). `src/team_stats/serve.py` is a thin back-compat wrapper around `draft_assistant/serve.py`; there is one local server, not two.
- **Compare** (`compare/`) — rebased onto the selected board (`compare_prepare.py`), so a comparison never quotes pre-ensemble numbers.
- **Sleepers** (`sleepers/`) — the deep board, ranked by upside against cost.

### Sleepers gate — `scripts/evaluate_deep_band_accuracy.py`

The Sleepers page is **not** shown on the strength of looking useful; it is gated on measured accuracy in the bands it actually covers, written to `output/deep_band_accuracy/report.json` → `draft_assistant/data/deep_band_accuracy.json`.

- Banding is by **projected season-points rank within season**, not board `overall_rank`. Bands: `top60`, `deep_core` (61–120), `deep_primary` (121–200), `deep_speculative` (201–300), `tail`.
- Admission metric is `p_startable_100` ≥ 0.15 at n ≥ 30. MAE and Spearman both improve as a band nears the zero floor and cannot separate skill from a dead tail; a hit rate can.
- Admitted today: `deep_core`, `deep_primary`, `deep_speculative` — i.e. **projected rank 61–300**. `top60` is excluded as not-sleeper.
- **Quarterbacks are excluded.** They post large raw season totals while being replacement-rich, so a points-rank band listed deep QBs at the top on raw upside while they carried roughly −210 VORP. The measurement reports a matching RB/WR/TE-only population for the rates the page quotes.

Band edges in the page are fallbacks; the real edges come from the measurement file.

## 7b. External comparison — diagnostic, never an acceptance criterion

`src/comparison/` compares the board against Sleeper's free public projections API (the only free service found with a clean bulk per-player endpoint, and it exposes `gsis_id` directly, so the join is trivial). This has earned its keep — it surfaced the share-denominator bug, the trade-vacancy bug and the Diggs/Okonkwo/White triple-boost — but **no number it prints is a gate**.

The two systems are not measuring the same quantity. Sleeper projects a full slate: `gp = 18` for roughly 9,370 of the 9,402 players it tracks, and it allocates about **96.8%** of team carries to named players against this system's **83.8%**. This system projects *expected value*, including the probability a player does not play. So agreement is not accuracy and divergence is not, by itself, a defect.

`spot_check.py` encodes that distinction structurally. It used to be, in effect, a Sleeper-agreement regression suite — the watchlist annotated each player with the direction he was expected to move *toward Sleeper*. It now flags **incoherence, never mere disagreement**: a fixed named watchlist checked against our own board, plus a Sleeper divergence report that is descriptive and cannot fail the run. The value was never the agreement — it was that a named watchlist catches structural failures aggregate metrics hide (the Phase 5 rookie-filter bug and the Phase 6 team-change bugs both showed up as one well-known player being missing or impossible while the MAE tables looked clean).

**Market metrics — `market_metrics.py`.** The separate, legitimate market surface: model-versus-contemporaneous-ADP/ECR draft-edge metrics over the top 120, with name normalisation for the join. It does not touch training. Holdout MAE and Spearman remain secondary accuracy checks; preseason draft value against market rank is the primary decision surface. This is also the module the accuracy-first ensemble (§4a) and decision-quality evaluation (§9) match against.

## 7c. Sentiment — diagnostic only, `src/sentiment/`

`markdown_market_v1` scores reviewed research summaries plus the ECR-versus-ADP gap into a **within-position, residual** score (`+50` ≈ 75th percentile of residual sentiment, not a 50% projection bump). No evidence stores as null / `coverage=none`, never as neutral. It appears on the board (`sentiment_score`, `sentiment_coverage`, `sentiment_confidence`, `sentiment_as_of`, `sentiment_model_active`) as a secondary Sleepers column and is never a sort key.

`models/sentiment_manifest.json` is the only activation surface and every position is `false`: the score alters no rate, availability, rookie, team-anchor, fantasy-point, VORP, tier or ensemble calculation. Activation needs three distinct preseason seasons, 200 non-null player-seasons and 40% coverage per position — passing the data audit alone never flips it. Dated snapshots under `data/sentiment/` are **tracked on purpose** (an explicit `.gitignore` negation) because they are the point-in-time history the gate is accumulating; until that negation existed, every run wrote to an ignored path and the history could not build at all.

---

## 8. The file chain, end to end

```
train.py                per-stat + team-total models          -> models/*.joblib
backtest.py             rolling residuals, interval models    -> models/interval_models/,
                                                                 output/backtest/residuals_rolling.parquet
fit_reconcile_alpha.py  alpha by rolling origin               -> models/reconcile_calibration.json (0.75)
fit_v3_uncertainty.py   team / share / availability params    -> models/v3/uncertainty/manifest.json
calibrate_v3_distribution.py
                        scores the production simulator       -> output/backtest/
                                                                 v3_fantasy_interval_calibration.json
                                                    selected_distribution_mode = joint_bootstrap

publish.py              publish() - one transaction
  predict.project_season()      veterans / rookies / replacement  -> per-stat pred_pg
    composition.compose_board()   anchors, reconcile, concentration,
                                  TD clips, season totals, identities
  validate_projection_contract()
  _git_revision()               captured BEFORE anything is written
  simulate.write_simulation_outputs()   generative + joint bootstrap
  fantasy_points.compute_fantasy_points()
  draft_assistant.prepare.export_draft_data()   ensemble -> vorp -> tiers -> overlay
                                                    -> players_2026.json
                                                       (atomic swap, manifest last)
  release_report.build/write_release_report_simulation()
```

Release monitoring is two-stage and merged: `publish` writes `release_report_simulation_<season>.json`, `prepare` writes `release_report_board_<season>.json`, and the two merge into `release_report_<season>.json`.

Entry point:

```bash
python -m src.projection.publish --season 2026

# 10k sealed bundle (does not activate)
python -m src.projection.publish --season 2026 --simulation-profile publish --artifact-namespace <ns>
python scripts/validate_release_bundle.py --season 2026 --artifact-namespace <ns>
python -m src.projection.promote_release --season 2026 --artifact-namespace <ns>
python scripts/validate_release_bundle.py --season 2026 --artifact-namespace <ns> --require-active

# 10k release candidate, namespaced and non-public
python -m src.projection.publish --season 2026   --simulation-profile release_candidate   --artifact-namespace <ns> --rollout-label <label>
```

A non-`dev` profile requires `--artifact-namespace`. `publish` writes an immutable `release_bundle_manifest_v2` (promotion-eligible) under `output/model_v3/release_bundles/` and copies browser artifacts to `draft_assistant/data/releases/<ns>/` without rewriting `active_release_<season>.json`. Older `release_bundle_manifest_v1` bundles remain readable but cannot be promoted (`promotion_eligible=false`). `promote_release` runs six fail-closed promotion invariants, copies browser artifacts through a temp public namespace with rehashing, requires a clean git tree with either initial equality or restore ancestry against the bundle's `source_commit`, atomically replaces the pointer, and writes a tracked promotion receipt. `release_candidate` additionally requires `--rollout-label` and routes through `src/projection/release_candidate.py`.

Artifacts are staged in a temp directory and atomically swapped, with the run manifest moved last as the commit marker; an interrupted run is rejected rather than half-published.

## 8a. The release layer — what the browser actually reads

Publishing used to mean overwriting `draft_assistant/data/players_2026.json` in place. It no longer does. A release is now an **immutable, hash-sealed namespace**, and going live is a **pointer swap** — which makes rollback a repoint rather than a republish.

```
publish --artifact-namespace <ns>
  |
  |-- output/model_v3/release_bundles/season=<s>/namespace=<ns>/   (local full bundle; gitignored)
  |     release_bundle_manifest.json   <- release_bundle_manifest_v2 (new), v1 legacy
  |     release_bundle_validation.json <- mutable attestation sidecar; never restore authority
  |     ...56 artifacts, each sha256 + byte_size
  |
  '-- draft_assistant/data/releases/<ns>/      (public browser copies; tracked)
        players / team_stats / comparison / deep_band_accuracy / manifest

validate_release_bundle.py       attestation checks + promotion_invariants (6 named checks)
promote_release --artifact-namespace <ns>
  |
  |-- draft_assistant/data/active_release_<s>.json          <- pointer swap
  '-- draft_assistant/data/promotion_receipts/<s>/<ns>.json <- tracked activation receipt
```

**Promotion invariants** (`evaluation/promotion_invariants.py`) — all six must pass before pointer movement; no force flag:

1. `overlay_coverage_alignment` — per-field non-null counts in manifest/report match final `players_<season>.json`
2. `selected_board_hash_alignment` — canonical `selected_board_sha256` in manifest, players meta, simulation manifest, and release report
3. `simulation_profile_identity` — `profile_key`, `profile_label`, `draw_count`, `chunk_size`, `configuration_hash`, `policy_hash` agree with sealed rollout decision and `config/simulation.json`
4. `ensemble_source_provenance` — sealed `v2_points`, `adp_source`, `ensemble_weights`, and contract hashes agree
5. `browser_artifact_completeness` — every `browser_consumed` artifact appears exactly once in the public namespace with matching bytes
6. `git_provenance` — clean tree, `source_dirty=false`, and either **initial** (`HEAD == source_commit`) or **restore** (`source_commit` is an ancestor of `HEAD`)

**Provenance modes.** A namespace that has never been promoted stays on strict equality. Restore mode is derived only when the exact namespace, release ID, and manifest hash match the active pointer's current or previous release, or a git-tracked `release_promotion_receipt_v1`. The mutable `release_bundle_validation.json` sidecar never authorizes restore. See [`PROMOTION_PROVENANCE_2026-08-30.md`](decisions/PROMOTION_PROVENANCE_2026-08-30.md).

**The manifest carries identity, never status** (`release_bundle.py`). `release_bundle_manifest_v2` records the namespace, `release_id`, model id, the `runs` pair (`projection_run_id`, `simulation_run_id`), canonical `selected_board_sha256` (plus `selected_points_vector_hash` separately), `overlay_coverage`, `ensemble` provenance, `git` provenance, the simulation block (profile identity, draw count, configuration hash, policy hash, joint-donor hash, calibration hashes) and the overlay population hash — plus per-artifact `sha256`/`byte_size`/`media_type`/`required`/`browser_consumed`. Keys named `status`, `active` or `inactive` are **forbidden on the document by construction** (`FORBIDDEN_MUTABLE_KEYS`), so "which release is live" can never be a field someone rewrites inside a sealed bundle. **The v2 board is sealed with everything else.** `v2_pred` comes from a separate repository (`../fantasy-projections-2`, synced in by `from_v2.py`) and carries 0.55 of the published WR mean and 0.30 of RB. It used to be read as a "frozen source" without being hashed into the application contract or copied into the bundle, so that one input could be swapped and every other hash in the chain would still validate. It is now pinned as `source_hashes.v2_points_<season>` and sealed as the `v2_points` artifact, and `validate_release_bundle` checks the sealed copy against the pin. Bundles sealed before the pin have neither and skip the check rather than failing it.

Every file in the namespace must be enumerated — an unlisted file fails the seal outright, which is exactly how the rehearsal caught `simulations_2026.parquet` being written but not declared.

**Mutable status lives on the pointer** (`active_release.py`, `promote_release.py`). `active_release_<season>.json` is `active_release_pointer_v1`: namespace, release id, manifest path and its `manifest_sha256`, `activated_at`, the `public_urls` map, and a `previous` block naming the namespace/release it replaced (optional `previous.manifest_sha256`). `promote_release` revalidates the sealed bundle and verifies the public copies still hash-match before it atomically replaces that one file, then writes a tracked promotion receipt.

**The browser fails visibly, not quietly** (`draft_assistant/js/release_loader.js`, loaded by all five pages). Its contract, in order:

1. Fetch the pointer with cache revalidation.
2. Pointer **absent** (404/network) → bootstrap the legacy loose files. This is the only path that reads `players_<season>.json` directly.
3. Pointer present but malformed, or the manifest/files fail validation → **fail visibly. Never fall back to legacy.** A stale board rendered as if current is the failure mode being prevented.
4. Freeze the namespace at the first successful pointer read, so a promotion landing mid-page-load cannot mix two bundles on one screen.

`evaluation/release_pointer.py` implements the same repoint-don't-republish rule for the `output/model_v3/releases/` side.

Profiles map onto this directly: `publish` seals a bundle without touching the pointer; `promote_release` flips it; `release_candidate` (`release_candidate.py`) additionally requires `--rollout-label` and lands under `output/model_v3/release_candidates/`, non-public by construction.

## 9. Future modeling (out of scope for shipped overlays)

The Phase 2 simulation overlays (finish probabilities, simulated VORP) are built on the existing rate-forecast point engine and recentered v3 draw distribution. Further modeling work stays in separate tracks:

- **Draw-count stability** — nested-prefix evaluation at 1k/2k/5k/10k via `scripts/evaluate_draw_stability.py`; decision recorded in `output/model_v3/draw_count_decision.json`.
- **Intermediate draw sweep** — 7.5k/10k/15k vs validated 20k reference via `scripts/evaluate_draw_stability.py --gate-mode production_v20k --skip-generate-reference --sweep-phase intermediate_v20k`; artifact `draw_stability_intermediate_v20k_<season>.json`. Frozen copy + `freeze_manifest.json` under `output/model_v3/frozen/draw_stability_intermediate_v20k_<season>/` via `scripts/freeze_draw_stability_evidence.py`.
- **Draw-count rollout (RC)** — namespaced non-public 10k publish via `python -m src.projection.publish --simulation-profile release_candidate --artifact-namespace <ns> --rollout-label <label>`; artifacts under `output/model_v3/release_candidates/season=<season>/namespace=<ns>/`. Rollout policy in `output/model_v3/draw_count_rollout_decision.json` (`draw_count_rollout_decision_v2`, Phase 2 closed; revisited 2026-08-29). Human records: [`DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md`](decisions/DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md), [`DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-29.md`](decisions/DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-29.md). Production profile: `decision_stable_compromise_10000` (10k draws; human-approved operational compromise — decision-stable at 10k, but the strict numerical gate against 20k did **not** pass). Closure via `scripts/write_draw_count_rollout_closure.py`.
- **Decision-change diagnostics** — per-event threshold analysis with 20k reference validation via `scripts/evaluate_draw_decision_diagnostics.py`; artifacts `output/model_v3/decision_change_events_<season>.parquet`, `player_stability_diagnostics_<season>.parquet`, and `decision_change_diagnostics_<season>.json`. The 20k reference extends the existing 10k checkpoint (full-board generation, top-120 evaluation). Provenance failure yields `verdict: hold`.
- **Decision quality (E1)** — top-N precision/recall grids, tier calibration, one-dimensional segments and roster-independent ADP-choice regret over rolling-origin folds, reusing the leakage-safe populations from `fantasy_evaluation`, the market join from `market_metrics` and the authoritative VORP/tier contracts from `draft_assistant` (`evaluation/decision_quality.py`; `scripts/evaluate_decision_quality.py`). The gate (`decision_quality_gate.py`) derives its thresholds **from frozen baseline evidence rather than hard-coded constants**, and returns `hold` on any missing fold, contract hash, market snapshot or baseline reference.
- **QB context retrain (E2)** — `evaluation/qb_context_evaluation.py` + `qb_context_gate.py` against `FROZEN_BASELINE_ID` over folds 2023–2025; best verdict is `qb_context_review_ready`, worst is `hold_qb_context` (§3). `scripts/evaluate_qb_context.py`.
- **Segment calibration** — one-dimensional coverage/pinball segments on the recentered distribution, gated only at n ≥ 50 (`evaluation/calibration_segments.py` → `output/evaluation/season=<s>/calibration_segments.parquet` + `calibration_segment_summary.json`). Its `segment_report_hash` is one of the four calibration hashes pinned on the simulation manifest and the release bundle.
- **Evidence freezing** — `evaluation/evidence_freeze.py` copies draw-stability evidence into an immutable hashed bundle under `output/model_v3/frozen/`, so a threshold can be re-derived later from the evidence it was actually derived from.
- **Release monitoring** — two-stage artifacts `release_report_simulation_<season>.json` (publish) and `release_report_board_<season>.json` (prepare), merged into `release_report_<season>.json`. Newly built reports propagate an explicit 10k risk flag whenever `strict_gate_promotion=false`. The currently sealed active release report predates that propagation; correcting it requires a future or separately authorized replacement namespace.
- **Opportunity-first / compositional mean modeling** — Shadow 0A/0B research branch under `output/shadow_opportunity_mean/`; remains `hold` until it beats the accuracy-first incumbent on leakage-safe top-120 evaluation (do not wire into compose/publish until then).
- **Weekly hierarchical models** — weekly feature as-of audit is green (`targets_share_roll3` / `carries_share_roll3` shifted); hierarchical weekly ROS modeling may begin. Gate: `scripts/audit_weekly_features.py` → `output/weekly_audit/audit_report.json`.

## 10. Live state — 2026, as of 2026-08-30

Machine-readable control-plane snapshot (guarded by `tests/test_pipeline_map_live_state.py`):

<!-- LIVE_STATE_TABLE_BEGIN -->
| key | value |
|---|---|
| namespace | v2_baseline_20260830 |
| previous_namespace | v2_candidate_20260830 |
| release_id | e92edd22-40d9-4219-87f6-47a651489d15 |
| manifest_sha256 | a951ca5093a12e8c2d8637de8515ff12c0b82a3b7a5883ccf95ae155dbaf3a37 |
| model_id | accuracy_first_ensemble |
| draw_count | 10000 |
| overlay_population | 778 |
| strict_gate_promotion | false |
<!-- LIVE_STATE_TABLE_END -->

### What the browser serves — sealed bundle `v2_baseline_20260830`

Read from `draft_assistant/data/active_release_2026.json` → `data/releases/v2_baseline_20260830/`.

| Surface | State |
|---|---|
| Pointer | `active_release_pointer_v1`, status `active`, activated 2026-08-30T09:42:41Z |
| Release ID | `e92edd22-40d9-4219-87f6-47a651489d15`; manifest `a951ca5093…` |
| Rollback target | `previous.namespace = v2_candidate_20260830` (`4348bf20-…`) |
| Published mean | **`accuracy_first_ensemble`** — RB weights 0.10/0.30/0.60 (v1/v2/ADP), WR weights 0/0.55/0.45; ADP alignment ~0.93/0.85; RB/WR repair track closed, not solved |
| Simulation | 10000 draws, profile `publish` / `decision_stable_compromise_10000`, human-approved operational compromise (not a passed strict 20k gate) |
| Overlay population | 778 players with finish / VORP overlays populated |
| Sealed release report risk field | **Missing** on the active sealed report (predates automatic 10k risk propagation). Do not rewrite sealed bytes; next namespace picks it up |
| Full local bundle | Required under `output/model_v3/release_bundles/...` for validate / promote / rollback; public copies alone are enough to serve and smoke-test |

### Previous release — `v2_candidate_20260830`

Promoted briefly, then rolled back. Tracked promotion receipt retained so restore provenance can re-authorize it from ancestry + receipt/pointer identity. Same 10k operational draw-count policy.

### Permanently non-promotable — schema-v1 phase1 bundles

`phase1_rehearsal_20260829` and `phase1_rehearsal_prior` remain on disk as historical schema-v1 namespaces. They are readable for archaeology but **`promotion_eligible=false`** — no receipts were created for them, and they cannot be restored through the v2 promotion path.

### The loose legacy files — bootstrap path only

Read from `draft_assistant/data/players_2026.json` · `meta` when the pointer is absent. A *malformed* pointer does not fall back to them (§8a). Unshipped by choice: the `v3_means` cutover and every §9 research track.

---

## Related decisions

- [`SIMULATION_MODE_2026-08-26.md`](decisions/SIMULATION_MODE_2026-08-26.md) — retiring the interim arm, the coverage investigation, and the calibration steps
- [`QB_TEAM_VOLUME_SHARE_2026-08-26.md`](decisions/QB_TEAM_VOLUME_SHARE_2026-08-26.md) — the 1.000 → 0.941/0.942 revert and its paired-test evidence
- [`V3_PROBABILISTIC_PIPELINE.md`](decisions/V3_PROBABILISTIC_PIPELINE.md) — v3 scaffolding, the means backtest, and why the point engine is still v1
- [`PHASE1_PRODUCTION_REHEARSAL_2026-08-29.md`](decisions/PHASE1_PRODUCTION_REHEARSAL_2026-08-29.md) — the full build → validate → promote → rollback acceptance test, including the unlisted-artifact seal defect
- [`PHASE1_RELEASE_SIGN_OFF_2026-08-29.md`](decisions/PHASE1_RELEASE_SIGN_OFF_2026-08-29.md) — hash alignment table and the player-facing approval for `phase1_rehearsal_20260829`
- [`PROMOTION_PROVENANCE_2026-08-30.md`](decisions/PROMOTION_PROVENANCE_2026-08-30.md) — initial vs restore provenance, receipts, and local-bundle dependency
- [`DRAW_COUNT_ROLLOUT_2026-08-28.md`](decisions/DRAW_COUNT_ROLLOUT_2026-08-28.md) and the two dated human-decision records — how 10k became the production draw count
- [`V1_PRODUCTION_ROLE_2026-08-29.md`](decisions/V1_PRODUCTION_ROLE_2026-08-29.md) — v1 structural role and closed RB/WR shadow repair track
- [`ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`](decisions/ACCURACY_FIRST_ENSEMBLE_2026-08-27.md) — selected RB/WR weights and holdout evidence
