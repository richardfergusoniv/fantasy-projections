# The Projection Pipeline

How raw NFL play-by-play becomes a calibrated draft board: the data, the models, the reconciliation that makes a team's players add up, the Monte Carlo layer that turns point estimates into distributions, and the acceptance gates that decide whether any of it is allowed to ship.

**Updated:** 2026-08-26

**Stage flow:** Raw nflverse data → Feature engineering → Point models → Compose + reconcile → Simulate (1000 draws) → Acceptance gate → Draft board (VORP + tiers)

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

## 5. The simulation layer

The composed board is a point estimate. This stage turns it into a distribution — 1000 draws per season — by representing both *sampling* noise and the larger uncertainty that the projection itself is wrong.

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
                                                       x 1000 draws
                                                                 |
                       joint bootstrap correction  <--------------+
              whole donor residual vector, re-centred on generative p50
                                   |
                                   v
                    p10 · p25 · p50 · p75 · p90

  point estimate, VORP and tiers never pass through this layer
```

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
| No position below 0.70 at n ≥ 50 | QB .793 · RB .767 · WR .748 · TE .734 |
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

The published board carries `fantasy_pts_p10/p25/p50/p75/p90`, `volatility_flag` and `p_top12/24/36` for all 778 players — a distributional overlay beside means and ranks that come only from v1.

---

## 8. The file chain, end to end

```
train.py                per-stat + team-total models          -> models/*.joblib
backtest.py             rolling residuals, interval models    -> interval_models/,
                                                                 residuals_rolling.parquet
fit_reconcile_alpha.py  alpha by rolling origin               -> reconcile_calibration.json (0.75)
fit_v3_uncertainty.py   team / share / availability params    -> v3/uncertainty/manifest.json
calibrate_v3_distribution.py
                        scores the production simulator       -> v3_fantasy_interval_calibration.json
                                                    selected_distribution_mode = joint_bootstrap

publish.py              publish() - one transaction
  predict.project_season()      veterans / rookies / replacement  -> per-stat pred_pg
    composition.compose_board()   anchors, reconcile, concentration,
                                  TD clips, season totals, identities
  validate_projection_contract()
  _git_revision()               captured BEFORE anything is written
  simulate.write_simulation_outputs()   generative + joint bootstrap
  fantasy_points.compute_fantasy_points()
  draft_assistant.prepare.export_draft_data()   vorp -> tiers -> overlay
                                                    -> players_2026.json
                                                       (atomic swap, manifest last)
```

Entry point:

```bash
python -m src.projection.publish --season 2026
```

Artifacts are staged in a temp directory and atomically swapped, with the run manifest moved last as the commit marker; an interrupted run is rejected rather than half-published.

## Related decisions

- [`SIMULATION_MODE_2026-08-26.md`](decisions/SIMULATION_MODE_2026-08-26.md) — retiring the interim arm, the coverage investigation, and the calibration steps
- [`QB_TEAM_VOLUME_SHARE_2026-08-26.md`](decisions/QB_TEAM_VOLUME_SHARE_2026-08-26.md) — the 1.000 → 0.941/0.942 revert and its paired-test evidence
- [`V3_PROBABILISTIC_PIPELINE.md`](decisions/V3_PROBABILISTIC_PIPELINE.md) — v3 scaffolding, the means backtest, and why the point engine is still v1
