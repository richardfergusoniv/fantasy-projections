# The Projection Pipeline

How raw NFL play-by-play becomes a draft-day rank sheet: every data source, every engineered feature, the math behind each model, and the reconciliation steps that turn independent stat predictions into one coherent board.

**Stage flow:** Raw NFL data &rarr; Feature engineering &rarr; Role-rate + team models &rarr; Rookie rule + corrections &rarr; Compose + reconcile &rarr; Season totals &rarr; Fantasy points &rarr; VORP + tiers

---

## 1. Raw data sources

Everything nflverse publishes gets pulled through `src/ingest/sources.py`, cached, and loaded into a local SQLite database. Depth-chart eligibility comes from a hand-curated file the model never edits itself.

### Ingested via `src/ingest/sources.py` &rarr; SQLite

| Source | nflverse call | Table | What it's used for |
|---|---|---|---|
| Play-by-play | `import_pbp_data` | `pbp` | Attempts, rush/red-zone/air-yards, EPA, opponent features |
| Weekly box scores | `import_weekly_data` | `weekly` | Per-player-week stats; falls back to pbp-derived stats before the official 2025 file publishes |
| Participation | nflverse-data release | `participation` | Personnel/formation data, 2016+ |
| Snap counts | `import_snap_counts` | `snap_counts` | Offensive snap %, keyed by pfr_player_id |
| Depth charts | `import_depth_charts` | `depth_charts` | Preseason/weekly charts, 2016&ndash;2026 (two schema eras) |
| Seasonal rosters | `import_seasonal_rosters` | `seasonal_rosters` | Season snapshots &mdash; age, position, rookie_season |
| Weekly rosters | `import_weekly_rosters` | `weekly_rosters` | Weekly status (ACT/INA/DEV/RES/CUT/RET) &mdash; eligibility & role-zero logic |
| Schedules | `import_schedules` | `schedules` | Game-by-game opponent list, for opponent-strength features |
| Next Gen Stats | `import_ngs_data` | `ngs_*` | Loaded, not directly used in feature build |
| FTN charting | `import_ftn_data` | `ftn` | 2022+ charting data |
| PFR advanced | `import_weekly_pfr` / `import_seasonal_pfr` | `weekly_pfr_*` / `seasonal_pfr_*` | 2018+ advanced stats |
| Draft picks | `import_draft_picks` | `draft_picks` | Draft capital &mdash; rookie path only |
| Injuries | `import_injuries` | `injuries` | Weekly injury report &mdash; feeds injury_durability_rate |
| Combine | `import_combine_data` | `combine_data` | Athletic testing &mdash; rookie tier |
| Player IDs | `import_ids` / `import_players` | `players` | gsis/pfr crosswalk, career position, rookie_season |

### Hand-curated &mdash; `src/depth_chart/`

| File | Purpose |
|---|---|
| `starters_2026.csv` | Curated base depth chart &mdash; eligibility / room membership / formation role, never auto-edited |
| `live_depth_2026.csv` | Derived chart after applying injury events |
| `status_overrides_2026.csv` | Dated games-caps / zero overrides from Sleeper status ingest |

---

## 2. Feature engineering

Raw tables get aggregated to one row per player-season (2016&ndash;2025) in `src/projection/features.py`, built on top of `data_prep.py`, `ol_quality.py`, and `depth_history.py`.

**Usage shares** (`features.py` &middot; `FEATURE_COLS`) &mdash; volume as a share of team/position opportunity, not raw counts, so it's portable across role changes:
- `carry_share`, `target_share`
- `rz_carry_share`, `rz_target_share`
- `rz_carry_monopoly`, `rz_target_monopoly`
- `air_yards_share`, `adot`
- `peak_receiving_yards_share`

**Team & scheme context** (`data_prep.py` &middot; `OC_METRICS`) &mdash; prior-season opponent and offensive-scheme signal, decoupled from the player's own performance:
- `opp_def_pass_epa_prior`, `opp_def_rush_epa_prior`
- `pass_oe`, `pass_oe_neutral`, `neutral_sec_per_play`
- `play_action_rate`
- `personnel_11_rate` / `12_rate` / `21_rate` / `other_rate`
- `qb_designed_run_rate`

**Player quality & health** (`data_prep.py`, `ol_quality.py`) &mdash; snap trust, protection quality, durability, and aging; these move slowly across seasons:
- `snap_pct` (mean offense_pct)
- `ol_pass_protection_score`, `ol_run_blocking_score`
- `ol_confidence_low_churn`
- `injury_durability_rate` = (missed games + 0.4&times;flagged-but-played) / team games
- `age`, `career_year`

**Role stability & depth** (`transitions.py`, `depth_history.py`) &mdash; what the player was already doing, and where the depth chart says he sits; the strongest priors for next season:
- `prior_{stat}_pg` lag features, every target stat
- `prior_role_rate`, `prior_role_rate_3y` for stable stats
- `target_depth_rank` (availability feature)
- `DEPTH_TIER_COLUMN` (volume/role feature)
- Two-season blend: `w = min(1, games / 8)`, applied to RB/WR/TE only

---

## 3. What each model actually computes

Six distinct mathematical approaches, matched to how much data and how much smoothness each target has &mdash; not one model family applied everywhere.

### Role-rate & share models &mdash; LightGBM, 24 models
One `LGBMRegressor` per (position, stat) &mdash; e.g. `QB_passing_yards`, `RB_carries` &mdash; fit on season-N&rarr;N+1 transition pairs (2021&ndash;2025). Heavily regularized because each cell has only a few hundred rows:

```
n_estimators=100, max_depth=3, num_leaves=8, min_child_samples=10,
subsample=0.8, colsample_bytree=0.8, reg_alpha=reg_lambda=0.1
```

Target is a *per-eligible-week* rate (season total &divide; rostered, non-reserve weeks, zeros included), not per-appearance. Some receiving stats are reframed to predict **share of team volume** instead of an absolute rate, composed with the team model downstream.
&mdash; `src/projection/train.py :: fit_one()`

### Team-total anchors &mdash; RidgeCV, 4 models
Team-season passing yards, pass attempts, carries, rushing yards &mdash; smooth, autocorrelated targets on a tiny dataset (32 teams &times; 3&ndash;4 transitions &asymp; 100&ndash;128 rows), so linear regularized regression beats gradient boosting.

```
RidgeCV(alphas = logspace(-2, 3, 20))   # regularization chosen by leave-one-out CV
```
&mdash; `src/projection/train.py :: fit_team_total()` &rarr; `models/team_*.joblib`

### Availability (games played) &mdash; LightGBM, 4 models
One model per position, predicting next season's games played from usage history plus the *target* season's preseason depth chart. Feeds the games multiplier used to turn a per-game rate into a season total.
&mdash; `src/projection/train.py :: fit_availability()` &rarr; `models/{POS}_games.joblib`

### Rookie path &mdash; rule-based, not ML
Rookies lack the trailing-history features every other model depends on, so this is a two-parameter linear fit plus an explicit vacancy adjustment, not a trained regressor:

```
rate = intercept + slope * log(effective_pick)      # OLS via np.linalg.lstsq, per position/stat
rate_adj = rate * clip(player_vacated_share / bucket_historical_vacated_share, VACATED_CLIP)
```

Upward scaling (taking &gt;1&times; the bucket rate) only applies to rookies a curated depth chart lists as starter/committee. Rookie claims on a team's vacated volume are netted against what veteran models already claimed for the same opening, split proportionally when multiple rookies compete for it.
&mdash; `src/projection/rookies.py`

### Elite-shrinkage correction &mdash; OLS, WR/TE receiving yards only
A single-parameter post-hoc correction fit through cross-fitted, leave-one-transition-out residuals, applied only above a per-position ypg knot (WR: 60, TE: 50):

```
residual ≈ beta * max(0, observed_ypg - knot)        # OLS through the origin
```

Ships only if it clears three gates: &ge;15 rows above the knot, positive beta, and cross-season consistency &ge; 2.0 SE. Applied additively at inference, capped at 8.0 ypg.
&mdash; `src/projection/corrections.py`

### Uncertainty intervals &mdash; empirical quantiles, 80% band
**Veterans** &mdash; additive: pool strictly-forward, cross-fitted residuals (actual &minus; pred) across rolling backtest folds, take the empirical 10th/90th percentile per (position, stat):

```
pred_low/high = pred_pg + quantile(residuals, 0.10 / 0.90)
```

**Rookies** &mdash; multiplicative, since there's no naive baseline to draw residuals from: empirical p10/p90 of `actual_pg / bucket_mean_pg` per (position, round bucket, stat), applied as a ratio to the point estimate. Cells under the row-count minimum (30 veteran / 20 rookie) get a `low_n_flag`.
&mdash; `src/projection/backtest.py`, `rookies.py` &middot; `models/interval_residuals.csv`

---

## 4. Composing one coherent board

Every stat above is predicted independently. This stage is what keeps a team's players from collectively over- or under-claiming the team's own predicted volume &mdash; without silently redistributing playing time between teammates.

Team totals set the ceiling; players are pulled toward it, not overwritten by it &mdash; each team-position group's summed volume moves halfway to the anchor, individual ordering inside the group is untouched.

```
Team-total anchor (RidgeCV, per team)
        |
        |  reframed stats: share x team total
        v
Player A pred_pg   Player B pred_pg   Player C pred_pg   (independent predictions)
        \                |                /
         \               |               /
          v              v              v
              reconcile_team_volume
     team sum(players) -> pulled halfway to anchor, alpha=0.5
                          |
                          v
   concentration -> TD-rate clip -> child<=parent stats -> season totals
```

1. **Attach team anchors** &mdash; every player row gets its team's predicted pass attempts, passing yards, carries, and rushing yards for the season.
2. **Compose reframed receiving stats** &mdash; a player's predicted share of team passing yards is normalized against the team's summed shares (capped so shares can't exceed 1), then multiplied by the team total. Rookie shares enter the same denominator, so an incoming rookie squeezes veteran shares.
3. **Reconcile team volume &mdash; &alpha; = 0.5** &mdash; for attempts, passing yards, carries, and rushing yards, each team's summed player-level volume is pulled halfway toward the team anchor. QB rooms protect the tier-1 starter; RB rooms scale the whole committee together. `TEAM_RECONCILE_ALPHA` in `contracts.py`.
4. **Concentration calibration** &mdash; within a team-position room, how volume concentrates onto the top player is sharpened or flattened against a fitted calibration, with exact conservation of the room's total.
5. **TD-rate & stat-identity constraints** &mdash; pass-TD-rate and rush-TD-per-carry get clipped to historical bands; completions &le; attempts and receptions &le; targets are enforced.
6. **Season totals & team identities** &mdash; `pred_season = pred_pg * projected_games`. A final geometric split forces team-level season identities to hold (receiving_yards == passing_yards, receptions == completions).

---

## 5. Scoring, VORP, and tiers

The composed board is per-stat predictions, not fantasy points. Everything from here is deterministic math on top of the model output &mdash; no further ML.

### Fantasy points &mdash; weighted dot product

| Stat | Weight |
|---|---|
| Passing yards | 1 / 25 |
| Passing TD | 4 |
| Interception | &minus;2 |
| Rushing yards | 1 / 10 |
| Rushing TD | 6 |
| Receiving yards | 1 / 10 |
| Receiving TD | 6 |
| Reception | 0.5 |

```
fantasy_pts = sum( pred[stat] * weight[stat] )
```

Half-PPR, 4-pt passing TD. Fumbles lost and 2-pt conversions aren't modeled upstream, so they're absent by construction, not silently zeroed. Low/high bands are scored the same way per endpoint, correctly flipped for interceptions' negative weight.
&mdash; `src/projection/fantasy_points.py`

### VORP & tiers

```
replacement_rank = roster_math(position) * (17 / mean_projected_games)
vorp = vorp_input_pts - replacement_pts        # signed -- not floored at zero
```

Roster math combines required starters (QB1/RB2/WR3/TE1) with weighted flex demand; the availability factor deepens the replacement level for positions whose starters miss more games. Flooring VORP at zero was tried and rejected &mdash; it collapsed most of the board to ties. TE gets a shape correction blended toward a historical surplus curve; QB/RB/WR are left alone.
&mdash; `src/draft_assistant/vorp.py`

Tiers break wherever the point gap to the next player exceeds an absolute or relative threshold (`DEFAULT_TIER_GAPS`, ~3&ndash;4% relative) &mdash; computed separately for the whole board, within-position, and the RB/WR/TE flex pool.
&mdash; `src/draft_assistant/tiers.py`

---

## 6. The file chain, end to end

```
train.py                 fits per-stat + team-total models              -> models/*.joblib
backtest.py               builds residual quantiles, forward coverage    -> interval_residuals.csv
predict.py                project_season() orchestrates:
  veterans.py / rookies.py / replacement.py                              -> raw per-stat pred_pg rows
  composition.py          compose_board()
    team_reconcile.py     team anchors, share composition, volume reconcile,
                           concentration, TD-clip, stat constraints, season totals
publish.py                publish() orchestrates:
  fantasy_points.py       compute_fantasy_points()                       -> fantasy_points_<season>.csv
  draft_assistant/prepare.py   export_draft_data()
    vorp.py                add_vorp_columns()
    tiers.py                add_tier_columns()
                                                                           -> draft_assistant/data/players_<season>.json
```

Entry point: `python -m src.projection.publish --season 2026`. Output is staged in a temp directory and atomically swapped in, with the run manifest as the final commit marker.
