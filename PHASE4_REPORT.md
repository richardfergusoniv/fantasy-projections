# Phase 4 — Player Projection Models (LightGBM, opportunity x efficiency)

## What's built

- `src/projection/data_prep.py` — shared player-week/season usage table
  (2016-2025, REG season only), team-season pbp totals, red-zone usage,
  snap %.
- `src/projection/ol_quality.py` — team-season OL quality feature, built
  from Phase 2's pooled coefficients weighted by actual snap share.
- `src/projection/features.py` — the player-season feature table (QB/RB/WR/TE,
  opportunity + efficiency-conditioning features + target per-game rates).
- `src/projection/rookies.py` — the separate rookie rule-based path.
- `src/projection/transitions.py` — season N -> N+1 pairing logic shared by
  training and backtesting.
- `src/projection/train.py` — fits and saves the production LightGBM
  models to `models/*.joblib` (gitignored, binary artifacts).
- `src/projection/backtest.py` — the 2025 holdout backtest (run via
  `python -m src.projection.backtest`).
- `src/projection/predict.py` — Phase 5's entry point
  (`project_season(conn, target_season)` or
  `python -m src.projection.predict --season <year>`).

Run order: `python -m src.projection.train` (writes `models/`), then
`python -m src.projection.backtest` for the honesty check (does not touch
`models/` — it fits its own held-out copies internally). `predict.py`
reads the saved `models/` artifacts, so `train.py` must run first.

## Granularity: player-SEASON, not player-season-week

Chosen because the target stats are explicitly per-GAME rates and every
upstream Phase 2/3 signal (OC tendencies, OL quality) is already
season-level — a weekly grain would require inventing week-level share/OL
features that don't exist anywhere upstream, adding noise without adding
real signal for what is fundamentally a next-SEASON projection task.

## Train/predict framing

Each training row is (player's season N feature vector) -> (player's
season N+1 per-game rate). This is a genuine forward projection, not
same-season leakage: everything in the feature vector is fully observed by
the end of season N, and the label is strictly season N+1's outcome.

**Judgment call, stated plainly**: season N's own observed
`oc_tendency_profiles` / OL-quality features are used as season N+1's
feature vector — i.e., "this team's scheme/OL context last year" is the
proxy for "this team's scheme/OL context next year," because next year's
actual observed values obviously don't exist yet at prediction time for a
genuinely future season. This is what any real preseason projection has to
do. `oc_tendency_profiles.inherited_*` (Phase 3) exists specifically to
give a better proxy when a team's OC situation is *known* to be changing
entering the target season — but substituting that in is a Phase 5 concern
(it requires knowing, at call time, which teams have a new play-caller for
the season being projected), not something `predict.py` does automatically.
This is flagged in `predict.py`'s docstring for Phase 5 to pick up.

## 2021-2025-only training scope (explicit scope decision)

**Chosen: option (a) — restrict ALL training (every position/stat model,
not just OL-conditioned ones) to the 2021-2025 window**, i.e. transitions
2021->22, 22->23, 23->24, 24->25. 2016-2020 has coordinator-tendency
features but no OL quality at all (Phase 2's window starts 2021). The
alternative considered — fit opportunity-only models back to 2016 and a
separate OL-conditioned model for 2021+ — was rejected: it would mean
building and maintaining two full model families per stat instead of one,
for a project whose own architecture explicitly wants efficiency
conditioned on OL quality as a first-class input, not an optional add-on.
Mixing eras into one model (leaving OL columns NaN for ~2016-2020 rows)
was also rejected, since LightGBM would then have to learn to ignore those
columns for 40%+ of rows rather than getting to use them consistently.

**Honest cost of this choice**: with only 4 total transitions available
(2021->22 ... 2024->25) and 3 used for training (2024->25 held out for the
backtest), the production models are fit on somewhere between ~250 rows
(QB) and ~660 rows (WR) — see the per-model row counts in `train.py`'s
printed output. This is a small training set for a gradient-boosted tree
model. It is mitigated by very shallow, heavily regularized trees (see
Hyperparameters below), but the user should treat these as a first-pass
model, not a mature one — more transitions will accumulate one season at a
time as new data lands.

## Opportunity features

Team volume (`oc_tendency_profiles`, keyed by season+team, observed —
not `inherited_*` — for a completed season): `pass_oe`, `pass_oe_neutral`,
`neutral_sec_per_play`, `play_action_rate` (NaN pre-2022, real Phase 3
gap, not imputed), and the four personnel-rate columns.

Player share, computed against team-season pbp totals (`pass_attempt`/
`rush_attempt` counts by `posteam`, REG season, from `pbp` directly rather
than summing `weekly` — avoids double-counting/mislabeling from the 2025
pbp-fallback rows' missing team column):
- `carry_share` = player season carries / team season rush attempts
- `target_share` = player season targets / team season pass attempts
- `rz_carry_share` / `rz_target_share` = same, restricted to `pbp.yardline_100 <= 20`
- `snap_pct` = average offensive snap % across the season, joined via
  `players.pfr_id` -> `snap_counts.pfr_player_id` (99.6% match rate per
  Phase 1's crosswalk finding).

**Caveat**: shares are computed against the player's *resolved season
team* (the team with the most active weeks — see `data_prep.season_aggregate`),
not per-week. A player traded mid-season has their shares computed against
only one team's totals, understating/misattributing volume for the other
team's stretch. Rare in this position group, not specifically quantified,
flagged here rather than silently accepted as correct.

## Efficiency features conditioned on OL quality and scheme

**OL quality** (`src/projection/ol_quality.py`): for each team-season, every
lineman who logged OL-position (`G`/`T`/`C`/`OT`/`OG`/`OL`) offensive snaps
in `snap_counts` that season is joined to `ol_coefficients_pooled` (their
pooled coefficient) + `ol_season_effects_pooled` (that season's fixed
effect) to get a fitted per-player score, then averaged across the team's
OL, weighted by each player's share of the team's total OL offensive
snaps that season — this is the "who actually blocked for them" weighting
the spec asked for, not a league-wide average. Produces
`ol_pass_protection_score` and `ol_run_blocking_score` per team-season
(2021-2025 only — Phase 2's window).

**`confidence_flag` handling (judgment call, reasoning stated either way)**:
`ol_coefficients_pooled.confidence_flag` flags *individual* players whose
credit within a low-churn (fixed-starting-five) team-season block isn't
statistically identified — ridge has no information to separate them from
their linemates for those plays. The **team-season aggregate** is used
regardless of this flag, because arbitrary mis-splits of credit within a
collinear block largely cancel out in a weighted sum (if player A is
over-credited at player B's expense, their *combined* contribution to the
team total is roughly unaffected — the ridge fit still had to explain the
same total outcome over those shared plays). What this reasoning does
**not** cover: a systematic bias in how well the whole unit is captured
relative to another team's whole unit — that's a real, unaddressed
limitation of the aggregate, not something this weighting scheme fixes.
To let the downstream model account for it anyway, `ol_team_season_churn`'s
own team-level `confidence_flag` (not the per-player rollup) is carried
through as a binary `ol_confidence_low_churn` feature, so LightGBM can
learn to trust the aggregate less on the 4/160 (2.5%) team-seasons flagged
`unit_level`.

**Scheme conditioning**: reuses the same `oc_tendency_profiles` columns as
the opportunity block (`pass_oe`, personnel rates, pace) rather than
building a duplicate copy — scheme affects both how much opportunity a
player gets and how efficiently they convert it, and there's no principled
reason to have two separate feature sets for the same underlying columns.

## Target stats

Per-game rate = season total / `games_played`, where `games_played` is
counted on the position-relevant *usage* stat (QB: `attempts > 0`;
RB/WR/TE: `carries > 0` or `targets > 0`) rather than "any row exists in
`weekly`" — a player can have an all-zero row for an inactive/injured week.

| Position | Stats |
|---|---|
| QB | attempts, completions, passing_yards, passing_tds, interceptions |
| RB | carries, rushing_yards, rushing_tds, targets, receptions, receiving_yards, receiving_tds |
| WR / TE | targets, receptions, receiving_yards, receiving_tds |

## Rookie handling (hard project rule, separate path)

For a player whose draft season equals their first season with any active
NFL week (`src/projection/rookies.py::identify_rookie_seasons`), the
veteran feature vector is structurally unavailable (no prior season), so
this is a **distinct rule-based path**, not a smaller LightGBM model:

- **Inputs**: draft round (bucketed `round_1` / `round_2_3` / `round_4_7` /
  `undrafted`) and pick, plus **vacated team opportunity** — the share of
  the player's new team's season-(N-1) carries/targets that belonged to
  players who did NOT have an active week for that same team in season N
  (retired, cut, signed elsewhere, or simply lost the role — this doesn't
  distinguish why, only that the volume moved). This uses only
  season-(N-1) data plus who's active on the roster in season N — never
  the rookie's or anyone else's season-N production — so no forward
  information leaks into a genuine preseason projection.
- **No college production, no same-season NFL stats** — enforced by
  construction: college production isn't modeled anywhere in this DB, and
  the rookie path never touches the `weekly`/`pbp` rows for season N at
  all, only season N-1's departures.
- **Model**: historical mean per-game rate for rookies in the same
  (position, draft-round-bucket), fit only on seasons strictly before the
  target season, scaled by the ratio of this player's team's vacated
  opportunity to the bucket's own historical average vacated opportunity
  (clipped to 0.3x-2.5x to avoid small-sample blowups). This is
  deliberately simple — see "Why rule-based, not LightGBM" below.
- **Every rookie output row is flagged** `source='rookie_rule'`,
  `low_confidence=True` in `predict.py`'s combined output — never mixed in
  as an equal-confidence number alongside veteran-model rows.

**Why rule-based, not LightGBM**: pooling all rookie-seasons 2016-2024
across QB/RB/WR/TE gives only 629 total rows (74 QB, 179 RB, 264 WR, 112
TE — see counts below), split further across 3-4 round buckets per
position. A gradient-boosted model with that little data, on a feature
vector this thin (round, pick, one vacated-share number), would either
memorize noise or degenerate to something functionally identical to a
bucket mean anyway — the rule-based bucket-mean-times-scale approach is
more transparent about exactly that limitation instead of dressing it up
as a fitted model.

Rookie sample sizes (all seasons 2016-2025, before any train/test split):

| Position | round_1 | round_2_3 | round_4_7 (incl. UDFA*) | total |
|---|---|---|---|---|
| QB | 33 | 14 | 27 | 74 |
| RB | 13 | 49 | 117 | 179 |
| WR | 42 | 90 | 132 | 264 |
| TE | 11 | 41 | 60 | 112 |

*`draft_picks` only covers drafted players — undrafted rookies who make an
active roster are not represented in this rookie path at all (they'd need
a `round_bucket = 'undrafted'` baseline, which exists in the code but has
zero training rows in this window since `identify_rookie_seasons` only
matches players present in `draft_picks`). This means **undrafted rookies
currently fall through both paths silently absent from `predict.py`'s
output** — a real gap the user should weigh in on (options: extend
`identify_rookie_seasons` to also catch undrafted first-year players via
`players.rookie_season` and use the `undrafted` bucket with a distinct,
even-lower-confidence baseline, or accept that UDFAs are out of scope for
now since they're rarely fantasy-relevant in their rookie year).

## Backtest: 2025 holdout, model vs. naive baseline

Train on 2021->22, 22->23, 23->24 transitions; predict 2025 per-game rates
from 2024 features; naive baseline = 2024's own per-game rate carried
forward unchanged. Ground-truth caveat (same one from Phase 1, restated
because the backtest's "truth" inherits it): 2025's `weekly` rows are the
pbp-fallback aggregation, not nflverse's official release — this doesn't
change the backtest's structure (model and naive baseline are scored
against the identical numbers) but the ground truth itself carries the
fallback methodology's own caveats (no fumbles/2pt logic, a close-but-not-
byte-identical replica of nflverse's own aggregation).

| Position | Stat | n | Model MAE | Naive MAE | Model wins? |
|---|---|---|---|---|---|
| QB | attempts | 61 | 6.88 | 7.43 | **yes** |
| QB | completions | 61 | 4.21 | 4.77 | **yes** |
| QB | passing_yards | 61 | 49.03 | 55.82 | **yes** |
| QB | passing_tds | 61 | 0.419 | 0.522 | **yes** |
| QB | interceptions | 61 | 0.259 | 0.321 | **yes** |
| RB | carries | 94 | 2.65 | 2.75 | **yes** |
| RB | rushing_yards | 94 | 12.58 | 13.63 | **yes** |
| RB | rushing_tds | 94 | 0.150 | 0.168 | **yes** |
| RB | targets | 94 | 0.815 | 0.808 | **no** |
| RB | receptions | 94 | 0.692 | 0.674 | **no** |
| RB | receiving_yards | 94 | 5.94 | 6.04 | yes (marginal) |
| RB | receiving_tds | 94 | 0.061 | 0.064 | yes (marginal) |
| WR | targets | 170 | 1.16 | 1.21 | **yes** |
| WR | receptions | 170 | 0.849 | 0.901 | **yes** |
| WR | receiving_yards | 170 | 11.42 | 13.30 | **yes** |
| WR | receiving_tds | 170 | 0.127 | 0.159 | **yes** |
| TE | targets | 97 | 0.788 | 0.870 | **yes** |
| TE | receptions | 97 | 0.689 | 0.768 | **yes** |
| TE | receiving_yards | 97 | 7.58 | 8.66 | **yes** |
| TE | receiving_tds | 97 | 0.129 | 0.137 | **yes** |

**Plain statement, as required**: the model beats the naive carry-forward
baseline on **18 of 20** position/stat combinations. **RB targets and RB
receptions are the two exceptions — the model loses to naive there** (MAE
0.815 vs 0.808 for targets, 0.692 vs 0.674 for receptions; both losses are
small in absolute terms but real). A plausible explanation: RB receiving
work is heavily role-dependent (a specific "third-down back" role can
persist almost unchanged year-to-year for a given player independent of
team-level share features), so last year's own rate is already a strong
predictor and the model's added features aren't adding enough signal to
beat it — this is a hypothesis, not verified further given the phase's
time budget, and is exactly the kind of result the project owner asked to
see stated plainly rather than buried.

Rookie-path MAE (no naive-baseline comparison exists for rookies — there's
no prior season to carry forward):

| Position | Stat | n | Model MAE |
|---|---|---|---|
| QB | attempts | 7 | 9.20 |
| QB | completions | 7 | 4.67 |
| QB | passing_yards | 7 | 53.08 |
| QB | passing_tds | 7 | 0.349 |
| QB | interceptions | 7 | 0.229 |
| RB | carries | 22 | 4.73 |
| RB | rushing_yards | 22 | 23.25 |
| RB | rushing_tds | 22 | 0.222 |
| RB | targets | 22 | 1.37 |
| RB | receptions | 22 | 1.01 |
| RB | receiving_yards | 22 | 8.64 |
| RB | receiving_tds | 22 | 0.075 |
| WR | targets | 25 | 1.52 |
| WR | receptions | 25 | 0.967 |
| WR | receiving_yards | 25 | 12.41 |
| WR | receiving_tds | 25 | 0.146 |
| TE | targets | 11 | 1.64 |
| TE | receptions | 11 | 1.25 |
| TE | receiving_yards | 11 | 12.33 |
| TE | receiving_tds | 11 | 0.112 |

n is tiny (7-25 rookies per position in the single 2025 test season) —
these numbers are directional at best, not a reliable performance
estimate. QB attempts MAE (9.2) is notably worse than the veteran model's
(6.9), consistent with rookie QB playing time being close to binary
(starter vs. clipboard-holder) and not well captured by a continuous
draft-round bucket average.

## Hyperparameters

Not tuned beyond a fixed, deliberately conservative choice, given the
small per-model training sets (250-660 rows):
`n_estimators=100, learning_rate=0.05, max_depth=3, num_leaves=8,
min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
reg_lambda=0.1` (see `src/projection/train.py::LGBM_PARAMS`). These were
chosen to bias toward under-fitting rather than over-fitting on a small
sample, not selected via any cross-validation search — a proper
hyperparameter sweep (even a small grid on `max_depth`/`num_leaves`/
`learning_rate` via the existing transition pairs as CV folds) is future
work if the user wants to push backtest performance further.

## Addendum: QB rushing added (post-launch)

A Sleeper-projection comparison (`src/comparison/sleeper_compare.py`) found our
QB fantasy points were pure-passing - `TARGET_STATS['QB']` never included
`carries`/`rushing_yards`/`rushing_tds`, despite the raw totals and share
features already being computed generically for every position. This
systematically underrated every mobile/dual-threat QB (Lamar Jackson, Josh
Allen, Jayden Daniels, Kyler Murray, Caleb Williams showed the largest
Sleeper-higher deltas in the whole dataset). Added the three stats and
retrained; other positions/stats are bit-for-bit unaffected (verified: their
backtest MAE numbers are unchanged).

New backtest results (2025 holdout, same train/test split as above):

| Stat | n | Model MAE | Naive MAE | Model wins? |
|---|---|---|---|---|
| carries | 61 | 1.29 | 1.50 | **yes** |
| rushing_yards | 61 | 8.47 | 7.30 | **no** |
| rushing_tds | 61 | 0.115 | 0.149 | **yes** |

Stated plainly per the project's own rule: **QB rushing_yards loses to the
naive baseline** (small-sample QB rushing volume is evidently closer to a
carry-forward-last-year's-rate situation than the model currently captures) -
the third QB loss alongside the pre-existing RB targets/receptions losses.

Effect on the star QBs that motivated this: the Sleeper-delta roughly halved
for all five (Allen -8.3 -> -3.9, Lamar -7.9 -> -4.5, Daniels -7.1 -> -3.5,
Caleb Williams -7.6 -> -3.2 pts/game); Kyler Murray improved less (-7.3 ->
-6.1) and remains a real residual gap.

## Addendum 2: rookie QB over-projection root-caused and fixed

The issue flagged above (Kaliakmanis-style deep-bench QBs at implausible
starter-level volume) had two distinct, compounding causes, both in
`rookies.py`/`predict.py`:

1. **Wrong vacated-opportunity proxy.** `predict_rookies` scaled every
   non-RB rookie's projection by `vacated_target_share` - the team's
   WR/TE/RB *receiving-corps* turnover. For a QB, that's a category error:
   whether a backup QB plays has nothing to do with how many receivers left.
   Concretely, WAS's 2026 `vacated_target_share` (0.52, driven by real
   WR/TE departures) was scaling Kaliakmanis's whole passing line toward
   starter volume, even though WAS's actual 2026 starter (Jayden Daniels)
   never left. Fixed: `team_vacated_opportunity` now also computes
   `vacated_attempts_share` (how much of the team's PASSING volume belonged
   to departed QBs specifically - `attempts` is ~exclusively a QB stat, so
   this isolates QB-room turnover), and `predict_rookies` uses it for QB
   instead of the receiving-corps proxy.
2. **No depth-chart check on the boost, mirroring the Phase 6 veteran bug.**
   Even with the QB-correct proxy, a real vacancy (e.g. Tua Tagovailoa
   leaving MIA for ATL) would boost ANY rookie/UDFA QB on that roster
   toward the opening's volume, with no check on whether that specific
   player is actually a candidate for it - Mark Gronowski (an anonymous MIA
   UDFA) got scaled to 45 attempts/game purely because MIA's real starter
   departed. Fixed the same way Phase 6 fixed it for team-changing
   veterans: `predict_rookies` now takes the curated
   `src/depth_chart/starters_2026.csv` and only allows the upward half of
   the scale (>1.0) for a rookie the table actually lists for their
   (team, position); everyone else still gets scaled down for a
   below-average situation, capped at 1.0 on the upside.

Verified: Kaliakmanis 27.9 -> 5.7 attempts/game, Gronowski 45.4 -> 18.2
(still capped-scale, see residual limitation below), Cole Payton/Garrett
Nussmeier/Behren Morton (all uncurated) now correctly land on the same
unscaled bucket mean rather than inflated individually. Sleeper-comparison
correlation improved overall (0.863 -> 0.888) and specifically for QB
(0.863 -> 0.912, mean abs delta 2.69 -> 2.16 - better than even the
pre-rushing-addition baseline of 2.17).

## Addendum 3: rookie QB survivorship bias fixed via Sleeper play-probability

The residual limitation flagged above (the UDFA-QB bucket mean itself is
survivorship-biased, since `fit_rookie_baselines` only sees rookie-seasons
with `games_played > 0`) is now fixed for QB, using the same idea the user
proposed: borrow Sleeper's own judgment about whether a player will see the
field at all, since that's real depth-chart/beat-reporter knowledge this
project has no other free source for.

**First attempt (wrong), corrected before shipping**: Sleeper's `gp`
(games-played) field looked like the natural "play probability" signal
(`gp / 17`), but checking the actual distribution showed `gp` is not a
real per-player estimate - 9370 of 9402 players with a `gp` value have
EXACTLY `gp=18`, including rookie QBs with zero other projected stats at
all (Athan Kaliakmanis, Mark Gronowski: `gp=18`, no `pass_att`, no
`pts_half_ppr`, nothing else). `gp` is a bookkeeping default for anyone
Sleeper tracks for ADP, not an expected-playing-time estimate - using it
as designed would have been a no-op (every rookie multiplied by ~1.0).

**What actually works**: whether Sleeper bothers projecting real pass-
attempt volume for the player at all. `fetch_sleeper_play_probability`
(`src/comparison/sleeper_compare.py`) now re-reads the raw projections
JSON directly and checks for the presence of `pass_att` in that player's
record - present -> `play_prob=1.0`, absent -> `play_prob=0.05` (same
value/reasoning as the existing no-match default). `predict_rookies`
(`rookies.py`) multiplies a QB rookie's whole predicted line by this
factor. Needs the player's real name to join to Sleeper (`identify_
target_season_rookie_class` now carries a `name` column specifically for
this - the 2026 draft class's own `gsis_id` is a known-bad placeholder per
Phase 5, so name is the only usable join key for this year's rookies);
gracefully no-ops (with a printed warning) if the Sleeper fetch fails, and
is skipped entirely for `backtest.py`'s historical rookie evaluation
(that path's `rookie_df` has no `name` column - by design, this is a
target-season prediction-quality fix, not a training-time change).

The resulting `qb_sleeper_play_prob` is carried into the final output
(`projections_<year>.csv`) so a reader can see exactly why a given rookie
QB's number was discounted, rather than a silently blended value.

**Verified**: Kaliakmanis attempts/game 5.67 -> 0.28, Gronowski 18.16 ->
0.91 - both now realistically near-zero instead of camp-arm-with-a-real-
shot numbers. Real veterans (Cousins, McCarthy) are untouched (rookie-only
fix). Sleeper-comparison correlation improved again: overall 0.888 -> 0.894,
QB specifically 0.912 -> 0.923 (mean abs delta 2.16 -> 1.72). The 2025
rookie backtest MAE table (re-measured after this fix, replacing the
addendum-2 numbers which reflected an interim mid-fix state) is now:

| Stat | n | Model MAE |
|---|---|---|
| attempts | 9 | 6.85 |
| completions | 9 | 3.46 |
| passing_yards | 9 | 42.60 |
| passing_tds | 9 | 0.25 |
| interceptions | 9 | 0.30 |
| carries | 9 | 1.34 |
| rushing_yards | 9 | 7.62 |
| rushing_tds | 9 | 0.17 |

All materially better than the original (pre-any-fix) numbers in the first
addendum's rookie table, not just less embarrassing on inspection.

## Judgment calls and caveats for the user to weigh in on

1. **2021-2025-only training window** (see above) — 3 training transitions
   is thin; performance and feature importances should be revisited as
   2026, 2027, etc. seasons accumulate more transitions.
2. **Season N's own features stand in for season N+1's conditions**
   (train/predict framing) — reasonable for a completed-season backtest,
   but Phase 5 needs to consciously decide whether/how to substitute
   `inherited_*` OC-tendency rows for teams with a genuinely new play-caller
   before projecting a season that hasn't happened yet.
3. **Undrafted rookies are currently absent from `predict.py`'s output
   entirely** — not silently defaulted to zero, just not projected at all.
   Flagged above; needs a decision before Phase 5 relies on complete rookie
   coverage.
4. **In-season trades**: a traded player's shares are computed against
   only their season-resolved team (most active weeks), not per-stint —
   understates/misattributes volume for the other stint. Not quantified,
   rare for this position group, but not fixed here.
5. **RB targets/receptions lose to the naive baseline** in the backtest —
   reported plainly above, not smoothed over. Worth a closer look (e.g. a
   role-persistence feature, like last season's own receiving-work share
   of the team's overall RB receiving work) if the user wants to chase
   that specific gap.
6. **OL-quality confidence-flag handling** (aggregate used regardless of
   individual `confidence_flag`, team-level churn flag passed through as a
   feature instead) is a considered judgment call, not a default — see the
   full reasoning above; the residual risk (systematic whole-unit bias
   between teams) is explicitly not addressed by this scheme.
7. **Hyperparameters are fixed, not tuned** — acceptable given the phase's
   priority order (backtest and rookie logic over hyperparameter search),
   but a real lever left on the table if backtest performance needs to
   improve further.
8. **2025 ground truth is the pbp-fallback aggregation**, not the official
   nflverse release — inherited from Phase 1, restated here since it's the
   backtest's actual test-set label.
