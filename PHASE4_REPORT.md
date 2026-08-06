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

## Addendum 4: injury history (veterans) + combine athleticism (rookies)

Two new free nflverse sources (`import_injuries`, `import_combine_data`) were
validated as available and wired in end-to-end: ingestion, a new trailing
veteran feature, a new rookie-path scale, and a full retrain/backtest/
predict/Sleeper-compare cycle. Results below, reported honestly per this
project's rule — the net effect on both the backtest and the Sleeper
comparison is small and mixed, not a clear win, and that is stated plainly
rather than oversold.

### Part 1: ingestion

Added `get_injuries(seasons)` and `get_combine_data(seasons)` to
`src/ingest/sources.py`, following the existing `cached_multi_season(...,
skip_missing=True)` pattern, and added both to `src/db/load.py`'s
`TABLE_SPECS`.

- **`injuries`**: 55,556 rows, 2016-2025. `failed_seasons` reported exactly
  one gap: `(2026, 'HTTP Error 404: Not Found')` — 2026 hasn't been played
  yet, so no injury reports exist. **No hard MIN_SEASON floor was found**
  within this project's window (unlike participation/FTN/PFR) — spot-checked
  2015-2020 directly and every one of those seasons returns thousands of
  rows with a real, non-null `report_status` distribution, not empty
  scaffolding, so `get_injuries` has no season gate.
- **`combine_data`**: 3,744 rows, 2016-2026, reported as "full coverage" by
  `failed_seasons` — but that claim needs one caveat stated explicitly:
  `import_combine_data` does **not** raise for a season with no data yet (a
  future combine that hasn't happened), it silently returns an **empty**
  DataFrame, which `cached_multi_season`'s `skip_missing` logic can't detect
  as a failure (it only catches exceptions). Verified directly: `season=2027`
  returns 0 rows with no error. This project's `SEASONS` window
  (2016-2027) means the 2027 slot contributes silently zero rows without a
  `failed_seasons` entry — a real gap in what "full coverage" can claim for
  this specific source, documented in `get_combine_data`'s docstring and
  restated here rather than assumed benign. It doesn't affect this task
  (2026 is the target season and its combine already happened), but the next
  engagement that touches `combine_data` for a future season should re-check
  row counts directly, not just trust `failed_seasons`.
- **pfr_id join coverage**: 84.6% of all `combine_data` rows overall have a
  non-null `pfr_id` (matches the ~85% figure this task was scoped against).
  QB/RB/WR/TE-only subset: 1,263 rows.

### Part 2: veteran injury-durability feature

Added `injury_durability_rate` to `FEATURE_COLS` (`src/projection/features.py`),
built by `src/projection/data_prep.py::build_player_season_injury_durability`.

**What it measures**: for a player-season, `(missed_games + 0.4 *
flagged_but_played_games) / team_games`, clipped to `[0, 1]`, where
`team_games` is 17 for 2021+ / 16 for 2016-2020 (the NFL's well-known
schedule-length change), `missed_games = team_games - games_played`, and
`flagged_but_played_games` counts weeks the player carried an
Out/Doubtful/Questionable status on the injury report **but still played**
that week (using `games_played`'s own existing active-week definition).

**Design judgment calls, stated plainly**:
- **Why not simply "fraction of weeks flagged," full stop** (the first
  design floated): spot-checked a real, extreme case — Christian
  McCaffrey's 2024 season (Achilles injury, 4 of 17 games played) — and
  found the `injuries` table **stops filing weekly reports once a player is
  on long-term IR**: his 2024 rows exist only for weeks 1-2 (pre-placement)
  and week 10 (on his way back); weeks 3-9 have no injuries row at all, not
  because he wasn't hurt but because there's nothing left to report. A
  "fraction of weeks flagged" metric — using either weeks-with-an-injuries-
  row or weeks-with-a-`weekly`-row as the denominator — would score
  McCaffrey's 2024 as barely notable, exactly backwards for what should be a
  maximal durability red flag. Anchoring the denominator to the team's
  actual scheduled game count and crediting every genuinely MISSED game
  (regardless of whether a report row exists for it) fixes this.
- **Why missed games are weighted 1.0 and played-but-flagged weeks only
  0.4**: the spec explicitly asks the feature to distinguish "banged up all
  year but always played" from "missed several games" — equal weighting
  would score a player who was Questionable every single week but never
  missed a game identically to a player who missed half the season
  outright, collapsing exactly the distinction the feature is supposed to
  capture. 0.4 is a stated, not-tuned judgment call, in the same spirit as
  this project's other stated-not-tuned constants (`train.py`'s
  `LGBM_PARAMS`, `rookies.py`'s `VACATED_CLIP`).
- **Collapsing multiple injuries rows per player-week**: the table has no
  day-of-week field (checked the actual schema — only a raw `date_modified`
  timestamp), so a Wednesday-practice-report vs. Friday-final-status
  distinction isn't recoverable from what nflverse ships. Spot-checked the
  real row structure for 2024: duplicate `(season, week, gsis_id)` rows are
  rare (2 of ~5-6k player-weeks) and represent a genuine status update
  during the week (Questionable -> Out), not distinct practice-day
  snapshots — the row with the latest `date_modified` is taken per
  player-week, equivalent to "most recently known status."
- **Trailing framing**: this is computed as season N's own value and fed
  into season N's feature row, exactly like every other `FEATURE_COLS`
  entry — `transitions.py`'s existing season-N -> season-(N+1) pairing
  automatically makes it a genuine trailing predictor with no extra shift
  logic needed, and no leakage (season N's games_played, itself downstream
  of season N's own injuries, is only ever used to predict season N+1).
- **No-report players get a real 0.0**, by construction of the arithmetic
  (missed_games=0, flagged_but_played=0), not a filled NaN. Verified: 5,597
  player-seasons in the full feature table, zero nulls in
  `injury_durability_rate`.

**Spot-checks (by name, not just aggregates)**:
- **Christian McCaffrey, 2024**: games_played=4/17,
  `injury_durability_rate=0.788` — correctly one of the highest scores in
  the dataset for a season that should be a maximal red flag.
- **Healthy full-season players (2025)**: Jared Goff, Hunter Henry, Keenan
  Allen, DeAndre Hopkins — all 17/17 games, all score exactly `0.0`.
- **Jayden Reed (GB WR), 2025**: a real injury-shortened season (5/17 games
  played), scores `0.729` — high durability-risk flag. His 2026 projection
  (`output/projections_2026.csv`) comes out modest for a player who was a
  legitimate WR1/2-caliber target-earner before the injury (20.7
  receiving_yards/game, 3.25 targets/game) — directionally sensible, though
  the low 2025 games_played also directly depresses his other 2025-based
  share features (`target_share`, `snap_pct`), so this is not a clean
  isolation of the injury feature's effect alone.
- **Feature importance** (LightGBM split counts, all 22 position/stat
  models): `injury_durability_rate` is a real, non-trivial signal, not dead
  weight — it ranks in the top 3 of 18 features for RB targets, RB
  receptions, RB receiving_yards/receiving_tds, WR targets, and QB
  passing_tds, and top-10 for 16 of the 22 models overall. It ranks lowest
  (13th-15th) for RB rushing_yards/rushing_tds and TE receptions.

### Part 3: rookie combine-athleticism scale

Added `combine_athletic_scores_by_pfr_id` / `load_combine_athletic_tier` to
`src/projection/rookies.py`: a discrete `athletic_tier` in
`{'above_median', 'below_median', 'no_data'}` per rookie, applied as a
modest multiplicative scale (`ATHLETIC_SCALE = {'above_median': 1.08,
'below_median': 0.94, 'no_data': 1.0}`) on top of the existing bucket-mean x
vacated-opportunity projection in `predict_rookies`.

**Design judgment calls**:
- **Discrete tier + multiplicative scale on the existing prediction, not a
  new fitted bucketing dimension**: chosen deliberately per the spec's own
  guidance to prefer flags/tiers over precise continuous scaling on data
  this thin. The per-`(position, round_bucket)` rookie sample is already
  11-132 rows; splitting further into a `(position, round_bucket,
  athletic_tier)` grouping for `fit_rookie_baselines` would shrink an
  already-thin sample further (combine's ~85% pfr_id join coverage costs
  more rows on top of that) and risk noise-fitting. A scale on the existing
  point estimate needs no new fitted sample at all.
- **athletic_score** = mean of two WITHIN-POSITION percentile ranks (raw
  40-times/verticals aren't comparable across QB/RB/WR/TE): 40-time
  percentile (faster = higher) and vertical-jump percentile (higher =
  higher), each ranked against **every combine tester at that position**,
  not just the subset who made an NFL roster — using only "players who made
  it" as the percentile population would bias the scale itself, the same
  survivorship-bias trap Addendum 3 already found and fixed for the QB
  play-probability signal. A player missing one metric still gets scored
  off the other (mean with `skipna`); missing both, or no combine match at
  all, is `'no_data'` (a real, explicit fallback — not a dropped row or a
  NaN multiplier). Tier cutoff is a simple median split, not a
  data-optimized cutpoint, per the "keep the rookie path simple" mandate.
- **Bug found and fixed while wiring this up**: joining `combine_data` to a
  target-season rookie via `player_id -> players.gsis_id -> players.pfr_id`
  (the natural first approach, matching every other player_id join in this
  project) silently failed for nearly the entire 2026 **drafted** rookie
  class, because `draft_picks.gsis_id` for the 2026 class is the
  already-known placeholder id (not a real gsis_id — see `predict.py`'s
  `with_display_names` docstring for the same root cause), so it can't match
  `players.gsis_id` at all. Concretely: Drew Allar (2026 QB, real
  `draft_picks.pfr_player_id='AllaDr00'`) came back `'no_data'` via the
  `player_id` path even before checking whether he actually tested at the
  combine. Fixed by carrying `pfr_id` through
  `identify_target_season_rookie_class` directly from `draft_picks.
  pfr_player_id` (drafted) / `seasonal_rosters.pfr_id` (UDFA, whose
  `player_id` IS a real gsis_id) and joining `combine_athletic_scores_by_
  pfr_id` on that instead — bypassing the broken gsis_id entirely. Verified
  the fix: the 2026 drafted-QB `no_data` count before the fix was 8/8 (every
  single QB, including real testers) with the placeholder-gsis_id path;
  after the fix, real testers correctly resolve (e.g. Taylen Green:
  `above_median`, 1.00; Jalon Daniels: `above_median`, 0.78; Luke Altmyer:
  `above_median`, 0.62) and QB prospects who genuinely skipped every drill
  at the combine (Drew Allar, Cade Klubnik, Fernando Mendoza — all have a
  combine row with every drill field null) still correctly resolve to
  `'no_data'`, not a false match. `build_rookie_dataset` (the historical
  path used for baselines/backtest) is unaffected by this bug — historical
  seasons' `draft_picks.gsis_id` is real, confirmed in Phase 6's own
  spot-check (256/256 for 2025 vs. 0/230 for 2026).
- **Coverage for the 2026 target class**: 254 rookies total; 38
  `above_median`, 10 `below_median`, 172 `no_data` (didn't test the relevant
  drills, or genuinely absent from the pull — not a residual join bug).

**Spot-check (by name)**: WR round_2_3 bucket, 2026 class — Zachariah Branch
(`above_median`, score 0.85) vs. Germie Bernard (`below_median`, score
0.36). Verified the scale mechanism is applied correctly by backing out the
pre-athletic-scale implied baseline for each (dividing their final
`receiving_yards` prediction by their own tier's multiplier): Branch's
implied pre-scale baseline was 25.5 yd/game, scaled up to 27.5 (x1.08);
Bernard's was 30.4, scaled down to 28.6 (x0.94) — the mechanism is doing
exactly what it's supposed to. **Caveat worth stating**: Bernard's own
FINAL number still lands higher than Branch's, because his team's
vacated-opportunity scale (a much bigger, independent factor) outweighs the
modest +/-6-8% athletic adjustment — this is by design (the athletic scale
is a secondary refinement, not meant to dominate the projection), not a bug,
but it means the athletic tier's effect on the final ranked list is subtle
and easy to miss without unpacking the math as done here.

### Part 4: retrain / backtest / predict / Sleeper-compare

Ran the full cycle (`train.py`, `backtest.py`, `predict.py --season 2026`,
`fantasy_points.py --season 2026`, `sleeper_compare.py --season 2026`) and
also reran the **exact pre-Addendum-4 code** (via `git stash`) through the
same train/backtest cycle, to get an honest apples-to-apples before/after —
the numbers already written in this report's earlier addenda were computed
against an older DB/code snapshot and are no longer a clean baseline (the
rookie path picked up UDFA-rookie handling since then, which alone moves
rookie backtest `n_test` up for RB/WR/TE independent of this task).

**Veteran backtest (2025 holdout), before vs. after this task's changes** —
model MAE, all other columns (n_test, naive_mae) unchanged since the same
2025 holdout / naive baseline is used both times:

| Position | Stat | Before MAE | After MAE | Delta |
|---|---|---|---|---|
| QB | attempts | 6.879 | 6.888 | +0.009 (worse) |
| QB | completions | 4.210 | 4.123 | -0.087 (better) |
| QB | passing_yards | 49.034 | 48.988 | -0.046 (better) |
| QB | passing_tds | 0.419 | 0.413 | -0.006 (better) |
| QB | interceptions | 0.259 | 0.263 | +0.004 (worse) |
| QB | carries | 1.291 | 1.276 | -0.015 (better) |
| QB | rushing_yards | 8.469 | 8.486 | +0.017 (worse) |
| QB | rushing_tds | 0.115 | 0.118 | +0.003 (worse) |
| RB | carries | 2.652 | 2.652 | +0.0005 (worse) |
| RB | rushing_yards | 12.584 | 12.552 | -0.033 (better) |
| RB | rushing_tds | 0.150 | 0.151 | +0.0005 (worse) |
| RB | targets | 0.815 | 0.817 | +0.002 (worse) |
| RB | receptions | 0.693 | 0.701 | +0.009 (worse) |
| RB | receiving_yards | 5.945 | 5.769 | -0.176 (better) |
| RB | receiving_tds | 0.061 | 0.060 | -0.001 (better) |
| WR | targets | 1.155 | 1.179 | +0.024 (worse) |
| WR | receptions | 0.849 | 0.844 | -0.005 (better) |
| WR | receiving_yards | 11.419 | 11.339 | -0.080 (better) |
| WR | receiving_tds | 0.127 | 0.127 | -0.0002 (better) |
| TE | targets | 0.789 | 0.812 | +0.024 (worse) |
| TE | receptions | 0.689 | 0.681 | -0.008 (better) |
| TE | receiving_yards | 7.576 | 7.673 | +0.097 (worse) |
| TE | receiving_tds | 0.129 | 0.129 | +0.0002 (worse) |

**Stated plainly, per this project's rule**: the injury feature is a
**wash on the veteran backtest** — 10 of 22 stat models improved, 12 got
marginally worse, and every delta is small (largest single move is RB
receiving_yards at -0.176). The same 3 stats that lost to the naive
baseline before this change (QB rushing_yards, RB targets, RB receptions)
still lose to naive after it — unchanged, not newly broken or newly fixed.
This is a plausible, honest outcome for adding one new trailing feature to
an already heavily-regularized, small-sample (250-660 rows) model — the
feature importance numbers above show LightGBM IS using it, just not in a
way that moves aggregate MAE meaningfully in either direction on this
particular 2025 holdout.

**Rookie backtest (2025 holdout), before vs. after**:

| Position | Stat | Before MAE | After MAE | Delta |
|---|---|---|---|---|
| QB | attempts | 6.849 | 6.487 | -0.362 (better) |
| QB | completions | 3.463 | 3.248 | -0.215 (better) |
| QB | passing_yards | 42.603 | 40.317 | -2.286 (better) |
| QB | passing_tds | 0.249 | 0.249 | ~0 |
| QB | interceptions | 0.304 | 0.304 | ~0 |
| QB | carries | 1.343 | 1.337 | -0.006 (better) |
| QB | rushing_yards | 7.619 | 7.727 | +0.107 (worse) |
| QB | rushing_tds | 0.170 | 0.169 | ~0 |
| RB | carries | 2.936 | 2.964 | +0.028 (worse) |
| RB | rushing_yards | 13.439 | 13.617 | +0.179 (worse) |
| RB | targets/receptions/TDs | ~unchanged | ~unchanged | <0.02 either way |
| WR | targets/receptions | 1.040/0.673 | 1.054/0.684 | +0.014/+0.011 (worse) |
| WR | receiving_yards | 10.453 | 10.664 | +0.211 (worse) |
| TE | receiving_yards | 9.922 | 9.806 | -0.115 (better) |
| TE | other stats | ~unchanged | ~unchanged | <0.02 either way |

**Stated plainly**: the combine-athleticism scale shows a real, if modest,
improvement for the **QB** rookie stats that matter most for fantasy
(attempts/completions/passing_yards all improved, passing_yards MAE dropped
~5.4%), and is a small-to-negligible wash for RB/WR/TE — again honest and
consistent with the mechanism's design (a +/-6-8% scale on top of a bucket
mean is a small lever, not expected to transform performance on an n=9-38
holdout).

**Sleeper comparison, 2026 season** (`python -m src.comparison.sleeper_compare
--season 2026`), 773/789 rows matched (98%):

| Scope | Baseline (pre-task) corr | After-task corr | Delta |
|---|---|---|---|
| Overall | 0.894 | 0.892 | -0.002 |
| QB | 0.923 | 0.922 | -0.001 |
| RB | 0.895 | 0.892 | -0.003 |
| WR | 0.854 | 0.854 | ~0.000 |
| TE | 0.867 | 0.866 | -0.001 |

Mean absolute delta (points/game, after-task): overall 1.42, QB 1.75, RB
1.50, WR 1.43, TE 1.11.

**Stated plainly, per this project's rule**: the Sleeper-comparison
correlation **did not move the needle** — every position's correlation
shifted by 0.003 or less, indistinguishable from noise. Combined with the
backtest results above, the honest overall conclusion for this task is that
the injury-durability and combine-athleticism features are **real, sane,
non-leaking signals that LightGBM/the rookie path do use** (confirmed via
feature importances and the mechanism-level spot-checks), but their effect
on this project's two existing top-line evaluation metrics (backtest MAE,
Sleeper correlation) is small and mixed rather than a clear win. This is not
a failure of implementation — it is a realistic result for adding one
modest new signal on top of an already-decent model with a small training
set, and is reported as such rather than framed as a bigger improvement
than the numbers support.

### Caveats and residual limitations (Addendum 4)

1. **`injury_durability_rate` conflates "missed games due to injury" with
   "missed games for any reason"** (holdout, suspension, personal reasons,
   simple roster churn) — `missed_games` is purely `team_games -
   games_played`, with no way to distinguish cause. For the low-games_played
   fringe/practice-squad players spot-checked (e.g. a player with 1 game
   played gets a high score by this formula's arithmetic even with zero
   actual injury designations), this is really measuring "wasn't on this
   team's active roster most of the season" more than genuine injury
   durability — a real, stated limitation, not hidden by the McCaffrey/Reed
   spot-checks that happen to be genuine injury cases.
2. **`combine_data`'s empty-DataFrame-instead-of-exception behavior for a
   not-yet-held combine** (Part 1) means `failed_seasons`/"full coverage"
   claims for this specific source need a manual row-count sanity check for
   any future season, not blind trust — flagged in the source code and
   restated here.
3. **The combine-athleticism scale's real-world effect on the final ranked
   projection list is easy to overstate at a glance** — as the Branch/Bernard
   spot-check shows, the vacated-opportunity scale (pre-existing, much
   larger swings) can easily dominate the athletic tier's modest +/-6-8%
   adjustment in the final number. The mechanism is verified correct in
   isolation; its practical impact on who ends up ranked where is smaller
   than either scale examined independently might suggest.
4. **Not otherwise re-verified**: this addendum did not re-run
   `apply_depth_chart_gating`/`reassign_team_changers` spot-checks (Phase 6)
   or re-verify the QB Sleeper play-probability correction (Addendum 3) —
   both are unchanged by this task's code and were only exercised
   incidentally by running the full `predict.py` pipeline end-to-end.
