# Phase 5 — 2026 Projections: Intervals, UDFA, OC Framing, Final Deliverable

Target season: **2026** (unplayed as of this build — real-world early August
2026, draft already happened, no games played yet). Source season for the
veteran path is 2025.

## 1. 2026 draft-pick availability — checked, available, pulled

`draft_picks` had **no 2026 rows** before this phase (`SEASONS` in
`src/db/load.py` stopped at 2025). Checked directly with
`nfl_data_py.import_draft_picks([2026])` — **it works**: 257 picks, a full
7-round class (Fernando Mendoza, Jeremiyah Love, Caleb Downs, etc. — this
year's actual draft class).

Fix: bumped `src/db/load.py`'s `SEASONS` from `range(2016, 2026)` to
`range(2016, 2027)` and reran `python -m src.db.load` (the standard
pipeline, not a parallel one-off fetch). This is safe because every
`src.ingest.sources` fetcher already uses `skip_missing=True` —
tables where 2026 data genuinely doesn't exist yet (pbp, weekly,
snap_counts, participation, ftn, weekly_pfr_*: the season hasn't been
played) correctly report 2026 as a gap in `failed_seasons` rather than
erroring the whole load; `draft_picks`, `schedules`, `seasonal_rosters`,
and `weekly_rosters` for 2026 all loaded with full coverage (2026 rosters
are already public pre-season). Verified post-load: `draft_picks` has 257
rows for season=2026 (matching every other season's count), `weekly`/`pbp`
have **zero** 2026 rows (correctly — no games played yet, not silently
fabricated).

**Data-quality gap found in this pull, worked around, documented rather
than hidden**: `draft_picks.gsis_id` for the 2026 class is **not a real
gsis_id** — spot-checked 230/230 non-null 2026 rows against the
`00-0#######` format used by every other season (256/256 match for 2025)
and **0/230 match** for 2026. nflverse hasn't back-filled real gsis_ids for
this draft class yet. Consequences handled:
- These player_ids don't exist in `players.gsis_id` at all → display names
  for drafted 2026 rookies are resolved via `draft_picks.pfr_player_name`
  instead (see §5).
- These player_ids also aren't in `seasonal_rosters` → team for drafted
  2026 rookies falls back to `draft_picks.team` for every single pick, not
  just edge cases.
- `draft_picks.team` turned out to use **PFR-style abbreviations**
  (`KAN`/`GNB`/`LAR`/`LVR`/`NOR`/`NWE`/`SFO`/`TAM`) instead of the standard
  nflverse codes used everywhere else in this project's tables
  (`KC`/`GB`/`LA`/`LV`/`NO`/`NE`/`SF`/`TB`) — found by noticing 39 distinct
  "teams" in a first output draft (32 real ones). Fixed with an explicit
  `TEAM_ABBR_FIX` map in `src/projection/rookies.py`, applied before the
  team is used anywhere (output labeling AND the vacated-opportunity roster
  join, which would otherwise have silently mismatched a Rams rookie
  tagged `LAR` against the team's actual vacated-share row keyed `LA`).

## 2. UDFA rookie baseline — added

**Old gap** (per PHASE4_REPORT.md): `identify_rookie_seasons` only matched
players present in `draft_picks`, so UDFAs were silently absent from
`predict.py`'s output entirely.

**Identification approach, and why the obvious one was rejected**: the
naive definition ("first active `weekly` season, with no `draft_picks` row
for that player") was tried first and produces 484 false "2016 UDFA
rookies" — spot-checked and nearly all of them are established veterans
(Tom Brady, Drew Brees, Antonio Gates, etc.) whose real draft/rookie season
predates 2016, `draft_picks`'s own coverage window. `draft_picks` only goes
back to 2016, so a player drafted earlier just isn't in it — looking
"undrafted" purely by absence in a left-censored table is wrong.
Fixed by using `players.rookie_season` and `players.draft_round` instead —
both are nflverse master-roster fields, not windowed by this project's
ingestion range. `identify_udfa_rookie_seasons` (`src/projection/rookies.py`)
now defines a UDFA rookie-season as: `draft_round` is null AND
`rookie_season` equals the player's first active `weekly` season (so it's
a real production year, not just a training-camp body). Re-running this
definition gives a stable ~15-38 UDFA rookie-seasons/year across
2016-2025, no left-censoring spike.

**Sample size** (2016-2025, actual production rookie-seasons):

| Position | n |
|---|---|
| QB | 12 |
| RB | 87 |
| TE | 56 |
| WR | 116 |
| **Total** | **271** |

RB/TE/WR sample sizes are comparable to or larger than several of the
existing drafted-round buckets (e.g. RB `round_1`=13, RB UDFA=87). QB (n=12)
is thin — flagged as low-n below (§3). `build_rookie_dataset` now includes
these rows alongside drafted rookies, so `fit_rookie_baselines`'
`groupby("round_bucket")` naturally produces a populated `undrafted`
bucket instead of the previously-empty one.

**Confidence tiering**: added `rookie_tier` column (`'drafted'` /
`'udfa'`), distinct from the existing `low_confidence` flag (True for
both — the hard project rule). All rookie output rows still carry
`source='rookie_rule'` as required.

**A structural bug found and fixed along the way**: the existing
`identify_rookie_seasons` requires the target season's rookies to already
have an active week in `weekly` for that season — which is true for
*historical* backtesting but **structurally impossible for a genuinely
future season** (2026 has zero played games). Verified directly: calling
`identify_rookie_seasons(conn, [2026])` returns **0 rows** before this fix
— `predict.py`'s entire rookie path would have silently produced **zero**
rookie projections for 2026, drafted or UDFA. Fixed with a dedicated
`identify_target_season_rookie_class(conn, target_season)` that reads the
target season's rookie class directly off `draft_picks` (drafted) and
`seasonal_rosters` (UDFA: `years_exp==0` and `draft_number` is null — both
fields are populated pre-season, unlike anything derived from `weekly`).
Historical rookie identification (used only to fit the baselines) is
untouched and still requires confirmed production, which is the right
behavior there. `team_vacated_opportunity` similarly needed a
roster-based fallback for a season with zero played games (added: falls
back to `seasonal_rosters[season]` to determine "still on this team" when
no `weekly` rows exist for that season).

## 3. Prediction intervals

**Veteran path**: empirical, from the SAME 2025 held-out backtest already
in `backtest.py` (train 2021-22/22-23/23-24, predict 2025) — not a second
quantile-regression model, per the spec's own reasoning (training rows are
already thin at 250-660 per position/stat; fitting a second model family
would halve the effective sample further). `backtest.residual_quantiles()`
computes signed residuals (`actual - pred`) on the held-out 2025 test set
per (position, stat), and takes the **10th/90th percentile** (an 80%
interval — chosen as a reasonably informative width without over-trusting
tail behavior on n=61-170 samples; narrower e.g. 25th/75th would be
sharper but arguably overstate confidence given the sample sizes).
`pred_pg_low = clip(pred_pg + resid_p10, floor=0)`,
`pred_pg_high = pred_pg + resid_p90`. Saved to
`models/interval_residuals.csv` by `python -m src.projection.backtest`
(same gitignored `models/` artifact convention as the trained models).

All 20 position/stat combos have n_test = 61-170 (same table as
PHASE4_REPORT.md's MAE backtest) — **none fell below the
`INTERVAL_MIN_N=30` threshold**, so no position/stat needed the
normal-approximation fallback; `resid_std` is computed and carried anyway
in case a smaller future test set needs it later.

**Rookie path**: no naive-baseline backtest exists for rookies (no prior
season to carry forward — same reasoning PHASE4_REPORT.md gives), so no
additive residual distribution exists either. Used a different,
**multiplicative** fallback instead: `rookie_interval_ratios()` computes,
per (position, round_bucket, stat), the empirical 10th/90th percentile of
`actual_pg / bucket_mean_pg` across historical rookies in that bucket.
Applied multiplicatively (`pred_pg_low = pred_pg * ratio_low`) rather than
additively, because the rookie point prediction is itself
`bucket_mean * vacated_opportunity_scale` — a flat additive band would
ignore how much an individual player's number was already scaled up or
down, whereas a ratio stays consistent with that scaling logic.

Bucket sample sizes here are smaller than the veteran backtest (same 11-132
range documented in PHASE4_REPORT.md's rookie table) — buckets below
`ROOKIE_INTERVAL_MIN_N=20` are flagged via `interval_low_n_flag=True`
rather than presented at equal confidence: **QB `round_2_3` (n=14), QB
`undrafted` (n=12), RB `round_1` (n=13), TE `round_1` (n=11)**. In the
final 2026 output this flags 83/1887 rows (all rookie rows — no veteran
row is ever flagged, since all 20 veteran combos cleared the n=30
threshold). A bucket/stat with fewer than 3 historical values (none occur
in this run, but structurally possible) falls back further to a
deliberately wide `(0.2, 3.0)` ratio, always flagged.

## 4. OC-inheritance framing — stated limitation, not implemented

Checked `src/coordinator/oc_assignments.csv`: it covers **2016-2025 only**
(built in Phase 3, before this phase). It does **not** cover 2026.

**Stated plainly, as required**: this 2026 projection uses each team's
**2025 observed** scheme/OL context as-is (`oc_tendency_profiles`,
observed not `inherited_*`, for season 2025 — the same framing
train.py/transitions.py already document for any N→N+1 transition). It
does **not** reflect any coordinator hires or departures for the 2026
season, since `oc_assignments.csv` doesn't cover 2026 and building that
out (researching every team's 2026 OC situation, scoring
new/returning/inherited scheme tendencies) was explicitly out of scope for
this phase per the spec. This is a real, live limitation of the current
`output/projections_2026.csv` for any team that hired a new offensive
playcaller this offseason — those teams' veteran-model rows are using
last year's staff's tendencies as the stand-in for a coaching staff that
may no longer be there.

## 5. Final deliverable: `output/projections_2026.csv`

Generated via `export_projections()` in `src/projection/predict.py`
(`with_display_names()` joins `players.display_name`, falling back to
`draft_picks.pfr_player_name` then `seasonal_rosters.player_name` for the
2026 rookie class whose placeholder ids aren't in `players` at all — see
§1's gsis_id caveat). **3344 rows** (see the bug-fix section below - this reflects the corrected
run, not the phase's original draft), one row per (player, position, stat):

| source | rookie_tier | rows |
|---|---|---|
| veteran_model | — | 2308 |
| rookie_rule | drafted | 334 |
| rookie_rule | udfa | 702 |

Columns: `player_id, display_name, team, position, stat, pred_pg,
pred_pg_low, pred_pg_high, source, low_confidence, rookie_tier,
interval_low_n_flag, season`. No nulls anywhere in the final file
(verified) except `rookie_tier`, which is legitimately null for veteran
rows.

**How to regenerate**: from repo root, with `.venv` active:
```
python -m src.db.load               # only if refreshing raw data
python -m src.projection.train      # trains models/*.joblib (skip if already trained)
python -m src.projection.backtest   # builds models/interval_residuals.csv (required for intervals)
python -m src.projection.predict --season 2026
```
The last command writes `output/projections_2026.csv` and prints a
summary. `--out <path>` overrides the default path if needed.

## Bug found and fixed during review: veteran stars silently missing

Before accepting this phase's deliverable, a spot-check for well-known
players (Josh Allen, Ja'Marr Chase, Christian McCaffrey) found **all three
completely absent** from `output/projections_2026.csv`, despite having
complete, valid 2025 feature rows. Root cause: `identify_rookie_seasons()`
in `src/projection/rookies.py` accepted a `seasons` parameter but never
actually filtered by it - it always returned every historically-drafted
rookie across all of 2016-2025 (629 player-seasons), regardless of what was
passed in. `project_veterans()` in `predict.py` calls this with
`[source_season]` expecting only that one season's rookie class back, to
build an exclusion set (rookies get projected via the separate rule-based
path, not the veteran model). Because the function ignored the season
filter, the exclusion set instead contained **every player ever drafted
2016 or later**, permanently barring them from the veteran-model path for
their entire career - Josh Allen (2018 rookie), Ja'Marr Chase (2021), and
Christian McCaffrey (2017) were all silently dropped from every season's
projection, forever, not just their actual rookie year. This had been
present since Phase 4 but only surfaced when a human read the actual
output file (all the aggregate backtest MAE numbers in `PHASE4_REPORT.md`
were unaffected - `train.py`/`backtest.py` never call this function).

Fixed with a one-line change (`rookies = rookies[rookies["draft_season"].isin(seasons)]`
before the return), verified the sibling function `identify_udfa_rookie_seasons`
already filtered correctly (it wasn't affected), then reran the full
pipeline (`train.py` -> `backtest.py` -> `predict.py --season 2026`) to
regenerate everything from a clean state. Effect: unique veteran players in
the output went from **177 to 476**; Josh Allen/Chase/McCaffrey (and every
other player drafted since 2016) now correctly appear as `veteran_model`
rows. Total output rows: **3344** (up from 1887) - `source`/`rookie_tier`
breakdown and all other methodology in this report is otherwise unchanged
(backtest MAE, interval quantiles, and training row counts were all
confirmed identical before/after, since the bug never touched those code
paths).

## Other judgment calls/caveats surfaced during this phase

- Veteran point predictions and interval floors are clipped to 0 (a
  per-game rate can't be negative; LightGBM has no such constraint and did
  produce one slightly-negative `receiving_tds` prediction before this fix).
- `backtest.py`'s rookie MAE table now reflects the UDFA-inclusive rookie
  dataset (n changed from PHASE4_REPORT.md's 7/22/25/11 per position to
  9/29/38/17) — a side effect of §2's fix, not separately re-tuned.
- The 2026 UDFA target class (147 candidates from `seasonal_rosters`:
  `years_exp==0`, `draft_number` null) includes every player currently on
  an offseason roster in that bucket, most of whom will never make a
  53-man roster — this is the same "project everyone identified, let the
  vacated-opportunity scaling do the differentiation" approach the drafted
  path already uses, not a new judgment call, but worth knowing when
  reading the output (a late-camp UDFA long-shot gets a real row here).
