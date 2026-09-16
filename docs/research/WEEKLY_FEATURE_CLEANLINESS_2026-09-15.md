# Weekly feature cleanliness (as-of / leakage / integrity)

**Date:** 2026-09-15
**Scope:** research + audit only. No model training, no promote, no reseal. League Value board, freeze knobs, and production defaults were not changed.

## Verdict: **partial**

Week-level **lagged usage features are clean enough to build on** for a new per-game model. The shift-then-roll contract holds in code and in synthetic tests: a target week's roll3 / `_l3` feature does not contain that week's own box score.

What is **not** sealed: live `projections.db` and `player_week_panel.parquet` were **missing in this environment**, so uniqueness, share bounds, and box-vs-pbp coverage were **not** re-checked on real rows. PIPELINE_MAP's "as-of audit is green" claim is weaker than it reads.

Richard can treat lagged weekly shares as a trustworthy **feature recipe**. He should not treat master as having a live-data cleanliness seal.

## Why `output/weekly_audit/audit_report.json` 404s on master

The file is produced by `scripts/audit_weekly_features.py` and is listed in `.gitignore` (`output/weekly_audit/`). The 404 is expected from git, not proof the audit never ran.

This PR force-adds the artifacts so they exist for review.

## What was checked

### 1. Official weekly feature audit (static)

Ran `scripts/audit_weekly_features.py` → **pass**.

That script does **not** open the database. It walks a hardcoded contract registry and marks `targets_share_roll3` / `carries_share_roll3` green if the notes mention shift / leakage-safe. Other registered names (`snap_pct`, `injury_durability_rate`, `opp_def_pass_epa_prior`, `depth_chart_status`) auto-pass because they are not `week < target_week` rules. Three of those have empty notes.

So PIPELINE_MAP is technically right that the **script exits 0**. It is not a live as-of scan of weekly rows.

Artifacts: `output/weekly_audit/audit_report.json`, `output/weekly_audit/feature_contract.json`.

### 2. Code contracts (v3 weekly path and v2 panel)

| Builder | Leakage design | Grain |
|---|---|---|
| `src/projection/data/features_weekly.py` | `shift(1).rolling(3)` on target/carry **share** | groupby **player_id** (can use prior-season weeks at week 1) |
| `src/projection/weekly/features/rolling.py` | `shift(1)` then rolling mean | groupby **gsis_id + season** (week 1 `_l3` is null; prior season is a separate column) |
| `src/projection/weekly/features/leakage.py` | `filter_as_of` drops the target season-week | used by tests / as-of filters |
| `src/projection/weekly/models/volume.py` | `VOLUME_FEATURE_CANDIDATES` uses `_l3` / `_l5` / prior / pregame | same-week `targets` / `target_share` are labels, not features |
| `src/projection/weekly/draws/feature_outcome_split.py` | denylist blocks raw same-week box, shares, and team aggregates (`team_attempts`, `team_carries`, `team_targets`, `team_air_yards`) at inference | panel still **stores** labels on the same row; lagged `_l3` forms remain allowed |

2025 regular-season weeks in the v3 usage loader are inferred as `week <= 18` (POST dropped). That is hardcoded for 2025 only, because the pbp-fallback `weekly` rows ship with null `season_type`.

Duplicate player-week aliases from the 2025 pbp fallback are collapsed in `load_weekly_usage` (`_canonicalize_player_weeks` sums stats, keeps one identity row).

### 3. Synthetic integrity extension (no DB required)

Ran `scripts/audit_weekly_integrity_extension.py` on in-memory fixtures.

**Passed**

- 2025 week 19 dropped as POST; weeks 1–18 kept
- Canonical usage is unique on `(player_id, season, week)`; alias rows are summed
- Team-week target/carry shares stay in `[0, 1]` and sum to 1 when volume > 0
- Poisoning week 3's target share does **not** change week 3 `targets_share_roll3`
- v2 `_l3` ignores the current week and does not cross seasons
- Volume-model feature list has no raw same-week box/share columns
- Inference denylist blocks same-week `team_attempts` / `team_carries` / `team_targets` / `team_air_yards` (`is_allowed_prediction_column`)
- Canonicalize repairs pbp-fallback name-alias duplicates
- Cutoff/vintage and market-snapshot families are **stubbed** (skipped: helpers/columns not yet in tree). This is not a live-data seal.

**Failed (real defects; see below)**

- Raw pbp-fallback aggregator can emit duplicate player-weeks
- v2 `add_team_pass_rate` attaches same-week `team_attempts` / `team_carries` onto the panel

Artifact: `output/weekly_audit/integrity_extension.json`.

### 4. Unit tests

`tests/test_weekly_feature_audit.py`, `tests/test_data_prep_appearances.py`, `tests/test_weekly_event_cohort_eval_repair.py`: **28 passed**. `tests/test_weekly_panel_leakage.py` **skipped** (no panel parquet).

The roll3 unit tests are synthetic and correctly show shift-before-roll. `week1_uses_only_prior_season` compares a number to itself; it does not prove week-1 behavior on real data.

## What could not be checked

There is **no** `data/projections.db` and **no** `data/processed/player_week_panel.parquet` in this environment (`FANTASY_PROJECTIONS_DATA_DIR` unset). Those paths are gitignored.

Not verified on live rows:

- Duplicate `(player_id, season, week)` remaining after canonicalize
- Shares > 1 from nflverse-supplied `target_share` (v2 keeps the upstream value when present)
- Team-week join coverage: weekly box vs pbp
- How complete 2025 pbp-fallback weeks are vs official player_stats
- Whether nflverse week-W injury reports used as `is_out` / `play_prob` include any post-kickoff updates
- Historical weekly depth charts joined on the same week (snapshot depth for 2025+ **does** cut `dt.date() <= kickoff date`)

Re-run on a machine with the DB:

```bash
uv run python scripts/audit_weekly_features.py
uv run python scripts/audit_weekly_integrity_extension.py
```

The extension will add live uniqueness, share-range, roll3-vs-current-share, and box-vs-pbp coverage checks when the DB is present.

Cutoff/vintage (`available_at`) and snapshot-dated ADP / season-market checks are now in the same script (`as_of_cutoff_checks`, `market_snapshot_checks`). They skip until those helpers/columns exist. Lag-only `filter_as_of` and depth `dt <= kickoff` are **not** that contract. Skipping is not a seal.

## Real defects found

These are code-contract issues, not live-table measurements.

1. **Official audit is a notes check, and the evidence file is gitignored.** PIPELINE_MAP should not be read as "live weekly tables were scanned and sealed." The gate path it names is not on master.

2. **Pbp fallback can split one player-week into two rows** when passer/rusher/receiver names differ for the same GSIS id (`aggregate_weekly_stats_from_pbp` groups by `player_id` **and** `player_name`). Downstream canonicalize fixes this for v3 usage. Anyone reading the raw `weekly` table, or skipping canonicalize, can double-count.

3. **v2 team-week panel can see same-week team volume.** `add_team_pass_rate` correctly lags `team_pass_rate_l5`, then also left-joins current-week `team_attempts` and `team_carries`. Those columns are not in the volume-model feature list, but they sit on the feature panel. Sitting on the panel is not the same as being blocked at inference.

4. **Inference denylist hole for `team_attempts` / `team_air_yards` (enforceable, now closed).** `SAME_WEEK_OUTCOME_DENYLIST` had `team_targets` and `team_carries` but not `team_attempts` or `team_air_yards`. `is_allowed_prediction_column` therefore allowed same-week `team_attempts` into the prediction frame. That is not an advisory “don’t use these” note — it is the gate that inference uses. Both names are now on the denylist, and the integrity extension plus `test_inference_denylist_blocks_same_week_team_aggregates` assert the block.

5. **Two weekly recipes disagree at week 1.** v3 roll3 can include last season's games. v2 `_l3` is null at week 1 and uses `*_prior_season` instead. Fine if intentional; not interchangeable.

None of these are "roll3 leaks the target week's own share." That specific claim in PIPELINE_MAP holds in the builders.

## Build-on guidance for the per-game model

Safe starting point:

- Labels: same-week box / shares
- Features: lagged rolls (`*_l3` / `*_l5` / `*_roll3`), prior-season means, pregame schedule (spread, total, rest), as-of depth snapshots
- Keep Vegas as the weekly benchmark; do not wire this into League Value

Do not start from:

- Raw same-week `targets` / `target_share` / `team_attempts` as features
- The static `audit_report.json` as a substitute for a DB-backed scan
- Assuming v3 `features_weekly.py` and v2 `rolling.py` are the same week-1 feature

**Next cheap check, when a DB is available:** run the integrity extension on real rows before any training.

**Next extension (now stubbed in the audit, not a seal):** when injury/practice/depth/inactives gain `available_at` (or equivalent) and a vintage filter, and when historical ADP / season-market rows gain a snapshot date and a filter, the integrity extension will assert those Rule 1 contracts instead of skipping.
