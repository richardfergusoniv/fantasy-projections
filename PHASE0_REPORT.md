# Phase 0 — Data Validation Gate

## Summary

**The OL attribution model is viable as originally designed.** Play-level
personnel data exists for every play from 2021 onward, including 2023-2025,
with 0% nulls on the roster fields. No fallback to game-level snap counts is
needed.

One caveat: `nfl_data_py` v0.3.3 (the installed Python package) does **not**
wrap this dataset — it has no `import_pbp_participation` function. The data
must be pulled directly from the nflverse-data GitHub release assets as
parquet files. This is a supported, stable nflverse distribution channel (not
scraping), just not exposed through the Python convenience wrapper. Ingestion
code will fetch it directly:
`https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet`

## Coverage matrix

| Source | Field | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| participation | `offense_players` / `defense_players` (roster IDs per play) | 0% null, 50,714 rows | 0% null, 50,150 rows | 0% null, 46,168 rows | 0% null, 45,919 rows | 0% null, 45,184 rows |
| participation | `offense_personnel` (position-group counts) | present | present | present | present | present |
| participation | `offense_names`/`positions`/`numbers` (readable labels) | absent | absent | present | present | present |
| participation | `was_pressure` | 61.4% null | 62.2% null | 0% null | 0% null | 0% null |
| participation | `time_to_throw` | 61.4% null | 62.2% null | 56.0% null | 57.0% null | 57.1% null |
| participation | `route` | 62.7% null | 63.8% null | 0% null | 0% null | 0% null |
| FTN charting | all fields (`is_play_action`, `is_screen_pass`, `n_blitzers`, `n_pass_rushers`, etc.) | **not available** (FTN starts 2022) | 0% null, 41,643 rows | 0% null, 48,225 rows | 0% null, 48,031 rows | 0% null, 47,316 rows (current season, in progress) |
| NGS passing | `avg_time_to_throw` (player-week aggregate, not play-level) | 0% null, 608 rows | 0% null, 603 rows | 0% null, 620 rows | 0% null, 614 rows | 0% null, 605 rows |

## Notes on `time_to_throw` nulls

The ~57-63% null rate for `time_to_throw` in the participation file is **not
a data quality gap** — it is null on every non-dropback-pass play by
construction (rushes, kneels, etc. have no throw to time). Verified against
2024 play-by-play: pass plays are 40.4% of all plays, closely matching the
~43-44% non-null rate on `time_to_throw`. For pass plays specifically, this
field is fully populated in 2023-2025 and roughly 38% populated in 2021-2022
(the pre-2023 gap is real: NGS changed its charting/tracking coverage before
2023, which is also visible in `was_pressure` and `route` for those two
years).

## Practical implications for Phase 2 (OL model)

- **2023-2025** is the clean, fully-charted window: `was_pressure`,
  `time_to_throw` (on pass plays), and `route` are all populated, plus
  human-readable `offense_names`/`positions` for QA/debugging joins.
- **2021-2022** still has full roster participation (`offense_players`,
  `offense_personnel`) but `was_pressure`/`time_to_throw`/`route` are ~60%
  null — usable for "which linemen were on the field" but not for
  pressure-based outcome modeling without accepting a smaller charted subset.
- Recommendation: train the ridge regression on 2023-2025 as the primary
  window (3 seasons, full charting), and treat 2021-2022 as optional
  supplementary data only for plays where those fields happen to be
  populated. This does not require a design change — it's just an
  effective sample-window decision within Phase 2.
- Player positions (needed to identify which of the 11 on-field offensive
  players are linemen) are not in the participation file itself and must be
  joined from roster data (`import_seasonal_rosters` / `import_players`) by
  `gsis_id`. This join needs to be validated in Phase 1 but is a standard,
  well-supported nflverse join — not a new risk.

## Gate decision

No fallback needed. Proceeding to Phase 1 as designed, using season window
**2021-2025** for general ingestion and **2023-2025** as the primary charted
window for the Phase 2 OL regression.
