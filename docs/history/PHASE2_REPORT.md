# Phase 2 Report: OL Attribution Ridge Regression

## Sample sizes and drops per season

| Season | Total run/pass plays | Dropped (not exactly 5 OL) | Kept plays | Pass-pro N | Pass-pro alpha | Run-block N | Run-block alpha | Pressure fallback rate |
|---|---|---|---|---|---|---|---|---|
| 2021 | 35500 | 1417 | 34083 | 20550 | 1000.0 | 13532 | 562.3 | 6.3% |
| 2022 | 35193 | 1087 | 34106 | 20076 | 562.3 | 14030 | 1000.0 | 6.7% |
| 2023 | 35378 | 955 | 34423 | 20404 | 1000.0 | 14019 | 1000.0 | 0.0% |
| 2024 | 34789 | 1055 | 33734 | 19673 | 1778.3 | 14061 | 562.3 | 0.0% |
| 2025 | 34415 | 1888 | 32527 | 19178 | 1778.3 | 13349 | 562.3 | 0.0% |

Plays are dropped when the offense doesn't resolve to exactly 5 identifiable linemen (jumbo/6-7 OL packages, unidentifiable personnel, or malformed participation rows). This is ~3-5% of run/pass plays per season, except 2025 (~5.5%, likely partial-season roster/personnel labeling noise since 2025 data is mid-season as of this run).

## OL identification method (data-quality note)

`participation.offense_positions` (a per-play position label aligned to `offense_players`) is **entirely NULL for 2021 and 2022** in this DB - only populated 2023-2025. This was not previously flagged in Phase 0/1 docs and is a real gap, not a bug. For 2021-2022 we fall back to `players.position` (career/latest-known position, codes OT/G/C/OL) keyed on gsis_id. Coverage check: 100% of offense_players ids resolved via this fallback (no unresolved ids), and the resulting distribution of OL-count-per-play (5 OL ~97%, 6 OL ~3% jumbo packages) closely matches the 2023-2025 distribution obtained directly from offense_positions, which supports using the fallback but doesn't make it identical to a true point-in-time label - a player who changed position mid-career could be mislabeled in the season they switched.

## Pass protection sub-model

Outcome: `was_pressure` where non-null. `was_pressure`/`time_to_throw`/`route` are ~61-62% null in 2021-2022 (NGS tracking coverage change before 2023) and near-fully populated in 2023-2025 (confirmed again here). For 2021-2022 rows where `was_pressure` is null, we substitute `sack` (near-zero nulls across all years) as a coarser fallback outcome. This means the 2021-2022 pass-protection coefficients are estimated on a noisier, more conservative outcome that misses hurries/hits not resulting in a sack - treat 2021-2022 pass-protection coefficients as lower-confidence than 2023-2025.

Controls: down, ydstogo, score_differential, game_seconds_remaining (game script), and opponent pass-rush quality (defense's leave-one-out pressure rate against all OTHER offenses that season, excluding the current posteam-defteam matchup entirely to avoid the O-line's own performance leaking into its own opponent-quality control).

**Judgment call**: `time_to_throw` is NOT used as a control, despite being mentioned as an option in the spec. Pressure causally shortens time_to_throw (and sacks/scrambles truncate it outright), so it is a post-treatment variable relative to pressure - controlling for it would bias the lineman coefficients rather than clean them up. Open for the user to reconsider if a different causal framing is preferred.

## Run blocking sub-model

Outcome: raw `rushing_yards`. This DB's pbp table has no literal "rushing yards over expected" field - checked `PRAGMA table_info(pbp)`; the only expected-yardage fields are `xyac_*` (expected yards AFTER CATCH, for receptions, not rushing). Controls (down, ydstogo, defenders_in_box, score_differential) partial out the situational component in the ridge fit, so lineman coefficients approximate yards-over-expected conditional on those controls, but this is a documented substitute for the literal field the spec described, not the field itself.

## Stability testing across years

### pass_protection
- Linemen appearing in 2+ seasons: 413
- 2021 -> 2022: n=251, correlation=0.218
- 2022 -> 2023: n=243, correlation=0.203
- 2023 -> 2024: n=248, correlation=0.205
- 2024 -> 2025: n=247, correlation=0.025

### run_blocking
- Linemen appearing in 2+ seasons: 415
- 2021 -> 2022: n=254, correlation=0.110
- 2022 -> 2023: n=244, correlation=0.202
- 2023 -> 2024: n=249, correlation=0.116
- 2024 -> 2025: n=248, correlation=0.076

Average year-over-year coefficient correlation across both sub-models: 0.144. This is low - consistent with ridge coefficients on single-season, play-level data being noisy per-player estimates rather than stable trait measurements. Do not treat any single season's coefficient for a given lineman as a reliable individual rating; at most, use multi-season averages and even those with caution.

## Other caveats

- ftn play-charting data (play-action, screen, blitz counts) was NOT joined into either sub-model. It's 2022+ only (missing 2021 entirely), and adding it would either force dropping 2021 or leaving nulls for one season out of five - decided against it for this phase to keep the 2021-2025 window consistent. A natural Phase 3 extension.
- Ridge coefficients are relative to the (implicit) baseline of the excluded/average lineman in that season's design matrix; they are not on an interpretable absolute scale and should only be compared within the same season/submodel, not pooled across seasons without care.
- Sacks are rare relative to non-sack pressures; the pressure_outcome fallback in 2021-2022 means those seasons' positive-class rate is much lower than 2023-2025's, so alpha selection and coefficient scale are not directly comparable across the NGS-coverage boundary.
