# Draft capital removed from the player-level models — 2026-08-14

## Change

`draft_round` and `draft_pick` are no longer inputs to any player-level model.
They were added to `FEATURE_COLS` in Phase 5 of the consensus-gap work on the
theory that draft capital keeps predicting a year-2 leap after the rookie
boundary. Driver review says otherwise: once a player has NFL snaps on record,
his own observed usage, role, and experience are what predict next season —
where he was drafted is a rookie-path input and belongs only there.

One file changed, `src/projection/features.py`:

- `FEATURE_COLS` drops `"draft_round"`, `"draft_pick"`. `career_year`
  (`season - rookie_season`) **stays** — that is experience, not draft slot,
  and it is the honest carrier of the year-2/year-3 curve.
- `build_player_season_features` now selects only `rookie_season` from
  `players`; the round/pick columns are no longer read at all.

Scope of the removal, by model family:

| Family | Affected | Why |
|---|---|---|
| 22 per-(position, stat) rate models | yes | schema is `ALL_FEATURES` |
| 4 per-position availability models | yes | schema is `AVAILABILITY_FEATURES` |
| 4 team-grain models | no | `TEAM_FEATURES` never contained draft columns |
| Rookie rule path | no | `rookies.py` reads round/pick from `draft_picks` directly |

The rookie path is untouched by construction — it sources draft capital from
the `draft_picks` table, not from this feature table. Draft capital still
drives rookie projections exactly as before.

## Regeneration

Full documented pipeline was re-run: `train` → `backtest` → `predict --season
2026` → `fantasy_points --season 2026` → `sleeper_compare --season 2026` →
`fantasy_evaluation`. All 36 model binaries in `models/` were refit; the
feature schema stored in each joblib is now 2 columns narrower.

`pytest`: **63 passed**.

## Held-out evidence

### 2024→2025 veteran holdout (per-game rate MAE)

12 of 23 stats improved, 11 worsened — a wash on the headline, with the
movement concentrated as expected: receiving-side stats improved, QB/RB volume
stats gave a little back.

The standing loss list gets shorter. Two stats lost to naive carry-forward
before this change; one does now.

| | Before | After |
|---|---|---|
| Losses vs naive | QB `rushing_yards`, RB `targets` | QB `rushing_yards` |

RB `targets` MAE 0.7774 → 0.7382 (−5.0%), which flips it to a win. Largest
regressions: QB `passing_yards` 51.93 → 54.56 (+5.1%), RB `rushing_yards`
11.23 → 11.69 (+4.1%), QB `rushing_yards` 7.23 → 7.51 (+3.9%, already a loss
before and after).

Rolling-origin (3 folds, the more robust read) tells the same story: small
moves both ways, RB `targets` fold-win-rate 0.33 → 0.67, WR `receiving_yards`
10.74 → 10.58. No stat changed its win/loss status against naive in that view.

### Leakage-safe 2024→2025 fantasy evaluation (the freeze's gate)

`all_eligible`, model method, against the frozen 2026-08-13 baseline:

| Position | Spearman | Points MAE | Tier hits | VORP MAE |
|---|---|---|---|---|
| QB | 0.7917 → 0.7826 (−.009) | 41.72 → 42.40 (+0.68) | 6/12 → 6/12 | 47.98 → 46.41 (−1.57) |
| RB | 0.7288 → 0.7331 (+.004) | 33.27 → 33.58 (+0.32) | 15/24 → 15/24 | 35.74 → 35.20 (−0.54) |
| WR | 0.7690 → 0.7725 (+.004) | 23.11 → 22.72 (−0.39) | 22/36 → 22/36 | 25.72 → 24.59 (−1.13) |
| TE | 0.8369 → 0.8327 (−.004) | 16.55 → 16.57 (+0.02) | 5/12 → 5/12 | 39.28 → 35.16 (−4.12) |

Stated plainly: Spearman and points MAE are a wash (four sub-0.01 correlation
moves, MAE within ±0.7 points across a full season). Tier hits are unchanged
at every position. **VORP MAE improves at all four positions**, most of all at
TE (−4.12), which the freeze called out as the build's weakest column. The
model still beats carry-forward and availability-adjusted baselines on
Spearman and points MAE at all four positions, and still does not beat
carry-forward on QB tier hits (6/12 v 7/12).

### Sleeper agreement (2026, informational)

Matched-player season-total correlation 0.9481 → 0.9477, mean absolute delta
14.77 → 14.84, bias −4.54 → −4.64. Per-position bias on the `sleeper_50_plus`
stratum: QB −10.2 → −8.0, RB −0.5 → −1.5, WR −30.7 → −31.0, TE −21.5 → −21.3.
Noise-level in every column — this change does not move consensus agreement.

## What actually moved in the 2026 board

Mean absolute change is 2.18 season points across 768 player-position rows; 38
rows move more than 10 points. The direction is exactly what removing a
pedigree term should produce.

Veteran rows only, mean season-point change by draft round:

| Round | n | mean Δ |
|---|---:|---:|
| 1 | 86 | −1.70 |
| 2 | 67 | −2.15 |
| 3 | 59 | −1.49 |
| 4+ / undrafted | 329 | +0.88 |

The effect concentrates in sophomores, where the pedigree bump was largest —
round-1 sophomores −5.35, round-3 −3.69, round-2 −2.44, round-4+/undrafted
+1.27.

Largest individual moves, and each is the same story: recent high picks lose a
bump their own usage did not earn, late-round veterans with real usage gain
one they were being denied.

| Player | Pos | Before | After | Δ |
|---|---|---:|---:|---:|
| Ashton Jeanty | RB | 254.2 | 228.5 | −25.7 |
| Isiah Pacheco | RB | 77.6 | 102.1 | +24.6 |
| Dillon Gabriel | QB | 46.8 | 25.8 | −21.0 |
| Tony Pollard | RB | 171.0 | 191.8 | +20.8 |
| Kyler Murray | QB | 246.9 | 228.0 | −19.0 |
| Cam Ward | QB | 234.4 | 217.1 | −17.3 |

## Honest caveats

- This is not a measured accuracy win. It is a wash on the two headline
  metrics, a real gain on VORP MAE, and a change made for a stated modeling
  reason rather than to chase a backtest number.
- The Phase-5 addition it reverses **was** validated as a global win on an
  extended 2016-2024 leave-one-transition-out window (8 transitions, n>1000),
  a wider window than the 3-4 transitions the production models see. The
  evidence above comes from the production window and the leakage-safe
  evaluation, not that extended window, so this does not refute the Phase-5
  measurement on its own terms — it says the production-window and
  fantasy-evaluation cost of dropping the feature is ~zero, and the removal
  rests on the driver argument.
- The freeze's known limits are unchanged. Nothing here touches the rookie
  path, the curated depth chart, role discounts, or the cross-player share
  reconciliation gap.

## Artifacts regenerated

`output/projections_2026.csv`, `output/fantasy_points_2026.csv`,
`output/sleeper_comparison_2026.csv`, `output/fantasy_evaluation_2025.csv`,
`output/fantasy_evaluation_summary_2025.{csv,json}`, all of `models/`.
The `FREEZE_2026-08-13.md` artifact hashes no longer describe the working
tree — that document stands as the dated snapshot it is, and this file is the
newly versioned evaluation its release-gate rule asks for.
