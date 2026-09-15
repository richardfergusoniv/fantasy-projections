# Season-projection regression re-explore — 2026-09-14

**Status:** research only. Nothing in this note changes the live sealed board,
freeze knobs, or production defaults.

> **Follow-up (2026-09-15):** per-position miss brief Richard can use for
> design decisions:
> [`SEASON_PROJECTION_POSITION_MISS_BRIEF_2026-09-15.md`](SEASON_PROJECTION_POSITION_MISS_BRIEF_2026-09-15.md).
> Same freeze. Same artifacts. Structured by QB / RB / WR / TE.

| Live constraint | Value |
|---|---|
| Sealed browser board | `v2_baseline_20260830` (accuracy-first ensemble) |
| Season freeze through 2026 outcomes | RB/WR weights, availability α = 0, v3-means cutover off, 10k draws |
| This pass | Read code + committed artifacts. Did **not** run `promote_release`, did **not** reseal, did **not** edit `EXPOSURE_BLEND_ALPHA` / ensemble weights / draw count |
| Leakage-safe `fantasy_evaluation` re-run | **Not run.** `data/projections.db` is not present in this environment. Numbers below are from on-disk artifacts |

Evidence tags: **[code]** read from current source; **[artifact]** committed file under `output/` or `models/`; **[doc]** an existing repo document, which may itself be stale.

---

## 1. Executive summary (plain English)

The regression model does **not** spit out “this player will score 280 fantasy
points.” It predicts **how busy and how efficient a player is per game in a
given role** — attempts, catches, yards, touchdowns — from last year’s usage,
scheme, line, age, and depth. A separate availability model guesses how many
games he will play. Those two pieces are then turned into a season total.

On the **live draft board**, that native regression number is only part of the
story:

- For **quarterbacks and tight ends**, the board still mostly trusts this
  repo’s model (blended with the sibling v2 model).
- For **running backs and wide receivers in the top 120 of ADP**, the board
  mostly trusts **the market (ADP) plus v2**, with only a small (RB) or zero
  (WR) weight on this regression. That mix was chosen because it beat the old
  blend on the 2025 holdout, not because the regression was thrown away.
- Everyone outside that top-120 slice stays on the older v1/v2 mix.
- The “how sure are we / finish probability” layer is a third model (v3). It
  is **not** allowed to move the point estimate.

**What the regression is good at today**

It orders a **whole position** better than “just copy last year.” That is a
real result, repeated on 2023, 2024, and 2025 holdouts. Rate models also beat
a naive carry-forward on 22 of 23 player-stat cells (the only loss is RB
receiving touchdowns). Team pass-volume anchors beat naive too.

**What it is weak at — the draft-relevant part**

Knowing who the *best dozen* are, and how many points the stars will score, is
much harder than ordering the whole list:

- **QB:** Rank correlation on **per-game rates** falls from 0.77 across all
  eligible QBs (`all_eligible.rate_spearman` 0.765) to ~0.22 among actual
  starters (`starter_8plus_games.rate_spearman` 0.217). The 0.78 figure usually
  quoted is the all-eligible **season-points** Spearman (`metrics[].spearman`)
  — same backup-vs-starter story, not a second collapse. The model missed
  2025’s bounce-back / breakout QBs (Stafford, Maye, Lawrence) and over-trusted
  players who then missed time (Burrow, Daniels). A dedicated QB repair already
  tried and **failed** its starter gates (NO-GO, 2026-09-03). Do not promote
  those arms.
- **RB:** Whole-board rank is fine; identifying the RB1 tier is one of the
  better cells (17/24). Value-over-replacement is *slightly worse* than copy-
  last-year. Christian McCaffrey was the loud miss (186 projected vs 364
  actual). v1 is **not** the approved ranking signal for top-120 RB.
- **WR:** Same pattern — good whole-board order, **worse** than copy-last-year
  at picking the top 36 (19 vs 22). Ceiling names (Puka, JSN) were low;
  injured names (Nabers, Hunter) stayed high. v1 has **zero** selected weight
  on top-120 WR.
- **TE:** Best whole-board order of the four positions, worst at the top of
  the draft. Hit only 4 of 12 TE1s. Replacement-level TE is still too low
  (about 110 projected vs 133 actual), which warps VORP. Availability-adjusted
  last-year points beat the model on TE VORP.

**The freeze is doing its job.** The live RB/WR mix, the “price every healthy
player as 17 games” choice, the “do not let v3 move the mean,” and the 10k
draw count are locked until 2026 games are in. Reopening them in-season would
be fitting the board to the season it is trying to predict.

**After 2026 outcomes, the already-named path is:** run the **same**
accuracy-first selector on untouched 2026 results. Do not auto-promote. Do not
reopen the closed RB/WR “shadow repair” loop. Use starter-conditional QB
metrics, not the flattering all-eligible Spearman.

---

## 2. Current pipeline map (brief)

Two boards exist. Do not mix them up.

```
nflverse / depth / status
        │
        ▼
FEATURES     player-season rates, shares, OL, coordinator, depth tier
        │
        ▼
v1 MODELS    LightGBM per (position, stat) role-rate
             Ridge team pass/rush anchors
             LightGBM games-played (Gate A)     ← lives in projected_games_raw
             rookie rules + replacement rows
        │
        ▼
COMPOSE      hygiene + partial team-volume pull + identities
             pred_season = pred_pg × 17 (except IR/PUP/suspension)
        │
        ├── native CSVs:  output/projections_2026.csv
        │                 output/fantasy_points_2026.csv     ← v1 only
        │
        ▼
SELECTED POINT BOARD
        accuracy-first ensemble (top-120 ADP):
          QB/TE = incumbent v1/v2
          RB    = 0.10 v1 / 0.30 v2 / 0.60 ADP
          WR    = 0 v1 / 0.55 v2 / 0.45 ADP
          v3    = 0 point weight
        │
        ▼
v3 OVERLAY   p10/p50/p90, finish probabilities (10k draws in production)
        │
        ▼
SEALED RELEASE   draft_assistant/data/releases/v2_baseline_20260830/
                 pointer: active_release_2026.json
```

**[code]** `src/projection/{train,predict,veterans,composition,fantasy_points}.py`,
`src/projection/evaluation/accuracy_first.py`,
`docs/decisions/{V1_PRODUCTION_ROLE_2026-08-29,ACCURACY_FIRST_ENSEMBLE_2026-08-27}.md`.

### 2.1 How a per-game rate is produced

1. Training pairs are **season N features → season N+1 role rate**, 2021–2025
   transitions only (OL quality does not exist before 2021). **[code]**
   `train.py`.
2. The label is **per eligible week**, not per appearance week, so a backup is
   not scored as a starter just because the only weeks he appears are the weeks
   he started. **[code]** `features.py` (`{stat}_per_elig`).
3. One shallow LightGBM per (position, stat). Hyperparameters are fixed and
   **untuned** (`n_estimators=100`, `max_depth=3`, `num_leaves=8`, …) because
   each cell is a few hundred rows. **[code]** `train.py` `LGBM_PARAMS`.
4. Receiving yards for WR/TE/RB are **not** predicted as yards. They are
   predicted as a **share of team passing yards**, then multiplied by the
   Ridge team-passing-yards anchor, after an exposure-weighted share cap that
   also sees incoming rookies. **[code]** `REFRAMED_SHARE_STATS`,
   `_compose_reframed_receiving_predictions`.
5. Depth is a **model input** (`depth_tier`), not a post-hoc multiplier. The
   old Gate B ladder is retired in production (`role_discount_factor ≡ 1.0`).
   **[doc]** `V1_PRODUCTION_ROLE_2026-08-29.md`.
6. Elite residual correction exists but is **TE-only** and is omitted on
   leakage-safe folds. **[code]** `corrections.joblib` / eval coverage limits.

### 2.2 How that becomes a season projection

Shipped `compose_board` is a **slim hygiene pipeline**, not the 15-stage
hierarchical pass/rush mixer described in `STATE_OF_BUILD.md` §2.1. That
section is historical. Current stages, in order **[code]**
`composition.run_compose_stages`:

| # | Stage | What it does to the number |
|---|---|---|
| 1 | `apply_full_season_games_baseline` | Draft exposure = 17 games (`α = 0`). Gate A stays in `projected_games_raw` |
| 2 | `apply_status_overrides` | IR → 0, PUP cap, suspension → 0. 2026-only file |
| 3 | `propagate_team_anchors` | Copy Ridge team totals onto every row (metadata + later volume pull) |
| 4 | `reconcile_team_volume` | Partial top-down pull (`α = 0.5`) of QB attempts/yards and RB carries/yards toward the team anchor. Protects the QB1; does **not** protect an RB1 |
| 5 | `apply_concentration` | Could steepen within-room shares. **Currently a no-op:** every cell `promoted=false`, exponent 1.0 **[artifact]** `models/concentration_calibration.json` |
| 6 | `reconcile_td_rate_constraints` | Clip impossible TD rates (QB pass TD/attempt, rush TD/carry) |
| 7 | `reconcile_stat_constraints` | Child ≤ parent counting stats (completions ≤ attempts, …) |
| 8 | `add_projected_season_totals` | `pred_season = pred_pg × projected_volume_games` |
| 9 | `reconcile_team_season_identities` | Restore pass/catch identities on **season columns only**. Rates are untouched |

Fantasy season points are scored from `pred_season`, so the identity step
reaches the board. **[code]** `fantasy_points.compute_fantasy_points`.

Half-PPR, 4-point passing TD. Fumbles and two-point conversions are not
modeled. **[code]** `fantasy_points.SCORING`.

### 2.3 What the live board actually ranks

Native v1 remains the **component-statline engine** (roster, depth,
availability, identities, simulation structure, QB/TE ensemble, fallback for
players with no v2/ADP). It is **not** the approved top-120 RB/WR ranking
signal. **[doc]** `V1_PRODUCTION_ROLE_2026-08-29.md`.

The accuracy-first selector (fit 2024, untouched 2025 holdout, then refit
2024–2025 for 2026) is what moved RB/WR:

| Position | Arm | v1 | v2 | ADP | v3 |
|---|---|---:|---:|---:|---:|
| QB | incumbent | 0.40 | 0.60 | — | 0 |
| RB | market_no_v3 | 0.10 | 0.30 | 0.60 | 0 |
| WR | market_no_v3 | 0.00 | 0.55 | 0.45 | 0 |
| TE | incumbent | 0.90 | 0.10 | — | 0 |

**[artifact]** `output/accuracy_first_2026/ensemble_weights.json`. Applied to
93 of 778 2026 rows (the overlapping top-120 ADP slice); everyone else stays
on the incumbent mix.

---

## 3. Findings

### 3.1 Quote the on-disk eval JSON, not the old published tables

`STATE_OF_BUILD.md` §3.3 and `FANTASY_EVALUATION_2025_REPORT.md` still warn
that the 2025 tables are stale. That warning is itself partly stale.

The file `output/fantasy_evaluation_summary_2025.json` **does** contain
`composition_pipeline`, `composition_artifact_provenance`, and
`composition_stage_coverage`. It describes the **slim** compose path (volume
reconcile + identities), not the retired 15-stage mixer. Last git touch:
`b416c28` (2026-08-25). `STATE_OF_BUILD` was edited later (2026-08-30) and
still quotes older ρ/MAE/tier numbers.

**Do not quote the FREEZE_2026-08-13 or 2025 report tables as current.** Quote
the JSON. Remaining caveats on that JSON:

- Coverage map is missing `apply_concentration` (added to compose after this
  eval). In practice concentration is identity, so the scored path should
  match.
- Curated 2026 depth/status, elite correction, and intervals no-op on a 2025
  fold, by design.
- `roster_moves` / vacancy boosts / replacement construction are still
  **outside** the leakage-safe harness.
- This environment has no `data/projections.db`, so the harness was not
  re-run here.

### 3.2 v1 leakage-safe fantasy holdout (all-eligible)

Source: **[artifact]** `output/fantasy_evaluation_summary_{2023,2024,2025}.json`,
scope `all_eligible`, method `model` vs `carry_forward`. Half-PPR season
points. Population = earliest Week-1 roster.

**2024 → 2025 (the one that selected the live ensemble)**

| Pos | Model ρ | Carry ρ | Model MAE | Carry MAE | Model tier | Carry tier | Model VORP MAE | Best baseline VORP |
|---|---:|---:|---:|---:|---|---|---:|---:|
| QB | 0.780 | 0.651 | 39.21 | 50.70 | 7/12 | 7/12 | 39.81 | 63.21 (carry) |
| RB | 0.691 | 0.553 | 35.25 | 37.28 | **17/24** | 14/24 | 40.66 | **38.04 (carry)** |
| WR | 0.780 | 0.682 | 24.68 | 28.17 | 19/36 | **22/36** | 28.27 | 32.59 (carry) |
| TE | 0.835 | 0.693 | 17.28 | 21.99 | 4/12 | 3/12 | 30.24 | **24.55 (avail.)** |

Direction is stable across 2023 and 2024 folds: model wins Spearman and MAE at
all four positions every year; tier and VORP do not.

| Fold | QB ρ / MAE / tier | RB | WR | TE |
|---|---|---|---|---|
| 2022→2023 | 0.669 / 55.8 / 5/12 | 0.783 / 33.9 / 15/24 | 0.781 / 24.7 / 25/36 | 0.765 / 18.5 / 7/12 |
| 2023→2024 | 0.714 / 44.5 / 7/12 | 0.810 / 31.2 / 16/24 | 0.796 / 26.0 / 21/36 | 0.801 / 17.9 / 6/12 |
| 2024→2025 | 0.780 / 39.2 / 7/12 | 0.691 / 35.3 / 17/24 | 0.780 / 24.7 / 19/36 | 0.835 / 17.3 / 4/12 |

TE replacement (13th) vs actual 13th:

| Fold | Predicted TE13 | Actual TE13 |
|---|---:|---:|
| 2023 | 101.3 | 110.8 |
| 2024 | 95.8 | 112.4 |
| 2025 | 109.6 | 133.0 |

The undershoot is persistent. That is why TE VORP MAE can look bad even when
ordering is good: VORP is centered on each method’s own replacement.

### 3.3 Strong vs weak, by position, on 2025

Player names below are from **[artifact]** `output/fantasy_evaluation_2025.csv`
(`model_points_end_to_end` vs `actual_points`). They illustrate the JSON
metrics; they are not a new scoreboard.

**QB — orders the room, misses the starters that matter**

- Like-for-like **rate** Spearman: `all_eligible.rate_spearman` 0.765 vs
  `starter_8plus_games.rate_spearman` **0.217**. The usually quoted 0.78 is
  all-eligible **season-points** Spearman (`metrics[].spearman` 0.780). Points
  MAE 39.2 vs **61.7** on that starter slice. **[artifact]** JSON
  `qb_starter_metrics`.
- Predicted top 12: Burrow, Mayfield, Lamar, Hurts, Herbert, Nix, Mahomes,
  Daniels, Allen, Caleb, Purdy, Darnold.
- Actual top 12 the model missed: Stafford (pred rank 21), Maye (23),
  Lawrence (20), Dak (13), Goff (17). Allen was ranked 9th but **undercut by
  ~115 points** (253 vs 369).
- False-positive injuries/down years: Burrow 298 vs 134 (8 games), Daniels
  255 vs 116 (7 games).
- α = 0 (flat 17) is the mechanism: the mean board does not haircut those
  outcomes. Gate A still sits in `projected_games_raw` for simulation.
- 2026-09-03 QB repair: experimental allocation / mobile-rush / multi-season
  prior arms **lost** starter MAE to baseline. Verdict NO-GO.
  **[doc]** `docs/QB_PROJECTION_FINAL_REPAIR_REPORT.md`.

**RB — best tier cell on v1; still not the ranking signal**

- Tier 17/24 beats carry-forward 14/24. Spearman win is real.
- VORP MAE loses to carry-forward (40.7 vs 38.0).
- Loud miss: McCaffrey 186 vs 364 (pred rank 17). Kamara 247 vs 88 is the
  other side (aging / role change the prior over-trusts).
- Accuracy-first on the top-120 ADP slice: incumbent MAE 71.3 / ρ 0.454 →
  selected 65.3 / 0.526, by putting 0.60 on ADP. **[artifact]** accuracy
  report. That is why v1 is structural-only for top-120 RB.

**WR — whole board good, elite tier worse than last year**

- Spearman/MAE win; **tier 19/36 vs carry-forward 22/36**.
- Predicted high, then hurt: Nabers 211 vs 48, Travis Hunter 201 vs 50
  (rookie two-way / injury).
- Actual ceiling under-shot: Puka 188 vs 312, JSN 198 vs 302, Pickens 128 vs
  243, Olave 106 vs 218.
- Accuracy-first: v1 weight **zero**. Selected WR MAE 46.5 / ρ 0.527 vs
  incumbent 54.6 / 0.232 on the ADP slice. v3 coefficient also zero; the
  simpler market_no_v3 arm won the tie-break.

**TE — best ρ, worst draft-relevant miss**

- ρ 0.835 is the strongest of the four, every fold.
- Tier 4/12. Hits: McBride, Kelce, Pitts, Bowers. Misses include Goedert plus
  several 2025 leap/rookies (Warren, Fannin, Loveland, Ferguson, Henry).
- McBride 151 vs 253 — even the hit is a massive under-shoot of the outlier.
- Availability-adjusted baseline **wins TE VORP**. Model TE13 = 109.6 vs
  actual 133.0.

### 3.4 Rate-model backtest (not the fantasy harness)

**[artifact]** `output/backtest/veteran_holdout_2025.csv` (held-out copies of
the LightGBM/Ridge models, not the production `models/` files).

22 of 23 player-stat cells beat naive carry-forward (24 of 25 rows in the
holdout file, which also includes two team passing-volume cells). The only
loss is **RB receiving TDs** (MAE 0.0445 vs 0.0433). Older prose that says
“RB targets and RB receptions lose” (`STATE_OF_BUILD` §3.1, `PHASE4_REPORT`)
is superseded.

Team pass attempts and passing yards also win. This file does **not** report
team carries / rushing yards even though those Ridge models exist
**[code]** `train.py` `fit_team_total` for `TEAM_CARRIES_LABEL`.

### 3.5 Accuracy-first ensemble (what the live board uses for RB/WR)

**[artifact]** `output/accuracy_first_2026/report.json`, top-120 ADP, 2025
untouched holdout:

| | MAE | Spearman |
|---|---:|---:|
| Incumbent v1/v2 | 58.72 | 0.504 |
| Selected ensemble | **53.14** | **0.602** |
| Bootstrap 95% (candidate − incumbent) | [−9.81, −1.66] | [+0.026, +0.179] |

v3 adds no measured marginal point accuracy. Production RB/WR weights are
this selected mix, frozen.

### 3.6 v1 vs v2 on the same 2025 population (context, not a selector)

**[artifact]** `output/model_accuracy_compare_2025.json` (note: this file’s
v1 numbers are from an older eval CSV than §3.2; treat as directional).

On the v1 Week-1 population, v2 has the better WR MAE/tier and much better QB
MAE; v1 has the better TE Spearman. That is consistent with keeping v1 on TE,
letting v2/ADP take WR, and not using either model alone for QB starters.

### 3.7 Compose-stage effects that matter for season totals

From closed RB/WR shadow work and the QB repair, both **[doc]**:

- Raw rate and Gate A availability are co-dominant and largely cancel once
  exposure is flattened to 17.
- Team-volume reconcile can **cut a QB1** when backups still hold implausible
  attempt rates (Burrow on the 2026 board: raw 14.0 PPG → 10.9 after volume
  → 9.4 after TD clip). Protect-starter exists and is not enough when the
  bench floor still leaves residual volume.
- `reconcile_team_season_identities` moves season fantasy points with
  **corr = 1** vs the “finalization remainder”; it does not touch `pred_pg`.
- Depth-ladder is not applied. Concentration is not applied (identity).

On the 2025 eval CSV, `projected_games` is still the Gate A raw number
(never 17) while `projected_volume_games` is exactly 17 for every covered
row. Season scoring uses `pred_season` / `model_points_end_to_end`, not
`rate × projected_games`. Do not read that CSV’s `projected_games` column as
draft exposure.

---

## 4. Gaps and risks

### 4.1 Freeze-blocked (do not touch until 2026 outcomes)

Named in `STATE_OF_BUILD.md` “Season architecture freeze”:

| Knob | Live value | Why it is frozen |
|---|---|---|
| RB/WR accuracy-first weights | 0.10/0.30/0.60 and 0/0.55/0.45 | Selected on 2025; changing them now is fitting 2026 |
| Availability blend α | 0 (flat 17; Gate A in `projected_games_raw`) | Shadow Gate-A blend failed nested-fit / freeze gates |
| v3 means cutover | off (`--v3-means` exists, default off) | Generative means did not beat v1 **and** the blend on every rolling fold |
| Production draw count | 10,000 | Decision-stable compromise; strict 20k numerical gate did not pass |

Also closed, not a freeze knob but the same “do not reopen” class:

- `shadow_v1_rb_wr` repair track (`further_repair_authorized=false`).
- In-season QB architecture promotion (NO-GO already recorded).

### 4.2 Fair game later (or as labeled research now)

These do **not** authorize a live-board change. They are the honest leftover
work once outcomes exist, plus diagnostics that are allowed now.

| Gap | Why it matters | Blocked now? |
|---|---|---|
| All-eligible Spearman is the wrong QB headline | Rate Spearman 0.765 all-eligible vs ~0.22 among starters (8+ games). The 0.78 figure is **points** Spearman on the same all-eligible set, not a starter rate. | Research now; promotion after 2026 only if starter gates pass |
| TE replacement / VORP | TE13 too low every fold; avail. baseline wins VORP | Model-policy (do not retune TE weights in-season). Research OK |
| WR/RB ceiling vs injury | Elite correction is TE-only; concentration is identity | Research OK; do not reopen closed RB/WR repair |
| Unscored production stages | vacancy boosts, roster_moves, curated 2026 depth, elite correction | Extending the harness is research; cannot historically score 2026-only curated files |
| 17 of 30 historical knobs unmeasured | Many retired with volume composition; remaining live knobs (reconcile α = 0.5, QB volume shares 0.941/0.942, TD clips, vacancy α) still bind | Ablations already exist for some; no in-season retune |
| Eval artifacts vs docs | STATE_OF_BUILD §2.1 still lists 15 stages; §3.3 quotes superseded numbers | Docs-only; this note is the correction. Do not “fix” by resealing |
| Team rush-anchor holdout table | Ridge models exist; `veteran_holdout_2025.csv` omits them | Measurement gap |
| Joint fantasy interval | Componentwise 80% envelope, not a joint score interval | Known limitation, not freeze |
| Sentiment | Fields ship; model inactive (one season of research) | Unrelated to this path |
| No `projections.db` in this agent environment | Cannot refresh the 2025 JSON against current `compose_board` | Operational, not a model finding |

### 4.3 Structural measurement limits (will still be true after 2026)

- Curated depth (`starters_2026.csv`) cannot be scored on a historical fold.
- Rookie MAE is directional (small n, `low_confidence=True` always).
- Sleeper agreement is **not** an accuracy metric. Policy: diagnostic only.
  Past allocation decisions did use it; new work must not.
- August camp cuts are outside the eval universe (Week-1 roster).

---

## 5. Recommended next experiments AFTER freeze unlock

No auto-promote. Each item is “run, write a decision record, then a human
chooses.” Ranked by how much of the live board it can honestly move, and by
whether the repo already named the path.

| Rank | Experiment | What it can change | Evidence required before any promote | Already named? |
|---:|---|---|---|---|
| 1 | **Unchanged accuracy-first selector** on 2026 outcomes (`scripts/evaluate_accuracy_first_ensemble.py` chronology: fit on earlier seasons, 2026 untouched holdout, then optional refit for 2027) | RB/WR (and only then QB/TE) point weights; still v3 = 0 unless it wins a marginal test | Same contract as 2026-08-27: MAE ≤ incumbent **and** Spearman ≥ incumbent, overall and by position; paired bootstrap excluding zero; hashes; **no** 2026 in the fit | Yes — `V1_PRODUCTION_ROLE_2026-08-29.md`, `STATE_OF_BUILD` scheduled follow-up |
| 2 | **Leakage-safe v1 `fantasy_evaluation`** for 2023/24/25/**26**, publish coverage map + starter slices | Nothing on the board. This is the scoreboard other experiments must beat | Current `compose_board` coverage keys (including concentration); `all_eligible` **and** starter-conditional tables; TE13 vs actual | Yes as an action item; still outstanding |
| 3 | **QB starter-conditional selector** (reuse `src/projection/qb_repair/` arms only if they beat baseline on 2026 *without* using 2026 for selection) | QB v1 rates or QB ensemble weights | Starter 8+ games MAE + Spearman gates that the 2026-09-03 repair already uses; non-QB invariance; no all-eligible Spearman as a pass | Partially — repair is NO-GO; the *metric* should become standard |
| 4 | **v3 means gate rematch** (`scripts/backtest_v3_means.py` + `scripts/v3_promotion_gate.py`) | Only if `promote_v3_means` is true on **every** rolling fold vs v1 **and** the then-incumbent blend | Existing gate, not a new one. Keep `--v3-means` default off until then | Yes — `V3_PROBABILISTIC_PIPELINE.md` |
| 5 | **Availability α nested-fit**, with the 2026-08-30 failure mode as a tripwire | `EXPOSURE_BLEND_ALPHA` only if a nested fit does **not** collapse to flat-17 after cold start, and starter + top-120 MAE both improve | The failed `shadow_availability_gate_a_blend_v1` hold is the negative control. Do not ship oracle “availability_only” | Yes as a closed candidate, not as a live knob |
| 6 | **TE replacement / VORP calibration** (research first: is the defect the 13th-TE level, the outlier McBride-class ceiling, or rookie TEs?) | TE prior / replacement rows / maybe TE ensemble — **not** RB/WR | Hold out 2026. Must not buy VORP by wrecking ρ. Compare to availability-adjusted baseline, which already wins VORP | Named as a watch item since the 2025 report |
| 7 | **WR/RB ceiling without reopening shadow repair** | Possibly concentration exponents, TE-style elite correction for WR, or “leave it to ADP” | Nested rolling-origin on rates **and** top-K tier hits. Current concentration fit already **rejected** promotion (`promoted=false`). A new WR elite correction needs the same three gates as TE (n above knot, positive β, consistency ≥ 2 SE) | Ceiling is a known miss; the closed repair track is **not** the vehicle |
| 8 | **Harness extensions** (roster_moves, vacancy α, curated-depth proxy) | Makes freeze knobs *measurable* next cycle | A historical proxy that does not use 2026 research files; fail closed if the proxy would leak | Provenance audit / ABLATION_RESULTS |

**Explicit non-experiments**

- Do not `promote_release` from this note.
- Do not reseal `v2_baseline_20260830` in place (including to add the 10k-draw
  risk field).
- Do not delete `output/shadow_v1_rb_wr/repair_track_closed.json` to “reopen”
  RB/WR.
- Do not fit new RB/WR weights on in-season 2026 weekly data.
- Do not treat Sleeper agreement as a gate.

---

## 6. Hypotheses (labeled; not used as findings)

These are consistent with the artifacts and **not** independently proven:

1. **H1.** Flattening exposure to 17 is correct for *draft EV* and is why
   Burrow/Daniels look “too high” on a mean board. A mid-α blend may help MAE
   and hurt decision quality. Needs nested fit + decision-quality gate, not
   another Sleeper delta.
2. **H2.** QB all-eligible **points** Spearman will keep looking great as long
   as the model can identify non-starters. Starter-conditional **rate** Spearman
   (and starter MAE) are the only honest promotion surface.
3. **H3.** WR zero-v1-weight is not “v1 is useless at WR”; it is “v1’s WR
   error is dominated by ceiling/injury that ADP already prices.” A v1 WR
   change that does not beat ADP on 2026 top-120 should not get weight.
4. **H4.** TE VORP is mostly a **replacement-level** problem plus a few
   outliers, not a failure to order the position. Fixing TE13 without a
   ceiling model will not recover McBride 2025.
5. **H5.** `reconcile_team_season_identities` is doing real work on season
   points and is invisible if you only look at `pred_pg`. Any future rate
   retune must re-score from `pred_season`.

---

## 7. Technical appendix

### 7.1 Training contract

- Population: skill-position players with source-season features, zeros kept
  when they are *role* zeros.
- Features: opportunity shares, RZ monopoly, air yards, snap %, OL scores,
  OC tendencies (inherited 60/40 for new seats), opponent EPA priors, age,
  `prior_{stat}_pg` in **label units**, depth tier.
- Models: 24 LightGBM rate/share cells + 4 Ridge team anchors + 4 LightGBM
  availability cells + OLS elite shrinkage (TE only) + rookie OLS-on-log-pick.
- Production models train on all four transitions; `backtest.py` refits on
  the first three to score 2025. `predict.py` refuses to run without
  `models/interval_residuals.csv`.

### 7.2 Share → yards

For `(WR|TE|RB, receiving_yards)`:

```
pred_pg_yards = pred_share_elig × share_scale × team_passing_yards_pg_pred
```

`share_scale` ≤ 1 when exposure-weighted WR+TE+RB shares (plus rookies’
implied shares) exceed the cap. Intervals are added in **rate units** after
composition.

### 7.3 Season identity

`TEAM_IDENTITY_PAIRS` **[code]** `team_reconcile.py`:

- receiving yards = passing yards
- receptions = completions
- receiving TDs = passing TDs
- targets = 0.952 × attempts (throwaways)

Split is geometric (half the log-gap on each side). `pred_pg` unchanged.

### 7.4 Scoring used in this note

- Eval MAE / Spearman / tier / VORP: `fantasy_evaluation` on
  `model_points_end_to_end` (composed `pred_season`).
- Tier K: QB12, RB24, WR36, TE12. Ties at the cutoff expand the set.
- VORP replacement: method-specific kth score (QB13, RB25, WR37, TE13).
- Accuracy-first: top-120 ADP only; isotonic ADP→points curves fit on earlier
  seasons; ECR diagnostic, zero weight.

### 7.5 What STATE_OF_BUILD still gets wrong (for this path)

| Claim | Current tree |
|---|---|
| `compose_board` is 15 stages including L2/L3 pass/rush mix | Slim 9-stage hygiene list in §2.2 |
| 2025 eval JSON lacks coverage keys / predates composition rewrite | JSON has coverage keys; numbers in §3.2 |
| Rate models lose RB targets and receptions | They win; RB receiving TDs lose |
| `FREEZE_2026-08-13.md` is the live manifest | Void; live seal is `v2_baseline_20260830` |

The freeze banner at the top of `STATE_OF_BUILD` (sealed namespace, α = 0,
RB/WR weights, 10k draws) **is** current.

---

## 8. Commands to reproduce cited numbers

All from repo root. None of these write production defaults.

```bash
# Tables in this note and in the 2026-09-15 per-position brief
# (stdlib; no DB, no models fit).
# Requires the committed output/ and models/ artifacts listed in the script;
# missing files exit with a relative-path list, not a traceback.
python3 scripts/research/dump_season_projection_regression_tables.py

# Inspect the JSON directly
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("output/fantasy_evaluation_summary_2025.json").read_text())
print("alpha", d["metadata"]["exposure_blend_alpha"])
print("coverage", json.dumps(d["metadata"]["composition_stage_coverage"], indent=2))
print("qb_starter", json.dumps(d["metadata"]["qb_starter_metrics"], indent=2))
for r in d["metrics"]:
    if r["scope"]=="all_eligible":
        print(r["position"], r["method"], round(r["spearman"],4),
              round(r["points_mae"],2), f"{r['tier_hits']}/{r['tier_rank']}",
              round(r["vorp_mae"],2))
PY

# Rate-model holdout (22/23 player cells; 24/25 rows including team pass)
column -s, -t output/backtest/veteran_holdout_2025.csv | head

# Accuracy-first holdout + weights
python3 - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("output/accuracy_first_2026/report.json").read_text())
g = r["holdout_2025"]["overall_gate"]
print("incumbent", g["incumbent"]["points_mae"], g["incumbent"]["spearman"])
print("selected", g["proposed"]["points_mae"], g["proposed"]["spearman"])
print(json.dumps(r["holdout_2025"]["final"]["bootstrap"], indent=2))
print(Path("output/accuracy_first_2026/ensemble_weights.json").read_text())
PY

# Confirm concentration is identity
python3 - <<'PY'
import json
from pathlib import Path
c = json.loads(Path("models/concentration_calibration.json").read_text())
for k,v in c["cells"].items():
    print(k, "exp", v["exponent"], "fitted", v["fitted_exponent"], "promoted", v["promoted"])
PY

# Live pointer (must remain v2_baseline_20260830)
python3 - <<'PY'
import json
from pathlib import Path
print(json.dumps(json.loads(Path("draft_assistant/data/active_release_2026.json").read_text()), indent=2))
PY
```

**Needs the NFL database — not run on this pass**

```bash
# Leakage-safe v1 scoreboard (refits through source_season; does not touch models/)
python -m src.projection.fantasy_evaluation
# optional: --source-season 2024 --target-season 2025

# Accuracy-first bake-off (does not overwrite native v1 CSVs)
python scripts/evaluate_accuracy_first_ensemble.py

# v3 means gate (does not enable --v3-means)
python scripts/backtest_v3_means.py
python scripts/v3_promotion_gate.py
```

`predict` / `compose_board` / `prepare` / `promote_release` were **not**
invoked.

---

## 9. Sources (read for this note)

- `STATE_OF_BUILD.md` (live freeze banner; stale §2.1 / §3.3)
- `docs/decisions/V1_PRODUCTION_ROLE_2026-08-29.md`
- `docs/decisions/ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`
- `docs/decisions/V3_PROBABILISTIC_PIPELINE.md`
- `FANTASY_EVALUATION_2025_REPORT.md` (historical run; correction banner)
- `docs/QB_PROJECTION_FINAL_REPAIR_REPORT.md`
- `docs/PIPELINE_MAP.md` (useful for features; compose-stage list not current)
- `src/projection/{train,predict,veterans,composition,team_reconcile,contracts,fantasy_points,fantasy_evaluation,evaluation/accuracy_first}.py`
- Follow-up brief: `docs/research/SEASON_PROJECTION_POSITION_MISS_BRIEF_2026-09-15.md`
- Artifacts: `output/fantasy_evaluation_summary_202{3,4,5}.json`,
  `output/fantasy_evaluation_2025.csv`,
  `output/backtest/veteran_holdout_2025.csv`,
  `output/accuracy_first_2026/{report,ensemble_weights}.json`,
  `output/model_accuracy_compare_2025.json`,
  `models/concentration_calibration.json`,
  `draft_assistant/data/active_release_2026.json`,
  `output/shadow_v1_rb_wr/repair_track_closed.json`
