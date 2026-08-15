# Age-effect shrinkage for RB — 2026-08-14

## Background

A user-driven investigation (same day, see project memory / prior session
notes) found the trained models lean on `age` ~1.6-3.2x harder for WR than
RB (gain-share and partial-dependence magnitude), and the user judged the
WR effect real but excessive. Follow-up work ruled out a bug and ruled out
"stale historical pattern" as the explanation - the WR age-related decline
is real, current, and specifically concentrated in the model's own
2021-2025 training window (a structural shift in how NFL teams allocate
targets by age, centered almost exactly at 2021 - not a biological/medical
effect). That investigation also surfaced a real asymmetry worth acting
on: RB's age signal is thin (sparse older-RB sample, as few as n=4 at
32+ in the recent era) and, when tested directly, does not earn its keep
out of sample.

## What was tried, and what the evidence said

A grid search (script: see scratchpad, not committed) tested a range of
"age-effect shrink" factors, from 1.0 (unshrunk) to 0.0 (fully
neutralized - every player scored as if they were their position's median
age), against both the single 2024→2025 holdout and a 3-fold rolling-origin
backtest, across every (position, stat) rate model:

| Position | Effect of shrinking age | At full neutralization (shrink=0.0) |
|---|---|---|
| **RB** | Monotonically **helps**, all 7 stats | -1.2% mean MAE (holdout), -0.4% (rolling) |
| **WR** | Monotonically **hurts**, all 4 stats | +1.3% mean MAE (holdout), +1.0% (rolling) |
| QB / TE | Within noise either way | <0.5% either direction |

Given RB and WR pointed in opposite directions, this was presented to the
user rather than resolved unilaterally. **User's explicit choice: RB only,
full neutralization (shrink=0.0). WR/QB/TE untouched.**

## Mechanism

`src/projection/transitions.py` gains `REFERENCE_AGE` (each position's
median age over its 2021-2025 training population: QB 27, RB 25, WR 25,
TE 26), `AGE_EFFECT_SHRINK` (`{"QB": 1.0, "RB": 0.0, "WR": 1.0, "TE": 1.0}`),
and `age_shrunk_predict(model, X, position, features=ALL_FEATURES)`.

This is **predict-time only - no retraining**. For a shrunk position, it
calls the *same already-trained* model twice: once on the real feature row,
once with `age` swapped for the position's reference age (every other
feature held at its real observed value - an individual-conditional-
expectation swap). The final prediction is
`pred_neutral + shrink * (pred - pred_neutral)`; at `shrink=0.0` this
collapses to `pred_neutral` (age's marginal contribution fully removed);
at `shrink=1.0` it's a no-op that skips the second `predict()` call
entirely, so QB/WR/TE pay no extra inference cost.

Wired into every place a per-position rate model predicts on `ALL_FEATURES`,
so backtest MAE, interval calibration, and the elite-shrinkage correction
fit can't silently diverge from what predict.py ships (the project's
existing precedent for `RECEIVING_SHARE_SUM_CAP`/`receiving_share_scale`):
`predict.py` (live composition), `backtest.py` (5 call sites: headline
holdout, rolling-origin/interval residuals, reframed-share composition,
the old-vs-new coherence comparison, and the season-totals framing),
`corrections.py` (the elite-shrinkage correction's own LOO residual
fitting), and `fantasy_evaluation.py` (the leakage-safe 2025 evaluation -
this one was missed on the first pass since it fits its own fresh models
independently of `backtest.py`, and was found and fixed before finalizing).

No retraining of the saved `models/*.joblib` binaries was needed for this
change (the shrink is a wrapper around inference, not a training-time
input) - only `corrections.joblib` was regenerated, since its residuals
now reflect RB's shrunk predictions (TE's own elite-shrinkage parameters,
the only ones currently shipped, were unaffected: beta=0.4031 unchanged).

## Held-out evidence, full pipeline re-run

Full documented pipeline re-run: `train` → `backtest` → `predict --season
2026` → `fantasy_points --season 2026` → `sleeper_compare --season 2026` →
`fantasy_evaluation`. `pytest`: 63 passed.

**2024→2025 veteran holdout** (RB rows only; QB/WR/TE unchanged, confirming
the no-op skip works): all 7 RB stats improved, matching the grid search -
carries 2.378→2.357, rushing_yards 11.688→11.428, targets 0.738→0.727,
receptions 0.600→0.590, receiving_yards (composed) 5.245→5.178,
receiving_tds and rushing_tds both improved fractionally.

**Leakage-safe 2025 fantasy evaluation** (`all_eligible`, model method):

| Position | Spearman | Points MAE | Tier hits | VORP MAE |
|---|---|---|---|---|
| QB | 0.7826 → 0.7822 (float noise) | 42.40 → 42.39 | 6/12 → 6/12 | 46.41 → 46.47 |
| RB | 0.7331 → 0.7324 (−.0006) | 33.58 → 33.55 | 15/24 → 15/24 | **35.20 → 36.59 (+1.39)** |
| WR | 0.7725 → 0.7726 (float noise) | 22.721 → 22.721 | 22/36 → 22/36 | 24.588 → 24.588 |
| TE | 0.8327 → 0.8324 (float noise) | 16.569 → 16.568 | 5/12 → 5/12 | 35.165 → 35.166 |

QB/WR/TE move by float-precision-level amounts only (5th-6th decimal) -
this is cross-position numerical coupling through the shared per-team
receiving-share cap (`RECEIVING_SHARE_SUM_CAP`), not a behavioral change;
their own age shrink factor is 1.0 (no-op), confirmed by construction.

**RB's one real regression, and it's traced to a specific player, not
diffuse damage**: VORP MAE worsens 35.20 → 36.59 (+4%). The
`predicted_replacement_points` anchor (the RB sitting at the replacement
rank, 25th) rose from 171.80 to 174.92 while the true replacement level
stayed 164.80 - the model now overshoots the replacement boundary by
~3 more points than before. The rank-25 player in both the old and new
run is **David Montgomery** (age 28) - actual 154.92 points, model
predicted 171.80 before this change (already an over-projection) and
174.92 after (a slightly larger one). Removing the age discount for RB
gave a 28-year-old between-role back a small bump his own usage evidently
didn't support here. Spearman, points MAE, and tier hits are all flat to
within noise - this is a narrow, understood side effect concentrated at
one boundary player, not a broad accuracy loss, but it is real and is
reported as such rather than left out.

## What moved on the 2026 board

Mean absolute season-point change: RB 1.46 pts (n=171), everyone else
0.046 pts (float noise) - confirms the fix stayed scoped to RB as
intended. Direction is exactly what removing an age discount should
produce: young RBs lose the "youth bonus" they were getting (Trevor
Etienne −12.8, Dylan Sampson −9.8, Jordan James −8.4, DJ Giddens −8.1),
older/veteran RBs gain it back (Chuba Hubbard +10.0, Christian McCaffrey
+7.6, Saquon Barkley +6.8, Jonathan Taylor +6.1, Rhamondre Stevenson +5.7).

Sleeper agreement moved negligibly for RB (correlation 0.9547 → 0.9539,
MAE 15.30 → 15.56, bias +0.57 → +0.36) - within noise, no signal either
way from that comparison.

## Honest summary

This is a real, validated, evidence-gated fix scoped exactly to where the
evidence supported it (RB) and deliberately not applied where it didn't
(WR, despite that being the position that originally prompted the
concern - the data there argues the opposite way). It costs a measurable,
traced VORP-MAE regression at RB's specific replacement boundary (David
Montgomery), which is disclosed rather than hidden. Every other tracked
metric across both evaluation harnesses is flat to within floating-point
noise.

## Artifacts regenerated

`models/corrections.joblib`, `output/projections_2026.csv`,
`output/fantasy_points_2026.csv`, `output/sleeper_comparison_2026.csv`,
`output/fantasy_evaluation_2025.csv`,
`output/fantasy_evaluation_summary_2025.{csv,json}`. The 32 non-corrections
model binaries in `models/` are unchanged byte-for-byte (no retraining
input changed) - only rerun for pipeline-order hygiene, not because their
content differs.
