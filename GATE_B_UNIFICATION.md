# Gate B depth-rate ladder: one rule, three paths

**Date:** 2026-08-15
**Scope:** `depth_gating.py`, `depth_rates.py`, `veterans.py`, `fantasy_evaluation.py`,
`backtest.py`, `corrections.py`, `team_reconcile.py` (docstring only)
**Test baseline:** 136 passed → **155 passed, 4 subtests** (19 added, none weakened,
skipped or deleted)

---

## 0. Verdict in one paragraph

The curated-chart gate on the Gate-B ladder was a **latent bug**, and the git history
shows exactly where it was introduced and where its justification expired. It has been
removed; the ladder now applies from `nfl_depth_rank` alone on all three paths, through
one shared helper. `backtest.py` and `corrections.py` were refit on the discounted basis
production actually ships. **The leakage-safe 2025 fold is bit-identical before and
after** — the acceptance test the ablation predicted, and it passes exactly. The 2026
board's point predictions move on **152 of 4,039 cells, all TE, max 0.76 season yards**;
the top-50 fantasy-points board is unchanged in membership *and* order. Prediction
intervals move materially and in the direction the fix implies. **Recommendation: ship**,
with one caveat about what this does and does not prove, in §8.

---

## 1. Phase 1 — bug or deliberate choice?

**Latent bug.** Not a judgment call — the guard predates the change that invalidated it,
and can be dated.

### 1.1 The guard's original justification was true when written

`05bdc91` ("Phase 6: 2026 team reassignment + curated depth chart gating") introduced
`apply_depth_chart_gating` together with `if depth_chart.empty: … return`. At that
commit the multiplier came from the **curated table**: rows off the table got
`DEEP_BENCH_DISCOUNT = 0.15`, rows on it got `ROLE_VOLUME_DISCOUNT[role]`. With no
curated table there is no `role`, so "no chart ⇒ no multiplier" was simply correct, and
the `depth_chart_status = "not_curated_no_table"` label the branch sets — documented in
that same commit as *"gating is a no-op, not a claim about this player's role"* — was an
accurate statement.

### 1.2 Gate B moved the multiplier off the curated table and did not revisit the guard

`720fa8e` ("Gate B: fit the volume discounts against outcomes instead of asserting
them") replaced both constants with `DEPTH_RATE_LADDER`, keyed on `nfl_depth_rank`.
Its own commit body states the re-keying explicitly:

> `DEPTH_RATE_LADDER` is keyed on the nflverse preseason rank, not the curated role,
> because rank resolves a gradient the role tiers cannot […] The curated chart stays
> authoritative for membership, team, and displayed role.

The diff of that commit (verified with `git show 720fa8e -- src/projection/predict.py`)
rewrites the body of `apply_depth_chart_gating` — the factor lookup, the discount mask,
the `low_confidence` semantics — and **leaves the `depth_chart.empty` early return
untouched**, including its `role_discount_factor = 1.0`. Nothing in the commit body
discusses the empty-chart case.

### 1.3 The signal the guard suppresses exists for every season, by design

`8be9c63` ("Gate A") exists precisely to establish that:

> The "no historical curated depth chart exists" blocker cited in
> `train.fit_availability` and `backtest.py` was false. `depth_charts` in the project DB
> covers 2016-2026.

`veterans.py:109-116` states the split in the shipped code today: the curated file
"exists only for 2026, so it can never be a trained-on feature. The nflverse chart
exists for every season, which is the only reason the availability model can be honestly
held out on it."

So the pre-fix code gated a **chart-independent, historically-calibrated** factor on the
existence of a **2026-only research file**. `load_depth_chart` (`depth_gating.py:34`)
returns an empty frame for every season except 2026 by an explicit `if target_season !=
2026` test, so the ladder ran on the shipped path for exactly one season in the project's
history.

### 1.4 Nothing in the docs claims the gate was intentional

Searched `docs/history/PHASE6_REPORT.md`, all eight files in `docs/decisions/`,
`PROVENANCE_AUDIT.md`, `SLEEPER_RETIREMENT.md` and `ABLATION_RESULTS.md` for
`not_curated_no_table`, `depth_chart.empty`, `Gate B`, `DEPTH_RATE_LADDER`.
`PHASE6_REPORT.md:164` documents the *label*, in the pre-Gate-B sense. No document
argues for gating the ladder on the curated chart. `ABLATION_RESULTS.md` §5 independently
reached the same "latent bug, with a benign cover" conclusion.

**Was it dormant?** On the shipped 2026 board, yes — 2026 has a chart, so the ladder
already applied. It becomes live the moment `starters_2027.csv` does not yet exist, or
anyone runs `project_season` for a historical season. **Verified**: `predict --season
2025` now runs the previously-dead branch end to end and discounts **348 of 562 veteran
players (62%)**, mean factor QB 0.881 / RB 0.848 / TE 0.931 / WR 0.951. (`ABLATION_RESULTS.md`
reports 287/661 = 43% on the *evaluation harness* population; different frame, same
order of magnitude — both are large.)

---

## 2. Phase 2 — refitting `backtest.py` on the shipped basis

### 2.1 What was mismatched

`backtest.py` imported neither `depth_gating` nor `depth_rates`. Every prediction it
made was undiscounted, and two production artifacts are fit from it:

| Artifact | Fit as | Consumed as |
|---|---|---|
| `models/interval_residuals.csv` | `actual − pred_undiscounted` | added to `pred_discounted` |
| `models/corrections.joblib` (β) | OLS on `actual − pred_undiscounted` above a knot | **additive** yards/game term added to `pred_discounted`, then scaled by the same factor (`team_reconcile.py:264`) |

The corrections case is the sharper one: β is additive, so a coefficient fit against
undiscounted residuals was being fit for one prediction and added to a different one.

### 2.2 A prerequisite finding: both on-disk artifacts were already stale

Before changing anything, I re-ran `backtest.py` unmodified. It did **not** reproduce
the checked-in `models/interval_residuals.csv`: QB rows matched exactly, but RB
316→315, TE 319→317 and WR 537→539 rows. `models/corrections.joblib` on disk carried
TE β = 0.4031 (consistency 2.578); recomputed on current code with the ladder still off
it is β = 0.3396 (consistency 2.285). Both artifacts predate some upstream change to the
transition/feature population. `models/` is gitignored, so this drift is invisible to
`git status`.

Every "before" number below is therefore reported against the **recomputed
ladder-off** artifacts, not the stale on-disk ones, so the delta is attributable to the
ladder rather than to accumulated drift. Both are shown where the distinction matters.

### 2.3 Interval residuals — before vs after

Widths narrow slightly and the band shifts right, exactly as a discounted prediction
implies (predictions are smaller, so `actual − pred` is more positive).

| Position | mean width change | rows |
|---|---|---|
| QB | −6.2% | 192 |
| RB | −0.7% | 315 |
| TE | −3.1% | 317 |
| WR | −2.1% | 539 |
| **All 23 (position, stat) cells** | **mean −3.29%, median −2.99%** | 19 narrowed, 4 widened |

Largest single move: `QB passing_yards` low −103.47 → −83.48, high +78.21 → +92.77
(width −3.0%, but the centre shifts +17 yards/game).

**Forward interval coverage** (`interval_forward_coverage.csv`, 46 cells, calibrated on
earlier folds only, 80% nominal):

| | before (ladder off) | after (ladder on) |
|---|---|---|
| **Overall** | **0.8162** | **0.8102** |
| QB | 0.8348 | 0.8329 |
| RB | 0.8211 | 0.8091 |
| TE | 0.8033 | 0.7964 |
| WR | 0.7837 | 0.7802 |
| 2024 fold | 0.7815 | 0.7733 |
| 2025 fold | 0.8510 | 0.8470 |

The previously documented figure was 0.820; the recomputed ladder-off baseline is 0.8162
and the unified basis is 0.8102. **This is a −0.6pp move on 46 cells of 60–190 rows
each; I read it as noise, and I would not claim the fix improved coverage.** It remains
within one point of nominal. Largest per-cell swings go both ways (`QB attempts` 2025
+6.4pp, `QB passing_tds` 2024 −6.3pp).

### 2.4 Corrections β — before vs after

Fit via `corrections.compute_loo_receiving_residuals` over `ALL_PAIRS`, with the ladder
applied to the share before renormalization (the order production uses).

| | on-disk (stale) | recomputed, ladder OFF | recomputed, ladder ON |
|---|---|---|---|
| TE β | 0.4031 | 0.3396 | **0.3822** (+12.5% vs ladder-off) |
| TE `season_consistency` | 2.578 | 2.285 | **2.947** |
| TE n above knot | 23 | 23 | 23 |
| WR | not fit | not fit (consistency 0.856) | not fit (consistency 0.865) |

**This is the finding the brief asked to be watched for, and it points the good way.**
`ABLATION_RESULTS.md` flagged that β "sits at consistency 2.1 against a gate of 2.0 —
one fold from failing its own test." On the recomputed ladder-off basis it is 2.285;
on the unified basis it is **2.947**, roughly 50% more headroom over
`MIN_SEASON_CONSISTENCY = 2.0`. The mechanism is visible in the per-transition means:
ladder off, TE above-knot residuals run +4.77 / +5.97 / +1.72 / +0.03 across
2021→2024; ladder on, +4.77 / +7.71 / +3.18 / +0.99. The effect is present in every
transition rather than carried by two, which is precisely what the consistency gate was
built to distinguish. WR still correctly fails.

### 2.5 An interval-construction consequence, fixed rather than routed around

Once the residual band is calibrated on a *discounted* prediction, the shipped
non-reframed interval `low = (pred + resid) · f` double-applies the discount: it hands
a QB2 an `0.77×`-narrowed version of a band already calibrated for a QB2. Reframed
receiving rows never had this problem — `_compose_reframed_receiving_predictions` adds
the residual absolutely, and its docstring said so while explicitly naming the
undiscounted calibration as the reason it was the "more faithful reading."

Endpoint construction for non-reframed veteran rows therefore moved **after** the ladder
(`veterans._attach_veteran_intervals`), matching the reframed convention. Point
predictions are untouched by this; only `pred_pg_low`/`pred_pg_high` move. Quantified
in §7.

### 2.6 What was deliberately *not* changed

`backtest.depth_rate_calibration` still fits on undiscounted predictions. It is the
ladder's own calibration table; feeding it the ladder would be circular. **Verified
byte-identical** before and after, and pinned by
`test_depth_rate_calibration_stays_undiscounted`.

---

## 3. Phase 3 — the unified rule

> The Gate-B factor is a function of `(position, nfl_depth_rank)` and nothing else. It
> applies to every veteran row, in every season, on every path. The curated chart governs
> what it is authoritative for — membership, team, displayed role, formation role,
> `depth_chart_status`, `low_confidence` — and none of those select a multiplier.
> Rookie rows are excluded, because the ladder was fit on veteran transition pairs and
> measured harmful on the rookie test.

One helper, in `depth_rates.py`:

```python
LADDER_SCALED_COLUMNS = ("pred_pg", "pred_pg_low", "pred_pg_high")
depth_rate_factor(position, rank)            # scalar (unchanged)
depth_rate_factors(positions, ranks)         # vector
attach_depth_rate_factor(df, …)              # writes role_discount_factor, scales nothing
apply_depth_rate_ladder(df, …)               # attach + scale: THE application
```

| Path | before | after |
|---|---|---|
| `depth_gating.apply_depth_chart_gating`, curated branch | inline list comprehension + inline scaling loop | `apply_depth_rate_ladder(df)` |
| `depth_gating.apply_depth_chart_gating`, empty-chart branch | `role_discount_factor = 1.0`, early return | `apply_depth_rate_ladder(df)`; curated fields still no-op and say so |
| `fantasy_evaluation._veteran_forecasts` | inline per-position list comprehension | `depth_rate_factors(...)` |
| `backtest` non-reframed (`backtest_position_stat`, `rolling_residual_rows`, `backtest_season_totals`, `coherence_ratio_backtest` old arm) | not applied | `× depth_ladder_factors(...)` |
| `backtest._predict_all_reframed_receiving` | not applied | factor on the **share**, before `receiving_share_scale`; elite `adj` scaled by the same factor, mirroring `team_reconcile.py:264` |
| `corrections.compute_loo_receiving_residuals` | not applied | factor on the share, same order |
| `backtest.depth_rate_calibration` | not applied | **still not applied** (circular) |

`attach_depth_rate_factor` raises when `nfl_depth_rank` is absent instead of defaulting
to NaN, because NaN means "off the preseason chart" and carries a real discount — the
same guard the curated branch already had, now covering the empty branch too. This is a
behaviour change for any external caller that passed a rank-less frame with an empty
chart; the error names the fix.

### 3.1 Regression tests added — `tests/test_depth_rate_ladder_unification.py`

19 tests in four groups: the factor's inputs (`position`, `rank`, nothing else, unknown
position ⇒ 1.0, misaligned vectors rejected, missing rank column raises); the regression
itself (empty chart still discounts, curated fields still report ignorance, same player
gets the same factor with and without a curated file); source-level guards that each of
the five paths still routes through the shared helper *and* that
`depth_rate_calibration` does not; and the interval convention (raw residual on a
discounted prediction, reframed rows deferred, missing residual flags instead of
borrowing, endpoint attachment ordered after gating).

The source-level guards are deliberate: re-implementing the ladder inline on one path is
exactly how the three-way disagreement happened.

---

## 4. Acceptance test — leakage-safe 2025 fold, all four positions

Run: `python -m src.projection.fantasy_evaluation` (source 2024 → target 2025),
`scope = all_eligible`, `method = model`.

| Position | Spearman before | Spearman after | Points MAE before | Points MAE after | Tier hits before | Tier hits after |
|---|---|---|---|---|---|---|
| QB | 0.791726 | 0.791726 | 40.604510 | 40.604510 | 8 / 12 | 8 / 12 |
| RB | 0.689090 | 0.689090 | 35.787058 | 35.787058 | 14 / 24 | 14 / 24 |
| WR | 0.786135 | 0.786135 | 23.520593 | 23.520593 | 22 / 36 | 22 / 36 |
| TE | 0.831807 | 0.831807 | 17.147358 | 17.147358 | 4 / 12 | 4 / 12 |

`scope = forecast_covered`, same picture (QB 0.797119 / 46.252116 / 8, RB 0.739600 /
40.183370 / 14, WR 0.796457 / 25.429802 / 22, TE 0.838407 / 18.902167 / 4).

**Max |Δ| across every metric and every row of the summary — including `carry_forward`
and `availability_adjusted` methods, VORP MAE and VORP Spearman — is exactly 0.**

**Unchanged, not improved. RB does not degrade — but RB is also not tested here.** That
is the intended result and it is worth being blunt about what it means: the harness was
already applying the ladder unconditionally, so unification moved the other two paths
onto the behaviour that was already being measured, rather than onto a fourth. The
invariance proves the direction of the merge. It supplies **no new evidence** about
whether the ladder itself is a good idea.

---

## 5. Supporting measurements from the refit backtest

These are not the acceptance test. They are what the refit made visible, and they cut
against the ablation on RB, so both are reported.

### 5.1 Per-game rate MAE, 2024→2025 holdout

21 of 25 rows improve; mean −3.17%. QB `attempts` (+0.45%) and QB `carries` (+0.75%)
worsen slightly; the two TEAM rows are unaffected by construction.
Largest: TE `receiving_yards` 6.723 → 6.256 (−7.0%), RB `receiving_yards` 5.307 → 4.983
(−6.1%), QB `passing_yards` 54.56 → 51.29 (−6.0%).

### 5.2 Rolling-origin fold means (expanding window, 3 folds)

23 of 25 rows improve; mean −4.17%. The two non-improving rows are the TEAM rows
(unchanged). **Every player-level (position, stat) improves, including all seven RB
rows.**

### 5.3 Season-total framing, shipped `rate × predicted games`, `all_source_players`

| Position | stat | before | after |
|---|---|---|---|
| QB | passing_yards | 651.78 | **629.32** |
| RB | rushing_yards | 165.62 | **164.56** |
| WR | receiving_yards | 130.89 | **130.22** |
| TE | receiving_yards | 79.59 | **76.56** |

All four improve.

### 5.4 How to read §5 honestly — and the RB contradiction

`DEPTH_RATE_LADDER`'s constants were fit leave-one-transition-out on 2021–2025, and
these folds sit inside that window. §5 is therefore a **consistency check, not
independent validation**: the ladder was fit to drive `sum(actual)/sum(pred)` toward 1
by rank, and it does. It also measures *rate and single-stat season yards*, not fantasy
points across a composed board.

`ABLATION_RESULTS.md` measured the opposite sign for RB on the thing that actually
matters — removing the ladder **improves** RB fantasy-points MAE by 0.653 (inside the
bootstrap interval, so individually insignificant, but pointing the same way as two
other RB-allocation arms). §5.2 and §5.3 say RB rate and RB season rushing yards both
get *better* with the ladder. **These are different measurements and they disagree.
Neither is decisive.** The disagreement is not resolved by this work and should not be
read as resolved: the ablation's fantasy-points view is the more decision-relevant one,
and its RB signal stands. §9 keeps that follow-up open.

---

## 6. What did *not* change on any path

* 2026 point predictions for QB, RB and WR: **zero cells moved.**
* `models/depth_rate_calibration.csv`: byte-identical.
* Injury-cohort gate: NOT tripped before and after (mean resid −4.42 → −3.51, 45%
  positive both ways).
* Positions shipping an elite correction: `['TE']` before and after. WR still fails its
  consistency gate.
* Team-level backtest rows (`TEAM passing_yards`, `TEAM pass_attempts`): unchanged.

---

## 7. 2026 board delta

Baseline is the on-disk `output/projections_2026.csv` snapshotted before any edit — the
board regenerated with `INCUMBENT_VACANCY_ALPHA["carry"] = 0.0`. **Verified deterministic
first**: re-running `predict --season 2026` against the untouched baseline artifacts
reproduced that file with **0 cells differing** across all 83 columns and 4,039 rows.

### 7.1 Total, baseline → shipped

4,039 rows × 83 columns. **9,914 cells moved**, of which:

| Column | cells moved | up | down | max abs | mean signed |
|---|---|---|---|---|---|
| `pred_pg` | 152 | 124 | 28 | 0.0592 | +0.0001 |
| `pred_season` | 152 | 124 | 28 | 0.758 | ~0 |
| `pred_pg_low` | 1,317 | 1,124 | 193 | 39.84 | +0.366 |
| `pred_pg_high` | 2,832 | 2,712 | 120 | 92.34 | +1.157 |
| `elite_correction_pg` | 7 | 0 | 7 | 0.232 | −0.0016 |
| reconciler scale/ratio columns | 1,132 | — | — | ≤0.0068 | ~0 |
| `stat_constraint_applied` (flag) | 21 | — | — | — | — |

**Point predictions, by position:**

| Position | `pred_pg` moved / total | up | down | mean Δ | max abs Δ (`pred_season`) |
|---|---|---|---|---|---|
| QB | **0 / 768** | — | — | — | — |
| RB | **0 / 1,267** | — | — | — | — |
| WR | **0 / 1,308** | — | — | — | — |
| TE | 152 / 696 | 124 | 28 | +0.0014 pg | 0.758 season yards |

Every point-prediction move is TE, and every one traces to the corrections β
(0.4031 → 0.3822) plus the reconcilers redistributing within the team pass budget —
`pred_season` sums to zero change, as the anchors require. Largest movers: Brock Bowers
−0.76, Tucker Kraft −0.65, George Kittle −0.57, Michael Mayer +0.52 season receiving
yards.

**Interval endpoints, by position** (`pred_pg_low` / `pred_pg_high` rows moved of total):

| Position | low moved | high moved | mean Δ low | mean Δ high |
|---|---|---|---|---|
| QB | 306 / 768 | 519 / 768 | +3.417 | +5.665 |
| RB | 400 / 1,267 | 917 / 1,267 | +0.132 | +1.401 |
| TE | 206 / 696 | 536 / 696 | +0.644 | +0.072 |
| WR | 405 / 1,308 | 860 / 1,308 | +0.613 | +0.476 |

Interval **width** change on veteran rows splits cleanly along the discount, which is
the signature of the §2.5 fix:

| Cohort | n | mean width Δ | median |
|---|---|---|---|
| `role_discount_factor < 1` | 1,802 | **+15.19%** | +12.02% |
| `role_discount_factor = 1` | 1,085 | **−1.86%** | −1.29% |

Undiscounted rows narrow by the ~2–3% the residual refit implies. Discounted rows widen
by ~15% because they stop having the discount applied to their band twice.

### 7.2 Attribution (extra run, ladder-off artifacts + new code)

* **Artifact staleness + interval-ordering fix**: 9,265 cells; 172 TE `pred_pg` rows.
* **Ladder-basis refit alone**: 8,822 cells; 172 TE `pred_pg` rows (mean −0.0028 pg),
  1,262 `pred_pg_low` and 2,229 `pred_pg_high`.

Neither component moves a QB, RB or WR point prediction.

### 7.3 Downstream

`output/fantasy_points_2026.csv` regenerated: **38 of 778 player rows move, all TE,
max 0.13 season fantasy points, mean ≈ 0.** The top-50 board is **identical in
membership and in order**. `draft_assistant/data/players_2026.json` regenerated from
the new board.

---

## 8. Verified vs inferred

**Verified by running it:**

* Baseline board reproduces bit-for-bit (0/335,237 cells differ) — so every delta above
  is signal, not run-to-run noise.
* 136 → 155 tests pass; no test weakened, skipped or deleted.
* 2025 fold metrics identical to machine precision on every position, scope, method and
  metric.
* Every number in §2.3, §2.4, §5, §6 and §7 is from a run in this session.
* On-disk `interval_residuals.csv` and `corrections.joblib` were stale relative to
  current code, independent of this change.
* `predict --season 2025` executes the previously dormant branch without error and
  discounts 348/562 veterans.
* `depth_rate_calibration.csv` byte-identical.

**Established from git, not from a run:**

* The intent verdict in §1. The chain — guard introduced in `05bdc91` when it was
  correct, invalidated by `720fa8e`, never revisited — is read off commit diffs and
  bodies. I did not find a document arguing the gate was deliberate, but absence of
  evidence in the docs I searched is weaker than the positive evidence in the diffs.

**Inferred / not established:**

* That the unified rule is *better*. The 2025 fold is invariant by construction, so it
  cannot say. §5 is inside the ladder's own fitting window.
* That RB's ladder rungs are fine. §5 says they help rate and season yards; the ablation
  says they hurt fantasy points. Unresolved.
* Coverage: 0.8162 → 0.8102 is reported as noise, not as an improvement or a regression.
  I did not bootstrap it.
* The predict-path change for non-2026 seasons is **unmeasured by any harness**.
  `fantasy_evaluation` reaches the ladder through `_veteran_forecasts`, not through
  `apply_depth_chart_gating`; `compose_board` never calls the latter. The empty-chart
  branch is covered by unit tests and a smoke run, not by a scored fold.

---

## 9. Recommendation

**Ship it.** The reasoning, in order of weight:

1. **The correctness argument is independent of the performance argument.** Fitting an
   additive correction and a residual band against one prediction and applying them to
   a different one is wrong regardless of which prediction is better. §2 fixes that, and
   it is the half of this work that would be worth doing even if the ladder were
   deleted tomorrow.
2. **The blast radius on what ships is close to nil.** 152 TE cells, max 0.76 season
   yards, top-50 fantasy board unchanged in order. QB, RB and WR point predictions do
   not move at all.
3. **The acceptance test passes exactly.** The three paths were merged onto the
   behaviour the evaluation harness was already measuring, not onto a fourth one.
4. **The corrections β came out of the refit with more evidence behind it, not less** —
   consistency 2.285 → 2.947 against a gate of 2.0, and present in all four transitions
   rather than two. That was the identified risk, and it resolved favourably.
5. **The latent half is now safe.** Before this, generating a 2027 board before
   `starters_2027.csv` existed would have silently dropped a calibrated signal from 60%
   of veterans with no warning anywhere in the output.

**Caveats to carry forward, not to bury:**

* Ship the **refit artifacts together with the code**. `models/` is gitignored, so a
  machine with the old `interval_residuals.csv` and the new `veterans.py` gets intervals
  that are neither convention. Re-run `python -m src.projection.backtest` and re-fit
  `corrections.joblib` (via `train.py`, or the fit path in §2.4) on any checkout that
  ships this.
* `models/` drift is a real, invisible hazard — both artifacts were already stale before
  I touched anything. Worth a manifest/hash check as its own piece of work.
* **The RB rungs remain open.** `ABLATION_RESULTS.md`'s recommendation 3 — re-fit
  `DEPTH_RATE_LADDER` per position scored on fantasy points, starting with RB's
  `{1: 1.00, 2: 0.98, 3: 0.73}` and `DEPTH_RATE_DEEP["RB"] = 0.70` — is unaffected by
  this change and is still the better use of a run than anything on this page. This work
  makes that re-fit *possible to do honestly*, because all three harnesses now agree on
  what "the ladder is applied" means.
* Consider extending `fantasy_evaluation` to route veteran gating through
  `apply_depth_chart_gating` rather than calling the factor lookup directly, so the
  empty-chart branch is exercised by a scored fold rather than only by unit tests.
