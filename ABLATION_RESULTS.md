# Ablation of the allocation layer against the leakage-safe 2025 fold — 2026-08-15

Scope: `src/projection/`. Harness: `src.projection.fantasy_evaluation.run_evaluation()`
(2024 → 2025, leakage-safe), which since the composition unification runs
`composition.compose_board` — the same stage sequence `predict.project_season`
ships.

**Method.** No constant was edited on disk to measure it. Each arm ran in its
own Python process and rebound the relevant module-level constant or stage
function immediately before `run_evaluation()`. Nothing can leak between arms
and nothing is left changed. The runner is
`<scratchpad>/ablate.py`; per-arm JSON summaries and per-player error dumps are
in `<scratchpad>/results/`.

**Determinism.** The pipeline is bit-deterministic across processes: every arm
that turned out to be inert reproduced the baseline's total-points fingerprint
(39529.48289411105) exactly, and an independent baseline re-run reproduced it
too. So every delta below is a real, repeatable consequence of the knob — the
open question for each is generalisation, not measurement noise in the run.

**One fold.** There is exactly one fold here. A paired bootstrap over the 2025
player population (below) says how much of each delta survives resampling
*those players*; it says nothing about other seasons. Spearman deltas are the
weakest evidence of all — the marginal SE of Spearman at n=185 is ≈0.074, and
no arm moves any position's Spearman by more than 0.027.

---

## 1. Baseline

Scope `all_eligible`, method `model`. This is the current working tree,
**before** the Task-2 change.

| Position | n | Spearman | Points MAE | Tier hits | VORP MAE |
|---|---:|---:|---:|---:|---:|
| QB | 107 | 0.791726 | 40.6045 | 8 / 12 | 46.84 |
| RB | 185 | 0.689090 | 35.7871 | 14 / 24 | 43.67 |
| WR | 298 | 0.786135 | 23.5206 | 22 / 36 | 23.99 |
| TE | 165 | 0.831807 | 17.1467 | 4 / 12 | 26.02 |

Scope `forecast_covered`: QB 0.7971 / 46.252 / 8; RB 0.7396 / 40.183 / 14;
WR 0.7965 / 25.430 / 22; TE 0.8384 / 18.901 / 4. Every conclusion below is the
same on both scopes, so only `all_eligible` is tabulated from here on.

### A discrepancy with the task brief, stated up front

The brief gives the current baseline as **RB 34.95 points MAE and 13 tier
hits**, improving to 34.07 / 16 with the stage-order fix. I measure the
baseline at **RB 35.787 / 14**, and the stage-order fix moves RB by **exactly
zero** (§4). I could not reproduce either the briefed baseline or the briefed
improvement on this tree. I did not chase the divergence to a commit; what I
can say is that the numbers below are internally consistent, deterministic,
and reproduced twice.

---

## 2. What is measurable on this fold, and what is not

Six of the briefed candidates produced a result **byte-identical to baseline**.
That is not a neutral result — those knobs are *unreachable* on a 2025 fold.
Reporting them as "no effect" would be the same category error the audit warns
about, so the mechanism was verified for each rather than inferred
(`<scratchpad>/probe.py`):

| Candidate | Result | Why it is unreachable |
|---|---|---|
| `FORMATION_ROLE_BLEND_W` 0.5 → 0.0 | identical | The composition input frame has **no `formation_role` column at all** for 2025. `_allocate_wr_by_formation_role` sees `known.any() == False` and falls straight through to `_allocate_within_group`. `formation_role` comes only from the curated chart, which is 2026-only. |
| `INCUMBENT_VACANCY_ALPHA["carry"]` 1.0 → 0.0 | identical | `fantasy_evaluation.py` does not import `roster_moves` at all. `apply_incumbent_vacancy_boost` is called from `veterans.project_veterans`, which is on the shipped forecast path only — upstream of `compose_board`, which is where unification stopped. |
| `INCUMBENT_VACANCY_ALPHA["target"]` 0.5 → 0.0 | identical | same |
| `TEAM_CHANGE_VACANCY_ALPHA` → {0, 0} | identical | same (`reassign_team_changers`); the harness already documents this in `coverage_limits`. |
| `USAGE_SHARE_CURATED_W` 0.5 → 0.0 | identical | `apply_usage_share_prior` runs, but with `USAGE_SHARE_BLEND_W = 0.0` and no curated `usage_share_prior` column it returns at the early exit. `usage_share_blend_factor` is non-null on **0 of 3,476** rows. The whole stage is inert on this fold. |
| `NAMED_RUSH_COVERAGE` 0.814 → 0.0 | identical | The floor never binds. **0 of 32** teams have named carry supply below 0.814× the anchor (min ratio 0.889, median 0.973); 0 of 32 for rush yards (min 0.827). |
| `RUSH_ATTEMPTS_PER_APPEARANCE_MAX` / `RUSH_YARDS_PER_CARRY_MAX` → unbounded | identical | 0 of 247 carry rows sit on a ceiling. Confirms the audit's "currently non-binding". (2 of 91 QB attempt rows *do* sit on `QB_ATTEMPTS_PER_VOLUME_GAME_MAX`.) |

The harness's own `composition_stage_coverage` map agrees, and names the cause:
`_curated_depth_chart: absent — starters_2025.csv does not exist`.

**Consequence for §5 of the audit.** Its rank-1 candidate
(`FORMATION_ROLE_BLEND_W` + `WR_FORMATION_ROLE_PRIORS`) and its rank-3
(`INCUMBENT_VACANCY_ALPHA["carry"]`) remain **unmeasurable**, even after the
unification. The unification moved the boundary to `compose_board`; the vacancy
alphas and the roster-move layer sit upstream of it. The audit's §4.5 item 1 is
therefore only half done.

---

## 3. Per-candidate deltas (measurable arms)

Deltas vs. baseline, `all_eligible`. Negative MAE = the ablation is better than
what ships. Positive tier-hit delta = the ablation is better.

| Arm | QB ΔSpear / ΔMAE / ΔHits | RB | WR | TE |
|---|---|---|---|---|
| `depth_rate_off` (Gate B ladder → 1.0) | −0.0034 / +0.158 / −1 | +0.0028 / **−0.653** / 0 | −0.0002 / +0.023 / 0 | −0.0055 / **+0.567** / 0 |
| `depth_rate_ladder_only` (on-chart rungs + DEEP → 1.0) | −0.0034 / +0.158 / −1 | +0.0034 / −0.665 / 0 | −0.0009 / +0.058 / 0 | −0.0048 / +0.553 / 0 |
| `depth_rate_off_chart_only` (OFF_CHART → 1.0) | 0 / 0 / 0 | −0.0009 / +0.011 / 0 | +0.0006 / −0.040 / 0 | −0.0008 / +0.014 / 0 |
| `no_hier_pass_l3` (L3 distribution off, L2 mix attached) | 0 / 0 / 0 | −0.0002 / −0.045 / 0 | −0.0002 / −0.088 / 0 | +0.0004 / −0.058 / 0 |
| `no_hier_pass_l2l3` (whole pass mix layer off) | 0 / 0 / 0 | −0.0024 / **−1.259** / +1 | −0.0020 / +0.096 / 0 | +0.0043 / −0.064 / 0 |
| `no_hier_rush_l3` (L3 distribution off) | −0.0085 / **+1.192** / **−2** | −0.0025 / −0.795 / **−2** | 0 / 0 / 0 | 0 / 0 / 0 |
| `no_hier_rush_l2l3` (whole rush mix layer off) | −0.0085 / +1.192 / −2 | −0.0025 / −0.795 / −2 | 0 / 0 / 0 | 0 / 0 / 0 |
| `named_rec_coverage_off` (all three REC floors → 0.0) | 0 / 0 / 0 | −0.0002 / −0.688 / 0 | −0.0001 / +0.015 / 0 | −0.0003 / −0.200 / **+1** |
| `named_rec_receptions_only_off` (0.98 → 0.0) | 0 / 0 / 0 | 0 / 0 / 0 | −0.0000 / +0.028 / 0 | 0 / −0.021 / 0 |
| `named_all_coverage_off` (REC + RUSH → 0.0) | 0 / 0 / 0 | −0.0002 / −0.688 / 0 | −0.0001 / +0.015 / 0 | −0.0003 / −0.200 / +1 |
| `named_all_coverage_full` (all floors → 1.0) | −0.0011 / +0.379 / 0 | −0.0006 / **+0.786** / −1 | +0.0003 / +0.043 / 0 | −0.0003 / +0.054 / 0 |
| `vacated_clip_floor_0.0` | −0.0210 / +0.709 / 0 | −0.0113 / −0.000 / 0 | −0.0156 / +0.270 / 0 | −0.0004 / +0.040 / 0 |
| `vacated_clip_floor_0.2` | −0.0005 / +0.222 / 0 | −0.0024 / +0.037 / 0 | −0.0019 / +0.108 / 0 | −0.0012 / +0.038 / +1 |
| `vacated_clip_floor_0.5` | −0.0009 / −0.303 / 0 | +0.0014 / −0.210 / −1 | +0.0027 / −0.046 / 0 | +0.0005 / −0.178 / 0 |
| `vacated_scale_off` (clip → (1.0, 1.0)) | −0.0023 / −0.500 / −1 | −0.0046 / −0.517 / 0 | −0.0003 / +0.341 / +1 | −0.0068 / −0.229 / 0 |

Notes on individual arms:

* **Hierarchical pass**: almost the entire effect is the L2 mix *columns*, not
  the L3 distribution. Turning off L3 alone moves RB MAE by 0.045; also
  dropping the `wr/te/rb_target_share` columns moves it by 1.259, because
  `normalize_team_passing_volume` and `reconcile_team_pass_receive_counts` both
  branch on those columns' presence and switch from per-(team, position)
  scaling to a single team-wide factor. **The mix layer's leverage is mostly in
  the reconcilers it re-keys, not in the redistribution it performs.** That is
  not how the audit (or the module docstring) describes it.
* **Hierarchical rush**: L2 and L3 are indistinguishable, which is expected —
  the rush mix columns are consumed by nothing but L3.
* **`VACATED_CLIP` floor**: 0.30 is not obviously the right value, but 0.0 is
  clearly worse (QB +0.709 MAE, and Spearman down at three of four positions —
  the largest Spearman movement anywhere in this study). 0.5 looks mildly
  better than 0.3 at every position on MAE while costing one RB tier hit. This
  is a knob worth a proper sweep, not a deletion candidate.

### Paired bootstrap (4,000 resamples of the 2025 population)

Only where the arm actually moved a number. `ΔMAE < 0` = ablation better.
`*` = 95% interval excludes zero.

| Arm | Pos | ΔMAE | 95% CI | P(ablation better) |
|---|---|---:|---|---:|
| `depth_rate_off` | TE | +0.567 | [+0.211, +0.925] | 0.000 `*` |
| `depth_rate_off` | RB | −0.653 | [−1.408, +0.101] | 0.957 |
| `depth_rate_off` | QB | +0.158 | [−0.758, +1.046] | 0.381 |
| `depth_rate_off` | WR | +0.023 | [−0.154, +0.191] | 0.393 |
| `named_rec_coverage_off` | RB | −0.688 | [−1.557, +0.171] | 0.941 |
| `named_rec_coverage_off` | TE | −0.200 | [−0.550, +0.140] | 0.877 |
| `no_hier_pass_l2l3` | RB | −1.265 | [−3.064, +0.497] | 0.918 |
| `no_hier_pass_l2l3` | WR | +0.099 | [−1.091, +1.186] | 0.417 |
| `no_hier_rush_l3` | QB | +1.192 | [−0.464, +3.168] | 0.085 |
| `no_hier_rush_l3` | RB | −0.795 | [−2.258, +0.646] | 0.874 |
| `vacated_clip_floor_0.0` | QB | +0.709 | [−0.043, +1.691] | 0.045 |
| `vacated_clip_floor_0.5` | QB | −0.303 | [−0.764, +0.018] | 0.955 |
| `vacated_clip_floor_0.5` | TE | −0.178 | [−0.464, +0.021] | 0.951 |
| triple joint (§3.1) | RB | −1.868 | [−5.041, +1.269] | 0.880 |
| triple joint | WR | −0.986 | [−2.486, +0.476] | 0.901 |

**Exactly one result in this entire study clears a 95% paired-bootstrap bar:
the Gate-B depth-rate ladder genuinely helps TE (+0.567 MAE if removed).**
Everything else — including every RB improvement that looks large in the delta
table — has a confidence interval straddling zero. Arms whose interval is
`[0.000, 0.000]` are the inert ones and should be read as "no effect", not as
significance.

### 3.1 Joint tests — is there a compensating pair?

Top three by individual effect: hierarchical rush (L3), hierarchical pass
(L2+L3), receiving coverage floors.

| Arm | QB ΔMAE / ΔHits | RB ΔMAE / ΔHits | WR ΔMAE / ΔHits | TE ΔMAE / ΔHits |
|---|---|---|---|---|
| rush alone | +1.192 / −2 | −0.795 / **−2** | 0 / 0 | 0 / 0 |
| pass alone | 0 / 0 | −1.259 / **+1** | +0.096 / 0 | −0.064 / 0 |
| rec-coverage alone | 0 / 0 | −0.688 / 0 | +0.015 / 0 | −0.200 / +1 |
| rush + pass | +1.192 / −2 | −1.722 / **+2** | +0.096 / 0 | −0.064 / 0 |
| rush + rec | +1.192 / −2 | −1.273 / −2 | +0.015 / 0 | −0.200 / +1 |
| pass + rec | 0 / 0 | −1.797 / +1 | −0.986 / +1 | −0.591 / +1 |
| all three | +1.192 / −2 | −1.868 / **+2** | −0.986 / +1 | −0.592 / +1 |
| `depth_rate_off` + rec | −0.158→+0.158 / −1 | −1.338 / 0 | +0.037 / 0 | +0.337 / 0 |
| `depth_rate_off` + rush | +1.353 / −3 | −1.366 / −1 | +0.023 / 0 | +0.567 / 0 |

**Yes, there is one, and it is in tier hits, not MAE.** Removing the rush mix
alone costs **2 RB tier hits**. Removing the pass mix alone gains 1. Removing
**both** gains **2** — a four-hit swing relative to the rush-only arm. The two
layers are jointly mis-allocating the RB room: the rush mix inflates RB carry
volume, the pass mix's re-keying of the receiving reconcilers inflates RB
receiving volume, and each partly conceals the other's error at the RB tier
boundary. Removing one alone is worse than removing both. This is exactly the
"two errors can cancel" shape already in project memory, and it means **the
single-knob table above understates the case against the mix layers.**

A second, milder interaction: `pass + rec` is the only pair that improves WR
(−0.986 MAE) and TE (−0.591) together, while neither does alone. Both act
through the same receiving reconcilers.

On MAE the joints are consistently **sub-additive** (all three: −1.868 RB vs.
−2.742 if additive), i.e. the layers are competing for the same error, not
compounding it. No sign inversion on MAE anywhere.

**Caveat that governs the whole subsection**: a 2-hit swing out of 24 is well
inside binomial noise on one season, and the RB MAE joint interval is
[−5.041, +1.269]. This is a strong enough signal to justify a second fold and
a re-decision; it is not strong enough to delete a layer on.

---

## 4. Stage-order fix (Task 2)

### Implemented

`src/projection/composition.py` — `compose_board` now calls
`reconcile_stat_constraints` a second time, after
`reconcile_team_pass_receive_counts`, so the last stage that changes a number
is guarded. The rationale is in the code: the count reconciler rescales
receptions and receiving TDs with a `(team, position)` factor their *parent*
stats do not share, so it can put a child back above its parent after the
first guard has run.

A second, unbudgeted change was required to make the first one correct.
`reconcile_stat_constraints` opened with `out["stat_constraint_applied"] =
False`. Called twice, the second call **erased the first call's audit trail**:
on the 2026 board that silently flipped 28 genuine caps from `True` to `False`
(20 RB / 4 TE receptions, 4 QB completions), and `stat_constraint_applied` is
shipped in `OUTPUT_COLUMNS` and aggregated into `fantasy_points`'
`any_stat_constraint_applied`. The flag is now **sticky** — it means "this row
was capped somewhere in composition", which is what the column name and its
consumers already assume. This changes no projected number.

### Measured effect — the briefed improvement does not reproduce

| Position | Baseline MAE → post-fix | Baseline hits → post-fix | Spearman |
|---|---|---|---|
| QB | 40.604510 → 40.604510 | 8 → 8 | unchanged |
| RB | 35.787058 → 35.787058 | 14 → 14 | unchanged |
| WR | 23.520593 → 23.520593 | 22 → 22 | unchanged |
| TE | 17.146722 → **17.147358** | 4 → 4 | unchanged |

The fix changes **one row** on the 2025 fold: player `00-0036754` (TE),
receptions 0.819231 → 0.805760 per game. Total model points across the whole
population move by 0.105. TE points MAE gets **0.0006 worse**, not better.

The briefed result (RB 34.95 → 34.07, 13 → 16 tier hits) is not reproducible
on this tree. RB is untouched to full float precision. I am reporting the fix
as **structurally correct and numerically inert**, not as an improvement.

Instrumentation backing this: at the output boundary the pre-fix 2025 board has
0 rows with `completions > attempts` and **1** row with
`receptions > targets`. The first `reconcile_stat_constraints` call flags 3
rows; the trailing call flags 1.

### 2026 board delta

Pre-change board snapshotted to
`<scratchpad>/projections_2026_PRE_STAGEORDER.csv` before any edit. To isolate
this change from anything the two concurrent agents may have done, I also
generated a counterfactual board from the *current* tree with only the trailing
call removed (`<scratchpad>/board_pre.py`).

```
rows: 4039   columns: 83   cells: 335,237
CHANGED CELLS: 0   in 0 of 4039 rows
```

**Zero cells changed** — both against the counterfactual board and against the
session-start snapshot. Distribution of the change: empty. Identity check on
both boards: 0 rows with `completions > attempts`, 0 with
`receptions > targets`.

So the 2026 board did **not** need this guard: the constraint already held at
its output boundary. The fix earns its place as a guard against a violation
that provably occurs on other data (1 row on the 2025 fold), not as a
correction to what currently ships. The task expected the board to change; it
does not, and I would rather say so than present the 28 flag flips — which were
a *regression* my sticky-flag change removes — as the expected movement.

### Tests

- `tests/test_composition_unification.py::EXPECTED_STAGE_ORDER` updated to the
  16-stage sequence (the pinned order is the point of that list).
- Added `test_last_numeric_stage_is_the_stat_constraint_guard` — asserts the
  guard is the final numeric stage, that it also runs before the count
  reconciler, and that it appears exactly twice. Fails with a message about the
  guard rather than as one line of a long list diff.
- Added `test_trailing_guard_catches_a_violation_the_count_stage_introduces` —
  behavioural: a child stat inflated past its parent by the final numeric stage
  is still capped in the returned board.
- Added `test_stat_constraint_flag_survives_the_second_call` — pins the sticky
  flag, and that a second pass moves no `pred_pg`.

`.venv/Scripts/python.exe -m pytest -q` → **136 passed**. (Briefed baseline was
127; the delta is my 3 new tests plus 6 added by the concurrent agents.) No
existing test was weakened, skipped or deleted.

---

## 5. Gate-B ladder: analysis only, no behaviour changed

### The three harnesses disagree three ways

| Path | Applies `depth_rate_factor`? |
|---|---|
| `predict.project_season` for **2026** | **Yes** — `veterans.project_veterans` → `depth_gating.apply_depth_chart_gating`, curated-chart branch, keyed on `nfl_depth_rank` (line 295). |
| `predict.project_season` for **any other season** | **No** — `apply_depth_chart_gating` early-returns `role_discount_factor = 1.0` at line 257 whenever the curated chart is empty, and the curated chart is 2026-only by construction (`load_depth_chart` line 34: `if target_season != 2026: return <empty>`). |
| `fantasy_evaluation` (2025 fold) | **Yes, unconditionally** — `_veteran_forecasts` calls `depth_rate_factor(position, rank)` directly (lines 348–352) with no curated-chart gate at all. |
| `backtest.py` | **No, never.** It imports neither `depth_gating` nor `depth_rates`; `depth_rate_calibration` only *fits/diagnoses* the ladder from `nfl_depth_rank`. Its MAE tables, `interval_residuals.csv` and (via `corrections.py`) the elite-shrinkage fit are all computed on **undiscounted** predictions. |

**So yes — `backtest.py` and `fantasy_evaluation.py` currently disagree about
this, and both disagree with the `predict` path for non-2026 seasons.** Three
harnesses, three answers.

### Is the early return intentional or a latent bug?

**Latent bug, with a benign cover.** The evidence:

1. The factor is computed from `nfl_depth_rank`, not from the curated chart.
   The function's own Gate-B comment (line 291–294) says this explicitly and
   gives the reason: the nflverse rank "is the only rank that exists for
   historical seasons, so it is the only one this can be fit and validated on".
   Gating a chart-independent signal on chart presence contradicts the comment
   directly above it.
2. `nfl_depth_rank` is populated for **3,326 of 3,476** rows on the 2025 fold.
   Applied unconditionally, **287 of 661** veteran players (43%) receive a
   factor below 1.0 — QB {0.77: 28, 0.84: 11}, RB {0.70: 32, 0.73: 17,
   0.86: 7, 0.98: 28}, WR {0.79: 4, 0.86: 21, 0.94: 60, 0.97: 26},
   TE {0.77: 3, 0.83: 23, 0.90: 27}. This is not a marginal signal.
3. The empty-chart branch also sets `depth_chart_status = "not_curated_no_table"`,
   whose docstring says "gating is a no-op, **not a claim about this player's
   role**". That is a correct statement about the *curated* role — and it was
   then used to suppress a factor that does not come from the curated role.
   The most likely history is that the branch predates the Gate-B change that
   re-keyed the factor from `role` to `nfl_depth_rank`, and was not revisited.

It is currently **dormant on the shipped path** — 2026 has a chart, so the
ladder applies. It becomes live the moment `starters_2027.csv` does not exist
yet, or anyone runs `project_season` for a historical season.

### What changing it would do — measured

`fantasy_evaluation` already applies the ladder unconditionally, so the
*baseline* is the unconditional case and the `depth_rate_off` arm is the
counterfactual (what `depth_gating`'s early return produces).

| Position | Ladder ON (baseline) | Ladder OFF | Effect of HAVING the ladder |
|---|---|---|---|
| QB | 40.6045 / 8 hits / 0.79173 | 40.7625 / 7 hits / 0.78833 | −0.158 MAE, +1 hit — helps |
| RB | 35.7871 / 14 / 0.68909 | 35.1341 / 14 / 0.69189 | **+0.653 MAE — hurts** |
| WR | 23.5206 / 22 / 0.78614 | 23.5436 / 22 / 0.78594 | −0.023 MAE — neutral |
| TE | 17.1467 / 4 / 0.83181 | 17.7137 / 4 / 0.82631 | **−0.567 MAE — helps, CI [+0.211, +0.925], the one significant result in this study** |

Decomposition:

* `depth_rate_off_chart_only` (`DEPTH_RATE_OFF_CHART` → 1.0): essentially nil
  (max |ΔMAE| 0.040, all four positions). The off-chart factor is not where
  the value is.
* `depth_rate_ladder_only` (on-chart rungs + `DEPTH_RATE_DEEP` → 1.0):
  reproduces ~95% of the full ablation. **The on-chart rungs and the DEEP
  factor carry the entire Gate-B effect.**

### Recommendation

1. **Do not change `depth_gating.py:257` in this pass.** Confirmed correct — it
   would change `backtest.py` semantics, which is the fit path for
   `interval_residuals.csv`, the corrections β, and the ladder's own
   calibration table. Applying the ladder in `predict` for historical seasons
   while `backtest` continues to fit residuals on undiscounted predictions
   would create a train/serve mismatch of exactly the kind
   `OL_TRAILING_SEASONS` already introduces, and the elite correction's
   `role_discount_factor` scaling (`team_reconcile.py:264`) assumes a
   consistent convention.

2. **Fix the three-way disagreement in one change, deliberately, as its own
   piece of work**, with this order of operations:
   a. Make `apply_depth_chart_gating` apply the ladder from `nfl_depth_rank`
      unconditionally, deleting the `depth_chart.empty` early return's
      `role_discount_factor = 1.0` while keeping everything else it sets
      (`depth_chart_status`, `role`, `low_confidence` are genuinely
      chart-dependent and must stay gated).
   b. In the same change, apply the same factor in `backtest.py`'s prediction
      path, and **refit `interval_residuals.csv` and `corrections.joblib`**
      afterwards. The corrections β has already been refit three times as
      upstream composition moved and sits at consistency 2.1 against a gate of
      2.0 — one fold from failing its own test — so this refit must be watched,
      not assumed benign.
   c. Re-run the 2025 fold. It should be *unchanged*, because
      `fantasy_evaluation` already does (a). That invariance is the acceptance
      test for the change: it proves the three paths were unified onto the
      behaviour that was already being measured, rather than onto a fourth.
   d. Expected 2026 board impact: **zero**, since 2026 has a curated chart and
      already takes the ladder branch.

3. **Independently, re-open the ladder's per-position values.** The measured
   picture is not uniform: TE is the only position where the ladder is
   defensible on this evidence (significant, +0.567 MAE), QB is a weak positive,
   WR is nil, and **RB is a measured negative (+0.653 MAE) — inside the
   bootstrap interval, but pointing the same way as the joint tests in §3.1,
   which also indict RB allocation.** The audit already flags that the shipped
   ladder is the fit "capped at 1.0 and shaded toward more discount by hand",
   and that its acceptance evidence was Sleeper MAE. A per-position re-fit
   scored on fantasy points — with RB's `{0.70, 0.73, 0.86, 0.98}` rungs as
   the first thing tested — is a better use of a run than any deletion on this
   list.

---

## 6. Ranked verdict

Confidence is in the *conclusion*, not in the size of the effect.

### Recommend deleting

Nothing, on this evidence. No candidate produced a removal that is better than
what ships at 95% on the one available fold. That is itself the headline
result: the ablation did not find a layer that is measurably harmful, and it
did not find a layer that is measurably helpful either, with one exception.

### Recommend keeping

| Item | Why | Confidence |
|---|---|---|
| **Gate-B ladder, TE rungs** | The only result in this study that clears a paired-bootstrap 95% bar. Removing it costs 0.567 TE points MAE, CI [+0.211, +0.925]. | **High** |
| **`VACATED_CLIP` lower bound (as a non-zero number)** | Setting the floor to 0.0 is the largest Spearman degradation measured anywhere here (QB −0.021, WR −0.016) and costs QB 0.709 MAE. The floor exists for a reason even if 0.30 is not the reason. | **High** for "not zero", **Low** for "0.30 specifically" — 0.5 beat 0.3 on MAE at all four positions and deserves a proper sweep. |
| **`NAMED_RUSH_COVERAGE = 0.814` as a value** | Cannot bind on this fold and so cannot be wrong on it. The audit's judgment that the *estimate* is sound stands; the open question remains whether the mechanism should exist, and that is still unanswerable here. | **High** (that it is currently inert) |
| **`reconcile_stat_constraints` trailing call** | Structurally correct; costs nothing; catches a real violation on one fold. | **High** |

### Genuinely neutral

| Item | Evidence | Confidence |
|---|---|---|
| `NAMED_REC_RECEPTIONS_COVERAGE = 0.98` | Removing it moves nothing above 0.031 MAE at any position. The audit's rank-9 concern (asserted by analogy, never measured) is real as *provenance*, but the constant is doing almost nothing. | **High** |
| `DEPTH_RATE_OFF_CHART` | Max |ΔMAE| 0.040 across all positions. | **High** |
| `apply_hierarchical_pass_distribution` (L3 alone) | Max |ΔMAE| 0.088. The audit's rank-2 — "the single largest multiplier in the shipped board" — is correct about the *multiplier* and wrong about the *leverage*: the L2 mix columns move ~25× more than the L3 redistribution does, via the reconcilers they re-key. | **High** for the fold; the 0.00-valued `hierarchical_pass_scale` rows the audit flags are still worth a separate look. |
| `RUSH_ATTEMPTS_PER_APPEARANCE_MAX`, `RUSH_YARDS_PER_CARRY_MAX` | 0 of 247 rows bind, on this fold and on the 2026 board. Undocumented (audit rank 7) but currently inert. | **High** for "inert now", **Low** for "safe to remove" — they are a tail guard and the fold has no tail event. |
| `PASS_CATCH_COHERENCE_BAND` | Excluded from the study as the audit recommends; 0 of 4,039 rows flagged. | **High** |

### Needs more evidence

| Item | What is missing | Confidence in the gap |
|---|---|---|
| **L2 pass mix + L2/L3 rush mix, jointly** | The strongest single finding here (§3.1: rush-only −2 RB tier hits, pass-only +1, both +2) is a compensating pair on a metric with ~24 trials on one season. Needs a second fold before anyone acts. Both layers are also unmeasurable at their L2-gate level: `build_team_*_mix_profiles` never consults `validate_*_mix_model`, and the pass-mix LOSO number is still unrecorded anywhere in the repo. | **Medium-High** that something is wrong with RB allocation; **Low** on which of the two layers is at fault |
| **`FORMATION_ROLE_BLEND_W`, `WR_FORMATION_ROLE_PRIORS`** | Structurally unmeasurable — no `formation_role` for 2025. The audit's rank-1 argument (byte-identical to a prior measured to lose, re-entering through an uncommitted constant) stands entirely on provenance and is **not weakened or strengthened by this study**. To score it you need a curated `formation_role` for a historical season, i.e. a research file, not a code change. | **High** that it is unmeasurable |
| **Both vacancy alphas, `TEAM_CHANGE_VACANCY_ALPHA`** | Unreachable: they live in `roster_moves`, upstream of `compose_board`, which the unification did not extend to. Audit §4.5 item 1 is half done. Extending the harness to run the roster-move layer is the single highest-value next step, and it unblocks the audit's rank 3 and F1/F3. | **High** |
| **`USAGE_SHARE_CURATED_W`** | Inert on this fold (needs a curated `usage_share_prior` column). Note this contradicts the audit's rank-6 note that it "is actually scoreable today" — the *mechanism* runs, but with no curated input it exits early. | **High** |
| **`DEEP_BENCH_GAMES_CAP`, replacement rows, elite corrections, prediction intervals** | Confirmed structurally unmeasurable, as briefed. Not attempted. | **High** |
| **RB allocation overall** | Three independent arms point the same way — the rush mix, the pass mix's reconciler re-keying, and the Gate-B RB rungs all make RB slightly worse. None is individually significant. Together they are the most consistent signal in the study and the best-motivated target for the next fold. | **Medium** |

---

## Appendix: what was run

25 distinct arms plus 2 baseline replications, ~88s each. Artefacts in the
session scratchpad: `ablate.py` (runner), `compare.py` (delta tables),
`boot.py` (paired bootstrap), `probe.py` (reachability/binding instrumentation),
`gateb.py` (Gate-B factor distribution), `board_pre.py` + `diff_board.py`
(2026 board isolation), `results/*.json`, `results/*_rows.csv`,
`projections_2026_PRE_STAGEORDER.csv`.

Files changed in the working tree by this task, all under `src/projection/` and
`tests/`, nothing committed or staged:

- `src/projection/composition.py` — trailing `reconcile_stat_constraints`;
  docstring updated.
- `src/projection/team_reconcile.py` — `stat_constraint_applied` made sticky.
- `tests/test_composition_unification.py` — stage order updated to 16 stages;
  three tests added.

No constant in `contracts.py` was modified. No deletion from the ablation list
was applied.
