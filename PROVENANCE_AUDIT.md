# Provenance audit of the post-model allocation layer — 2026-08-15

Read-only forensic audit. No source file was modified, no pipeline was run, no
commit was made. Every claim below is sourced to a commit SHA, a doc section, a
code comment, or a static read of `output/projections_2026.csv`.

## Classification scheme

| Class | Meaning |
|---|---|
| **A** | Fitted/validated against real outcomes — held-out actual NFL production or fantasy points |
| **B** | Fitted/validated against a **proxy** metric — share MAE, tendency MAE, OL-score persistence, coverage fraction. Real data, wrong objective. |
| **C** | Tuned toward Sleeper/consensus agreement |
| **D** | Hand-set by judgment, no measurement |
| **E** | Not a tuning knob (path, column list, enum) — excluded from the counts |

A doc calling something "measured" earns class A only if a held-out **outcome**
metric is identifiable. Where the fit and the shipped value differ, the shipped
value governs the classification and the discrepancy is stated.

---

## 0. Two structural findings that frame everything else

### 0.1 The entire allocation layer is uncommitted

`git ls-files src/projection/` returns 13 files. The following are **untracked**
(`??` in `git status`) and therefore exist only in the working tree:

```
src/projection/contracts.py      src/projection/team_reconcile.py
src/projection/roster_moves.py   src/projection/replacement.py
src/projection/depth_gating.py   src/projection/depth_rates.py
src/projection/veterans.py       src/projection/team_pass_mix.py
src/projection/team_rush_mix.py  src/projection/artifacts.py
src/coordinator/inheritance.py
```

Consequences for this audit:

- `contracts.py` has no git history of its own. Provenance for each constant had
  to be traced through `git log -S` against `HEAD:src/projection/predict.py`,
  where these values previously lived.
- Four constants (`OL_TRAILING_SEASONS`, `WR_FORMATION_ROLES`,
  `WR_FORMATION_ROLE_PRIORS`, `FORMATION_ROLE_BLEND_W`,
  `DEPTH_RANK_TO_WR_FORMATION_ROLE`) return **no** `git log -S` hit at all. They
  have never been committed in any form. Their only documentation is
  `OL_TRAILING_2026-08-14.md` and `HIERARCHICAL_PASS_MIX_2026-08-14.md`, also
  untracked.
- `FREEZE_2026-08-13.md`'s artifact hashes describe commit `df37452`, which
  predates the whole layer. Nothing in the current tree is under the freeze.

### 0.2 The only class-A harness does not exercise the highest-leverage layers

`src/projection/fantasy_evaluation.py` — the leakage-safe 2024→2025 evaluation,
the project's only criterion that matches how the output is used — imports and
runs:

`depth_rate_factor`, `apply_usage_share_prior`, `normalize_team_passing_volume`,
`normalize_team_rushing_volume`, `reconcile_team_pass_receive_counts`,
`reconcile_stat_constraints`, `reconcile_qb_projected_volume_games`.

It does **not** import `team_pass_mix`, `team_rush_mix`, `roster_moves`,
`replacement`, `depth_gating`, or `corrections`. Neither does `backtest.py`
(which does run `corrections`).

So the following can not currently be scored against outcomes by any harness in
the repo, no matter what one wanted to measure:

- hierarchical pass mix L2/L3 — `hierarchical_pass_scale` mean **1.467**, range
  **0.00–4.05** across 2,728 shipped rows
- hierarchical rush mix L2/L3 — `hierarchical_rush_scale` mean **1.354**, range
  0.72–2.58 across 831 rows
- every vacancy alpha and the team-change reassignment path
- replacement-level rows, `DEEP_BENCH_GAMES_CAP`, formation-role blending

This is the root cause of the pattern the rest of this audit documents. When the
only outcome-based scoreboard cannot see a layer, the layer's authors reached for
the scoreboard that could: Sleeper.

---

## 1. All 43 constants in `src/projection/contracts.py`

Line numbers are current working-tree `src/projection/contracts.py`.

| # | Constant | Value | Line | Class | Evidence |
|---|---|---|---|---|---|
| 1 | `REPO_ROOT` | derived path | 10 | **E** | path |
| 2 | `MODELS_DIR` | `models/` | 11 | **E** | path |
| 3 | `OUTPUT_DIR` | `output/` | 12 | **E** | path |
| 4 | `INTERVAL_RESIDUALS_PATH` | path | 13 | **E** | path |
| 5 | `CORRECTIONS_PATH` | path | 14 | **E** | path |
| 6 | `DEPTH_CHART_PATH` | path | 15 | **E** | path |
| 7 | `LIVE_DEPTH_CHART_PATH` | path | 16 | **E** | path |
| 8 | `STATUS_OVERRIDES_PATH` | path | 17 | **E** | path; added `0329ee6` |
| 9 | `CURATED_RESEARCH_DEPTH` | `{QB:2,RB:2,WR:3,TE:2}` | 22 | **D** | `0329ee6` comment cites `PHASE6_REPORT.md`. That report describes the *scope of manual research* ("I researched every one of the 32 teams' QB/RB/WR/TE groups to the [relevant] depth"), not a measurement. This is the boundary of human effort, promoted to a hard review-failure gate in `enforce_availability_chart_review`. |
| 10 | `DEEP_BENCH_GAMES_CAP` | `6.0` | 23 | **B** | `0329ee6` comment: "Mean REG games among off-chart RB/WR/TE with any weekly usage, 2017-2025 ≈ 5.51." A real descriptive statistic of outcomes, but (a) it is games, not fantasy accuracy; (b) 5.51 was rounded **up** to 6.0 with no stated reason; (c) it is a hard clamp layered on top of the Gate A availability model, which *was* validated. Never scored held-out. |
| 11 | `ROOKIE_RATIO_FALLBACK` | `(0.2, 3.0)` | 24 | **D** | `dbbe949`; own comment says "deliberately wide". Affects `pred_pg_low/high` only, never the point estimate. |
| 12 | `VACATED_CLIP` | `(0.3, 2.5)` | 27 | **D** | `55953fa` (`rookies.py`), module docstring: "clipped to avoid small-sample blowups". No measurement, then or since. **Binds hard**: 317 of 1,134 rookie rows ship at exactly the 0.30 lower bound. |
| 13 | `TEAM_CHANGE_SHARE_CLIP` | `= VACATED_CLIP` | 28 | **D** | `05bdc91` comment: "deliberately reused from `rookies.VACATED_CLIP` … rather than inventing a new number". Reuse of an unmeasured number is not evidence. |
| 14 | `DEPTH_RATE_LADDER` | see file | 31–36 | **A** | Gate B, `720fa8e`. Estimand is `sum(actual_pg)/sum(pred_pg)` among players at a given preseason rank who played — real held-out production, leave-one-transition-out 2021–2025 (QB/RB on 2017–2025). **Shipped ≠ fitted**: commit states "two deliberate departures from the raw fit: capped at 1.0 … and shaded toward more discount where per-fold spread is wide or n<30". WR rank 4 fit at 1.11 and ships at 1.00. **Acceptance evidence in the commit body is season-scale MAE vs Sleeper by role tier** (committee 54.1→28.3 etc.), i.e. class-C reporting on a class-A fit. |
| 15 | `DEPTH_RATE_DEEP` | `{QB:.84,RB:.70,WR:.94,TE:1.00}` | 37 | **A** | Same fit as #14, same shipped-vs-fitted caveat. `role_discount_factor` min in the shipped board is 0.70, so this binds. |
| 16 | `DEPTH_RATE_OFF_CHART` | `{QB:1.00,RB:.86,WR:.79,TE:.77}` | 38 | **A** | Same fit. QB=1.00 is the 1.0 clamp of a raw fit of ~1.08 (`720fa8e` states this explicitly). |
| 17 | `BOOST_ELIGIBLE_ROLES` | `{starter, committee}` | 40 | **D** | `9d5e533`/`737eadf`. Which curated roles receive the incumbent vacancy boost is asserted, never measured. It is a tuning knob, not an enum: it decides who gets scaled. |
| 18 | `INCUMBENT_VACANCY_ALPHA` | `{target:0.5, carry:1.0}` | 45 | **C** | **target=0.5 is B**: `9d5e533` grid-searched it over 2017→2025 transitions against observed share change — share MAE −1.85% vs carry-forward, consistency 2.06. Share MAE is a proxy. **carry=1.0 is C**: the same commit measured carry α=1.0 as "STRONGEST evidence in the module" (share MAE −13.6%, 9/9 folds) and **shipped it disabled at 0.0 anyway**, because "Live it made 2 of 3 RB metrics worse (corr 0.848→0.833, MAD 2.367→2.444)" — those are Sleeper metrics. It was re-enabled to 1.0 on 2026-08-14; the justifying document `RB_CARRY_VACANCY_2026-08-14.md` is a five-row table in which four rows are Sleeper comparisons, produced by `scripts/diag_rb_carry_vacancy.py`, which reads only `output/sleeper_comparison_2026.csv`. Both the disable and the re-enable were adjudicated on Sleeper. |
| 19 | `TEAM_CHANGE_VACANCY_ALPHA` | `{target:0.35, carry:0.25}` | 46 | **B** | `4f3e452` measured both over every 2017→2025 transition on target/carry **share** MAE vs carry-forward: 0.05460 at α=0.35, 0.19082 at α=0.25. Proxy metric. C-contamination: the commit defends carry=0.25 partly on live Sleeper numbers ("0.25 is the best available setting on both historical and live evidence"; "RB MAD regresses 2.386 → 2.490"). |
| 20 | `INCUMBENT_VACANCY_NET_CLIP` | `0.75` | 47 | **D** | Introduced in `9d5e533`. That commit enumerates exactly what was measured (alpha) and this is not in it. No comment, no doc, no derivation found anywhere in the repo. |
| 21 | `INCUMBENT_VACANCY_SCALE_CAP` | `2.0` | 48 | **D** | Same as #20. Undocumented. |
| 22 | `REPLACEMENT_POSITIONS` | `("RB","WR","TE")` | 50 | **D** | `fca3525`. The QB exclusion has a strong *mechanistic* argument (QB rooms have a fixed 17-game budget, so a QB row is a claim against the starter) and an observed live effect (Watson 5.9 games, Sanders 13.6→8.0), but there is no ground truth in that observation — it is reasoning plus a look at the board. Defensible judgment, still judgment. |
| 23 | `REPLACEMENT_MIN_CELL` | `15` | 51 | **D** | `fca3525`. No derivation. Matches `corrections.MIN_N_ABOVE_KNOT = 15` by convention, and that one is also unfitted. |
| 24 | `REPLACEMENT_DEPTH_BANDS` | `((1,r1),(2,r2),(99,r3+))` | 52 | **D** | `fca3525`. Banding scheme mirrors the Gate B keying by analogy; the cut points were not tested against alternatives. |
| 25 | `PASS_CATCH_COHERENCE_BAND` | `(0.8, 1.35)` | 54 | **D** | `8b28673`, and the code comment is unusually honest: "a stated, un-tuned judgment call, not fit to any target". **Zero shipped impact** — diagnostic flag only; `team_pass_catch_coherence_flag` is `False` on all 4,039 rows. |
| 26 | `NAMED_REC_YARDS_COVERAGE` | `0.98` | 55 | **B** | `0329ee6` comment: measured on nflverse weekly 2016–2024 — top-12 target boards cover ~97.8% of receiving yards, all WR/TE/RB ~99.3%. A real empirical coverage fraction, but a coverage fraction is not fantasy accuracy, and 0.98 is a point pick inside a 97.8–99.3 range with no stated rule. |
| 27 | `NAMED_REC_RECEPTIONS_COVERAGE` | `0.98` | 56 | **D** | Same commit comment, but for receptions it says only "Receptions track yards, not the TD leak" — **asserted by analogy, never separately measured**. |
| 28 | `NAMED_REC_TDS_COVERAGE` | `0.96` | 57 | **B** | Measured ~97.1% (top-12) / ~98.7% (all). Shipped at 0.96, i.e. *below both measurements* — an unstated conservative shade. |
| 29 | `QB_ATTEMPTS_PER_VOLUME_GAME_MAX` | `42.0` | 59 | **B** | `c57ecd0` comment: "Conservative ceiling just below the observed all-time full-season pace (~42.4 attempts/game)." Historical support bound, shaded down. **It binds**: `c57ecd0` records CLE's three QBs sitting on the clamp before the two-sided fix. |
| 30 | `RUSH_ATTEMPTS_PER_APPEARANCE_MAX` | `{QB:12,RB:25,WR:5,TE:3}` | 60 | **D** | Introduced `c57ecd0` with **no comment at all**, and no mention in `PHASE7_REMEDIATION_REPORT.md`. Undocumented. RB=25.0 is the exact ceiling Josh Jacobs was pinned to (`10303b5`), i.e. this number silently set a headline projection for at least two players. |
| 31 | `RUSH_YARDS_PER_CARRY_MAX` | `{QB:10,RB:7,WR:15,TE:15}` | 61 | **D** | Same commit, same absence of any comment or doc. WR/TE at 15.0 yards per carry is implausibly loose and appears to be a placeholder. |
| 32 | `NAMED_RUSH_COVERAGE` | `0.814` | 62 | **B** | `10303b5`: share of a team's season-N carries taken by players active in season N-1, every 2017–2025 transition, mean 0.814, range 0.776–0.869, no trend. This is the best-estimated constant in the file. **But the decision to have the mechanism at all was Sleeper-gated**: the alternative (no fill) was rejected because "every lead back fell well under consensus — RB season-total MAE 15.6 → 19.0", which is MAE against Sleeper. `DEPTH_CHART_ALLOCATION_2026-08-14.md` §"Investigated and found not to be a bug" states outright: "The decisive test was Sleeper itself." |
| 33 | `OL_TRAILING_SEASONS` | `3` | 68 | **B** | `OL_TRAILING_2026-08-14.md`. Ablation metric is **next-season OL-score persistence MAE** (0.00506→0.00482 pass-pro, 0.05897→0.05368 run-block) — a proxy two steps removed from fantasy points. Additional risk: the trailing average is applied to the **live predict path only**, historical/backtest rows stay exact-season, so live features are drawn from a different distribution than training. Never committed to git. |
| 34 | `USAGE_SHARE_BLEND_W` | `0.0` | 70 | **A** | `482f626` and `DEPTH_CHART_ALLOCATION_2026-08-14.md` §4. **The exemplar of correct practice in this repo.** The fitted rank prior won on LOSO share MAE (−9.3% targets, −8.8% carries, 9/9 folds) and was then re-scored on the leakage-safe 2025 fantasy evaluation, where at w=0.25 it *lost*: RB points MAE +1.25%, WR +0.73%, mean VORP MAE +1.5%, one fewer tier hit. Shipped at 0.0. |
| 35 | `USAGE_SHARE_CURATED_W` | `0.5` | 71 | **D** | `482f626`: justified purely by analogy — "the same precedence `reassign_team_changers` already gives a curated 'starter' over the vacancy heuristic". No measurement. **Currently live**: `usage_share_blend_factor` is non-null on 50 shipped rows, range 0.764–1.263, driven by 11 hand-reviewed chart rows (DAL, GB, JAX, PHI) whose priors are round hand-typed numbers (0.24, 0.18, 0.06, 0.52, 0.22…). |
| 36 | `WR_FORMATION_ROLES` | `("LWR","RWR","SWR")` | 77 | **E** | enum |
| 37 | `WR_FORMATION_ROLE_PRIORS` | `{LWR:.1554, RWR:.0667, SWR:.0386}` | 78 | **B** | Never committed; documented only in `HIERARCHICAL_PASS_MIX_2026-08-14.md`. **These are the fitted rank-prior defaults** — the identical triple `SLOT = [0.1554, 0.0667, 0.0386]` in `HEAD:scripts/apply_wr_usage_priors.py`, whose fit was LOSO share MAE and which was **measured to lose on the fantasy evaluation** and consequently disabled at `USAGE_SHARE_BLEND_W = 0.0` (#34). It re-enters the live path here at weight 0.5, and again as `WR_USAGE_SLOTS` in `src/depth_chart/live.py:18`. Same numbers, same proxy fit, negative fantasy-evaluation result, different door. |
| 38 | `FORMATION_ROLE_BLEND_W` | `0.5` | 79 | **D** | Never committed. `HIERARCHICAL_PASS_MIX_2026-08-14.md` names it and gives no number, no gate, no ablation. Active on the shipped board: 352 rows carry a `formation_role` (120 LWR / 120 RWR / 112 SWR). |
| 39 | `DEPTH_RANK_TO_WR_FORMATION_ROLE` | `{1:LWR,2:RWR,3:SWR}` | 80 | **E** | **Dead code** — zero consumers anywhere in `src/`, `tests/`, or `scripts/`. Also encodes an assertion the curated chart's own notes column contradicts ("listed order is formation-based … and is NOT usage order"). |
| 40 | `USAGE_SHARE_FAMILIES` | stat→prior mapping | 82–93 | **E** | structural mapping |
| 41 | `USAGE_SHARE_MAX_RANK` | `5` | 94 | **D** | `482f626`. No derivation given. |
| 42 | `TEAM_ANCHOR_OUTPUT_COLS` | column list | 96 | **E** | schema |
| 43 | `OUTPUT_COLUMNS` | column list | 103–146 | **E** | schema |

---

## 2. Fitted layers

| Layer | Where | Class | Exact metric it was gated on | Notes |
|---|---|---|---|---|
| **Gate A availability model** | `train.fit_availability`, `depth_history.py` | **A** | Held-out games-played MAE, leave-one-transition-out 2017–2025, 8/8 folds at every position (QB 3.514→2.553 etc.); plus Phase 11's held-out season-**total** MAE against actuals including zeros (WR 252.3→154.4). | Genuine outcome validation. `8be9c63` also reports "Season-scale MAE vs Sleeper 24.8 → 22.9" as supporting evidence — unnecessary and contaminating, but not decisive. |
| **Gate B depth-rate ladder** | `depth_rates.py`, `contracts.py:31-38` | **A** (fit) / **D** (shipped) | `sum(actual_pg)/sum(pred_pg)` by preseason rank among players who played, LOTO 2021–2025. | Shipped values are the fit clamped at 1.0 and "shaded toward more discount" by hand. Headline acceptance table in `720fa8e` is Sleeper season-scale MAE. |
| **Elite shrinkage correction** | `corrections.py` | **A** | Held-out 2024→2025 receiving MAE (TE 7.77→7.53; elite TE subset 16.80→12.10, n=5), plus an explicit `MIN_SEASON_CONSISTENCY = 2.0` per-transition stability gate that **rejected WR** after the pooled fit made the holdout worse (15.31→15.40). | Best-provenance layer in the stack, and the only one with a codified anti-pooling guard. Caveat: β has been refit three times (0.4903 → 0.4747 → 0.3770 → 0.4031) as upstream composition changed, and `df37452` records its consistency at 2.1 against a gate of 2.0 — one fold from failing its own test. `MIN_SEASON_CONSISTENCY=2.0`, `ELITE_KNOTS`, `ELITE_CORRECTION_CAP=8.0`, `MIN_N_ABOVE_KNOT=15` are themselves class **D** (the cap's own comment: "the cap is protection against the tail, not a fitted value"). |
| **`RECEIVING_SHARE_SUM_CAP = 1.2`** | `transitions.py:117` | **A** | Held-out 2024→2025 backtest MAE, monotone improvement down to ~1.1 (WR receiving 10.72→10.46, TE 8.04→7.96, RB 5.70→5.60). | `6b48273`, user decision at gate. 1.2 chosen over the 1.1 optimum as margin — a D-flavoured shade on an A-grade fit. The commit's "honest live trade-off" paragraph is entirely Sleeper-framed. |
| **Age-effect shrink (`AGE_EFFECT_SHRINK`)** | `transitions.py:199` | **A** | Grid search on the 2024→2025 holdout **and** a 3-fold rolling-origin backtest, per (position, stat); then the leakage-safe 2025 fantasy evaluation. RB monotone improvement, WR monotone harm → RB only. | `AGE_EFFECT_SHRINKAGE_2026-08-14.md`. Discloses its own regression (RB VORP MAE 35.20→36.59, traced to David Montgomery at the replacement boundary). Sleeper explicitly recorded as "no signal either way". |
| **Usage-share fitted rank prior** | `predict.fit_usage_share_priors`, `USAGE_SHARE_BLEND_W` | **A** | Leakage-safe 2025 fantasy evaluation: points MAE, VORP MAE, tier hits. Rejected. | The right process. Undercut in practice — see `WR_FORMATION_ROLE_PRIORS` below. |
| **Reviewed curated usage priors** | `starters_2026.csv`, `USAGE_SHARE_CURATED_W = 0.5` | **D** | None. | 11 chart rows reviewed by hand (DAL/GB/JAX/PHI), moving 50 shipped stat rows by 0.76×–1.26×. The weight 0.5 is asserted by analogy. |
| **WR formation-role priors + blend** | `team_pass_mix._allocate_wr_by_formation_role` | **B** (priors) / **D** (blend weight) | Priors: LOSO share MAE, from the fit that *lost* the fantasy evaluation. Blend weight 0.5: nothing. | The disabled prior, reinstated at half weight through a different constant. Highest-priority finding in this audit after §0.2. |
| **Hierarchical pass mix L2** | `team_pass_mix.py` | **B** | `validate_mix_model`: leave-one-season-out MAE on team WR/TE/RB **target share** vs league-mean and prior-season baselines. | Two problems. (i) The gate is **advisory only** — `build_team_pass_mix_profiles` never calls `validate_mix_model`; `predict.project_season` attaches the scheme mix unconditionally. `mix_source == 'scheme_model'` on **100%** of 4,039 shipped rows. (ii) **No pass-mix LOSO number is recorded anywhere in the repo.** `HIERARCHICAL_PASS_MIX_2026-08-14.md` gives a validation *command* and a conditional ("Ship the scheme mix only when LOSO MAE beats prior-season mix") but no result. The rush-mix doc has a table; the pass-mix doc does not. |
| **Hierarchical pass distribution L3** | `apply_hierarchical_pass_distribution` | **D** | None. | No gate of any kind. Leverage is the largest in the stack: `hierarchical_pass_scale` mean **1.467**, min 0.00, max **4.05**, on 2,728 rows. |
| **Hierarchical rush mix L2** | `team_rush_mix.py` | **B** | LOSO MAE on team **rush share**: scheme+lag 0.0343 vs prior-season 0.0347 vs league-mean 0.0415. | The task brief is right to flag this. A 1.2% edge on a share proxy, on ~9 seasons, called a "measured win". Same advisory-gate problem: `build_team_rush_mix_profiles` never consults `validate_rush_mix_model`; `rush_mix_source == 'scheme_model'` on 100% of rows. `hierarchical_rush_scale` mean 1.354, max 2.58. |
| **OC inheritance weights** | `coordinator/inheritance.py` | **B** | LOSO grid over team weight ∈ {0.3…0.7}, scored on SD-scaled **tendency metric** MAE plus pass-mix column MAE vs observed season values. Best 0.6/0.6 at 0.419 vs judgment 0.7/0.3 at 0.430 vs team-only 0.437. | `OC_INHERITANCE_FIT_2026-08-14.md`. Honest and well-run, but the objective is coordinator tendency reproduction, not player accuracy. Margin over the judgment default is 2.6%. Grid is coarse (5 points) and the module docstring still describes the superseded 70/30 default. |
| **OL trailing average** | `ol_quality.trailing_for_seasons` | **B** | Next-season OL-score persistence MAE, 2023–2025 holds, 5/6 fold×metric cells. | Live-path only; introduces train/serve feature skew. |
| **Replacement-level baselines** | `replacement.py` | **B** | Conditional means of real historical per-game outcomes by (position, preseason depth band), full-cohort availability separately. No held-out gate. | The *estimator* is sound and mirrors the rookie/Gate-B pattern. The *scope decisions* (`REPLACEMENT_POSITIONS`, `REPLACEMENT_MIN_CELL`, `REPLACEMENT_DEPTH_BANDS`) are D. Acceptance evidence in `fca3525` is explicitly Sleeper: "Sleeper agreement improves on every metric — overall correlation 0.950 → 0.951, RB correlation 0.953 → 0.955 … RB MAE 16.63 → 16.38." Ships 14 rows. |
| **Incumbent vacancy boost** | `roster_moves.apply_incumbent_vacancy_boost` | **C** | target α: share MAE vs carry-forward (B). carry α: share MAE (B) for the *value*, Sleeper for the *ship decision*, twice. | See constant #18. |
| **Team-changer vacancy netting + damping** | `roster_moves.reassign_team_changers` | **B** | Share MAE over 2017→2025 transitions vs naive carry-forward; netted+damped 0.05460 targets / 0.19082 carries. | Strong internal logic (netting alone loses to naive; damping alone loses to naive; only net-then-shrink wins). Metric is still share, and live tie-breaks used Sleeper corr/MAD. |
| **Rookie residual vacancy netting** | `predict._attach_rookie_residual_vacancy` | **D** | None — an accounting-identity argument. | `4803957`. The argument is correct and important (the same opening was being spent twice). The stated evidence is nonetheless "WR correlation with Sleeper 0.934 → 0.939, the first WR movement of this remediation". |
| **Named-supply rushing reconcile** | `team_reconcile.normalize_team_rushing_volume` | **B** | Coverage fraction 0.814 measured 2017–2025 (B); existence of the mechanism decided on RB season MAE vs Sleeper (C). | See constant #32. |
| **Named-supply receiving reconcile** | `team_reconcile.normalize_team_passing_volume` | **B** | Coverage fractions measured on nflverse 2016–2024 (yards, TDs); receptions asserted by analogy. | `team_passing_volume_scale` mean 1.159, max **3.11** — this layer moves numbers a great deal. |

---

## 3. Counts

Of 43 constants, **13 are class E** (8 paths, 3 schema/enum lists, `USAGE_SHARE_FAMILIES`, and the dead `DEPTH_RANK_TO_WR_FORMATION_ROLE`). That leaves **30 real tuning knobs**:

| Class | Count | Share of knobs | Constants |
|---|---:|---:|---|
| **A** — real outcomes | **4** | 13% | `DEPTH_RATE_LADDER`, `DEPTH_RATE_DEEP`, `DEPTH_RATE_OFF_CHART` (all three: fit A, shipped values hand-shaded), `USAGE_SHARE_BLEND_W` |
| **B** — proxy metric | **8** | 27% | `DEEP_BENCH_GAMES_CAP`, `TEAM_CHANGE_VACANCY_ALPHA`, `NAMED_REC_YARDS_COVERAGE`, `NAMED_REC_TDS_COVERAGE`, `QB_ATTEMPTS_PER_VOLUME_GAME_MAX`, `NAMED_RUSH_COVERAGE`, `OL_TRAILING_SEASONS`, `WR_FORMATION_ROLE_PRIORS` |
| **C** — Sleeper-tuned | **1** | 3% | `INCUMBENT_VACANCY_ALPHA` |
| **D** — judgment, unmeasured | **17** | 57% | `CURATED_RESEARCH_DEPTH`, `ROOKIE_RATIO_FALLBACK`, `VACATED_CLIP`, `TEAM_CHANGE_SHARE_CLIP`, `BOOST_ELIGIBLE_ROLES`, `INCUMBENT_VACANCY_NET_CLIP`, `INCUMBENT_VACANCY_SCALE_CAP`, `REPLACEMENT_POSITIONS`, `REPLACEMENT_MIN_CELL`, `REPLACEMENT_DEPTH_BANDS`, `PASS_CATCH_COHERENCE_BAND`, `NAMED_REC_RECEPTIONS_COVERAGE`, `RUSH_ATTEMPTS_PER_APPEARANCE_MAX`, `RUSH_YARDS_PER_CARRY_MAX`, `USAGE_SHARE_CURATED_W`, `FORMATION_ROLE_BLEND_W`, `USAGE_SHARE_MAX_RANK` |

For the 19 fitted layers in §2: **A 6, B 9, C 1, D 3**.

Two summary observations.

1. **The C count is misleadingly small.** Only one constant was *set* by fitting to
   Sleeper. But Sleeper is the **acceptance criterion** in the commit body or
   report for `DEPTH_RATE_LADDER`/`DEEP`/`OFF_CHART` (Gate B), Gate A,
   `NAMED_RUSH_COVERAGE`, the replacement-level path, the rookie vacancy netting,
   both vacancy alphas, and every one of the 11 "WR consensus gap" phase commits.
   Counting *decisions* rather than *constants*, Sleeper is the deciding evidence
   in roughly half the allocation layer.
2. **The D majority is the bigger problem than the C.** 17 of 30 knobs have no
   measurement of any kind, and several of them bind on the shipped board
   (`VACATED_CLIP` at its floor on 317 rookie rows; `FORMATION_ROLE_BLEND_W` on
   352 WR rows; `USAGE_SHARE_CURATED_W` on 50 rows; `RUSH_ATTEMPTS_PER_APPEARANCE_MAX`
   set the headline carry number for Jacobs and Jeanty before `10303b5`).

---

## 4. Sleeper feedback-loop inventory

### 4.1 Sleeper feeding a fitted value or a tuning decision (must be retired)

| # | Site | What Sleeper decides |
|---|---|---|
| F1 | `INCUMBENT_VACANCY_ALPHA["carry"]` | Disabled at 0.0 in `9d5e533` on live Sleeper RB corr/MAD despite the module's strongest share-MAE evidence; re-enabled to 1.0 on `RB_CARRY_VACANCY_2026-08-14.md`, whose diagnostic table is 4/5 Sleeper rows. |
| F2 | `scripts/diag_rb_carry_vacancy.py` | The instrument behind F1. Reads `output/sleeper_comparison_2026.csv` and nothing else. Every printed statistic is a delta against Sleeper. Untracked. |
| F3 | `TEAM_CHANGE_VACANCY_ALPHA["carry"] = 0.25` | Historical share MAE picked the value; the live Sleeper RB MAD regression (2.386→2.490) is what the commit argues about, and the α=0 counterfactual was rejected on Sleeper corr/MAD. |
| F4 | `NAMED_RUSH_COVERAGE` mechanism | The no-fill alternative was rejected on "RB season-total MAE 15.6 → 19.0" — vs Sleeper. `DEPTH_CHART_ALLOCATION_2026-08-14.md`: "The decisive test was Sleeper itself." |
| F5 | Gate B ladder acceptance (`720fa8e`) | The commit's headline results table is season-scale MAE vs Sleeper by role tier. The underlying fit is class A; the *ship* argument is Sleeper. |
| F6 | Gate A acceptance (`8be9c63`) | "Season-scale MAE vs Sleeper 24.8 → 22.9 overall, starters 46.2 → 39.6" is cited as corroboration. |
| F7 | Replacement-level rows (`fca3525`) | Sole quantitative result: "Sleeper agreement improves on every metric." |
| F8 | Rookie vacancy netting (`4803957`) | Sole quantitative result: "WR correlation with Sleeper 0.934 → 0.939." |
| F9 | `DEPTH_CHART_ALLOCATION_2026-08-14.md` "Net effect" table | 3 of 6 rows are Sleeper correlations. This is the summary document for the whole 5-phase remediation. |
| F10 | The 11 "WR consensus gap" commits (`6a15b2d`…`9d5e533`) | Per-player delta vs Sleeper is the stop/go criterion in every one. Includes two "user decision at gate" commits (`1f7f6e8` Gainwell backup→committee; `6b48273` share cap 1.5→1.2) where the presented evidence was Sleeper deltas. |
| F11 | `src/comparison/spot_check.py` | The standing "run after every pipeline change" gate. Its `WATCHLIST` is annotated with the direction players are expected to move **toward Sleeper**; `CONTROLS` must not move relative to Sleeper. Structurally a Sleeper-agreement regression suite. |

### 4.2 Dead Sleeper coupling (delete)

| Site | Status |
|---|---|
| `sleeper_compare.fetch_sleeper_play_probability`, `NO_STATS_PLAY_PROB = 0.05`, `HAS_STATS_PLAY_PROB = 1.0` | `f5a5d09` multiplied a rookie QB's entire projection line by a Sleeper-derived binary. Removed from `rookies.py` in `df37452`. The function and both constants survive in `src/comparison/sleeper_compare.py:185-229` with **zero callers**. This is a loaded gun: the docstring still describes it as a prediction-quality fix. |

### 4.3 Sleeper as a data feed, not a target (keep, but name it)

| Site | Role |
|---|---|
| `src/depth_chart/sleeper_status.py` → `refresh.py`, `events.py` | Ingests Sleeper's player **injury/status** fields to propose depth-chart and `status_overrides_2026.csv` changes. This is Sleeper-as-news, categorically different from Sleeper-as-target, and legitimate. It does move projections (via `apply_status_overrides` and the live depth chart), so it belongs in any dependency inventory — but it is not a fitting objective and should not be removed. |
| `src/crosswalk.py` | ID resolution. Infrastructure. |

### 4.4 Sleeper as read-only diagnostic (keep as-is)

`src/comparison/sleeper_compare.py` season-total comparison,
`output/sleeper_comparison_2026.csv`, `output/sleeper_snapshots/*`. The
content-addressed snapshotting with SHA-256 metadata (`VALIDATION_EVALUATION_REMEDIATION.md`)
is good practice. Sleeper genuinely found the share-denominator bug and the
trade-vacancy bug (`9835a72`, `737eadf`) and the Diggs/Okonkwo/White triple-boost
(`4f3e452`); it earns its place as a bug detector.

### 4.5 What must change to retire Sleeper as a target

Ordered by how much each unblocks the others.

1. **Extend `fantasy_evaluation.py` to run the full allocation stack.** Today it
   runs the reconcilers but not `team_pass_mix`, `team_rush_mix`, `roster_moves`,
   `replacement`, or `depth_gating` (§0.2). Until it does, no vacancy alpha and no
   mix layer *can* be class A, and Sleeper will keep filling the vacuum. This is
   the prerequisite for items 2–4. The blocker named in `VALIDATION_EVALUATION_REMEDIATION.md`
   is real — "the historical path has no dated curated role file, full rookie-room
   composition, or production QB/team reconciliation entry point" — and the fix it
   names is a `project_season_as_of` interface. Build that.
2. **Re-decide F1, F3, F4 on that harness.** Specifically:
   `INCUMBENT_VACANCY_ALPHA["carry"] ∈ {0.0, 1.0}`,
   `TEAM_CHANGE_VACANCY_ALPHA["carry"] ∈ {0.0, 0.25, 1.0}`,
   `NAMED_RUSH_COVERAGE ∈ {0.0, 0.814, 1.0}` — scored on points MAE, VORP MAE,
   Spearman, and tier hits. Accept whatever it says, including reverting.
3. **Re-point `spot_check.py` at truth.** The named-player watchlist is a genuinely
   good idea and has caught real silent-drop bugs twice. Keep the mechanism;
   replace the Sleeper reference column with actual prior-season outcomes and
   an explicit "does this player exist in the output at all" assertion. A
   named-player check does not need a consensus to be useful.
4. **Rewrite `scripts/diag_rb_carry_vacancy.py`** against actuals, or delete it.
   As written it can only ever answer "do we agree with Sleeper".
5. **Delete `fetch_sleeper_play_probability`, `NO_STATS_PLAY_PROB`,
   `HAS_STATS_PLAY_PROB`** from `sleeper_compare.py` (§4.2).
6. **Adopt a documentation rule**: no Sleeper number may appear in a "Net effect",
   "Results", or "Decision" section of any report. Sleeper deltas go in a section
   explicitly headed as a diagnostic, alongside the standing note that Sleeper
   projects full slates (`gp = 18.00` for 9,370 of 9,402 players) and allocates
   96.8% of team carries to named players against our 83.8%. `FREEZE_2026-08-13.md`
   already states the principle — "Sleeper remains a rough season-total comparison,
   not the optimization target" — and eight commits since then violate it.
7. **Record the pass-mix LOSO result.** `HIERARCHICAL_PASS_MIX_2026-08-14.md` asserts
   a ship condition and reports no number. Either the gate was run and the result
   was not written down, or it was not run. Both are fixable in one command.

---

## 5. Ranked deletion candidates

Ranked by (weakness of provenance) × (statically inferable leverage on the shipped
board). Confidence is my confidence in the *classification and leverage estimate*,
not a prediction of the ablation result.

| Rank | Target | Class | Leverage evidence | Confidence | Why |
|---|---|---|---|---|---|
| **1** | `FORMATION_ROLE_BLEND_W = 0.5` + `WR_FORMATION_ROLE_PRIORS` | D / B | 352 shipped WR rows carry a `formation_role`; blend reweights the entire within-WR budget | **High** | The clearest documented contradiction in the repo. These are the same numbers as the fitted rank prior that was measured to *lose* on the leakage-safe fantasy evaluation and deliberately disabled (`USAGE_SHARE_BLEND_W = 0.0`), re-entering at weight 0.5 through an unmeasured constant that has never been committed to git. Ablate to `FORMATION_ROLE_BLEND_W = 0.0` first — the code already falls back to `_allocate_within_group`, and `tests/test_team_pass_mix.py` already covers that path. |
| **2** | L3 `apply_hierarchical_pass_distribution` | D | `hierarchical_pass_scale` mean **1.467**, range **0.00–4.05**, 2,728 rows | **High** (leverage) / Medium (that removal is safe) | The single largest multiplier in the shipped board and it has **no gate at all**. Its L2 input has an advisory-only proxy gate whose result is unrecorded. A `hierarchical_pass_scale` of 0.00 on any row is worth investigating independently of the ablation. |
| **3** | `INCUMBENT_VACANCY_ALPHA["carry"] = 1.0` | C | Doc's own example: `v_net = 0.30 → scale 1.43` on high-turnover RB rooms | **High** | The only pure class-C constant. Disabled on Sleeper evidence, re-enabled on Sleeper evidence, never once scored on fantasy points. Project memory records the RB lead-back level bias as the standing reason it was off. Flipping to 0.0 is a one-line ablation and `4803957` guarantees the rookie-netting coupling stays consistent either way. |
| **4** | `VACATED_CLIP = (0.3, 2.5)` | D | **317 of 1,134 rookie rows ship at exactly 0.30** — 28% of the rookie board is sitting on an unmeasured clamp | **High** | Highest ratio of "binding on real output" to "evidence" in the whole file. Introduced in `55953fa` with a five-word docstring. `TEAM_CHANGE_SHARE_CLIP` aliases it, so one ablation covers two constants. Vary the floor over {0.0, 0.2, 0.3, 0.5} and watch rookie points MAE. |
| **5** | L2 `team_rush_mix` scheme model | B | `hierarchical_rush_scale` mean 1.354, max 2.58, 831 rows; `rush_mix_source` = `scheme_model` on 100% of rows | **High** | A 1.2% edge (0.0343 vs 0.0347) on team rush *share* over ~9 LOSO folds, called a "measured win", driving a mean 1.35× multiplier on every RB. The prior-season baseline it beat is the natural ablation arm and is already implemented in `validate_rush_mix_model`. |
| **6** | `USAGE_SHARE_CURATED_W = 0.5` + the 11 reviewed rows | D | `usage_share_blend_factor` non-null on 50 rows, 0.764–1.263 | **High** | The weight is asserted by analogy; the 11 priors are hand-typed round numbers (0.24, 0.18, 0.06…). `482f626` shipped this deliberately inert; it has since been switched on for four teams. Low blast radius, so cheap to ablate — but note the mechanism *does* run inside `fantasy_evaluation.py`, so unlike ranks 1–5 this one is actually scoreable today. |
| **7** | `RUSH_ATTEMPTS_PER_APPEARANCE_MAX`, `RUSH_YARDS_PER_CARRY_MAX` | D | Currently 0 rows capped (`CAPPED` tripwire silent) — but RB=25.0 set Jacobs' and Jeanty's headline carries before `10303b5` | **High** (provenance) / **Low** (current leverage) | Completely undocumented — no comment, no report, no derivation. WR/TE at 15.0 yd/carry looks like a placeholder. Currently non-binding, which makes them cheap to remove and a good test of whether "non-binding" holds under other ablations. Removing them while ablating rank 5 is the interesting combination. |
| **8** | `INCUMBENT_VACANCY_NET_CLIP = 0.75`, `INCUMBENT_VACANCY_SCALE_CAP = 2.0` | D | Both clamp `apply_incumbent_vacancy_boost`, which fires on curated starters/committee | **Medium** | Introduced in `9d5e533`, which carefully enumerates what it measured and does not include these. No comment anywhere. Their binding rate is not statically determinable from the shipped CSV — needs instrumentation. |
| **9** | `NAMED_REC_RECEPTIONS_COVERAGE = 0.98` | D | Feeds `_named_supply_target` for receptions; `team_pass_receive_count_scale` mean 1.011, range **0.418–3.431** | **Medium** | Asserted by analogy to the yards figure and never separately measured, unlike its two siblings. The 0.418–3.431 range on the count scale says this reconciler moves real numbers. Ablate by measuring the receptions coverage fraction directly rather than by deletion. |
| **10** | `DEEP_BENCH_GAMES_CAP = 6.0` | B | Applies to 2,463 rows with `depth_chart_status == 'deep_bench_discounted'` | **Medium** | Measured at 5.51 and shipped at 6.0. More importantly it is an unvalidated hard clamp on top of the *validated* Gate A availability model — exactly the "two errors can cancel" shape in project memory. Worth measuring whether Gate A alone already produces the right games for this cohort. |
| **11** | `OL_TRAILING_SEASONS = 3` | B | Feature-level; live-path only | **Medium** | Gated on OL-score persistence MAE (a proxy) and applied only to live features, creating train/serve skew. Never committed. Ablate to 0 — the doc names it as the kill-switch. Leverage on fantasy points is genuinely unknown; this is a measurement request more than a deletion candidate. |
| **12** | `DEPTH_RANK_TO_WR_FORMATION_ROLE` | E | Zero consumers | **High** | Dead code. Delete outright, no ablation needed. Also encodes an assertion the curated chart's own notes column contradicts. |
| **13** | `PASS_CATCH_COHERENCE_BAND = (0.8, 1.35)` | D | 0 of 4,039 rows flagged | **High** | Self-declared "un-tuned judgment call". Zero effect on numbers today, so **not** a deletion candidate for the ablation — listed here so it can be explicitly excluded and not waste a run. Keep it as a diagnostic. |

### Explicitly *not* deletion candidates

`USAGE_SHARE_BLEND_W = 0.0` (the model process working correctly),
the elite shrinkage correction and its `MIN_SEASON_CONSISTENCY` guard (best
provenance in the stack, and the guard is the mechanism that caught the WR
pooling artifact), `AGE_EFFECT_SHRINK`, `RECEIVING_SHARE_SUM_CAP = 1.2`, the Gate A
availability model, and `NAMED_RUSH_COVERAGE = 0.814` **as a value** — the 0.814
estimate is sound and stable (0.776–0.869, no trend); what needs re-deciding is
whether the fill mechanism should exist at all, and that is item 2 of §4.5, not a
deletion.

### One caveat on the whole list

Ranks 1–5 and 8–11 cannot be scored by any harness currently in the repo (§0.2).
Running them as ablations against `output/projections_2026.csv` would measure
*board movement*, not accuracy — which is the same mistake in a new costume. The
ablation study should be sequenced behind §4.5 item 1, or its results read
strictly as sensitivity analysis rather than as evidence about correctness.

---

## Undetermined

- The pass-mix L2 LOSO result. Not recorded in any file; not computed here because
  doing so requires running the fitting path.
- The binding rate of `INCUMBENT_VACANCY_NET_CLIP` and `INCUMBENT_VACANCY_SCALE_CAP`
  on the 2026 board — not emitted to any output column.
- The derivation of `RUSH_ATTEMPTS_PER_APPEARANCE_MAX`, `RUSH_YARDS_PER_CARRY_MAX`,
  `INCUMBENT_VACANCY_NET_CLIP`, `INCUMBENT_VACANCY_SCALE_CAP`, `REPLACEMENT_MIN_CELL`,
  `USAGE_SHARE_MAX_RANK`, and `FORMATION_ROLE_BLEND_W`. Searched commits, reports,
  code comments, and docstrings; found nothing. These are recorded as D on the
  absence of evidence, not on evidence of absence.
- Why `starters_2026.csv` currently has 11 reviewed rows when `0329ee6` describes
  activating "reviewed WR usage priors for all 32 rooms". The file is modified in
  the working tree; the intervening state was not reconstructed.
