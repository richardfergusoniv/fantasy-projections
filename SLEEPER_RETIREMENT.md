# Retiring Sleeper as a fitting and acceptance target — 2026-08-15

Companion to `PROVENANCE_AUDIT.md` §4. That audit catalogued 11 fitting/decision
uses of Sleeper (F1–F11), one dead coupling, and the legitimate uses. This
document records what was actually executed, what was re-assessed as newly
unblocked, and what remains blocked with the specific reason.

**Settled position this work implements:** Sleeper projects full slates
(`gp = 18` for ~9,370 of the 9,402 players it tracks) and allocates ~96.8% of
team carries to named players against our ~83.8%. This system projects expected
value, including the probability a player does not play. The two are
differently framed, so **agreement is not accuracy**. Sleeper stays as a
read-only bug detector — it genuinely found the share-denominator bug, the
trade-vacancy bug and the Diggs/Okonkwo/White triple-boost — and stops being an
objective.

**Scope constraint:** `src/projection/` was owned by another agent during this
work and was not modified. Nothing in this document changed a file under
`src/projection/`. Everything requiring such a change is in the hand-off list.

---

## 1. What changed

| File | Change |
|---|---|
| `src/comparison/sleeper_compare.py` | Deleted the dead Sleeper→prediction coupling; reframed all summary output as descriptive divergence, never loss. |
| `src/comparison/spot_check.py` | Rewritten. Was a Sleeper-agreement regression suite; is now a coherence gate on our own board with a non-failing Sleeper divergence report. |
| `scripts/diag_rb_carry_vacancy.py` | Quarantined with a header stating it cannot inform a ship decision. Not deleted (see §5). |
| `tests/test_validation_evaluation_integrity.py` | Added `SleeperIsNotATargetTests` — 6 tests locking the new failure semantics. No existing test weakened or removed. |

### 1.1 Dead coupling removed (audit §4.2, path item 5)

Deleted from `src/comparison/sleeper_compare.py`:
`fetch_sleeper_play_probability`, `NO_STATS_PLAY_PROB = 0.05`,
`HAS_STATS_PLAY_PROB = 1.0`.

This multiplied a rookie QB's entire projection line by a binary derived from
whether Sleeper published a `pass_att` field (`f5a5d09`). Its consumer in
`rookies.py` was removed in `df37452`; the function survived with a docstring
still describing it as a prediction-quality fix.

**Verified, not inferred.** Before deleting I ran a repo-wide search (excluding
`.venv/` and `.git/`) for `fetch_sleeper_play_probability`, `NO_STATS_PLAY_PROB`,
`HAS_STATS_PLAY_PROB`, `play_prob` and `PLAY_PROB` across `src/`, `tests/`,
`scripts/`, `draft_assistant/`, notebooks and all Markdown. Results:

- **Zero** callers in `src/`, `tests/` or `scripts/`.
- The only live references were the definitions themselves.
- Remaining hits are documentation-only: `PHASE4_REPORT.md:401-416` (historical
  record of the removed mechanism) and `PROVENANCE_AUDIT.md` (this finding).
- Because the last dead-code claim on this repo missed a transitive chain, I
  also searched for dynamic dispatch — `getattr(`, `importlib`, `__import__`,
  `eval(`, `globals()[` — across every `.py` file outside `.venv/`. **No
  matches anywhere in the repo.** There is no indirect call path.
- Traced the full inbound edge set of `sleeper_compare` rather than only direct
  imports of the function. Three modules import from it, and all three import
  only survivors: `src/depth_chart/sleeper_status.py` (`SNAPSHOT_DIR`,
  `_fetch_json`, `_normalize_name`, `PLAYERS_URL`),
  `src/depth_chart/events.py` (`_normalize_name`),
  `src/depth_chart/refresh.py` (`_normalize_name`). Confirmed post-deletion by
  importing all three plus `spot_check` successfully.

Nothing under `src/projection/` was involved — the dead code lived entirely in
`src/comparison/`, which I own. **No hand-off needed for this item.**

A comment marks the deletion site so the next reader sees why it must not
return, and `test_sleeper_derived_play_probability_multiplier_is_gone` fails if
any of the three names reappears on the module.

### 1.2 `sleeper_compare.py` reframed (audit F9-adjacent, path item 6)

The comparison itself is kept unchanged — the join logic, the two-tier
gsis_id/name matching, the content-addressed SHA-256 snapshotting, and every
emitted column are untouched. What changed is everything that read as an
objective:

| Before | After | Why |
|---|---|---|
| `season_mae` column | `mean_abs_divergence` | "MAE" names a loss; a loss implies a target. |
| `season_bias` column | `mean_signed_divergence` | "Bias" asserts one side is correct. |
| `"evaluation strata … cannot hide relevant-player error"` | `"divergence strata … cannot hide the relevant players"` | Divergence is not error. |
| `"Mean absolute season-total delta"` | `"Mean absolute season-total divergence"` | Same. |
| per-position `season_mean_abs_delta` | `mean_abs_divergence` | Same. |
| Module docstring: endpoints only | Adds an explicit "what this is and is not" section | The framing was previously carried only by convention, and eight commits since `FREEZE_2026-08-13.md` violated it. |
| No banner | `DIAGNOSTIC_BANNER` printed first | The first thing a reader sees is that nothing below is an acceptance criterion. |
| Ends with `Written -> path` | Also points at `fantasy_evaluation.py` for accuracy decisions and `spot_check` for a pass/fail gate | Removes the vacuum that made Sleeper the default scoreboard. |

`comparison_summary_strata` keeps its name (no external callers — verified) so
the function is still findable from the reports that cite it.

**Note:** the tool has no failure mode at all — `sleeper_compare` writes a CSV
and exits 0 regardless. It never was a gate, so nothing needed to be removed
from its exit semantics; the problem was purely that its *vocabulary* invited
readers to treat its numbers as loss. That is what was fixed.

---

## 2. `spot_check.py`: before / after failure semantics

This is the substantive change and the one the audit ranked as F11.

### Before

| Aspect | Behaviour |
|---|---|
| Data source | `output/sleeper_comparison_<season>.csv`, filtered to `matched_sleeper == True`. **Could not run at all without Sleeper.** |
| Watchlist framing | Header printed `"watchlist (should converge toward Sleeper)"`; each entry annotated with the direction it was expected to move toward consensus. |
| Controls framing | `"Controls (must NOT move materially)"` — i.e. must not move *relative to Sleeper*. |
| Reference column | `sleeper_fpts_season`, `sleeper_rec_yards` — a third party's forecast of the season being projected. |
| Summary | `position_bias`: mean/median `ours − Sleeper`, headed `"Position bias"`. A number to drive toward zero. |
| Exit code | `sys.exit(1)` only when a watched player was missing from the **Sleeper-matched** frame. So a player present in our board but unmatched by Sleeper's join failed the run. |
| Net effect | A standing "run after every pipeline change" gate whose only failure condition was mediated by Sleeper, wrapped in language that made converging on Sleeper the goal. |

### After

| Aspect | Behaviour |
|---|---|
| Data source | `output/fantasy_points_<season>.csv` — **our own board**. The Sleeper file is optional; if absent the script says so and continues. |
| Failure conditions | **Only incoherence**, four rules, all computed without any external reference (§2.1). |
| Watchlist framing | Annotations describe the *structural risk each row is a canary for* (short-season availability handling, depth-chart gating drops, rookie-cohort carry-through). No entry asserts a direction relative to anything. |
| Controls → `STABILITY_ANCHORS` | Reported for a human to read. Explicitly **not** a failure condition, because this file contains no ground truth that could say what the right number is. |
| Reference column | **Actual production in the prior completed season**, computed from our own database (`data_prep.load_weekly_usage` → `season_aggregate`, scored with `fantasy_points.SCORING`). This is the audit's path item 3 executed literally: real outcomes replace the consensus column. Degrades to a printed explanation if the DB is unavailable — never silently back to Sleeper. |
| Divergence report | Retained, printed under `=== DIAGNOSTIC ===`, sorted largest-first so real bugs stay loud, preceded by the framing note. **Cannot touch the exit code.** |
| `position_bias` → `position_divergence` | Same numbers, renamed, with a docstring stating that non-zero is the expected consequence of the framing difference and is not to be driven toward zero. |
| Join audit | Kept as-is and explicitly labelled `"a join defect IS actionable"` — a name collision is a bug in our matching, not a disagreement. |

### 2.1 The four coherence rules

Each is a statement about our board alone that cannot be true of a real NFL
season. None reads a `sleeper_` column.

| Rule | Fires when |
|---|---|
| `missing_from_output` | A watched player is absent from the board. This is an existence check on our own output — the exact shape of the Phase 5 rookie-filter bug and the Phase 6 team-change bugs — and is preserved as a hard failure. It is *not* mediated by Sleeper any more: a player present in our board but unmatched by Sleeper's join no longer fails the run. |
| `negative_volume` | Any `pg_*`, `our_*`, `projected_games`, `projected_volume_games`, `fantasy_pts` or `fantasy_pts_season` is negative. |
| `player_exceeds_team_total` | A player's season total exceeds his own team's projected total for that stat. Checked for carries, rushing yards, targets vs pass attempts, receiving yards, pass attempts, passing yards. |
| `team_over_allocated` | A team's named players are collectively allocated more than the team anchor — the "shares summing past 1" check, expressed against the anchor rather than a normalized share column, so it still fires if a share column is itself wrong. |

**Threshold calibration (measured, not guessed).** Per-game rates are converted
to season totals with each player's own exposure *before* summing, because a
per-game rate is conditional on playing and does not sum across a roster —
skipping that step would have produced false alarms on every team. Measured on
the live 778-row board:

- per-player max ratio to team total: carries 0.720, rushing yards 0.713,
  targets 0.325, receiving yards 0.352, attempts 0.984, passing yards 0.987.
  Zero rows over 1.0 → strict threshold (1.0 + float epsilon) is safe.
- team allocation ratios: the reconcilers pin named supply at the anchor, so
  these sit at 0.899–1.000 with float noise producing a handful of nominal
  `> 1.0` readings. `TEAM_ALLOCATION_TOLERANCE = 0.02` sits above the noise
  and below any real over-allocation; **zero teams exceed 1.05 today**.

### 2.2 Verification

`python -m src.comparison.spot_check --season 2026` runs clean, exit 0: no
incoherence, all 15 watched players present, prior-season actuals resolved for
every one. The divergence section still surfaces the large gaps loudly — Parker
Washington −116.4, Jayden Reed −80.2, Chris Godwin −60.8 — as information.

Incidental finding while verifying: `output/sleeper_comparison_2026.csv` is
**stale relative to `output/fantasy_points_2026.csv`** (e.g. Malik Nabers
181.0 in the comparison vs 186.2 on the board). Not caused by this work — the
comparison CSV is a cached artifact from an earlier pipeline run. It is a
further reason the coherence gate should read the board directly, which it now
does. Not fixed here (regenerating it needs a network fetch and would rewrite a
tracked output artifact).

---

## 3. Status of the 11 catalogued fitting uses

Legend: **Retired** — the mechanism is gone. **Reframed** — the artifact
survives, its status as an objective does not. **Blocked** — still decided on
Sleeper, with the specific blocker named.

| # | Site | Status | Detail |
|---|---|---|---|
| **F1** | `INCUMBENT_VACANCY_ALPHA["carry"] = 1.0` | **Blocked** | The constant lives in `src/projection/contracts.py` (not mine to edit) and, more fundamentally, is unscoreable: `apply_incumbent_vacancy_boost` is called from `veterans.py`, which sits on the **forecast** side of `composition.compose_board`. `fantasy_evaluation.py` builds its veteran rows via `_forecast_from_history` and never executes it. Unblocked by: bringing `roster_moves` inside the harness's forecast stage. |
| **F2** | `scripts/diag_rb_carry_vacancy.py` | **Quarantined** | Header now states it reads only `sleeper_comparison_2026.csv`, that every statistic is a Sleeper delta, and that it is structurally incapable of informing the ship decision its original docstring posed. Kept rather than deleted (§5). Rewriting against actuals is blocked behind F1's blocker. |
| **F3** | `TEAM_CHANGE_VACANCY_ALPHA["carry"] = 0.25` | **Blocked** | Same blocker as F1 — `reassign_team_changers` is likewise called from `veterans.py`, and `fantasy_evaluation.py:738` states explicitly that it is not run (the target team comes from the frozen Week-1 roster instead). |
| **F4** | `NAMED_RUSH_COVERAGE` mechanism | **NEWLY UNBLOCKED — not yet executed** | This is the material re-assessment. `NAMED_RUSH_COVERAGE` is consumed by `team_reconcile._named_supply_target` inside `normalize_team_rushing_volume`, which **is** in `compose_board` (line 262) and **is** therefore executed by `fantasy_evaluation.py`. The audit wrote this off with items 2–4 as blocked behind harness work; that harness work has landed. `NAMED_RUSH_COVERAGE ∈ {0.0, 0.814, 1.0}` can be scored on held-out points MAE / VORP MAE / Spearman / tier hits **today**. Not executed here: it requires editing `src/projection/contracts.py`. See hand-off. |
| **F5** | Gate B ladder acceptance (`720fa8e`) | **Reframed (historical)** | The underlying fit is class A and stands. The Sleeper acceptance table is in a commit body and cannot be rewritten without rewriting history. Governed going forward by the documentation rule (§4). `DEPTH_RATE_*` are consumed by `depth_rates.depth_rate_factor`, which the harness already runs, so a re-acceptance on outcomes is available whenever wanted. |
| **F6** | Gate A acceptance (`8be9c63`) | **Reframed (historical)** | Sleeper was corroboration, not the deciding evidence; the held-out games-played MAE (8/8 folds) is genuine class A. No action needed beyond the documentation rule. |
| **F7** | Replacement-level rows (`fca3525`) | **Blocked** | Sole quantitative result was "Sleeper agreement improves on every metric". `replacement.py` is imported by `predict.py` only, not by `compose_board`, and replacement rows exist only for curated players — and `starters_<season>.csv` exists for 2026 alone, so no historical fold can construct them. Two stacked blockers. |
| **F8** | Rookie vacancy netting (`4803957`) | **Blocked** | Sole quantitative result was "WR correlation with Sleeper 0.934 → 0.939". `_attach_rookie_residual_vacancy` lives in `predict.py`, forecast-side, outside `compose_board`. The accounting-identity argument for the mechanism is independently correct; only its *evidence* was Sleeper. |
| **F9** | `DEPTH_CHART_ALLOCATION_2026-08-14.md` "Net effect" table | **Documented, not rewritten** | 3 of 6 rows are Sleeper correlations in a section headed "Net effect". Left intact as a historical record; the standing rule in §4 forbids the pattern in new reports. Rewriting a dated report would destroy the audit trail the provenance work depends on. |
| **F10** | The 11 "WR consensus gap" commits | **Reframed (historical)** | Commit bodies are immutable. Two of them (`1f7f6e8`, `6b48273`) were user decisions at a gate on Sleeper-delta evidence and would need re-deciding on the harness; `6b48273`'s `RECEIVING_SHARE_SUM_CAP` is in `transitions.py`, which the harness does reach, so it is re-scoreable today. `1f7f6e8` (Gainwell backup→committee) is a curated-chart edit and is unscoreable on any historical fold. |
| **F11** | `src/comparison/spot_check.py` | **RETIRED** | Fully executed. See §2. It is no longer a Sleeper-agreement regression suite; Sleeper cannot fail it. |

**Summary: 1 retired (F11), 1 quarantined (F2), 1 newly unblocked and ready
(F4), 4 blocked on the same structural cause (F1, F3, F7, F8), 4 historical
records governed by the documentation rule (F5, F6, F9, F10).**

Plus, outside the F-list: the §4.2 dead coupling is **deleted**.

### 3.1 Tests: no agreement-with-Sleeper pass condition was found

The brief flagged this as a likely finding. **I checked and it is not the case
in the test suite** — reporting it explicitly because a negative result here is
load-bearing.

Every Sleeper-touching test asserts *mechanics*, not agreement:

- `tests/test_runtime_output_correctness.py:709` —
  `test_sleeper_comparison_is_season_total_and_invalidates_fake_rate` asserts
  delta *arithmetic* on synthetic rows (`fantasy_pts_season_delta == 10.0`,
  `our_passing_yards_season == 1600.0`) and, importantly, that the fake `gp=18`
  rate is invalidated to NaN. This test is *part of the defence*, not part of
  the problem.
- `tests/test_validation_evaluation_integrity.py` — asserts name/team
  disambiguation, collision handling, and snapshot SHA-256/timestamp presence.

No test asserted a threshold on divergence, a correlation floor, or a
convergence direction. **No test was weakened or deleted.** The agreement-gate
behaviour lived entirely in `spot_check.py`'s `sys.exit(1)`, which was a script
exit code rather than a test, and is now driven only by incoherence.

Six tests were **added** (`SleeperIsNotATargetTests`), pinning the contract:
the deleted names must stay deleted; a coherent board passes however wildly it
diverges from Sleeper; and each of the four incoherence rules fires.

---

## 4. Re-assessment of the audit's 7-step path

The audit wrote items 2–4 as blocked behind item 1. `src/projection/composition.py`
has since landed and unifies the shipped and evaluated paths. Re-assessed:

| Step | Audit status | Status now | Reason |
|---|---|---|---|
| **1.** Extend `fantasy_evaluation.py` to run the full allocation stack | Blocked | **Substantially landed, with a named residual** | `compose_board` is now the single stage list, and `fantasy_evaluation.py:452` calls it. Newly inside the harness: `team_pass_mix` L2+L3, `team_rush_mix` L2+L3, `depth_gating`, `apply_usage_share_prior`, all reconcilers. **Still outside:** `roster_moves` and `replacement`, which `composition.py`'s own docstring places deliberately on the FORECAST side. Separately, `leakage_safe_context` loads `starters_<target>.csv`, which exists for 2026 only — so on any historical fold the curated-dependent stages degrade to pass-throughs. To its credit the harness records this honestly in `composition_stage_coverage` and `coverage_limits` rather than faking it. |
| **2.** Re-decide F1, F3, F4 on that harness | Blocked | **F4 unblocked; F1 and F3 still blocked** | `NAMED_RUSH_COVERAGE` runs inside `compose_board` → scoreable now. Both vacancy alphas run in `veterans.py` → still invisible to the harness. |
| **3.** Re-point `spot_check.py` at truth | Blocked | **DONE** | §2. Did not actually depend on item 1: the coherence rules need only our own board, and the reference column needs only the existing database. |
| **4.** Rewrite `diag_rb_carry_vacancy.py` against actuals, or delete | Blocked | **Quarantined; rewrite still blocked** | Blocked by the same cause as F1 — there is no actuals-based measurement of `apply_incumbent_vacancy_boost` to rewrite it against. |
| **5.** Delete `fetch_sleeper_play_probability` + 2 constants | Ready | **DONE** | §1.1. |
| **6.** Adopt the documentation rule | Ready | **DONE — stated below** | |
| **7.** Record the pass-mix LOSO result | Ready | **Not done — hand-off** | `validate_mix_model` lives in `src/projection/team_pass_mix.py`. Running it is one command but recording the result belongs with the module owner. |

### The documentation rule (step 6)

> **No Sleeper number may appear in a "Net effect", "Results", "Decision" or
> acceptance section of any report or commit body.** Sleeper deltas belong in a
> section explicitly headed as a diagnostic, alongside the standing note that
> Sleeper projects full slates (`gp = 18` for 9,370 of 9,402 players) and
> allocates 96.8% of team carries to named players against our 83.8%.
>
> A change is accepted on `src/projection/fantasy_evaluation.py` (held-out
> actual outcomes) or on a stated mechanistic argument — never on movement
> toward consensus. "It agrees with Sleeper more" is not a result. If a layer
> cannot be scored by the harness, say so and ship on a named judgment; do not
> substitute the scoreboard that happens to be available.

`FREEZE_2026-08-13.md` already stated the principle and eight commits violated
it, so the rule is now enforced in code at the two places output is produced:
`sleeper_compare` prints `DIAGNOSTIC_BANNER` before any number, and
`spot_check` can no longer fail on a Sleeper comparison at all.

---

## 5. Hand-off list for `src/projection/` (not edited — another agent owns it)

Ordered by value. Each is a change I would have made and did not.

1. **Execute F4 — the one newly available re-decision.** Ablate
   `NAMED_RUSH_COVERAGE` in `src/projection/contracts.py:62` over
   `{0.0, 0.814, 1.0}` and score each arm on `fantasy_evaluation.py` (points
   MAE, VORP MAE, Spearman, tier hits). The value 0.814 is well estimated
   (2017–2025, range 0.776–0.869, no trend); what was never tested on outcomes
   is whether the fill **mechanism** should exist. `DEPTH_CHART_ALLOCATION_2026-08-14.md`
   says outright "The decisive test was Sleeper itself" — replace that
   sentence with a harness result, and accept whatever it says including
   reverting. This is the highest-value item because it is the only one of the
   audit's re-decisions that is unblocked today.

2. **Bring `roster_moves` inside the measured path.** This single change
   unblocks F1, F3, F8 and the rewrite of `diag_rb_carry_vacancy.py`. The
   forecast stage of `fantasy_evaluation._forecast_from_history` would need to
   call `reassign_team_changers` and `apply_incumbent_vacancy_boost` on the
   held-out fold. Note the caveat already recorded at
   `fantasy_evaluation.py:738`: the harness deliberately takes the target team
   from the frozen Week-1 roster because that is a *stricter* preseason source
   than the `seasonal_rosters` lookup the shipped path uses. That is a real
   design tension, not an oversight — resolving it is the actual work. Until
   then, `INCUMBENT_VACANCY_ALPHA["carry"] = 1.0` is a live class-C constant
   that was disabled on Sleeper evidence and re-enabled on Sleeper evidence and
   has never been scored on fantasy points.

3. **Record the pass-mix LOSO result (path item 7).** Run `validate_mix_model`
   in `src/projection/team_pass_mix.py` and write the number into
   `HIERARCHICAL_PASS_MIX_2026-08-14.md`, which currently asserts the ship
   condition "ship the scheme mix only when LOSO MAE beats prior-season mix"
   and reports no result — while `mix_source == 'scheme_model'` on 100% of
   4,039 shipped rows. Either the gate was run and not written down, or it was
   not run. Also worth fixing: `build_team_pass_mix_profiles` never calls
   `validate_mix_model`, so the gate is advisory only.

4. **Delete `DEPTH_RANK_TO_WR_FORMATION_ROLE`** from
   `src/projection/contracts.py:80`. I independently re-confirmed the audit's
   finding: zero consumers in `src/`, `tests/` or `scripts/`. Unrelated to
   Sleeper, but it is dead code in a file being touched anyway, and it encodes
   an assertion the curated chart's own notes column contradicts.

5. **Consider a `predict.py` counterpart to the coherence rules.** The four
   rules in `spot_check.py` currently run post-hoc on the written CSV. They are
   cheap and would be stronger as tripwires at the end of
   `predict.project_season`, where they could name the stage that broke the
   invariant rather than only the row.

None of these are blocking for the work in §1 — all of it is complete and
verified independently of `src/projection/`.

---

## 6. What was preserved, deliberately

- **`src/depth_chart/sleeper_status.py` is untouched and working.** Sleeper's
  injury/status fields are a **news feed**, categorically different from a
  fitting target, and legitimately move projections via `apply_status_overrides`
  and the live depth chart. Verified post-change that `sleeper_status`,
  `refresh` and `events` all still import and that every symbol they take from
  `sleeper_compare` survived the deletion.
- **`sleeper_compare.py`'s comparison logic is unchanged.** Only vocabulary and
  framing moved. The join, the strata, the snapshotting all behave identically.
- **`output/sleeper_snapshots/` and `output/sleeper_comparison_2026.csv` were
  not deleted.** They are records.
- **The named watchlist mechanism was kept.** It has caught real silent-drop
  bugs twice; the audit is right that "a named-player check does not need a
  consensus to be useful". It now doesn't have one.
- **The divergence report was kept and made louder** (sorted largest-first).
  The goal was to stop Sleeper being an objective, not to stop looking at it.

---

## 7. Verification

| Check | Result |
|---|---|
| Test suite before | **127 passed** (stated baseline, matched on first run) |
| Test suite after | **133 passed** — 127 preserved + 6 added. Zero failures, zero skips, zero tests weakened or removed. |
| `python -m src.comparison.spot_check --season 2026` | Exit **0**. No incoherence; 15/15 watched players present; prior-season actuals resolved for all 15. |
| Import integrity | `sleeper_compare`, `spot_check`, `sleeper_status`, `refresh`, `events` all import cleanly; the three deleted names are absent; all six symbols the depth-chart modules depend on are present. |
| Git | Nothing committed, staged, or reverted. No `checkout`/`reset`/`stash`/`clean` run. All changes are in the working tree for review. |

### Verified vs inferred

**Verified by execution or exhaustive search:**
- Zero callers of the deleted function/constants (repo-wide, including dynamic
  dispatch and every inbound importer of the module).
- The coherence thresholds, calibrated against the actual 778-row board.
- That no test asserts agreement with Sleeper (read every Sleeper-touching test
  in full).
- That `NAMED_RUSH_COVERAGE` reaches `fantasy_evaluation.py` — traced
  `contracts` → `team_reconcile._named_supply_target` →
  `normalize_team_rushing_volume` → `compose_board:262` →
  `fantasy_evaluation:452`.
- That the vacancy alphas do **not** reach it — traced to `veterans.py`, which
  the harness does not call, corroborated by the harness's own
  `coverage_limits` text.
- Test counts and spot_check exit code, both by running them.

**Inferred, not verified:**
- The *outcome* of the F4 re-decision. Nothing here predicts whether
  `NAMED_RUSH_COVERAGE` survives its ablation — only that the ablation is now
  possible.
- That F5/F6/F10's underlying class-A fits are sound. I took the audit's
  classification of the commit-body evidence at face value and did not re-run
  those fits.
- The claim that the stale `sleeper_comparison_2026.csv` is a cached artifact
  of an earlier run rather than a bug. I confirmed the numbers differ from the
  current board; I did not trace which pipeline run produced it.
- That `roster_moves` could be moved inside the harness without disturbing the
  Week-1-roster design decision. That tension is real and I have not resolved
  it — hand-off item 2 is a problem statement, not a plan.
