# Phase 6 — 2026 Team Reassignment, Curated Depth Chart, and Output Gating

Fixes the two bugs the project owner found by eyeballing `output/projections_2026.csv`:
Kirk Cousins showing on Atlanta (his 2025 team) instead of Las Vegas (his real
2026 team), and no mechanism to stop three Arizona QBs (Brissett/Minshew/Slovis)
from all getting full-confidence "starter" projections.

## 1. Team reassignment (Task 1)

**Source of truth**: `seasonal_rosters` for `season=2026`, joined by `player_id`.
`load_target_roster_map()` (`src/projection/predict.py`) prefers a player's
`status='ACT'` row when a player has more than one 2026 roster row (e.g. a
practice-squad stint); non-`ACT` statuses (`RES`, `RET`, `CUT`, `E14`) are
**kept, not dropped** — a `RES`/PUP player isn't necessarily out for a season
that hasn't started, and even a `RET`/`CUT` row is more informative surfaced
than silently vanished. The chosen team and raw status are both carried into
the output as `roster_status`, so a reader can judge a `RET` row differently
from an `ACT` one themselves rather than have that judgment made silently
upstream.

**What gets re-pointed for a team-changer** (`reassign_team_changers()`,
applied once per veteran-model call before the LightGBM predict step):

1. **Output team label** → the 2026 roster team (was: stale 2025 team).
2. **Team-context features** (`oc_tendency_profiles`, OL pass-pro/run-block
   score, `ol_confidence_low_churn`) → the **new** team's season-2025
   observed row, not the old team's. This is the same "last observed season
   stands in for the unplayed next one" judgment call `train.py`/
   `transitions.py` already document for the no-change case — just correctly
   re-pointed at the team the player is actually walking into for 2026,
   instead of the one they left.
3. **Player share features** (`carry_share`, `target_share`, `rz_carry_share`,
   `rz_target_share`, `snap_pct`) → see below, the hard part.

**Share-transfer method for team-changers** — the real modeling judgment call:
a traded/signed veteran's *old* team share number doesn't mean anything at a
new team with a different depth chart and different available volume, so it
is not carried forward unchanged (that would just be the same bug in a
subtler form). Instead, this reuses `rookies.py`'s `team_vacated_opportunity`
(how much of a team's carries/targets belonged to players not returning to
that team this season — already has a roster-based fallback for a season
with zero played games, which 2026 is) as the "how much room is actually open
here" signal, and treats the player's own established share **at their old
team** as a "quality tier" signal (how much volume this player is capable of
commanding, once given a role):

```
scale       = clip(new_team_vacated_share / league_avg_vacated_share, 0.3, 2.5)
new_share   = old_team_share * scale
```

`league_avg_vacated_share` (the mean vacated share across all 32 teams for
the target season) is the baseline, not the player's own old team's vacated
share — this mirrors `rookies.predict_rookies`, which scales a bucket mean
against the **historical bucket average** vacated share, not the specific
player's own prior situation. The 0.3–2.5 clip is the same band
`rookies.VACATED_CLIP` already uses, reused rather than inventing a new
number for the same kind of small-sample safety clamp. `carry_share`/
`rz_carry_share` are scaled by the carry-vacancy ratio; `target_share`/
`rz_target_share` by the target-vacancy ratio; `snap_pct` is scaled by
whichever of the two matches the position's primary opportunity type
(RB→carry, WR/TE→target, QB→left unscaled, since a starting QB's snap_pct is
~100% regardless of team and Task 3's depth-chart gating — not this share
model — is what actually separates a new team's QB1 from its QB3). All
scaled shares are clipped to a max of 1.0.

**Stated limitation**: this does not, and structurally cannot, capture
**scheme fit** — "the new team's offense throws far more to the slot than
the old team's did" is a real, unaddressed residual on top of normal
projection error for every one of the 108 team-changers in this output. It
also can't distinguish "this team has a big opening because their WR1 left"
from "this team has a big opening because their whole passing game
collapsed" — `vacated_share` only measures departed volume, not incoming
quality of the offense around the new player.

Implementation lives entirely in `predict.py` (`reassign_team_changers`,
called once from `project_veterans` before the model's `X` matrix is built)
— `features.py` itself is untouched, since its historical-training path must
keep using each season's own resolved team (that's the correct behavior for
fitting the models; only the *predict*-time team assignment for the
not-yet-played target season needed fixing).

**Rookie path confirmed untouched**: `identify_target_season_rookie_class`
already resolves team from `seasonal_rosters[target_season]` (falling back
to `draft_picks.team` only when a placeholder rookie id isn't in
`seasonal_rosters` at all) — verified by reading `rookies.py` and by the
2026 rookie rows in the rerun output; no changes made there.

## 2. Curated 2026 depth chart (Task 2) — `src/depth_chart/starters_2026.csv`

Built from `ourlads.com`'s per-team 2026 depth-chart pages (fetched
individually for all 32 teams, `2026-08-05`), cross-checked against
`seasonal_rosters`/`players` for `gsis_id` resolution. **262 rows, all 32
teams, 0 unresolved `gsis_id`** (name-normalization required a couple of
manual fixes — e.g. `players.display_name` "Deebo Samuel Sr." / "Stefon
Diggs" don't currently appear in `seasonal_rosters[2026]` at all despite
showing up on their new teams' ourlads pages; resolved instead via the
`players` master table by name, and flagged as a known roster-ingestion lag
below).

Coverage matches the spec exactly: QB1 for every team (QB2 only where a real
competitive/injury-contingency situation exists — 12 of 32 teams), RB1–2
with a `role` of `starter`/`committee`/`backup` (not forcing a bell-cow where
the backfield is genuinely split), WR1–3, TE1–2.

Confidence breakdown (262 rows):

| position | high | medium | low | total |
|---|---|---|---|---|
| QB | 20 | 12 | 6 | 38 |
| RB | 15 | 44 | 5 | 64 |
| WR | 24 | 51 | 21 | 96 |
| TE | 14 | 35 | 15 | 64 |
| **all** | **73** | **142** | **47** | **262** |

`confidence=low` rows are genuine, named uncertainty (e.g. Cleveland's
Sanders/Watson QB competition, Indianapolis's Jones/Richardson QB
competition, several true rookie WR3/TE1 slots with no track record) — every
low-confidence row's `notes` column states exactly what's unresolved, per the
project's `oc_assignments.csv` precedent. No slot was silently omitted for
being uncertain.

**Judgment call**: `ourlads` lists WR by field position (Left/Right/Slot),
not by expected target share. Where I had strong outside signal that this
misorders the real target-volume hierarchy (e.g. Baltimore's Zay Flowers,
Detroit's Amon-Ra St. Brown, Green Bay's Jayden Reed — all listed at "slot"
but are each team's clear top target), I reordered and said so in `notes`.
For the rest, positional order was kept as the rank-1/2/3 proxy — a real,
acknowledged imprecision for teams I didn't specifically re-rank.

**Known gap, stated not hidden**: `seasonal_rosters[2026]` does not yet
contain Deebo Samuel or Stefon Diggs at all (both very recent free-agent
signings per ourlads — SF and WAS respectively) despite `ourlads` showing
them on their new teams — the roster ingestion appears to lag the very
latest transactions by some days/weeks. Their depth-chart rows are included
(gsis_id resolved via `players`), but the Task-1 team-reassignment path
depends on `seasonal_rosters` for its "what team is this player on" signal —
if the roster table hasn't caught up for a specific player, `predict.py`'s
`reassign_team_changers` will silently keep that player on their **old**
team rather than the curated depth chart's team, since it doesn't consult
`starters_2026.csv` for team assignment (by design — that would make the
depth chart a second source of truth for team, which risks the two files
drifting apart silently). This is a real, live inconsistency for exactly
these two players in the current run — flagged here rather than
silently accepted.

## 3. Output gating (Task 3)

**Mechanism chosen**: keep every row (never silently drop a real player —
the project's hard rule, and the exact bug `PHASE5_REPORT.md` already found
once), but any veteran not matched in the curated depth chart for their
`(new team, position)` gets `pred_pg`/`pred_pg_low`/`pred_pg_high` multiplied
by `DEEP_BENCH_DISCOUNT = 0.15`, `low_confidence` forced `True`, and
`role='deep_bench'`. Full exclusion was considered and rejected: a
heavily-discounted, clearly-flagged row is auditable in a way a missing row
is not, and a "starters only" consumer can trivially filter on
`depth_chart_status` themselves. A new `depth_chart_status` column
distinguishes:
- `curated` — matched the table, carries real `depth_rank`/`role`.
- `deep_bench_discounted` — the team+position group IS covered by the table
  (I researched every one of the 32 teams' QB/RB/WR/TE groups to the
  specified depths) but this player fell outside the curated top-N —
  **confirmed** outside the relevant depth, not merely unresearched.
- `not_curated_no_table` — target season has no curated table at all (any
  season other than 2026); gating is a no-op there, not a claim about role.
- `rookie_path` — rookie-rule rows; not gated (already `low_confidence=True`
  by construction), but carries `depth_rank`/`role` informationally when the
  rookie happens to be in the curated table.

Because the 2026 curation is complete for all 32 teams at every specified
depth, there is currently no "couldn't research this in time" bucket to
distinguish from "confirmed buried" — that distinction is architecturally
present (`depth_chart_status`) and will matter the moment a future rerun's
curation is partial.

**Result**: 245 unique veteran players in `curated` status, 324 unique
veteran players discounted as `deep_bench_discounted`.

## 4. Bug found and fixed along the way (not part of Tasks 1–3, found during Task 4's required spot-checks)

Required random-player spot-checks (Task 4) turned up a real, silent gap:
**Ashton Jeanty — a real 2025 1st-round rookie with a full 17-game, 267-carry
season — was completely absent from the 2026 output**, both before and
independent of any Task 1–3 change. Root cause: `project_veterans` excluded
every player whose *own* rookie season equaled `source_season`
(`identify_rookie_seasons`/`identify_udfa_rookie_seasons([source_season])`),
on the stated reasoning "source_season IS their only season, so they have no
real trailing features." That reasoning is wrong — a rookie season is a
complete, real season of production, exactly the trailing data the veteran
model needs to project a player's **sophomore** season. The effect: every
player's second NFL season was silently dropped from every year's
projection, forever (same *class* of bug `PHASE5_REPORT.md` already found
and fixed once for long-tenured veterans, but its sibling for players
exactly one year removed from their rookie year — not caught by that
review's Allen/Chase/McCaffrey spot-checks since none of them were in their
second season at the time this was checked).

Fix: removed the exclusion entirely. It's safe with no replacement needed —
`target_season`'s real incoming rookie class structurally has no
`source_season` feature row at all (no NFL history before `target_season`),
so there's no risk of double-counting them via both paths. Verified:
unique veteran players in the output went from 476 (pre-fix, matching
`PHASE5_REPORT.md`'s number) to **569**; Ashton Jeanty now appears correctly
(LV, `curated`, `role=starter`, `depth_rank=1`). Total rows: 3812 (up from
3344 pre-fix, both numbers after this phase's Task 1–3 changes were already
applied — the two fixes are independent and both included in the final
rerun).

## 5. Rerun and required spot-checks

Ran the full pipeline per `PHASE5_REPORT.md`'s documented order.
`train.py`/`backtest.py` were **not rerun** — confirmed by reading both
files that neither imports or calls anything changed in this phase
(`project_veterans`, `reassign_team_changers`, `apply_depth_chart_gating`,
and the depth-chart file are all `predict.py`/Task-2-only additions); the
trained models and interval residuals are unaffected by feature-construction
or output-gating changes made downstream of them.

```
python -m src.projection.predict --season 2026
```
→ **3812 projection rows** (2776 `veteran_model`, 1036 `rookie_rule`), 569
unique veteran players, 220 unique rookie players.

**Kirk Cousins**: team=`LV` (was `ATL`), `team_changed=True`,
`roster_status=ACT`, `depth_chart_status=curated`, `role=starter`,
`depth_rank=1`. Fixed.

**Kyler Murray**: team=`MIN` (was `ARI`), `team_changed=True`,
`roster_status=ACT`, `depth_chart_status=curated`, `role=starter`,
`depth_rank=1`. Fixed.

**Arizona QB room** (`attempts` per-game shown):
| player | pred_pg | role | depth_chart_status |
|---|---|---|---|
| Jacoby Brissett | 27.73 | starter | curated |
| Gardner Minshew | 8.81 | backup | curated |
| Kedon Slovis | 1.97 (down from an undiscounted ~13) | deep_bench | deep_bench_discounted |

One clear primary (Brissett), one real competitive backup at meaningfully
lower volume (Minshew — reflecting the genuine camp battle, not equal
weighting), and the third QB heavily discounted and flagged rather than
standing shoulder-to-shoulder with the other two. Fixed.

**Two random no-change players** (per the project's standing rule to check
the common path isn't broken by an edge-case fix):
- **Josh Allen** (BUF): unchanged team, `team_changed=False`,
  `depth_chart_status=curated`, `role=starter`, `depth_rank=1`,
  `pred_pg`(attempts)=28.0 — in line with his established volume.
- **Christian McCaffrey** (SF): unchanged team, `team_changed=False`,
  `depth_chart_status=curated`, `role=starter`, `depth_rank=1`, full 7-stat
  RB row present and undiscounted.

Also re-confirmed **Ja'Marr Chase** and **Patrick Mahomes** unaffected
(unchanged team, `curated`/`starter`, normal undiscounted values) as an
extra check against the PHASE5 bug's exact former victims.

## 6. Other caveats / open items for the user

- **65 of 569 veteran players** (~11%) have no 2026 `seasonal_rosters` row
  at all (retired, out of the league, or a late signing the roster table
  hasn't caught up to — see the Deebo Samuel/Diggs note above). These fall
  back to keeping their old (2025) team via `reassign_team_changers`'s
  `no_info` branch, which is the best available default but is provably
  wrong for anyone who actually retired (a couple of names in this bucket,
  e.g. Philip Rivers, look like they may be stale/incorrect entries in the
  underlying `weekly`/`pbp` data itself — a pre-existing data-quality
  question outside this phase's scope, not something introduced here).
- The `DEEP_BENCH_DISCOUNT=0.15` multiplier is a stated, un-tuned judgment
  call (same spirit as `train.py`'s un-tuned LightGBM hyperparameters) — it
  wasn't backtested against any held-out season, since there's no historical
  curated depth chart to validate it against.
- The team-changer share-transfer method's scheme-fit limitation (see §1)
  is real and unaddressed for all 108 team-changers in this output.
- WR ranking order defaults to `ourlads`' Left/Right/Slot field-position
  order except where explicitly overridden in `notes` — a residual
  imprecision for any team where I didn't have strong outside signal to
  reorder (see §2).
