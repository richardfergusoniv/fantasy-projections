# Shrinking the consensus spread — 2026-08-23

**Goal.** The projections are meant to expose mispricing in ADP. They cannot do
that while they are themselves mispricing players, and the spread against ECR
and ADP was far wider than a working board should produce. This pass closes that
gap by fixing the model, with no per-player overrides and **no blending toward
consensus** — model-only fixes were an explicit choice, so that whatever spread
remains is signal rather than something manufactured by construction.

**Headline.** ECR rank correlation 0.695 → 0.859, ADP 0.559 → 0.766, with **zero
cost to held-out accuracy**. Everything that shipped is in this repo. A separate
attack on the projection engine's target-share accounting is documented at the
bottom: it was diagnosed in detail, built, measured, and **reverted**, because it
cost accuracy and bought nothing on the actual goal.

---

## How the spread is measured

`scripts/consensus_spread.py`, against a frozen snapshot in
`data/consensus/consensus_2026.json` (FantasyPros PPR ECR via
nflverse/DynastyProcess; Fantasy Football Calculator half-PPR 12-team ADP, 2810
drafts). `/data/` is otherwise gitignored; the snapshot is carved out because no
measurement here is reproducible without it.

Three measurement rules, all load-bearing:

1. **Both sides are re-ranked inside the matched set.** Our board and the market
   rank different-size universes. Comparing raw ranks across them manufactures a
   constant negative bias at every position — the first reading of −46 to −60 for
   QB/RB/WR/TE alike was entirely this artifact, not a model signal.
2. **Before/after is scored on the common player set** (`--restrict-to-board`).
   The old board carried 356 players and the new one 908; without this the old
   board flatters itself by simply not containing the hard players.
3. **`--max-market-rank`** restricts to the draftable range, because a metric
   averaged over players nobody drafts is dominated by noise.

---

## Result

Common matched set, identical consensus snapshot:

| | ECR rho | ECR mean abs d | ADP rho | ADP mean abs d |
|---|---|---|---|---|
| Before | 0.695 | 39.4 | 0.559 | 23.7 |
| Phase 1 — spread | 0.839 | 29.8 | 0.728 | 19.3 |
| Phase 2 — coherence + TE | **0.859** | **28.4** | **0.766** | **18.0** |

Position bias (ADP): QB −0.7 → −1.0, RB +7.0 → +4.7, WR −0.5 → +2.9,
TE −17.7 → −21.9.

Against the consensus-free reference — a board built from realized historical
curves under identical VORP rules — the mean overall-rank gap by position is now
WR +0.6, **TE −1.0** (from −18.0), RB −7.6. QB reads +32.0 and is left alone on
purpose; see Known-remaining.

Held-out accuracy is unchanged, because nothing that shipped touches the
projection engine — v2's `accounting.py` is byte-identical to where it started,
and its leave-one-season-out preseason backtest still passes its promotion gate
(weighted MAE 3.918, rank corr 0.493).

---

## What shipped

### 1. The board was stale, and that was most of it

`output/fantasy_points_2026.csv` had been built from an older v2 export that
carried no `projected_games`: all 356 players sat at exactly 17.0, and real
starters were present at 0.0 points (George Kittle among them). Re-syncing from
the current v2 output — 908 rows with real availability — moved ECR rho from
0.684 to 0.852 on its own.

This is worth dwelling on: the single largest source of "the model is mispricing
players" was that the board being looked at was not the model's current output.

### 2. VORP was floored at zero

`add_vorp_columns` clipped surplus at 0, tying **73% of the board** (261 of 356)
at a single value. Everything from roughly round 9 down was ordered by whatever
the sort tiebreak happened to be — not monotone in points, let alone in value.

VORP is now signed. It also ranks **season points rather than a per-game rate**:
a draft pick buys a season, and with availability restored a 9-game player is
worth about half of an identical 17-game one. Tier gaps in `tiers.py` were
rescaled by a 17-game season so tier granularity is unchanged by the change of
basis.

### 3. Replacement level ignored availability

A starting slot consumes more than one player across 17 weeks, because starters
miss games. Measured from the board's own projected games (~15 per position),
replacement ranks deepen by ~12%: QB 13→14, RB 29→33, WR 43→48, TE 14→15.
Derived from the model's own availability output, not fitted to ADP. Worth ECR
rho 0.805 → 0.830 and RB bias +9.3 → +3.2 at the time it was added.

Both the nominal and the used ranks are published in the board metadata, along
with the per-position availability factors.

### 4. The browser recomputed VORP with the old bug

`draft_assistant/js/app.js` re-derived VORP client-side whenever league size
changed, using `Math.max(0, ppg - baseline)` — the same clip, on per-game points.
So changing league size in the UI silently reintroduced the defect. Now matches
the server: season points, signed, availability-adjusted.

### 5. Per-team pass/catch totals now tie out

A team's receiving yards *are* its passing yards. League-wide the projection
balanced almost exactly, but per team the ratio ran sd 0.154 against a realized
0.060 — 25 of 32 teams off by more than 5%, Cleveland at 1.63, its receivers
1,232 yards ahead of its quarterbacks.

The cause is not the engine's allocation, which is exact: summed per team, v2's
per-game receiving yards equal its per-game passing yards to the last decimal on
all 32 teams. The identity breaks only when each player's rate is multiplied by
*his own* `projected_games`. Cleveland's starting quarterback is projected for
8.85 games and its backup for 15.0 at 2.3 yards per game, so the quarterback
room never covers a 17-game season while the receivers accumulate over ~15 games
each.

`reconcile_team_season_identities` (`from_v2.py`) restores the identity in the
season totals, splitting the correction geometrically between the two sides.
Three deliberate properties:

* **The target is the identity itself, not whatever the rates currently imply.**
  The first version reconciled toward each team's own per-game ratio, on the
  reasoning that the engine already enforced the identity there — which it did,
  exactly, at the time. Mid-session the engine gained a
  `season_availability_scale_*` on the receiving side, the per-game ratio drifted
  to 1.20, and preserving it faithfully preserved the drift. Reconciling to the
  identity is robust to an engine that changes underneath.
* **A pair may opt out of 1.0, but not out of arithmetic.** `targets` against
  `attempts` is not an identity, so it reconciles to the realized 0.965. It
  cannot reconcile to "whatever the engine produced", because on the current run
  that is **1.032** — more targets than pass attempts, which cannot happen.
* **The split is symmetric.** Rescaling receivers onto the quarterback outright
  was tried earlier in this project and backtested worse, because it propagates
  the quarterback model's error onto every receiver. A geometric split moves
  each side half as far.

All four pairs now hold on every team. The correction carries through to season
fantasy points by each player's own before/after ratio on the scored line
(`apply_identity_scale_to_points`) rather than by rescoring, which would discard
v2's fitted per-position points calibration. The factor ships as
`team_identity_scale` (mean 1.003, sd 0.058).

### 6. TE VORP no longer pushes tight ends up the board

Measured against the fitted historical curves with the board's own replacement
ranks, TE's replacement level was right (106.4 against 107.2) and so was its
tail — but **TE2 through TE8 carried roughly twice the surplus they should**
(ratios 1.95, 1.85, 2.12, 2.05, 2.33, 2.14). Eight tight ends sat between 174
and 245 points where history spreads them 143 to 182. WR and RB needed nothing:
their surplus ratios run 0.85 to 1.11 across the same probes.

This is a *shape* error at the top of one position's curve, and the two obvious
instruments are both wrong for it. Scaling points fails — it shrinks surplus and
deficit alike, so sub-replacement tight ends become *less* negative and rank
higher, which is backwards. Moving the replacement rank fails too: it is already
correct.

`POSITION_CURVE_WEIGHT` in `vorp.py` blends a position's surplus curve toward
the fitted historical shape at the same positional rank, both anchored at that
position's own replacement level. Enabled for TE only, at full weight. The
magnitudes being replaced are not information worth keeping — they come from an
elite-TE target-share floor firing on roughly seven teams a season — while the
**ordering, which tight end is best, is ours and is preserved** (both curves are
monotone in rank, so the blend cannot reorder). Ranks past the fitted depth keep
our own surplus.

Trey McBride moves from 5th overall to 12th, against a historical-implied 14th.
Curves are fitted by `scripts/fit_position_curves.py` into
`src/draft_assistant/position_curves.json` and published in the board metadata,
so the browser can apply the identical blend.

### 7. The client no longer re-derives a board it cannot reproduce

`applyLiveVorp` recomputed VORP on every load, and the client lacks inputs the
server has — it ignored the rookie ranking haircut, so the rendered board
disagreed with the published one about where rookies belong. At the published
league size the client now uses the server ranks verbatim and only recomputes
when the user actually changes league size; the recompute path gained both the
rookie haircut and the position-curve blend. Verified in the browser: rendered
order matches the published order exactly through the top of the board.

### 8. A calibration harness that does not lie to itself

`scripts/` gains `_history.py` (shared realized-history loader),
`consensus_spread.py`, `board_calibration.py`, `receiving_share_check.py`,
`rank_share_check.py`, `conservation_check.py`, `efficiency_check.py`,
`team_coherence_check.py`, `fit_position_curves.py`, `fit_position_rank_scale.py`.

---

## Investigated, built, measured, and reverted: the v2 target-share accounting

A real and well-evidenced calibration defect, whose obvious correction made the
model worse. Recorded in full because the diagnosis stands and the next attempt
should start from here rather than rediscover it.

### The defect is real

The survivorship-free test — league-wide totals and shares, which obey hard
identities (receiving yards = passing yards) and have no order-statistic bias —
shows every stat at a uniform ~0.84 of realized (the legitimate expected-value
and availability discount) **except** the receiving split: WR at 0.72–0.75, TE at
1.03–1.18. Against the recent window, TE's top-18 season points run **1.19x**
realized and **1.45x** relative to WR. The projected TE curve puts *eight* tight
ends above 200 points; historically only the league's TE1 clears 200.

Three mechanisms in `fantasy-projections-2/src/projections/pipeline/accounting.py`:

- **`_TARGET_DEPTH_WEIGHTS` is one shared table.** `depth_rank` ranks a player
  *within* his position, so a shared curve gives a team's TE1 and its WR1 the
  same weight of 1.0 — while RB/WR/TE are normalized against one pooled team
  target budget. Realized 2019-2024 share says a TE1 is worth 0.60 of a WR1 and
  an RB1 0.46. The fitting script (`scripts/fit_depth_share_weights.py`) cannot
  discover this: it pools positions and pins `weights[1] = 1.0` by construction.
- **The elite TE1 floor mints share.** It raises the TE1 in place with an
  explicit "do not require WR donors" — but every receiver is renormalized to one
  budget immediately afterwards, so the minted share comes out of the WRs anyway,
  silently, on every team with a qualifying TE.
- **`MAX_TARGET_SHARE["TE"] = 0.28` is TE's record, not its tail** (p95 is 0.229;
  0.287 is the single highest season ever observed). A cap set at the record does
  not constrain anything.

Two further findings about the gate, both independently useful:

- **`_anchor_share` returns `max(prior_season, last_5_games)`.** The maximum of
  two noisy windows estimates their upper envelope, not the level, so marginal
  players clear on whichever window broke their way.
- **The gate is p77, not p90.** `TE1_ELITE_PRIOR_SHARE = 0.22` reads as strict
  but is the 77th percentile of realized TE1 target share; eight of 32 teams
  clear it.

### Why it was reverted

The corrections did what they were supposed to — TE went from 1.45x to 1.12x
relative to WR, cross-position calibration spread 0.447 → 0.281 — and then the
leave-one-season-out preseason backtest said they made the model worse:

| | ECR rho | ADP rho | calib spread | backtest MAE | rank corr | promotion |
|---|---|---|---|---|---|---|
| Freshness + VORP only | **0.839** | 0.728 | 0.447 | **3.918** | **0.493** | **pass** |
| ＋ accounting fixes | 0.827 | **0.741** | **0.281** | 4.007 | 0.475 | **fail** |

The accounting work buys +0.013 ADP rank correlation and −0.012 ECR, i.e. a wash
on the goal, in exchange for 2.3% worse MAE, 0.018 worse rank correlation, and
dispersion falling from 0.713/0.719 (already marginal) to 0.667/0.664 against a
policy floor of 0.70 — flipping the project's own promotion gate from pass to
fail. Reverted; `accounting.py` is byte-identical to where it started, verified,
and the projection is deterministic to 1e-14 so that revert is exact.

The gate percentile was retried at three values (0.22 as-found, 0.2667, 0.2124)
and the accuracy regression is present at all three, so it is not the gate.

**The most likely reading** is that the TE inflation is compensating for
under-dispersion elsewhere in the model — removing one half of a compensating
pair made things worse. The dispersion numbers say exactly this: the corrected
version predicts a narrower spread than reality, and it was the inflation
supplying the width. A real fix has to find the missing dispersion first, which
is a larger piece of work than this pass took on.

**If someone picks this up:** bisect the bundle. The five changes were applied
together and only measured together, so it is entirely possible one of them (the
donor-funded floor, say, which is conservation-correct on its own terms) is free
or positive while another carries the whole cost. Each backtest run is ~15
minutes. The reverted diff is reconstructable from this document.

---

## Known-remaining, stated honestly

- **TE still reads about 22 spots above ADP, and that residual is now believed
  to be real.** The consensus-free reference puts our TE placement within one
  rank of where realized curves say it belongs (gap −18.0 → −1.0). What is left
  is the market drafting tight ends later than surplus-over-replacement says it
  should — the kind of disagreement this tool exists to surface, not a defect to
  tune away.
- **Published TE point projections still read high**, because the correction
  above lives in the ranking layer only. The underlying level defect is the
  reverted accounting work; the board no longer inherits it, the projections
  still do. Anyone reading `fantasy_points_2026.csv` directly should know that.
- **QB and RB read +13 and +23 against ECR but −1.0 and +4.7 against ADP.** ECR
  is a ranking opinion; ADP is the price. Where they disagree this sharply,
  weight ADP.
- **QB reads +32 against the historical-implied board** and is deliberately not
  corrected. A realized QB curve is the steepest in football precisely because
  which of 32 near-equal starters finishes QB1 is close to random, so no honest
  expected-value projection can match its shape. ADP agreeing with the board
  (bias −1.0) is the tiebreaker.
- **The QB curve is too flat through ranks 5–24** (1.36x the WR-relative scale).
  Some is real compression; much is the survivorship caveat at its strongest,
  since which of 32 near-equal starters finishes QB1 is close to random and no
  honest expected-value curve can be as steep as the realized one. ADP agrees
  with our QB placement, so this was not chased.
- **TE catch rate runs 1.08x realized** (WR 1.04, RB 1.03).
- **The projection universe is far deeper than the league** (385 WR / 209 RB /
  195 TE against realized populations of 219 / 125 / 120), and rank-6+ players
  hold ~13.6% of projected team targets against ~1.7% in reality. Lowering the
  tail floors barely moves it, because `DEPTH_PRIOR_STRENGTH = 0.55` blends
  toward identity — even a weight of zero only cuts a player to 45% of the
  model's raw prediction, so the depth prior *cannot* express the needed
  concentration. That blend is the lever, and it was not touched.

---

## Traps hit during this work

**Position shares are not stationary.** TE's share of the receiving pie ran 21.1%
(2021), 21.7% (2023), 22.5% (2024), **24.1% (2025)**; receiving yards moved 20.9%
to 23.9%. Benchmarking a 2026 projection against a 2019-2024 mean of 22.8%
understated the current level by more than a point and drove an over-correction
that only surfaced when the window was narrowed. The calibration scripts now
default to the recent window and say so.

**Percentiles must be computed on the quantity actually being thresholded.** p90
of the raw prior share is 0.2667; p90 of the blended anchor the gate actually
reads is 0.2124. Using the former admitted one team in 32 instead of three.

**Realized rank curves are survivorship-inflated.** The realized "WR12" is the
ex-post winner of a health race; a calibrated expected-value projection is
legitimately flatter and lower. Raw ratios read WR 0.80, RB 0.86, QB 1.14, TE
1.22 and invited a chase after WR "under-projection" that was mostly artifact.
Compare positions *to each other*, or use totals and shares.

**`weekly` drops 2025 twice over.** `season_type` is NULL for every 2025 row and
`position` is NULL for all 5,636 of them (a pbp-fallback ingest). Any query
filtering on `season_type = 'REG'` **or** on position silently loses the entire
most recent season. `src/projection/data_prep.load_weekly_usage` already handles
both; `scripts/_history.py` now applies the same two rules.

---

## Reproducing

```bash
python scripts/consensus_spread.py --restrict-to-board <old-board.json>
```

```bash
python scripts/board_calibration.py --curve
```

```bash
python scripts/receiving_share_check.py
```

```bash
python scripts/conservation_check.py
```

All read `FANTASY_PROJECTIONS_DB_PATH` (or `FANTASY_PROJECTIONS_DATA_DIR`).
