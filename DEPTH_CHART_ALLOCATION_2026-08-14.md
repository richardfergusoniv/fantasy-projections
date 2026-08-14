# Depth-chart allocation remediation (2026-08-14)

The curated depth chart gated *eligibility* — who is real, who is off-chart —
but never allocated *opportunity* within a team. Three symptoms traced to
that one gap, and the final reconcilers amplified whichever modeled players
happened to remain.

Five phases, one commit each. Every constant below is measured; the two that
did not survive measurement ship disabled and say so.

## 1. Rushing allocated to measured roster coverage (`10303b5`)

`normalize_team_rushing_volume` pushed a team's **entire** carry anchor into
whichever players were modeled. Josh Jacobs was inflated 1.72x onto the exact
25.0 carries/game ceiling because Green Bay's charted committee back had no
projection row to absorb his share; Ashton Jeanty was pinned the same way.

Removing the fill entirely was the opposite error, not the fix — named
players then covered only 74.8% of league rushing and every lead back fell
under consensus (RB season MAE 15.6 → 19.0).

So the fill level was measured. Across every 2017–2025 transition, the share
of a team's season-N carries taken by players active in season N-1 — exactly
the universe the veteran path can project — averages **0.814**, stable season
to season (0.776–0.869, no trend). `NAMED_RUSH_COVERAGE` is applied
asymmetrically because the evidence is: a floor to fill up to, never a
ceiling to cut a healthy room down to, since team p90 reaches 0.99.

Result: no player pinned to a ceiling, Jacobs 338.5 → 275.5 season carries.

## 2. Replacement-level rows for charted players nothing projected (`fca3525`)

A curated player with no source-season production has no veteran feature row
and is not in the rookie class, so nothing projected him — and his share was
never held open, so the reconcilers gave it away. `MarShawn Lloyd` is the
case that surfaced it.

`fit_replacement_level_baselines` groups historical outcomes by position and
preseason depth band, the same way the rookie baselines and Gate B ladder are
fit. Rows are marked `source='replacement_level'`, flagged low-confidence, and
printed by name.

**QB is excluded, and testing showed why.** Quarterback appearances are
sequential and every room is allocated exactly 17 volume-games, so a QB row is
not an extra candidate — it is a claim against the starter. Filling
Cleveland's curated QB2 from a band mean gave Deshaun Watson 5.9 games and cut
Shedeur Sanders from 13.6 to 8.0. Missing charted QBs are surfaced for a human
instead.

Green Bay: Jacobs 275.5 → 211.8, Lloyd holding 88.1. Carolina: Hubbard
218.8 → 144.8, Brooks holding 126.2.

## 3. Rookie vacancy netted against existing claims (`4803957`)

The rookie path was the only claimant on a team's vacancy that never netted
itself. Arrivals net out competitors and incumbents net out arrivals, but a
rookie got the whole gross vacancy — so the same opening was spent twice.
Philadelphia lost A.J. Brown, credited DeVonta Smith as an incumbent
absorbing part of it, and scaled Makai Lemon by all of it anyway: 118.9
targets against Smith's 97.9.

The arithmetic already existed and simply was not being read:
`residual = (gross − arrivals) × (1 − INCUMBENT_VACANCY_ALPHA)`. It equals 1.0
exactly when the rookie really is the only claimant, and carries sit at 1.0
today purely because the carry alpha ships disabled — the correct coupling, so
enabling it cannot silently reintroduce the double-count.

Only the upward half is netted: a below-average opening is a true fact about a
rookie's situation and no other player's claim makes it less true.

Lemon 118.9 → 79.3, behind Smith as charted.

## 4. Usage-share prior — curated live, fitted disabled (`482f626`)

The chart had no quantitative signal. `depth_rank` is formation order (the
file says so in its own notes), and Gate B gives WR1 and WR2 the same 1.00
multiplier, having been fit to calibrate a rate rather than rank a room.

**The fitted rank prior tested beautifully and then lost where it counted.**
Leave-one-season-out over 2017–2025, w=0.4 cut share MAE 9.3% (targets) and
8.8% (carries), all 9 folds improving. But that scores against a
carried-forward share — the naive baseline — and this pipeline does not ship
naive. Re-scored against the actual models on the leakage-safe 2025
evaluation, the same blend is a straight loss: at w=0.25, RB points MAE
+1.25%, WR +0.73%, mean VORP MAE +1.5%, one fewer tier hit. The models already
read depth and usage history; a rank prior mostly re-tells them what they
know. `USAGE_SHARE_BLEND_W = 0.0`.

What beats the models is a human who has looked at a specific room, so
`usage_share_prior` / `usage_share_reviewed` were added to the chart and a
**reviewed** value carries weight 0.5. All 288 rows are populated with their
fitted default so the numbers are visible and editable; all 288 ship
unreviewed, because an unreviewed default is a starting point for research,
not a claim.

Dallas — where the curated chart lists Pickens above Lamb — is fixed by
reviewing two rows, not by a league-wide rule that costs accuracy.

## 5. Board-level tripwires (`d33f831`)

The existing tripwires watch one player or one stage at a time, which is why
none of them caught anything here: every bug was the reconcilers agreeing with
each other and being wrong together. Four checks on the finished board,
stderr-only, never changing a number: `CAPPED`, `MISSING`, `RB SHARE`,
`NEWCOMER`.

The 2026 board fires 16. `CAPPED` and `RB SHARE` are both silent — the rushing
fix reporting itself. The `NEWCOMER` list is worth a human pass: A.J. Brown
out-projecting a curated New England starter says the *chart* needs updating,
and Wan'Dale Robinson over Calvin Ridley is a case already known to be an
overshoot.

## Investigated and found not to be a bug

Phase 1 left a ~23% rushing residual even on teams with complete charted
coverage, which looked like systematic rate under-prediction. It is not.
Every component checks out: the team anchor is right (456.8 vs 456.0 actual
2025), RB carries per game played is right (8.18 vs 8.23; top-32 14.74 vs
14.68), and projected games for top-32 RBs is if anything *generous* (13.68
vs 12.26).

The first pass got this wrong by comparing against RBs ranked
**retrospectively** by outcome, which showed bell-cows playing 15.5 games —
survivorship. Ranked by prior-season usage, the honest figure is 12.26.

The decisive test was Sleeper itself: **Sleeper allocates 96.8% of every
team's carries to named players; we allocate 83.8%**, and their `gp` field is
literally 18.00 for every player. Sleeper projects "what this player does if
he plays"; we project expected value including injury risk. The persistent
negative bias against Sleeper — at all four positions — is a framing
difference, not a calibration error. `NAMED_RUSH_COVERAGE = 0.814` is
validated as correct rather than a patch.

## Net effect

| | before | after |
|---|---|---|
| Sleeper correlation | 0.947 | **0.952** |
| WR correlation | 0.934 | **0.939** |
| RB correlation | 0.954 | **0.955** |
| players pinned to a capacity ceiling | 2 | **0** |
| curated RB/WR/TE with no projection row | 2 | **0** |
| tests | 63 | **81** |

2025 leakage-safe evaluation is unchanged at every position, as intended —
Phases 4 and 5 are verified no-ops on the board (pred_pg identical to twelve
decimal places), and Phases 1–3 are preseason-allocation changes the
evaluation's own reconcile path already reflects.

## Still open

- **The RB lead-back level bias** noted in project memory still blocks
  `INCUMBENT_VACANCY_ALPHA['carry']`. Phase 3 is now coupled to it correctly,
  so enabling it is a one-constant change.
- **No curated row is reviewed**, so Phase 4's mechanism is inert until the
  first room is researched. Dallas is the obvious first candidate.
- **Three curated QBs have no projection row** by design (Watson, Bennett,
  DeVito). The `MISSING` tripwire reports them every run.
