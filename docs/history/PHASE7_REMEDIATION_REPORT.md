# Phase 7 - Projection Integrity Remediation

This phase implements the 2026 projection-audit fixes, retrains the models,
regenerates production artifacts, and separates enforced accounting identities
from independent diagnostics.

## Corrected

- Historical OL features use exact-season coefficients rather than a fit pooled
  across future seasons.
- Historical rookie cohorts are defined from preseason draft/roster records and
  retain zero-game rookies; production and backtest no longer use realized
  target-season participation to decide who is a rookie.
- Drafted-rookie placeholder IDs are crosswalked through PFR IDs before roster
  and depth joins. Any remaining placeholder now emits a runtime warning and an
  output flag.
- Drafted players are removed from the UDFA bucket before baseline fitting. The
  current historical inputs contain 107 overlapping player-seasons before this
  deduplication and zero afterward.
- Rookie availability uses the schema-harmonized, truncated preseason depth
  rank. A depth-band cell must contain at least five players; smaller cells fall
  back to the position/draft-bucket mean and expose the cell size and fallback
  flag in output.
- The veteran depth-rate ladder is not applied to rookies. The dedicated rookie
  test found it neutral-to-harmful, so depth affects rookie availability only.
- Upward rookie vacancy scaling requires a curated starter/committee role, not
  mere chart membership.
- `games_played` means offensive appearances derived from snaps;
  `opportunity_games` separately records games with a pass/carry/target.
- The snap augmentation falls back only for the supported missing-schema case.
  SQL, crosswalk, merge, and augmentation errors otherwise fail loudly.
- Target, carry, red-zone, air-yard, monopoly, and receiving-yard share
  denominators all use offensive-appearance weeks. Zero denominators become
  missing rather than infinity.
- The production and backtest receiving-share guard both use explicit
  participation weights. The held-out backtest now has zero capped rows, rather
  than capping 70 rows on a denominator production did not use.
- A team model forecasts official QB attempts, aggregated at week/team grain so
  sacks are excluded and traded-QB attempts stay with the correct club.
- The independently forecast QB appearances are preserved unchanged in
  `projected_games_raw`. A separate, explicitly audited two-sided room
  allocation reconciles every resolved team to exactly 17 mutually exclusive
  `projected_volume_games`; its direction and player-level scale ship beside
  the raw estimate. Attempt allocation then operates on those 17 games and
  fails loudly if a resolved room cannot reach its anchor below 42 attempts/game.
- Team receiving yards and QB passing yards use one shared team-yardage anchor.
  The pre-normalization ratio/flag remains visible as the diagnostic; the
  post-normalization ratio/flag is documented as an accounting assertion.
- Canonical `pred_season` and fantasy season totals use
  `projected_volume_games` (falling back to `projected_games` when unavailable).
  CLI output displays both exposure columns beside the season total.
- Negative-scoring stats use the correct opposite interval endpoint. Raw
  negative fantasy lower bounds remain auditable while the shipped lower bound
  is floored at zero.
- Final output enforces completions <= attempts and receptions <= targets for
  point and interval endpoints and records every adjustment.
- Sleeper is evaluation-only. Comparisons use raw season totals, and the invalid
  feed-wide `gp=18` bookkeeping value is not presented as a conditional rate.
- Rolling-origin folds, target-stat lag features, exact-season OL isolation,
  active LightGBM row subsampling, and bare output-path handling are covered by
  the rebuilt pipeline and regression tests.
- **Team-model grain collision.** `TEAM_MODEL_FEATURES` ended in `naive_pred`,
  a name the *player* pair-builder already uses for its own carry-forward
  baseline. `backtest.py` and `corrections.py` scored the team-grain model on
  player-grain frames; that raised nothing and silently supplied a player's
  prior receiving rate (~30 yd/g) where the model expected a team's prior
  passing volume (~230 yd/g). Held-out team totals came out ~40% low (mean
  142.9 against an actual 225.6; MAE 82.7 against 22.3 with the correct input),
  and because the reframed path composes `share x team_total`, every held-out
  receiving prediction inherited the bias. The feature is now
  `team_naive_pred` so the same mistake raises `KeyError`, and player rows are
  scored through `transitions.team_model_inputs`. A separate production defect
  also existed: the live anchor selected an arbitrary first player after team
  reassignment, so an arrival could contribute his old team's lag. Production
  now scores a canonical, shuffle-invariant one-row-per-source-team frame and
  fails loudly on missing or conflicting team inputs. Downstream interval
  residuals and the elite-shrinkage correction were refit after both fixes.

## Verification

The completed sequence was:

1. `python -m unittest discover -s tests -v`
2. `python -m src.projection.train`
3. `python -m src.projection.backtest`
4. `python -m src.projection.predict --season 2026`
5. `python -m src.projection.fantasy_points --season 2026`
6. `python -m src.comparison.sleeper_compare --season 2026`

Current checks:

- 57 focused tests pass under the complete pytest suite.
- 3,969 projection rows and 768 player-position fantasy rows ship.
- Zero duplicate `(player_id, position, stat)` keys, negative point
  predictions, completion/attempt violations, or reception/target violations.
- 26 stat rows were reconciled and explicitly flagged.
- Projected team attempts have mean 573.7, minimum 533.3, and maximum 615.6.
  All 32 named QB rooms sum to exactly 17 allocated games and named attempt
  rates are capped at 42.0/game. The four formerly underfilled rooms (CLE,
  DAL, MIA, NE) reconcile upward while retaining their lower raw availability
  totals; Cleveland now allocates its full 575-attempt anchor with no residual.
- Passing/receiving yardage identity error is below `1e-12` for every team.
  Eight teams trigger the independent pre-normalization diagnostic; zero fail
  the enforced post-normalization invariant.
- Fantasy season-total formula error is below `9e-14` on every row. No negative
  lower bound ships; 57 raw negative lower envelopes were floored and flagged.
- The target rookie class contains all 80 drafted players and 147 UDFAs with
  zero duplicate IDs. Three players use the minimum-cell fallback.
- Kendrick Law resolves to canonical ID `00-0041446`; all seven formerly
  omitted drafted players now ship. Only Seydou Traore remains explicitly
  unresolved and warned.
- Historical exposure-weighted receiving-yard share has a maximum team-season
  sum of 1.140 and no 1.2 cap breaches.
- Sleeper matches 752/768 players (98%). Season-total correlation is 0.949 and
  mean absolute season-total delta is 14.67 half-PPR points. Conditional-rate
  comparison is correctly reported unavailable.

Rolling-origin results after the grain fix. The model beats carry-forward in all
three folds for QB attempts/completions/passing yards/interceptions, WR
targets/receptions/TDs, both team passing models, and - now - WR receiving
yards (10.88/11.81/9.39 against 11.10/13.04/11.12). TE and RB receiving yards
win two folds of three and win pooled (TE 7.30 against 7.51, RB 5.35 against
5.61).

The earlier finding that WR and TE share models lost to carry-forward "across
all three folds" was an artifact of the grain collision above, not a
model-selection result: the composed predictions being scored were built on
team totals that were ~40% low. On the held-out 2024->2025 pair the reframed
stats move from losing to winning - WR 11.35 -> 9.39 against a naive 11.12,
TE 7.91 -> 6.69 against 7.23, RB 5.15 -> 5.06 against 5.40. Every veteran stat
in the table now beats carry-forward except QB rushing yards.

Interval residuals were re-fit on the corrected composition. The three reframed
stats had shipped at 4.5x-6.1x one-sidedness (WR -4.13/+25.34) while their
non-reframed siblings sat near 1.0x - the signature of systematically low
predictions, visible in `models/interval_residuals.csv` for two review rounds
with nothing reading it. They are now 1.0x/1.3x/1.0x, and the non-reframed
rows are bit-identical. In shipped terms, the median receiving upside/downside
ratio moves 2.46 -> 1.65; Puka Nacua's season interval moves from
[224.2, 326.6] around a 233.4 point estimate to [187.4, 286.7]. The
elite-shrinkage beta, fit on the same composition, is now 0.3577.

## Structural follow-ups

- Route historical folds through a full `project_season_as_of` path with
  historical roster snapshots, rookie competition, and curated-equivalent role
  inputs. The current backtest shares the corrected weighted guard but still
  cannot reproduce every live roster decision.
- Watch the elite-shrinkage correction. Refitting on the corrected composition
  moved its season-consistency to 2.1 against a `MIN_SEASON_CONSISTENCY` gate of
  2.0. It still ships, but it is one fold from failing its own evidence test and
  should not be treated as settled.
- The team-total model is the highest-leverage remaining target for receiving
  accuracy. The composition is multiplicative, so its error lands on every
  receiver of a team at once and in the same direction, where per-player error
  does not. It is fit on ~32 rows per season. Damping it toward the team's own
  prior is the cheapest experiment worth running.
- Replace componentwise fantasy intervals with correlated stat and availability
  simulation. The current strictly-forward marginal coverage is 0.820 against
  a nominal 0.800 target, but it is not a joint fantasy-score interval.
- Add fantasy-rank and replacement-value evaluation beyond stat/season-total
  MAE.
- Build explicit dated 2026 coordinator, offensive-line, and schedule context
  instead of relying primarily on 2025 team context.
