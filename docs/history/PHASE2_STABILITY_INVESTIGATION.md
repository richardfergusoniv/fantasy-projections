# Phase 2 Stability Investigation: Why is year-over-year correlation only 0.144?

This is a diagnostic follow-up to `PHASE2_REPORT.md`'s honest but un-root-caused finding
that year-over-year correlation of OL coefficients averages 0.144 (range 0.025-0.218)
across the pass-protection and run-blocking sub-models. Four hypotheses were tested against
the actual 2021-2025 data in `data/projections.db`. Diagnostics originally lived in
`scratch_stability_diag.py` (deleted; findings below are retained).

## 1. Split-half reliability (the decisive check)

For 2023, each sub-model's plays were split by **game** (not by play, so a game's context
never leaks across both halves) into two random halves 5 times, RidgeCV refit
independently on each half, and coefficients correlated between the two halves for players
common to both.

| Sub-model | Split-half correlation (5 runs) | Year-over-year correlation (2023-adjacent pairs) |
|---|---|---|
| pass_protection | 0.397, 0.240, 0.376, 0.259, 0.380 -> **mean 0.330** | 0.203 (2022->23), 0.205 (23->24) |
| run_blocking | 0.324, 0.328, 0.301, 0.141, 0.360 -> **mean 0.291** | 0.202 (2022->23), 0.116 (23->24) |
| **Overall** | **0.310** | **0.144** (all 8 pairs, both sub-models) |

**Finding: split-half reliability (~0.31) is roughly 2x the year-over-year correlation
(~0.144), but it is itself still low in absolute terms (well under the 0.3 threshold the
existing report used as a "some real signal" cutoff, right at the boundary).** This is the
key result and it points to two things happening simultaneously, not one:

- A large share of the year-over-year instability is **pure estimation noise** that exists
  even within a single, unchanging season — two random halves of the *same* players in the
  *same* season only agree at r≈0.31. Since nothing about the players' true ability changed
  between the two halves (they're the same games, same season, same roster), this ceiling
  represents how much of *any* two independent samples' agreement is attributable to
  non-noise signal. Year-over-year agreement (0.144) is meaningfully lower than even this
  noisy same-season ceiling, meaning there IS some additional real degradation from year to
  year (real personnel/scheme/health change) beyond pure sampling noise, but it's a
  relatively small increment on top of an already noisy base.
- Because split-half itself sits at ~0.31, the 0.144 figure is **mostly noise, with a
  smaller real year-over-year signal layered on top** — not "the model is measuring real
  traits well and players genuinely bounce around year to year." If the true individual
  contribution to pressure/yards-over-expected were being cleanly estimated, split-half
  reliability would be much higher (>0.6-0.7 is the typical bar for a stable individual
  trait estimate at this play-level sample size).

## 2. Sample size per lineman

Play-count-per-player-per-season in the 2023 design matrix (314 pass-pro linemen, 317
run-block linemen):

- Pass-pro: median 310 plays/season, IQR 102-547, min 1, max 795.
- Run-block: median 209 plays/season, IQR 68-368, min 1, max 553.

Restricting year-over-year correlation to only linemen with >=300 plays in **both**
adjacent seasons materially improves correlation, and >=500 improves it further for
pass-pro:

| Threshold | Pass-pro avg YoY corr (n pairs) | Run-block avg YoY corr (n pairs) |
|---|---|---|
| >=0 (full set) | 0.163 (989 player-seasons) | 0.126 (995 player-seasons) |
| >=300 plays | 0.218 (423 player-seasons) | 0.254 (275 player-seasons) |
| >=500 plays | 0.263 (207 player-seasons) | not enough plays/season to test robustly at run-block volumes |

**Finding: stability does improve for high-snap linemen (roughly 1.5-2x higher
correlation), confirming sample-size-driven estimation noise is a real contributor** — a
depth lineman on 100-150 plays is simply too small a sample for ridge to pin down an
individual coefficient precisely, and the full-set 0.144 average is dragged down by
including these low-snap players. But even the >=500-play subset only reaches ~0.26-0.27,
still below the split-half ceiling of ~0.31 for the full population and nowhere near a
"stable trait" correlation — so sample size alone does not fully explain the low number.

## 3. Multicollinearity / identifiability for stable starting fives

2023 pass-protection design matrix: 314 lineman indicator columns, only 266 are
numerically independent (48 near-zero singular values, condition number ~6.75e27,
i.e. numerically singular past floating-point precision even before ridge regularization
is applied). RidgeCV's L2 penalty makes the fit well-posed despite this, but a
near-singular indicator matrix means the *unpenalized* likelihood cannot distinguish many
of these columns from each other, and the ridge solution's split of credit among
collinear columns is sensitive to arbitrary details of the penalty and data (exactly what
would produce instability across resamples/splits/years even if the training procedure is
"correct").

The root of the singularity: only 2 of 32 teams (BUF, CIN) had the same 5 linemen on
>=90% of pass snaps in 2023. Checked BUF directly — their most common 5-man lineup
appeared on 629 of 639 team plays, and for those 629 plays the five players' indicator
columns are **row-for-row identical** (every play has all five of them at 1, everyone else
at 0). This confirms the mechanism directly: within that block of plays, ridge has zero
statistical information to attribute performance to any one of the five over another —
whatever the model assigns to one of them individually is an arbitrary (penalty- and
noise-dependent) split of what is really a single team/unit-level effect for those plays.

**Finding: for teams with low lineup churn (the majority — only 2/32 teams hit the 90%
threshold; most teams used 8-23 distinct lineups over the season due to injuries and
rotation), individual-lineman attribution is not fully statistically identified from
pure ridge regression on indicator columns.** Teams with more lineup churn (NYJ: 21
distinct lineups, MIA: 20, NO: 15, etc.) actually provide *more* identifying variation
because injuries/rotation break up the otherwise-collinear block and let the model
separate individuals — ironically, the teams whose coefficients should be most trustworthy
are the ones with messier, less stable lineups, not the ones with a clean stable starting
five.

## 4. Alpha sensitivity

RidgeCV chose alpha=1000.0 for both 2023 sub-models (consistent with the range 562-1778
logged across all seasons in `PHASE2_REPORT.md`).

- Coefficients at 10x lower alpha (100) correlate 0.83-0.84 with the CV-chosen coefficients;
  10x higher alpha (10000) correlates 0.94-0.95. Coefficient spread (std) shrinks
  substantially as alpha increases (expected — that's what ridge regularization does).
- Split-half **stability** (the metric that matters for Phase 4, not just similarity to the
  CV pick) monotonically *improves* with higher alpha: pass-pro split-half correlation goes
  from 0.160 (alpha=100) to 0.333 (alpha=1000, RidgeCV's pick) to 0.360 (alpha=10000) to
  0.364 (alpha=100000). Run-blocking shows the same pattern: 0.194 -> 0.330 -> 0.358 -> 0.355
  (roughly flattening out by 100x).

**Finding: RidgeCV's cross-validated alpha is tuned for predictive fit on held-out plays,
not for coefficient stability, and the two objectives diverge here** — a materially higher
alpha (10-100x) than RidgeCV selects would produce more stable, more shrunk-toward-zero
individual coefficients, at some cost to the model's ability to explain in-season predictive
variance. The gains from over-shrinking flatten out fast (10000->100000 barely moves the
needle), so there's a real but bounded stability improvement available by fixing alpha
higher than CV's predictive-optimal choice — this alone would not fix the underlying
identifiability problem in #3, but it's a free, low-risk lever.

## Overall diagnosis

0.144 is **not** simply "real season-to-season volatility in lineman performance." The
evidence points to three compounding, mostly-artifact causes:

1. **Estimation noise dominates.** Split-half reliability on the *same* season (~0.31) is
   already low and roughly 2x the year-over-year number — most of the instability exists
   even with zero true change in the underlying player or season.
2. **Sample size matters but doesn't rescue it.** High-snap-count linemen show
   meaningfully better (but still not "stable trait"-grade) year-over-year agreement.
3. **Structural identifiability limit.** For the many teams with low lineup churn (30/32
   teams in 2023 didn't hit even a 90% same-five threshold, and several had a near-fixed
   starting five for long stretches), the design matrix cannot statistically separate
   individual linemen from a shared unit effect — this is a hard ceiling on what any
   per-play, per-season indicator-column ridge regression can recover, independent of alpha
   or sample size.
4. Alpha is a secondary, fixable contributor — RidgeCV is optimized for the wrong objective
   (predictive fit, not coefficient stability) and a higher fixed alpha buys some real
   stability, but it's a minor lever relative to #1 and #3.

## Recommendation for Phase 4

Given all four diagnostics point the same direction, the plain recommendation is:

**Do not treat per-season per-player coefficients as individual trait scores to be
averaged post-hoc. Move to a pooled multi-season regression with player and season fixed
effects (or a player-level random-effects/partial-pooling model) instead of five
independent per-season fits.** Concretely:

- Fit one pooled model per sub-model across all 2021-2025 plays, with lineman indicator
  columns as before but adding season dummies/fixed effects (and ideally team-season or
  season-level controls already used) so the same player's plays across multiple seasons
  jointly inform their coefficient, rather than being estimated from a single season's
  ~300 plays in isolation. This directly addresses cause #1 (sample-size-driven noise) by
  pooling more plays per player, and gives a principled way to let true year-over-year
  change show up (via player x season interaction terms or shrinkage toward a player-level
  mean) rather than manually averaging noisy independent estimates after the fact.
- Raise ridge alpha above RidgeCV's predictive-fit optimum for the final attribution
  model (roughly 10x higher, based on where the split-half stability curve flattens out) —
  cheap, already-validated stability gain, do this regardless of the pooling decision.
- Accept and document the identifiability ceiling from #3: **for teams/seasons with a
  low-churn stable starting five, treat the coefficients as more reliable at the
  unit (5-man line) level than the individual level.** A useful concrete signal to carry
  into Phase 4: compute each team-season's "lineup churn" (number of distinct 5-man
  combinations, or top-lineup snap share, as computed here) and use it as a confidence flag
  — individual coefficients from high-churn team-seasons are more statistically
  identified and should be weighted/trusted more than those from low-churn team-seasons,
  where credit-splitting among a fixed five is largely arbitrary no matter how the model is
  tuned.
- If a full pooled refit is out of scope for now, a trailing 2-3 season average, weighted
  by each season's play count, is a reasonable stopgap that should recover part of the
  sample-size gains from #2 — but it does not address the identifiability ceiling in #3,
  so should be treated as a temporary mitigation, not a fix.
