# Retire the interim simulation arm; ship generative percentiles

**Date:** 2026-08-26
**Verdict:** `ship_generative_percentiles`
**Scope:** `src/projection/inference/simulate.py` — `SIMULATION_MODE` is now `"full"`

## Why interim was retired

The interim arm drew `clip(pred + residual, 0)` per stat, independently.

Two defects followed from that, and they invert between grains:

- **The zero floor is asymmetric.** `E[max(0, X)] > E[X]`. 73.5% of stat rows
  have residual pools that can go negative and 28.9% of draws are clipped,
  worth **+8.33** fantasy points per player against an observed sim-mean-vs-v1
  gap of **+8.65**.
- **Residual right-skew** pushes each stat's *median* below its point estimate.

Summing eight stats removes most of the skew, so the total's median migrates
off the second bias onto the first: per-stat medians sum to **−3.93** against
v1 while the median of the sum is **+4.26**, a gap of +8.19 for 90.5% of
players. Scoring per-stat rates gives the opposite answer to scoring summed
fantasy points, and the shipped artifact is the latter.

Underneath both: per-stat residuals were drawn **independently**, while within
a player they correlate **+0.876** (passing yards/TDs), **+0.871** (receiving
yards/receptions), **+0.672** (rushing yards/TDs). Independence understates
the summed spread by **31%**.

## Measured, held out on 2025

Board built from history through 2024 only, scored against realised points
(`scripts/compare_simulation_modes.py`):

| mode | coverage | width | p50 MAE | p50 bias | rho |
|---|---|---|---|---|---|
| interim | 0.5050 | 62.23 | 34.563 | +1.036 | .6956 |
| **full (generative)** | **0.5367** | **60.60** | **33.863** | −2.065 | **.7414** |

Generative wins on every metric. Switched.

## Open defect: the band is not calibrated

**Coverage is 0.537 against a 0.80 target.** Shipping continues with that
known, on the reasoning that generative is strictly better than what it
replaces and the band is a secondary display column — not that it is correct.

The switch was recommended on the argument that a shared volume draw makes a
player's stats move together, so correlation would come for free. Measurement
says that argument is wrong twice over:

1. **No projection uncertainty.** Generative's spread is sampling noise around
   a **fixed** team volume and **fixed** conversion rates. The dominant real
   uncertainty — that the projection itself is wrong — is not represented at
   all. The residual bootstrap does represent it and merely scrambles the
   correlation, which is why a **joint bootstrap** (resampling a whole donor
   player-season's residual vector) reaches **0.757** on the same grain.
2. **Season-scale efficiency noise decorrelates the stats.** The conversion
   draws multiply volume by a fresh lognormal whose sigma (0.35 receiving,
   0.25 rushing, 0.20 passing) was chosen when the path emitted *per-game*
   lines. Over a season, volume is nearly deterministic (CV ~5%) while that
   factor carries CV ~36%, so it dominates: simulated yards correlate with
   receptions at **0.13**, against +0.871 in real residuals. Shrinking sigma
   restores it (0.15 → 0.31, 0.05 → 0.71).

So the two flaws partly offset — too narrow from (1), too noisy on efficiency
from (2) — and the net is a band that is both too tight overall and wrongly
shaped within a player.

Pinned by `test_season_scale_efficiency_noise_decorrelates_volume_stats`,
which asserts the defect and fails once it is fixed.

## Step 1 done: sigmas recalibrated, and it is NOT the coverage fix

Measured with `scripts/fit_conversion_sigmas.py` as the SD of
log(actual season efficiency / predicted) over held-out player-seasons. The
earlier claim in this document — that the sigmas were uniformly too large for
season scale — was wrong. Only receiving was:

| conversion | was | measured |
|---|---|---|
| receiving | 0.35 | 0.279 (WR .273 / TE .242 / RB .315) |
| passing | 0.20 | **0.267** (too small) |
| rushing | 0.25 | **0.468** (RB .367 / **QB .636**) (much too small) |

Held out on 2025, before → after:

| metric | before | after |
|---|---|---|
| coverage | 0.5367 | **0.5383** |
| p50 MAE | 33.863 | **33.098** |
| rho | .7414 | **.7493** |
| width | 60.60 | 60.93 |

**Coverage moved 0.16pp — nothing.** Point accuracy and rank improved
usefully (MAE −0.77, rho +0.008), so the recalibration earns its place, but
it does not touch the interval defect.

That isolates the cause. The band is not too narrow because efficiency
dispersion was mis-set; it is too narrow because **volume carries only
sampling uncertainty**. Measured on the same residuals, real volume
dispersion is ~0.78 log-SD for WR targets (≈91% CV) against ~42% CV in the
simulated draws — roughly a factor of two, and it accounts for essentially
the whole remaining gap. (An earlier note in this work put that gap at an
order of magnitude; that was wrong — the Dirichlet share draw contributes
more spread than it credited.)

## Step 2 done: projection uncertainty closes most of the gap

Implemented as `src/projection/models/uncertainty.py`, fit leakage-safely by
`scripts/fit_v3_uncertainty.py` from rolling OOF folds:

- **Team environment**: correlated multivariate-normal residual on team
  (pass_attempts, carries), covariance fitted from OOF team-level residuals
  — captures the two stats moving together, not just each independently.
- **Opportunity shares**: Dirichlet concentration fitted by method-of-moments
  from actual-vs-predicted share variance, per pool (`qb_attempts`,
  `receiving_targets`, `rb_carries`), replacing a flat `concentration=10.0`.
- **Availability**: beta-binomial games draw, concentration fitted from
  actual-vs-expected games variance, bucketed by position and a fragility
  flag (expected games < 14).
- **Conversion sigmas**: same measurement as step 1, now fit inside the same
  leakage-safe manifest rather than hard-coded, with the step-1 constants
  kept as fallback.
- **Replacement sink**: `allocate_opportunities` tracks volume that could not
  be allocated when a room's prior is zero, rather than inventing a share
  for a player who is not there.

Fit through 2024 (`training_seasons: [2023, 2024]`), measured held out on
2025, confirmed stable across two seeds and two draw counts (300 and 500):

| position | interim | full (this step) | target |
|---|---|---|---|
| QB | 0.48–0.50 | 0.57–0.58 | 0.80 |
| RB | 0.41 | 0.69–0.71 | 0.80 |
| TE | 0.54–0.55 | 0.83–0.84 | 0.80 |
| WR | 0.53 | 0.73–0.75 | 0.80 |
| **overall** | 0.50 | **0.72–0.73** | 0.80 |

Coverage gap closed from −0.26 to roughly −0.08 overall. p50 MAE is flat to
slightly worse (34.6 vs 33.1 after step 1 alone) and band width nearly
doubled (60.6 → ~93) — expected, since representing "the projection could be
wrong" necessarily widens the band; the useful number is coverage, not width.

**Correction to an in-progress misdiagnosis, left in for the record.** A
first pass reported QB coverage identical to four decimal places between
interim and full (0.5244 both) and read that as a shared defect — the
`qb_attempts` share concentration hit its fitted floor of 1.0, so the
worry was that uncertainty wasn't reaching QB at all. It was a coincidental
tie: both modes happened to cover exactly 43 of 82 QBs at that seed. A
different seed gave 0.500 vs 0.561, with 38% of individual QBs flipping
their covered/uncovered verdict between modes — ordinary sampling noise
around a real rate, not two mechanisms failing the same way. Confirmed by
re-running at 500 draws.

**QB remains the weakest position** (0.57 vs 0.80) even after correcting
that misdiagnosis. The `concentration=1.0` floor may still be part of it —
QB rooms are the most top-heavy of the three pools, so a wide Dirichlet
occasionally hands a backup a large, unrealistic share — but real QB outcome
variance (benchings, injuries, in-season role changes) may also just exceed
what the fitted covariance captures. Not chased further here.

## Next, in order

1. ~~Recalibrate the conversion sigmas for season aggregates.~~ Done; helps
   p50, not coverage on its own.
2. ~~Add projection uncertainty.~~ Done; overall coverage 0.538 -> 0.72-0.73.
   RB/TE/WR land near or above target; QB remains under at ~0.57.
3. Investigate the QB shortfall specifically — start with whether
   `qb_attempts` concentration is a fitting artifact (only 3 pools total,
   QB is the most top-heavy) versus genuine unmodeled QB variance.
4. Re-measure after any QB-specific fix; if overall coverage still falls
   short, take the band from a joint bootstrap (0.757 on the old grain) for
   the remaining gap and keep generative for p50 and rank.
5. Re-point the calibration gate at fantasy-points coverage. It currently
   reports 0.8013, which is genuine but per-stat-rate; the percentiles it
   authorises cover ~0.72, not that number.

## Note on interim

Retired as a shipping mode, kept callable so
`scripts/compare_simulation_modes.py` can still score it. `--mode interim` on
`scripts/run_v3_simulation.py` reaches it for comparison only.
