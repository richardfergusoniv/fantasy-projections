# Per-position miss brief — 2026 season / League Value

**Date:** 2026-09-15
**Audience:** Richard (design decisions)
**Status:** research only

> **No live board change.** This note does not touch the sealed browser board
> (`v2_baseline_20260830`), freeze knobs (RB/WR weights, availability α = 0,
> v3-means cutover off, 10k draws), production defaults, or any promote/reseal
> path. It extends the 2026-09-14 pipeline note; it does not replace it.

| Live constraint | Value |
|---|---|
| Sealed board | `v2_baseline_20260830` (accuracy-first ensemble) |
| This pass | Read committed artifacts + decisions docs. No `promote_release`, no reseal, no freeze-knob edits |
| 2026 outcomes scoreboard | **Not run.** `data/projections.db` is absent. There is no committed 2026 `fantasy_evaluation`. Player names below are the **2025 leakage-safe v1 holdout**, which is what selected the live mix |
| Reproduce | `python3 scripts/research/dump_season_projection_regression_tables.py` |

Evidence tags: **[artifact]** committed file; **[doc]** existing decision/report; **[code]** current source. Hypotheses are labeled **hypothesis** and are not findings.

---

## How to read this

Two different boards get mixed up. This brief keeps them apart.

1. **Native v1** (this repo’s regression) is what the 2025 evaluation CSV
   scores. Player names and “pred vs actual” come from that file.
2. **The live draft / League Value board** is already a mix. For top-120 ADP
   running backs and receivers it mostly trusts the market and v2, not v1.
   Quarterbacks and tight ends still mostly use this repo’s model.

“What went wrong” here means: where v1’s 2025 season-points forecast missed
the players who matter for **rank**, **star/tier**, and **VORP** (value over
a replacement-level player). It is **not** a 2026 in-season recap. Week-1
2026 games exist in the real world; they are not in these artifacts.

Half-PPR, 4-point passing TD. Tier cutoffs: QB12 / RB24 / WR36 / TE12.
Replacement: the 13th / 25th / 37th / 13th player on each method’s own list.

---

## One-page scorecard (2025 v1 holdout)

All-eligible Week-1 roster. Model vs copy-last-year (carry-forward).
**[artifact]** `output/fantasy_evaluation_summary_2025.json`

| | Rank (Spearman ρ) | Points error (MAE) | Star/tier hits | VORP MAE (best baseline) | Live top-120 mix |
|---|---|---|---|---|---|
| **QB** | Strong: 0.78 vs 0.65 | Wins: 39 vs 51 | 7/12, tied with last year | Wins vs last year (40 vs 63) | **0.40 v1 / 0.60 v2** (no ADP) |
| **RB** | Strong: 0.69 vs 0.55 | Wins: 35 vs 37 | **Best cell: 17/24** vs 14 | **Loses** to last year (41 vs 38) | **0.10 v1 / 0.30 v2 / 0.60 ADP** |
| **WR** | Strong: 0.78 vs 0.68 | Wins: 25 vs 28 | **Loses: 19/36** vs 22 | Wins vs last year (28 vs 33) | **0 v1 / 0.55 v2 / 0.45 ADP** |
| **TE** | **Best rank: 0.84** vs 0.69 | Wins: 17 vs 22 | **Worst: 4/12** vs 3 | **Loses** to availability-adjusted (30 vs 25) | **0.90 v1 / 0.10 v2** (no ADP) |

Same pattern on 2023 and 2024 folds: v1 orders the whole room; it does not
reliably pick the stars or set replacement. **[artifact]** summary JSON for
those years.

Draft-relevant slice (top-120 ADP, 2025 untouched holdout) is a different
scoreboard and is what shipped: selected mix MAE 53.1 / ρ 0.60 vs incumbent
58.7 / 0.50. **[artifact]** `output/accuracy_first_2026/report.json`

---

## Quarterback

### 1. Strong vs weak

v1 is good at telling **starters from backups**. That is why the whole-room
rank looks excellent (ρ 0.78 on season points, 0.77 on per-game rates).

It is weak at telling **which starters will be the good ones**:

- Among depth-tier-1 QBs who played 8+ games, **rate** Spearman falls to
  **0.22** and points MAE rises to **62** (from 39). **[artifact]** JSON
  `qb_starter_metrics` / `starter_8plus_games`.
- The 0.78 figure is all-eligible **points** Spearman, not a second starter
  metric. Do not use it as a promotion gate.
- Star/tier: 7 of 12. Replacement (13th) was almost calibrated in 2025
  (pred 241 vs actual 244), so QB VORP is not the broken cell. 2023–2024
  replacement was a bit low (244 vs 263; 251 vs 273).

On the **live** top-120 ADP slice, even the shipped v1/v2 mix only reaches
ρ **0.18** (n=17). ADP-assisted arms were worse (negative Spearman).
**[artifact]** accuracy-first `position_evidence.QB`.

### 2. Notable 2025 misses (v1 native)

Source: **[artifact]** `output/fantasy_evaluation_2025.csv`,
`model_points_end_to_end` vs `actual_points`. Predicted top 12 vs actual top 12.

**Hits that still under-shot the ceiling:** Josh Allen was ranked 9th
(inside QB12) at 253 projected vs **369** actual. Caleb Williams 247 vs 318.

**Missed the actual top 12 (healthy, 17 games):**

| Player | Pred rank | Actual finish | Pred | Actual | Games |
|---|---:|---:|---:|---:|---:|
| Matthew Stafford | 21 | 2 | 201 | 356 | 17 |
| Drake Maye | 23 | 3 | 198 | 353 | 17 |
| Trevor Lawrence | 20 | 4 | 205 | 342 | 17 |
| Dak Prescott | 13 | 6 | 241 | 312 | 17 |
| Jared Goff | 17 | 7 | 221 | 307 | 17 |

**False-positive injuries / short seasons** (projected as QB1, then missed time).
Per-game **rates** for Burrow and Daniels were close; the season miss is games:

| Player | Pred | Actual | Games | Gate A games (CSV) | Rate pred → actual |
|---|---:|---:|---:|---:|---|
| Joe Burrow | 298 | 134 | 8 | 16.3 | 16.5 → 16.8 |
| Jayden Daniels | 255 | 116 | 7 | 11.0 | 16.0 → 16.6 |
| Brock Purdy | 246 | 179 | 9 | 15.1 | 13.2 → 19.9 |
| Anthony Richardson | 237 | 2 | 2 | 16.1 | (almost no season) |
| Kyler Murray | 228 | 80 | 5 | 13.5 | 13.7 → 16.0 |

Lamar Jackson (278 vs 219, 13 games) is mixed: some missed time, rate nearly
flat. Sam Darnold (243 vs 243, 17 games) is a **rank** miss at the cutoff,
not a points miss.

**Role / who-starts misses:** Jacoby Brissett 39 vs 233 (14 games), Daniel
Jones 61 vs 230 (13), rookie Jaxson Dart 96 vs 244 (14). Those are backup
priors meeting unexpected playing time.

Among the predicted top 12, the three with ≤10 games have mean error **+123**
points; the seven with ≥15 games have mean error **−38**. Availability and
under-shooting healthy stars are both in the same board.

### 3. Cause categories (hypotheses)

- **Availability / games (supported for the injury names).** α = 0 prices
  every healthy player at 17 games. Gate A in the CSV also failed to haircut
  Burrow (16.3 vs 8) and Richardson (16.1 vs 2). **Hypothesis H-QB1:** a
  mid-α blend might cut MAE on these names and still be the wrong *draft*
  number. Nested fit already collapsed to flat-17. **[doc]** v1 production
  role / shadow availability hold.
- **Rate forecast wrong (supported for Stafford / Maye / Lawrence / Allen).**
  Full-season games, large per-game under-shoot. **Hypothesis H-QB2:** the
  model is sticky to last year’s usage and does not price bounce-backs or
  second-year leaps. Mobile-rush / multi-season-prior arms were tried and
  **lost** starter MAE. **[doc]** `docs/QB_PROJECTION_FINAL_REPAIR_REPORT.md`
  (NO-GO, 2026-09-03).
- **Depth-chart / role (supported for Brissett / Jones / Dart).** Not a
  starter-rate problem; the Week-1 depth prior did not own the snaps.
- **Team-volume reconcile (supported on the 2026 *board*, not proven as the
  2025 miss).** On the sealed 2026 compose path, Burrow drops raw 14.0 PPG →
  10.9 after volume → 9.4 after TD clip because backups still claim attempts.
  Protect-starter is not enough. **[doc]** QB repair stage attribution.
  Lamar’s rushing is lost in the **raw rate**, then never restored.
- **ADP / consensus:** not the live QB mix. On 2025 top-120, market arms
  *lost* to incumbent v1/v2. **[artifact]** accuracy-first QB arms.
- **Replacement / VORP:** not the 2025 headline defect (13th almost exact).
- **Ensemble weights:** live 40/60 v1/v2 is the incumbent that beat the
  alternatives on that holdout. Changing it now would be fitting 2026.

### 4. What the live sealed board already does

QB in the overlapping top-120 ADP slice: **0.40 v1 / 0.60 v2**, v3 = 0, no
ADP weight. Everyone outside that slice stays on the older v1/v2 incumbent.
v1 remains the component-statline / simulation engine. **[artifact]**
`ensemble_weights.json`. **[doc]** `V1_PRODUCTION_ROLE_2026-08-29.md`,
`ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`.

The 2026-09-03 repair (allocation, mobile rush, multi-season prior) is on
disk under `output/qb_repair/` and is **not** on the sealed board.

### 5. After freeze unlock vs closed now

**Worth exploring after 2026 outcomes (authorized / already named):**

- Unchanged accuracy-first selector, with **starter 8+ games** MAE + Spearman
  as the QB gate — not all-eligible ρ.
- Reuse `src/projection/qb_repair/` arms only if they beat baseline on 2026
  *without* using 2026 for selection. Default prior is NO-GO until then.
- Leakage-safe `fantasy_evaluation` refresh (needs the database).

**Closed mid-season:** promoting those QB arms; resealing; using Sleeper
agreement as a gate; turning on `--v3-means`.

---

## Running back

### 1. Strong vs weak

v1’s **best draft-relevant cell** is RB tier: **17/24** vs last year’s 14/24.
Whole-board rank also wins (0.69 vs 0.55).

It is weak at **how many points the stars are worth** (VORP MAE 40.7 vs
last year’s **38.0**). Replacement was a bit low (pred RB25 151 vs actual
165). Christian McCaffrey is the exhibit: still inside the predicted RB24
(rank 17) and **177 points** light.

On the live top-120 ADP slice, incumbent v1/v2 was MAE 71.3 / ρ 0.45;
the selected mix is **65.3 / 0.53** by putting **0.60 on ADP**.
**[artifact]** accuracy-first RB arms.

Rate models beat naive carry-forward on every RB cell except **receiving
TDs** (the only player-stat loss in the 2025 veteran holdout).
**[artifact]** `output/backtest/veteran_holdout_2025.csv`.

### 2. Notable 2025 misses (v1 native)

**Ceiling under-shot, full season (rate + season):**

| Player | Pred rank | Actual | Pred | Actual | Games | Rate pred → actual |
|---|---:|---:|---:|---:|---:|---|
| Christian McCaffrey | 17 | 1 | 186 | 364 | 17 | 11.4 → 21.4 |
| Jonathan Taylor | 8 | 2 | 205 | 339 | 17 | 12.0 → 20.0 |
| Jahmyr Gibbs | 7 | 4 | 209 | 330 | 17 | 12.4 → 19.4 |
| James Cook | 15 | 5 | 188 | 292 | 17 | 11.1 → 17.2 |
| De'Von Achane | 14 | 6 | 189 | 289 | 16 | 10.5 → 18.0 |

Those five are **inside** the predicted RB24. The miss is stars/VORP, not
“left off the board.”

**Left off the actual RB24:** Travis Etienne 125 vs 236 (rank 33→10);
Javonte Williams 122 vs 229 (35→11); Jaylen Warren 120 vs 197; rookies
TreVeyon Henderson 48 vs 191 and RJ Harvey 119 vs 185; Kenneth Gainwell
**40 vs 187** from depth-tier **3**; Zach Charbonnet 107 vs 169 (depth 2).

**False-positive injuries / aging / zero season:**

| Player | Pred | Actual | Games | Notes |
|---|---:|---:|---:|---|
| Joe Mixon | 182 | 0 | 0 | Gate A still 14.7 |
| James Conner | 182 | 29 | 3 | Gate A 14.0 |
| Alvin Kamara | 247 | 88 | 11 | Rate also over: 14.6 → 8.0 |
| Chuba Hubbard | 187 | 112 | 15 | Rate over: 11.0 → 7.5; not an injury story |
| Bucky Irving | 195 | 126 | 10 | Mixed games + leftover rank |
| Najee Harris | 151 (the predicted RB25) | 10 | 3 | Replacement-level name, then gone |

Ashton Jeanty (rookie) was projected 2nd at 257 vs 220 actual — a hit on
tier with a modest over-shoot. Omarion Hampton was a false-positive rookie
(162 vs 120, 9 games).

Predicted top 24 with ≤10 games: mean error **+111**. With ≥15 games: **−40**.
Same two-sided pattern as QB.

### 3. Cause categories (hypotheses)

- **Rate forecast (supported for CMC / Taylor / Gibbs / Cook / Achane).**
  Healthy, 16–17 games, large per-game under-shoot. **Hypothesis H-RB1:**
  last-year injury (CMC) and “committee prior” shrink the RB1 rate. Elite
  correction is **TE-only**; concentration is identity (`promoted=false`).
  **[artifact]** `models/concentration_calibration.json`. **[doc]** v1 role.
- **Availability / games (supported for Mixon / Conner / Irving).** α = 0
  plus Gate A that still expected ~14 games. **Hypothesis H-RB2:** haircutting
  those names with Gate A would also haircut CMC-class bounce-backs (Gate A
  had CMC at 11.7 games; he played 17). That cancellation is why the shadow
  blend failed. **[doc]** RB/WR shadow closeout.
- **Depth-chart / role (supported for Gainwell depth 3, Charbonnet depth 2,
  Etienne rebound, several rookies).** **[doc]** `RB_ROLE_RATE_DIAGNOSIS`:
  deep RBs are already **over**-predicted at the rate level; an RB-wide
  receiving bump was rejected because it would worsen tiers 2/4/5.
- **Replacement / VORP (supported as a metric, not a single-player cause).**
  Carry-forward wins VORP. Predicted RB25 sat on Najee Harris.
- **ADP / ensemble (supported as the live mitigation).** 0.60 ADP was
  selected because it beat v1 on the draft-relevant slice. **Hypothesis
  H-RB3:** v1 remains useful structurally (identities, simulation) and is
  the wrong *ranking* signal for top-120 RB — already policy.
- **RB receiving TDs:** the one rate cell that loses to naive. Small MAE
  gap (0.0445 vs 0.0433). Not enough, by itself, to explain CMC.

### 4. What the live sealed board already does

Top-120 ADP RB: **0.10 v1 / 0.30 v2 / 0.60 ADP-implied points**, v3 = 0.
Outside that slice: incumbent (historically v1-heavy). Shadow `shadow_v1_rb_wr`
repair is **closed** (`further_repair_authorized=false`).
**[artifact]** `output/shadow_v1_rb_wr/repair_track_closed.json`.

On the 2026 *component* file (not outcomes), ADP/v2 already lift Jeanty and
Chase Brown vs native incumbent and cut some of v1’s Jeremiyah Love / Cam
Skattebo enthusiasm. That is market disagreement, **not** a 2026 grade.
**[artifact]** `output/accuracy_first_2026/fantasy_points_2026.csv`.

### 5. After freeze unlock vs closed now

**Worth exploring after 2026 outcomes:** unchanged accuracy-first selector
on untouched 2026 (the **only** authorized RB weight path). Optional:
WR/RB ceiling work **without** reopening shadow repair (concentration
already rejected; a new elite-correction cell would need the same TE gates).

**Closed mid-season:** deleting the repair-track seal; fitting new RB
weights on weekly 2026 data; Gate-A α retune; RB-only receiving multiplier.

---

## Wide receiver

### 1. Strong vs weak

Whole-board rank and MAE win (0.78 / 25 vs last year’s 0.68 / 28).
**Star/tier loses:** 19/36 vs last year’s **22/36**. That is the
draft-relevant miss.

VORP MAE still beats last year (28 vs 33). Replacement is a bit low
(pred WR37 121 vs actual 133).

On the live top-120 ADP slice the incumbent (v1-heavy) was the **worst**
WR rank of the four positions: ρ **0.23**, MAE 54.6. Selected mix
(0 v1 / 0.55 v2 / 0.45 ADP) is ρ **0.53**, MAE 46.5. The “full” arm tied
only because its v3 weight was zero; the simpler market arm won the
tie-break. **[artifact]** accuracy-first WR arms.

Caveat: 2025 `starter_depth_tier_1` WR n=31 vs ~77–82 in 2023–2024, so
do **not** quote that slice’s 0.18 Spearman as a headline. All-eligible
tier 19/36 is the honest WR star metric.

### 2. Notable 2025 misses (v1 native)

**Ceiling under-shot (healthy or near-healthy):**

| Player | Pred rank | Actual | Pred | Actual | Games | Rate pred → actual |
|---|---:|---:|---:|---:|---:|---|
| Puka Nacua | 9 | 1 | 188 | 312 | 16 | 11.1 → 19.5 |
| Jaxon Smith-Njigba | 7 | 2 | 198 | 302 | 17 | 11.5 → 17.8 |
| George Pickens | 34 | 5 | 128 | 243 | 17 | 7.8 → 14.3 |
| Chris Olave | 44 | 6 | 106 | 218 | 16 | 6.3 → 13.6 |

JSN and Puka were still inside WR36; Pickens and Olave were **tier misses**.

**Injury / availability false positives** (several have *calibrated rates*):

| Player | Pred | Actual | Games | Rate pred → actual |
|---|---:|---:|---:|---|
| Malik Nabers | 211 | 48 | 4 | 11.8 → 12.0 |
| Travis Hunter | 201 | 50 | 7 | 13.2 → 7.1 (rookie two-way + injury) |
| Garrett Wilson | 210 | 82 | 7 | 13.0 → 11.6 |
| Tyreek Hill | 171 | 43 | 4 | 8.7 → 10.8 |
| Calvin Ridley | 155 | 39 | 7 | 8.9 → 5.5 |
| Mike Evans | 150 | 70 | 8 | 9.0 → 8.7 |
| Terry McLaurin | 178 | 95 | 10 | 9.7 → 9.5 |

**Full-season role misses (not injury):** Jerry Jeudy 180 vs 98 **in 17
games** (rate 10.3 → 5.7); Brian Thomas Jr. 192 vs 115 in 14 games
(12.6 → 8.2); Darnell Mooney 130 vs 66 in 15.

**Leap / depth misses:** rookie Emeka Egbuka 37 vs 162 (rank 105→21);
Michael Wilson 63 vs 182 (depth 2); Wan'Dale Robinson 81 vs 172; Parker
Washington 44 vs 144 (depth 4); Troy Franklin 57 vs 141 (depth 4).

Ten of the predicted WR36 played ≤10 games (mean error **+103**). The
21 with ≥15 games still under-shoot on average (**−17**) because of
Puka/JSN/Pickens-class ceiling.

Justin Jefferson finished 25th at 160 vs 113 predicted — a points under-shoot
with 17 games, still a tier hit if the cutoff is 36. Do not over-read the
CSV `model_rate_points` column for every WR; season scoring is
`model_points_end_to_end`.

### 3. Cause categories (hypotheses)

- **Availability (supported for Nabers / Wilson / Hill / Evans / McLaurin).**
  Rates often fine; α = 0 and Gate A ~14–15 games. Same H1 as QB/RB.
- **Rate / ceiling (supported for Puka / JSN / Pickens / Olave).**
  **Hypothesis H-WR1:** share-of-team-passing-yards plus an identity
  reconcile **and** no WR elite correction caps outliers. Concentration
  exponents were fitted and **not promoted**.
- **ADP / consensus (supported as the live mitigation).** Zero v1 weight
  on top-120 WR is not “v1 is useless at WR.” It is “v1’s WR error is
  dominated by ceiling/injury that ADP already prices.” **Hypothesis H-WR2:**
  a v1 WR change that does not beat ADP on 2026 top-120 should not get
  weight back. **[doc]** 2026-09-14 note H3, kept.
- **Depth / role (supported for Egbuka, Wilson, Washington, Franklin, Jeudy).**
  Curated 2026 depth cannot be scored on this fold (file is 2026-only).
- **Replacement / VORP:** milder than TE. Model still wins VORP vs
  carry-forward. Not the WR headline.
- **Ensemble:** v3 point weight already zero.

### 4. What the live sealed board already does

Top-120 ADP WR: **0 v1 / 0.55 v2 / 0.45 ADP**, v3 = 0. That is the strongest
downgrade of v1 on the live board. 2026 component file (not outcomes) shows
the mix lifting Amon-Ra St. Brown (+69 vs incumbent) and Ja'Marr Chase
(+34), and cutting George Pickens (−32) and Jaxon Smith-Njigba (−12).

### 5. After freeze unlock vs closed now

**Worth exploring after 2026 outcomes:** same accuracy-first selector.
Research-only now: WR ceiling **without** reopening `shadow_v1_rb_wr`
(new elite-correction cell would need n-above-knot, positive β, consistency
≥ 2 SE — the TE gates).

**Closed mid-season:** giving v1 WR weight back from this note; in-season
weight fit; promoting concentration (`promoted=false` is current).

---

## Tight end

### 1. Strong vs weak

Best **whole-board rank** of the four positions, every fold (2025 ρ
**0.835**). MAE also wins.

Worst **draft-relevant** miss: **4 of 12** TE1s (last year 3/12, so this is
persistent, not a one-year fluke). 2023 was 7/12; 2024 was 6/12. The top of
the TE board got harder.

VORP is the other broken cell: availability-adjusted last-year points
**win** VORP (24.6 vs model 30.2). Predicted TE13 is too low **every** fold:

| Fold | Predicted TE13 | Actual TE13 |
|---|---:|---:|
| 2023 | 101 | 111 |
| 2024 | 96 | 112 |
| 2025 | 110 | 133 |

On the live top-120 ADP slice, n=10 is thin. Incumbent (90/10 v1/v2) won
the selector on Spearman even though a 100% ADP arm had a lower MAE
(32.9 vs 36.8) and a slightly worse ρ. The rule is MAE **and** Spearman,
by position — ADP did not clear Spearman, so TE stayed incumbent.
**[artifact]** accuracy-first TE arms.

Starter-conditional TE is kinder than all-eligible tier: depth-tier-1 /
8+ games hits **7/12** with rate Spearman **0.61** (2025). The all-eligible
4/12 is the rookies and the second-tier leap, not a collapse among known
starters.

Gate A **lost** to naive on 2025 TE games for roster-eligible players
(MAE 4.24 vs 3.93). Unique among the four positions on that fold.
**[artifact]** `output/backtest/availability_rolling.csv` (2025,
`target_roster_eligible`).

### 2. Notable 2025 misses (v1 native)

Predicted TE12 hits: Brock Bowers, Travis Kelce, Trey McBride, Kyle Pitts.

**Even the hits can be huge under-shoots:** McBride 151 vs **253** (rate
8.6 → 14.9, 17 games). Pitts 114 vs 167. Kelce was essentially exact
(153 vs 153). Bowers 154 vs 142 in 12 games.

**Cutoff curiosity, not a points disaster:** George Kittle 136 vs 133 in
11 games — predicted 6th, actual **13th**. League Value / VORP *uses* that
13th as replacement (133). The model’s replacement name was Dallas Goedert
at 110.

**Missed the actual TE12:**

| Player | Pred rank | Actual | Pred | Actual | Games | Rookie? |
|---|---:|---:|---:|---:|---:|---|
| Dallas Goedert | 13 | 3 | 110 | 155 | 15 | no |
| Harold Fannin Jr. | 31 | 5 | 57 | 152 | 16 | **yes** |
| Tyler Warren | 48 | 6 | 34 | 150 | 17 | **yes** |
| Jake Ferguson | 24 | 7 | 67 | 149 | 17 | no |
| Hunter Henry | 14 | 8 | 98 | 149 | 17 | no |
| Juwan Johnson | 22 | 9 | 77 | 145 | 17 | no |
| Dalton Schultz | 23 | 11 | 70 | 137 | 17 | no |
| Colston Loveland | 18 | 12 | 86 | 136 | 16 | **yes** |

Zero rookies in the predicted TE12; three in the actual TE12.

**False positives (full or near-full seasons, so not just α = 0):**
Jonnu Smith 138 vs 66 **in 17 games** (rate 7.8 → 3.9); Mark Andrews 139 vs
107 in 17; Cade Otton 131 vs 93 in 16; David Njoku 111 vs 70 in 12.
Sam LaPorta (125 vs 87, 9 games) and Tucker Kraft (113 vs 101, 8 games)
are mixed availability.

### 3. Cause categories (hypotheses)

- **Replacement / VORP (supported).** TE13 too low every year.
  Availability-adjusted baseline wins VORP. **Hypothesis H-TE1:** this is
  mostly a replacement-level and second-tier problem, plus a few outliers.
  Raising TE13 without a ceiling model will not recover McBride 2025.
- **Rate / ceiling (supported for McBride; partial for Pitts / Ferguson
  / Henry).** Elite residual correction exists and is **TE-only**, but it
  is omitted on leakage-safe folds — so this 2025 scoreboard **does not
  include** it. **[code]** eval coverage limits. Do not blame the live
  2026 TE board for a correction the holdout never applied.
- **Rookie / role (supported for Warren / Fannin / Loveland).** Rookie
  path is always `low_confidence`. **Hypothesis H-TE2:** 2025 was a
  historically rookie-heavy TE1 class; the 4/12 tier is partly that, not
  a sudden veteran-ordering failure (starter slice 7/12).
- **Depth / role (supported for Jonnu, 17 games, rate cut in half).**
- **Availability (mixed).** LaPorta / Kraft / Kittle / Bowers missed
  games; Gate A lost on 2025 TE. α = 0 is not the main TE1 miss — several
  false positives played 16–17.
- **ADP / ensemble:** live TE has **no** ADP weight. The 2025 top-120 ADP
  arm was not allowed to replace incumbent because Spearman did not
  improve. **Hypothesis H-TE3:** after 2026, TE is the position where an
  accuracy-first *could* move toward market if both gates pass — it is
  not authorized to do that from this note.
- **WR/RB-style shadow repair:** does not apply; v1 **is** the TE ranking
  signal.

### 4. What the live sealed board already does

TE top-120: **0.90 v1 / 0.10 v2**, v3 = 0, no ADP. v1 is the approved
TE ranking signal. 2026 component file (not outcomes) shows v2 much higher
on McBride (302 vs incumbent 193), Kittle (280 vs 95), and Loveland
(250 vs 130) — the live mix mostly **ignores** that v2 lift.

### 5. After freeze unlock vs closed now

**Worth exploring after 2026 outcomes:**

- Accuracy-first selector (TE weights only if **both** MAE and Spearman
  beat incumbent).
- TE replacement / VORP study: is the defect the 13th-TE level, the
  McBride-class ceiling, or rookie TEs? Must not buy VORP by wrecking ρ.
  Compare to the availability-adjusted baseline that already wins VORP.
- Confirm whether live elite correction (TE-only, on the shipped path)
  moves 2026 in a way the leakage-safe CSV never measured.

**Closed mid-season:** retuning TE ensemble toward ADP from 2025 MAE
alone; treating Sleeper TE deltas as accuracy.

---

## Cross-position: what is already handled vs still open

### The freeze is doing its job

The live mix already treats v1 as:

- **QB/TE ranking signal** (with v2 blend).
- **RB/WR structural engine**, not the top-120 ranking signal.

Reopening RB/WR weights, α, v3 means, or draw count **in-season** would be
fitting the board to the season it is trying to predict.

### Authorized after 2026 outcomes (do not auto-promote)

Ranked as in the 2026-09-14 note; repeated here so this brief stands alone:

| | Experiment | Positions it can honestly move |
|---|---|---|
| 1 | Unchanged accuracy-first selector on untouched 2026 | RB/WR first; QB/TE only if they beat incumbent on the same contract |
| 2 | Leakage-safe v1 `fantasy_evaluation` including 2026 | Scoreboard only |
| 3 | QB starter-conditional gates (repair arms only if they pass) | QB |
| 4 | v3 means gate rematch (`promote_v3_means` on every rolling fold) | Means overlay, still not a silent default |
| 5 | Availability α nested-fit, with the 2026-08-30 collapse as tripwire | Exposure only if it does **not** collapse to 17 |
| 6 | TE replacement / VORP calibration | TE, not RB/WR |
| 7 | WR/RB ceiling **without** shadow repair | Possibly concentration or a new elite-correction cell with TE-class gates |
| 8 | Harness extensions (roster_moves, vacancy, curated-depth proxy) | Measurement, not a board change |

### Explicitly closed now

- `promote_release` / reseal `v2_baseline_20260830`.
- Delete `output/shadow_v1_rb_wr/repair_track_closed.json`.
- In-season QB architecture promotion (NO-GO already recorded).
- Fit new weights on 2026 weekly data.
- Sleeper agreement as a gate.

---

## What this brief did not do

- Did not invent a 2026 fantasy evaluation. No `projections.db`.
- Did not re-score the **live ensemble** player-by-player on 2025: the
  accuracy-first player parquet is not in the tree, only position-level
  MAE/Spearman in `report.json`. Names above are **v1 native**.
- Did not treat 2026 ADP vs v1 gaps as outcomes. Those rows in the dump
  are labeled market/model disagreement.
- Did not quote `STATE_OF_BUILD.md` §3.3 or `FREEZE_2026-08-13.md` as
  current numbers (superseded; see the 2026-09-14 note).

---

## Reproduce the cited tables

```bash
# Stdlib; no DB; no train/predict/promote. Missing files → relative-path list.
python3 scripts/research/dump_season_projection_regression_tables.py
```

The dump now prints, from the same committed files: 2023–2025 all-eligible
and starter slices; 2025 per-position predicted-top / false-positive /
false-negative names; accuracy-first weights; 2026 component disagreement
labeled as not outcomes.

Commands that **need** `data/projections.db` were not run:

```bash
python -m src.projection.fantasy_evaluation   # do not use to “refresh” live numbers
python scripts/evaluate_accuracy_first_ensemble.py
```

---

## Sources

- This follow-up plus `docs/research/SEASON_PROJECTION_REGRESSION_REEXPLORE_2026-09-14.md`
- `STATE_OF_BUILD.md` freeze banner (live); do not use stale §2.1 / §3.3 tables
- `docs/decisions/V1_PRODUCTION_ROLE_2026-08-29.md`
- `docs/decisions/ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`
- `docs/decisions/RB_ROLE_RATE_DIAGNOSIS_2026-08-17.md`
- `docs/QB_PROJECTION_FINAL_REPAIR_REPORT.md`
- Artifacts: `output/fantasy_evaluation_summary_202{3,4,5}.json`,
  `output/fantasy_evaluation_2025.csv`,
  `output/backtest/{veteran_holdout_2025,availability_rolling}.csv`,
  `output/accuracy_first_2026/{report,ensemble_weights,fantasy_points_2026}.json/csv`,
  `models/concentration_calibration.json`,
  `draft_assistant/data/active_release_2026.json`,
  `output/shadow_v1_rb_wr/repair_track_closed.json`,
  `output/qb_repair/selection_decision.json`
