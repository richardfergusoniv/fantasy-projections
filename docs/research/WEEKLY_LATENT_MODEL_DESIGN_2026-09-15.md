# Weekly latent model design (locked 2026-09-15)

**Date:** 2026-09-15  
**Status:** locked design + Milestone 1 research/shadow implementation + Milestone 2 research/shadow (see `WEEKLY_LATENT_M2_TEAM_WEEK_LATENT_2026-09-16.md`)  
**Does not:** replace Vegas weekly props, promote/reseal League Value, change freeze knobs or production defaults, train a model, or wire into the PWA.

Sealed League Value stays `v2_baseline_20260830`. This track reads it as **scaffolding + prior**, not as a board to defend as “correct.”

---

## 1. Product split (non-negotiable)

| Surface | Role in this track |
|---|---|
| **Vegas weekly props** | External **weekly benchmark**. M1–M3 do not replace it. |
| **League Value / season ensemble** | Scaffolding + prior. Do not promote, reseal, or freeze-knob it from this work. |
| **ADP and season-long Vegas** | **Not model drivers.** Later they may appear only as outside **market-sanity guardrails** (empirical bands; widen when ADP and season Vegas disagree). They are not blended into Milestone 1 outputs. |

Research/shadow only. No production wiring.

---

## 2. Causal chain

The unit of fantasy is a **scheduled week**, not a 17-game average. Season totals are sums of weeks.

1. **Schedule / opponent / location / environment** set the game context for team \(t\) in week \(w\).
2. That context drives **team plays, pass/rush mix, and scoring**.
3. Mix and scoring become **team opportunity**: attempts, targets, carries, red-zone chances.
4. Each player \(i\) is **active** \(A_{i,w}\) and holds a **role share** \(s_{i,k,w}\) of opportunity type \(k\).
5. **Player opportunity** is active × share × team opportunity.
6. **Conversions** (completion rate, YPA/YPC, catch rate, TD rates) turn opportunity into box stats.
7. Box stats score to **weekly fantasy**.
8. **Season = sum of weeks.** A bye is \(A_{i,w}=0\), not \(\mathrm{final}\times 17\).

```text
schedule, opponent, location, env
        → team plays / pass-rush / scoring
        → team attempts / targets / carries / RZ
        → player active A_i,w + role share
        → player opportunity
        → conversions
        → weekly fantasy
        → season = Σ weeks
```

---

## 3. Formulas

Season points are a sum of weeks, with byes as inactivity:

$$
FP_i = \sum_{w=1}^{18} A_{i,w}\, f(x_{i,w})
$$

Bye weeks are not a 17-game multiplier:

$$
A_{i,w} = 0 \quad \text{on a bye (and other inactive weeks).}
$$

A weekly mean is a function of context, team opportunity, share, and availability:

$$
x_{i,w} = f\!\left(\text{schedule}_{t,w},\; V_{t,\cdot,w},\; s_{i,\cdot,w},\; A_{i,w}\right)
$$

Team opportunity conservation (named players plus an explicit other bucket):

$$
\sum_{i \in t} s_{i,k,w} + s_{\mathrm{other},k,w} = 1
$$

Milestone 1 team-volume conservation, **before** any matchup is allowed to change season mass:

$$
\sum_{w=1}^{18} V_{t,k,w} = V_{t,k}^{\mathrm{season}}
$$

Weekly weights (home/away × shrunk opponent) are renormalized so they cannot move the season total:

$$
\tilde{\omega}_{t,k,w}
= \frac{A_{t,w}\, m_{t,w}^{\mathrm{HA}}\, m_{t,k,w}^{\mathrm{opp}}}
       {\sum_{w'} A_{t,w'}\, m_{t,w'}^{\mathrm{HA}}\, m_{t,k,w'}^{\mathrm{opp}}},
\qquad
V_{t,k,w} = \tilde{\omega}_{t,k,w}\, V_{t,k}^{\mathrm{season}}
$$

Preseason opponent strength is shrunk toward 1.0 as the week gets farther away:

$$
m_{t,k,w}^{\mathrm{opp}} = 1 + \lambda_w \left(s_{\mathrm{opp}(t,w),k} - 1\right)
$$

with \(\lambda_w\) high in weeks 1–4 and heavily shrunk in weeks 15–18 (see §6).

---

## 4. Phases A–C (model layers)

These are layers of the **same causal chain**, not competing boards.

### Phase A — Team environment (team-week latent)

Schedule, opponent, location, rest, surface, and (in-season) weather / lines / injuries → team plays, pass/rush mix, scoring, then team attempts / targets / carries / RZ.

M1 does **not** fit this layer. It **allocates** existing season team volume onto the 2026 schedule with simple, renormalized multipliers.

### Phase B — Availability and role

\(A_{i,w}\) and compositional shares. Shares plus `other` equal 1 inside a team-week. M1 uses **current season role shares** from the sealed board, constant across weeks, with \(A_{i,w}=0\) on byes.

### Phase C — Conversions and fantasy

Opportunity → yards / TDs / receptions at the player’s own season rates → weekly half-PPR. Season is the sum of those weeks.

Later phases can make conversion rates week-varying. M1 keeps them at season rates so player box stats stay linear in allocated opportunity.

---

## 5. Milestones 1–3 (roadmap)

### Milestone 1 (this lock) — Deterministic weekly schedule allocation, no new ML

Use the **2026 schedule** to turn **existing season team volume + player role shares** into weekly means, then aggregate back.

**Implemented conservation (locked first):**

1. \(\sum_w\) team pass attempts / carries / pass yards / rush yards \(=\) sealed season team volume (`team_*_pg_pred` × 17).
2. Within each team-week, named shares + `other` \(= 1\) for attempts, targets, carries, and the yardage pools.
3. Bye volume is exactly 0.
4. Matchup multipliers **renormalize**; they reshape weeks and **do not** change season team totals.

Player-season points reconcile to the sealed `pred_season` when named volume did not overflow the team pool. Where the sealed board’s named players already exceed `team_pg × 17`, M1 **rescales onto the simplex** and reports `rescaled_overflow`. The conserved object in that case is the **team pool**, not the overflowing named sum. That is intentional: we will not invent extra team plays to paper over board inconsistency.

**Not in M1:** training, same-week realized volume features, Vegas replacement, League Value promote, PWA wiring, ADP/season-Vegas blending.

### Milestone 2 — Team-week latent that may move season totals

After M1 conservation is proven, allow opponent/environment to change \(V_{t,k}^{\mathrm{season}}\) instead of only reshaping weeks. Opponent defense priors must be **lagged / preseason**. Still no ADP or season Vegas as drivers. Vegas weekly props remain the benchmark, not the target to copy.

**Implemented (2026-09-16, research/shadow, not a promotion):** [`WEEKLY_LATENT_M2_TEAM_WEEK_LATENT_2026-09-16.md`](WEEKLY_LATENT_M2_TEAM_WEEK_LATENT_2026-09-16.md). Same package (`src/projection/weekly_latent/`), CLI `scripts/run_weekly_schedule_m2.py`. M2 does **not** renormalize matchup multipliers. `available_at` is advanced via max of board, schedule-env, and per-row prior vintages (never moved earlier). Backtests inside M2: synthetic rolling-origin plus a 2025-shaped historical schedule with prior-only opponent features. Still not 6–8 live weeks, still not a promotion.

### Milestone 3 — Weekly availability + conversions; compare, don’t replace, Vegas

Week-varying \(A_{i,w}\) and conversion latents. Publish a shadow weekly board. Score it against **Vegas weekly props** as the external benchmark. Optional empirical market-sanity bands vs ADP / season Vegas (widen when those two disagree). Still no promote/reseal of League Value from this track.

---

## 6. Schedule feature table

M1 **attaches** these features to every team-week. Only **opponent, home/away, and bye** enter the M1 volume multipliers. The rest are scaffolding for Phases A–C / M2–M3.

| Feature | M1 use | Notes |
|---|---|---|
| Opponent | Multiplier (if a lagged prior exists; else 1.0) | Shrunk toward 1.0 by week. Never same-week box. |
| Home / away | Multiplier 1.03 / 0.97 | Listed-home at a **neutral/international** site uses 1.00, not 1.03. |
| Week | Shrinkage schedule \(\lambda_w\) | 1–4: 1.00; 5–8: 0.70; 9–11: 0.45; 12–14: 0.25; 15–18: 0.10. |
| Bye | \(A_{t,w}=0\) | One bye per team on the 18-week grid. |
| Division | Feature only | `div_game` from schedule + conference/division from `TEAM_META`. |
| Rest | Feature only | `home_rest` / `away_rest` from nflverse schedule. |
| Travel / international | Feature only | 2026 neutrals in the fixture are international sites. |
| Surface / roof / stadium | Feature only | Pregame venue. |
| Weather (temp/wind) | In-season update | Not a M1 driver. |
| Spread / total | In-season update | **Not a M1 driver.** Vegas stays the weekly benchmark, not an input to blend. |
| Starting QB / injuries | In-season update | Future \(A_{i,w}\) / role; not M1. |

**Cutoff / `available_at`.** Every M1 team-week and player-week row carries `available_at`. For M1 this is the sealed preseason snapshot (`2026-08-30T00:00:00+00:00`, `v2_baseline_20260830`), **not kickoff**. M1 features are schedule / home-away / bye / opponent scaffolding plus allocated season volume — no same-week realized volume. A Tuesday vintage and a 90-minute-pre-kickoff vintage are different cutoffs; M1 does not distinguish them because it has no in-season features.

**M2 must overwrite `available_at`** when it attaches opponent priors or in-season updates, because those are knowable later than the preseason snapshot. Adding the column now (one writer: `allocate_team_weeks`) is cheaper than retrofitting it across M2 inputs.

---

## 7. Explicit correction: as-of shares vs same-week team volume

Source: PR #70 / `docs/research/WEEKLY_FEATURE_CLEANLINESS_2026-09-15.md` (local integrity findings).

**Weekly roll3 shares are passed as-of.** `targets_share_roll3` / `carries_share_roll3` and v2 `*_l3` are `shift(1)` then rolling. A target week’s roll3 does **not** contain that week’s own box. That recipe is safe to build on **as a lagged feature**, when we get there.

**Do not train on same-week `team_attempts` / `team_carries` attached by `add_team_pass_rate`.** That helper correctly lags `team_pass_rate_l5`, then also left-joins **current-week** `team_attempts` and `team_carries` onto the panel. Lagged pass-rate is OK. The same-week volume columns are not. M1 does not call `add_team_pass_rate` and does not put those column names on its tables. `FORBIDDEN_SAME_WEEK_TRAINING_FEATURES` is a literal copy of the team-prefixed `SAME_WEEK_OUTCOME_DENYLIST` names (`team_targets`, `team_carries`, `team_attempts`, `team_air_yards`) so `allocate.py` does not import the polars pipeline; `tests/test_weekly_latent_m1.py` is the drift alarm.

M1 role shares come from the **sealed season board**, not from in-season roll3. Roll3 becomes relevant in M2/M3 as a lagged in-season updater, still as-of.

---

## 8. Milestone 1 implementation seams (no second data stack)

| Input | Seam |
|---|---|
| Season player totals + team pg volume | Public sealed long CSV `draft_assistant/data/releases/v2_baseline_20260830/projections_2026.csv` |
| 2026 REG schedule | Committed slim fixture `src/projection/weekly_latent/fixtures/nfl_schedules_2026_reg.csv` (nflverse `games.csv` REG 2026, **scores/lines/QBs stripped**) |
| Team-weeks + byes | Pandas analogue of `explode_schedules_to_team_weeks` **without** Vegas implied totals and **without** `add_team_pass_rate` |
| Division / aliases | Same maps as `src/team_stats/prepare.py` `TEAM_META` and `src/projection/weekly/data/teams.py` |
| Scoring | `ScoringConfig` half-PPR from `src/projection/weekly/config/scoring.py` |
| Conservation spirit | Same identities as `src/projection/weekly/draws/conservation.py` / `normalize_shares`, at season-week grain rather than draw grain |

Code: `src/projection/weekly_latent/`. CLI: `scripts/run_weekly_schedule_m1.py`. Tests: `tests/test_weekly_latent_m1.py`. Shadow output: `output/shadow_weekly_schedule_m1/` (large CSVs gitignored; summary + sample committed).

Default opponent factors are **1.0** in this environment (no `projections.db`). Home/away + bye still reshape weeks. Optional `--` priors / local DB path may supply lagged opponent factors later; M1 will **not** join same-week realized volume to get them.

---

## 9. Milestone 1 runbook

From repo root (Linux/macOS):

```bash
uv run python scripts/run_weekly_schedule_m1.py --dry-run
uv run python scripts/run_weekly_schedule_m1.py --season 2026
uv run pytest tests/test_weekly_latent_m1.py -q
```

On Richard’s Windows machine, if the cloud VM lacks `D:\fantasy-projections-data\projections.db` (it does; M1 does not need the DB for the sealed-board path):

```bat
set FANTASY_PROJECTIONS_DATA_DIR=D:\fantasy-projections-data
set FANTASY_PROJECTIONS_DB_PATH=D:\fantasy-projections-data\projections.db
uv run python scripts/run_weekly_schedule_m1.py --season 2026
```

The DB is **optional**. It is not used to train. If present, M1 still refuses a same-week `team_attempts` join and keeps opponent factors at 1.0 unless a true lagged prior can be formed. The sealed public projections CSV in this repo is enough for the 2026 allocation.

Override paths:

```bash
uv run python scripts/run_weekly_schedule_m1.py --season 2026 \
  --projections draft_assistant/data/releases/v2_baseline_20260830/projections_2026.csv \
  --schedule src/projection/weekly_latent/fixtures/nfl_schedules_2026_reg.csv \
  --output output/shadow_weekly_schedule_m1
```

---

## 10. What “done” means for this lock

- This document is the dated design lock.
- M1 code + conservation tests + shadow output path (and this runbook if the DB is missing).
- Live League Value, freeze knobs, production defaults, and the PWA are unchanged.
