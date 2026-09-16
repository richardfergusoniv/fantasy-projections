# Weekly latent Milestone 2 — team-week latent that may move season mass

**Date:** 2026-09-16  
**Status:** research/shadow implementation (not a promotion)  
**Locked parent:** [`WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md`](WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md) §5 Milestone 2, §6–7  
**Promotion roles:** [`WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md)

**Does not:** replace Vegas weekly props, promote/reseal League Value, change freeze knobs or `APP_PROJECTION_SOURCE`, train a hierarchical Monte Carlo / M3 availability+conversion layer, use ADP or season-long Vegas as drivers, or wire into the PWA.

---

## 1. What changed vs Milestone 1

M1 proved conservation first. Matchup multipliers (home/away × shrunk opponent) were **renormalized** so they could only reshape weeks:

\[
\tilde{\omega}_{t,k,w}
= \frac{A_{t,w}\, m_{t,w}^{\mathrm{HA}}\, m_{t,k,w}^{\mathrm{opp}}}
       {\sum_{w'} A_{t,w'}\, m_{t,w'}^{\mathrm{HA}}\, m_{t,k,w'}^{\mathrm{opp}}},
\qquad
V_{t,k,w}^{\mathrm{M1}} = \tilde{\omega}_{t,k,w}\, V_{t,k}^{\mathrm{sealed}}
\]

\[
\sum_w V_{t,k,w}^{\mathrm{M1}} = V_{t,k}^{\mathrm{sealed}}
\]

M2 keeps the same schedule grid, the same sealed board as a **prior**, and the same compositional shares. It **stops renormalizing**. Opponent and environment multipliers are allowed to change season team volume:

\[
V_{t,k,w}^{\mathrm{M2}}
= A_{t,w}\cdot \frac{V_{t,k}^{\mathrm{sealed}}}{\sum_{w'} A_{t,w'}}
\cdot m_{t,w}^{\mathrm{HA}}\cdot m_{t,k,w}^{\mathrm{opp}}\cdot m_{t,w}^{\mathrm{env}}
\]

On a full 17-game slate \(\sum_{w'} A_{t,w'} = 17\). On the synthetic dry-run grid it is the number of active weeks in the frame, so a 2-game toy slate is not accidentally scaled by \(2/17\).

Season mass identity (what M2 **does** enforce):

\[
\sum_w V_{t,k,w}^{\mathrm{M2}}
= V_{t,k}^{\mathrm{sealed}} \cdot \operatorname{mean}_{w: A_{t,w}=1}\!\left(m^{\mathrm{HA}}\, m^{\mathrm{opp}}\, m^{\mathrm{env}}\right)
\]

That mean can differ from 1. The delta vs sealed (and vs M1 with the **same** priors, which still conserves) is measured in `conservation.json`. It is not a failure.

Unchanged from M1:

- Named shares + `other` \(= 1\) inside each team-week.
- Bye volume is exactly 0.
- Same-week realized volume is forbidden.
- Vegas weekly props remain the external benchmark, not a blend target.

---

## 2. Features and `available_at`

| Feature | Role in M2 | `available_at` rule |
|---|---|---|
| Sealed season team volume / role shares | Prior \(V^{\mathrm{sealed}}\) and shares | `2026-08-30T00:00:00+00:00` (`v2_baseline_20260830`) |
| Home / away / neutral | \(m^{\mathrm{HA}}\) = 1.03 / 0.97 / 1.00 (same as M1) | Schedule vintage = sealed preseason (`M2_SCHEDULE_ENV_AVAILABLE_AT`) |
| Opponent defensive EPA (prior season) | \(m^{\mathrm{opp}}\) after z-score × \(\kappa=0.06\), then M1 \(\lambda_w\) shrinkage | `2026-01-05T00:00:00+00:00` (end of 2025 REG week 18) — **earlier** than the sealed board |
| Rest days | \(m^{\mathrm{env}}\) short week \(<6\)d → 0.97; \(10+\)d → 1.02; else 1.00 | Schedule vintage |
| International / neutral site | travel 0.98 on top of HA 1.00 | Schedule vintage |
| In-season lagged EPA (backtest only) | \(m^{\mathrm{opp}}\) from weeks **strictly before** \(t\) | max `gameday` of those lagged weeks — **later** than the preseason stamp |

**Overwrite rule (design lock §6).** `allocate_team_weeks_m2` does **not** stamp `M1_AVAILABLE_AT` blindly. Row `available_at` is `max(board, schedule-env, prior)`. A later-knowable prior (in-season L5 / as-of EPA) moves the cutoff forward. A prior-season EPA dated January 2026 does **not** make the row earlier than the sealed board when the board is attached; the max is still August 30. Tests cover both directions.

**Not M2 drivers:** spread, total, implied team total, ADP, season-long Vegas, weather, injuries, same-week `team_attempts` / `team_carries` / `team_targets` / `team_air_yards`. Those names fail closed if they appear on the M2 graph.

Opponent shrinkage \(\lambda_w\) is the M1 preseason schedule (weeks 1–4 keep the prior; 15–18 shrink hard). The 2026 shadow run is an August-vintage board: far-out 2026 opponent *form* is not knowable yet, even though 2025 EPA is. Both rolling-origin harnesses use \(\lambda_w=1\) because that vintage is as-of week \(t\). The historical harness passes `schedule_env_available_at` as the 2025 preseason stamp so 2025 rows are not branded with `M1_AVAILABLE_AT` (2026-08-30).

---

## 3. Opponent EPA source

Prefer the same lagged defensive EPA already in the feature stack (`opp_def_pass_epa_prior` / `opp_def_rush_epa_prior` grain: opponent’s **prior-season** EPA allowed).

1. If `projections.db` has `pbp`, mean REG `epa` by `defteam` for season \(S-1\) (2025 for a 2026 run). No join to same-week `team_attempts`.
2. Else the committed fixture `src/projection/weekly_latent/fixtures/opp_def_epa_prior_2025.csv`.

The fixture is nflverse `stats_team_week_2025` REG, offensive `passing_epa` / `rushing_epa` averaged by `opponent_team` (the defense). That is **game-level** EPA, the same family `add_opponent_defense_features` uses, not per-play. Cloud agents usually lack `D:\fantasy-projections-data\projections.db`; the fixture is the offline path. On Richard’s Windows machine the DB can replace it:

```bat
set FANTASY_PROJECTIONS_DATA_DIR=D:\fantasy-projections-data
set FANTASY_PROJECTIONS_DB_PATH=D:\fantasy-projections-data\projections.db
uv run python scripts/run_weekly_schedule_m2.py --season 2026
```

Factor map (deterministic, not fitted on 2026 outcomes):

\[
m^{\mathrm{opp}} = \operatorname{clip}\bigl(1 + \kappa \cdot z(\mathrm{EPA}_{\mathrm{allowed}}),\; 0.85,\; 1.15\bigr)
\]

Higher EPA allowed = worse defense = more offense volume for the team facing them. \(\kappa = 0.06\) so \(\pm 1\) SD is about \(\pm 6\%\) team volume.

---

## 4. Backtest reported *inside* M2

Not a renamed M3. Not a random split. Two leakage-safe rolling-origin
evaluations of the object M2 actually ships (`allocate_team_weeks_m2`,
deterministic volume that may move season mass):

| Harness | What it is | What it is not |
|---|---|---|
| **Synthetic** (`backtest_synthetic.json`) | 2-team, weeks \(t=1..4\). Same-week `team_attempts` poison must raise. MAE vs even-split naive of the season prior. Later weeks advance `available_at` (max of attached vintages). | A model result. Demonstrates the harness. |
| **Historical schedule** (`backtest_historical.json`) | 2025-shaped 5-team / 5-week slate (home/away, rest, byes). Opponent EPA for week \(t\) uses weeks \(< t\), or prior-season if that opponent has no lagged game. Season volume is a **prior-season analogue**, not the sum of the realized weeks (that would leak weeks \(t+1..\) into week 1). Outcomes live in a separate table (`realized_pass_attempts`, never `team_attempts` / `team_carries` / `team_targets` / `team_air_yards` on the feature frame). `available_at` is max(board, schedule-env, prior); schedule-env is the 2025 preseason stamp so 2025 rows are not branded with the 2026 M1 cutoff. | 32-team nflverse box scores. 6–8 live 2026 weeks. A promotion argument. |

Shared fail-closed rule: injecting same-week `team_attempts` / `team_carries` / `team_targets` / `team_air_yards` onto the prediction frame raises (`refuse_forbidden_m2_columns`). Both harnesses must pass for `run_milestone2` to report `backtest_passes`.

On this compact historical slate, M2 weekly pass-attempt MAE is about 1.50 vs even-split naive 2.80. That shows the shipped multipliers use lagged opponent features. It is **not** evidence the 2026 board is ready to serve.

**Still not claimed**

- Not 6–8 live 2026 shadow weeks (promotion gate, decision note §3).
- Not a Vegas weekly-props comparison (role 2). Not pinball / CRPS / coverage.
- Not trailing-3 / EWM / hierarchical-mean / public expected-opportunity baselines — only even-split of the season prior.
- Not a full historical season of realized nflverse team-week volume.
- Not a real 2025 preseason projection board as \(V^{\mathrm{sealed}}\) — the volume table is a prior-season analogue constructed so week 1 cannot see weeks 2–5.
- Not M3 availability, conversions, or probabilistic draws.
- Reconstructing an earlier vintage of the same row (Tuesday vs Friday vs 90-min-pre-kickoff) is out of scope; a single scalar `available_at` is a max.

A historical schedule backtest of M1 does not replace this. This does not replace the promotion gate.

---

## 5. Code and runbook

Same package as M1: `src/projection/weekly_latent/` (no second stack).

| Piece | Path |
|---|---|
| Allocation | `allocate.allocate_team_weeks_m2` |
| Env / cutoff | `environment.py` |
| Priors | `priors.py` + `fixtures/opp_def_epa_prior_2025.csv` |
| Identities | `conservation.evaluate_m2` |
| Backtest | `backtest.py` (synthetic + historical schedule) |
| CLI | `scripts/run_weekly_schedule_m2.py` |
| Tests | `tests/test_weekly_latent_m2.py` |
| Shadow output | `output/shadow_weekly_schedule_m2/` (large CSVs gitignored) |

```bash
uv run python scripts/run_weekly_schedule_m2.py --dry-run
uv run python scripts/run_weekly_schedule_m2.py --season 2026
uv run pytest tests/test_weekly_latent_m1.py tests/test_weekly_latent_m2.py -q
```

---

## 6. Gate

**Still shadow. Not a promotion.** A synthetic harness, a compact 2025-shaped
schedule backtest, and a 2026 preseason allocation do not clear the 6–8
live-week bar. Vegas weekly props stay what the app serves.
