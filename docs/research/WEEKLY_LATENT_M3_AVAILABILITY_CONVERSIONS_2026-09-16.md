# Weekly latent Milestone 3 — availability + conversions (shadow board)

**Date:** 2026-09-16  
**Status:** research/shadow implementation (not a promotion)  
**Locked parent:** [`WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md`](WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md) §5 Milestone 3, §2–4 Phases B–C, §6–7  
**M2 prior:** [`WEEKLY_LATENT_M2_TEAM_WEEK_LATENT_2026-09-16.md`](WEEKLY_LATENT_M2_TEAM_WEEK_LATENT_2026-09-16.md)  
**Promotion roles:** [`WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md)

**Does not:** replace Vegas weekly props, promote/reseal League Value, change freeze knobs or `APP_PROJECTION_SOURCE`, implement Role 3 blend, use ADP or season-long Vegas as drivers, or wire into the PWA.

---

## 1. What changed vs Milestone 2

M2 ships a team-week latent that may move season mass:

\[
V_{t,k,w}^{\mathrm{M2}}
= A_{t,w}\cdot \frac{V_{t,k}^{\mathrm{sealed}}}{\sum_{w'} A_{t,w'}}
\cdot m_{t,w}^{\mathrm{HA}}\cdot m_{t,k,w}^{\mathrm{opp}}\cdot m_{t,w}^{\mathrm{env}}
\]

M1/M2 player-weeks then set \(A_{i,w}=A_{t,w}\) (1 on active team-weeks, 0 on bye) and convert at **season** rates. Yards were a share of the team yardage pool.

M3 is Phase B–C on the **same package**. Team volume stays the M2 latent. Player availability and conversions become week-varying:

\[
A_{i,w}=0 \quad\text{on a bye (overrides cannot un-zero a bye).}
\]

\[
A_{i,w}=\operatorname{clip}\bigl(A_i^{\mathrm{base}}\cdot m_{i,w}^{\mathrm{rest}},\,0,1\bigr)
\quad\text{otherwise, unless a lagged sit override is attached.}
\]

\(A_i^{\mathrm{base}}=1\). Season exposure is **not** applied again here. Role shares are built from sealed `pred_season` (`pred_pg × projected_games`), so `projected_games / 17` is already inside \(s_{i,k}\). Multiplying weekly \(A\) by that ratio would square the discount and smear an 8-game projection across all 17 weeks. Week-level \(A_{i,w}\) is only play/sit: rest haircut, lagged override, or bye. Short rest (`< 6` days) is a sit-risk haircut (starters 0.97, backups 0.90). Extra rest does **not** create games. This is player-level availability, not a second copy of M2 `env_mult` (which already moves team volume).

A player projected for 8 games therefore keeps the smaller share and \(A=1\) on active weeks (unless sit/rest/bye), not \(A=8/17\) on every week. Expected games are the sum of weekly \(A\), not a second 17-game multiplier.

Opportunity:

\[
x_{i,k,w}=A_{i,w}\,s_{i,k}\,V_{t,k,w}^{\mathrm{M2}}
\]

for attempts / targets / carries (role shares + `other` still sum to 1). Box stats use **season rates × opponent-only conversion multipliers**, not the team yardage-pool share (that would double-count the M2 matchup into YPA):

\[
m_{i,w}^{\mathrm{pass\,conv}}=1+\kappa_{\mathrm{conv}}(m_{t,w}^{\mathrm{opp,pass}}-1),\qquad \kappa_{\mathrm{conv}}=0.50
\]

clipped to \([0.90,1.10]\). Rush conversions use the rush opponent factor. INT rate uses the **inverse** pass factor (tougher pass D → more INTs). Completions cannot exceed attempts; receptions cannot exceed targets; INTs cannot exceed attempts − completions; TDs cannot exceed the matching count.

**Season = sum of weeks**, including \(A_{i,w}=0\) byes. Conservation compares \(\sum_w\) weekly fantasy to an independent `fantasy_points_board` from `season_box_from_players` on the **summed weekly box**, not a second groupby of the same `fantasy_points` column. That check can fail. It is not sealed-`pred_season` reconciliation: M2/M3 may move season mass.

Unchanged from M2:

- Named shares + `other` \(= 1\) inside each team-week (role simplex).
- Same-week realized volume is forbidden.
- Vegas weekly props remain the **external benchmark**, not a blend target.
- `available_at` advances via max of attached vintages and never moves earlier. Component columns (`available_at_board`, `env_available_at`, `prior_available_at`, plus M3 `avail_available_at` / `conv_available_at`) stay on the row.

---

## 2. Features and `available_at`

| Feature | Role in M3 | `available_at` rule |
|---|---|---|
| M2 team-week volume | \(V^{\mathrm{M2}}\) prior for opportunity | max(board, schedule-env, opponent prior) as in M2 |
| Sealed `projected_games` | already inside role share (`pred_season`); **not** a second \(A^{\mathrm{base}}\) scale | board vintage (`v2_baseline_20260830` / `M1_AVAILABLE_AT`) |
| Rest sit-risk | \(m^{\mathrm{rest}}\) | schedule-env vintage |
| Lagged sit override | replaces \(A_{i,w}\) on that player-week | override `available_at` required and must be \(\le\) kickoff (`gameday`/`kickoff_at`); missing or post-kickoff refuses. Merge key is `(player_id, season, week)` when season is on the slate. Row `available_at` still advances via max of vintages and never moves earlier. |
| Opponent pass/rush factor | conversion multipliers (not HA/env) | `conv_available_at` = prior vintage (or board if none) |
| Bye | \(A_{i,w}=0\) | — |

Row `available_at` is the max of board / env / prior / availability / conversion stamps. A January 2026 EPA prior does not make an August board row earlier.

**Not M3 drivers:** spread, total, implied team total, ADP, season-long Vegas, weather, same-week `team_attempts` / `team_carries` / `team_targets` / `team_air_yards`. Those names fail closed.

---

## 3. Backtest reported *inside* M3

Not a renamed M2. Not a random split. Two leakage-safe rolling-origin evaluations of the object M3 actually ships (`allocate_players_m3`):

| Harness | What it is | What it is not |
|---|---|---|
| **Synthetic** (`backtest_synthetic.json`) | AAA WR, weeks \(t=1..4\). Week-3 sit is attached **only** when predicting week 3 (as-of). Same-week `team_attempts` poison must raise. MAE vs M2 `allocate_players` (constant \(A=A_{t,w}\), season rates). Later weeks advance `available_at`. | A model result. Demonstrates the harness. |
| **Historical schedule** (`backtest_historical.json`) | Same 2025-shaped 5-team / 5-week slate as M2, plus one WR per team. Opponent EPA for week \(t\) uses weeks \(< t\). BUF WR sits week 3 (not a team bye); sit vintage is after week 2. Bye \(A=0\). Season fantasy = sum of weeks. Outcomes live in a separate table (`realized_receptions`). | 32-team nflverse box scores. 6–8 live 2026 weeks. A promotion argument. |

Shared fail-closed rule: injecting same-week `team_attempts` / `team_carries` / `team_targets` / `team_air_yards` raises. Both harnesses must pass for `run_milestone3` to report `backtest_passes`.

Beating the M2-player naive on these slates shows the sit + conversion layer uses lagged features. It is **not** evidence the 2026 board is ready to serve.

---

## 4. Vegas weekly props — Role 2 hook, not Role 3

M3 publishes `output/shadow_weekly_schedule_m3/shadow_board_role2.csv` (`player_id`, `season`, `week`, `market`, `model_mean`) so it can be fed to PR #83's `compare_shadow_vegas_props.py --board` later. This PR **does not depend on #83 merging**.

If `src.projection.weekly_eval` is importable, `compare_m3_to_vegas_props` calls `compare_shadow_to_vegas`. Otherwise a local fixture scorer (`fixtures/vegas_props_m3_synthetic.csv`) reports MAE vs timestamped `implied_mean`, failing closed on missing `as_of` / `kickoff_at` / post-kickoff snapshots. Non-empty `blend_weights` raise (Role 3 forbidden).

This is **compare, do not replace**. Vegas weekly props stay what the app serves.

**Market-sanity bands vs ADP / season Vegas** are **stubbed / deferred** (`market_sanity.json`, `used_as_driver=false`). Allowed later as outside alarms (widen when ADP and season Vegas disagree). Not used as drivers or training targets in M3.

---

## 5. Code and runbook

Same package as M1/M2: `src/projection/weekly_latent/` (no second stack).

| Piece | Path |
|---|---|
| Availability | `availability.py` |
| Conversions | `conversions.py` |
| Player-weeks | `allocate.allocate_players_m3` |
| Identities | `conservation.evaluate_m3` |
| Backtest | `backtest_m3.py` |
| Role 2 hook | `vegas_hook.py` |
| CLI | `scripts/run_weekly_schedule_m3.py` |
| Tests | `tests/test_weekly_latent_m3.py` |
| Shadow output | `output/shadow_weekly_schedule_m3/` (large CSVs gitignored) |

```bash
uv run python scripts/run_weekly_schedule_m3.py --dry-run
uv run python scripts/run_weekly_schedule_m3.py --season 2026
uv run pytest tests/test_weekly_latent_m1.py tests/test_weekly_latent_m2.py tests/test_weekly_latent_m3.py -q
```

---

## 6. Gate

**Still shadow. Not a promotion.** A synthetic harness, a compact 2025-shaped
player-week backtest, and a 2026 preseason allocation do not clear the 6–8
live-week bar. Vegas weekly props stay what the app serves. Role 3 blend is
not implemented.
