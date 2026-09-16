# Weekly-model promotion and Vegas weekly props roles — 2026-09-15

**Date:** 2026-09-15
**Scope:** north-star + gate checklist. No production code, sealed-board, or pointer change.
**Does not:** skip Milestone 1 or jump to hierarchical Monte Carlo.

## Context

Direction: leave frozen League Value; build the weekly game-level model in
**shadow**; Vegas weekly props remain the live product board for now.

This note locks two things that later PRs keep conflating:

1. **Promotion discipline** — what must be true before the independent weekly
   model leaves shadow and is served in the app.
2. **How market data may be used** — three distinct roles for weekly props, and
   a tighter rule for ADP / season-long Vegas.

It is distilled from an independent Perplexity governance review (2026-09-15)
and the prior locked weekly-latent design (milestones M1–M3). The Perplexity
pass was a manual subscription-chat review; it is not yet committed under
`docs/research/`. It is a **checklist**. It does **not** redefine those
milestones or supersede the design lock. It does not authorize skipping
Milestone 1 (deterministic schedule allocation / conservation) or jumping
straight to hierarchical Monte Carlo.

## Decision

Serve Vegas weekly props until the independent weekly model clears the
promotion gate below. Do not blend market into the model until a market-free
challenger has been reported. Research merges and M1–M3 shadow work are not
promotions until the gate below clears.

## 1. Vegas weekly props — three roles (do not conflate)

| Role | What it is | What it is not |
|---|---|---|
| **1. Production benchmark** | What the app serves today while the independent weekly model is shadow | A training target, a feature to copy, or proof the model is “done” |
| **2. Evaluation comparator** | Timestamped market snapshots per player/stat; de-vig when possible; compare model distributions to **realized outcomes** and to **market-implied distributions** | A live scrape used as if it were the historical as-of line |
| **3. Optional ensemble** | Allowed only **after** independence testing: a market-free challenger is reported first | A default blend. Without the challenger you cannot tell whether football features add information or merely copy the benchmark |

Role 1 is product. Role 2 is measurement. Role 3 is a later modeling choice
that is forbidden until Role 2 has been run on a market-free candidate.

## 2. ADP / season-long Vegas

**Allowed as**

- Preseason / early-season priors
- Missing-player imputation
- Sanity-check alarms (position bands; **widen** when ADP and season-long
  Vegas disagree)

**Not allowed as**

- Weekly targets
- Primary covariates after evidence arrives
- Optimization objectives (do not fit or select toward ADP / season-market
  agreement)

Snapshot dates are required. Later-season market information must not leak
into earlier historical rows.

## 3. Promotion gate

This is the defensible bar **before the independent weekly model leaves
shadow and is served in the app**. Clearing it is not implied by a green
unit test, a research merge, or a short early-season sample.

| Check | Bar |
|---|---|
| Live shadow sample | Minimum **6–8 live shadow weeks**. Prefer a **full season** for broad claims. |
| Integrity | No material leakage or missingness regressions. |
| Baselines | Better than naive baselines (last game / trailing-3, season-to-date rate, EWM, hierarchical position/team mean) **and** a public expected-opportunity baseline where available, across most positions. |
| Vs weekly-prop benchmark | Statistically credible **improvement or parity** on primary losses. Not RMSE alone — include pinball / CRPS / coverage / calibration, and prop Brier where relevant. |
| Calibration | Intervals within prespecified tolerances. |
| Subgroups | No severe failure on backups, Q-tagged, or low-volume players. |
| Thin 2026 sample | With only **1–2 weeks** of 2026 live data: **update the model, do not declare superiority**. Estimate on historical seasons; treat 2026 as sequential external validation. |

This gate is about promoting the **independent weekly model**. Vegas props
have their own scrape / freshness gates; those are a separate mechanism.

## 4. Milestone alignment

This note **does not supersede** the locked weekly-latent design. Milestone
meanings and wording below are those of
[PR #71](https://github.com/richardfergusoniv/fantasy-projections/pull/71).
Judge weekly-model PRs against the locked milestone they land in. This gate
does not authorize skipping M1.

| Milestone | Scope | Promotion? |
|---|---|---|
| **M1** | Deterministic weekly schedule allocation, no new ML | No |
| **M2** | Team-week latent that may move season totals | No |
| **M3** | Weekly availability + conversions; compare, don't replace, Vegas | Still shadow until the gate above clears |

**Backtesting is a requirement inside each milestone**, not a separate
ladder rung. Each milestone must report leakage-safe, rolling-origin
evaluation of the object it actually ships. A historical schedule backtest
does not replace M2.

### Milestone 1 — Deterministic weekly schedule allocation, no new ML

Use the **2026 schedule** to turn **existing season team volume + player
role shares** into weekly means, then aggregate back. Matchup multipliers
**renormalize**; they reshape weeks and **do not** change season team
totals.

**Not in M1:** training, same-week realized volume features, Vegas
replacement, League Value promote, PWA wiring, ADP/season-Vegas blending.

### Milestone 2 — Team-week latent that may move season totals

After M1 conservation is proven, allow opponent/environment to change
\(V_{t,k}^{\mathrm{season}}\) instead of only reshaping weeks. Opponent
defense priors must be **lagged / preseason**. Still no ADP or season
Vegas as drivers. Vegas weekly props remain the benchmark, not the target
to copy.

### Milestone 3 — Weekly availability + conversions; compare, don't replace, Vegas

Week-varying \(A_{i,w}\) and conversion latents. Publish a shadow weekly
board. Score it against **Vegas weekly props** as the external benchmark.
Optional empirical market-sanity bands vs ADP / season Vegas (widen when
those two disagree). Still no promote/reseal of League Value from this
track.

The promotion gate applies to **serving** the independent model in the app.
Research merges and M1–M3 shadow work are not promotions until that gate
clears.

## Related

These design/process notes are **not yet on `master`**. Until they land,
cite the open PRs rather than relative paths that 404 from this file.

- Weekly latent design lock — [PR #71](https://github.com/richardfergusoniv/fantasy-projections/pull/71)
  (lands at `docs/research/WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md`)
- Review pipeline note — [PR #73](https://github.com/richardfergusoniv/fantasy-projections/pull/73)
  (lands at `docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md`)
- [`docs/PIPELINE_MAP.md`](../PIPELINE_MAP.md) — related-decisions list
