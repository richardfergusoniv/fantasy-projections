# fantasy-projections

NFL fantasy projections, simulation, and release promotion. Python API + jobs
(`src/`), React PWA (`web/`).

This file is the short cross-tool entry for agents (Cursor, Claude Code,
Compound Engineering, and others). **`CLAUDE.md` remains the detailed review
bible.** Do not symlink, replace, or duplicate it here.

## North star

The live product is **Vegas weekly props**. The work now is a weekly,
game-level projection model we can update as 2026 results come in; that model
stays in **shadow** until it clears the promotion gate. ADP and season-long
Vegas are **sanity checks only** — not weekly targets, not optimization
objectives, and not evidence of weekly accuracy.

## Where to look

- Review rules (leakage, model shape, shadow-until-promoted, conservation,
  validation, Vegas props correctness): [`CLAUDE.md`](CLAUDE.md)
- Promotion gate, three Vegas-props roles, ADP allowed / not-allowed:
  [`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md)
- Locked M1–M3 weekly-latent design:
  [`docs/research/WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md`](docs/research/WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md)
- PR description contract:
  [`docs/ops/CURSOR_PR_CONTRACT.md`](docs/ops/CURSOR_PR_CONTRACT.md) and
  [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
- Review pipeline (Cursor implements, Claude reviews code, Perplexity is
  **manual** subscription chat + GitHub connector — no API Action):
  [`docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md`](docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md)

## Standing agent habits

- Prefer **fewer theme PRs**. Once a theme PR is open, land follow-up work on
  that branch instead of opening a parallel PR for the same theme.
- After a plan is locked, execute in **mission** style: fewer confirmation
  gates; keep going until the plan is done or a true blocker.
- Interrupt a human only for merges, spend or secrets, promote-to-production,
  or a genuine product fork. Do not pause for routine implementation choices
  the locked plan already covers.
- A **silent-green Claude** run (CI green with no review comment) is an
  **incident**, not a pass. The Claude Code Review Action must leave a comment
  on every PR it reviews.

## Compound Engineering

Team defaults live in [`.compound-engineering/config.yaml`](.compound-engineering/config.yaml).
The committed template is [`.compound-engineering/config.example.yaml`](.compound-engineering/config.example.yaml).
Machine-local overrides go in `.compound-engineering/config.local.yaml` (gitignored).
Re-run `/ce-setup` to health-check the plugin and refresh local config.
