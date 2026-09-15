# Cursor agents: PR description contract

Use this when opening or updating a pull request.

GitHub fills the description from [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md). Do not replace that template with a free-form summary. Fill every section.

## Required

1. **Change summary** — what changed and why, in concrete terms.
2. **Scope** — production / research / diagnostic; modules, artifacts, and public schema.
3. **Contracts preserved** — every checkbox. If a contract is out of scope, mark that line `N/A` (for example `- [ ] Selected-board hash behavior — N/A`). Do not skip rows or leave them blank.
4. **Evaluation** — tests, backtests, holdout usage, new artifacts, gate verdict. Write `none` when nothing was run; do not omit the fields.
5. **Risk and rollback** — known limitations and how to undo the change.

## Research / weekly PRs

Weekly, shadow, and other research PRs still use this template.

- Mark production contracts **N/A** when the change does not touch the selected board, simulation overlays, sealed bundles, or the active release pointer.
- Still fill **Evaluation** (what was measured, holdout, leakage evidence) and **Risk and rollback**.
- Weekly / hierarchical / opportunity-first work stays shadow until it beats the accuracy-first incumbent under leakage-safe evaluation. See [`docs/PIPELINE_MAP.md`](../PIPELINE_MAP.md) §9 and [`docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md`](../decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md).

## Do not

- Do not invent a shorter description because the PR is “docs only” or “research only.”
- Do not leave production contract boxes unchecked without `N/A`.
- Do not change sealed boards, release pointers, or curated `starters_YYYY.csv` unless the PR is explicitly about that work and the contracts above are filled as in-scope.
