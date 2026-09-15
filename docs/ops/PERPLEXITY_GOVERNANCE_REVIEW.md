# Perplexity governance review — ops note

Automatic, **informational** PR comment from `.github/workflows/perplexity-governance-review.yml`. It is not a required merge check.

## How this differs from Claude Code Review

| | Claude (`.github/workflows/claude-code-review.yml`) | Perplexity (this Action) |
|---|---|---|
| Job | Code quality | Independent model / systems governance |
| Question | Is the diff readable, tested, and free of obvious bugs? | Does this PR preserve the mathematical, calibration, artifact, and release-control contract? |
| Typical findings | Names, tests, CI, logic nits | Leakage / as-of, board hash / provenance, calibration binding, fail-closed gates, weekly shadow rules, finish-prob / VORP when relevant |
| Merge gate | Existing Claude workflow | **Never** — Phase 1 always exits 0. `REQUEST CHANGES` is comment severity only |

Do not treat a green Perplexity job as “Claude already covered it.” The two reviews are supposed to disagree in focus. If Perplexity’s comment is a paraphrase of Claude, the governance pass is incomplete.

## Secret (required before reviews actually run)

Richard must add a **repository** Actions secret named `PERPLEXITY_API_KEY`.

1. Create an API key in [Perplexity API settings](https://www.perplexity.ai/account/api).
2. GitHub → this repo → **Settings → Secrets and variables → Actions → New repository secret**.
3. Name: `PERPLEXITY_API_KEY`. Value: the key.

Until that secret exists, the job still succeeds and posts:

> Perplexity review skipped: `PERPLEXITY_API_KEY` not configured

That skip is intentional so a missing key cannot block merge.

## Cursor agents

When this Action is on the default branch (or otherwise running on the PR), **Cursor agents do not need to call Perplexity manually**. Opening or updating the PR is enough: the workflow collects the packet, calls Sonar, and upserts one PR comment.

Agents should still fill the PR contract template (production contracts `N/A` when they are out of scope). The Action reads that description.

Manual copy-paste of the prompt in [`docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md`](../decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md) is only needed if the Action did not run (secret missing, workflow not on the PR’s base, or GitHub Actions disabled).

## What the Action does

1. Checkout with enough history for `base..head`.
2. Build a review packet (`gh`): PR metadata, changed files, a size-capped unified diff (priority to `src/`, `docs/decisions/`, `docs/research/`, `scripts/`, `tests/`, output manifests), CI check summary, plus `docs/PIPELINE_MAP.md` and changed decision/research docs.
3. `POST https://api.perplexity.ai/v1/sonar` with model `sonar-pro` and the governance prompt. Search is disabled so the model reviews the packet, not the public web.
4. Upsert a PR comment titled `## Perplexity governance review (informational)`.
5. Exit 0. Write a job summary with the recommendation.

`cursor[bot]` PR updates are allowed (same reason Claude sets `allowed_bots: cursor`).

## Phase 1 vs a hard blocker

Hard required status check is **deferred until the review is calibrated**. Do not add this workflow to branch protection yet.
