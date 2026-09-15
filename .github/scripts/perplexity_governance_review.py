#!/usr/bin/env python3
"""Phase 1 Perplexity governance review for pull requests.

This is NOT a code-quality reviewer. Claude already owns that job
(.github/workflows/claude-code-review.yml). This script asks Perplexity
one contract question: does the PR preserve the mathematical, calibration,
artifact, and release-control contract?

Phase 1 is informational. The caller (the GitHub Action) always treats
this process as a success, even when the model says REQUEST CHANGES.
Severity belongs in the PR comment, not in a red required check.

stdlib only — no extra pip packages on the runner.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Current documented Sonar chat endpoint and a stable model.
# https://docs.perplexity.ai/docs/sonar/quickstart
# https://docs.perplexity.ai/api-reference/sonar-post
SONAR_URL = "https://api.perplexity.ai/v1/sonar"
SONAR_MODEL = "sonar-pro"

COMMENT_MARKER = "<!-- perplexity-governance-review -->"
COMMENT_TITLE = "## Perplexity governance review (informational)"

# Unified-diff budget. Huge generated files are summarized, not dumped.
DIFF_CHAR_CAP = 200_000
# GitHub issue-comment hard limit is 65536; leave room for the header.
COMMENT_CHAR_CAP = 60_000
# Extra docs we attach in full (PIPELINE_MAP + changed decision/research).
DOC_CHAR_CAP = 80_000

# When the diff is huge, keep these paths first.
PRIORITY_PREFIXES = (
    "src/",
    "docs/decisions/",
    "docs/research/",
    "docs/ops/",
    "docs/PIPELINE_MAP.md",
    "scripts/",
    "tests/",
    "output/",
    "config/",
    ".github/workflows/",
    "draft_assistant/data/",
)

DECISION_DOC = "docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md"

SYSTEM_PROMPT = """\
You are the independent model-governance reviewer for Fantasy Decisions
(fantasy-projections). This is NOT a code-style review and NOT a second
Claude Code Review. Claude already reviews readability, tests, and obvious
bugs. Do not restate Claude-style nits (naming, formatting, missing type
hints, “add a unit test for this rename”) unless you are disagreeing with
Claude or the nit is actually a contract break.

Question you must answer:
Does this PR preserve the mathematical, calibration, artifact, and
release-control contract?

Review only the packet in the user message (PR metadata, diff, CI, and
attached docs). Do not invent files, hashes, holdout results, or CI
outcomes that are not in the packet. If evidence is missing, say so
under Missing tests / evidence.

Classify the PR as one of: modeling/calibration, simulation/draw
infrastructure, finish probability/VORP, weekly/shadow, docs/ops only,
or mixed. Apply the matching checklist.

Governance checklist (summarized from the review-pipeline decision and
PIPELINE_MAP; out-of-scope is valid when the PR marks those contracts N/A):

1. Leakage / as-of
   - Target-week or post-cutoff outcomes must not enter features used to
     predict that week or season (shift-then-roll; unused holdout).
   - Rolling-origin / leakage-safe folds only. Do not select weights on
     the holdout that will later “prove” them.

2. Board / hash / provenance
   - selected_board_hash / selected_board_sha256 alignment across
     simulation manifest, players meta, and release report.
   - canonical_projection_run_id on the simulation manifest matches the
     board it claims to describe.
   - Recenter transform version stays v1_median_correction unless the PR
     is an explicit, evidenced version change.
   - WR calibration artifact binding (wr_calibration_version /
     wr_calibration.json hash) stays pinned when that path is in scope.
   - Partition / per-partition hashes on draw artifacts.
   - Sealed bundles are immutable; going live is a pointer swap, not an
     in-place rewrite of sealed bytes.

3. Calibration / holdout / fail-closed
   - Calibration / donor / WR-scale / gate hashes still bind to the
     artifacts they name.
   - A hold or missing verdict zeroes uncertainty or withholds overlay
     rather than shipping a partial candidate.
   - Promotion has no force flag. Fail-closed paths stay fail-closed.
   - Statistical validity is vs the frozen baseline, never vs Sleeper
     agreement (external comparison is diagnostic, never a gate).

4. Weekly / shadow
   - Weekly / hierarchical / opportunity-first work stays shadow until
     it beats the accuracy-first incumbent on a documented leakage-safe
     gate. Do not wire into compose / publish / auto-publish on a green
     unit test alone.

5. Finish-probability / VORP (when relevant)
   - Finish-probability fields attach only after the finish gate is ready.
   - Simulated VORP requires that finish gate plus its own gate.
   - Overlays are computed on recentered draws anchored on the
     accuracy-first selected forecast.
   - Deterministic vorp / ranks / tiers remain authoritative.
   - Rank tie policies stay distinct (first vs min). Do not unify them
     silently.

6. Release control
   - No silent mutation of curated depth-chart source
     (src/depth_chart/starters_YYYY.csv).
   - No silent rewrite of active_release_*.json or sealed namespace
     bytes unless the PR is explicitly about promotion and shows the
     six promotion invariants.

Output structured markdown exactly in this order:

### Merge recommendation
One of: APPROVE / REQUEST CHANGES / ESCALATE
(Use ESCALATE when a human must decide a contract change that the packet
cannot settle.)

### Risk level
One of: Low / Medium / High / Critical

### Summary
A short paragraph answering the contract question with evidence.

### Findings
Severity-ranked list: Blocker, High, Medium, Low.
Each finding: severity, file/line when possible, why it is a contract
issue, what evidence is missing. If there are no contract findings,
write “None.”

### Missing tests / evidence
Holdout, as-of audit, hash alignment, gate JSON, CI gaps.

### GitHub-ready comment body
A concise comment Richard can read on the PR. Repeat the recommendation
and the important findings. Do not dump the whole checklist.

Approve only if the contract question is answered with evidence from the
PR description, tests, and pinned artifacts — or if the PR is explicitly
docs/ops with production contracts marked N/A. Do not approve because
the code “looks clean.”
"""


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def write_step_summary(text: str) -> None:
    path = env("GITHUB_STEP_SUMMARY")
    if not path:
        print(text)
        return
    Path(path).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def is_priority(path: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in PRIORITY_PREFIXES)


def split_unified_diff(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into (path, chunk) pairs."""
    chunks: list[tuple[str, str]] = []
    current_path = ""
    current: list[str] = []

    def flush() -> None:
        nonlocal current_path, current
        if current:
            chunks.append((current_path or "(unknown)", "".join(current)))
        current_path = ""
        current = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            # diff --git a/foo b/foo
            parts = line.strip().split()
            if len(parts) >= 4:
                current_path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
            current.append(line)
        else:
            if line.startswith("+++ b/") and not current_path:
                current_path = line.strip()[6:]
            current.append(line)
    flush()
    return chunks


def prioritize_diff(diff: str, cap: int = DIFF_CHAR_CAP) -> tuple[str, list[str]]:
    chunks = split_unified_diff(diff)
    if not chunks:
        clipped = diff[:cap]
        note = [] if len(diff) <= cap else ["(unparsed diff truncated)"]
        return clipped, note

    ordered = [c for c in chunks if is_priority(c[0])] + [c for c in chunks if not is_priority(c[0])]
    kept: list[str] = []
    omitted: list[str] = []
    size = 0
    for path, chunk in ordered:
        if size + len(chunk) <= cap:
            kept.append(chunk)
            size += len(chunk)
        else:
            omitted.append(path)
    return "".join(kept), omitted


def collect_changed_docs(changed_files: list[str], repo_root: Path) -> str:
    """Attach PIPELINE_MAP plus any changed decision/research/ops docs."""
    wanted: list[str] = []
    pipeline = repo_root / "docs" / "PIPELINE_MAP.md"
    if pipeline.is_file():
        wanted.append("docs/PIPELINE_MAP.md")
    decision = repo_root / DECISION_DOC
    if decision.is_file():
        wanted.append(DECISION_DOC)
    for path in changed_files:
        if path.startswith(("docs/decisions/", "docs/research/", "docs/ops/")):
            if path not in wanted:
                wanted.append(path)

    parts: list[str] = []
    for rel in wanted:
        full = repo_root / rel
        if not full.is_file():
            parts.append(f"\n----- {rel} (missing on HEAD) -----\n")
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        if len(text) > DOC_CHAR_CAP:
            text = text[:DOC_CHAR_CAP] + f"\n\n[truncated {rel} at {DOC_CHAR_CAP} chars]\n"
        parts.append(f"\n----- {rel} -----\n{text}")
    return "".join(parts)


def collect_packet(repo_root: Path) -> dict[str, str]:
    pr = env("PR_NUMBER")
    packet: dict[str, str] = {}

    code, out, err = run(
        [
            "gh",
            "pr",
            "view",
            pr,
            "--json",
            "title,body,number,author,isDraft,labels,url,baseRefName,headRefName,additions,deletions,changedFiles",
        ]
    )
    packet["pr_json"] = out if code == 0 else f"(gh pr view failed: {err.strip() or out.strip()})"

    title = body = author = ""
    draft = "unknown"
    labels = ""
    try:
        meta = json.loads(packet["pr_json"]) if packet["pr_json"].startswith("{") else {}
        title = str(meta.get("title") or "")
        body = str(meta.get("body") or "")
        author = str((meta.get("author") or {}).get("login") or "")
        draft = str(bool(meta.get("isDraft")))
        labels = ", ".join(
            item.get("name", "") for item in (meta.get("labels") or []) if item.get("name")
        )
    except json.JSONDecodeError:
        meta = {}

    code, out, err = run(["gh", "pr", "diff", pr, "--name-only"])
    changed_text = out if code == 0 else ""
    changed_files = [line.strip() for line in changed_text.splitlines() if line.strip()]
    packet["changed_files"] = "\n".join(changed_files) if changed_files else "(none / unavailable)"

    code, out, err = run(["gh", "pr", "diff", pr], timeout=180)
    raw_diff = out if code == 0 else f"(gh pr diff failed: {err.strip()})"
    clipped, omitted = prioritize_diff(raw_diff)
    if omitted:
        clipped += (
            "\n\n[diff truncated over "
            f"{DIFF_CHAR_CAP} chars; omitted {len(omitted)} file(s), "
            "priority given to src/, docs/decisions/, docs/research/, "
            "scripts/, tests/, output manifests]\n"
            + "\n".join(f"- {path}" for path in omitted[:80])
        )
        if len(omitted) > 80:
            clipped += f"\n- … {len(omitted) - 80} more\n"
    packet["diff"] = clipped

    code, out, err = run(["gh", "pr", "checks", pr])
    if code == 0 and out.strip():
        packet["ci"] = out.strip()
    else:
        packet["ci"] = (
            "(CI check summary unavailable yet — this review may be racing "
            f"other jobs. {err.strip() or out.strip()})"
        )

    packet["docs"] = collect_changed_docs(changed_files, repo_root)
    packet["title"] = title
    packet["body"] = body
    packet["author"] = author
    packet["draft"] = draft
    packet["labels"] = labels or "(none)"
    packet["number"] = pr
    return packet


def build_user_prompt(packet: dict[str, str]) -> str:
    return f"""Review packet for fantasy-projections PR #{packet.get("number", "?")}.

PR title: {packet.get("title") or "(missing)"}
Author: {packet.get("author") or "(missing)"}
Draft: {packet.get("draft") or "(missing)"}
Labels: {packet.get("labels") or "(none)"}

----- PR description (contract checklist lives here) -----
{packet.get("body") or "(empty description)"}

----- Changed files -----
{packet.get("changed_files") or "(none)"}

----- CI checks (may be incomplete if this job started early) -----
{packet.get("ci") or "(none)"}

----- Unified diff (priority-trimmed if huge) -----
{packet.get("diff") or "(none)"}

----- Attached pipeline / decision / research text -----
{packet.get("docs") or "(none)"}
"""


def call_sonar(system_prompt: str, user_prompt: str, api_key: str) -> tuple[str, str | None]:
    """Return (assistant_text, error_message). error_message is None on success."""
    payload = {
        "model": SONAR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8000,
        # Review the packet, not the public web. Contract facts are in-repo.
        "disable_search": True,
    }
    request = urllib.request.Request(
        SONAR_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        return "", f"Perplexity API HTTP {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return "", f"Perplexity API network error: {exc.reason}"
    except TimeoutError:
        return "", "Perplexity API timed out after 180s"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "", f"Perplexity API returned non-JSON: {raw[:500]}"

    choices = data.get("choices") or []
    if not choices:
        return "", f"Perplexity API returned no choices: {raw[:500]}"
    content = ((choices[0] or {}).get("message") or {}).get("content") or ""
    if not str(content).strip():
        return "", "Perplexity API returned an empty message"
    return str(content).strip(), None


def extract_recommendation(text: str) -> str:
    match = re.search(r"\b(APPROVE|REQUEST CHANGES|ESCALATE)\b", text)
    return match.group(1) if match else "UNKNOWN"


def extract_risk(text: str) -> str:
    match = re.search(r"Risk level[:\s*]*\**\s*(Critical|High|Medium|Low)\b", text, re.I)
    return match.group(1).title() if match else "Unknown"


def format_comment(body: str, extra_note: str | None = None) -> str:
    header = (
        f"{COMMENT_MARKER}\n"
        f"{COMMENT_TITLE}\n\n"
        "Phase 1 informational review. **This is not a required merge blocker.** "
        "The workflow job succeeds even when the recommendation is "
        "`REQUEST CHANGES` or `ESCALATE`. Claude Code Review remains the "
        "code-quality pass; this comment is independent model/systems "
        "governance and should not merely echo Claude.\n"
    )
    if extra_note:
        header += f"\n{extra_note}\n"
    text = header + "\n" + body.strip() + "\n"
    if len(text) > COMMENT_CHAR_CAP:
        text = text[: COMMENT_CHAR_CAP - 80] + "\n\n[comment truncated for GitHub's size limit]\n"
    return text


def skip_comment() -> str:
    return format_comment(
        "Perplexity review skipped: `PERPLEXITY_API_KEY` not configured.\n\n"
        "Richard: add a **repository** Actions secret named "
        "`PERPLEXITY_API_KEY` (Settings → Secrets and variables → Actions). "
        "Create the key in [Perplexity API settings](https://www.perplexity.ai/account/api). "
        "Until that secret exists, this job will keep succeeding and posting "
        "this skip note so it cannot block merge."
    )


def find_existing_comment_id(pr: str) -> str | None:
    # --jq streams one id per line and stays valid across --paginate pages.
    code, out, err = run(
        [
            "gh",
            "api",
            f"repos/{env('GH_REPO')}/issues/{pr}/comments",
            "--paginate",
            "--jq",
            (
                '.[] | select((.body | contains("'
                + COMMENT_MARKER
                + '")) and '
                + '(.user.login == "github-actions[bot]" or .user.login == "github-actions")) | .id'
            ),
        ]
    )
    if code != 0:
        print(f"comment lookup failed: {err.strip()}", file=sys.stderr)
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[-1] if lines else None


def upsert_comment(pr: str, body: str) -> None:
    comment_id = find_existing_comment_id(pr)
    payload = json.dumps({"body": body}).encode("utf-8")
    if comment_id:
        cmd = [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{env('GH_REPO')}/issues/comments/{comment_id}",
            "--input",
            "-",
        ]
    else:
        cmd = [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{env('GH_REPO')}/issues/{pr}/comments",
            "--input",
            "-",
        ]
    proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=60, check=False)
    if proc.returncode != 0:
        # Last-resort: still try gh pr comment so Richard sees something.
        fallback = subprocess.run(
            ["gh", "pr", "comment", pr, "--body-file", "-"],
            input=body.encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=False,
        )
        if fallback.returncode != 0:
            print(
                "failed to post PR comment: "
                f"{proc.stderr.decode('utf-8', errors='replace')} "
                f"{fallback.stderr.decode('utf-8', errors='replace')}",
                file=sys.stderr,
            )


def main() -> int:
    pr = env("PR_NUMBER")
    repo_root = Path(env("GITHUB_WORKSPACE") or ".").resolve()
    api_key = env("PERPLEXITY_API_KEY")

    if not pr:
        write_step_summary("Perplexity governance review: missing PR_NUMBER.")
        return 0

    if not api_key:
        comment = skip_comment()
        upsert_comment(pr, comment)
        write_step_summary(
            "### Perplexity governance review\n\n"
            "Skipped: `PERPLEXITY_API_KEY` is not configured. "
            "Job succeeded (informational). Richard must add the repo secret "
            "before reviews run.\n"
        )
        print("Perplexity review skipped: PERPLEXITY_API_KEY not configured")
        return 0

    packet = collect_packet(repo_root)
    user_prompt = build_user_prompt(packet)
    review, error = call_sonar(SYSTEM_PROMPT, user_prompt, api_key)

    if error:
        comment = format_comment(
            "Perplexity API call failed. This is **not** a merge blocker.\n\n"
            f"```\n{error}\n```\n\n"
            "If this is 401/403, check that repo secret `PERPLEXITY_API_KEY` "
            "is a valid key from Perplexity API settings."
        )
        upsert_comment(pr, comment)
        write_step_summary(
            "### Perplexity governance review\n\n"
            f"API call failed (informational, job still green):\n\n```\n{error}\n```\n"
        )
        print(f"Perplexity API failed (informational): {error}")
        return 0

    recommendation = extract_recommendation(review)
    risk = extract_risk(review)
    comment = format_comment(review)
    upsert_comment(pr, comment)
    write_step_summary(
        "### Perplexity governance review (informational)\n\n"
        f"- Merge recommendation: **{recommendation}**\n"
        f"- Risk level: **{risk}**\n"
        f"- Model: `{SONAR_MODEL}`\n"
        "- This job always succeeds in Phase 1. Severity is in the PR comment.\n"
    )
    print(f"Posted Perplexity governance review: {recommendation} / {risk}")
    return 0


if __name__ == "__main__":
    # Informational: never become a merge blocker from an unexpected exception.
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — last-resort comment, then green
        pr = env("PR_NUMBER")
        message = f"{type(exc).__name__}: {exc}"
        if pr:
            upsert_comment(
                pr,
                format_comment(
                    "The governance reviewer crashed before it could finish. "
                    "This is **not** a merge blocker.\n\n"
                    f"```\n{message}\n```\n"
                ),
            )
        write_step_summary(
            "### Perplexity governance review\n\n"
            f"Script error (informational):\n\n```\n{message}\n```\n"
        )
        print(f"Perplexity reviewer crashed (informational): {message}")
        sys.exit(0)
