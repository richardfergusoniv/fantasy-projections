"""Fail-closed gate for Claude Code Review on ready pull requests.

A green ``claude-review`` check must mean ``claude[bot]`` left a visible
PR comment or review on *this run*. Draft silence remains allowed. PRs
that change ``.github/workflows/claude-code-review.yml`` are waived
because ``anthropics/claude-code-action`` skips when that file does not
match the default branch (workflow-validation).

Live GitHub fetching uses stdlib only so the review job does not need uv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Mapping, Sequence

CLAUDE_LOGINS = frozenset({"claude[bot]", "claude"})
WORKFLOW_VALIDATION_SKIP_PATHS = frozenset(
    {".github/workflows/claude-code-review.yml"}
)
DEFAULT_CLOCK_SLACK = timedelta(seconds=90)
GITHUB_API = "https://api.github.com"

GateStatus = Literal["pass", "fail", "skip_draft", "skip_workflow_validation"]


@dataclass(frozen=True)
class ReviewArtifact:
    kind: str
    login: str
    created_at: datetime
    updated_at: datetime | None
    url: str


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    message: str

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "fail" else 0


def is_claude_login(login: str | None) -> bool:
    if login is None:
        return False
    return login.strip().lower() in {name.lower() for name in CLAUDE_LOGINS}


def parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_since(value: str) -> datetime:
    parsed = parse_github_datetime(value)
    if parsed is None:
        raise ValueError(f"invalid --since timestamp: {value!r}")
    return parsed


def truthy_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def artifact_is_from_this_run(
    artifact: ReviewArtifact,
    *,
    since: datetime,
    slack: timedelta = DEFAULT_CLOCK_SLACK,
) -> bool:
    cutoff = since - slack
    latest = artifact.updated_at or artifact.created_at
    return artifact.created_at >= cutoff or latest >= cutoff


def claude_artifacts_for_this_run(
    artifacts: Sequence[ReviewArtifact],
    *,
    since: datetime,
    slack: timedelta = DEFAULT_CLOCK_SLACK,
) -> list[ReviewArtifact]:
    return [
        artifact
        for artifact in artifacts
        if is_claude_login(artifact.login)
        and artifact_is_from_this_run(artifact, since=since, slack=slack)
    ]


def evaluate_gate(
    *,
    draft: bool,
    changed_files: Sequence[str],
    artifacts: Sequence[ReviewArtifact],
    since: datetime,
    slack: timedelta = DEFAULT_CLOCK_SLACK,
) -> GateResult:
    if draft:
        return GateResult(
            status="skip_draft",
            message="Draft PR: plugin silence is expected. Skipping post gate.",
        )

    if any(path in WORKFLOW_VALIDATION_SKIP_PATHS for path in changed_files):
        return GateResult(
            status="skip_workflow_validation",
            message=(
                "WAIVED: this PR changes .github/workflows/claude-code-review.yml. "
                "claude-code-action skips until that file matches the default "
                "branch (workflow-validation). Not a review pass."
            ),
        )

    matched = claude_artifacts_for_this_run(artifacts, since=since, slack=slack)
    if matched:
        urls = ", ".join(item.url for item in matched)
        return GateResult(
            status="pass",
            message=f"Claude posted on this run ({len(matched)} artifact(s)): {urls}",
        )

    return GateResult(
        status="fail",
        message=(
            "claude-review is fail-closed: this ready PR has no claude[bot] "
            "issue comment, review, or inline review comment created during "
            "this job. A green check requires visible Claude output."
        ),
    )


def _user_login(payload: Mapping[str, Any]) -> str | None:
    user = payload.get("user")
    if not isinstance(user, Mapping):
        return None
    login = user.get("login")
    return login if isinstance(login, str) else None


def artifacts_from_issue_comments(
    comments: Iterable[Mapping[str, Any]],
) -> list[ReviewArtifact]:
    out: list[ReviewArtifact] = []
    for comment in comments:
        created = parse_github_datetime(comment.get("created_at"))
        if created is None:
            continue
        out.append(
            ReviewArtifact(
                kind="issue_comment",
                login=_user_login(comment) or "",
                created_at=created,
                updated_at=parse_github_datetime(comment.get("updated_at")),
                url=str(comment.get("html_url") or comment.get("url") or ""),
            )
        )
    return out


def artifacts_from_reviews(
    reviews: Iterable[Mapping[str, Any]],
) -> list[ReviewArtifact]:
    out: list[ReviewArtifact] = []
    for review in reviews:
        created = parse_github_datetime(
            review.get("submitted_at") or review.get("created_at")
        )
        if created is None:
            continue
        out.append(
            ReviewArtifact(
                kind="review",
                login=_user_login(review) or "",
                created_at=created,
                updated_at=created,
                url=str(review.get("html_url") or review.get("url") or ""),
            )
        )
    return out


def artifacts_from_review_comments(
    comments: Iterable[Mapping[str, Any]],
) -> list[ReviewArtifact]:
    out: list[ReviewArtifact] = []
    for comment in comments:
        created = parse_github_datetime(comment.get("created_at"))
        if created is None:
            continue
        out.append(
            ReviewArtifact(
                kind="review_comment",
                login=_user_login(comment) or "",
                created_at=created,
                updated_at=parse_github_datetime(comment.get("updated_at")),
                url=str(comment.get("html_url") or comment.get("url") or ""),
            )
        )
    return out


def _github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "fantasy-projections-claude-review-gate",
    }


def _request_json(url: str, token: str) -> tuple[Any, Mapping[str, str]]:
    request = urllib.request.Request(url, headers=_github_headers(token), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
            return body, headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API {exc.code} for {url}: {detail[:500]}"
        ) from exc


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        start = section.find("<")
        end = section.find(">", start + 1)
        if start >= 0 and end > start:
            return section[start + 1 : end]
    return None


def _paginate(url: str, token: str) -> list[Any]:
    items: list[Any] = []
    current: str | None = url
    while current:
        body, headers = _request_json(current, token)
        if not isinstance(body, list):
            raise RuntimeError(f"expected a JSON list from {current}")
        items.extend(body)
        current = _next_link(headers.get("link"))
    return items


def fetch_changed_files(owner: str, repo: str, pr_number: int, token: str) -> list[str]:
    url = (
        f"{GITHUB_API}/repos/{urllib.parse.quote(owner)}/"
        f"{urllib.parse.quote(repo)}/pulls/{pr_number}/files?per_page=100"
    )
    files = _paginate(url, token)
    paths: list[str] = []
    for item in files:
        if isinstance(item, Mapping) and isinstance(item.get("filename"), str):
            paths.append(item["filename"])
    return paths


def fetch_artifacts(owner: str, repo: str, pr_number: int, token: str) -> list[ReviewArtifact]:
    owner_q = urllib.parse.quote(owner)
    repo_q = urllib.parse.quote(repo)
    issue_comments = _paginate(
        f"{GITHUB_API}/repos/{owner_q}/{repo_q}/issues/{pr_number}/comments?per_page=100",
        token,
    )
    reviews = _paginate(
        f"{GITHUB_API}/repos/{owner_q}/{repo_q}/pulls/{pr_number}/reviews?per_page=100",
        token,
    )
    review_comments = _paginate(
        f"{GITHUB_API}/repos/{owner_q}/{repo_q}/pulls/{pr_number}/comments?per_page=100",
        token,
    )
    return [
        *artifacts_from_issue_comments(
            item for item in issue_comments if isinstance(item, Mapping)
        ),
        *artifacts_from_reviews(item for item in reviews if isinstance(item, Mapping)),
        *artifacts_from_review_comments(
            item for item in review_comments if isinstance(item, Mapping)
        ),
    ]


def split_repo(repo: str) -> tuple[str, str]:
    owner, _, name = repo.partition("/")
    if not owner or not name or "/" in name:
        raise ValueError(f"expected owner/repo, got {repo!r}")
    return owner, name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--since", required=True, help="UTC timestamp when the job started")
    parser.add_argument(
        "--draft",
        default="false",
        help="true if the pull request is still a draft",
    )
    parser.add_argument(
        "--token",
        default="",
        help="GitHub token; defaults to GITHUB_TOKEN or GH_TOKEN",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("GITHUB_TOKEN or GH_TOKEN is required", file=sys.stderr)
        return 1

    since = parse_since(args.since)
    owner, repo = split_repo(args.repo)
    draft = truthy_flag(args.draft)

    try:
        changed_files = fetch_changed_files(owner, repo, args.pr, token)
        artifacts = fetch_artifacts(owner, repo, args.pr, token)
    except (OSError, TimeoutError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(
            f"claude-review is fail-closed: could not list PR artifacts: {exc}",
            file=sys.stderr,
        )
        return 1

    result = evaluate_gate(
        draft=draft,
        changed_files=changed_files,
        artifacts=artifacts,
        since=since,
    )
    stream = sys.stderr if result.exit_code else sys.stdout
    print(result.message, file=stream)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
