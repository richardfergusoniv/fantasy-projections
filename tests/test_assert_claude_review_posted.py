"""Fail-closed Claude review gate — the #80 silent-green class."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.assert_claude_review_posted import (
    ReviewArtifact,
    artifacts_from_issue_comments,
    artifacts_from_review_comments,
    artifacts_from_reviews,
    evaluate_gate,
    is_claude_login,
    parse_since,
)

SINCE = datetime(2026, 9, 16, 16, 10, 24, tzinfo=timezone.utc)


def _artifact(
    *,
    login: str = "claude[bot]",
    kind: str = "issue_comment",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    url: str = "https://github.com/example/comment/1",
) -> ReviewArtifact:
    stamp = created_at or SINCE + timedelta(seconds=40)
    return ReviewArtifact(
        kind=kind,
        login=login,
        created_at=stamp,
        updated_at=updated_at,
        url=url,
    )


def test_ready_tooling_pr_with_no_claude_output_fails():
    """PR #80 shape: ready, not a workflow-file PR, zero claude[bot] artifacts."""
    result = evaluate_gate(
        draft=False,
        changed_files=[
            ".compound-engineering/config.example.yaml",
            ".compound-engineering/config.yaml",
            ".gitignore",
            "AGENTS.md",
        ],
        artifacts=[],
        since=SINCE,
    )
    assert result.status == "fail"
    assert result.exit_code == 1
    assert "fail-closed" in result.message


def test_draft_silence_is_expected():
    result = evaluate_gate(
        draft=True,
        changed_files=["src/app.py"],
        artifacts=[],
        since=SINCE,
    )
    assert result.status == "skip_draft"
    assert result.exit_code == 0


def test_workflow_file_change_is_waived():
    result = evaluate_gate(
        draft=False,
        changed_files=[".github/workflows/claude-code-review.yml"],
        artifacts=[],
        since=SINCE,
    )
    assert result.status == "skip_workflow_validation"
    assert result.exit_code == 0


def test_claude_issue_comment_from_this_run_passes():
    result = evaluate_gate(
        draft=False,
        changed_files=["AGENTS.md"],
        artifacts=[_artifact()],
        since=SINCE,
    )
    assert result.status == "pass"
    assert result.exit_code == 0


def test_claude_review_from_this_run_passes():
    result = evaluate_gate(
        draft=False,
        changed_files=["src/app.py"],
        artifacts=[
            _artifact(
                kind="review",
                url="https://github.com/example/pull/80#pullrequestreview-1",
            )
        ],
        since=SINCE,
    )
    assert result.status == "pass"


def test_claude_inline_review_comment_from_this_run_passes():
    result = evaluate_gate(
        draft=False,
        changed_files=["src/app.py"],
        artifacts=[
            _artifact(
                kind="review_comment",
                url="https://github.com/example/pull/80#discussion_r1",
            )
        ],
        since=SINCE,
    )
    assert result.status == "pass"


def test_vercel_comment_does_not_count():
    result = evaluate_gate(
        draft=False,
        changed_files=["AGENTS.md"],
        artifacts=[_artifact(login="vercel[bot]")],
        since=SINCE,
    )
    assert result.status == "fail"


def test_prior_run_claude_comment_does_not_count():
    stale = _artifact(created_at=SINCE - timedelta(hours=2))
    result = evaluate_gate(
        draft=False,
        changed_files=["AGENTS.md"],
        artifacts=[stale],
        since=SINCE,
    )
    assert result.status == "fail"


def test_comment_just_before_job_start_counts_within_clock_slack():
    almost = _artifact(created_at=SINCE - timedelta(seconds=30))
    result = evaluate_gate(
        draft=False,
        changed_files=["AGENTS.md"],
        artifacts=[almost],
        since=SINCE,
    )
    assert result.status == "pass"


def test_updated_sticky_comment_from_this_run_counts():
    sticky = _artifact(
        created_at=SINCE - timedelta(hours=3),
        updated_at=SINCE + timedelta(seconds=20),
    )
    result = evaluate_gate(
        draft=False,
        changed_files=["AGENTS.md"],
        artifacts=[sticky],
        since=SINCE,
    )
    assert result.status == "pass"


def test_is_claude_login_accepts_bot_suffix():
    assert is_claude_login("claude[bot]")
    assert is_claude_login("claude")
    assert not is_claude_login("cursor[bot]")
    assert not is_claude_login(None)


def test_main_fails_closed_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    from scripts.assert_claude_review_posted import main

    assert (
        main(
            [
                "--repo",
                "richardfergusoniv/fantasy-projections",
                "--pr",
                "80",
                "--since",
                "2026-09-16T16:10:24Z",
                "--draft",
                "false",
            ]
        )
        == 1
    )


def test_main_ready_pr_without_claude_output_exits_1(monkeypatch):
    from scripts import assert_claude_review_posted as gate

    monkeypatch.setattr(
        gate,
        "fetch_changed_files",
        lambda *args, **kwargs: ["AGENTS.md"],
    )
    monkeypatch.setattr(gate, "fetch_artifacts", lambda *args, **kwargs: [])
    assert (
        gate.main(
            [
                "--repo",
                "richardfergusoniv/fantasy-projections",
                "--pr",
                "80",
                "--since",
                "2026-09-16T16:10:24Z",
                "--draft",
                "false",
                "--token",
                "ghs_test",
            ]
        )
        == 1
    )


def test_payload_parsers_extract_claude_bot():
    comments = artifacts_from_issue_comments(
        [
            {
                "user": {"login": "claude[bot]"},
                "created_at": "2026-09-16T16:11:00Z",
                "html_url": "https://example/comment",
            }
        ]
    )
    reviews = artifacts_from_reviews(
        [
            {
                "user": {"login": "claude[bot]"},
                "submitted_at": "2026-09-16T16:11:00Z",
                "html_url": "https://example/review",
            }
        ]
    )
    inlines = artifacts_from_review_comments(
        [
            {
                "user": {"login": "claude[bot]"},
                "created_at": "2026-09-16T16:11:00Z",
                "html_url": "https://example/inline",
            }
        ]
    )
    assert comments[0].kind == "issue_comment"
    assert reviews[0].kind == "review"
    assert inlines[0].kind == "review_comment"
    assert parse_since("2026-09-16T16:10:24Z") == SINCE
