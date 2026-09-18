"""Fail-closed Claude review gate — the #80 silent-green class."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from scripts.assert_claude_review_posted import (
    ReviewArtifact,
    artifacts_from_issue_comments,
    artifacts_from_review_comments,
    artifacts_from_reviews,
    evaluate_gate,
    explain_silent_run,
    is_claude_login,
    parse_since,
    summarize_execution_messages,
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


def _transcript(**result_overrides):
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 10,
        "total_cost_usd": 1.53,
        "result": "Stopped without posting.",
        "permission_denials": [
            {"tool_name": "Bash", "tool_input": {"command": "gh api repos/x/y/files"}},
            {"tool_name": "WebFetch", "tool_input": {"url": "https://example.com"}},
        ],
    }
    result.update(result_overrides)
    return [
        {"type": "system", "subtype": "init", "model": "claude-sonnet-5"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Checking eligibility."},
                    {"type": "tool_use", "name": "Task", "input": {"subagent_type": "haiku"}},
                ]
            },
        },
        result,
    ]


def test_summary_names_the_denied_command_and_final_text():
    lines = "\n".join(summarize_execution_messages(_transcript()))
    assert "turns=10" in lines
    assert "Stopped without posting." in lines
    assert "permission denials (2)" in lines
    assert "gh api repos/x/y/files" in lines
    assert "Task" in lines
    assert "Checking eligibility." in lines


def test_summary_does_not_echo_tool_inputs_that_could_carry_file_contents():
    # A public Actions log: only inputs that name the denied action are shown.
    lines = "\n".join(summarize_execution_messages(_transcript()))
    assert "https://example.com" not in lines


def test_summary_reports_absence_of_denials_rather_than_omitting_it():
    lines = "\n".join(summarize_execution_messages(_transcript(permission_denials=[])))
    assert "permission denials: none recorded" in lines


def test_summary_survives_a_transcript_with_no_result_message():
    lines = "\n".join(summarize_execution_messages([{"type": "system"}]))
    assert "tools called: none" in lines


def test_explain_silent_run_never_raises_on_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    assert "no execution log" in "\n".join(explain_silent_run())

    (tmp_path / "claude-execution-output.json").write_text("{not json")
    assert "could not read execution log" in "\n".join(explain_silent_run())

    (tmp_path / "claude-execution-output.json").write_text(json.dumps({"a": 1}))
    assert "unexpected execution log shape" in "\n".join(explain_silent_run())

    (tmp_path / "claude-execution-output.json").write_text(json.dumps(_transcript()))
    assert "permission denials (2)" in "\n".join(explain_silent_run())

    monkeypatch.delenv("RUNNER_TEMP")
    assert "not running on a GitHub runner" in "\n".join(explain_silent_run())
