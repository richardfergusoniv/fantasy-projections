from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from unittest import mock

import pandas as pd

from src.sentiment.gate import audit_snapshots
from src.sentiment.markdown import (
    RESEARCH_AS_OF,
    TEAM_RESEARCH_FILES,
    parse_research_directory,
    score_sentiment,
)
from src.sentiment.snapshot import (
    SENTIMENT_OUTPUT_COLS,
    attach_sentiment,
    build_sentiment_snapshot,
)
from src.sentiment.refresh_outputs import refresh_outputs


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_research_registry_covers_all_32_teams():
    assert len(TEAM_RESEARCH_FILES) == 32
    assert len(set(TEAM_RESEARCH_FILES.values())) == 32
    root = REPO_ROOT / "perplexity research"
    assert all((root / filename).exists() for filename in TEAM_RESEARCH_FILES.values())


def test_objective_injury_negative_is_not_sentiment():
    assert score_sentiment("Strongly bearish", context="Repeat hamstring injury") is None
    assert score_sentiment("Negative", context="Waived/injured on August 24") is None
    assert score_sentiment("Elite positive, injury caution", context="Role remains central") == 1.0
    assert score_sentiment("Neutral") == 0.0


def test_section_heading_does_not_override_an_explicit_label():
    """A "## Top positive signals" heading must not flip a bearish row positive.

    The heading is a fallback for rows that carry no label of their own. Scoring
    label and heading together matched the positive tier on the heading's wording
    and returned before the negative branch, so an explicit "Strongly bearish"
    row scored +0.55. Headings of this form appear in 18+ of the 32 summaries.
    """
    assert score_sentiment("Strongly bearish", "Top positive signals") == -0.9
    assert score_sentiment("Bearish", "Strongest positives") == -0.55
    assert score_sentiment("Neutral", "Strongest positive signals") == 0.0
    # ... while an unlabelled row still inherits the heading as its polarity.
    assert score_sentiment("Jonah Coleman", "Strongest positives") == 0.55
    assert score_sentiment("someone", "Strongly bearish outlook") == -0.9
    # ... and a qualifying heading still tempers a positive row beneath it.
    assert score_sentiment("Positive", "Positive but role-dependent") == 0.3
    assert score_sentiment("Positive", "Positive, but conditional") == 0.3
    assert score_sentiment("Positive", "Strongest positives") == 0.55


def test_markdown_parser_finds_signal_for_every_team():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv")
    players = players.drop_duplicates("player_id")
    parsed = parse_research_directory(
        players,
        REPO_ROOT / "perplexity research",
        as_of=RESEARCH_AS_OF,
    )
    assert set(parsed["team"]) == set(TEAM_RESEARCH_FILES)
    assert parsed.groupby("team")["player_id"].nunique().min() >= 1
    assert parsed["player_id"].is_unique


def test_snapshot_has_explicit_missing_rows_and_inactive_gate():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv")
    players = players.drop_duplicates("player_id")
    snapshot = build_sentiment_snapshot(players, season=2026, as_of="2026-08-24")
    assert len(snapshot) == len(players)
    assert snapshot["player_id"].is_unique
    assert snapshot["team"].nunique() == 32
    assert snapshot["sentiment_score"].notna().sum() >= 300
    missing = snapshot[snapshot["sentiment_coverage"].eq("none")]
    assert len(missing) > 0
    assert missing["sentiment_score"].isna().all()
    assert not snapshot["sentiment_model_active"].any()


def test_pre_cutoff_snapshot_does_not_use_future_markdown(tmp_path):
    players = pd.DataFrame(
        [{"player_id": "p1", "display_name": "Josh Allen", "team": "BUF", "position": "QB"}]
    )
    snapshot = build_sentiment_snapshot(
        players,
        season=2026,
        as_of=date(2026, 8, 23),
        consensus_dir=tmp_path,
        manifest_path=tmp_path / "missing.json",
    )
    row = snapshot.iloc[0]
    assert row["sentiment_coverage"] == "none"
    assert pd.isna(row["sentiment_score"])


def test_attach_sentiment_is_many_to_one():
    frame = pd.DataFrame(
        [
            {"player_id": "p1", "display_name": "Unknown One", "team": "BUF", "position": "QB", "stat": "attempts"},
            {"player_id": "p1", "display_name": "Unknown One", "team": "BUF", "position": "QB", "stat": "passing_yards"},
        ]
    )
    attached = attach_sentiment(frame, season=2026, as_of="2026-08-23")
    assert len(attached) == 2
    assert set(SENTIMENT_OUTPUT_COLS) <= set(attached.columns)
    assert attached["sentiment_coverage"].eq("none").all()


def test_gate_requires_three_seasons(tmp_path):
    path = tmp_path / "sentiment_2026_2026-08-24.csv"
    pd.DataFrame(
        [
            {
                "player_id": f"p{i}",
                "position": "WR",
                "sentiment_feature": 0.5,
                "sentiment_as_of": "2026-08-24",
            }
            for i in range(250)
        ]
    ).to_csv(path, index=False)
    report = audit_snapshots([path])
    assert not report["ready_for_ablation"]
    assert not report["by_position"]["WR"]["prerequisites_met"]


def test_refresh_outputs_preserves_projection_values(tmp_path):
    source = pd.read_csv(REPO_ROOT / "output" / "projections_2026.csv")
    sample_ids = source["player_id"].drop_duplicates().head(12)
    source = source[source["player_id"].isin(sample_ids)].copy()
    before = source[["player_id", "stat", "pred_pg", "pred_season"]].copy()
    projection_path = tmp_path / "projections.csv"
    fantasy_path = tmp_path / "fantasy.csv"
    manifest_path = tmp_path / "projection_run_2026.json"
    source.to_csv(projection_path, index=False)
    manifest_path.write_text(json.dumps({
        "files": {"projections": {}, "fantasy_points": {}}
    }), encoding="utf-8")
    # Contract rejection has dedicated tests; this test isolates the promise
    # that an accepted sentiment-only refresh cannot alter numeric forecasts.
    with mock.patch("src.sentiment.refresh_outputs.validate_projection_contract"):
        refresh_outputs(
            season=2026,
            as_of="2026-08-24",
            projections_path=projection_path,
            fantasy_path=fantasy_path,
            manifest_path=manifest_path,
        )
    after = pd.read_csv(projection_path)
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        after[["player_id", "stat", "pred_pg", "pred_season"]].reset_index(drop=True),
        check_dtype=False,
    )
    assert set(SENTIMENT_OUTPUT_COLS) <= set(after.columns)
    assert fantasy_path.exists()
