from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from unittest import mock

import pandas as pd

from src.sentiment.diagnostics import (
    EVIDENCE_LEGACY_ONLY,
    EVIDENCE_MARKET_ONLY,
    EVIDENCE_NONE,
    attach_diagnostic_labels,
    sentiment_evidence_tier,
)
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
from src.draft_assistant.prepare import build_sentiment_meta


from src.sentiment.refresh_outputs import refresh_outputs


REPO_ROOT = Path(__file__).resolve().parents[1]
SLEEPERS_DIR = REPO_ROOT / "draft_assistant" / "sleepers"


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


def test_positive_label_never_renders_bearish():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    snapshot = build_sentiment_snapshot(players, season=2026, as_of="2026-08-24")
    positive = snapshot[snapshot["text_sentiment_raw"] > 0]
    assert not (positive["sentiment_tone"] == "bearish").any()
    below = positive[positive["sentiment_peer_label"] == "below_peers"]
    assert len(below) >= 69
    assert (below["sentiment_tone"] == "bullish").all()


def test_peer_label_is_independent_of_tone():
    frame = pd.DataFrame(
        [
            {
                "text_sentiment_raw": 0.8,
                "sentiment_score": -50.0,
                "text_sentiment_z": 1.0,
                "market_gap_z": None,
            }
        ]
    )
    labeled = attach_diagnostic_labels(frame).iloc[0]
    assert labeled["sentiment_tone"] == "bullish"
    assert labeled["sentiment_peer_label"] == "below_peers"


def test_market_only_evidence_tier():
    frame = pd.DataFrame(
        [{"text_sentiment_z": None, "market_gap_z": 0.5, "text_sentiment_raw": None, "sentiment_score": -10.0}]
    )
    labeled = attach_diagnostic_labels(frame).iloc[0]
    assert labeled["sentiment_evidence_tier"] == EVIDENCE_MARKET_ONLY
    assert labeled["sentiment_tone"] == "unavailable"


def test_evidence_tier_covers_all_four_combinations():
    assert sentiment_evidence_tier(has_text=True, has_market=True) == "legacy_plus_market"
    assert sentiment_evidence_tier(has_text=True, has_market=False) == EVIDENCE_LEGACY_ONLY
    assert sentiment_evidence_tier(has_text=False, has_market=True) == EVIDENCE_MARKET_ONLY
    assert sentiment_evidence_tier(has_text=False, has_market=False) == EVIDENCE_NONE


def test_sentiment_meta_block_matches_consensus_metadata():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    snapshot = build_sentiment_snapshot(players, season=2026, as_of="2026-08-24")
    generated_at = "2026-08-30T12:00:00+00:00"
    meta = build_sentiment_meta(2026, snapshot, generated_at=generated_at)
    consensus = json.loads((REPO_ROOT / "data" / "consensus" / "consensus_2026.json").read_text())
    assert meta["ecr_date"] == consensus["meta"]["ecr"]["scrape_date"]
    assert meta["adp_end_date"] == consensus["meta"]["adp"]["end_date"]
    assert meta["status"] == "diagnostic"
    assert meta["model_active"] is False
    assert meta["generated_at"] == generated_at


def test_release_artifact_carries_diagnostic_fields():
    path = REPO_ROOT / "draft_assistant" / "data" / "players_2026.json"
    assert path.exists(), "run prepare export before this test"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "sentiment" in payload["meta"]
    sample = next(p for p in payload["players"] if p.get("sentiment_tone"))
    for field in ("sentiment_tone", "sentiment_peer_label", "sentiment_evidence_tier"):
        assert field in sample


def test_sleeper_diagnostic_columns_are_not_sortable():
    html = (SLEEPERS_DIR / "index.html").read_text(encoding="utf-8")
    for label in ("Tone", "Peer buzz", "Evidence"):
        start = html.index(label)
        row = html.rfind("<th", 0, start)
        end = html.index("</th>", start)
        header = html[row:end]
        assert 'class="sortable' not in header
        assert "data-sort" not in header
    assert 'data-sort="sentiment' not in html


def test_sleeper_colspans_match_header_count():
    html = (SLEEPERS_DIR / "index.html").read_text(encoding="utf-8")
    import re

    header_count = len(re.findall(r"<th[\s>]", html))
    app_js = (SLEEPERS_DIR / "js" / "app.js").read_text(encoding="utf-8")
    colspans = [int(value) for value in re.findall(r'colspan="(\d+)"', app_js)]
    assert colspans
    assert all(value == header_count for value in colspans)


def test_no_hardcoded_research_date_in_browser_assets():
    for path in SLEEPERS_DIR.rglob("*"):
        if path.suffix in {".js", ".html", ".css"}:
            assert "2026-08-24" not in path.read_text(encoding="utf-8"), path


def test_diagnostic_fields_do_not_change_projections():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    cols = ["fantasy_pts_season", "fantasy_pts", "position", "team"]
    before = players.set_index("player_id")[cols]
    attached = attach_sentiment(players, season=2026, as_of="2026-08-24")
    after = attached.set_index("player_id")[cols]
    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def _write_snapshot(tmp_path, rows: list[dict], name: str) -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_gate_rejects_duplicate_season_snapshots(tmp_path):
    row = {
        "player_id": "p1",
        "position": "WR",
        "sentiment_as_of": "2026-08-24",
    }
    first = _write_snapshot(tmp_path, [row], "sentiment_2026_2026-08-24.csv")
    second = _write_snapshot(tmp_path, [row], "sentiment_2026_2026-08-25.csv")
    try:
        audit_snapshots([first, second])
        assert False, "expected duplicate season rejection"
    except ValueError as exc:
        assert "Duplicate snapshot" in str(exc)


def test_gate_three_claims_count_once_in_numerator(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    claims = [
        {
            "season": 2026,
            "position": "WR",
            "player_id": "p1",
            "training_eligible": True,
            "evidence_tier": "verified",
        }
        for _ in range(3)
    ]
    claims_path.write_text("\n".join(json.dumps(c) for c in claims), encoding="utf-8")
    snap = _write_snapshot(
        tmp_path,
        [{"player_id": "p1", "position": "WR", "sentiment_as_of": "2026-08-24"}],
        "sentiment_2026_2026-08-24.csv",
    )
    report = audit_snapshots([snap], claim_paths=[claims_path])
    assert report["by_position"]["WR"]["covered_player_seasons"] == 1


def test_gate_position_with_one_season_fails_even_if_others_have_three(tmp_path):
    snap_2026 = _write_snapshot(
        tmp_path,
        [{"player_id": "p1", "position": "WR", "sentiment_as_of": "2026-08-24"}],
        "sentiment_2026_2026-08-24.csv",
    )
    for season in (2024, 2025):
        _write_snapshot(
            tmp_path,
            [{"player_id": f"qb{season}", "position": "QB", "sentiment_as_of": f"{season}-08-24"}],
            f"sentiment_{season}_{season}-08-24.csv",
        )
    report = audit_snapshots(
        [
            snap_2026,
            tmp_path / "sentiment_2024_2024-08-24.csv",
            tmp_path / "sentiment_2025_2025-08-24.csv",
        ]
    )
    assert len(report["by_position"]["WR"]["seasons"]) == 1
    assert not report["by_position"]["WR"]["prerequisites_met"]


def test_gate_coverage_exactly_40_percent_passes(tmp_path):
    eligible = 10
    covered = 4
    rows = [{"player_id": f"p{i}", "position": "WR", "sentiment_as_of": "2026-08-24"} for i in range(eligible)]
    snap = _write_snapshot(tmp_path, rows, "sentiment_2026_2026-08-24.csv")
    claims_path = tmp_path / "claims.jsonl"
    claim_rows = [
        {
            "season": 2026,
            "position": "WR",
            "player_id": f"p{i}",
            "training_eligible": True,
            "evidence_tier": "verified",
        }
        for i in range(covered)
    ]
    claims_path.write_text("\n".join(json.dumps(c) for c in claim_rows), encoding="utf-8")
    report = audit_snapshots([snap], claim_paths=[claims_path])
    assert report["by_position"]["WR"]["coverage"] == 0.4
    assert report["by_position"]["WR"]["prerequisites_met"] is False  # still needs seasons + count


def test_frozen_wr_population_has_105_records():
    from scripts.freeze_sentiment_spike_population import EXPECTED_WR_COUNT, freeze_population

    payload = freeze_population(2024, "WR")
    assert payload["denominator"] == EXPECTED_WR_COUNT == 105


def test_spike_validator_rejects_post_cutoff_and_missing_hash(tmp_path):
    from scripts.freeze_sentiment_spike_population import SPIKE_CUTOFF_2024, freeze_population
    from scripts.validate_spike_claims import validate_claim

    population_path = tmp_path / "population.json"
    population_path.write_text(json.dumps(freeze_population(2024, "WR")), encoding="utf-8")
    allowed = {p["player_id"] for p in json.loads(population_path.read_text())["players"]}
    player_id = next(iter(allowed))
    cutoff = __import__("datetime").datetime.fromisoformat(SPIKE_CUTOFF_2024.replace("Z", "+00:00"))
    late = {
        "player_id": player_id,
        "evidence_tier": "verified",
        "training_eligible": True,
        "source_url": "https://example.com/a",
        "excerpt": "Coach praised his route development role",
        "publication_timestamp": "2024-12-01T00:00:00Z",
        "captured_content_hash": "abc",
        "reviewer": "rf",
        "parsed_label": "positive",
    }
    assert "after spike cutoff" in validate_claim(late, allowed_players=allowed, cutoff=cutoff)[0]
    missing_hash = {**late, "publication_timestamp": "2024-08-01T00:00:00Z", "captured_content_hash": ""}
    assert any("captured_content_hash" in err for err in validate_claim(missing_hash, allowed_players=allowed, cutoff=cutoff))


def test_spike_report_empty_claims_is_infeasible(tmp_path):
    from scripts.freeze_sentiment_spike_population import freeze_population
    from scripts.report_sentiment_spike import VERDICT_INFEASIBLE, build_report

    population_path = tmp_path / "population.json"
    population_path.write_text(json.dumps(freeze_population(2024, "WR")), encoding="utf-8")
    report = build_report(population_path=population_path, claims_path=None, attempts_path=None)
    assert report["verified_numerator"] == 0
    assert report["frozen_denominator"] == 105
    assert report["verdict"] == VERDICT_INFEASIBLE


def test_patch_release_adds_diagnostic_fields_to_players_payload():
    from src.sentiment.patch_release import patch_players_payload

    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    ).head(20)
    payload = {
        "meta": {"generated_at": "2026-08-30T12:00:00+00:00", "season": 2026},
        "players": [{"player_id": row.player_id, "display_name": row.display_name} for row in players.itertuples()],
    }
    patched = patch_players_payload(payload, season=2026, fantasy_points=players)
    assert "sentiment" in patched["meta"]
    sample = next(p for p in patched["players"] if p.get("sentiment_tone"))
    for field in ("sentiment_tone", "sentiment_peer_label", "sentiment_evidence_tier"):
        assert field in sample
