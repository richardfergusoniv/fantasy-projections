from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.sentiment.ledger import (
    DAILY_IMPORTER_VERSION,
    DAILY_RESEARCH_FILES,
    EVIDENCE_TIER_LEGACY,
    assert_training_eligible_allowed,
    build_legacy_claim,
    import_legacy_ledger,
    import_legacy_daily_ledger,
    write_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_registered_files_appear_in_ledger_source_file():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    records = import_legacy_ledger(players, season=2026)
    source_files = {row["source_file"] for row in records}
    from src.sentiment.markdown import TEAM_RESEARCH_FILES

    assert source_files == set(TEAM_RESEARCH_FILES.values())


def test_reimport_is_byte_identical(tmp_path):
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    first = import_legacy_ledger(players, season=2026)
    out_a = tmp_path / "legacy_a.jsonl"
    out_b = tmp_path / "legacy_b.jsonl"
    write_ledger(first, out_a)
    second = import_legacy_ledger(players, season=2026)
    write_ledger(second, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()
    assert {row["claim_id"] for row in first} == {row["claim_id"] for row in second}
    assert {row["content_hash"] for row in first} == {row["content_hash"] for row in second}
    assert len(first) == len(second)


def test_no_legacy_row_is_training_eligible():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    records = import_legacy_ledger(players, season=2026)
    assert all(row["training_eligible"] is False for row in records)
    with pytest.raises(ValueError, match="cannot be training-eligible"):
        assert_training_eligible_allowed(
            {"evidence_tier": EVIDENCE_TIER_LEGACY, "training_eligible": True}
        )


def test_multiple_mentions_produce_multiple_rows():
    players = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "Test Player",
                "team": "BUF",
                "position": "WR",
            }
        ]
    )
    claim_a = {
        "player_id": "p1",
        "display_name": "Test Player",
        "team": "BUF",
        "position": "WR",
        "source_file": "bills.md",
        "line_number": 10,
        "parsed_label": "Positive",
        "polarity": 0.55,
        "context": "Test Player: positive outlook",
        "extraction_method": "bullet",
    }
    claim_b = {
        **claim_a,
        "line_number": 20,
        "parsed_label": "Bearish",
        "polarity": -0.55,
        "context": "Test Player: bearish outlook",
    }
    rows = [build_legacy_claim(season=2026, claim=claim_a), build_legacy_claim(season=2026, claim=claim_b)]
    assert len(rows) == 2
    assert rows[0]["claim_id"] != rows[1]["claim_id"]


def test_source_url_and_publication_date_are_explicit_nulls():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    records = import_legacy_ledger(players, season=2026)
    for row in records:
        assert row["source_url"] is None
        assert row["publication_date"] is None


def test_audit_expectations_match_current_corpus():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    records = import_legacy_ledger(players, season=2026)
    frame = pd.DataFrame(records)
    by_player = frame.groupby("player_id").size()
    assert frame["player_id"].nunique() == 402
    assert int((by_player == 1).sum()) == 392
    assert int((by_player > 1).sum()) == 10
    assert frame["source_file"].nunique() == 32
    assert int((frame["extraction_method"] == "table").sum()) == 266
    assert int((frame["extraction_method"] == "bullet").sum()) == 146


def test_daily_registry_covers_august_26_through_september_4():
    assert list(DAILY_RESEARCH_FILES) == [
        "2026-08-26",
        "2026-08-27",
        "2026-08-28",
        "2026-08-29",
        "2026-08-30",
        "2026-08-31",
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
        "2026-09-04",
    ]
    root = REPO_ROOT / "perplexity research" / "daily"
    assert all((root / filename).exists() for filename in DAILY_RESEARCH_FILES.values())


def test_daily_reports_import_as_training_ineligible_legacy_evidence(tmp_path):
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    first = import_legacy_daily_ledger(players, season=2026)
    second = import_legacy_daily_ledger(players, season=2026)
    assert first
    assert first == second
    assert {row["research_cutoff"] for row in first} == set(DAILY_RESEARCH_FILES)
    assert {row["importer_version"] for row in first} == {DAILY_IMPORTER_VERSION}
    assert all(row["evidence_tier"] == EVIDENCE_TIER_LEGACY for row in first)
    assert all(row["training_eligible"] is False for row in first)
    assert all(row["source_url"] is None for row in first)
    assert all(row["publication_date"] is None for row in first)

    out_a = tmp_path / "daily_a.jsonl"
    out_b = tmp_path / "daily_b.jsonl"
    write_ledger(first, out_a)
    write_ledger(second, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()


def test_daily_import_contains_known_directional_mentions():
    players = pd.read_csv(REPO_ROOT / "output" / "fantasy_points_2026.csv").drop_duplicates(
        "player_id"
    )
    records = import_legacy_daily_ledger(players, season=2026)
    by_name = {}
    for row in records:
        by_name.setdefault(row["display_name"], []).append(row)
    assert any(row["polarity"] > 0 for row in by_name["Breece Hall"])
    assert any(row["polarity"] > 0 for row in by_name["D'Andre Swift"])
    assert any(row["polarity"] < 0 for row in by_name["Rome Odunze"])
