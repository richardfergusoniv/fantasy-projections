"""Fixture-based tests for Role 2 live props export (no live S3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.providers.base import FixturePropProvider
from src.projection.weekly_eval.live_export import (
    build_identity_map_from_records,
    canonicalize_role2_market,
    export_role2_snapshot_rows,
    is_gsis_shaped,
    load_schedule_kickoffs,
    missing_live_export_env,
    rows_to_frame,
    write_live_snapshots_csv,
)
from src.projection.weekly_eval.schema import load_prop_snapshots

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "data" / "props" / "fixtures" / "providers"
SCHEDULE = (
    REPO_ROOT
    / "src"
    / "projection"
    / "weekly_latent"
    / "fixtures"
    / "nfl_schedules_2026_reg.csv"
)


def _quote(
    *,
    source: str = "draftkings",
    player_id: str | None = "00-0034857",
    player_name: str = "Josh Allen",
    team: str = "BUF",
    market: str = "pass_yards",
    line: float = 265.5,
    fetched_at: str = "2026-09-16T18:00:00+00:00",
    event_start: str | None = "2026-09-17T20:15:00+00:00",
) -> NormalizedQuote:
    return NormalizedQuote(
        source=source,
        sportsbook=source,
        event_id="e1",
        game_id="g1",
        player_name_raw=player_name,
        player_id=player_id,
        team=team,
        opponent="DET",
        market=market,
        period="full_game",
        line=line,
        over_odds=-110,
        under_odds=-110,
        fetched_at=datetime.fromisoformat(fetched_at),
        event_start=(
            None if event_start is None else datetime.fromisoformat(event_start)
        ),
        source_url=None,
        raw_record_hash=f"{source}:{player_id}:{market}:{line}",
        over_prob_novig=0.5,
        under_prob_novig=0.5,
        kind="book",
    )


def _snap(
    quotes: list[NormalizedQuote],
    *,
    source: str = "draftkings",
    season: int = 2026,
    week: int = 2,
    fetched_at: str = "2026-09-16T18:00:00+00:00",
) -> ProviderSnapshot:
    return ProviderSnapshot(
        source=source,
        season=season,
        week=week,
        fetched_at=datetime.fromisoformat(fetched_at),
        urls=(),
        quotes=tuple(quotes),
        success=True,
    )


def test_canonicalize_role2_market_aliases():
    assert canonicalize_role2_market("passing_yards") == "pass_yards"
    assert canonicalize_role2_market("receiving_yards") == "rec_yards"
    assert canonicalize_role2_market("carries") == "rush_attempts"
    assert canonicalize_role2_market("bogus_market") is None


def test_is_gsis_shaped():
    assert is_gsis_shaped("00-0034857")
    assert not is_gsis_shaped("sleeper-123")
    assert not is_gsis_shaped(None)


def test_export_drops_post_kickoff_and_non_gsis():
    kickoffs = {
        "BUF": datetime(2026, 9, 17, 20, 15, tzinfo=UTC),
    }
    snaps = [
        _snap(
            [
                _quote(fetched_at="2026-09-16T18:00:00+00:00"),  # ok
                _quote(
                    fetched_at="2026-09-17T21:00:00+00:00",  # post-kickoff
                    line=270.5,
                ),
                _quote(
                    player_id="sleeper-999",
                    player_name="Unknown Backup",
                    line=10.5,
                    market="rush_yards",
                ),
            ]
        )
    ]
    rows = export_role2_snapshot_rows(
        snaps, kickoffs_by_team=kickoffs, mode="consensus", require_gsis=True
    )
    assert len(rows) == 1
    assert rows[0]["player_id"] == "00-0034857"
    assert rows[0]["market"] == "pass_yards"
    assert rows[0]["line"] == 265.5
    assert rows[0]["as_of"] <= rows[0]["kickoff_at"]


def test_export_consensus_merges_books(tmp_path):
    kickoffs = {
        "BUF": datetime(2026, 9, 17, 20, 15, tzinfo=UTC),
    }
    snaps = [
        _snap([_quote(source="draftkings", line=265.5)], source="draftkings"),
        _snap([_quote(source="fanduel", line=267.5)], source="fanduel"),
    ]
    rows = export_role2_snapshot_rows(
        snaps, kickoffs_by_team=kickoffs, mode="consensus"
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "consensus"
    assert rows[0]["line"] == 266.5  # median of two
    out = write_live_snapshots_csv(rows, tmp_path / "live_snapshots.csv")
    loaded = load_prop_snapshots(out)
    assert len(loaded) == 1
    assert loaded[0].player_id == "00-0034857"


def test_export_single_keeps_source_column():
    kickoffs = {"BUF": datetime(2026, 9, 17, 20, 15, tzinfo=UTC)}
    snaps = [
        _snap([_quote(source="draftkings", line=265.5)], source="draftkings"),
        _snap([_quote(source="fanduel", line=267.5)], source="fanduel"),
    ]
    rows = export_role2_snapshot_rows(
        snaps, kickoffs_by_team=kickoffs, mode="single"
    )
    assert len(rows) == 2
    sources = {r["source"] for r in rows}
    assert sources == {"draftkings", "fanduel"}


def test_identity_map_resolves_name_to_gsis():
    identity = build_identity_map_from_records(
        [
            {
                "player_id": "00-0034857",
                "gsis_id": "00-0034857",
                "name": "Josh Allen",
                "team": "BUF",
                "position": "QB",
            }
        ]
    )
    kickoffs = {"BUF": datetime(2026, 9, 17, 20, 15, tzinfo=UTC)}
    snaps = [
        _snap(
            [
                _quote(
                    player_id=None,
                    player_name="Josh Allen",
                    team="BUF",
                    line=260.5,
                )
            ]
        )
    ]
    rows = export_role2_snapshot_rows(
        snaps, identity_map=identity, kickoffs_by_team=kickoffs
    )
    assert len(rows) == 1
    assert rows[0]["player_id"] == "00-0034857"


def test_schedule_kickoffs_week2():
    kickoffs = load_schedule_kickoffs(SCHEDULE, season=2026, week=2)
    assert "BUF" in kickoffs
    assert kickoffs["BUF"].year == 2026
    assert kickoffs["DET"] == kickoffs["BUF"]


def test_fixture_providers_export_passes_schema(tmp_path):
    """Committed DK/FD fixtures already carry gsis ids — export must schema-validate."""
    dk = FixturePropProvider(
        "draftkings", FIXTURE_DIR / "draftkings.json"
    ).fetch(season=2026, week=1)
    fd = FixturePropProvider("fanduel", FIXTURE_DIR / "fanduel.json").fetch(
        season=2026, week=1
    )
    assert dk.success and fd.success
    # Fixture event_start is in 2099; use schedule week-1 kickoffs instead.
    kickoffs = load_schedule_kickoffs(SCHEDULE, season=2026, week=1)
    # Shift fetched_at onto a pre-kickoff stamp so schedule kickoffs apply.
    from dataclasses import replace

    def _retime(snap: ProviderSnapshot) -> ProviderSnapshot:
        fetched = datetime(2026, 9, 8, 18, 0, tzinfo=UTC)
        quotes = tuple(replace(q, fetched_at=fetched, event_start=None) for q in snap.quotes)
        return replace(snap, fetched_at=fetched, quotes=quotes, week=1)

    rows = export_role2_snapshot_rows(
        [_retime(dk), _retime(fd)],
        kickoffs_by_team=kickoffs,
        mode="consensus",
    )
    assert len(rows) > 0
    assert all(is_gsis_shaped(r["player_id"]) for r in rows)
    assert all(r["as_of"] <= r["kickoff_at"] for r in rows)
    out = write_live_snapshots_csv(rows, tmp_path / "live_snapshots.csv")
    loaded = load_prop_snapshots(out)
    assert len(loaded) == len(rows)


def test_missing_live_export_env_reports_keys(monkeypatch):
    for key in (
        "DATABASE_URL",
        "JOB_DATABASE_URL",
        "ARTIFACT_BACKEND",
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    missing = missing_live_export_env()
    assert "DATABASE_URL|JOB_DATABASE_URL" in missing
    assert "ARTIFACT_BACKEND=s3" in missing
    assert "S3_ACCESS_KEY_ID" in missing


def test_rows_to_frame_empty_has_columns():
    frame = rows_to_frame([])
    assert list(frame.columns) == [
        "player_id",
        "player_name",
        "season",
        "week",
        "market",
        "line",
        "implied_p_over",
        "implied_mean",
        "as_of",
        "kickoff_at",
        "source",
    ]


def test_cli_snapshot_json(tmp_path):
    import json
    import runpy

    out = tmp_path / "live_snapshots.csv"
    snap_path = tmp_path / "dk.json"
    payload = ProviderSnapshot(
        source="draftkings",
        season=2026,
        week=2,
        fetched_at=datetime(2026, 9, 16, 18, 0, tzinfo=UTC),
        urls=(),
        quotes=(
            _quote(fetched_at="2026-09-16T18:00:00+00:00", event_start=None),
        ),
        success=True,
    ).to_dict()
    snap_path.write_text(json.dumps(payload), encoding="utf-8")
    mod = runpy.run_path(str(REPO_ROOT / "scripts" / "export_role2_live_props.py"))
    rc = mod["main"](
        [
            "--season",
            "2026",
            "--week",
            "2",
            "--snapshot-json",
            str(snap_path),
            "--schedule",
            str(SCHEDULE),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.is_file()
    loaded = load_prop_snapshots(out)
    assert len(loaded) == 1
    assert loaded[0].player_id == "00-0034857"


def test_cli_check_env_fails_without_creds(monkeypatch):
    import runpy

    for key in (
        "DATABASE_URL",
        "JOB_DATABASE_URL",
        "ARTIFACT_BACKEND",
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    mod = runpy.run_path(str(REPO_ROOT / "scripts" / "export_role2_live_props.py"))
    assert mod["main"](["--check-env"]) == 1
