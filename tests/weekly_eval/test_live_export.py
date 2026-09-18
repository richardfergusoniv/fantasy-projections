"""Fixture-based tests for Role 2 live props export (no live S3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.providers.base import FixturePropProvider
from src.projection.weekly_eval.live_export import (
    build_identity_map_from_records,
    build_identity_map_from_records_with_stats,
    canonicalize_role2_market,
    canonicalize_team_abbrev,
    eastern_wallclock_to_utc,
    export_role2_snapshot_rows,
    is_gsis_shaped,
    kickoffs_by_team_from_schedule,
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
ET = ZoneInfo("America/New_York")


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
    over_prob_novig: float | None = 0.5,
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
        raw_record_hash=f"{source}:{player_id}:{market}:{line}:{over_prob_novig}",
        over_prob_novig=over_prob_novig,
        under_prob_novig=None if over_prob_novig is None else 1.0 - over_prob_novig,
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


def test_canonicalize_team_abbrev_book_aliases():
    assert canonicalize_team_abbrev("LAR") == "LA"
    assert canonicalize_team_abbrev("JAC") == "JAX"
    assert canonicalize_team_abbrev("WSH") == "WAS"
    assert canonicalize_team_abbrev("buf") == "BUF"


def test_eastern_wallclock_sunday_1300_et_to_utc():
    """13:00 ET Sunday → 17:00Z (EDT) — not naive-as-UTC 13:00Z."""
    naive = datetime(2026, 9, 13, 13, 0, 0)
    utc = eastern_wallclock_to_utc(naive)
    assert utc == datetime(2026, 9, 13, 17, 0, 0, tzinfo=UTC)
    # Round-trip via ZoneInfo for the same wall clock.
    assert naive.replace(tzinfo=ET).astimezone(UTC) == utc


def test_schedule_kickoffs_sunday_1300_et_not_naive_utc():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "PIT",
                "away_team": "ATL",
            }
        ]
    )
    kickoffs = kickoffs_by_team_from_schedule(frame, season=2026, week=1)
    assert kickoffs["ATL"] == datetime(2026, 9, 13, 17, 0, 0, tzinfo=UTC)
    assert kickoffs["PIT"] == kickoffs["ATL"]


def test_schedule_kickoffs_week2_buf_thursday_night_et():
    kickoffs = load_schedule_kickoffs(SCHEDULE, season=2026, week=2)
    assert "BUF" in kickoffs
    # 2026-09-17 20:15 ET → 2026-09-18 00:15Z (EDT, UTC-4)
    assert kickoffs["BUF"] == datetime(2026, 9, 18, 0, 15, 0, tzinfo=UTC)
    assert kickoffs["DET"] == kickoffs["BUF"]


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


def test_export_drops_when_team_missing_schedule_even_with_event_start():
    """Fail closed: book event_start is not a leakage boundary."""
    snaps = [
        _snap(
            [
                _quote(
                    team="ZZZ",
                    event_start="2099-09-14T17:00:00+00:00",
                )
            ]
        )
    ]
    rows = export_role2_snapshot_rows(
        snaps, kickoffs_by_team={"BUF": datetime(2026, 9, 17, 20, 15, tzinfo=UTC)}
    )
    assert rows == []


def test_export_resolves_lar_alias_to_schedule_la():
    kickoffs = {"LA": datetime(2026, 9, 10, 0, 35, tzinfo=UTC)}
    snaps = [
        _snap(
            [
                _quote(
                    team="LAR",
                    player_id="00-0033077",
                    player_name="Matthew Stafford",
                    market="pass_yards",
                    line=255.5,
                    fetched_at="2026-09-09T18:00:00+00:00",
                    event_start=None,
                )
            ],
            week=1,
        )
    ]
    rows = export_role2_snapshot_rows(snaps, kickoffs_by_team=kickoffs)
    assert len(rows) == 1
    assert rows[0]["kickoff_at"] == "2026-09-10T00:35:00Z"


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


def test_consensus_implied_p_over_matches_consensus_line_only():
    """4.5@0.62 / 5.5@0.48 / 5.5@0.47 → line 5.5, p_over ≈ 0.475 (not ~0.523)."""
    kickoffs = {"BUF": datetime(2026, 9, 17, 20, 15, tzinfo=UTC)}
    snaps = [
        _snap(
            [
                _quote(
                    source="draftkings",
                    market="receptions",
                    line=4.5,
                    over_prob_novig=0.62,
                )
            ],
            source="draftkings",
        ),
        _snap(
            [
                _quote(
                    source="fanduel",
                    market="receptions",
                    line=5.5,
                    over_prob_novig=0.48,
                )
            ],
            source="fanduel",
        ),
        _snap(
            [
                _quote(
                    source="oddschecker",
                    market="receptions",
                    line=5.5,
                    over_prob_novig=0.47,
                )
            ],
            source="oddschecker",
        ),
    ]
    rows = export_role2_snapshot_rows(
        snaps, kickoffs_by_team=kickoffs, mode="consensus"
    )
    assert len(rows) == 1
    assert rows[0]["line"] == 5.5
    assert rows[0]["implied_p_over"] == pytest.approx(0.475)
    assert rows[0]["implied_p_over"] != pytest.approx((0.62 + 0.48 + 0.47) / 3)


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


def test_identity_map_leaves_colliding_team_keys_unmapped():
    identity, ambiguous = build_identity_map_from_records_with_stats(
        [
            {
                "player_id": "00-0000001",
                "gsis_id": "00-0000001",
                "name": "Alex Smith",
                "team": "WAS",
                "position": "QB",
            },
            {
                "player_id": "00-0000002",
                "gsis_id": "00-0000002",
                "name": "Alex Smith",
                "team": "WAS",
                "position": "QB",
            },
        ]
    )
    assert ambiguous >= 1
    # Team-qualified key must not silently last-write-wins.
    assert "alex smith|WAS" not in identity
    assert "alex smith|WAS|QB" not in identity
    # Bare gsis keys still present.
    assert identity["00-0000001"]["player_id"] == "00-0000001"
    assert identity["00-0000002"]["player_id"] == "00-0000002"


def test_identity_json_merges_into_existing_map():
    """DB entries survive a small --identity-json patch (merge, not replace)."""
    base = build_identity_map_from_records(
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
    patch = build_identity_map_from_records(
        [
            {
                "player_id": "00-0033280",
                "gsis_id": "00-0033280",
                "name": "Christian McCaffrey",
                "team": "SF",
                "position": "RB",
            }
        ]
    )
    merged = dict(base)
    merged.update(patch)
    assert "00-0034857" in merged
    assert "00-0033280" in merged
    assert merged["00-0034857"]["name"] == "Josh Allen"


def test_fixture_providers_export_passes_schema(tmp_path):
    """Committed DK/FD fixtures already carry gsis ids — export must schema-validate."""
    dk = FixturePropProvider(
        "draftkings", FIXTURE_DIR / "draftkings.json"
    ).fetch(season=2026, week=1)
    fd = FixturePropProvider("fanduel", FIXTURE_DIR / "fanduel.json").fetch(
        season=2026, week=1
    )
    assert dk.success and fd.success
    # Fixture event_start is in 2099 and must not be trusted; schedule only.
    kickoffs = load_schedule_kickoffs(SCHEDULE, season=2026, week=1)
    from dataclasses import replace

    def _retime(snap: ProviderSnapshot) -> ProviderSnapshot:
        fetched = datetime(2026, 9, 8, 18, 0, tzinfo=UTC)
        # Keep event_start in the distant future — schedule must still win;
        # rows without a schedule team are dropped, not rescued by event_start.
        quotes = tuple(replace(q, fetched_at=fetched) for q in snap.quotes)
        return replace(snap, fetched_at=fetched, quotes=quotes, week=1)

    rows = export_role2_snapshot_rows(
        [_retime(dk), _retime(fd)],
        kickoffs_by_team=kickoffs,
        mode="consensus",
    )
    assert len(rows) > 0
    assert all(is_gsis_shaped(r["player_id"]) for r in rows)
    assert all(r["as_of"] <= r["kickoff_at"] for r in rows)
    # Kickoffs must be ET→UTC (Sunday 13:00 ET → 17:00Z), not naive 13:00Z.
    sunday_kickoffs = {r["kickoff_at"] for r in rows if "T17:00:00Z" in r["kickoff_at"]}
    assert sunday_kickoffs, "expected at least one 13:00 ET → 17:00Z kickoff"
    out = write_live_snapshots_csv(rows, tmp_path / "live_snapshots.csv")
    loaded = load_prop_snapshots(out)
    assert len(loaded) == len(rows)


def test_missing_live_export_env_reports_keys(monkeypatch):
    for key in (
        "DATABASE_URL",
        "JOB_DATABASE_URL",
        "ARTIFACT_BACKEND",
        "S3_ENDPOINT_URL",
        "S3_BUCKET",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    missing = missing_live_export_env()
    assert "DATABASE_URL|JOB_DATABASE_URL" in missing
    assert "ARTIFACT_BACKEND=s3" in missing
    assert "S3_ACCESS_KEY_ID" in missing
    assert "S3_BUCKET" in missing
    assert "S3_REGION" in missing


def test_missing_live_export_env_requires_bucket_and_region(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://example.test/s3")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "sk")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_REGION", raising=False)
    missing = missing_live_export_env()
    assert "S3_BUCKET" in missing
    assert "S3_REGION" in missing


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


def test_cli_rejects_snapshot_week_mismatch(tmp_path):
    import json
    import runpy

    snap_path = tmp_path / "dk.json"
    payload = ProviderSnapshot(
        source="draftkings",
        season=2026,
        week=1,  # mismatch vs CLI --week 2
        fetched_at=datetime(2026, 9, 8, 18, 0, tzinfo=UTC),
        urls=(),
        quotes=(_quote(fetched_at="2026-09-08T18:00:00+00:00", event_start=None),),
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
            str(tmp_path / "out.csv"),
        ]
    )
    assert rc == 1


def test_cli_identity_json_merges_not_replaces(tmp_path, monkeypatch):
    """Simulate DB map + --identity-json patch; DB entries must survive."""
    import json
    import runpy

    # Build a tiny snapshot that needs the DB identity for name resolution,
    # plus a patch that only adds CMC — Josh Allen must still resolve from base.
    out = tmp_path / "live_snapshots.csv"
    snap_path = tmp_path / "dk.json"
    identity_path = tmp_path / "patch.json"
    payload = ProviderSnapshot(
        source="draftkings",
        season=2026,
        week=2,
        fetched_at=datetime(2026, 9, 16, 18, 0, tzinfo=UTC),
        urls=(),
        quotes=(
            _quote(
                player_id=None,
                player_name="Josh Allen",
                team="BUF",
                fetched_at="2026-09-16T18:00:00+00:00",
                event_start=None,
            ),
            _quote(
                player_id=None,
                player_name="Christian McCaffrey",
                team="SF",
                market="rush_yards",
                line=72.5,
                fetched_at="2026-09-16T18:00:00+00:00",
                event_start=None,
            ),
        ),
        success=True,
    ).to_dict()
    snap_path.write_text(json.dumps(payload), encoding="utf-8")
    # Patch only has CMC — if replace wins, Josh Allen vanishes.
    identity_path.write_text(
        json.dumps(
            [
                {
                    "player_id": "00-0033280",
                    "gsis_id": "00-0033280",
                    "name": "Christian McCaffrey",
                    "team": "SF",
                    "position": "RB",
                }
            ]
        ),
        encoding="utf-8",
    )

    # Inject a pre-seeded identity_map by patching main's DB path is heavy;
    # instead exercise merge semantics the script uses.
    base = build_identity_map_from_records(
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
    patch = build_identity_map_from_records(
        json.loads(identity_path.read_text(encoding="utf-8"))
    )
    merged = dict(base)
    merged.update(patch)
    kickoffs = load_schedule_kickoffs(SCHEDULE, season=2026, week=2)
    snap = ProviderSnapshot.from_dict(payload)
    rows = export_role2_snapshot_rows(
        [snap], identity_map=merged, kickoffs_by_team=kickoffs
    )
    pids = {r["player_id"] for r in rows}
    assert "00-0034857" in pids
    assert "00-0033280" in pids

    # Also run CLI with only the patch (no DB) — Josh Allen unresolved → dropped.
    mod = runpy.run_path(str(REPO_ROOT / "scripts" / "export_role2_live_props.py"))
    rc = mod["main"](
        [
            "--season",
            "2026",
            "--week",
            "2",
            "--snapshot-json",
            str(snap_path),
            "--identity-json",
            str(identity_path),
            "--schedule",
            str(SCHEDULE),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    loaded = load_prop_snapshots(out)
    pids_cli = {p.player_id for p in loaded}
    assert "00-0033280" in pids_cli
    # Without DB base, Josh Allen is unresolved and dropped.
    assert "00-0034857" not in pids_cli


def test_cli_check_env_fails_without_creds(monkeypatch):
    import runpy

    for key in (
        "DATABASE_URL",
        "JOB_DATABASE_URL",
        "ARTIFACT_BACKEND",
        "S3_ENDPOINT_URL",
        "S3_BUCKET",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    mod = runpy.run_path(str(REPO_ROOT / "scripts" / "export_role2_live_props.py"))
    assert mod["main"](["--check-env"]) == 1


def test_cli_check_env_incomplete_without_bucket(monkeypatch):
    import runpy

    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://example.test/s3")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "sk")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_REGION", raising=False)
    mod = runpy.run_path(str(REPO_ROOT / "scripts" / "export_role2_live_props.py"))
    assert mod["main"](["--check-env"]) == 1
