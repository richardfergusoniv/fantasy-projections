"""Deterministic tests for shadow sync safety and draft-rule persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.app.league.sleeper.owner_config import SleeperOwnerConfig
from src.app.league.sleeper.shadow_sync import (
    ShadowSyncOptions,
    _sqlite_db_path,
    assert_database_target,
    assert_opt_in,
    assert_read_only_client,
)
from src.app.league.sleeper.sync import SleeperSyncService
from src.app.persistence.models import LeagueDraftRule
from src.app.releases.gates import GateResult
from src.app.releases.publication import Candidate, CandidateRow, active_pointer, publish

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "config" / "sleeper_owner.example.json"


def test_read_only_client_assertion_passes():
    assert_read_only_client()


def test_shadow_sync_requires_opt_in(monkeypatch):
    monkeypatch.delenv("LIVE_SLEEPER_SHADOW", raising=False)
    with pytest.raises(RuntimeError, match="LIVE_SLEEPER_SHADOW"):
        assert_opt_in()


def test_production_database_refused_without_ack():
    with pytest.raises(RuntimeError, match="production-looking"):
        assert_database_target("sqlite+pysqlite:///./local_app.db", allow_production=False)


def test_sqlite_db_path_ignores_postgresql_urls():
    assert _sqlite_db_path("postgresql+psycopg://fantasy:fantasy@localhost:5432/fantasy_app") is None
    assert _sqlite_db_path("sqlite+pysqlite:///output/live_shadow/shadow_app.db") == Path(
        "output/live_shadow/shadow_app.db"
    )


def test_resolve_player_ids_skips_sleeper_empty_slot_sentinel(db_session):
    sync = SleeperSyncService(db_session, use_fixtures=True)
    resolved = sync.resolve_player_ids(["0", "", None, "fixture-qb-1"])
    assert resolved == ["fixture-qb-1"]
    assert "0" not in sync.unresolved_player_ids


def test_owner_draft_rule_persistence_is_idempotent(db_session):
    sync = SleeperSyncService(db_session, use_fixtures=True)
    first = sync.persist_owner_confirmed_draft_rule("fixture-dynasty", "reverse_standings")
    second = sync.persist_owner_confirmed_draft_rule("fixture-dynasty", "reverse_standings")
    assert first.id == second.id
    rows = (
        db_session.query(LeagueDraftRule)
        .filter(LeagueDraftRule.league_id == "fixture-dynasty")
        .all()
    )
    assert len(rows) == 1


def test_owner_draft_rule_change_updates_confirmed_at(db_session):
    sync = SleeperSyncService(db_session, use_fixtures=True)
    first = sync.persist_owner_confirmed_draft_rule("fixture-dynasty", "max_pf")
    confirmed_at = first.confirmed_at
    updated = sync.persist_owner_confirmed_draft_rule("fixture-dynasty", "reverse_standings")
    assert updated.rule == "reverse_standings"
    assert updated.confirmed_at >= confirmed_at


def test_redraft_league_config_has_no_persisted_rule(db_session):
    config = SleeperOwnerConfig.model_validate(json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")))
    sync = SleeperSyncService(db_session, use_fixtures=True)
    for entry in config.leagues:
        if entry.league_type == "dynasty" and entry.rookie_pick_rule:
            sync.persist_owner_confirmed_draft_rule(entry.league_id, entry.rookie_pick_rule)
    redraft_ids = [entry.league_id for entry in config.leagues if entry.league_type == "redraft"]
    for league_id in redraft_ids:
        assert (
            db_session.query(LeagueDraftRule)
            .filter(LeagueDraftRule.league_id == league_id)
            .count()
            == 0
        )


def test_failure_injection_does_not_advance_active_pointer(db_session):
    from src.app.persistence.models import ActiveProjectionPointer, ProjectionRun

    season = 2099
    week = 3
    run = ProjectionRun(
        id="weekly-baseline-shadow-test",
        mode="weekly",
        season=season,
        week=week,
        as_of=datetime.now(UTC),
        model_version="fixture",
        input_hash="abc",
        status="active",
        manifest_uri="fixture://baseline",
        artifact_mode="fixture",
    )
    db_session.add(run)
    db_session.flush()
    db_session.add(
        ActiveProjectionPointer(mode="weekly", season=season, week=week, run_id=run.id)
    )
    db_session.flush()

    candidate = Candidate(
        mode="weekly",
        season=season,
        week=week,
        run_id="weekly-fail-shadow-test",
        model_version="fixture",
        input_hash="dead",
        manifest_uri="fixture://fail",
        artifact_mode="fixture",
        partition_mode="weekly",
        rows=(
            CandidateRow(
                player_id="00-test",
                team="TST",
                opponent=None,
                availability_probability=1.0,
                mean_json={"points": 1.0},
                quantiles_json={"p50": 1.0},
            ),
        ),
    )
    result = publish(
        db_session,
        candidate,
        gates={"promotion": GateResult(passed=False, failures=["injected"])},
        register_partitions=False,
        validate_partitions=False,
    )
    pointer = active_pointer(db_session, mode="weekly", season=season, week=week)
    assert result.promoted is False
    assert pointer is not None
    assert pointer.run_id == "weekly-baseline-shadow-test"


def test_shadow_sync_options_defaults_use_isolated_paths():
    options = ShadowSyncOptions(config_path=EXAMPLE_PATH)
    assert "live_shadow" in options.database_url
    assert "live_shadow" in options.artifact_root
