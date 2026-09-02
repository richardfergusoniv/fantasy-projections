"""Incremental simulation tests."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session


def test_build_impact_set_expands_teammates():
    from src.app.projections.loader import PlayerSummary
    from src.app.releases.incremental import build_impact_set

    players = {
        "p1": PlayerSummary("p1", "A", "RB", "KC", 10.0, {"p50": 10.0}),
        "p2": PlayerSummary("p2", "B", "WR", "KC", 8.0, {"p50": 8.0}),
        "p3": PlayerSummary("p3", "C", "RB", "BUF", 9.0, {"p50": 9.0}),
    }
    impact = build_impact_set({"p1"}, players)
    assert "p1" in impact.affected_player_ids
    assert "p2" in impact.affected_player_ids
    assert "p3" not in impact.affected_player_ids
    assert impact.affected_teams == frozenset({"KC"})


def test_incremental_promotion_updates_active_pointer(db_session: Session, monkeypatch, tmp_path):
    from src.app.availability.service import AvailabilityService
    from src.app.persistence.models import SourceSnapshot
    from src.app.projections.loader import ReleaseBundleLoader
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge
    from src.app.releases.incremental import IncrementalSimulationService, build_impact_set
    from src.app.persistence.repositories import ProjectionRepository

    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(tmp_path / "empty_models"))
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(tmp_path / "empty_outputs"))

    if ReleaseBridge(db_session).sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    weekly_run_id = WeeklyProjectionService(db_session).promote_week(2026, week=1, automatic=False)
    assert weekly_run_id

    snapshot = SourceSnapshot(
        endpoint="players/nfl",
        request_params_json={},
        fetched_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        body_hash="incremental-test",
        artifact_uri="local://test",
        health_verdict="healthy",
        is_complete=True,
    )
    db_session.add(snapshot)
    db_session.flush()
    AvailabilityService(db_session).activate_event(
        player_id="00-0034857",
        event_type="injury_status",
        source_snapshot_id=snapshot.id,
        evidence_ids=[],
        policy={"play_probability": 0.5},
    )

    players = ReleaseBundleLoader(2026).load()
    player_id = next(iter(players))
    AvailabilityService(db_session).activate_event(
        player_id=player_id,
        event_type="injury_status",
        source_snapshot_id=snapshot.id,
        evidence_ids=[],
        policy={"play_probability": 0.4},
    )
    impact = build_impact_set({player_id}, players)
    inc_run_id = IncrementalSimulationService(db_session).promote_affected_week(
        2026,
        1,
        impact,
        base_run_id=weekly_run_id,
    )
    assert inc_run_id is not None
    assert inc_run_id.startswith(weekly_run_id)
    active = ProjectionRepository(db_session).active_run(mode="weekly", season=2026, week=1)
    assert active is not None
    assert active.id == inc_run_id
