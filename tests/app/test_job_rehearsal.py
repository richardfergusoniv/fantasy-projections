"""Two-week job rehearsal and projection rollback tests."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session


def test_projection_rollback_restores_previous_pointer(db_session: Session, monkeypatch, tmp_path):
    from src.app.availability.service import AvailabilityService
    from src.app.persistence.models import SourceSnapshot
    from src.app.persistence.repositories import ProjectionRepository
    from src.app.projections.loader import ReleaseBundleLoader
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge
    from src.app.releases.incremental import IncrementalSimulationService, build_impact_set
    from src.app.releases.rollback import ProjectionRollbackService

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
        body_hash="rollback-test",
        artifact_uri="local://test",
        health_verdict="healthy",
        is_complete=True,
    )
    db_session.add(snapshot)
    db_session.flush()
    players = ReleaseBundleLoader(2026).load()
    player_id = next(iter(players))
    AvailabilityService(db_session).activate_event(
        player_id=player_id,
        event_type="injury_status",
        source_snapshot_id=snapshot.id,
        evidence_ids=[],
        policy={"play_probability": 0.35},
    )
    impact = build_impact_set({player_id}, players)
    inc_run_id = IncrementalSimulationService(db_session).promote_affected_week(
        2026,
        1,
        impact,
        base_run_id=weekly_run_id,
    )
    assert inc_run_id
    assert inc_run_id != weekly_run_id

    active_before = ProjectionRepository(db_session).active_run(mode="weekly", season=2026, week=1)
    assert active_before is not None
    assert active_before.id == inc_run_id

    restored = ProjectionRollbackService(db_session).rollback("weekly", 2026, 1)
    assert restored == weekly_run_id
    active_after = ProjectionRepository(db_session).active_run(mode="weekly", season=2026, week=1)
    assert active_after is not None
    assert active_after.id == weekly_run_id


def test_two_consecutive_daily_refresh_jobs(db_session: Session, monkeypatch, tmp_path):
    from src.app.jobs.handlers import run_daily_refresh
    from src.app.jobs.runner import JobRunner
    from src.app.persistence.repositories import ProjectionRepository
    from src.app.seed import seed_development_data

    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(tmp_path / "empty_models"))
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(tmp_path / "empty_outputs"))

    seed_development_data(db_session, email="owner@example.com")
    runner = JobRunner(db_session)
    first = runner.run("daily-refresh", lambda: run_daily_refresh(db_session), idempotency_key="rehearsal-week-1")
    second = runner.run("daily-refresh", lambda: run_daily_refresh(db_session), idempotency_key="rehearsal-week-2")
    assert first.status == "succeeded"
    assert second.status == "succeeded"
    preseason = ProjectionRepository(db_session).active_run(mode="preseason", season=2026, week=None)
    assert preseason is not None
    # Production daily path promotes sealed preseason + status overlay, not weekly-v2.
    assert first.metadata_json.get("preseason_run_id")
    assert second.metadata_json.get("preseason_run_id")
    assert first.metadata_json.get("weekly_run_id") is None
    assert second.metadata_json.get("weekly_run_id") is None
    assert first.metadata_json.get("incremental", {}).get("mode") == "weekly_rnd_disabled"
    assert second.metadata_json.get("scoring_gate", {}).get("passed") is True
