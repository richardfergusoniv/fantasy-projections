"""Weekly projection run tests."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session


def test_weekly_run_promotes_pointer(db_session: Session):
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge

    bridge = ReleaseBridge(db_session)
    if bridge.sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    service = WeeklyProjectionService(db_session)
    run_id = service.promote_week(2026, week=1)
    assert run_id is not None
    from src.app.persistence.repositories import ProjectionRepository

    run = ProjectionRepository(db_session).active_run(mode="weekly", season=2026, week=1)
    assert run is not None
    assert run.id == run_id
    assert run.model_version == "weekly_v2_fixture"
    projections = ProjectionRepository(db_session).player_projections(run_id)
    assert len(projections) > 100
    from src.app.persistence.models import SimulationPartition

    partitions = (
        db_session.query(SimulationPartition)
        .filter(SimulationPartition.run_id == run_id)
        .all()
    )
    assert len(partitions) == 1
    assert partitions[0].draw_count > 0


def test_lineup_uses_weekly_run(db_session: Session):
    from src.app.decisions.services import LineupService
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge
    from src.app.seed import seed_development_data

    seed_development_data(db_session, email="owner@example.com")
    bridge = ReleaseBridge(db_session)
    if bridge.sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    WeeklyProjectionService(db_session).promote_week(2026, week=1)
    result = LineupService(db_session).recommend("fixture-standard", 1)
    assert result["projection_run_id"].startswith("weekly-")
    assert result["recommended_starters"]
