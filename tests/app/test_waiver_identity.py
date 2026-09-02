"""Waiver ownership identity reconciliation tests."""

from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import Session

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_PROJECTION_SOURCE", "sealed_release")


def test_waiver_blocks_when_roster_ids_do_not_resolve(db_session: Session):
    from src.app.decisions.services import LeagueContextError, WaiverService
    from src.app.persistence.models import League, LeagueRuleSnapshot, RosterSnapshot
    from src.app.seed import seed_development_data

    seed_development_data(db_session, email="owner@example.com")
    league = db_session.query(League).filter(League.league_id == "fixture-standard").one()
    snapshot = (
        db_session.query(LeagueRuleSnapshot)
        .filter(LeagueRuleSnapshot.league_id == league.league_id)
        .order_by(LeagueRuleSnapshot.fetched_at.desc())
        .first()
    )
    assert snapshot is not None
    db_session.add(
        RosterSnapshot(
            league_id=league.league_id,
            week=1,
            roster_id=1,
            fetched_at=snapshot.fetched_at,
            players=["unresolved-sleeper-id"],
            starters=[],
            reserve=[],
        )
    )
    db_session.commit()

    with pytest.raises(LeagueContextError, match="waiver_ownership_incomplete"):
        WaiverService(db_session).recommend(league.league_id, 1)
