"""Sleeper status injury research from synced player snapshots."""

from __future__ import annotations

from datetime import UTC, datetime

from src.app.availability.research import (
    MODE_SLEEPER,
    ResearchUnavailable,
    SleeperStatusInjuryResearchProvider,
)
from src.app.availability.service import AvailabilityService
from src.app.persistence.models import PlayerStatusSnapshot, SourceSnapshot


def _snapshot(db_session, **overrides):
    fields = {
        "endpoint": "players/nfl",
        "request_params_json": {},
        "fetched_at": datetime.now(UTC),
        "body_hash": f"hash-{datetime.now(UTC).timestamp()}",
        "artifact_uri": "local://test",
        "health_verdict": "healthy",
        "is_complete": True,
    }
    fields.update(overrides)
    snapshot = SourceSnapshot(**fields)
    db_session.add(snapshot)
    db_session.flush()
    return snapshot


def test_sleeper_status_provider_creates_cited_claim(db_session):
    _snapshot(db_session)
    db_session.add(
        PlayerStatusSnapshot(
            player_id="player-1",
            fetched_at=datetime.now(UTC),
            status="Active",
            injury_status="Questionable",
            practice="Limited",
            raw_json={"full_name": "Test Player", "injury_body_part": "ankle"},
        )
    )
    db_session.flush()

    result = SleeperStatusInjuryResearchProvider(db_session).research("player-1")
    assert result.mode == MODE_SLEEPER
    assert not result.synthetic
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.status == "questionable"
    assert claim.sources[0]["url"].startswith("https://api.sleeper.app/")


def test_sleeper_status_provider_rejects_missing_status(db_session):
    try:
        SleeperStatusInjuryResearchProvider(db_session).research("missing")
    except ResearchUnavailable as exc:
        assert "no_sleeper_injury_status" in str(exc)
    else:
        raise AssertionError("expected ResearchUnavailable")


def test_research_job_uses_sleeper_mode(db_session):
    from src.app.availability.research_job import research_changed_players

    snapshot = _snapshot(db_session)
    db_session.add(
        PlayerStatusSnapshot(
            player_id="player-2",
            fetched_at=datetime.now(UTC),
            status="Active",
            injury_status="Out",
            practice=None,
            raw_json={"full_name": "Out Player"},
        )
    )
    AvailabilityService(db_session).activate_event(
        player_id="player-2",
        event_type="injury_status",
        source_snapshot_id=snapshot.id,
        evidence_ids=[],
        policy={"play_probability": 0.0, "injury_status": "Out"},
    )
    db_session.flush()

    summary = research_changed_players(db_session, mode=MODE_SLEEPER)
    assert summary["status"] == "ok"
    assert summary["researched"] == 1
    assert summary["synthetic"] is False
