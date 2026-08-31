"""Weekly v2 bridge and research job tests."""

from __future__ import annotations


def test_weekly_v2_fixture_manifest_available():
    from src.app.projections.weekly_v2_bridge import (
        load_weekly_v2_manifest,
        weekly_v2_artifacts_available,
        weekly_v2_model_version,
    )

    assert weekly_v2_artifacts_available(2026, 1) is True
    manifest = load_weekly_v2_manifest(2026)
    assert manifest is not None
    assert manifest["model_version"] == "weekly_v2_fixture"
    assert weekly_v2_model_version(2026) == "weekly_v2_fixture"


def test_weekly_v2_bridge_defaults_without_artifacts():
    from src.app.projections.weekly_v2_bridge import (
        weekly_v2_artifacts_available,
        weekly_v2_model_version,
    )

    assert weekly_v2_artifacts_available(2099, 1) is False
    assert weekly_v2_model_version(2099) == "weekly_fixture_v1"


def test_research_changed_players(db_session):
    from src.app.availability.research_job import research_changed_players
    from src.app.availability.service import AvailabilityService
    from src.app.persistence.models import AvailabilityEvent, SourceSnapshot

    snapshot = SourceSnapshot(
        endpoint="players/nfl",
        request_params_json={},
        fetched_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        body_hash="abc",
        artifact_uri="local://test",
        health_verdict="healthy",
        is_complete=True,
    )
    db_session.add(snapshot)
    db_session.flush()
    service = AvailabilityService(db_session)
    event = service.activate_event(
        player_id="fixture-research-only-player",
        event_type="injury_status",
        source_snapshot_id=snapshot.id,
        evidence_ids=[],
        policy={"play_probability": 0.6},
    )
    assert event.id
    result = research_changed_players(db_session)
    assert result["researched"] >= 1
    assert result["mode"] == "fixture"
    assert result["synthetic"] is True
    assert result["status"] == "ok"
    evidence = service.repo.evidence_for_player("fixture-research-only-player")
    assert evidence
    assert all(row.claim_json["synthetic"] is True for row in evidence)
    assert all(row.source_url.startswith("fixture://") for row in evidence)
