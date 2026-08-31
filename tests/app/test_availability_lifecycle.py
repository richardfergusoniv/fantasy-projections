"""Availability evidence lifecycle: provenance, kickoff freeze, and clearing rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)


def _snapshot(session: Session, **overrides):
    from src.app.persistence.models import SourceSnapshot

    fields = {
        "endpoint": "players/nfl",
        "request_params_json": {},
        "fetched_at": datetime.now(UTC),
        "body_hash": f"hash-{len(overrides)}-{datetime.now(UTC).timestamp()}",
        "artifact_uri": "local://test",
        "health_verdict": "healthy",
        "is_complete": True,
    }
    fields.update(overrides)
    snapshot = SourceSnapshot(**fields)
    session.add(snapshot)
    session.flush()
    return snapshot


def _claim(player_id: str, **overrides):
    from src.app.availability.service import EvidenceClaim

    fields = {
        "player_id": player_id,
        "status": "questionable",
        "reported_injury": "hamstring",
        "expected_return_min": None,
        "expected_return_max": None,
        "claim_confidence": 0.8,
        "sources": [
            {
                "url": f"https://www.espn.com/nfl/story/_/id/{player_id}",
                "title": "Beat writer report",
                "published_at": "2026-09-11T13:05:00Z",
            }
        ],
        "publisher": "espn.com",
        "source_reliability": 0.9,
    }
    fields.update(overrides)
    return EvidenceClaim(**fields)


# ---------------------------------------------------------------------------
# A. Research provider selection and synthetic flagging
# ---------------------------------------------------------------------------


def test_fixture_research_marks_evidence_synthetic(db_session: Session):
    from src.app.availability.research_job import research_changed_players
    from src.app.availability.service import AvailabilityService, summarize_evidence

    service = AvailabilityService(db_session)
    player_id = "fixture-mode-player"
    service.activate_event(
        player_id=player_id,
        event_type="injury_status",
        source_snapshot_id=_snapshot(db_session).id,
        evidence_ids=[],
        policy={"play_probability": 0.6},
    )

    result = research_changed_players(db_session, mode="fixture")
    assert result["mode"] == "fixture"
    assert result["synthetic"] is True

    rows = service.repo.evidence_for_player(player_id)
    assert rows
    for row in rows:
        assert row.claim_json["synthetic"] is True
        assert row.claim_json["mode"] == "fixture"
        assert row.source_url.startswith("fixture://")
        assert "http" not in row.source_url
    assert all(item["synthetic"] is True for item in summarize_evidence(rows))


def test_live_research_without_configuration_reports_unavailable(db_session: Session, monkeypatch):
    from src.app.availability import research_job
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    player_id = "live-unconfigured-player"
    service.activate_event(
        player_id=player_id,
        event_type="injury_status",
        source_snapshot_id=_snapshot(db_session).id,
        evidence_ids=[],
        policy={"play_probability": 0.6},
    )

    class _Settings:
        app_env = "production"
        openai_api_key = None
        openai_balanced_model = "gpt-4.1"

    monkeypatch.setattr(research_job, "get_settings", lambda: _Settings())
    result = research_job.research_changed_players(db_session, mode="live")

    assert result["status"] == "unavailable"
    assert result["available"] is False
    assert result["researched"] == 0
    assert result["synthetic"] is False
    assert service.repo.evidence_for_player(player_id) == []


def test_fabricated_citation_cannot_be_stored_unflagged(db_session: Session):
    from src.app.availability.service import AvailabilityService, FabricatedCitationError

    service = AvailabilityService(db_session)
    fabricated = _claim(
        "fabricated-player",
        sources=[{"url": "https://example.com/injury-fixture", "title": "Fixture injury report"}],
    )
    with pytest.raises(FabricatedCitationError):
        service.add_evidence(fabricated)

    masquerading = _claim(
        "fabricated-player",
        sources=[{"url": "https://www.espn.com/nfl/story/_/id/fake", "title": "x"}],
        mode="fixture",
        synthetic=True,
    )
    with pytest.raises(FabricatedCitationError):
        service.add_evidence(masquerading)

    assert service.repo.evidence_for_player("fabricated-player") == []


# ---------------------------------------------------------------------------
# B. Evidence contract
# ---------------------------------------------------------------------------


def test_evidence_persists_published_at_publisher_and_reliability(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    row = service.add_evidence(
        _claim(
            "contract-player",
            published_at="2026-09-11T13:05:00Z",
            retrieved_at="2026-09-11T14:00:00Z",
        )
    )

    assert row.published_at is not None
    assert row.published_at.replace(tzinfo=UTC) == datetime(2026, 9, 11, 13, 5, tzinfo=UTC)
    assert row.claim_json["publisher"] == "espn.com"
    assert row.claim_json["source_reliability"] == 0.9
    assert row.claim_json["retrieved_at"] == "2026-09-11T14:00:00+00:00"
    # retrieved-at is the source read time and must not be conflated with the
    # row's ingest time.
    assert row.fetched_at.replace(tzinfo=UTC) != datetime(2026, 9, 11, 14, 0, tzinfo=UTC)


def test_uncited_return_date_claim_is_rejected(db_session: Session):
    from src.app.availability.service import AvailabilityService, UncitedClaimError

    service = AvailabilityService(db_session)
    with pytest.raises(UncitedClaimError, match="Uncited return-date"):
        service.add_evidence(
            _claim(
                "uncited-player",
                expected_return_min="2026-10-04",
                expected_return_max="2026-10-18",
                sources=[],
            )
        )
    with pytest.raises(UncitedClaimError):
        service.add_evidence(
            _claim(
                "uncited-player",
                expected_return_min="2026-10-04",
                sources=[{"title": "no url"}],
            )
        )
    assert service.repo.evidence_for_player("uncited-player") == []


# ---------------------------------------------------------------------------
# C. Post-kickoff evidence cannot move the frozen pregame evaluation
# ---------------------------------------------------------------------------


def test_post_kickoff_evidence_is_stored_but_does_not_change_pregame(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    player_id = "kickoff-player"

    baseline = service.evaluate_pregame(player_id, season=2026, week=2, kickoff_at=KICKOFF)
    assert baseline.play_probability == 1.0

    pre = service.add_evidence(
        _claim(
            player_id,
            status="questionable",
            published_at=(KICKOFF - timedelta(hours=20)).isoformat(),
        ),
        kickoff_at=KICKOFF,
    )
    pregame = service.evaluate_pregame(player_id, season=2026, week=2, kickoff_at=KICKOFF)
    assert pregame.play_probability == pytest.approx(0.65)
    assert pre.id in pregame.evidence_ids

    frozen = service.freeze_pregame_evaluation(player_id, season=2026, week=2, kickoff_at=KICKOFF)
    assert frozen.play_probability == pytest.approx(0.65)

    post = service.add_evidence(
        _claim(
            player_id,
            status="out",
            reported_injury="hamstring aggravated in warmups",
            sources=[
                {
                    "url": "https://www.nfl.com/news/kickoff-player-inactive",
                    "title": "Ruled out during pregame",
                }
            ],
            published_at=(KICKOFF + timedelta(minutes=30)).isoformat(),
        ),
        kickoff_at=KICKOFF,
    )

    assert post.id is not None
    assert post.claim_json["post_kickoff"] is True
    after = service.evaluate_pregame(player_id, season=2026, week=2, kickoff_at=KICKOFF)
    assert after.play_probability == pytest.approx(0.65)
    assert post.id not in after.evidence_ids
    # Stored, and still allowed to inform forward-looking availability.
    assert service.rest_of_season_probability(player_id) < 0.65

    unfrozen_next_week = service.evaluate_pregame(
        player_id, season=2026, week=3, kickoff_at=KICKOFF + timedelta(days=7)
    )
    assert unfrozen_next_week.play_probability < 0.65


# ---------------------------------------------------------------------------
# D. Namesake identity
# ---------------------------------------------------------------------------


def test_namesake_evidence_is_quarantined_not_guessed(db_session: Session):
    from src.app.availability.service import QUARANTINE_PLAYER_ID, AvailabilityService
    from src.app.persistence.models import PlayerIdentity

    db_session.add_all(
        [
            PlayerIdentity(
                player_id="twin-qb-buf",
                sleeper_id="twin-1",
                gsis_id="00-twin-1",
                name="Alex Rivers",
                position="QB",
                team="BUF",
            ),
            PlayerIdentity(
                player_id="twin-wr-jax",
                sleeper_id="twin-2",
                gsis_id="00-twin-2",
                name="Alex Rivers",
                position="WR",
                team="JAX",
            ),
        ]
    )
    db_session.flush()

    service = AvailabilityService(db_session)
    ambiguous = service.submit_named_evidence(_claim("", status="out"), name="Alex Rivers")

    assert ambiguous.status == "quarantined_ambiguous"
    assert ambiguous.applied is False
    assert sorted(ambiguous.candidates) == ["twin-qb-buf", "twin-wr-jax"]
    assert service.repo.evidence_for_player("twin-qb-buf") == []
    assert service.repo.evidence_for_player("twin-wr-jax") == []
    quarantined = service.repo.evidence_for_player(QUARANTINE_PLAYER_ID)
    assert quarantined
    assert quarantined[0].claim_json["quarantine"]["queried_name"] == "Alex Rivers"

    resolved = service.submit_named_evidence(_claim("", status="out"), name="Alex Rivers", team="BUF")
    assert resolved.status == "applied"
    assert resolved.player_id == "twin-qb-buf"
    assert len(service.repo.evidence_for_player("twin-qb-buf")) == 1
    assert service.repo.evidence_for_player("twin-wr-jax") == []


# ---------------------------------------------------------------------------
# E. Clearing rules
# ---------------------------------------------------------------------------


def _activate(service, player_id: str, snapshot) -> None:
    service.activate_event(
        player_id=player_id,
        event_type="injury_status",
        source_snapshot_id=snapshot.id,
        evidence_ids=[],
        policy={"play_probability": 0.25, "injury_status": "Doubtful"},
    )


def test_healthy_primary_snapshot_clears_without_residual_penalty(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    snapshot = _snapshot(db_session)
    _activate(service, "clear-player", snapshot)
    assert service.active_policy_for_player("clear-player")["play_probability"] == 0.25

    assert service.try_clear_for_player("clear-player", snapshot, player_count=11_000) == 1
    assert service.active_policy_for_player("clear-player")["play_probability"] == 1.0
    assert service.rest_of_season_probability("clear-player") == 1.0
    assert (
        service.evaluate_pregame(
            "clear-player", season=2026, week=4, kickoff_at=datetime.now(UTC) + timedelta(days=1)
        ).play_probability
        == 1.0
    )


def test_truncated_or_unhealthy_payload_does_not_clear(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    healthy = _snapshot(db_session)
    _activate(service, "truncated-player", healthy)

    truncated = _snapshot(db_session, is_complete=False, body_hash="truncated")
    assert service.try_clear_for_player("truncated-player", truncated, player_count=11_000) == 0
    assert service.clearance_check(truncated, record_count=11_000).reason == "incomplete_payload"

    small = _snapshot(db_session, body_hash="small")
    assert service.try_clear_for_player("truncated-player", small, player_count=12) == 0
    assert service.clearance_check(small, record_count=12).reason == "implausibly_small_payload"

    stale = _snapshot(db_session, health_verdict="stale", body_hash="stale")
    assert service.try_clear_for_player("truncated-player", stale, player_count=11_000) == 0

    unknown_size = _snapshot(db_session, body_hash="unknown-size")
    assert service.try_clear_for_player("truncated-player", unknown_size) == 0
    assert service.clearance_check(unknown_size).reason == "record_count_unknown"

    assert service.active_policy_for_player("truncated-player")["play_probability"] == 0.25


def test_unrecognized_endpoint_cannot_clear(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    healthy = _snapshot(db_session)
    _activate(service, "endpoint-player", healthy)

    foreign = _snapshot(db_session, endpoint="league/fixture-standard/rosters", body_hash="foreign")
    decision = service.clearance_check(foreign, record_count=50_000)
    assert decision.allowed is False
    assert decision.reason == "unrecognized_primary_source"
    assert service.try_clear_for_player("endpoint-player", foreign, player_count=50_000) == 0
    assert service.active_policy_for_player("endpoint-player")["play_probability"] == 0.25


# ---------------------------------------------------------------------------
# F. Duplicate, stale, and contradictory evidence
# ---------------------------------------------------------------------------


def test_duplicate_evidence_does_not_create_a_second_row(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    first = service.add_evidence(_claim("dupe-player"))
    second = service.add_evidence(_claim("dupe-player"))

    assert second.id == first.id
    assert len(service.repo.evidence_for_player("dupe-player")) == 1


def test_stale_evidence_does_not_override_newer_evidence(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    player_id = "stale-player"
    service.add_evidence(
        _claim(
            player_id,
            status="out",
            published_at="2026-09-12T18:00:00Z",
        )
    )
    assert service.rest_of_season_probability(player_id) == 0.0

    stale = service.add_evidence(
        _claim(
            player_id,
            status="healthy",
            reported_injury=None,
            published_at="2026-09-05T09:00:00Z",
            sources=[
                {
                    "url": "https://www.nfl.com/news/stale-player-cleared",
                    "title": "Cleared to practice (older report)",
                }
            ],
        )
    )

    assert stale.claim_json["superseded_by_newer_basis"] is True
    assert stale.claim_json["applied"] is False
    assert service.rest_of_season_probability(player_id) == 0.0
    assert len(service.repo.evidence_for_player(player_id)) == 2


def test_contradictory_sources_reduce_confidence_and_are_both_kept(db_session: Session):
    from src.app.availability.service import AvailabilityService

    service = AvailabilityService(db_session)
    player_id = "contradiction-player"
    optimistic = service.add_evidence(
        _claim(
            player_id,
            status="questionable",
            claim_confidence=0.8,
            published_at="2026-09-11T12:00:00Z",
        )
    )
    pessimistic = service.add_evidence(
        _claim(
            player_id,
            status="out",
            claim_confidence=0.8,
            published_at="2026-09-11T15:00:00Z",
            sources=[
                {
                    "url": "https://www.nfl.com/news/contradiction-player-out",
                    "title": "Expected to sit",
                }
            ],
        )
    )

    db_session.refresh(optimistic)
    assert optimistic.confidence < 0.8
    assert pessimistic.confidence < 0.8
    assert pessimistic.claim_json["contradicts"] == [optimistic.id]
    assert optimistic.claim_json["contradicted_by"] == [pessimistic.id]

    blended = service.rest_of_season_probability(player_id)
    assert 0.0 < blended < 0.65
