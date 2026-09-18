"""Persist shadow weekly_props candidates and honor env aliases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.app.config import Settings, get_settings
from src.app.persistence.models import (
    ActiveProjectionPointer,
    PlayerProjection,
    PromotionEvent,
    ProjectionRun,
)
from src.app.projections.source import WEEKLY_PROPS_SHADOW_ENV_KEYS, weekly_props_shadow_only
from src.app.releases.gates import GateResult
from src.app.releases.publication import (
    Candidate,
    CandidateRow,
    active_pointer,
    publish,
)
from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.projection.weekly_props.gates import validate_snapshots
from src.projection.weekly_props.persist import (
    activate_last_passing_weekly_props,
    load_latest_provider_snapshots,
    persist_provider_snapshots,
    richer_stored_snapshots,
)
from src.projection.weekly_props.provenance import weekly_props_context_fields
from src.projection.weekly_props.provenance import (
    WEEKLY_PROPS_POINTER_MODE,
    load_promoted_weekly_props_run,
)


def _gate(passed: bool = True) -> GateResult:
    return GateResult(passed=passed, failures=[] if passed else ["nope"], warnings=[])


def _candidate(*, run_id: str, n_players: int = 52, status_meta: str = "test") -> Candidate:
    rows = tuple(
        CandidateRow(
            player_id=f"p{i:03d}",
            team="BUF",
            opponent="NE",
            availability_probability=1.0,
            mean_json={"points": 10.0, "position": "WR", "name": f"Player {i}"},
            quantiles_json={"0.5": 10.0},
        )
        for i in range(n_players)
    )
    return Candidate(
        mode=WEEKLY_PROPS_POINTER_MODE,
        season=2026,
        week=1,
        run_id=run_id,
        model_version="weekly_props_v1",
        input_hash="abc123def4567890",
        manifest_uri="fixture://weekly-props-test",
        artifact_mode="market",
        partition_mode="weekly",
        rows=rows,
        as_of=datetime.now(UTC),
        metadata={"derivation": "weekly_props_v1", "note": status_meta},
    )


def test_weekly_prop_shadow_typo_env_does_not_enable_shadow(monkeypatch):
    monkeypatch.delenv("WEEKLY_PROPS_FORCE_SHADOW", raising=False)
    monkeypatch.delenv("WEEKLY_PROPS_SHADOW_ONLY", raising=False)
    monkeypatch.setenv("weekly_prop_shadow", "true")
    get_settings.cache_clear()
    assert weekly_props_shadow_only() is False
    get_settings.cache_clear()


def test_canonical_shadow_env_is_ignored_in_favor_of_auto_promote(monkeypatch):
    monkeypatch.delenv("WEEKLY_PROPS_FORCE_SHADOW", raising=False)
    monkeypatch.setenv("WEEKLY_PROPS_SHADOW_ONLY", "true")
    monkeypatch.setenv("weekly_prop_shadow", "true")
    get_settings.cache_clear()
    assert weekly_props_shadow_only() is False
    get_settings.cache_clear()


def test_force_shadow_env_enables_shadow(monkeypatch):
    monkeypatch.setenv("WEEKLY_PROPS_FORCE_SHADOW", "true")
    monkeypatch.setenv("WEEKLY_PROPS_SHADOW_ONLY", "false")
    get_settings.cache_clear()
    assert weekly_props_shadow_only() is True
    get_settings.cache_clear()


def test_settings_alias_accepts_weekly_prop_shadow(monkeypatch):
    monkeypatch.delenv("WEEKLY_PROPS_SHADOW_ONLY", raising=False)
    get_settings.cache_clear()
    settings = Settings(weekly_prop_shadow="false")  # type: ignore[call-arg]
    assert settings.weekly_props_shadow_only is False


def test_shadow_publish_persists_rows_without_swapping_weekly_v2(db_session):
    weekly_v2 = ProjectionRun(
        id="weekly-2026-w01-v2keep",
        mode="weekly",
        season=2026,
        week=1,
        as_of=datetime.now(UTC),
        model_version="weekly_v2_fixture",
        input_hash="v2",
        status="active",
        artifact_mode="trained",
    )
    db_session.add(weekly_v2)
    db_session.flush()
    db_session.add(
        ActiveProjectionPointer(
            mode="weekly", season=2026, week=1, run_id=weekly_v2.id
        )
    )
    db_session.flush()

    candidate = _candidate(run_id="weekly-props-2026-w01-shadowok")
    result = publish(
        db_session,
        candidate,
        gates={"ok": _gate(True)},
        activate=False,
    )
    assert result.promoted is False
    assert result.reason == "shadow_only"
    assert result.run_id == candidate.run_id

    stored = db_session.query(ProjectionRun).filter_by(id=candidate.run_id).one()
    assert stored.status == "shadow"
    assert db_session.query(PlayerProjection).filter_by(run_id=candidate.run_id).count() == 52

    weekly_pointer = active_pointer(db_session, mode="weekly", season=2026, week=1)
    assert weekly_pointer is not None
    assert weekly_pointer.run_id == weekly_v2.id
    props_pointer = active_pointer(
        db_session, mode=WEEKLY_PROPS_POINTER_MODE, season=2026, week=1
    )
    assert props_pointer is None
    assert load_promoted_weekly_props_run(db_session, season=2026, week=1) is None


def test_activate_closing_line_swaps_weekly_props_pointer_only(db_session):
    weekly_v2 = ProjectionRun(
        id="weekly-2026-w01-v2keep2",
        mode="weekly",
        season=2026,
        week=1,
        as_of=datetime.now(UTC),
        model_version="weekly_v2_fixture",
        input_hash="v2",
        status="active",
        artifact_mode="trained",
    )
    db_session.add(weekly_v2)
    db_session.flush()
    db_session.add(
        ActiveProjectionPointer(
            mode="weekly", season=2026, week=1, run_id=weekly_v2.id
        )
    )
    db_session.flush()
    candidate = _candidate(run_id="weekly-props-2026-w01-closeline")
    publish(db_session, candidate, gates={"ok": _gate(True)}, activate=False)

    result = activate_last_passing_weekly_props(db_session, season=2026, week=1)
    assert result is not None
    assert result.promoted is True
    assert result.reason == "activated_closing_line"

    promoted = load_promoted_weekly_props_run(db_session, season=2026, week=1)
    assert promoted is not None
    assert promoted.id == candidate.run_id
    weekly_pointer = active_pointer(db_session, mode="weekly", season=2026, week=1)
    assert weekly_pointer is not None
    assert weekly_pointer.run_id == weekly_v2.id


def test_weekly_v2_weekly_pointer_is_not_vegas_props(db_session):
    weekly_v2 = ProjectionRun(
        id="weekly-2026-w01-notprops",
        mode="weekly",
        season=2026,
        week=1,
        as_of=datetime.now(UTC),
        model_version="weekly_v2_fixture",
        input_hash="v2",
        status="active",
        artifact_mode="trained",
    )
    db_session.add(weekly_v2)
    db_session.flush()
    db_session.add(
        ActiveProjectionPointer(
            mode="weekly", season=2026, week=1, run_id=weekly_v2.id
        )
    )
    db_session.flush()
    assert load_promoted_weekly_props_run(db_session, season=2026, week=1) is None


def test_provider_snapshot_roundtrip_and_catalog(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_BACKEND", "local")
    monkeypatch.setenv("ARTIFACT_LOCAL_ROOT", str(tmp_path))
    get_settings.cache_clear()
    now = datetime.now(UTC)
    quote = NormalizedQuote(
        source="draftkings",
        sportsbook="DraftKings",
        event_id="e1",
        game_id="g1",
        player_name_raw="A.J. Brown",
        player_id="00-brown",
        team="PHI",
        opponent="DAL",
        market="rec_yards",
        period="full_game",
        line=70.5,
        over_odds=-110,
        under_odds=-110,
        fetched_at=now,
        event_start=now + timedelta(days=1),
        source_url="https://example.test",
        raw_record_hash="abc",
    )
    snap = ProviderSnapshot(
        source="draftkings",
        season=2026,
        week=1,
        fetched_at=now,
        urls=("https://example.test",),
        quotes=(quote,),
        success=True,
    )
    cloned = ProviderSnapshot.from_dict(snap.to_dict())
    assert cloned.source == "draftkings"
    assert cloned.quotes[0].player_name_raw == "A.J. Brown"
    uris = persist_provider_snapshots(db_session, [snap])
    assert uris
    loaded = load_latest_provider_snapshots(db_session, season=2026, week=1, sources=("draftkings",))
    assert len(loaded) == 1
    assert loaded[0].quotes[0].line == 70.5
    get_settings.cache_clear()


def test_persist_provider_snapshots_fails_closed_when_artifact_unreadable(
    db_session, tmp_path, monkeypatch
):
    """Do not catalog a healthy source_snapshot when verify-after-upload fails."""
    from unittest.mock import patch

    import pytest

    from src.app.artifacts.store import ArtifactError, LocalArtifactStore
    from src.app.persistence.models import SourceSnapshot

    monkeypatch.setenv("ARTIFACT_BACKEND", "local")
    monkeypatch.setenv("ARTIFACT_LOCAL_ROOT", str(tmp_path))
    get_settings.cache_clear()
    snap = _snap(n_quotes=5)

    original = LocalArtifactStore.verify_readable

    def wipe_then_verify(self, uri: str) -> None:  # noqa: ANN001
        path = self._resolve(uri)
        path.unlink(missing_ok=True)
        original(self, uri)

    with patch.object(LocalArtifactStore, "verify_readable", wipe_then_verify):
        with pytest.raises(ArtifactError, match="not durable"):
            persist_provider_snapshots(db_session, [snap])

    rows = (
        db_session.query(SourceSnapshot)
        .filter(SourceSnapshot.endpoint == "weekly_props:draftkings:2026:1")
        .all()
    )
    assert rows == [], "fail-closed must not leave a healthy/complete catalog row"
    get_settings.cache_clear()


def test_persist_provider_snapshots_verify_success_catalogs_healthy(
    db_session, tmp_path, monkeypatch
):
    from src.app.persistence.models import SourceSnapshot

    monkeypatch.setenv("ARTIFACT_BACKEND", "local")
    monkeypatch.setenv("ARTIFACT_LOCAL_ROOT", str(tmp_path))
    get_settings.cache_clear()
    snap = _snap(n_quotes=5)
    uris = persist_provider_snapshots(db_session, [snap])
    assert len(uris) == 1
    row = (
        db_session.query(SourceSnapshot)
        .filter(SourceSnapshot.endpoint == "weekly_props:draftkings:2026:1")
        .one()
    )
    assert row.health_verdict == "healthy"
    assert row.is_complete is True
    assert row.artifact_uri == uris[0]
    get_settings.cache_clear()


def _quote(i: int, *, fetched_at: datetime, source: str = "draftkings") -> NormalizedQuote:
    return NormalizedQuote(
        source=source,
        sportsbook="DraftKings",
        event_id="e1",
        game_id="g1",
        player_name_raw=f"Player {i}",
        player_id=f"00-p{i:03d}",
        team="PHI",
        opponent="DAL",
        market="rec_yards",
        period="full_game",
        line=70.5,
        over_odds=-110,
        under_odds=-110,
        fetched_at=fetched_at,
        event_start=fetched_at + timedelta(days=1),
        source_url="https://example.test",
        raw_record_hash=f"h{i}",
    )


def _snap(
    *,
    source: str = "draftkings",
    n_quotes: int = 10,
    fetched_at: datetime | None = None,
    success: bool = True,
) -> ProviderSnapshot:
    when = fetched_at or datetime.now(UTC)
    quotes = (
        tuple(_quote(i, fetched_at=when, source=source) for i in range(n_quotes))
        if success
        else ()
    )
    return ProviderSnapshot(
        source=source,
        season=2026,
        week=1,
        fetched_at=when,
        urls=("https://example.test",),
        quotes=quotes,
        success=success,
    )


def test_shadow_env_keys_are_shared_with_settings():
    from src.app.config import WEEKLY_PROPS_SHADOW_ENV_KEYS as from_config

    assert from_config is WEEKLY_PROPS_SHADOW_ENV_KEYS
    assert from_config[0] == "WEEKLY_PROPS_SHADOW_ONLY"


def test_richer_stored_when_live_failed():
    now = datetime.now(UTC)
    stored = [_snap(n_quotes=80, fetched_at=now - timedelta(hours=12))]
    live_failed = [_snap(n_quotes=0, fetched_at=now, success=False)]
    assert richer_stored_snapshots(stored, live_failed, now=now)
    assert richer_stored_snapshots(stored, [], now=now)


def test_richer_stored_when_stored_has_more_quotes():
    now = datetime.now(UTC)
    stored = [_snap(n_quotes=80, fetched_at=now - timedelta(hours=6))]
    live = [_snap(n_quotes=10, fetched_at=now)]
    picked = richer_stored_snapshots(stored, live, now=now)
    assert len(picked) == 1
    assert len(picked[0].quotes) == 80


def test_richer_stored_rejected_when_thinner():
    now = datetime.now(UTC)
    stored = [_snap(n_quotes=10, fetched_at=now - timedelta(hours=6))]
    live = [_snap(n_quotes=40, fetched_at=now)]
    assert richer_stored_snapshots(stored, live, now=now) == []


def test_richer_stored_rejected_when_same_age():
    now = datetime.now(UTC)
    stored = [_snap(n_quotes=80, fetched_at=now)]
    live = [_snap(n_quotes=10, fetched_at=now)]
    assert richer_stored_snapshots(stored, live, now=now) == []


def test_richer_stored_rejected_when_past_closing_line_age():
    now = datetime.now(UTC)
    stored = [_snap(n_quotes=80, fetched_at=now - timedelta(hours=80))]
    live = [_snap(n_quotes=0, fetched_at=now, success=False)]
    assert richer_stored_snapshots(stored, live, now=now) == []


def test_richer_stored_accepts_naive_and_aware_datetimes():
    now = datetime(2026, 9, 12, 18, 0, tzinfo=UTC)
    naive = datetime(2026, 9, 12, 10, 0, 0)  # treated as UTC
    stored = [_snap(n_quotes=80, fetched_at=naive)]
    live = [_snap(n_quotes=10, fetched_at=now)]
    assert richer_stored_snapshots(stored, live, now=now)


def test_ignore_age_records_stale_warning_but_still_uses_snapshot():
    now = datetime.now(UTC)
    snap = _snap(n_quotes=5, fetched_at=now - timedelta(hours=48))
    gated = validate_snapshots([snap], season=2026, week=1, now=now, ignore_age=True)
    assert gated.passed
    assert any(item.startswith("stale_snapshot:draftkings:") for item in gated.warnings)


def test_ignore_age_still_skips_closing_line_past_max_age():
    now = datetime.now(UTC)
    snap = _snap(n_quotes=5, fetched_at=now - timedelta(hours=80))
    gated = validate_snapshots([snap], season=2026, week=1, now=now, ignore_age=True)
    assert not gated.passed
    assert any(item.startswith("stale_snapshot:draftkings:") for item in gated.warnings)
    assert any(item.startswith("closing_line_too_stale:draftkings:") for item in gated.warnings)


def test_activate_closing_line_stamps_age_and_warns_past_live_freshness(db_session):
    candidate = _candidate(run_id="weekly-props-2026-w01-age48")
    publish(db_session, candidate, gates={"ok": _gate(True)}, activate=False)
    run = db_session.query(ProjectionRun).filter_by(id=candidate.run_id).one()
    run.as_of = datetime.now(UTC) - timedelta(hours=48)
    db_session.flush()

    result = activate_last_passing_weekly_props(db_session, season=2026, week=1)
    assert result is not None
    assert result.promoted is True
    freshness = result.gates["closing_line_freshness"]
    assert freshness["passed"] is True
    assert freshness["warnings"]
    event = (
        db_session.query(PromotionEvent)
        .filter(PromotionEvent.candidate_run_id == candidate.run_id)
        .order_by(PromotionEvent.created_at.desc())
        .first()
    )
    assert event is not None
    assert event.promoted is True
    assert event.validation_json["closing_line_age_hours"] >= 47
    assert event.validation_json["max_closing_line_age_hours"] == 72.0


def test_stale_closing_line_is_not_activated(db_session):
    candidate = _candidate(run_id="weekly-props-2026-w01-age120")
    publish(db_session, candidate, gates={"ok": _gate(True)}, activate=False)
    run = db_session.query(ProjectionRun).filter_by(id=candidate.run_id).one()
    run.as_of = datetime.now(UTC) - timedelta(hours=120)
    db_session.flush()

    result = activate_last_passing_weekly_props(db_session, season=2026, week=1)
    assert result is None
    assert load_promoted_weekly_props_run(db_session, season=2026, week=1) is None


def test_weekly_props_context_includes_snapshot_age():
    run = ProjectionRun(
        id="weekly-props-2026-w01-ctxage",
        mode="weekly_props",
        season=2026,
        week=1,
        as_of=datetime.now(UTC) - timedelta(hours=40),
        model_version="weekly_props_v1",
        input_hash="ctxagehash000000",
        status="active",
        artifact_mode="market",
    )
    fields = weekly_props_context_fields(run=run, week=1)
    assert fields["snapshot_age_hours"] is not None
    assert fields["snapshot_age_hours"] >= 39
    assert any(item.startswith("snapshot_age_hours:") for item in fields["caveats"])
    assert any(item.startswith("stale_closing_line:") for item in fields["caveats"])
