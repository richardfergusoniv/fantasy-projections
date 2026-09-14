"""Persist shadow weekly_props candidates and honor env aliases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.app.config import Settings, get_settings
from src.app.persistence.models import (
    ActiveProjectionPointer,
    PlayerProjection,
    ProjectionRun,
)
from src.app.projections.source import weekly_props_shadow_only
from src.app.releases.gates import GateResult
from src.app.releases.publication import (
    Candidate,
    CandidateRow,
    active_pointer,
    publish,
)
from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.projection.weekly_props.persist import (
    activate_last_passing_weekly_props,
    load_latest_provider_snapshots,
    persist_provider_snapshots,
)
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


def test_weekly_prop_shadow_typo_env_disables_shadow(monkeypatch):
    monkeypatch.delenv("WEEKLY_PROPS_SHADOW_ONLY", raising=False)
    monkeypatch.setenv("weekly_prop_shadow", "false")
    get_settings.cache_clear()
    assert weekly_props_shadow_only() is False
    get_settings.cache_clear()


def test_canonical_shadow_env_wins_over_typo(monkeypatch):
    monkeypatch.setenv("WEEKLY_PROPS_SHADOW_ONLY", "false")
    monkeypatch.setenv("weekly_prop_shadow", "true")
    get_settings.cache_clear()
    assert weekly_props_shadow_only() is False
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
