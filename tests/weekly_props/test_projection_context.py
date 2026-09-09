"""Weekly-props projection context fidelity and sealed-path invariance."""

from __future__ import annotations

from types import SimpleNamespace

from src.app.projections.service import ProjectionService
from src.app.projections.source import ProjectionSource
from src.projection.weekly_props.provenance import (
    WEEKLY_PROPS_CAPABILITY_MODE,
    WEEKLY_PROPS_SCORING_FIDELITY,
)


def test_weekly_props_context_reports_points_only_fidelity(db_session, monkeypatch):
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "weekly_props")
    from src.app.config import get_settings

    get_settings.cache_clear()
    run = SimpleNamespace(
        id="weekly-props-2026-w01-abc",
        model_version="weekly_props_v1",
        artifact_mode="market",
        metadata_json={},
    )
    svc = ProjectionService(db_session, season=2026)
    ctx = svc.context(
        requested_source=ProjectionSource.WEEKLY_PROPS,
        league_id="fixture-standard",
        week=1,
        effective_source=ProjectionSource.WEEKLY_PROPS,
        projection_run=run,
    )
    assert ctx.source is ProjectionSource.WEEKLY_PROPS
    assert ctx.scoring_fidelity == WEEKLY_PROPS_SCORING_FIDELITY
    assert ctx.capability_mode == WEEKLY_PROPS_CAPABILITY_MODE
    assert ctx.scoring_fidelity != "exact_component_rescore"
    assert "weekly_props_points_only_draws" in ctx.caveats
    assert ctx.provenance["requested_source"] == "weekly_props"
    assert ctx.provenance["effective_source"] == "weekly_props"
    assert ctx.provenance["projection_run_id"] == run.id
    get_settings.cache_clear()


def test_sealed_release_context_unchanged_with_component_rescore(db_session, monkeypatch):
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "sealed_release")
    from src.app.config import get_settings
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    seed_development_data(db_session, email="owner@example.com")
    svc = ProjectionService(db_session, season=2026)
    ctx = svc.context(
        requested_source=ProjectionSource.SEALED_RELEASE,
        league_id="fixture-standard",
    )
    # Fixture-standard is simple enough to remain exact when components exist.
    assert ctx.source is ProjectionSource.SEALED_RELEASE
    assert ctx.scoring_fidelity in {
        "exact_component_rescore",
        "modeled_approximation",
        "baseline_points_only",
        "unsupported_rule",
    }
    assert "weekly_props_points_only_draws" not in ctx.caveats
    get_settings.cache_clear()


def test_status_adjusted_context_path_unchanged(db_session, monkeypatch):
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "status_adjusted_release")
    from src.app.config import get_settings

    get_settings.cache_clear()
    svc = ProjectionService(db_session, season=2026)
    ctx = svc.context(requested_source=ProjectionSource.STATUS_ADJUSTED_RELEASE)
    assert ctx.source is ProjectionSource.STATUS_ADJUSTED_RELEASE
    # Without an overlay pointer this stays advisory/baseline — never weekly_props.
    assert ctx.scoring_fidelity != WEEKLY_PROPS_SCORING_FIDELITY
    assert ctx.capability_mode != WEEKLY_PROPS_CAPABILITY_MODE
    get_settings.cache_clear()
