"""Source resolution and weekly_props decision provenance."""

from __future__ import annotations

from src.app.projections.source import ProjectionSource, resolve_effective_source


def test_weekly_props_is_production_source():
    assert ProjectionSource.WEEKLY_PROPS.is_production
    assert not ProjectionSource.WEEKLY_PROPS.is_experimental


def test_parse_weekly_props():
    assert ProjectionSource.parse("weekly_props") is ProjectionSource.WEEKLY_PROPS


def test_resolve_effective_weekly_props():
    assert (
        resolve_effective_source(ProjectionSource.WEEKLY_PROPS)
        is ProjectionSource.WEEKLY_PROPS
    )


def test_weekly_prop_shadow_alias_is_honored(monkeypatch):
    from src.app.config import get_settings
    from src.app.projections.source import weekly_props_shadow_only

    monkeypatch.delenv("WEEKLY_PROPS_SHADOW_ONLY", raising=False)
    monkeypatch.setenv("WEEKLY_PROP_SHADOW_ONLY", "false")
    get_settings.cache_clear()
    assert weekly_props_shadow_only() is False
    get_settings.cache_clear()
