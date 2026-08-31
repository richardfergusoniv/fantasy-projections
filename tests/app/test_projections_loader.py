"""Release bundle loader and worker job tests."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("APP_ENV", "test")


def test_release_bundle_loader_reads_active_release():
    from src.app.projections.loader import ReleaseBundleLoader

    loader = ReleaseBundleLoader(season=2026)
    players = loader.load()
    if not players:
        pytest.skip("no active release bundle in workspace")
    assert len(players) > 100
    sample = next(iter(players.values()))
    assert sample.mean_points > 0
    assert sample.position in {"QB", "RB", "WR", "TE", "K", "DEF"}


def test_special_teams_draws_are_deterministic():
    from src.projection.special_teams.models import KickerContext, TeamContext, simulate_dst_draw, simulate_kicker_draw

    dst_a = simulate_dst_draw(TeamContext(), seed=42)
    dst_b = simulate_dst_draw(TeamContext(), seed=42)
    assert dst_a == dst_b
    k_a = simulate_kicker_draw(KickerContext(), seed=7)
    k_b = simulate_kicker_draw(KickerContext(), seed=7)
    assert k_a == k_b


def test_release_bridge_imports_players(db_session):
    from src.app.releases.bridge import ReleaseBridge

    bridge = ReleaseBridge(db_session)
    run_id = bridge.sync_preseason_pointer(2026)
    if run_id is None:
        pytest.skip("no active release pointer")
    from src.app.persistence.models import PlayerProjection

    count = db_session.query(PlayerProjection).filter(PlayerProjection.run_id == run_id).count()
    assert count > 0
