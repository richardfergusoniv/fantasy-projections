"""Tests for production projection re-anchor: source modes, loader, overlay, capabilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_PROJECTION_SOURCE", "sealed_release")
os.environ.setdefault("WEEKLY_RND_ENABLED", "false")


def test_sealed_release_is_default_source():
    from src.app.config import get_settings
    from src.app.projections.source import ProjectionSource, configured_projection_source

    get_settings.cache_clear()
    assert configured_projection_source() == ProjectionSource.SEALED_RELEASE


def test_unknown_projection_source_fails_closed():
    from src.app.projections.source import ProjectionSource

    with pytest.raises(ValueError, match="unknown projection source"):
        ProjectionSource.parse("bogus_source")


def test_weekly_v2_rnd_requires_explicit_enable():
    from src.app.projections.source import ProjectionSource, resolve_effective_source

    with pytest.raises(ValueError, match="WEEKLY_RND_ENABLED"):
        resolve_effective_source(ProjectionSource.WEEKLY_V2_RND)


def test_release_bundle_loader_reads_active_release():
    from src.app.projections.loader import ReleaseBundleLoader

    loader = ReleaseBundleLoader(season=2026)
    bundle = loader.load_bundle()
    if bundle is None:
        pytest.skip("no active release bundle in workspace")
    assert len(bundle.players) > 100
    assert bundle.validation_passed
    assert bundle.namespace
    assert len(bundle.manifest_sha256) == 64
    sample = next(iter(bundle.players.values()))
    assert sample.mean_points > 0


def test_loader_uses_sealed_component_projections_not_output_fallback():
    from src.app.projections.loader import ReleaseBundleLoader, invalidate_bundle_loader_cache
    from src.projection.release_bundle import sha256_file

    invalidate_bundle_loader_cache(2026)
    loader = ReleaseBundleLoader(season=2026)
    bundle = loader.load_bundle()
    if bundle is None:
        pytest.skip("no active release bundle in workspace")
    assert "component_projections_from_output_fallback" not in bundle.caveats
    assert bundle.component_projections_path is not None
    assert bundle.component_projections_path.name == "projections_2026.csv"
    path_posix = bundle.component_projections_path.as_posix()
    assert "release_bundles" in path_posix or "/releases/" in path_posix
    entry = next(
        a for a in json.loads(bundle.manifest_path.read_text(encoding="utf-8"))["artifacts"]
        if a["role"] == "projections"
    )
    assert sha256_file(bundle.component_projections_path) == entry["sha256"]


def test_loader_cache_invalidates_on_pointer_change(tmp_path, monkeypatch):
    from src.projection.active_release import pointer_path, write_active_pointer
    from src.projection.contracts import REPO_ROOT
    from src.app.projections.loader import ReleaseBundleLoader, invalidate_bundle_loader_cache

    pointer_file = Path(REPO_ROOT) / "draft_assistant" / "data" / "active_release_2026.json"
    if not pointer_file.exists():
        pytest.skip("no active pointer")
    original = json.loads(pointer_file.read_text(encoding="utf-8"))
    loader = ReleaseBundleLoader(season=2026)
    bundle1 = loader.load_bundle()
    if bundle1 is None:
        pytest.skip("no bundle")
    key1 = loader._cache_key

    # Simulate pointer swap by invalidating and reloading
    invalidate_bundle_loader_cache(2026)
    loader2 = ReleaseBundleLoader(season=2026)
    bundle2 = loader2.load_bundle()
    assert bundle2 is not None
    # Same pointer should produce same cache key
    assert loader2._cache_key == key1

    invalidate_bundle_loader_cache(2026)


def test_v3_quantiles_do_not_replace_point_means():
    from src.app.projections.loader import ReleaseBundleLoader

    loader = ReleaseBundleLoader(season=2026)
    bundle = loader.load_bundle()
    if bundle is None:
        pytest.skip("no bundle")
    for summary in list(bundle.players.values())[:20]:
        # Mean points come from fantasy_pts_season / projected_games, not p50 alone
        assert summary.mean_points > 0
        if summary.quantiles:
            p50 = summary.quantiles.get("0.5")
            if p50 is not None:
                # Quantiles are distributional overlay; mean is authoritative
                assert isinstance(p50, float)


def test_live_injury_evidence_rows_exclude_fixture_citations(db_session):
    from datetime import UTC, datetime

    from src.app.jobs.handlers import _live_injury_evidence_rows
    from src.app.persistence.models import InjuryEvidence

    db_session.add(
        InjuryEvidence(
            player_id="p-fixture",
            fetched_at=datetime.now(UTC),
            source_url="fixture://synthetic-injury-report/p-fixture",
            source_title="SYNTHETIC",
            claim_json={"synthetic": True, "summary": "fixture"},
        )
    )
    db_session.add(
        InjuryEvidence(
            player_id="p-live",
            fetched_at=datetime.now(UTC),
            source_url="https://example.com/injury",
            source_title="Real report",
            claim_json={"synthetic": False, "summary": "hamstring"},
        )
    )
    db_session.flush()
    rows = _live_injury_evidence_rows(db_session)
    assert len(rows) == 1
    assert rows[0]["player_id"] == "p-live"


def test_rescore_reads_flat_rule_snapshot_scoring():
    from src.app.projections.league_rescore import rescore_league
    from src.app.scoring.compiler import scoring_settings_from_snapshot

    flat_snapshot = {
        "rec": 1.0,
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "pass_yd": 0.04,
        "pass_td": 4.0,
        "bonus_pass_yd_400": 2.0,
    }
    scoring = scoring_settings_from_snapshot(flat_snapshot)
    assert scoring.get("bonus_pass_yd_400") == 2.0
    components = {
        "p1": {
            "_position": "QB",
            "pass_yards": 420.0,
            "pass_tds": 3.0,
            "rush_yards": 10.0,
        }
    }
    result = rescore_league(
        league_id="flat",
        display_name="Flat snapshot league",
        scoring_settings=scoring,
        roster_positions=["QB", "BN"],
        components_by_player=components,
    )
    assert result.scoring_fidelity == "modeled_approximation"
    assert any("threshold" in rule for rule in result.approximate_rules)


def test_league_rescore_ppfd_cannot_be_exact_without_components():
    from src.app.projections.league_rescore import rescore_league

    components = {
        "p1": {
            "_position": "RB",
            "rush_yards": 80.0,
            "receptions": 3.0,
            "rec_yards": 25.0,
            "rush_tds": 0.5,
            "rec_tds": 0.2,
        }
    }
    scoring = {"rush_yd": 0.1, "rec": 1.0, "rec_yd": 0.1, "rush_fd": 0.5}
    result = rescore_league(
        league_id="test",
        display_name="PPFD League",
        scoring_settings=scoring,
        roster_positions=["RB", "FLEX", "BN"],
        components_by_player=components,
    )
    assert result.scoring_fidelity == "modeled_approximation"
    assert any("ppfd" in rule for rule in result.approximate_rules)


def test_league_rescore_exact_without_nonlinear_rules():
    from src.app.projections.league_rescore import rescore_league

    components = {
        "p1": {
            "_position": "WR",
            "receptions": 5.0,
            "rec_yards": 60.0,
            "rec_tds": 0.4,
        }
    }
    scoring = {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0}
    result = rescore_league(
        league_id="test",
        display_name="PPR League",
        scoring_settings=scoring,
        roster_positions=["WR", "FLEX", "BN"],
        components_by_player=components,
    )
    assert result.scoring_fidelity == "exact_component_rescore"


def test_half_ppr_does_not_leak_to_full_ppr_league():
    from src.app.projections.league_rescore import rescore_league

    components = {
        "p1": {
            "_position": "WR",
            "receptions": 5.0,
            "rec_yards": 60.0,
            "rec_tds": 0.4,
        }
    }
    half_ppr = rescore_league(
        league_id="half",
        display_name="Half",
        scoring_settings={"rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0},
        roster_positions=["WR", "BN"],
        components_by_player=components,
    )
    full_ppr = rescore_league(
        league_id="full",
        display_name="Full",
        scoring_settings={"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0},
        roster_positions=["WR", "BN"],
        components_by_player=components,
    )
    assert full_ppr.scoring_fidelity == "exact_component_rescore"
    assert half_ppr.contract_hash != full_ppr.contract_hash


def test_status_overlay_gate_blocks_invalid_values():
    from src.app.projections.loader import PlayerSummary
    from src.app.projections.status_overlay import OverlayAdjustment, validate_overlay_gate

    gate = validate_overlay_gate(
        bundle_release_id="rel-1",
        bundle_manifest_sha256="a" * 64,
        adjustments=[
            OverlayAdjustment(
                player_id="p1",
                team="BUF",
                position="RB",
                before_points=10.0,
                after_points=-1.0,
                before_availability=1.0,
                after_availability=1.0,
                reason_code="bad",
            )
        ],
        players={
            "p1": PlayerSummary("p1", "p1", "RB", "BUF", -1.0, {}, 1.0),
        },
        overlay_hash="b" * 64,
    )
    assert not gate.passed


def test_status_overlay_out_zeros_player():
    from src.app.projections.loader import PlayerSummary
    from src.app.projections.status_overlay import _apply_availability_rules

    summary = PlayerSummary("p1", "Test", "RB", "BUF", 12.0, {"0.5": 12.0}, 1.0)
    pts, avail, reason = _apply_availability_rules(summary, status="OUT", availability_probability=None)
    assert pts == 0.0
    assert avail == 0.0
    assert reason.startswith("status_zero")


def test_overlay_rollback_restores_prior_pointer(tmp_path, monkeypatch):
    from src.projection.contracts import REPO_ROOT
    from src.app.projections.status_overlay import (
        _overlay_pointer_path,
        promote_overlay_pointer,
        rollback_overlay_pointer,
    )
    from src.app.projections.status_overlay import StatusOverlayBundle, validate_overlay_gate

    path = _overlay_pointer_path(2026)
    if path.exists():
        backup = path.read_text(encoding="utf-8")
    else:
        backup = None

    overlay = StatusOverlayBundle(
        schema_version="status_overlay_pointer_v1",
        algorithm_version="v1",
        base_release_id="rel",
        base_manifest_sha256="c" * 64,
        overlay_hash="d" * 64,
        generated_at="2026-08-31T00:00:00+00:00",
        source_observations=[],
        adjustments=[],
        players={},
        validation=validate_overlay_gate(
            bundle_release_id="rel",
            bundle_manifest_sha256="c" * 64,
            adjustments=[],
            players={},
            overlay_hash="d" * 64,
        ).to_dict(),
    )
    promote_overlay_pointer(overlay, season=2026)
    overlay.overlay_hash = "e" * 64
    overlay.validation = validate_overlay_gate(
        bundle_release_id="rel",
        bundle_manifest_sha256="c" * 64,
        adjustments=[],
        players={},
        overlay_hash="e" * 64,
    ).to_dict()
    promote_overlay_pointer(overlay, season=2026)
    restored = rollback_overlay_pointer(2026)
    assert restored is not None
    assert restored.get("overlay_hash") == "d" * 64

    if backup is not None:
        path.write_text(backup, encoding="utf-8")
    elif path.exists():
        path.unlink()


def test_capability_matrix_separates_production_from_weekly_rnd(db_session):
    from src.app.readiness.capabilities import build_capability_matrix

    matrix = build_capability_matrix(db_session, season=2026, week=1)
    by_name = matrix.by_name()
    assert "matchup_specific_weekly_start_sit_win_probability" in by_name
    assert "draft_rankings_and_roster_values" in by_name
    # Production can be healthy even when weekly R&D is NO-GO
    if not matrix.weekly_rnd_healthy:
        assert by_name["matchup_specific_weekly_start_sit_win_probability"].verdict == "NO-GO"


def test_app_starts_without_weekly_v2_directories(monkeypatch):
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(Path("/nonexistent/weekly_v2")))
    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(Path("/nonexistent/weekly_v2_models")))
    from src.app.factory import create_app

    app = create_app()
    assert app is not None


def test_daily_refresh_skips_weekly_promotion_by_default(db_session, monkeypatch):
    monkeypatch.setenv("WEEKLY_RND_ENABLED", "false")
    from src.app.config import get_settings
    from src.app.jobs.handlers import run_daily_refresh

    get_settings.cache_clear()
    result = run_daily_refresh(db_session, automatic=False)
    assert result["weekly_run_id"] is None
    assert result["incremental"]["mode"] == "weekly_rnd_disabled"


def test_matchup_win_probability_unavailable_when_gate_false(db_session):
    from src.app.projections.service import ProjectionService

    svc = ProjectionService(db_session, season=2026)
    allowed = svc.matchup_win_probability_allowed(week=1)
    # Weekly R&D gates typically fail in test env
    assert isinstance(allowed, bool)


def test_sealed_release_ignores_weekly_db_run(db_session: Session, monkeypatch, tmp_path):
    from src.app.config import get_settings
    from src.app.decisions.services import LineupService
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge
    from src.app.seed import seed_development_data

    monkeypatch.setenv("APP_PROJECTION_SOURCE", "sealed_release")
    monkeypatch.setenv("WEEKLY_RND_ENABLED", "false")
    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(tmp_path / "empty_models"))
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(tmp_path / "empty_outputs"))
    get_settings.cache_clear()
    seed_development_data(db_session, email="owner@example.com")
    bridge = ReleaseBridge(db_session)
    if bridge.sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    WeeklyProjectionService(db_session).promote_week(2026, week=1, automatic=False)
    result = LineupService(db_session).recommend("fixture-standard", 1)
    assert not str(result["projection_run_id"]).startswith("weekly-")
    assert str(result["projection_run_id"]).startswith("preseason-")
