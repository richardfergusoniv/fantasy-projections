"""Unit and contract tests for the calibrated v3 season distribution."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.projection.models.opportunity_shares import allocate_opportunities
from src.projection.models.uncertainty import (
    draw_availability,
    draw_team_environment,
    fit_uncertainty_manifest,
    joint_bootstrap_draws,
    load_uncertainty_manifest,
)


def _fit_rows():
    team = pd.DataFrame({
        "test_season": [2023] * 20,
        "pass_attempts_resid": np.linspace(-50.0, 50.0, 20),
        "carries_resid": np.linspace(30.0, -30.0, 20),
    })
    shares = pd.DataFrame({
        "test_season": [2023] * 18,
        "pool": ["qb_attempts"] * 6 + ["receiving_targets"] * 6 + ["rb_carries"] * 6,
        "pred_share": [0.7, 0.3] * 9,
        "actual_share": [0.6, 0.4, 0.8, 0.2, 0.65, 0.35] * 3,
    })
    availability = pd.DataFrame({
        "test_season": [2023] * 48,
        "position": ["QB"] * 24 + ["WR"] * 24,
        "role_bucket": (["starter"] * 12 + ["depth"] * 12) * 2,
        "fragility_bucket": (["standard"] * 6 + ["fragile"] * 6) * 4,
        "expected_games": [16.0, 15.0, 17.0, 16.0, 15.0, 17.0,
                           11.0, 12.0, 10.0, 11.0, 12.0, 10.0] * 4,
        "actual_games": [17.0, 14.0, 16.0, 15.0, 17.0, 13.0,
                         8.0, 15.0, 4.0, 13.0, 10.0, 6.0] * 4,
    })
    return team, shares, availability


def test_fit_uncertainty_has_psd_covariance_and_role_availability_cells():
    team, shares, availability = _fit_rows()
    manifest = fit_uncertainty_manifest(
        team, shares, availability, training_cutoff=2023)
    covariance = np.asarray(
        manifest["team_environment"]["residual_covariance"], dtype=float)
    assert np.linalg.eigvalsh(covariance).min() >= 0.0
    assert manifest["training_cutoff"] == 2023
    assert manifest["training_seasons"] == [2023]
    assert manifest["units"]["team_environment"] == "season_total_counts"
    assert "QB:starter:standard" in manifest["availability"]["cells"]
    assert manifest["opportunity_shares"]["pools"]["qb_attempts"]["concentration"] > 0
    assert len(manifest["artifact_hash"]) == 64


def test_team_and_availability_draws_are_seeded_nonnegative_and_bounded():
    team, shares, availability = _fit_rows()
    manifest = fit_uncertainty_manifest(
        team, shares, availability, training_cutoff=2023)
    environment = pd.DataFrame({
        "team": ["A", "B"],
        "team_pass_attempts_mean": [10.0, 600.0],
        "team_carries_mean": [5.0, 400.0],
    })
    first = draw_team_environment(
        environment, manifest, rng=np.random.default_rng(7))
    second = draw_team_environment(
        environment, manifest, rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(first, second)
    assert (first[["team_pass_attempts_mean", "team_carries_mean"]] >= 1.0).all().all()

    players = pd.DataFrame({
        "player_id": ["q", "q", "w"],
        "position": ["QB", "QB", "WR"],
        "role": ["starter", "starter", "depth"],
        "depth_tier": [1, 1, 4],
        "projected_games_raw": [16.0, 16.0, 11.0],
    })
    games_a = draw_availability(players, manifest, rng=np.random.default_rng(19))
    games_b = draw_availability(players, manifest, rng=np.random.default_rng(19))
    pd.testing.assert_series_equal(games_a, games_b)
    assert games_a.index.is_unique
    assert games_a.between(0, 17).all()


def test_inactive_room_goes_to_internal_replacement_sink():
    room = pd.DataFrame({
        "player_id": ["a", "b"],
        "team": ["KC", "KC"],
        "position": ["WR", "TE"],
        "stat": ["targets", "targets"],
        "_allocation_prior": [0.0, 0.0],
    })
    out = allocate_opportunities(
        room,
        550.0,
        rng=np.random.default_rng(1),
        group_cols=["team", "stat"],
        prior_col="_allocation_prior",
        allow_replacement_sink=True,
    )
    assert out["allocated_volume"].sum() == 0.0
    assert out.attrs["replacement_sink_volume"] == pytest.approx(550.0)


def test_unavailable_player_stays_zero_and_active_claims_conserve_volume():
    room = pd.DataFrame({
        "player_id": ["active", "inactive"],
        "team": ["KC", "KC"],
        "position": ["WR", "WR"],
        "stat": ["targets", "targets"],
        "_allocation_prior": [100.0, 0.0],
    })
    out = allocate_opportunities(
        room,
        500.0,
        rng=np.random.default_rng(2),
        prior_col="_allocation_prior",
        allow_replacement_sink=True,
    ).set_index("player_id")
    assert out.loc["inactive", "allocated_volume"] == 0.0
    assert out["allocated_volume"].sum() == pytest.approx(500.0)


def test_joint_bootstrap_uses_one_corrected_draw_set_centered_on_generative_p50():
    generative = pd.DataFrame({
        "player_id": ["p"] * 5,
        "position": ["WR"] * 5,
        "team": ["KC"] * 5,
        "draw": range(5),
        "fantasy_pts_season": [90.0, 95.0, 100.0, 105.0, 110.0],
    })
    players = pd.DataFrame({
        "player_id": ["p"], "position": ["WR"], "team": ["KC"],
        "depth_tier": [1], "role": ["starter"],
    })
    donors = pd.DataFrame({
        "position": ["WR"] * 12,
        "role_bucket": ["starter"] * 12,
        "fantasy_resid": np.arange(-60.0, 60.0, 10.0),
    })
    corrected = joint_bootstrap_draws(
        generative, players, donors, rng=np.random.default_rng(3))
    assert len(corrected) == 5
    assert corrected["draw"].tolist() == list(range(5))
    assert corrected["fantasy_pts_season"].ge(0).all()
    assert corrected["fantasy_pts_season"].median() == pytest.approx(100.0)


def test_corrupt_uncertainty_manifest_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        "src.projection.models.uncertainty.UNCERTAINTY_MANIFEST_PATH", path)
    assert load_uncertainty_manifest() == {}


def test_calibrator_and_live_writer_share_the_full_simulator():
    import scripts.calibrate_v3_distribution as calibration
    from src.projection.inference import simulate

    calibration_source = inspect.getsource(calibration)
    writer_source = inspect.getsource(simulate.write_simulation_outputs)
    assert "simulate_season_distributions" in calibration_source
    assert 'mode="full"' in calibration_source
    assert "simulate_season_distributions(" in writer_source
