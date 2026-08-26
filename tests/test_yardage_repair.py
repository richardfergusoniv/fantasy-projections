from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from src.projection.concentration import apply_concentration
from src.projection.contracts import (
    QB_STARTER_VOLUME_SHARES,
    TEAM_VOLUME_SHARES,
    TEAM_VOLUME_SIBLINGS,
)
from src.projection.publish import validate_projection_contract
from src.projection.team_reconcile import reconcile_team_volume
from src.projection.transitions import (
    ROLE_PRIOR_3Y_FEATURE,
    ROLE_PRIOR_FEATURE,
    role_features_for,
    trailing_role_rate,
)


def _calibration(gamma: float = 1.25) -> dict:
    return {
        "version": "test-v1",
        "cells": {
            "WR:receiving": {"exponent": gamma, "promoted": True},
            "RB:rushing": {"exponent": gamma, "promoted": True},
        },
    }


def test_concentration_conserves_orders_and_keeps_zero_zero():
    frame = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "team": ["X", "X", "X"],
        "position": ["WR", "WR", "WR"],
        "stat": ["receiving_yards"] * 3,
        "pred_pg": [60.0, 30.0, 0.0],
        "pred_pg_low": [50.0, 20.0, 0.0],
        "pred_pg_high": [70.0, 40.0, 0.0],
        "projected_games": [17.0] * 3,
        "projected_volume_games": [17.0] * 3,
    })
    out = apply_concentration(frame, _calibration())
    assert out["pred_pg"].sum() == pytest.approx(frame["pred_pg"].sum())
    assert out.loc[0, "pred_pg"] > frame.loc[0, "pred_pg"]
    assert out.loc[0, "pred_pg"] > out.loc[1, "pred_pg"]
    assert out.loc[2, "pred_pg"] == 0.0
    assert out.loc[2, "concentration_scale"] == 1.0
    assert set(out["concentration_calibration_version"]) == {"test-v1"}


def test_concentration_conserves_season_volume_with_status_exposure():
    frame = pd.DataFrame({
        "player_id": ["a", "b"],
        "team": ["X", "X"],
        "position": ["RB", "RB"],
        "stat": ["rushing_yards", "rushing_yards"],
        "pred_pg": [60.0, 40.0],
        "pred_pg_low": [50.0, 30.0],
        "pred_pg_high": [70.0, 50.0],
        "projected_games": [17.0, 8.0],
        "projected_volume_games": [17.0, 8.0],
    })
    before = (frame["pred_pg"] * frame["projected_volume_games"]).sum()
    out = apply_concentration(frame, _calibration())
    after = (out["pred_pg"] * out["projected_volume_games"]).sum()
    assert after == pytest.approx(before)


def test_qb_room_takes_the_measured_share_and_opportunity_does_not_move_tds():
    """The QB anchor share is the measured one, not the structural 1.000.

    1.000 shipped on the reasoning that reconciliation sums the whole room so
    the room owns the whole anchor. Paired scoring on held-out 2023-2025
    (n=312 QB pairs) put that at -0.478 QB mean abs-error, 95% CI
    [-0.86, -0.10], against the measured starter share -- so the reasoning
    lost to the measurement. See QB_STARTER_VOLUME_SHARES and
    scripts/ablate_qb_volume_share.py.
    """
    assert TEAM_VOLUME_SHARES[("QB", "attempts")][1] == QB_STARTER_VOLUME_SHARES["attempts"]
    assert (
        TEAM_VOLUME_SHARES[("QB", "passing_yards")][1]
        == QB_STARTER_VOLUME_SHARES["passing_yards"]
    )
    # TDs stay off the volume scale: reattaching them measured worse on QB
    # (+0.078, CI [-0.26, +0.42]) and partly cancelled the share gain.
    assert "passing_tds" not in TEAM_VOLUME_SIBLINGS[("QB", "attempts")]
    assert "rushing_tds" not in TEAM_VOLUME_SIBLINGS[("RB", "carries")]

    frame = pd.DataFrame({
        "player_id": ["q", "q"],
        "team": ["X", "X"],
        "position": ["QB", "QB"],
        "stat": ["attempts", "passing_tds"],
        "pred_pg": [20.0, 2.0],
        "pred_pg_low": [18.0, 1.0],
        "pred_pg_high": [22.0, 3.0],
        "projected_games": [17.0, 17.0],
        "projected_volume_games": [17.0, 17.0],
        "team_pass_attempts_pg_pred": [35.0, 35.0],
        "depth_tier": [1.0, 1.0],
    })
    out = reconcile_team_volume(frame)
    assert out.loc[out["stat"].eq("attempts"), "pred_pg"].iloc[0] != 20.0
    assert out.loc[out["stat"].eq("passing_tds"), "pred_pg"].iloc[0] == 2.0


def test_three_year_prior_is_weighted_and_stat_specific():
    feat = pd.DataFrame({
        "player_id": ["p", "p", "p"],
        "position": ["WR", "WR", "WR"],
        "season": [2023, 2024, 2025],
        "eligible_weeks": [10.0, 15.0, 5.0],
        "receiving_yards_share_elig": [0.10, 0.20, 0.40],
    })
    prior = trailing_role_rate(feat, ["p"], "WR", "receiving_yards", 2025)
    assert prior.iloc[0] == pytest.approx((0.10 * 10 + 0.20 * 15 + 0.40 * 5) / 30)
    assert ROLE_PRIOR_FEATURE in role_features_for("WR", "receiving_yards")
    assert ROLE_PRIOR_3Y_FEATURE in role_features_for("WR", "receiving_yards")
    assert ROLE_PRIOR_3Y_FEATURE not in role_features_for("WR", "receiving_tds")


def test_projection_contract_rejects_gate_a_as_canonical_exposure():
    frame = pd.DataFrame({
        "season": [2026],
        "projection_run_id": ["run"],
        "composition_version": ["v"],
        "status_override_applied": [False],
        "projected_games": [13.0],
        "projected_volume_games": [13.0],
        "projected_games_raw": [13.0],
    })
    with mock.patch("src.projection.publish.OUTPUT_COLUMNS", list(frame.columns)):
        with pytest.raises(ValueError, match="stale exposure artifact"):
            validate_projection_contract(frame, 2026)


def test_git_revision_is_captured_before_staging_exists():
    """`dirty` must describe the code, not publish's own staging directory.

    tempfile puts the staging dir inside REPO_ROOT so the final os.replace
    stays on one filesystem. Once files are staged there, `git status`
    reports it untracked -- so calling _git_revision() from inside the
    staging block returned dirty:true on every publish without exception.
    A provenance flag that can never say "clean" is worse than none.
    """
    import inspect
    from src.projection import publish as publish_mod

    src = inspect.getsource(publish_mod.publish)
    assert "code_revision = _git_revision()" in src, (
        "_git_revision must be called into a local before staging")
    assert src.index("code_revision = _git_revision()") < src.index(
        "tempfile.TemporaryDirectory"), (
        "_git_revision() must be called BEFORE the staging directory exists")
    assert '"code_revision": code_revision' in src


def test_staging_directory_is_ignored_by_git():
    """Belt and braces: the staging prefix must not register as a change."""
    import subprocess
    import tempfile
    from pathlib import Path
    from src.projection.publish import REPO_ROOT

    with tempfile.TemporaryDirectory(prefix="publish-9999-", dir=REPO_ROOT) as td:
        (Path(td) / "staged.csv").write_text("x\n", encoding="utf-8")
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False).stdout
    assert "publish-9999-" not in out, "staging dir shows as an untracked change"
