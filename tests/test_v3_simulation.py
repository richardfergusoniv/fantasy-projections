"""Tests for the v3 season simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.inference.simulate import (
    SIMULATION_MODE,
    simulate_season_distributions,
    summarize_simulations,
)


def _board():
    """Long board at the real grain: every player carries a volume stat."""
    rows = []
    for stat, value in (("targets", 9.0), ("receptions", 6.0),
                        ("receiving_yards", 60.0), ("receiving_tds", 0.4)):
        rows.append({"player_id": "p1", "position": "WR", "team": "AAA",
                     "stat": stat, "pred_pg": value, "projected_games": 16.0})
    for stat, value in (("carries", 14.0), ("rushing_yards", 60.0),
                        ("rushing_tds", 0.4)):
        rows.append({"player_id": "p2", "position": "RB", "team": "BBB",
                     "stat": stat, "pred_pg": value, "projected_games": 15.0})
    frame = pd.DataFrame(rows)
    frame["pred_season"] = frame["pred_pg"] * frame["projected_games"]
    return frame


def test_default_mode_is_generative():
    """interim is retired: it draws stats independently and covers 0.505."""
    assert SIMULATION_MODE == "full"


def test_simulate_produces_draws_for_every_player():
    draws = simulate_season_distributions(_board(), n_draws=50, seed=1)
    assert "fantasy_pts_season" in draws.columns
    assert set(draws["player_id"]) == {"p1", "p2"}
    assert len(draws) == 100


def test_interim_mode_still_runs_for_comparison():
    """Retired, not deleted -- compare_simulation_modes.py still scores it."""
    draws = simulate_season_distributions(_board(), n_draws=20, seed=1, mode="interim")
    assert not draws.empty
    assert "fantasy_pts_season" in draws.columns


def test_a_player_without_a_volume_stat_produces_no_draws():
    """Explicit, so it cannot be mistaken for a zero projection.

    Allocation keys off the volume stat (targets / carries / attempts). A row
    set lacking it has nothing to allocate, so the player emits no line at all
    and lands in the summary as NaN rather than as a low band. Every position
    on a real board carries its volume stat; this pins the behaviour so a
    future board that does not cannot fail silently.
    """
    board = _board()
    board = board[~((board["player_id"] == "p1") & (board["stat"] == "targets"))]
    draws = simulate_season_distributions(board, n_draws=10, seed=1)
    assert set(draws["player_id"]) == {"p2"}


def test_season_scale_efficiency_noise_decorrelates_volume_stats():
    """KNOWN DEFECT, pinned so a fix can be recognised when it lands.

    Retiring interim was argued on the grounds that a shared volume draw makes
    a player's stats move together. It does not, at season scale: the
    conversion draws multiply volume by a fresh lognormal whose sigma (0.35
    for receiving) was chosen when the path emitted PER-GAME lines. Over a
    season, volume is nearly deterministic (CV ~5%) while that efficiency
    factor carries CV ~36%, so it dominates and yards decorrelate from
    receptions -- measured at 0.13 against the +0.871 seen in real residuals.

    Shrinking sigma restores it (0.15 -> 0.31, 0.05 -> 0.71), so the fix is to
    recalibrate the conversion sigmas for season aggregates. Tracked in
    docs/decisions/SIMULATION_MODE_2026-08-26.md.

    This does not undo the switch: generative still beats interim on coverage,
    p50 MAE and rank. It bounds how much of the remaining coverage gap the
    shared volume draw can be expected to close.
    """
    draws = simulate_season_distributions(_board(), n_draws=300, seed=3)
    wr = draws[draws["player_id"].eq("p1")]
    corr = wr["receiving_yards"].corr(wr["receptions"])
    assert corr < 0.5, (
        f"correlation is {corr:.2f}; if this now exceeds 0.5 the season-scale "
        "sigmas have been recalibrated - update this test and the decision doc")


def test_summarize_percentiles():
    draws = pd.DataFrame({
        "player_id": ["p1"] * 10,
        "position": ["WR"] * 10,
        "team": ["AAA"] * 10,
        "fantasy_pts_season": list(range(100, 110)),
        "draw": list(range(10)),
    })
    summary = summarize_simulations(draws)
    assert "p50" in summary.columns
    assert summary.iloc[0]["p50"] == 104.5
