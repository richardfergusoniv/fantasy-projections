"""Monte Carlo simulation for season outcome distributions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR, MODEL_V3_DIR
from src.projection.fantasy_points import SCORING


def _score_wide_totals(wide: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=wide.index)
    for stat, weight in SCORING.items():
        if stat in wide.columns:
            total = total + pd.to_numeric(wide[stat], errors="coerce").fillna(0.0) * weight
    return total


def _load_residuals() -> pd.DataFrame:
    path = Path(BACKTEST_DIR) / "residuals_rolling.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["position", "stat", "team", "resid"])


def _row_residual_pools(stat_rows: pd.DataFrame, residuals: pd.DataFrame) -> list[np.ndarray]:
    """Map each projection row to a residual bootstrap pool."""
    if residuals.empty:
        return [np.array([0.0]) for _ in range(len(stat_rows))]
    by_team = {
        key: grp["resid"].to_numpy()
        for key, grp in residuals.groupby(["position", "stat", "team"], observed=True)
    }
    by_stat = {
        key: grp["resid"].to_numpy()
        for key, grp in residuals.groupby(["position", "stat"], observed=True)
    }
    pools: list[np.ndarray] = []
    for row in stat_rows.itertuples(index=False):
        pool = by_team.get((row.position, row.stat, row.team))
        if pool is None or len(pool) == 0:
            pool = by_stat.get((row.position, row.stat), np.array([0.0]))
        pools.append(pool)
    return pools


def simulate_season_distributions(
    projections: pd.DataFrame,
    *,
    n_draws: int = 1000,
    seed: int = 42,
    mode: str = "interim",
) -> pd.DataFrame:
    """Draw season stat totals and fantasy points from projection board.

    ``mode=interim`` bootstraps cross-fitted residuals by team-position room.
    ``mode=full`` uses the v3 generative reconcile path when team environment
    columns are present on the input frame.
    """
    if mode == "full":
        return _simulate_full_generative(projections, n_draws=n_draws, seed=seed)
    rng = np.random.default_rng(seed)
    residuals = _load_residuals()
    stat_rows = projections.copy()
    stat_rows["pred_pg"] = pd.to_numeric(stat_rows["pred_pg"], errors="coerce").fillna(0.0)
    stat_rows["projected_games"] = pd.to_numeric(
        stat_rows["projected_games"], errors="coerce"
    ).fillna(17.0)

    base_pred = stat_rows["pred_pg"].to_numpy()
    row_pools = _row_residual_pools(stat_rows, residuals)
    noise_matrix = np.vstack([
        rng.choice(pool, size=n_draws) if len(pool) else np.zeros(n_draws)
        for pool in row_pools
    ]).T  # shape (n_draws, n_rows)

    player_games = stat_rows.groupby("player_id", observed=True)["projected_games"].first()
    players = player_games.index.to_numpy()
    probs = np.clip(player_games.to_numpy() / 17.0, 0.05, 1.0)
    games_matrix = rng.binomial(17, probs, size=(n_draws, len(players)))

    meta = stat_rows[["player_id", "position", "team", "stat"]].copy()
    draws = []
    for draw_idx in range(n_draws):
        draw_games = stat_rows["player_id"].map(
            pd.Series(games_matrix[draw_idx], index=players)
        ).fillna(17.0)
        meta["season_total"] = (
            np.clip(base_pred + noise_matrix[draw_idx, :], 0, None) * draw_games.to_numpy()
        )
        wide = meta.pivot_table(
            index=["player_id", "position", "team"],
            columns="stat",
            values="season_total",
            aggfunc="first",
        ).reset_index()
        wide["fantasy_pts_season"] = _score_wide_totals(wide)
        wide["draw"] = draw_idx
        draws.append(wide)
    return pd.concat(draws, ignore_index=True)


def _simulate_full_generative(
    projections: pd.DataFrame,
    *,
    n_draws: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    from src.projection.inference.reconcile import reconcile_v3_generative
    from src.projection.models.availability import draw_games_played

    rng = np.random.default_rng(seed)
    players = projections.copy()
    team_env = (
        players[["team"]]
        .drop_duplicates()
        .assign(
            team_pass_attempts_mean=600.0,
            team_carries_mean=400.0,
        )
    )
    draws = []
    for draw_idx in range(n_draws):
        generative = reconcile_v3_generative(players, team_env, rng=rng)
        if generative.empty:
            continue
        games = draw_games_played(
            players.groupby("player_id")["projected_games"].first(), rng=rng
        )
        generative["draw"] = draw_idx
        wide = generative.groupby(["player_id", "position", "team"], observed=True).sum(numeric_only=True).reset_index()
        wide["fantasy_pts_season"] = _score_wide_totals(wide)
        wide["draw"] = draw_idx
        draws.append(wide)
    return pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()


def summarize_simulations(draws: pd.DataFrame) -> pd.DataFrame:
    """Aggregate draw-level fantasy points to player percentiles."""
    quantiles = draws.groupby(
        ["player_id", "position", "team"], observed=True
    )["fantasy_pts_season"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    quantiles.columns = ["p10", "p25", "p50", "p75", "p90"]
    return quantiles.reset_index()


def write_simulation_outputs(
    projections: pd.DataFrame,
    season: int,
    *,
    n_draws: int = 1000,
    mode: str = "interim",
) -> dict:
    out_dir = Path(MODEL_V3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    draws = simulate_season_distributions(projections, n_draws=n_draws, mode=mode)
    summary = summarize_simulations(draws)
    draws_path = out_dir / f"simulations_{season}.parquet"
    summary_path = out_dir / f"simulation_summary_{season}.csv"
    draws.to_parquet(draws_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "season": season,
        "n_draws": n_draws,
        "mode": mode,
        "draws_path": str(draws_path),
        "summary_path": str(summary_path),
    }
    (out_dir / f"simulation_manifest_{season}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
