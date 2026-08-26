"""Monte Carlo simulation for season outcome distributions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR, MODEL_V3_DIR, V3_MODELS_DIR
from src.projection.fantasy_points import SCORING


def _score_wide_totals(wide: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=wide.index)
    for stat, weight in SCORING.items():
        if stat in wide.columns:
            total = total + pd.to_numeric(wide[stat], errors="coerce").fillna(0.0) * weight
    return total


def _load_share_manifest() -> dict:
    """Fitted opportunity-share concentrations, when they exist on disk."""
    path = Path(V3_MODELS_DIR) / "opportunity_shares" / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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


SIMULATION_MODE = "full"


def simulate_season_distributions(
    projections: pd.DataFrame,
    *,
    n_draws: int = 1000,
    seed: int = 42,
    mode: str = SIMULATION_MODE,
) -> pd.DataFrame:
    """Draw season stat totals and fantasy points from projection board.

    ``mode=full`` (default) runs the v3 generative path: one team volume draw
    feeds a player's whole stat line, so a player's stats move together.

    ``mode=interim`` is RETIRED as a shipping mode and kept only so
    scripts/compare_simulation_modes.py can still score it. It bootstraps
    cross-fitted residuals per stat INDEPENDENTLY, which destroys the
    +0.62..+0.88 correlation between a player's own stats and understates the
    summed spread by 31%. Held out on 2025 it loses to generative on every
    metric: coverage 0.505 vs 0.537, p50 MAE 34.56 vs 33.86, rho .696 vs .741.

    KNOWN DEFECT, tracked and not fixed here: neither mode produces a
    calibrated band. Generative covers 0.537 against a 0.80 target because
    its uncertainty is sampling noise around a FIXED team volume and FIXED
    conversion rates -- the chance that the projection itself is wrong is not
    represented. See docs/decisions/SIMULATION_MODE_2026-08-26.md.
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
    from src.projection.inference.reconcile import (
        reconcile_v3_generative,
        team_environment_from_board,
    )

    rng = np.random.default_rng(seed)
    players = projections.copy()
    # Each team's fitted RidgeCV anchors, already on the board. The 600/400
    # constants this replaces gave every team the same volume draw.
    team_env = team_environment_from_board(players)
    share_manifest = _load_share_manifest()
    draws = []
    for draw_idx in range(n_draws):
        generative = reconcile_v3_generative(
            players, team_env, rng=rng, share_manifest=share_manifest)
        if generative.empty:
            continue
        # Availability is NOT drawn separately here. The generative path emits
        # season totals allocated from a season of team volume, and the share
        # prior is pred_season, which is already exposure-weighted -- drawing
        # games again and multiplying would apply the same discount twice.
        wide = generative.groupby(
            ["player_id", "position", "team"], observed=True
        ).sum(numeric_only=True).reset_index()
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
    mode: str = SIMULATION_MODE,
) -> dict:
    out_dir = Path(MODEL_V3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    draws = simulate_season_distributions(projections, n_draws=n_draws, mode=mode)
    summary = summarize_simulations(draws)
    draws_path = out_dir / f"simulations_{season}.parquet"
    summary_path = out_dir / f"simulation_summary_{season}.csv"
    draws.to_parquet(draws_path, index=False)
    summary.to_csv(summary_path, index=False)
    # Record WHICH board this was simulated from. Without it a summary from an
    # earlier board merges into a later one unnoticed: the 2026 percentiles
    # outlived a republish that moved QB projections, leaving p50 6.3 points
    # BELOW its own point estimate for QBs and above it everywhere else.
    source_run_id = None
    if "projection_run_id" in projections.columns:
        ids = projections["projection_run_id"].dropna().unique()
        if len(ids) == 1:
            source_run_id = str(ids[0])
    manifest = {
        "season": season,
        "n_draws": n_draws,
        "mode": mode,
        "source_projection_run_id": source_run_id,
        "draws_path": str(draws_path),
        "summary_path": str(summary_path),
    }
    (out_dir / f"simulation_manifest_{season}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
