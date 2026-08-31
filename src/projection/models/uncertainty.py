"""Leakage-safe projection-uncertainty fitting and draw helpers for v3.

The conversion models already represent sampling noise conditional on a point
forecast.  This module represents the other part of a predictive distribution:
the point forecast itself can be wrong.  Parameters are fitted only from
rolling-origin rows supplied by callers and are stored with their training
cutoff and units.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR, V3_MODELS_DIR
from src.projection.transitions import SEASON_GAMES


UNCERTAINTY_VERSION = "v3_projection_uncertainty_v1"
UNCERTAINTY_DIR = Path(V3_MODELS_DIR) / "uncertainty"
UNCERTAINTY_MANIFEST_PATH = UNCERTAINTY_DIR / "manifest.json"

TEAM_ROWS_PATH = Path(BACKTEST_DIR) / "v3_team_uncertainty_rolling.parquet"
SHARE_ROWS_PATH = Path(BACKTEST_DIR) / "v3_share_uncertainty_rolling.parquet"
AVAILABILITY_ROWS_PATH = Path(BACKTEST_DIR) / "v3_availability_uncertainty_rolling.parquet"
PLAYER_SEASON_ROWS_PATH = Path(BACKTEST_DIR) / "v3_player_season_residuals_rolling.parquet"
JOINT_DONORS_PATH = UNCERTAINTY_DIR / "joint_residual_donors.parquet"

POOL_SPECS = {
    "qb_attempts": ("QB", "attempts"),
    "receiving_targets": (("RB", "WR", "TE"), "targets"),
    "rb_carries": ("RB", "carries"),
}


def _role_bucket(frame: pd.DataFrame) -> pd.Series:
    """Stable broad role bucket available on historical and live boards."""
    tier = pd.to_numeric(frame.get("depth_tier"), errors="coerce")
    if tier is None:
        tier = pd.Series(np.nan, index=frame.index)
    role = frame.get("role", pd.Series("", index=frame.index)).fillna("").astype(str)
    out = pd.Series("depth", index=frame.index, dtype=object)
    out.loc[(tier <= 1) | role.eq("starter")] = "starter"
    out.loc[((tier > 1) & (tier <= 3)) | role.eq("committee")] = "rotation"
    return out


def _season_value(frame: pd.DataFrame) -> pd.Series:
    if "pred_season" in frame.columns:
        season = pd.to_numeric(frame["pred_season"], errors="coerce")
    else:
        season = pd.Series(np.nan, index=frame.index)
    games = pd.to_numeric(
        frame.get("projected_games", pd.Series(SEASON_GAMES, index=frame.index)),
        errors="coerce",
    ).fillna(SEASON_GAMES)
    return season.fillna(pd.to_numeric(frame["pred_pg"], errors="coerce").fillna(0.0) * games)


def extract_uncertainty_rows(
    long_board: pd.DataFrame,
    features: pd.DataFrame,
    target_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build row-level OOF team, share, and availability calibration rows."""
    board = long_board.copy()
    board["player_id"] = board["player_id"].astype(str)
    actual = features[features["season"].eq(target_season)].copy()
    actual["player_id"] = actual["player_id"].astype(str)

    # Team anchors and actuals are both per-game on their source frames; store
    # season-total residuals because that is the unit the simulator consumes.
    anchors = board.groupby("team", as_index=False).first()
    actual_team = actual.groupby("team", as_index=False).first()
    team = anchors[["team"]].copy()
    for pred_col, actual_col, stem in (
        ("team_pass_attempts_pg_pred", "team_pass_attempts_pg", "pass_attempts"),
        ("team_carries_pg_pred", "team_carries_pg", "carries"),
    ):
        team[f"{stem}_pred"] = (
            pd.to_numeric(anchors.get(pred_col), errors="coerce") * SEASON_GAMES
        )
        lookup = actual_team.set_index("team")[actual_col]
        team[f"{stem}_actual"] = (
            team["team"].map(lookup).pipe(pd.to_numeric, errors="coerce") * SEASON_GAMES
        )
        team[f"{stem}_resid"] = team[f"{stem}_actual"] - team[f"{stem}_pred"]
    team["test_season"] = int(target_season)
    team = team.dropna(subset=["pass_attempts_resid", "carries_resid"])

    # Player metadata is one row per player/position.  Expected games must use
    # Gate A's raw estimate rather than the healthy-season display value.
    meta_cols = [c for c in [
        "player_id", "position", "team", "depth_tier", "role",
        "projected_games_raw", "projected_games",
    ] if c in board.columns]
    meta = board[meta_cols].drop_duplicates(["player_id", "position"]).copy()
    meta["role_bucket"] = _role_bucket(meta)
    if "projected_games_raw" in meta.columns:
        expected = pd.to_numeric(meta["projected_games_raw"], errors="coerce")
    else:
        expected = pd.Series(np.nan, index=meta.index)
    meta["expected_games"] = expected.fillna(
        pd.to_numeric(meta.get("projected_games"), errors="coerce")
    ).fillna(SEASON_GAMES).clip(0, SEASON_GAMES)
    games_lookup = actual.set_index("player_id")["games_played"]
    meta["actual_games"] = meta["player_id"].map(games_lookup).pipe(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).clip(0, SEASON_GAMES)
    meta["fragility_bucket"] = np.where(meta["expected_games"] < 14.0, "fragile", "standard")
    meta["test_season"] = int(target_season)
    availability = meta[[
        "test_season", "player_id", "position", "team", "role_bucket",
        "fragility_bucket", "expected_games", "actual_games",
    ]].copy()

    # Opportunity shares are normalized inside the same modeled room on both
    # sides.  This estimates allocation error without confusing it with roster
    # coverage, which is handled by the internal replacement sink.
    actual_index = actual.set_index(["player_id", "position"])
    share_parts: list[pd.DataFrame] = []
    for pool, (positions, stat) in POOL_SPECS.items():
        positions = (positions,) if isinstance(positions, str) else tuple(positions)
        rows = board[board["position"].isin(positions) & board["stat"].eq(stat)].copy()
        if rows.empty:
            continue
        rows["pred_volume"] = _season_value(rows).clip(lower=0.0)
        keys = pd.MultiIndex.from_frame(rows[["player_id", "position"]])
        actual_values = actual_index.reindex(keys)[stat]
        rows["actual_volume"] = pd.Series(
            pd.to_numeric(actual_values.to_numpy(), errors="coerce"), index=rows.index
        ).fillna(0.0)
        pred_total = rows.groupby("team")["pred_volume"].transform("sum")
        actual_total = rows.groupby("team")["actual_volume"].transform("sum")
        rows["pred_share"] = np.where(pred_total > 0, rows["pred_volume"] / pred_total, 0.0)
        rows["actual_share"] = np.where(actual_total > 0, rows["actual_volume"] / actual_total, 0.0)
        role_lookup = meta.set_index(["player_id", "position"])["role_bucket"]
        rows["role_bucket"] = role_lookup.reindex(keys).fillna("depth").to_numpy()
        rows["pool"] = pool
        rows["test_season"] = int(target_season)
        share_parts.append(rows[[
            "test_season", "pool", "team", "player_id", "position", "role_bucket",
            "pred_volume", "actual_volume", "pred_share", "actual_share",
        ]])
    shares = pd.concat(share_parts, ignore_index=True) if share_parts else pd.DataFrame()
    return team, shares, availability


def extract_player_season_rows(
    long_board: pd.DataFrame,
    features: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """Persist complete OOF season-stat residual vectors for fallback draws."""
    board = long_board.copy()
    board["player_id"] = board["player_id"].astype(str)
    board["pred_season_value"] = _season_value(board)
    actual = features[features["season"].eq(target_season)].copy()
    actual["player_id"] = actual["player_id"].astype(str)
    actual_index = actual.set_index(["player_id", "position"])
    keys = pd.MultiIndex.from_frame(board[["player_id", "position"]])
    actual_values = []
    for key, stat in zip(keys, board["stat"]):
        try:
            value = actual_index.loc[key].get(stat, 0.0)
            if isinstance(value, pd.Series):
                value = value.iloc[0]
        except KeyError:
            value = 0.0
        actual_values.append(value)
    board["actual_season_value"] = pd.to_numeric(
        pd.Series(actual_values, index=board.index), errors="coerce").fillna(0.0)
    board["season_resid"] = board["actual_season_value"] - board["pred_season_value"]
    meta = board.drop_duplicates(["player_id", "position"]).copy()
    role_lookup = pd.Series(
        _role_bucket(meta).to_numpy(),
        index=pd.MultiIndex.from_frame(meta[["player_id", "position"]]),
    )
    board["role_bucket"] = role_lookup.reindex(keys).fillna("depth").to_numpy()
    board["test_season"] = int(target_season)
    return board[[
        "test_season", "player_id", "position", "team", "role_bucket", "stat",
        "pred_season_value", "actual_season_value", "season_resid",
    ]].copy()


def _fit_share_concentration(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"concentration": 10.0, "n": 0, "fallback": True}
    mu = frame["pred_share"].to_numpy(dtype=float)
    actual = frame["actual_share"].to_numpy(dtype=float)
    numerator = float(np.sum(mu * (1.0 - mu)))
    denominator = float(np.sum((actual - mu) ** 2))
    concentration = numerator / max(denominator, 1e-9) - 1.0
    return {
        "concentration": float(np.clip(concentration, 1.0, 500.0)),
        "n": int(len(frame)),
        "fallback": False,
    }


def _fit_availability_cell(frame: pd.DataFrame) -> dict:
    if len(frame) < 12:
        return {"concentration": 25.0, "rho": 1.0 / 26.0, "n": int(len(frame)), "fallback": True}
    n_games = float(SEASON_GAMES)
    p = np.clip(frame["expected_games"].to_numpy(dtype=float) / n_games, 0.01, 0.99)
    y = frame["actual_games"].to_numpy(dtype=float)
    base_var = np.sum(n_games * p * (1.0 - p))
    ratio = float(np.sum((y - n_games * p) ** 2) / max(base_var, 1e-9))
    rho = float(np.clip((ratio - 1.0) / max(n_games - 1.0, 1.0), 0.002, 0.35))
    return {
        "concentration": float(1.0 / rho - 1.0),
        "rho": rho,
        "n": int(len(frame)),
        "fallback": False,
    }


def fit_uncertainty_manifest(
    team_rows: pd.DataFrame,
    share_rows: pd.DataFrame,
    availability_rows: pd.DataFrame,
    *,
    training_cutoff: int,
    player_residuals: pd.DataFrame | None = None,
) -> dict:
    """Fit a serializable uncertainty manifest from earlier OOF folds."""
    residual_cols = ["pass_attempts_resid", "carries_resid"]
    if len(team_rows) >= 8:
        covariance = np.cov(team_rows[residual_cols].to_numpy(dtype=float), rowvar=False)
    else:
        covariance = np.diag([45.0 ** 2, 35.0 ** 2])
    covariance = np.asarray(covariance, dtype=float)
    # Numerical guard: project onto the positive-semidefinite cone.
    values, vectors = np.linalg.eigh(covariance)
    covariance = vectors @ np.diag(np.clip(values, 1e-6, None)) @ vectors.T

    pools = {
        pool: _fit_share_concentration(grp)
        for pool, grp in share_rows.groupby("pool", observed=True)
    } if not share_rows.empty else {}
    for pool in POOL_SPECS:
        pools.setdefault(pool, {"concentration": 10.0, "n": 0, "fallback": True})

    availability: dict[str, dict] = {}
    if not availability_rows.empty:
        for position, grp in availability_rows.groupby("position", observed=True):
            availability[f"{position}:all"] = _fit_availability_cell(grp)
        for (position, role), grp in availability_rows.groupby(
            ["position", "role_bucket"], observed=True
        ):
            cell = _fit_availability_cell(grp)
            if cell["fallback"]:
                cell = {**availability.get(f"{position}:all", cell), "fallback": True}
            availability[f"{position}:{role}:all"] = cell
        for (position, role, bucket), grp in availability_rows.groupby(
            ["position", "role_bucket", "fragility_bucket"], observed=True
        ):
            cell = _fit_availability_cell(grp)
            if cell["fallback"]:
                cell = {
                    **availability.get(
                        f"{position}:{role}:all",
                        availability.get(f"{position}:all", cell),
                    ),
                    "fallback": True,
                }
            availability[f"{position}:{role}:{bucket}"] = cell

    conversion_cells: dict[str, dict] = {}
    conversion_defaults: dict[str, dict] = {}
    residuals = player_residuals if player_residuals is not None else pd.DataFrame()
    for kind, numerator, denominator in (
        ("receiving", "receiving_yards", "receptions"),
        ("passing", "passing_yards", "completions"),
        ("rushing", "rushing_yards", "carries"),
    ):
        if residuals.empty:
            continue
        piv_pred = residuals.pivot_table(
            index=["player_id", "test_season", "position"], columns="stat", values="pred")
        piv_actual = residuals.pivot_table(
            index=["player_id", "test_season", "position"], columns="stat", values="actual")
        if numerator not in piv_pred or denominator not in piv_pred:
            continue
        frame = pd.DataFrame({
            "pred": piv_pred[numerator] / piv_pred[denominator],
            "actual": piv_actual[numerator] / piv_actual[denominator],
        }).replace([np.inf, -np.inf], np.nan).dropna()
        frame = frame[(frame["pred"] > 0) & (frame["actual"] > 0)]
        if frame.empty:
            continue
        logs = np.log(frame["actual"] / frame["pred"])
        conversion_defaults[kind] = {"sigma": float(logs.std()), "n": int(len(logs))}
        positions = frame.index.get_level_values("position")
        for position in sorted(set(positions)):
            sub = logs[positions == position]
            if len(sub) >= 30:
                conversion_cells[f"{position}:{kind}"] = {
                    "sigma": float(sub.std()), "n": int(len(sub))}

    manifest = {
        "version": UNCERTAINTY_VERSION,
        "training_cutoff": int(training_cutoff),
        "training_seasons": sorted(
            set(pd.to_numeric(team_rows.get("test_season"), errors="coerce").dropna().astype(int))
        ) if not team_rows.empty else [],
        "units": {
            "team_environment": "season_total_counts",
            "opportunity_shares": "simplex_share",
            "availability": "games_out_of_17",
        },
        "team_environment": {
            "stats": ["pass_attempts", "carries"],
            "residual_mean": [0.0, 0.0],
            "residual_covariance": covariance.tolist(),
            "n": int(len(team_rows)),
        },
        "opportunity_shares": {"pools": pools, "n": int(len(share_rows))},
        "availability": {"cells": availability, "n": int(len(availability_rows))},
        "conversion_sigmas": {
            "cells": conversion_cells,
            "defaults": conversion_defaults,
            "basis": "strictly_earlier_rolling_residuals",
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["artifact_hash"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def write_uncertainty_artifacts(
    manifest: dict,
    team_rows: pd.DataFrame,
    share_rows: pd.DataFrame,
    availability_rows: pd.DataFrame,
    player_season_rows: pd.DataFrame | None = None,
) -> Path:
    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    Path(BACKTEST_DIR).mkdir(parents=True, exist_ok=True)
    team_rows.to_parquet(TEAM_ROWS_PATH, index=False)
    share_rows.to_parquet(SHARE_ROWS_PATH, index=False)
    availability_rows.to_parquet(AVAILABILITY_ROWS_PATH, index=False)
    if player_season_rows is not None:
        player_season_rows.to_parquet(PLAYER_SEASON_ROWS_PATH, index=False)
    UNCERTAINTY_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return UNCERTAINTY_MANIFEST_PATH


def load_uncertainty_manifest() -> dict:
    if not UNCERTAINTY_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(UNCERTAINTY_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def draw_team_environment(
    team_environment: pd.DataFrame,
    manifest: dict,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add a correlated OOF team-volume residual to every team anchor."""
    out = team_environment.copy()
    cell = manifest.get("team_environment") or {}
    covariance = np.asarray(cell.get("residual_covariance", []), dtype=float)
    if covariance.shape != (2, 2) or out.empty:
        return out
    errors = rng.multivariate_normal(np.zeros(2), covariance, size=len(out))
    out["team_pass_attempts_mean"] = np.clip(
        pd.to_numeric(out["team_pass_attempts_mean"], errors="coerce").fillna(600.0)
        + errors[:, 0], 1.0, None,
    )
    out["team_carries_mean"] = np.clip(
        pd.to_numeric(out["team_carries_mean"], errors="coerce").fillna(400.0)
        + errors[:, 1], 1.0, None,
    )
    return out


def draw_availability(
    players: pd.DataFrame,
    manifest: dict,
    *,
    rng: np.random.Generator,
) -> pd.Series:
    """Draw beta-binomial season games, indexed by player id."""
    cols = [c for c in [
        "player_id", "position", "depth_tier", "role",
        "projected_games_raw", "projected_games",
    ] if c in players.columns]
    frame = players[cols].drop_duplicates("player_id").copy()
    if "projected_games_raw" in frame.columns:
        expected = pd.to_numeric(frame["projected_games_raw"], errors="coerce")
    else:
        expected = pd.Series(np.nan, index=frame.index)
    expected = expected.fillna(
        pd.to_numeric(frame.get("projected_games"), errors="coerce")
    ).fillna(SEASON_GAMES).clip(0, SEASON_GAMES)
    frame["fragility_bucket"] = np.where(expected < 14.0, "fragile", "standard")
    frame["role_bucket"] = _role_bucket(frame)
    cells = (manifest.get("availability") or {}).get("cells", {})
    draws = []
    for (_, row), mu in zip(frame.iterrows(), expected.to_numpy(dtype=float)):
        key = (
            f"{row.get('position', '')}:{row['role_bucket']}:"
            f"{row['fragility_bucket']}"
        )
        role_fallback_key = f"{row.get('position', '')}:{row['role_bucket']}:all"
        fallback_key = f"{row.get('position', '')}:all"
        # Accept the pre-v1 position:fragility shape as a read fallback, but
        # newly fitted manifests always use position + role + fragility.
        legacy_key = f"{row.get('position', '')}:{row['fragility_bucket']}"
        concentration = float(
            (
                cells.get(key)
                or cells.get(role_fallback_key)
                or cells.get(legacy_key)
                or cells.get(fallback_key)
                or {}
            ).get("concentration", 25.0)
        )
        p = float(np.clip(mu / SEASON_GAMES, 0.001, 0.999))
        active_p = rng.beta(p * concentration, (1.0 - p) * concentration)
        draws.append(int(rng.binomial(SEASON_GAMES, active_p)))
    return pd.Series(draws, index=frame["player_id"].astype(str), dtype=float)


def build_joint_donors(player_season_rows: pd.DataFrame, scoring: dict[str, float]) -> pd.DataFrame:
    """Collapse whole stat residual vectors to their jointly scored error."""
    frame = player_season_rows[player_season_rows["stat"].isin(scoring)].copy()
    frame["weighted_resid"] = frame["season_resid"] * frame["stat"].map(scoring)
    return frame.groupby(
        ["test_season", "player_id", "position", "role_bucket"], observed=True,
        as_index=False,
    )["weighted_resid"].sum().rename(columns={"weighted_resid": "fantasy_resid"})


def joint_bootstrap_draws(
    generative_draws: pd.DataFrame,
    players: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Center whole-vector season residuals on the generative p50."""
    if generative_draws.empty or donors.empty:
        return pd.DataFrame()
    n_draws = int(generative_draws["draw"].nunique())
    centers = generative_draws.groupby(
        ["player_id", "position", "team"], observed=True, as_index=False
    )["fantasy_pts_season"].median()
    meta = players.drop_duplicates(["player_id", "position"]).copy()
    meta["player_id"] = meta["player_id"].astype(str)
    meta["role_bucket"] = _role_bucket(meta)
    centers["player_id"] = centers["player_id"].astype(str)
    centers = centers.merge(
        meta[["player_id", "position", "role_bucket"]],
        on=["player_id", "position"], how="left",
    )
    by_cell = {
        key: grp["fantasy_resid"].to_numpy(dtype=float)
        for key, grp in donors.groupby(["position", "role_bucket"], observed=True)
    }
    by_position = {
        str(position): grp["fantasy_resid"].to_numpy(dtype=float)
        for position, grp in donors.groupby("position", observed=True)
    }
    rows = []
    for row in centers.itertuples(index=False):
        pool = by_cell.get((row.position, row.role_bucket))
        if pool is None or len(pool) < 12:
            pool = by_position.get(str(row.position), np.array([]))
        if len(pool) == 0:
            residual = np.zeros(n_draws)
        else:
            centered = pool - np.median(pool)
            residual = rng.choice(centered, size=n_draws, replace=True)
            # The published draw set, not merely its donor population, must
            # be centered on generative p50. A finite resample's median can
            # otherwise drift enough to change the displayed p50 and its
            # ranking metrics even though fallback is distribution-only.
            residual = residual - np.median(residual)
        values = np.clip(float(row.fantasy_pts_season) + residual, 0.0, None)
        rows.append(pd.DataFrame({
            "player_id": row.player_id,
            "position": row.position,
            "team": row.team,
            "fantasy_pts_season": values,
            "draw": np.arange(n_draws),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
