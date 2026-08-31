"""Leakage-safe WR residual-scale calibration for recentered distributions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import TOP_ADP, sha256_file
from src.projection.inference.recenter import (
    POINTS_COL,
    TRANSFORM_VERSION as RECENTER_VERSION,
    player_draw_medians,
    recenter_draws,
)

WR_RESIDUAL_SCALE_GRID = (
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.40,
    1.50,
    1.60,
    1.70,
    1.80,
)
NOMINAL_COVERAGE = 0.80
TRAINING_FOLDS = ((2023, 2024),)
TRANSFORM_VERSION = "v1_wr_residual_scale"
ARTIFACT_PATH = Path(OUTPUT_DIR) / "model_v3" / "wr_calibration.json"
ALPHA = 0.20


def _apply_median_correction(
    draws: pd.DataFrame,
    selected_points: pd.Series,
    *,
    points_col: str = POINTS_COL,
    player_col: str = "player_id",
) -> pd.DataFrame:
    out = draws.copy()
    for _ in range(12):
        post_medians = (
            out.groupby(player_col, observed=True)[points_col]
            .median()
            .astype(float)
        )
        correction = selected_points.reindex(post_medians.index).astype(float) - post_medians
        if correction.abs().max() < 1e-9:
            break
        out[points_col] = out[points_col] + out[player_col].map(correction).fillna(0.0)
        out[points_col] = np.maximum(out[points_col].to_numpy(dtype=float), 0.0)
    return out


def recenter_draws_wr_scaled(
    draws: pd.DataFrame,
    selected_points: Mapping[str, float] | pd.Series,
    *,
    wr_scale: float = 1.0,
    points_col: str = POINTS_COL,
    player_col: str = "player_id",
    position_col: str = "position",
) -> pd.DataFrame:
    """Recenter draws with an optional WR-only residual scale.

    For WR rows:

        y' = max(0, m_i + s_WR * (y_v3 - q50_v3))

    Other positions use ``s = 1``.  Median correction forces p50 = ``selected``.
    """
    if draws.empty:
        return draws.copy()
    if wr_scale <= 0:
        raise ValueError("wr_scale must be positive")
    if np.isclose(wr_scale, 1.0):
        return recenter_draws(
            draws,
            selected_points,
            points_col=points_col,
            player_col=player_col,
        )

    selected = pd.Series(selected_points, dtype=float)
    selected.index = selected.index.astype(str)
    v3_p50 = player_draw_medians(draws, points_col=points_col, player_col=player_col)
    v3_p50.index = v3_p50.index.astype(str)

    out = draws.copy()
    out[player_col] = out[player_col].astype(str)
    anchor = out[player_col].map(selected).astype(float)
    baseline = out[player_col].map(v3_p50).astype(float)
    residual = pd.to_numeric(out[points_col], errors="coerce").astype(float) - baseline
    is_wr = out[position_col].astype(str).eq("WR")
    scaled_residual = residual.copy()
    scaled_residual.loc[is_wr] = float(wr_scale) * residual.loc[is_wr]
    out[points_col] = np.maximum((anchor + scaled_residual).to_numpy(dtype=float), 0.0)
    return _apply_median_correction(
        out,
        selected,
        points_col=points_col,
        player_col=player_col,
    )


def _interval_score(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(
        np.mean(
            hi - lo
            + (2.0 / ALPHA) * np.maximum(lo - actual, 0.0)
            + (2.0 / ALPHA) * np.maximum(actual - hi, 0.0)
        )
    )


def score_wr_scale(
    draws: pd.DataFrame,
    holdout: pd.DataFrame,
    wr_scale: float,
    *,
    selected_col: str = "selected_pred",
) -> dict:
    """Score one WR scale on a holdout frame."""
    holdout = holdout.copy()
    holdout["player_id"] = holdout["player_id"].astype(str)
    selected = holdout.set_index("player_id")[selected_col].astype(float)
    recentered = recenter_draws_wr_scaled(draws, selected, wr_scale=wr_scale)
    quantiles = (
        recentered.groupby("player_id", observed=True)[POINTS_COL]
        .quantile([0.10, 0.50, 0.90])
        .unstack()
    )
    quantiles.columns = ["p10", "p50", "p90"]
    joined = (
        quantiles.reset_index()
        .merge(
            holdout[["player_id", "position", "actual_points"]],
            on="player_id",
            how="inner",
        )
        .dropna(subset=["actual_points", "p10", "p90"])
    )
    wr = joined[joined["position"].eq("WR")]
    if wr.empty:
        return {
            "wr_scale": float(wr_scale),
            "wr_n": 0,
            "wr_coverage": float("nan"),
            "wr_interval_score": float("nan"),
            "overall_coverage": float("nan"),
            "overall_interval_score": float("nan"),
        }
    y = wr["actual_points"].to_numpy(dtype=float)
    lo = wr["p10"].to_numpy(dtype=float)
    hi = wr["p90"].to_numpy(dtype=float)
    wr_cov = float(np.mean((lo <= y) & (y <= hi)))
    wr_is = _interval_score(y, lo, hi)

    y_all = joined["actual_points"].to_numpy(dtype=float)
    lo_all = joined["p10"].to_numpy(dtype=float)
    hi_all = joined["p90"].to_numpy(dtype=float)
    return {
        "wr_scale": float(wr_scale),
        "wr_n": int(len(wr)),
        "wr_coverage": wr_cov,
        "wr_interval_score": wr_is,
        "overall_coverage": float(np.mean((lo_all <= y_all) & (y_all <= hi_all))),
        "overall_interval_score": _interval_score(y_all, lo_all, hi_all),
    }


def select_wr_residual_scale(
    fold_scores: list[dict],
    *,
    wr_floor: float = 0.70,
    overall_floor: float = 0.75,
) -> dict:
    """Pick scale from training folds with coverage floors, then WR target proximity."""
    if not fold_scores:
        raise ValueError("fold_scores must not be empty")

    by_scale: dict[float, list[dict]] = {}
    for row in fold_scores:
        by_scale.setdefault(float(row["wr_scale"]), []).append(row)

    candidates: list[dict] = []
    for scale, rows in by_scale.items():
        total_wr_n = sum(int(r["wr_n"]) for r in rows)
        if total_wr_n == 0:
            continue
        wr_cov = sum(float(r["wr_coverage"]) * int(r["wr_n"]) for r in rows) / total_wr_n
        total_n = sum(int(r.get("overall_n", r["wr_n"])) for r in rows)
        overall_cov = (
            sum(float(r["overall_coverage"]) * int(r.get("overall_n", r["wr_n"])) for r in rows)
            / total_n
        )
        wr_is = sum(float(r["wr_interval_score"]) * int(r["wr_n"]) for r in rows) / total_wr_n
        candidates.append({
            "wr_scale": scale,
            "wr_coverage": wr_cov,
            "overall_coverage": overall_cov,
            "wr_interval_score": wr_is,
            "folds": len(rows),
            "wr_n": total_wr_n,
        })

    if not candidates:
        raise ValueError("no valid WR scale candidates")

    wr_eligible = [c for c in candidates if c["wr_coverage"] >= wr_floor]
    pool = wr_eligible or candidates
    overall_eligible = [c for c in pool if c["overall_coverage"] >= overall_floor]
    if overall_eligible:
        pool = overall_eligible

    def sort_key(row: dict) -> tuple:
        return (
            abs(float(row["wr_coverage"]) - NOMINAL_COVERAGE),
            float(row["wr_interval_score"]),
            float(row["wr_scale"]),
        )

    selected = min(pool, key=sort_key)
    return {
        "selected_wr_scale": float(selected["wr_scale"]),
        "selection_metric": {
            "target_coverage": NOMINAL_COVERAGE,
            "wr_coverage": selected["wr_coverage"],
            "overall_coverage": selected["overall_coverage"],
            "wr_interval_score": selected["wr_interval_score"],
            "distance_to_target": abs(selected["wr_coverage"] - NOMINAL_COVERAGE),
            "wr_floor": wr_floor,
            "overall_floor": overall_floor,
            "used_overall_floor": bool(overall_eligible),
        },
        "grid": WR_RESIDUAL_SCALE_GRID,
        "candidates": sorted(candidates, key=sort_key),
    }


def load_training_holdout(season: int) -> pd.DataFrame:
    """Top-ADP evaluation frame for a training target season."""
    eval_path = Path(OUTPUT_DIR) / "accuracy_first_2026" / "evaluation_players.parquet"
    frame = pd.read_parquet(eval_path)
    frame = frame[frame["season"].eq(int(season))].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    adp = pd.to_numeric(frame["adp"], errors="coerce")
    frame = frame.loc[adp.notna() & adp.le(TOP_ADP)].copy()
    if frame["selected_pred"].isna().all():
        frame["selected_pred"] = pd.to_numeric(frame["incumbent_pred"], errors="coerce")
    return frame


def generate_fold_draws(
    source_season: int,
    target_season: int,
    *,
    n_draws: int,
    out_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Generate/cache exact-path joint-bootstrap draws for one training fold."""
    from src.projection.data_prep import get_conn
    from src.projection.fantasy_evaluation import build_leakage_safe_long_board
    from src.projection.features import build_player_season_features
    from src.projection.fantasy_points import SCORING
    from src.projection.inference.simulate import simulate_season_distributions, slim_draw_frame
    from src.projection.models.uncertainty import (
        AVAILABILITY_ROWS_PATH,
        PLAYER_SEASON_ROWS_PATH,
        SHARE_ROWS_PATH,
        TEAM_ROWS_PATH,
        build_joint_donors,
        fit_uncertainty_manifest,
        joint_bootstrap_draws,
    )

    calibration_path = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
    calibration_hash = sha256_file(calibration_path)
    out_path = out_path or (
        Path(OUTPUT_DIR) / "model_v3" / f"training_draws_{target_season}.parquet"
    )
    meta_path = out_path.with_suffix(".json")
    if not force and out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            int(meta.get("n_draws", -1)) == n_draws
            and meta.get("calibration_sha256") == calibration_hash
            and meta.get("draws_sha256") == sha256_file(out_path)
        ):
            return pd.read_parquet(out_path)

    uncertainty_rows = {
        "team": pd.read_parquet(TEAM_ROWS_PATH),
        "share": pd.read_parquet(SHARE_ROWS_PATH),
        "availability": pd.read_parquet(AVAILABILITY_ROWS_PATH),
        "player_season": pd.read_parquet(PLAYER_SEASON_ROWS_PATH),
        "residuals": pd.read_parquet(Path(BACKTEST_DIR) / "residuals_rolling.parquet"),
    }
    conn = get_conn()
    features = build_player_season_features(conn)
    board = build_leakage_safe_long_board(conn, features, source_season, target_season)
    conn.close()
    training = lambda frame: frame[frame["test_season"].lt(target_season)].copy()
    manifest = fit_uncertainty_manifest(
        training(uncertainty_rows["team"]),
        training(uncertainty_rows["share"]),
        training(uncertainty_rows["availability"]),
        training_cutoff=target_season - 1,
        player_residuals=training(uncertainty_rows["residuals"]),
    )
    option_draws = simulate_season_distributions(
        board,
        n_draws=n_draws,
        seed=100 + target_season,
        mode="full",
        uncertainty_manifest=manifest,
        use_projection_uncertainty=True,
    )
    donors = build_joint_donors(
        uncertainty_rows["player_season"][
            uncertainty_rows["player_season"]["test_season"].lt(target_season)
        ],
        SCORING,
    )
    draws = joint_bootstrap_draws(
        option_draws,
        board,
        donors,
        rng=np.random.default_rng(900 + target_season),
    )
    slim = slim_draw_frame(draws)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out_path, index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_season": int(source_season),
        "target_season": int(target_season),
        "n_draws": int(n_draws),
        "calibration_sha256": calibration_hash,
        "draws_sha256": sha256_file(out_path),
        "n_rows": int(len(slim)),
        "path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return slim


def fit_wr_calibration(
    *,
    n_draws: int,
    force_regenerate: bool = False,
) -> dict:
    """Select WR residual scale from pre-holdout training folds only."""
    fold_scores: list[dict] = []
    fold_meta: list[dict] = []
    for source_season, target_season in TRAINING_FOLDS:
        draws = generate_fold_draws(
            source_season,
            target_season,
            n_draws=n_draws,
            force=force_regenerate,
        )
        holdout = load_training_holdout(target_season)
        selected_col = "selected_pred"
        if holdout["selected_pred"].isna().all():
            selected_col = "incumbent_pred"
        for scale in WR_RESIDUAL_SCALE_GRID:
            scored = score_wr_scale(draws, holdout, scale, selected_col=selected_col)
            scored["target_season"] = int(target_season)
            scored["overall_n"] = int(len(holdout))
            fold_scores.append(scored)
        fold_meta.append({
            "source_season": int(source_season),
            "target_season": int(target_season),
            "holdout_n": int(len(holdout)),
            "selected_col": selected_col,
        })

    selection = select_wr_residual_scale(fold_scores)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transform_version": TRANSFORM_VERSION,
        "recenter_version": RECENTER_VERSION,
        "training_folds": fold_meta,
        "fold_scores": fold_scores,
        **selection,
        "promotion_state": "wr_calibration_candidate",
    }
    return payload


def write_wr_calibration(payload: dict, path: Path | None = None) -> Path:
    out = Path(path or ARTIFACT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_wr_calibration(path: Path | None = None) -> dict | None:
    out = Path(path or ARTIFACT_PATH)
    if not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))
