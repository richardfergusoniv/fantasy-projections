"""Shadow 0A: team target pool and compositional receiving target shares."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SHADOW_VERSION = "shadow_0a_target_shares_v1"
TARGET_POOL_STAT = "targets"
RECEIVING_POSITIONS = ("WR", "TE", "RB")
OTHER_TARGET_BUCKET = "other_target_share"


def softmax_shares(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def allocate_target_shares(
    room: pd.DataFrame,
    *,
    depth_prior_col: str = "target_share",
    other_share: float = 0.0,
) -> pd.DataFrame:
    """Allocate team target shares across eligible WR/TE/RB players."""
    if room.empty:
        return room.copy()

    frame = room.copy()
    if depth_prior_col in frame.columns:
        prior = pd.to_numeric(frame[depth_prior_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        prior = np.ones(len(frame))
    if prior.sum() <= 0:
        prior = np.ones(len(frame))
    logits = np.log(np.clip(prior, 1e-6, None))
    player_shares = softmax_shares(logits)
    residual = max(0.0, min(1.0, float(other_share)))
    player_shares = player_shares * (1.0 - residual)
    frame["target_share_pred"] = player_shares
    frame[OTHER_TARGET_BUCKET] = residual
    return frame


def compose_team_target_pool(
    team_targets_pg: float,
    *,
    projected_games: float = 17.0,
) -> float:
    return float(team_targets_pg) * float(projected_games)


def predict_player_targets(
    long_board: pd.DataFrame,
    *,
    team_target_pool_col: str = "team_targets_season_pred",
    depth_prior_col: str = "target_share",
    other_share_by_team: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Shadow 0A output: player-level target season totals from compositional shares."""
    frame = long_board.copy()
    frame = frame[frame["position"].isin(RECEIVING_POSITIONS)]
    if frame.empty:
        return frame

    rows: list[pd.DataFrame] = []
    for team, team_grp in frame.groupby("team", observed=True):
        target_rows = team_grp[team_grp["stat"].eq(TARGET_POOL_STAT)]
        if target_rows.empty:
            continue
        pool = float(
            pd.to_numeric(
                target_rows[team_target_pool_col], errors="coerce"
            ).fillna(0.0).iloc[0]
        )
        eligible = team_grp[team_grp["stat"].eq(TARGET_POOL_STAT)].copy()
        if eligible.empty:
            eligible = team_grp.drop_duplicates("player_id").copy()
        if depth_prior_col not in eligible.columns and "pred_pg" in eligible.columns:
            eligible = eligible.copy()
            eligible[depth_prior_col] = pd.to_numeric(eligible["pred_pg"], errors="coerce")
        other_share = float((other_share_by_team or {}).get(str(team), 0.0))
        allocated = allocate_target_shares(
            eligible.drop_duplicates("player_id"),
            depth_prior_col=depth_prior_col,
            other_share=other_share,
        )
        allocated["pred_season_shadow"] = allocated["target_share_pred"] * pool
        rows.append(allocated)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def validate_share_simplex(
    allocated: pd.DataFrame,
    *,
    team_col: str = "team",
    share_col: str = "target_share_pred",
    other_col: str = OTHER_TARGET_BUCKET,
    tolerance: float = 1e-6,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for team, grp in allocated.groupby(team_col, observed=True):
        player_sum = float(pd.to_numeric(grp[share_col], errors="coerce").fillna(0.0).sum())
        other = float(pd.to_numeric(grp[other_col], errors="coerce").fillna(0.0).iloc[0])
        total = player_sum + other
        results[str(team)] = abs(total - 1.0) <= tolerance
    return results


def write_shadow_artifact(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
