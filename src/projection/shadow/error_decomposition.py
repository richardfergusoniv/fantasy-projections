"""Exact v1 error decomposition for RB/WR shadow attribution.

Components are defined so that, within numerical tolerance:

    raw_rate_error
  + composition_rate_effect
  + availability_effect
  + finalization_remainder
  = v1_prediction - actual_points
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ERROR_COMPONENT_COLS = (
    "raw_rate_error",
    "composition_rate_effect",
    "availability_effect",
    "finalization_remainder",
)

DEFAULT_ATOL = 1e-6


def decompose_prediction_error(
    frame: pd.DataFrame,
    *,
    v1_col: str = "v1_pred",
    actual_col: str = "actual_points",
    raw_rate_col: str = "raw_rate_ppg",
    composed_rate_col: str = "composed_rate_ppg",
    projected_games_col: str = "projected_games",
    actual_games_col: str = "actual_games_played",
    atol: float = DEFAULT_ATOL,
) -> pd.DataFrame:
    """Attach exact error components; raise if they do not sum to total error."""
    out = frame.copy()
    v1 = pd.to_numeric(out[v1_col], errors="coerce").fillna(0.0)
    actual = pd.to_numeric(out[actual_col], errors="coerce").fillna(0.0)
    raw_rate = pd.to_numeric(out[raw_rate_col], errors="coerce").fillna(0.0)
    composed_rate = pd.to_numeric(out[composed_rate_col], errors="coerce").fillna(0.0)
    projected_games = pd.to_numeric(out[projected_games_col], errors="coerce").fillna(0.0)
    actual_games = pd.to_numeric(out[actual_games_col], errors="coerce").fillna(0.0)

    out["total_error"] = v1 - actual
    out["raw_rate_error"] = raw_rate * actual_games - actual
    out["composition_rate_effect"] = (composed_rate - raw_rate) * actual_games
    out["availability_effect"] = composed_rate * (projected_games - actual_games)
    out["finalization_remainder"] = v1 - composed_rate * projected_games

    component_sum = (
        out["raw_rate_error"]
        + out["composition_rate_effect"]
        + out["availability_effect"]
        + out["finalization_remainder"]
    )
    residual = component_sum - out["total_error"]
    out["decomposition_residual"] = residual
    if not np.allclose(residual.to_numpy(dtype=float), 0.0, atol=atol, equal_nan=True):
        worst = float(np.nanmax(np.abs(residual.to_numpy(dtype=float))))
        raise ValueError(
            f"Error components do not sum to total error within atol={atol}; "
            f"max |residual|={worst}"
        )
    return out


def stage_point_deltas(
    stage_scores: dict[str, dict[str, dict]],
    *,
    player_ids: list[str] | None = None,
    value_key: str = "fantasy_ppg",
) -> pd.DataFrame:
    """Long-form deltas between consecutive compose checkpoints."""
    from src.projection.composition import COMPOSE_CHECKPOINT_NAMES

    stages = [name for name in COMPOSE_CHECKPOINT_NAMES if name in stage_scores]
    if len(stages) < 2:
        return pd.DataFrame()
    ids = player_ids or sorted(
        {pid for stage in stages for pid in stage_scores[stage]}
    )
    rows = []
    for pid in ids:
        prev_name = stages[0]
        prev_val = stage_scores[prev_name].get(pid, {}).get(value_key)
        for name in stages[1:]:
            val = stage_scores[name].get(pid, {}).get(value_key)
            rows.append({
                "player_id": pid,
                "stage": name,
                "prior_stage": prev_name,
                value_key: val,
                "delta_from_prior": (
                    None if val is None or prev_val is None else float(val) - float(prev_val)
                ),
                "position": stage_scores[name].get(pid, {}).get("position"),
            })
            prev_name = name
            prev_val = val
    return pd.DataFrame(rows)
