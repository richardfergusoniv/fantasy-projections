"""Leakage-safe evaluation for Shadow 0A target opportunity modeling."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.projection.evaluation.accuracy_first import TOP_ADP
from src.projection.shadow.reconciliation import reconciliation_burden_score, team_target_residual
from src.projection.shadow.share_model import predict_player_targets, validate_share_simplex
from src.projection.shadow.target_pool import estimate_team_target_pool


def evaluate_shadow_0a_on_long_board(
    long_board: pd.DataFrame,
    *,
    top_adp: int = TOP_ADP,
) -> dict[str, Any]:
    """Score Shadow 0A target predictions against incumbent board targets."""
    frame = long_board.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    pools = estimate_team_target_pool(frame)
    merged = frame.merge(pools, on="team", how="left")
    shadow = predict_player_targets(merged)
    incumbent = frame[frame["stat"].eq("targets")].copy()
    incumbent["pred_season"] = (
        pd.to_numeric(incumbent["pred_pg"], errors="coerce").fillna(0.0)
        * pd.to_numeric(incumbent["projected_games"], errors="coerce").fillna(17.0)
    )

    if "adp" in frame.columns:
        top_ids = set(
            frame.loc[pd.to_numeric(frame["adp"], errors="coerce") <= top_adp, "player_id"]
            .astype(str)
        )
        shadow_top = shadow[shadow["player_id"].isin(top_ids)]
        incumbent_top = incumbent[incumbent["player_id"].isin(top_ids)]
    else:
        shadow_top = shadow
        incumbent_top = incumbent

    compared = shadow_top.merge(
        incumbent_top[["player_id", "pred_season"]].rename(
            columns={"pred_season": "pred_season_incumbent"}
        ),
        on="player_id",
        how="inner",
    )
    target_mae = float(
        (compared["pred_season_shadow"] - compared["pred_season_incumbent"]).abs().mean()
    ) if not compared.empty else float("nan")
    residuals = team_target_residual(shadow, incumbent)
    burden = reconciliation_burden_score(residuals)
    simplex = validate_share_simplex(shadow) if not shadow.empty else {}
    simplex_pass_rate = (
        float(sum(simplex.values()) / len(simplex)) if simplex else float("nan")
    )
    return {
        "target_mae_top_adp": target_mae,
        "reconciliation_burden": burden,
        "team_simplex_pass_rate": simplex_pass_rate,
        "n_shadow_players": int(len(shadow_top)),
        "n_compared_players": int(len(compared)),
    }
