"""Stage attribution for projected QB1s across compose + ensemble layers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.projection.composition import compose_board_stages, shipped_context
from src.projection.fantasy_points import SCORING

REPO_ROOT = Path(__file__).resolve().parents[3]


def _score_wide(pg: dict[str, float]) -> float:
    total = 0.0
    for stat, pts in SCORING.items():
        total += float(pg.get(stat, 0.0) or 0.0) * float(pts)
    return total


def _player_stat_map(long_df: pd.DataFrame, player_id: str) -> dict[str, float]:
    sub = long_df[long_df["player_id"].astype(str).eq(str(player_id))]
    out = {}
    for _, row in sub.iterrows():
        out[str(row["stat"])] = float(pd.to_numeric(row["pred_pg"], errors="coerce") or 0.0)
    return out


def build_stage_attribution(
    *,
    raw_path: Path | None = None,
    fantasy_path: Path | None = None,
    v1_path: Path | None = None,
    v2_path: Path | None = None,
    target_season: int = 2026,
) -> pd.DataFrame:
    """Produce a stage-attribution table for every projected QB1."""
    raw_path = raw_path or (REPO_ROOT / "output" / "projections_2026_raw.csv")
    fantasy_path = fantasy_path or (
        REPO_ROOT / "output" / "accuracy_first_2026" / "fantasy_points_2026.csv"
    )
    v1_path = v1_path or (REPO_ROOT / "output" / "model_v1" / "fantasy_points_2026_composed.csv")
    if not v1_path.exists():
        v1_path = REPO_ROOT / "output" / "fantasy_points_2026_composed.csv"
    v2_path = v2_path or (REPO_ROOT / "output" / "model_v2" / "fantasy_points_2026.csv")

    raw = pd.read_csv(raw_path)
    for col in (
        "pred_season",
        "pred_season_low",
        "pred_season_high",
        "team_volume_scale",
        "td_rate_clip_applied",
    ):
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    ctx = shipped_context(conn=None, target_season=target_season)
    stages = compose_board_stages(raw, ctx, return_boards=True)
    boards = stages.get("_boards") or {}

    fantasy = pd.read_csv(fantasy_path)
    qbs = fantasy[fantasy["position"].astype(str).eq("QB")].copy()
    # Prefer curated depth_tier/rank == 1 as QB1 definition.
    if "depth_tier" in qbs.columns:
        qb1 = qbs[pd.to_numeric(qbs["depth_tier"], errors="coerce").eq(1.0)].copy()
    elif "depth_rank" in qbs.columns:
        qb1 = qbs[pd.to_numeric(qbs["depth_rank"], errors="coerce").eq(1.0)].copy()
    else:
        qb1 = qbs.copy()
    qb1 = qb1.sort_values(
        "fantasy_pts_season" if "fantasy_pts_season" in qb1.columns else "fantasy_pts",
        ascending=False,
    ).reset_index(drop=True)
    qb1["final_rank"] = qb1.index + 1

    v1 = pd.read_csv(v1_path) if v1_path.exists() else None
    v2 = pd.read_csv(v2_path) if v2_path.exists() else None

    rows = []
    stage_names = [
        "raw_forecast",
        "exposure_status_baseline",
        "team_volume_reconcile",
        "td_constraints",
        "season_total_finalization",
    ]
    for _, meta in qb1.iterrows():
        pid = str(meta["player_id"])
        entry = {
            "player_id": pid,
            "display_name": meta.get("display_name"),
            "team": meta.get("team"),
            "projected_games": float(pd.to_numeric(meta.get("projected_games"), errors="coerce") or 17.0),
            "final_rank": int(meta["final_rank"]),
            "final_ensemble_ppg": float(pd.to_numeric(meta.get("fantasy_pts"), errors="coerce") or 0.0),
            "final_ensemble_season": float(
                pd.to_numeric(meta.get("fantasy_pts_season"), errors="coerce") or 0.0
            ),
        }
        for name in stage_names:
            scored = stages.get(name) or {}
            cell = scored.get(pid) or {}
            entry[f"{name}_ppg"] = cell.get("fantasy_ppg")
            if name in boards:
                stats = _player_stat_map(boards[name], pid)
                entry[f"{name}_carries"] = stats.get("carries")
                entry[f"{name}_rushing_yards"] = stats.get("rushing_yards")
                entry[f"{name}_attempts"] = stats.get("attempts")
                entry[f"{name}_passing_yards"] = stats.get("passing_yards")
                entry[f"{name}_scored_ppg"] = _score_wide(stats)

        # Depth/availability treatment = exposure_status_baseline vs raw.
        if entry.get("raw_forecast_ppg") is not None and entry.get("exposure_status_baseline_ppg") is not None:
            entry["depth_availability_delta_ppg"] = (
                entry["exposure_status_baseline_ppg"] - entry["raw_forecast_ppg"]
            )
        if entry.get("team_volume_reconcile_ppg") is not None and entry.get("exposure_status_baseline_ppg") is not None:
            entry["team_volume_delta_ppg"] = (
                entry["team_volume_reconcile_ppg"] - entry["exposure_status_baseline_ppg"]
            )
        if entry.get("td_constraints_ppg") is not None and entry.get("team_volume_reconcile_ppg") is not None:
            entry["td_rate_delta_ppg"] = (
                entry["td_constraints_ppg"] - entry["team_volume_reconcile_ppg"]
            )

        if v1 is not None:
            v1row = v1[v1["player_id"].astype(str).eq(pid)]
            if not v1row.empty:
                entry["v1_ppg"] = float(pd.to_numeric(v1row.iloc[0].get("fantasy_pts"), errors="coerce") or 0.0)
                if "fantasy_pts_season" in v1row.columns:
                    entry["v1_season"] = float(
                        pd.to_numeric(v1row.iloc[0].get("fantasy_pts_season"), errors="coerce") or 0.0
                    )
        if v2 is not None:
            v2row = v2[v2["player_id"].astype(str).eq(pid)]
            if not v2row.empty:
                # model_v2 fantasy board uses fantasy_pts_season or similar.
                cols = [c for c in ("fantasy_pts", "fantasy_pts_season", "points") if c in v2row.columns]
                if "fantasy_pts" in v2row.columns:
                    entry["v2_ppg"] = float(pd.to_numeric(v2row.iloc[0]["fantasy_pts"], errors="coerce") or 0.0)
                if "fantasy_pts_season" in v2row.columns:
                    entry["v2_season"] = float(
                        pd.to_numeric(v2row.iloc[0]["fantasy_pts_season"], errors="coerce") or 0.0
                    )
        if "v2_pred" in meta.index and pd.notna(meta.get("v2_pred")):
            entry["v2_season"] = float(meta["v2_pred"])
            entry["v2_ppg"] = float(meta["v2_pred"]) / max(entry["projected_games"], 1.0)
        rows.append(entry)

    return pd.DataFrame(rows)
