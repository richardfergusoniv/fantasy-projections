"""Apply a selected arm to the 2026 board and compare vs sealed baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.composition import shipped_context
from src.projection.fantasy_points import SCORING
from src.projection.qb_repair.arms import ALL_ARMS, run_arm
from src.projection.qb_repair.rate_prior import build_qb_rate_priors

REPO_ROOT = Path(__file__).resolve().parents[3]


def score_long_to_fantasy(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a composed long board into half-PPR fantasy rows."""
    qb = long_df[long_df["position"].astype(str).eq("QB")].copy()
    if qb.empty:
        return pd.DataFrame()
    games = (
        qb.groupby("player_id")["projected_games"]
        .first()
        .rename("projected_games")
    )
    wide = qb.pivot_table(
        index=["player_id", "position"],
        columns="stat",
        values="pred_pg",
        aggfunc="first",
    ).reset_index()
    meta = (
        qb.sort_values("player_id")
        .drop_duplicates("player_id")[
            [
                c
                for c in (
                    "player_id",
                    "display_name",
                    "team",
                    "depth_tier",
                    "depth_rank",
                    "projected_games",
                )
                if c in qb.columns
            ]
        ]
    )
    out = wide.merge(meta, on="player_id", how="left")
    ppg = pd.Series(0.0, index=out.index)
    for stat, pts in SCORING.items():
        if stat in out.columns:
            ppg = ppg + pd.to_numeric(out[stat], errors="coerce").fillna(0.0) * float(pts)
    out["fantasy_pts"] = ppg
    out["fantasy_pts_season"] = ppg * pd.to_numeric(out["projected_games"], errors="coerce").fillna(17.0)
    out = out.sort_values("fantasy_pts_season", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def apply_arm_to_2026(arm: str, *, raw_path: Path | None = None) -> dict:
    raw_path = raw_path or (REPO_ROOT / "output" / "projections_2026_raw.csv")
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
    ctx = shipped_context(conn=None, target_season=2026)
    result = run_arm(raw, ctx, arm, target_season=2026)
    fantasy = score_long_to_fantasy(result.board)
    return {
        "arm": arm,
        "board": result.board,
        "fantasy": fantasy,
        "provenance": result.provenance,
    }


def compare_to_sealed(
    fantasy: pd.DataFrame,
    *,
    sealed_path: Path | None = None,
) -> dict:
    sealed_path = sealed_path or (
        REPO_ROOT / "output" / "accuracy_first_2026" / "fantasy_points_2026.csv"
    )
    sealed = pd.read_csv(sealed_path)
    sealed_qb = sealed[sealed["position"].astype(str).eq("QB")].copy()
    sealed_qb = sealed_qb.sort_values("fantasy_pts_season", ascending=False).reset_index(drop=True)
    sealed_qb["rank"] = sealed_qb.index + 1

    merged = fantasy.merge(
        sealed_qb[
            [
                c
                for c in (
                    "player_id",
                    "display_name",
                    "fantasy_pts",
                    "fantasy_pts_season",
                    "rank",
                    "pg_attempts",
                    "pg_passing_yards",
                    "pg_carries",
                    "pg_rushing_yards",
                    "pg_rushing_tds",
                )
                if c in sealed_qb.columns
            ]
        ],
        on="player_id",
        how="outer",
        suffixes=("_new", "_sealed"),
    )
    merged["season_delta"] = (
        pd.to_numeric(merged.get("fantasy_pts_season_new"), errors="coerce")
        - pd.to_numeric(merged.get("fantasy_pts_season_sealed"), errors="coerce")
    )
    merged["rank_delta"] = (
        pd.to_numeric(merged.get("rank_sealed"), errors="coerce")
        - pd.to_numeric(merged.get("rank_new"), errors="coerce")
    )
    focus_names = [
        "Lamar Jackson",
        "Joe Burrow",
        "Josh Allen",
        "Drake Maye",
        "Jalen Hurts",
        "Patrick Mahomes",
        "Bo Nix",
        "Jayden Daniels",
    ]
    name_col = "display_name_new" if "display_name_new" in merged.columns else "display_name"
    if name_col not in merged.columns and "display_name_sealed" in merged.columns:
        name_col = "display_name_sealed"
    focus = merged[merged[name_col].isin(focus_names)].copy() if name_col in merged.columns else merged.head(0)

    ppg = pd.to_numeric(fantasy["fantasy_pts"], errors="coerce")
    return {
        "n_qb": int(len(fantasy)),
        "ppg_thresholds": {
            "ge_18": int((ppg >= 18).sum()),
            "ge_20": int((ppg >= 20).sum()),
            "ge_22": int((ppg >= 22).sum()),
            "ge_24": int((ppg >= 24).sum()),
        },
        "largest_positive": merged.nlargest(10, "season_delta")[
            [c for c in (name_col, "player_id", "season_delta", "rank_new", "rank_sealed") if c in merged.columns]
        ].to_dict("records"),
        "largest_negative": merged.nsmallest(10, "season_delta")[
            [c for c in (name_col, "player_id", "season_delta", "rank_new", "rank_sealed") if c in merged.columns]
        ].to_dict("records"),
        "rank_moves_gt_5": merged[merged["rank_delta"].abs() > 5][
            [c for c in (name_col, "player_id", "rank_new", "rank_sealed", "rank_delta", "season_delta") if c in merged.columns]
        ].to_dict("records"),
        "focus_players": focus.to_dict("records"),
        "top15_new": fantasy.head(15)[
            [c for c in ("rank", "display_name", "team", "fantasy_pts", "fantasy_pts_season", "attempts", "passing_yards", "carries", "rushing_yards", "rushing_tds") if c in fantasy.columns]
        ].to_dict("records"),
    }


def non_qb_invariance_check(
    *,
    baseline_long: pd.DataFrame,
    candidate_long: pd.DataFrame,
    atol: float = 1e-9,
) -> dict:
    """Every non-QB player/component must be exactly unchanged."""
    def _key(df: pd.DataFrame) -> pd.DataFrame:
        out = df[~df["position"].astype(str).eq("QB")][
            ["player_id", "position", "stat", "pred_pg"]
        ].copy()
        out["player_id"] = out["player_id"].astype(str)
        out["stat"] = out["stat"].astype(str)
        return out.sort_values(["player_id", "position", "stat"]).reset_index(drop=True)

    a = _key(baseline_long)
    b = _key(candidate_long)
    if len(a) != len(b):
        return {"pass": False, "reason": "row_count_mismatch", "n_base": len(a), "n_cand": len(b)}
    merged = a.merge(b, on=["player_id", "position", "stat"], suffixes=("_base", "_cand"))
    if len(merged) != len(a):
        return {"pass": False, "reason": "key_mismatch", "n_merged": len(merged), "n_base": len(a)}
    delta = (
        pd.to_numeric(merged["pred_pg_cand"], errors="coerce")
        - pd.to_numeric(merged["pred_pg_base"], errors="coerce")
    ).abs()
    n_bad = int((delta > atol).sum())
    return {
        "pass": n_bad == 0,
        "n_compared": int(len(merged)),
        "n_changed": n_bad,
        "max_abs_delta": float(delta.max() if len(delta) else 0.0),
    }


def team_passing_conservation(
    long_df: pd.DataFrame,
    *,
    room_share: float = 0.941,
) -> dict:
    """Report QB-room season attempts vs team anchor claim."""
    qb = long_df[long_df["position"].eq("QB") & long_df["stat"].eq("attempts")].copy()
    if qb.empty or "team_pass_attempts_pg_pred" not in qb.columns:
        return {"n_teams": 0, "violations": []}
    exposure = pd.to_numeric(qb.get("projected_volume_games"), errors="coerce")
    exposure = exposure.fillna(pd.to_numeric(qb.get("projected_games"), errors="coerce")).fillna(17.0)
    qb = qb.assign(
        season_attempts=pd.to_numeric(qb["pred_pg"], errors="coerce") * exposure,
        team_target=pd.to_numeric(qb["team_pass_attempts_pg_pred"], errors="coerce") * 17.0 * room_share,
    )
    grouped = qb.groupby("team", as_index=False).agg(
        realized=("season_attempts", "sum"),
        target=("team_target", "first"),
    )
    grouped["rel_error"] = (grouped["realized"] - grouped["target"]) / grouped["target"].replace(0, np.nan)
    violations = grouped[grouped["rel_error"].abs() > 0.05]
    return {
        "n_teams": int(len(grouped)),
        "mean_abs_rel_error": float(grouped["rel_error"].abs().mean()),
        "violations": violations.to_dict("records"),
    }
