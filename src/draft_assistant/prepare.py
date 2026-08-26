"""Export projection CSV to draft-assistant JSON with tiers and metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

from src.draft_assistant.tiers import (
    DEFAULT_TIER_GAPS,
    FLEX_TIER_GAP,
    TierConfig,
    add_tier_columns,
)
from src.draft_assistant.vorp import (
    DEFAULT_TEAM_COUNT,
    load_position_curves,
    FLEX_SHARE,
    OVERALL_VORP_TIER_GAP,
    ROOKIE_RANK_SCALE,
    STARTERS,
    add_vorp_columns,
    replacement_ranks,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DRAFT_DATA_DIR = os.path.join(REPO_ROOT, "draft_assistant", "data")
DEFAULT_ENSEMBLE_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ensemble_weights.json"
)
MODEL_V3_DIR = os.path.join(OUTPUT_DIR, "model_v3")

EXPORT_COLS = [
    "player_id",
    "display_name",
    "position",
    "team",
    "fantasy_pts",
    "fantasy_pts_low",
    "fantasy_pts_high",
    "fantasy_pts_season",
    "projected_games",
    "source",
    "low_confidence",
    "role",
    "depth_chart_status",
    "vorp",
    "vorp_input_pts",
    "rookie_rank_scale",
    "replacement_pts",
    "vorp_curve_weight",
    "overall_rank",
    "overall_tier",
    "pos_rank",
    "pos_tier",
    "flex_rank",
    "flex_tier",
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_coverage",
    "sentiment_as_of",
    "sentiment_claim_count",
    "sentiment_source_count",
    "sentiment_model_active",
    "sentiment_version",
    "fantasy_pts_p10",
    "fantasy_pts_p25",
    "fantasy_pts_p50",
    "fantasy_pts_p75",
    "fantasy_pts_p90",
    "p_top12",
    "p_top24",
    "p_top36",
    "volatility_flag",
]


def load_projections(season: int, path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(OUTPUT_DIR, f"fantasy_points_{season}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing projection file: {path}")
    df = pd.read_csv(path)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df = df.sort_values("fantasy_pts_season", ascending=False).reset_index(drop=True)
    return df


def to_json_value(val, *, as_bool: bool = False):
    """Convert a pandas/scalar value to strict JSON-compatible Python types."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    if as_bool:
        return bool(val)
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 2)
    return str(val)


def build_player_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        rec = {}
        for col in EXPORT_COLS:
            raw = getattr(row, col, None)
            if col == "low_confidence":
                rec[col] = to_json_value(raw, as_bool=True) or False
            elif col in (
                "overall_rank",
                "overall_tier",
                "pos_rank",
                "pos_tier",
                "flex_rank",
                "flex_tier",
            ):
                rec[col] = int(raw) if pd.notna(raw) else None
            elif col in ("vorp", "replacement_pts"):
                rec[col] = to_json_value(raw)
            else:
                rec[col] = to_json_value(raw)
        records.append(rec)
    return records


def tier_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"overall": {}, "by_position": {}}
    overall_sorted = df.sort_values("vorp", ascending=False)
    for tier, group in overall_sorted.groupby("overall_tier", sort=False):
        summary["overall"][str(int(tier))] = {
            "count": int(len(group)),
            "top": group.sort_values("vorp", ascending=False).iloc[0]["display_name"],
            "vorp_range": [
                round(float(group["vorp"].max()), 2),
                round(float(group["vorp"].min()), 2),
            ],
        }
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df["position"] == pos]
        summary["by_position"][pos] = {}
        for tier, group in pos_df.groupby("pos_tier"):
            top = group.sort_values("vorp", ascending=False).iloc[0]
            summary["by_position"][pos][str(int(tier))] = {
                "count": int(len(group)),
                "top": top["display_name"],
                "vorp_range": [
                    round(float(group["vorp"].max()), 2),
                    round(float(group["vorp"].min()), 2),
                ],
            }
    flex_df = df[df["position"].isin(["RB", "WR", "TE"])]
    summary["flex"] = {}
    for tier, group in flex_df.groupby("flex_tier"):
        summary["flex"][str(int(tier))] = {
            "count": int(len(group)),
            "top": group.sort_values("vorp", ascending=False).iloc[0]["display_name"],
            "vorp_range": [
                round(float(group["vorp"].max()), 2),
                round(float(group["vorp"].min()), 2),
            ],
        }
    return summary


def load_ensemble_weights(path: str | None) -> dict | None:
    """Load position blend weights (v1/v2). Does not alter compose_board."""
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ensemble weights: {path}")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("weights") or payload


def default_v2_points_path(season: int) -> str:
    return os.path.join(OUTPUT_DIR, "model_v2", f"fantasy_points_{season}.csv")


def apply_ensemble_points(
    df: pd.DataFrame,
    weights: dict,
    *,
    v2_points_path: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Blend native board points with a v2 fantasy_points CSV.

    Returns ``(frame, applied)``. When v2 points are absent, returns the
    frame unchanged and ``applied=False``.
    """
    if not v2_points_path or not os.path.exists(v2_points_path):
        return df, False
    v2 = pd.read_csv(v2_points_path)
    v2 = v2[v2["position"].isin(["QB", "RB", "WR", "TE"])][
        ["player_id", "fantasy_pts_season"]
    ].rename(columns={"fantasy_pts_season": "v2_pts"})
    v2["player_id"] = v2["player_id"].astype(str)
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(v2, on="player_id", how="left")
    out["v2_pts"] = out["v2_pts"].fillna(out["fantasy_pts_season"])
    blended = []
    for _, row in out.iterrows():
        pos = str(row["position"])
        w = weights.get(pos) or {}
        # Weights keyed by v1_pred / v2_pred from ensemble_v1_v2.py
        w1 = float(w.get("v1_pred", 0.5))
        w2 = float(w.get("v2_pred", 0.5))
        blended.append(w1 * float(row["fantasy_pts_season"]) + w2 * float(row["v2_pts"]))
    out["fantasy_pts_season"] = blended
    # Keep per-game roughly consistent for display
    games = out["projected_games"].replace(0, pd.NA)
    out["fantasy_pts"] = out["fantasy_pts_season"] / games
    out["fantasy_pts"] = out["fantasy_pts"].fillna(out["fantasy_pts_season"] / 17.0)
    return out.drop(columns=["v2_pts"]), True


def resolve_ensemble_weights_path(
    *,
    season: int,
    ensemble_weights_path: str | None,
    use_ensemble: bool,
    ensemble_v2_points_path: str | None = None,
) -> str | None:
    """Default on when shipped weights + archived v2 points both exist."""
    if not use_ensemble:
        return None
    if ensemble_weights_path:
        return ensemble_weights_path
    v2_path = ensemble_v2_points_path or default_v2_points_path(season)
    if os.path.exists(DEFAULT_ENSEMBLE_WEIGHTS) and os.path.exists(v2_path):
        return DEFAULT_ENSEMBLE_WEIGHTS
    return None


def attach_v3_simulation_percentiles(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, bool]:
    """Merge v3 Monte Carlo percentiles when simulation summary exists."""
    path = os.path.join(MODEL_V3_DIR, f"simulation_summary_{season}.csv")
    if not os.path.exists(path):
        return df, False
    sim = pd.read_csv(path)
    rename = {
        "p10": "fantasy_pts_p10",
        "p25": "fantasy_pts_p25",
        "p50": "fantasy_pts_p50",
        "p75": "fantasy_pts_p75",
        "p90": "fantasy_pts_p90",
    }
    sim = sim.rename(columns=rename)
    sim["player_id"] = sim["player_id"].astype(str)
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    cols = ["player_id", *rename.values()]
    out = out.merge(sim[cols], on="player_id", how="left")
    out["volatility_flag"] = (
        pd.to_numeric(out["fantasy_pts_p90"], errors="coerce")
        - pd.to_numeric(out["fantasy_pts_p10"], errors="coerce")
    ) > 40.0
    tier_cutoffs = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
    draws_path = os.path.join(MODEL_V3_DIR, f"simulations_{season}.parquet")
    if os.path.exists(draws_path):
        try:
            draws = pd.read_parquet(draws_path)
        except Exception:
            draws = pd.DataFrame()
        if not draws.empty and "draw" in draws.columns:
            for pos, cutoff in tier_cutoffs.items():
                ranks = (
                    draws[draws["position"] == pos]
                    .groupby("draw")["fantasy_pts_season"]
                    .rank(ascending=False, method="first")
                )
                sub = draws[draws["position"] == pos].copy()
                sub["rank"] = ranks
                col = f"p_top{cutoff}"
                probs = (
                    sub.groupby("player_id")["rank"]
                    .apply(lambda s: float((s <= cutoff).mean()))
                    .rename(col)
                )
                out = out.merge(probs.reset_index(), on="player_id", how="left")
    return out, True


def apply_v3_means(
    df: pd.DataFrame,
    season: int,
    *,
    enabled: bool,
    require_gate: bool = True,
) -> tuple[pd.DataFrame, dict | None]:
    """Optionally replace draft mean points with v3 simulation p50.

    Default path keeps v1 / ensemble means. When ``enabled``, uses
    ``fantasy_pts_p50`` (or summary ``p50``) as ``fantasy_pts_season`` and
    recomputes per-game. Falls back to incumbent means if artifacts or the
    promotion gate are missing.
    """
    if not enabled:
        return df, None
    gate_path = os.path.join(MODEL_V3_DIR, "promotion_gate.json")
    gate = None
    if os.path.exists(gate_path):
        with open(gate_path, encoding="utf-8") as fh:
            gate = json.load(fh)
    if require_gate and (not gate or gate.get("verdict") != "promote_v3_means"):
        return df, {
            "applied": False,
            "reason": "gate_not_promote_v3_means",
            "gate_verdict": (gate or {}).get("verdict"),
            "fallback": "v1_or_ensemble_means",
        }
    summary_path = os.path.join(MODEL_V3_DIR, f"simulation_summary_{season}.csv")
    if not os.path.exists(summary_path):
        return df, {
            "applied": False,
            "reason": "missing_simulation_summary",
            "fallback": "v1_or_ensemble_means",
        }
    sim = pd.read_csv(summary_path)
    if "p50" not in sim.columns:
        return df, {
            "applied": False,
            "reason": "missing_p50",
            "fallback": "v1_or_ensemble_means",
        }
    sim = sim[["player_id", "p50"]].copy()
    sim["player_id"] = sim["player_id"].astype(str)
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(sim, on="player_id", how="left")
    missing = out["p50"].isna().mean()
    if missing > 0.5:
        return df, {
            "applied": False,
            "reason": "p50_coverage_too_low",
            "missing_frac": float(missing),
            "fallback": "v1_or_ensemble_means",
        }
    out["fantasy_pts_season"] = out["p50"].fillna(out["fantasy_pts_season"])
    games = out["projected_games"].replace(0, pd.NA)
    out["fantasy_pts"] = out["fantasy_pts_season"] / games
    out["fantasy_pts"] = out["fantasy_pts"].fillna(out["fantasy_pts_season"] / 17.0)
    out = out.drop(columns=["p50"])
    return out, {
        "applied": True,
        "source": summary_path.replace("\\", "/"),
        "gate_verdict": (gate or {}).get("verdict"),
        "note": "Draft mean points from v3 sim p50; VORP/tiers recomputed on this mean",
    }


def export_draft_data(
    season: int,
    *,
    tier_config: TierConfig | None = None,
    team_count: int = DEFAULT_TEAM_COUNT,
    rookie_rank_scale: float = ROOKIE_RANK_SCALE,
    ensemble_weights_path: str | None = None,
    ensemble_v2_points_path: str | None = None,
    use_ensemble: bool = True,
    use_v3_means: bool = False,
    require_v3_means_gate: bool = True,
    fantasy_path: str | None = None,
    out_path: str | None = None,
) -> str:
    df = load_projections(season, fantasy_path)
    ensemble_meta = None
    weights_path = resolve_ensemble_weights_path(
        season=season,
        ensemble_weights_path=ensemble_weights_path,
        use_ensemble=use_ensemble,
        ensemble_v2_points_path=ensemble_v2_points_path,
    )
    if weights_path:
        weights = load_ensemble_weights(weights_path)
        v2_path = ensemble_v2_points_path or default_v2_points_path(season)
        df, applied = apply_ensemble_points(df, weights, v2_points_path=v2_path)
        if applied:
            ensemble_meta = {
                "weights_path": weights_path.replace("\\", "/"),
                "v2_points_path": v2_path.replace("\\", "/"),
                "weights": weights,
                "note": "Draft post-process blend only; compose_board unchanged",
            }
    df, v3_applied = attach_v3_simulation_percentiles(df, season)
    df, v3_means_meta = apply_v3_means(
        df,
        season,
        enabled=use_v3_means,
        require_gate=require_v3_means_gate,
    )
    df = add_vorp_columns(df, team_count=team_count, rookie_rank_scale=rookie_rank_scale)
    df = add_tier_columns(
        df,
        points_col="vorp",
        config=tier_config,
        overall_points_col="vorp",
        overall_gap=OVERALL_VORP_TIER_GAP,
    )
    players = build_player_records(df)
    sources = (
        df["source"].dropna().astype(str).value_counts().to_dict()
        if "source" in df.columns
        else {}
    )
    if any(str(s).startswith("v2_") for s in sources):
        engine = "fantasy-projections-2 (team-first) — unexpected in native output/"
    else:
        engine = "fantasy-projections (rate-forecast / LightGBM)"

    payload = {
        "meta": {
            "season": season,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "player_count": len(players),
            "scoring": "half-PPR, 4pt passing TD",
            "source_file": f"output/fantasy_points_{season}.csv",
            "projection_engine": (
                "v3 simulation p50 means (flagged cutover); compose_board unchanged"
                if v3_means_meta and v3_means_meta.get("applied")
                else (
                    "v1/v2 draft ensemble (post-process); compose_board unchanged"
                    if ensemble_meta
                    else engine
                )
            ),
            "model_id": (
                "v3_means"
                if v3_means_meta and v3_means_meta.get("applied")
                else ("v1_v2_ensemble" if ensemble_meta else "v1_rate_forecast")
            ),
            "source_mix": sources,
            "roster": "1QB, 2RB, 3WR, 1TE, 1FLEX",
            "vorp_team_count": int(team_count),
            # The ranks actually used, which are deepened for availability --
            # not the nominal roster-math ranks.
            "vorp_replacement_ranks": df.attrs.get(
                "vorp_replacement_ranks", replacement_ranks(team_count)
            ),
            "vorp_replacement_ranks_nominal": replacement_ranks(team_count),
            "vorp_curve_weight": df.attrs.get("vorp_curve_weight", {}),
            # Published so the browser can reproduce the same blend when the
            # league size changes; a client-side recompute that skipped it would
            # silently re-inflate the position it corrects.
            "vorp_position_curves": load_position_curves(),
            "vorp_availability_factors": {
                k: round(v, 4)
                for k, v in (df.attrs.get("vorp_availability_factors") or {}).items()
            },
            "vorp_starters": STARTERS,
            "vorp_flex_share": FLEX_SHARE,
            "rookie_rank_scale": float(rookie_rank_scale),
            "ensemble": ensemble_meta,
            "v3_simulation": v3_applied,
            "v3_means": v3_means_meta,
        },
        "tier_gaps": {
            "overall_vorp": OVERALL_VORP_TIER_GAP,
            "flex": FLEX_TIER_GAP,
            **DEFAULT_TIER_GAPS,
        },
        "tier_summary": tier_summary(df),
        "players": players,
    }

    os.makedirs(DRAFT_DATA_DIR, exist_ok=True)
    out_path = out_path or os.path.join(DRAFT_DATA_DIR, f"players_{season}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draft assistant data")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--ensemble-weights",
        default=None,
        help=(
            "Position blend weights JSON (default: src/draft_assistant/"
            "ensemble_weights.json when output/model_v2 fantasy points exist)"
        ),
    )
    parser.add_argument(
        "--ensemble-v2-points",
        default=None,
        help="Optional v2 fantasy_points CSV (default output/model_v2/fantasy_points_<season>.csv)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Export the native v1 board only (skip v1/v2 post-process blend)",
    )
    parser.add_argument(
        "--v3-means",
        action="store_true",
        help=(
            "Replace draft mean points with v3 simulation p50 when "
            "promotion_gate.json verdict is promote_v3_means; otherwise keep "
            "v1/ensemble means"
        ),
    )
    parser.add_argument(
        "--force-v3-means",
        action="store_true",
        help="Use v3 p50 means even if promotion gate has not cleared promote_v3_means",
    )
    args = parser.parse_args()
    path = export_draft_data(
        args.season,
        ensemble_weights_path=args.ensemble_weights,
        ensemble_v2_points_path=args.ensemble_v2_points,
        use_ensemble=not args.no_ensemble,
        use_v3_means=bool(args.v3_means or args.force_v3_means),
        require_v3_means_gate=not args.force_v3_means,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
