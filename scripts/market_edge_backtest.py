"""Market-edge backtest: does model−ADP residual predict actual outcomes?

Loads frozen consensus snapshots and model/actual points, joins with the same
matched-set rules as consensus_spread, and reports draft-edge proxies.

Model sources per season (no retraining):
  - v1: output/fantasy_evaluation_{season}.csv when present
  - v2: fantasy-projections-2 outputs/preseason_oof.parquet season totals
  - blend: optional weights JSON from ensemble_v1_v2.py

Usage:
  python scripts/market_edge_backtest.py
  python scripts/market_edge_backtest.py --seasons 2023,2024,2025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.projection.market_metrics import (  # noqa: E402
    DEFAULT_MAX_MARKET_RANK,
    accuracy_block,
    apply_blend,
    draft_edge_proxy,
    market_agreement,
    matched_market_frame,
)

DEFAULT_V2_ROOT = REPO_ROOT.parent / "fantasy-projections-2"
OUT_DIR = REPO_ROOT / "output" / "test_before_rewrite"
CONSENSUS_DIR = REPO_ROOT / "data" / "consensus"


def load_consensus(season: int) -> pd.DataFrame:
    path = CONSENSUS_DIR / f"consensus_{season}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/fetch_consensus_snapshots.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["rows"]), payload.get("meta") or {}


def load_v1_eval(season: int) -> pd.DataFrame | None:
    path = REPO_ROOT / "output" / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "player_id": df["player_id"].astype(str),
            "display_name": df["display_name"],
            "position": df["preseason_position"],
            "v1_pred": pd.to_numeric(df["model_points_end_to_end"], errors="coerce").fillna(0.0),
            "carry_pred": pd.to_numeric(df["carry_forward_points"], errors="coerce").fillna(0.0),
            "actual_points": pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0),
        }
    )
    return out


def load_v2_oof(season: int, v2_root: Path) -> pd.DataFrame | None:
    path = v2_root / "outputs" / "preseason_oof.parquet"
    if not path.exists():
        return None
    oof = pd.read_parquet(path)
    oof = oof[oof["season"] == season].copy()
    if oof.empty:
        return None
    oof["gsis_id"] = oof["gsis_id"].astype(str)
    season_tot = (
        oof.groupby("gsis_id", as_index=False)
        .agg(
            v2_pred=("projected_fantasy_points", "sum"),
            actual_points=("actual_fantasy_points", "sum"),
            position=("position", "first"),
        )
    )
    season_tot = season_tot.rename(columns={"gsis_id": "player_id"})
    season_tot["display_name"] = season_tot["player_id"]
    return season_tot


def load_blend_weights(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_board(season: int, v2_root: Path, blend_weights: dict | None) -> pd.DataFrame:
    v1 = load_v1_eval(season)
    v2 = load_v2_oof(season, v2_root)
    if v1 is None and v2 is None:
        raise FileNotFoundError(f"No v1 eval or v2 OOF for season {season}")

    if v1 is not None and v2 is not None:
        board = v1.merge(
            v2[["player_id", "v2_pred", "position", "actual_points"]].rename(
                columns={
                    "position": "v2_position",
                    "actual_points": "v2_actual",
                }
            ),
            on="player_id",
            how="outer",
        )
        board["v1_pred"] = board["v1_pred"].fillna(0.0)
        board["v2_pred"] = board["v2_pred"].fillna(0.0)
        board["carry_pred"] = board["carry_pred"].fillna(0.0)
        board["position"] = board["position"].fillna(board["v2_position"])
        board["actual_points"] = board["actual_points"].fillna(board["v2_actual"]).fillna(0.0)
        board["display_name"] = board["display_name"].fillna(board["player_id"])
        board = board.drop(columns=["v2_position", "v2_actual"], errors="ignore")
    elif v1 is not None:
        board = v1.copy()
        board["v2_pred"] = 0.0
    else:
        assert v2 is not None
        board = v2.copy()
        board["v1_pred"] = 0.0
        board["carry_pred"] = 0.0
        if "display_name" not in board.columns:
            board["display_name"] = board["player_id"]

    if blend_weights:
        w = blend_weights.get("weights") or blend_weights
        board = apply_blend(board, w, model_cols=("v1_pred", "v2_pred"))
    else:
        board["blend_pred"] = 0.5 * board["v1_pred"].fillna(0.0) + 0.5 * board[
            "v2_pred"
        ].fillna(0.0)

    return board


def score_model(
    board: pd.DataFrame,
    consensus: pd.DataFrame,
    pred_col: str,
    *,
    max_market_rank: int,
    market_col: str = "adp",
) -> dict:
    frame = board.rename(columns={pred_col: "model_points"})
    # Keep actual_points
    matched = matched_market_frame(
        frame,
        consensus,
        market_col=market_col,
        model_points_col="model_points",
        actual_points_col="actual_points",
        max_market_rank=max_market_rank,
    )
    return {
        "agreement": market_agreement(matched),
        "edge": draft_edge_proxy(matched),
        "accuracy_on_matched": accuracy_block(
            matched.rename(columns={"model_points": pred_col}),
            pred_col,
            actual_col="actual_points",
        )
        if "actual_points" in matched.columns
        else {},
    }


def run_season(
    season: int,
    *,
    v2_root: Path,
    blend_weights: dict | None,
    max_market_rank: int,
) -> dict:
    consensus, meta = load_consensus(season)
    board = build_board(season, v2_root, blend_weights)
    # Enrich display names from consensus for v2-only rows
    name_map = (
        consensus.dropna(subset=["player_id"])
        .drop_duplicates("player_id")
        .set_index("player_id")["display_name"]
        .to_dict()
    )
    need = board["display_name"].isna() | (board["display_name"] == board["player_id"])
    board.loc[need, "display_name"] = board.loc[need, "player_id"].map(name_map).fillna(
        board.loc[need, "display_name"]
    )

    models = {}
    for label, col in (
        ("v1", "v1_pred"),
        ("v2", "v2_pred"),
        ("carry_forward", "carry_pred"),
        ("blend", "blend_pred"),
    ):
        if col not in board.columns:
            continue
        if board[col].abs().sum() == 0 and label != "blend":
            models[label] = {"skipped": True, "reason": "all-zero or missing"}
            continue
        models[label] = score_model(
            board, consensus, col, max_market_rank=max_market_rank
        )

    return {
        "season": season,
        "consensus_as_of": meta.get("as_of"),
        "consensus_meta": {
            "matched_adp": meta.get("matched_adp"),
            "matched_ecr": meta.get("matched_ecr"),
            "adp_drafts": (meta.get("adp") or {}).get("total_drafts"),
        },
        "board_n": int(len(board)),
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument(
        "--blend-weights",
        type=Path,
        default=OUT_DIR / "ensemble_weights.json",
    )
    parser.add_argument("--max-market-rank", type=int, default=DEFAULT_MAX_MARKET_RANK)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "market_edge_backtest.json",
    )
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    weights = load_blend_weights(args.blend_weights)

    report = {
        "metadata": {
            "max_market_rank": args.max_market_rank,
            "market": "adp",
            "blend_weights_path": str(args.blend_weights) if weights else None,
            "pass_fail_rule": (
                "Actionable edge requires edge_corr_residual_vs_actual_points < -0.05 "
                "in at least 2 seasons for the same model label"
            ),
        },
        "seasons": {},
    }
    for season in seasons:
        try:
            report["seasons"][str(season)] = run_season(
                season,
                v2_root=args.v2_root,
                blend_weights=weights,
                max_market_rank=args.max_market_rank,
            )
        except FileNotFoundError as exc:
            report["seasons"][str(season)] = {"error": str(exc)}

    # Aggregate pass/fail
    actionable = {}
    for label in ("v1", "v2", "blend", "carry_forward"):
        hits = []
        for season, block in report["seasons"].items():
            edge = ((block.get("models") or {}).get(label) or {}).get("edge") or {}
            if edge.get("actionable_points_edge"):
                hits.append(int(season))
        actionable[label] = {
            "seasons_with_points_edge": hits,
            "multi_season_edge": len(hits) >= 2,
        }
    report["actionable_summary"] = actionable

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"actionable_summary": actionable}, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
