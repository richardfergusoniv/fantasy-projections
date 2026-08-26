"""Out-of-fold v1/v2 ensemble: fit position blend weights on pre-holdout seasons.

Weights are nonnegative and sum to 1 per position. Fit on seasons strictly
before --holdout-season (default 2025), then score the holdout. Writes
ensemble_weights.json for draft-assistant post-process (does not touch
compose_board).
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
    accuracy_block,
    apply_blend,
    fit_nonnegative_blend_weights,
)

DEFAULT_V2_ROOT = REPO_ROOT.parent / "fantasy-projections-2"
OUT_DIR = REPO_ROOT / "output" / "test_before_rewrite"


def load_v1(season: int) -> pd.DataFrame | None:
    path = REPO_ROOT / "output" / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "player_id": df["player_id"].astype(str),
            "display_name": df["display_name"],
            "position": df["preseason_position"],
            "v1_pred": pd.to_numeric(df["model_points_end_to_end"], errors="coerce").fillna(0.0),
            "carry_pred": pd.to_numeric(df["carry_forward_points"], errors="coerce").fillna(0.0),
            "actual_points": pd.to_numeric(df["actual_points"], errors="coerce").fillna(0.0),
        }
    )


def load_v2(season: int, v2_root: Path) -> pd.DataFrame | None:
    path = v2_root / "outputs" / "preseason_oof.parquet"
    if not path.exists():
        return None
    oof = pd.read_parquet(path)
    oof = oof[oof["season"] == season].copy()
    if oof.empty:
        return None
    oof["gsis_id"] = oof["gsis_id"].astype(str)
    tot = (
        oof.groupby("gsis_id", as_index=False)
        .agg(
            v2_pred=("projected_fantasy_points", "sum"),
            actual_v2=("actual_fantasy_points", "sum"),
            position=("position", "first"),
        )
        .rename(columns={"gsis_id": "player_id"})
    )
    return tot


def merge_season(season: int, v2_root: Path) -> pd.DataFrame | None:
    v1 = load_v1(season)
    v2 = load_v2(season, v2_root)
    if v1 is None or v2 is None:
        return None
    merged = v1.merge(v2[["player_id", "v2_pred"]], on="player_id", how="inner")
    merged["season"] = season
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-seasons", default="2023,2024")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "ensemble_weights.json",
    )
    parser.add_argument(
        "--ship-out",
        type=Path,
        default=REPO_ROOT / "src" / "draft_assistant" / "ensemble_weights.json",
        help="Also write rounded production weights for draft prepare (default path)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=OUT_DIR / "ensemble_report.json",
    )
    args = parser.parse_args()
    fit_seasons = [int(s.strip()) for s in args.fit_seasons.split(",") if s.strip()]
    # Guard: never fit on holdout
    fit_seasons = [s for s in fit_seasons if s < args.holdout_season]
    if not fit_seasons:
        raise SystemExit("No fit seasons remain after excluding holdout")

    frames = []
    missing = []
    for season in fit_seasons:
        frame = merge_season(season, args.v2_root)
        if frame is None:
            missing.append(season)
        else:
            frames.append(frame)

    if not frames:
        raise SystemExit(
            f"No overlapping v1/v2 frames for fit seasons {fit_seasons}. "
            f"Missing or incomplete: {missing}. Run rolling_fantasy_eval first."
        )

    train = pd.concat(frames, ignore_index=True)
    weights = fit_nonnegative_blend_weights(train)

    holdout = merge_season(args.holdout_season, args.v2_root)
    report: dict = {
        "metadata": {
            "fit_seasons": fit_seasons,
            "fit_seasons_used": sorted(train["season"].unique().tolist()),
            "fit_seasons_missing_v1_or_v2": missing,
            "holdout_season": args.holdout_season,
            "rule": "Weights fit only on seasons < holdout; never tune on holdout",
        },
        "weights": weights,
        "train_accuracy": {
            "v1": accuracy_block(train, "v1_pred"),
            "v2": accuracy_block(train, "v2_pred"),
        },
    }
    train_blend = apply_blend(train, weights)
    report["train_accuracy"]["blend"] = accuracy_block(train_blend, "blend_pred")

    if holdout is not None:
        holdout_blend = apply_blend(holdout, weights)
        report["holdout_accuracy"] = {
            "v1": accuracy_block(holdout, "v1_pred"),
            "v2": accuracy_block(holdout, "v2_pred"),
            "carry_forward": accuracy_block(holdout, "carry_pred"),
            "blend": accuracy_block(holdout_blend, "blend_pred"),
        }
        # Position-level weight summary for draft UI
        report["holdout_n"] = int(len(holdout))
    else:
        report["holdout_accuracy"] = None
        report["holdout_error"] = f"Missing v1/v2 overlap for {args.holdout_season}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fit_seasons": report["metadata"]["fit_seasons_used"],
        "holdout_season": args.holdout_season,
        "weights": weights,
        "model_cols": ["v1_pred", "v2_pred"],
        "note": "Draft-assistant post-process only; does not alter compose_board",
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Rounded ship copy for prepare defaults
    ship = {
        "fit_seasons": payload["fit_seasons"],
        "holdout_season": args.holdout_season,
        "weights": {
            pos: {
                "v1_pred": round(float(w["v1_pred"]), 2),
                "v2_pred": round(float(w["v2_pred"]), 2),
            }
            for pos, w in weights.items()
        },
        "model_cols": ["v1_pred", "v2_pred"],
        "note": (
            "Draft-assistant post-process only; does not alter compose_board. "
            "See docs/decisions/TEST_BEFORE_REWRITE_2026-08-24.md."
        ),
    }
    args.ship_out.parent.mkdir(parents=True, exist_ok=True)
    args.ship_out.write_text(json.dumps(ship, indent=2), encoding="utf-8")
    args.report_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"weights": weights, "holdout": report.get("holdout_accuracy")}, indent=2))
    print(f"Wrote {args.out}, {args.ship_out}, and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
