"""Head-to-head accuracy: v1 rate-forecast vs v2 team-first on 2025 holdout.

Uses the leakage-safe Week-1 roster population and half-PPR season points from
``output/fantasy_evaluation_2025.csv`` as the shared scoreboard. v2 season
totals come from aggregating ``fantasy-projections-2/outputs/preseason_oof.parquet``
(strict preseason OOF, half-PPR by default).

Writes ``output/model_accuracy_compare_2025.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = REPO_ROOT.parent / "fantasy-projections-2"

TIER_RANKS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def _mae(a: pd.Series, b: pd.Series) -> float:
    return float(np.mean(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))))


def _tier_hits(actual: pd.Series, pred: pd.Series, n: int) -> dict:
    if len(actual) < n:
        return {"n": n, "hits": 0, "hit_rate": float("nan")}
    true_top = set(actual.nlargest(n).index)
    pred_top = set(pred.nlargest(n).index)
    hits = len(true_top & pred_top)
    return {"n": n, "hits": hits, "hit_rate": hits / float(n)}


def _score_block(frame: pd.DataFrame, pred_col: str) -> dict:
    out: dict = {"overall": {}, "by_position": {}}
    valid = frame.dropna(subset=[pred_col, "actual_points"]).copy()
    out["overall"] = {
        "n": int(len(valid)),
        "spearman": _spearman(valid["actual_points"], valid[pred_col]),
        "points_mae": _mae(valid["actual_points"], valid[pred_col]),
    }
    for pos, sub in valid.groupby("preseason_position"):
        sub = sub.copy()
        tier_n = TIER_RANKS.get(str(pos), 12)
        # Rank within position for tier hits
        sub_idx = sub.reset_index(drop=True)
        hits = _tier_hits(
            sub_idx["actual_points"],
            sub_idx[pred_col],
            min(tier_n, len(sub_idx)),
        )
        out["by_position"][str(pos)] = {
            "n": int(len(sub)),
            "spearman": _spearman(sub["actual_points"], sub[pred_col]),
            "points_mae": _mae(sub["actual_points"], sub[pred_col]),
            "tier": hits,
        }
    return out


def load_v1_population(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = [
        "player_id",
        "display_name",
        "preseason_position",
        "actual_points",
        "model_points_end_to_end",
        "carry_forward_points",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"v1 eval missing columns: {missing}")
    out = df[need].copy()
    out["player_id"] = out["player_id"].astype(str)
    out["v1_pred"] = pd.to_numeric(out["model_points_end_to_end"], errors="coerce").fillna(0.0)
    out["baseline_pred"] = pd.to_numeric(out["carry_forward_points"], errors="coerce").fillna(0.0)
    out["actual_points"] = pd.to_numeric(out["actual_points"], errors="coerce").fillna(0.0)
    return out


def load_v2_season_totals(oof_path: Path, season: int) -> pd.DataFrame:
    oof = pd.read_parquet(oof_path)
    oof = oof[oof["season"] == season].copy()
    if oof.empty:
        raise ValueError(f"No v2 OOF rows for season {season} in {oof_path}")
    oof["gsis_id"] = oof["gsis_id"].astype(str)
    oof["projected_fantasy_points"] = pd.to_numeric(
        oof["projected_fantasy_points"], errors="coerce"
    ).fillna(0.0)
    season_tot = (
        oof.groupby("gsis_id", as_index=False)
        .agg(
            v2_pred=("projected_fantasy_points", "sum"),
            v2_weeks=("projected_fantasy_points", "size"),
        )
    )
    return season_tot


def compare(
    *,
    v1_eval: Path,
    v2_oof: Path,
    season: int = 2025,
) -> dict:
    pop = load_v1_population(v1_eval)
    v2 = load_v2_season_totals(v2_oof, season)
    merged = pop.merge(v2, left_on="player_id", right_on="gsis_id", how="left")
    merged["v2_pred"] = merged["v2_pred"].fillna(0.0)
    merged["v2_covered"] = merged["gsis_id"].notna()

    common = merged[merged["v2_covered"]].copy()
    report = {
        "metadata": {
            "target_season": season,
            "scoring": "half-PPR season points",
            "population": (
                "v1 leakage-safe Week-1 roster (ACT/DEV/RES/INA/EXE); "
                "position/team frozen preseason"
            ),
            "v1_model": "fantasy-projections rate-forecast (LightGBM + composition gates)",
            "v2_model": "fantasy-projections-2 team-first (strict preseason OOF season totals)",
            "v1_source": str(v1_eval),
            "v2_source": str(v2_oof),
            "population_n": int(len(merged)),
            "v2_covered_n": int(merged["v2_covered"].sum()),
            "caveat": (
                "Same population and actuals. v1 points are the harness "
                "end-to-end season forecast; v2 points are summed week-level "
                "strict-preseason OOF projections. Populations/joins differ "
                "from each repo's native published summary tables."
            ),
        },
        "on_v1_population_missing_v2_as_zero": {
            "v1_rate_forecast": _score_block(merged, "v1_pred"),
            "v2_team_first": _score_block(merged, "v2_pred"),
            "carry_forward_baseline": _score_block(merged, "baseline_pred"),
        },
        "on_intersection_only": {
            "v1_rate_forecast": _score_block(common, "v1_pred"),
            "v2_team_first": _score_block(common, "v2_pred"),
            "carry_forward_baseline": _score_block(common, "baseline_pred"),
        },
    }

    # Headline winner per position (intersection, spearman then MAE)
    winners = {}
    for pos in ("QB", "RB", "WR", "TE"):
        a = report["on_intersection_only"]["v1_rate_forecast"]["by_position"].get(pos, {})
        b = report["on_intersection_only"]["v2_team_first"]["by_position"].get(pos, {})
        if not a or not b:
            continue
        if a["spearman"] > b["spearman"] + 1e-9:
            spearman_winner = "v1_rate_forecast"
        elif b["spearman"] > a["spearman"] + 1e-9:
            spearman_winner = "v2_team_first"
        else:
            spearman_winner = "tie"
        if a["points_mae"] < b["points_mae"] - 1e-9:
            mae_winner = "v1_rate_forecast"
        elif b["points_mae"] < a["points_mae"] - 1e-9:
            mae_winner = "v2_team_first"
        else:
            mae_winner = "tie"
        winners[pos] = {
            "spearman_winner": spearman_winner,
            "mae_winner": mae_winner,
            "v1_spearman": a["spearman"],
            "v2_spearman": b["spearman"],
            "v1_mae": a["points_mae"],
            "v2_mae": b["points_mae"],
            "v1_tier_hit_rate": a.get("tier", {}).get("hit_rate"),
            "v2_tier_hit_rate": b.get("tier", {}).get("hit_rate"),
        }
    report["winners_intersection"] = winners

    ov1 = report["on_intersection_only"]["v1_rate_forecast"]["overall"]
    ov2 = report["on_intersection_only"]["v2_team_first"]["overall"]
    report["headline"] = {
        "preferred_board": (
            "v1_rate_forecast"
            if ov1["spearman"] > ov2["spearman"]
            else "v2_team_first"
            if ov2["spearman"] > ov1["spearman"]
            else "tie"
        ),
        "metric": "season-points Spearman on intersection",
        "v1_spearman": ov1["spearman"],
        "v2_spearman": ov2["spearman"],
        "v1_mae": ov1["points_mae"],
        "v2_mae": ov2["points_mae"],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--v1-eval",
        type=Path,
        default=REPO_ROOT / "output" / "fantasy_evaluation_2025.csv",
    )
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=DEFAULT_V2_ROOT,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "model_accuracy_compare_2025.json",
    )
    args = parser.parse_args(argv)

    oof = args.v2_root / "outputs" / "preseason_oof.parquet"
    report = compare(v1_eval=args.v1_eval, v2_oof=oof, season=args.season)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["headline"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
