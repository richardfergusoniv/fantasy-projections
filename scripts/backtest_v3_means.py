"""Rolling fantasy holdout for v3 interim and generative means.

Fits generative artifacts on history through source_season, builds the
leakage-safe long board, then scores:

- v1: fantasy_evaluation model_points_end_to_end
- v3_interim: simulation p50 from interim residual bootstrap on the long board
- v3_generative: mean fantasy points from generative opportunity→conversion draws

Writes ``output/model_v3/means_backtest.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR, V3_MODELS_DIR
from src.projection.data_prep import get_conn
from src.projection.evaluation.v3_means_score import beats_incumbent, score_predictions
from src.projection.fantasy_evaluation import build_leakage_safe_long_board
from src.projection.fantasy_points import SCORING
from src.projection.features import build_player_season_features
from src.projection.inference.fit import fit_v3_models
from src.projection.inference.reconcile import reconcile_v3_generative
from src.projection.inference.simulate import simulate_season_distributions, summarize_simulations
from src.projection.transitions import SEASON_GAMES

FOLDS = [(2022, 2023), (2023, 2024), (2024, 2025)]
DEFAULT_DRAWS = 200


def _score_wide(wide: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=wide.index)
    for stat, weight in SCORING.items():
        if stat in wide.columns:
            total = total + pd.to_numeric(wide[stat], errors="coerce").fillna(0.0) * weight
    return total


def _load_eval(season: int) -> pd.DataFrame:
    path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = ["player_id", "actual_points", "model_points_end_to_end"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    out["actual_points"] = pd.to_numeric(out["actual_points"], errors="coerce").fillna(0.0)
    out["v1_pred"] = pd.to_numeric(out["model_points_end_to_end"], errors="coerce").fillna(0.0)
    if "preseason_position" not in out.columns and "position" in out.columns:
        out["preseason_position"] = out["position"]
    return out


def _blend_from_compare(season: int, pop: pd.DataFrame) -> pd.Series | None:
    """Use archived accuracy-compare v2 preds when available (2025)."""
    path = Path(OUTPUT_DIR) / f"model_accuracy_compare_{season}.json"
    if not path.exists():
        return None
    # Prefer model_v2 fantasy points if present for any season.
    v2_path = Path(OUTPUT_DIR) / "model_v2" / f"fantasy_points_{season}.csv"
    weights_path = Path(__file__).resolve().parents[1] / "src" / "draft_assistant" / "ensemble_weights.json"
    if not v2_path.exists() or not weights_path.exists():
        return None
    weights = json.loads(weights_path.read_text(encoding="utf-8")).get("weights", {})
    v2 = pd.read_csv(v2_path)
    v2 = v2[["player_id", "fantasy_pts_season", "position"]].copy()
    v2["player_id"] = v2["player_id"].astype(str)
    merged = pop.merge(v2, on="player_id", how="left")
    blended = []
    for _, row in merged.iterrows():
        pos = str(row.get("preseason_position") or row.get("position") or "")
        w = weights.get(pos) or {"v1_pred": 0.5, "v2_pred": 0.5}
        v1 = float(row["v1_pred"])
        v2p = float(row["fantasy_pts_season"]) if pd.notna(row.get("fantasy_pts_season")) else v1
        blended.append(float(w.get("v1_pred", 0.5)) * v1 + float(w.get("v2_pred", 0.5)) * v2p)
    return pd.Series(blended, index=pop.index)


def _interim_means(long_board: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    draws = simulate_season_distributions(long_board, n_draws=n_draws, mode="interim", seed=7)
    if draws.empty:
        return pd.DataFrame(columns=["player_id", "v3_interim_pred"])
    summary = summarize_simulations(draws)
    return summary.rename(columns={"p50": "v3_interim_pred"})[
        ["player_id", "v3_interim_pred"]
    ]


def _generative_means(long_board: pd.DataFrame, share_manifest: dict, n_draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    team_env = (
        long_board[["team"]]
        .drop_duplicates()
        .assign(team_pass_attempts_mean=600.0, team_carries_mean=400.0)
    )
    # Prefer fitted team environment means when present on disk.
    env_manifest = Path(V3_MODELS_DIR) / "team_environment" / "manifest.json"
    if env_manifest.exists():
        pass  # means drawn below still use defaults; fit artifacts used for shares
    scores = []
    for draw_idx in range(n_draws):
        gen = reconcile_v3_generative(
            long_board, team_env, rng=rng, share_manifest=share_manifest
        )
        if gen.empty:
            continue
        wide = gen.groupby(["player_id", "position", "team"], observed=True).sum(
            numeric_only=True
        ).reset_index()
        wide["fantasy_pts_season"] = _score_wide(wide)
        wide["draw"] = draw_idx
        scores.append(wide[["player_id", "fantasy_pts_season", "draw"]])
    if not scores:
        return pd.DataFrame(columns=["player_id", "v3_generative_pred"])
    all_draws = pd.concat(scores, ignore_index=True)
    means = all_draws.groupby("player_id", observed=True)["fantasy_pts_season"].mean().reset_index()
    return means.rename(columns={"fantasy_pts_season": "v3_generative_pred"})


def run_fold(conn, feat, source_season: int, target_season: int, n_draws: int) -> dict:
    eval_pop = _load_eval(target_season)
    if eval_pop.empty:
        return {"target_season": target_season, "skipped": True, "reason": "missing fantasy_evaluation"}

    print(f"  long board {source_season}->{target_season}...", flush=True)
    long_board = build_leakage_safe_long_board(conn, feat, source_season, target_season)
    if long_board.empty:
        return {"target_season": target_season, "skipped": True, "reason": "empty long board"}

    train_pairs = [(y, y + 1) for y in range(2021, source_season) if y + 1 <= source_season]
    history_board = long_board.copy()
    print(f"  fit generative through {source_season}...", flush=True)
    fitted = fit_v3_models(feat, train_pairs, long_board=history_board)
    share_manifest = fitted.get("opportunity_shares") or {}

    print(f"  interim sim ({n_draws} draws)...", flush=True)
    interim = _interim_means(long_board, n_draws)
    print(f"  generative sim ({n_draws} draws)...", flush=True)
    generative = _generative_means(long_board, share_manifest, n_draws)

    scored = eval_pop.merge(interim, on="player_id", how="left")
    scored = scored.merge(generative, on="player_id", how="left")
    scored["v3_interim_pred"] = scored["v3_interim_pred"].fillna(scored["v1_pred"])
    scored["v3_generative_pred"] = scored["v3_generative_pred"].fillna(scored["v1_pred"])

    # The blend arm exists to be a SECOND, independent incumbent. Copying
    # v1 into it when the v2 points are missing does not make the gate
    # conservative, it makes "beats blend" a silent restatement of "beats
    # v1" -- two checks reported, one check performed. Leave it absent
    # instead and fail its gates closed.
    blend = _blend_from_compare(target_season, scored)
    blend_available = blend is not None
    if blend_available:
        scored["blend_pred"] = blend

    metrics = {
        "v1": score_predictions(scored, "v1_pred"),
        "v3_interim": score_predictions(scored, "v3_interim_pred"),
        "v3_generative": score_predictions(scored, "v3_generative_pred"),
    }
    if blend_available:
        metrics["blend"] = score_predictions(scored, "blend_pred")

    # Belt and braces: a blend that reproduces v1's MAE to the last bit is
    # v1 under another name, however it got there.
    blend_degenerate = bool(
        blend_available
        and np.isclose(
            metrics["blend"]["points_mae"], metrics["v1"]["points_mae"], rtol=0.0, atol=1e-12
        )
    )
    blend_usable = blend_available and not blend_degenerate
    blend_reason = (
        None if blend_usable
        else ("blend_arm_unavailable" if not blend_available else "blend_arm_degenerate")
    )

    def _vs_blend(candidate: dict) -> dict:
        if not blend_usable:
            return {
                "mae_ok": False,
                "spearman_ok": False,
                "pass": False,
                "reason": blend_reason,
            }
        return beats_incumbent(candidate, metrics["blend"])

    return {
        "target_season": target_season,
        "source_season": source_season,
        "n_draws": n_draws,
        "blend_available": blend_available,
        "blend_degenerate": blend_degenerate,
        "blend_usable": blend_usable,
        "metrics": metrics,
        "gates": {
            "interim_beats_v1": beats_incumbent(metrics["v3_interim"], metrics["v1"]),
            "interim_beats_blend": _vs_blend(metrics["v3_interim"]),
            "generative_beats_v1": beats_incumbent(metrics["v3_generative"], metrics["v1"]),
            "generative_beats_blend": _vs_blend(metrics["v3_generative"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument(
        "--folds",
        default="2023,2024,2025",
        help="Comma-separated target seasons to score (default 2023,2024,2025)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(MODEL_V3_DIR) / "means_backtest.json",
    )
    args = parser.parse_args()
    wanted = {int(x.strip()) for x in args.folds.split(",") if x.strip()}
    conn = get_conn()
    feat = build_player_season_features(conn)
    folds = []
    for source, target in FOLDS:
        if target not in wanted:
            continue
        print(f"Fold {source}->{target}", flush=True)
        folds.append(run_fold(conn, feat, source, target, args.draws))
    conn.close()

    usable = [f for f in folds if not f.get("skipped")]
    blend_usable_all_folds = bool(usable) and all(f.get("blend_usable") for f in usable)
    summary = {
        "blend_usable_all_folds": blend_usable_all_folds,
        "blend_unusable_folds": [
            {"target_season": f["target_season"], "blend_available": f.get("blend_available")}
            for f in usable
            if not f.get("blend_usable")
        ],
        "interim_beats_v1_all_folds": all(
            f["gates"]["interim_beats_v1"]["pass"] for f in usable
        ) if usable else False,
        "interim_beats_blend_all_folds": all(
            f["gates"]["interim_beats_blend"]["pass"] for f in usable
        ) if usable else False,
        "generative_beats_v1_all_folds": all(
            f["gates"]["generative_beats_v1"]["pass"] for f in usable
        ) if usable else False,
        "generative_beats_blend_all_folds": all(
            f["gates"]["generative_beats_blend"]["pass"] for f in usable
        ) if usable else False,
        "n_folds": len(usable),
    }
    # Prefer generative as the candidate mean engine when promoting means.
    # blend_usable_all_folds is redundant with generative_beats_blend_all_folds
    # today (an unusable blend already fails that gate) and is required here
    # anyway, so a future edit that softens the per-fold gate cannot quietly
    # promote on a blend arm that was never computed.
    summary["promote_v3_means"] = bool(
        summary["generative_beats_v1_all_folds"]
        and summary["generative_beats_blend_all_folds"]
        and summary["blend_usable_all_folds"]
        and summary["n_folds"] >= 1
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": folds,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"promote_v3_means={summary['promote_v3_means']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
