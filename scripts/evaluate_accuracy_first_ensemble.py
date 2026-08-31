"""Evaluate and conditionally publish an accuracy-first 2026 ensemble.

Chronology is fixed:

* 2023 calibrates the first ADP -> points curves.
* 2024 fits candidate convex weights.
* 2025 selects arms without refitting them.
* after selection, 2024-2025 refit the accepted design for 2026.

The script never overwrites the native v1 board or the calibrated v3
distribution.  All outputs land below ``output/accuracy_first_2026``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.calibrate_v3_distribution import score_draws
from scripts.ensemble_v1_v2 import DEFAULT_V2_ROOT, load_v2
from src.projection.contracts import BACKTEST_DIR, MODEL_V3_DIR, OUTPUT_DIR
from src.projection.data_prep import get_conn
from src.projection.evaluation.accuracy_first import (
    POSITIONS,
    TOP_ADP,
    apply_market_curves,
    apply_position_weights,
    assemble_selected,
    bootstrap_deltas,
    canonical_json_hash,
    choose_position_arms,
    fit_market_curves,
    fit_position_weights,
    incumbent_points,
    load_consensus_snapshot,
    metric_block,
    paired_error_block,
    passes_metrics,
    resolve_candidate_inputs,
    sha256_file,
)
from src.projection.fantasy_evaluation import attach_actual_outcomes, build_leakage_safe_long_board
from src.projection.fantasy_points import SCORING
from src.projection.features import build_player_season_features
from src.projection.inference.simulate import simulate_season_distributions
from src.projection.models.uncertainty import (
    AVAILABILITY_ROWS_PATH,
    PLAYER_SEASON_ROWS_PATH,
    SHARE_ROWS_PATH,
    TEAM_ROWS_PATH,
    build_joint_donors,
    fit_uncertainty_manifest,
    joint_bootstrap_draws,
)


DEFAULT_OUT_DIR = Path(OUTPUT_DIR) / "accuracy_first_2026"
CALIBRATION_PATH = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
INCUMBENT_WEIGHTS_PATH = ROOT / "src" / "draft_assistant" / "ensemble_weights.json"
CONSENSUS_DIR = ROOT / "data" / "consensus"
HISTORICAL_FOLDS = ((2023, 2024), (2024, 2025))
ARM_MODELS = {
    "model_only": ("v1_pred", "v2_pred", "v3_p50"),
    "market_no_v3": ("v1_pred", "v2_pred", "adp_points"),
    "full": ("v1_pred", "v2_pred", "v3_p50", "adp_points"),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval(season: int) -> pd.DataFrame:
    path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{season}.csv"
    frame = pd.read_csv(path)
    out = pd.DataFrame({
        "player_id": frame["player_id"].astype(str),
        "display_name": frame["display_name"],
        "position": frame["preseason_position"].astype(str),
        "actual_points": pd.to_numeric(frame["actual_points"], errors="coerce").fillna(0.0),
        "v1_pred": pd.to_numeric(frame["model_points_end_to_end"], errors="coerce").fillna(0.0),
    })
    out["season"] = int(season)
    return out


def _load_v2_points(season: int) -> pd.DataFrame:
    frame = load_v2(season, DEFAULT_V2_ROOT)
    if frame is None or frame.empty:
        raise FileNotFoundError(
            f"Missing v2 OOF predictions for {season}: {DEFAULT_V2_ROOT / 'outputs' / 'preseason_oof.parquet'}"
        )
    out = frame[["player_id", "v2_pred"]].copy()
    out["player_id"] = out["player_id"].astype(str)
    return out.drop_duplicates("player_id")


def _actual_points(conn, features: pd.DataFrame, season: int) -> pd.DataFrame:
    population = pd.DataFrame({"player_id": features["player_id"].astype(str).unique()})
    actual = attach_actual_outcomes(population, features, season)
    actual["player_id"] = actual["player_id"].astype(str)
    return actual[["player_id", "actual_points"]]


def exact_v3_fold_p50(
    conn,
    features: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
    n_draws: int,
    uncertainty_rows: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict, dict]:
    """Regenerate the exact live-path joint-bootstrap p50 for one fold."""
    board = build_leakage_safe_long_board(conn, features, source_season, target_season)
    training = lambda frame: frame[frame["test_season"].lt(target_season)].copy()
    manifest = fit_uncertainty_manifest(
        training(uncertainty_rows["team"]),
        training(uncertainty_rows["share"]),
        training(uncertainty_rows["availability"]),
        training_cutoff=target_season - 1,
        player_residuals=training(uncertainty_rows["residuals"]),
    )
    option_draws = simulate_season_distributions(
        board,
        n_draws=n_draws,
        seed=100 + target_season,
        mode="full",
        uncertainty_manifest=manifest,
        use_projection_uncertainty=True,
    )
    donors = build_joint_donors(
        uncertainty_rows["player_season"][
            uncertainty_rows["player_season"]["test_season"].lt(target_season)
        ],
        SCORING,
    )
    draws = joint_bootstrap_draws(
        option_draws,
        board,
        donors,
        rng=np.random.default_rng(900 + target_season),
    )
    summary = (
        draws.groupby(["player_id", "position", "team"], observed=True)["fantasy_pts_season"]
        .quantile(0.50)
        .rename("v3_p50")
        .reset_index()
    )
    summary["player_id"] = summary["player_id"].astype(str)
    metrics = score_draws(draws, _actual_points(conn, features, target_season))
    provenance = {
        "source_season": int(source_season),
        "target_season": int(target_season),
        "n_draws": int(n_draws),
        "uncertainty_training_seasons": manifest.get("training_seasons") or [],
        "uncertainty_artifact_hash": manifest.get("artifact_hash"),
        "joint_donor_seasons": sorted(donors["test_season"].unique().astype(int).tolist()),
    }
    if any(int(season) >= target_season for season in provenance["uncertainty_training_seasons"]):
        raise RuntimeError(f"Uncertainty leakage detected for target {target_season}")
    if any(int(season) >= target_season for season in provenance["joint_donor_seasons"]):
        raise RuntimeError(f"Joint donor leakage detected for target {target_season}")
    return summary, metrics, provenance


def _parity_check(target_season: int, observed: dict, calibration: dict) -> dict:
    expected_fold = next(
        fold for fold in calibration["folds"] if int(fold["target_season"]) == target_season
    )
    expected = expected_fold["joint_bootstrap"]["overall"]
    actual = observed["overall"]
    keys = ("n", "coverage", "p50_mae", "p50_spearman")
    deltas = {
        key: float(actual[key]) - float(expected[key])
        for key in keys
    }
    passed = all(abs(delta) <= 1e-12 for delta in deltas.values())
    return {"pass": passed, "deltas": deltas, "expected": expected, "observed": actual}


def _market_history(target_season: int) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for season in range(2023, target_season):
        eval_frame = _load_eval(season)
        snapshot_path = CONSENSUS_DIR / f"consensus_{season}.json"
        consensus, meta = load_consensus_snapshot(snapshot_path, expected_season=season)
        merged = eval_frame.merge(
            consensus[["player_id", "adp"]], on="player_id", how="inner"
        ).dropna(subset=["adp"])
        rows.append(merged[["player_id", "position", "actual_points", "adp", "season"]])
        provenance.append({
            "season": season,
            "as_of": meta.get("as_of"),
            "sha256": sha256_file(snapshot_path),
            "n": int(len(merged)),
        })
    if not rows:
        raise ValueError(f"No earlier market seasons available for {target_season}")
    return pd.concat(rows, ignore_index=True), provenance


def attach_market_signal(frame: pd.DataFrame, *, target_season: int) -> tuple[pd.DataFrame, dict]:
    history, history_provenance = _market_history(target_season)
    curves = fit_market_curves(history)
    snapshot_path = CONSENSUS_DIR / f"consensus_{target_season}.json"
    consensus, meta = load_consensus_snapshot(snapshot_path, expected_season=target_season)
    out = frame.merge(
        consensus[["player_id", "adp", "ecr"]], on="player_id", how="left"
    )
    out["adp_points"] = apply_market_curves(out, curves)
    out["draft_relevant_top120"] = out["adp"].le(TOP_ADP) & out["adp"].notna()
    return out, {
        "target_season": target_season,
        "target_as_of": meta.get("as_of"),
        "target_sha256": sha256_file(snapshot_path),
        "calibration_seasons": history_provenance,
        "ecr_used_as_weight": False,
    }


def _incumbent_weights() -> dict[str, dict[str, float]]:
    return (_load_json(INCUMBENT_WEIGHTS_PATH).get("weights") or {})


def build_historical_frame(
    season: int,
    v3_p50: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    frame = _load_eval(season).merge(_load_v2_points(season), on="player_id", how="left")
    frame = frame.merge(v3_p50[["player_id", "v3_p50"]], on="player_id", how="left")
    coverage = {
        "n": int(len(frame)),
        "v2_raw": int(frame["v2_pred"].notna().sum()),
        "v3_raw": int(frame["v3_p50"].notna().sum()),
    }
    frame, market_meta = attach_market_signal(frame, target_season=season)
    frame = resolve_candidate_inputs(frame)
    frame["incumbent_pred"] = incumbent_points(frame, _incumbent_weights())
    coverage.update({
        "adp_raw": int(frame["adp"].notna().sum()),
        "draft_relevant_top120": int(frame["draft_relevant_top120"].sum()),
    })
    return frame, {"coverage": coverage, "market": market_meta}


def _fit_arms(train: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    eligible = train[train["draft_relevant_top120"]].copy()
    return {
        arm: fit_position_weights(eligible, columns)
        for arm, columns in ARM_MODELS.items()
    }


def _apply_arms(frame: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = frame.copy()
    for arm, columns in ARM_MODELS.items():
        out[f"{arm}_pred"] = np.nan
        eligible = out[out["draft_relevant_top120"]].copy()
        if eligible.empty:
            continue
        scored = apply_position_weights(eligible, weights[arm], out_col=f"{arm}_pred")
        out.loc[scored.index, f"{arm}_pred"] = scored[f"{arm}_pred"]
    return out


def evaluate_holdout(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    n_boot: int,
) -> tuple[pd.DataFrame, dict, dict]:
    fitted = _fit_arms(train)
    scored = _apply_arms(holdout, fitted)
    eligible = scored[scored["draft_relevant_top120"]].copy()
    candidate_cols = {arm: f"{arm}_pred" for arm in ARM_MODELS}
    proposed_selections, position_evidence = choose_position_arms(
        eligible,
        candidate_cols,
        arm_complexity={arm: len(columns) for arm, columns in ARM_MODELS.items()},
    )
    proposed = assemble_selected(eligible, proposed_selections, candidate_cols)
    incumbent_metrics = metric_block(proposed, "incumbent_pred")
    proposed_metrics = metric_block(proposed, "selected_pred")
    overall_pass = passes_metrics(proposed_metrics, incumbent_metrics)
    changed = any(arm != "incumbent" for arm in proposed_selections.values())
    if overall_pass and changed:
        verdict = "promote_accuracy_ensemble"
        final_selections = proposed_selections
    else:
        verdict = "hold_incumbent"
        final_selections = {position: "incumbent" for position in POSITIONS}
    final = assemble_selected(eligible, final_selections, candidate_cols)
    scored["selected_arm"] = "outside_top120"
    scored["selected_pred"] = scored["incumbent_pred"]
    scored.loc[final.index, "selected_arm"] = final["selected_arm"]
    scored.loc[final.index, "selected_pred"] = final["selected_pred"]

    by_position = {}
    for position in POSITIONS:
        cell = final[final["position"].eq(position)]
        by_position[position] = {
            "incumbent": metric_block(cell, "incumbent_pred"),
            "selected": metric_block(cell, "selected_pred"),
            "paired_error": paired_error_block(cell, "selected_pred"),
        }
    report = {
        "verdict": verdict,
        "proposed_position_selections": proposed_selections,
        "final_position_selections": final_selections,
        "position_evidence": position_evidence,
        "overall_gate": {
            "pass": bool(overall_pass and changed),
            "candidate_changes_any_position": changed,
            "incumbent": incumbent_metrics,
            "proposed": proposed_metrics,
        },
        "final": {
            "overall": metric_block(final, "selected_pred"),
            "by_position": by_position,
            "paired_error": paired_error_block(final, "selected_pred"),
            "bootstrap": bootstrap_deltas(final, "selected_pred", n_boot=n_boot),
        },
    }
    return scored, fitted, report


def _verify_live_v3_provenance(v1: pd.DataFrame) -> dict:
    manifest_path = Path(MODEL_V3_DIR) / "simulation_manifest_2026.json"
    manifest = _load_json(manifest_path)
    run_ids = v1.get("projection_run_id", pd.Series(dtype=str)).dropna().astype(str).unique()
    if len(run_ids) != 1:
        raise RuntimeError("2026 v1 board does not have exactly one projection_run_id")
    if str(manifest.get("source_projection_run_id")) != str(run_ids[0]):
        raise RuntimeError("2026 v3 summary is stale relative to the v1 board")
    return {
        "projection_run_id": str(run_ids[0]),
        "simulation_manifest_sha256": sha256_file(manifest_path),
        "distribution_mode": manifest.get("distribution_mode"),
        "n_draws": manifest.get("n_draws"),
    }


def _build_2026_signals() -> tuple[pd.DataFrame, dict]:
    v1_path = Path(OUTPUT_DIR) / "fantasy_points_2026.csv"
    v2_path = Path(OUTPUT_DIR) / "model_v2" / "fantasy_points_2026.csv"
    v3_path = Path(MODEL_V3_DIR) / "simulation_summary_2026.csv"
    v1 = pd.read_csv(v1_path)
    live_provenance = _verify_live_v3_provenance(v1)
    base = pd.DataFrame({
        "player_id": v1["player_id"].astype(str),
        "display_name": v1["display_name"],
        "position": v1["position"].astype(str),
        "v1_pred": pd.to_numeric(v1["fantasy_pts_season"], errors="coerce"),
    })
    v2 = pd.read_csv(v2_path)[["player_id", "fantasy_pts_season"]].rename(
        columns={"fantasy_pts_season": "v2_pred"}
    )
    v2["player_id"] = v2["player_id"].astype(str)
    v3 = pd.read_csv(v3_path)[["player_id", "p50"]].rename(columns={"p50": "v3_p50"})
    v3["player_id"] = v3["player_id"].astype(str)
    frame = base.merge(v2, on="player_id", how="left").merge(v3, on="player_id", how="left")
    raw_coverage = {
        "n": int(len(frame)),
        "v2_raw": int(frame["v2_pred"].notna().sum()),
        "v3_raw": int(frame["v3_p50"].notna().sum()),
    }
    frame = resolve_candidate_inputs(frame)
    frame["incumbent_pred"] = incumbent_points(frame, _incumbent_weights())
    frame, market_meta = attach_market_signal(frame, target_season=2026)
    raw_coverage.update({
        "adp_raw": int(frame["adp"].notna().sum()),
        "draft_relevant_top120": int(frame["draft_relevant_top120"].sum()),
    })
    return frame, {
        "coverage": raw_coverage,
        "market": market_meta,
        "live_v3": live_provenance,
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (v1_path, v2_path, v3_path)
        },
    }


def refit_selected_weights(
    history: pd.DataFrame,
    selections: dict[str, str],
) -> dict[str, dict]:
    eligible = history[history["draft_relevant_top120"]].copy()
    result: dict[str, dict] = {}
    incumbent = _incumbent_weights()
    refitted_arms = {
        arm: fit_position_weights(eligible, ARM_MODELS[arm])
        for arm in set(selections.values())
        if arm != "incumbent"
    }
    for position, arm in selections.items():
        if arm == "incumbent":
            result[position] = {"arm": arm, "weights": incumbent[position]}
            continue
        result[position] = {"arm": arm, "weights": refitted_arms[arm][position]}
    return result


def apply_selected_2026(
    frame: pd.DataFrame,
    selected_weights: dict[str, dict],
) -> pd.DataFrame:
    out = frame.copy()
    out["accuracy_ensemble_applied"] = False
    out["accuracy_ensemble_arm"] = "incumbent"
    out["accuracy_ensemble_pred"] = out["incumbent_pred"]
    eligible = out["draft_relevant_top120"]
    for position, spec in selected_weights.items():
        mask = eligible & out["position"].eq(position)
        if not mask.any() or spec["arm"] == "incumbent":
            continue
        total = pd.Series(0.0, index=out.index[mask])
        for column, weight in spec["weights"].items():
            total = total + float(weight) * out.loc[mask, column].astype(float)
        out.loc[mask, "accuracy_ensemble_pred"] = total
        out.loc[mask, "accuracy_ensemble_applied"] = True
        out.loc[mask, "accuracy_ensemble_arm"] = spec["arm"]
    return out


def write_2026_board(signals: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    native = pd.read_csv(Path(OUTPUT_DIR) / "fantasy_points_2026.csv")
    native["player_id"] = native["player_id"].astype(str)
    cols = [
        "player_id", "incumbent_pred", "accuracy_ensemble_pred",
        "accuracy_ensemble_applied", "accuracy_ensemble_arm", "adp", "adp_points",
        "v2_pred", "v3_p50",
    ]
    out = native.merge(signals[cols], on="player_id", how="left")
    out["accuracy_ensemble_points_before"] = out["fantasy_pts_season"]
    out["fantasy_pts_season"] = out["accuracy_ensemble_pred"].fillna(
        out["fantasy_pts_season"]
    )
    games = pd.to_numeric(out["projected_games"], errors="coerce").replace(0, np.nan)
    out["fantasy_pts"] = (out["fantasy_pts_season"] / games).fillna(
        out["fantasy_pts_season"] / 17.0
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def verify_published_board(board: pd.DataFrame, signals: pd.DataFrame) -> dict:
    expected = signals[["player_id", "accuracy_ensemble_pred"]].copy()
    expected["player_id"] = expected["player_id"].astype(str)
    actual = board[["player_id", "fantasy_pts_season"]].copy()
    actual["player_id"] = actual["player_id"].astype(str)
    joined = actual.merge(expected, on="player_id", how="inner")
    delta = (
        pd.to_numeric(joined["fantasy_pts_season"], errors="coerce")
        - pd.to_numeric(joined["accuracy_ensemble_pred"], errors="coerce")
    ).abs()
    result = {
        "pass": bool(len(joined) == len(board) and delta.max() <= 1e-10),
        "n": int(len(joined)),
        "max_abs_delta": float(delta.max()),
    }
    if not result["pass"]:
        raise RuntimeError(f"Published 2026 board does not reproduce selected weights: {result}")
    return result


def _market_rank_diagnostic(frame: pd.DataFrame) -> dict:
    eligible = frame[frame["draft_relevant_top120"]].copy()
    points = eligible["accuracy_ensemble_pred"]
    return {
        "n_adp": int(eligible["adp"].notna().sum()),
        "points_vs_adp_spearman": float(points.corr(-eligible["adp"], method="spearman")),
        "n_ecr": int(eligible["ecr"].notna().sum()),
        "points_vs_ecr_spearman": (
            float(points[eligible["ecr"].notna()].corr(
                -eligible.loc[eligible["ecr"].notna(), "ecr"], method="spearman"
            ))
            if eligible["ecr"].notna().sum() >= 3 else None
        ),
        "note": "ECR is diagnostic only and has zero fitted weight",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-apply-2026", action="store_true")
    args = parser.parse_args()

    calibration = _load_json(CALIBRATION_PATH)
    n_draws = int(args.draws or calibration["n_draws"])
    if n_draws != int(calibration["n_draws"]):
        raise SystemExit("Exact parity requires --draws to match the calibration artifact")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration_hash = sha256_file(CALIBRATION_PATH)
    # Fail cheap before entering either 1,000-draw fold.
    for target in (2024, 2025, 2026):
        history, _ = _market_history(target)
        fit_market_curves(history)
        load_consensus_snapshot(
            CONSENSUS_DIR / f"consensus_{target}.json", expected_season=target
        )
    for season in (2024, 2025):
        _load_v2_points(season)

    uncertainty_rows = {
        "team": pd.read_parquet(TEAM_ROWS_PATH),
        "share": pd.read_parquet(SHARE_ROWS_PATH),
        "availability": pd.read_parquet(AVAILABILITY_ROWS_PATH),
        "player_season": pd.read_parquet(PLAYER_SEASON_ROWS_PATH),
        "residuals": pd.read_parquet(Path(BACKTEST_DIR) / "residuals_rolling.parquet"),
    }
    conn = get_conn()
    features = build_player_season_features(conn)
    historical: dict[int, pd.DataFrame] = {}
    fold_meta: dict[str, dict] = {}
    for source, target in HISTORICAL_FOLDS:
        cache_path = args.out_dir / f"v3_exact_p50_{target}.parquet"
        cache_meta_path = args.out_dir / f"v3_exact_p50_{target}.json"
        cached = None
        if cache_path.exists() and cache_meta_path.exists():
            candidate = _load_json(cache_meta_path)
            if (
                int(candidate.get("n_draws", -1)) == n_draws
                and candidate.get("calibration_sha256") == calibration_hash
                and candidate.get("p50_sha256") == sha256_file(cache_path)
                and (candidate.get("parity") or {}).get("pass") is True
            ):
                cached = candidate
        if cached:
            print(f"Reuse parity-checked v3 p50 fold {source}->{target}...", flush=True)
            p50 = pd.read_parquet(cache_path)
            metrics = cached["metrics"]
            provenance = cached["provenance"]
            parity = cached["parity"]
        else:
            print(f"Exact v3 p50 fold {source}->{target} ({n_draws} draws)...", flush=True)
            p50, metrics, provenance = exact_v3_fold_p50(
                conn,
                features,
                source_season=source,
                target_season=target,
                n_draws=n_draws,
                uncertainty_rows=uncertainty_rows,
            )
            parity = _parity_check(target, metrics, calibration)
            if not parity["pass"]:
                raise RuntimeError(f"v3 exact-path parity failed for {target}: {parity['deltas']}")
            p50.to_parquet(cache_path, index=False)
            cache_meta = {
                "source_season": source,
                "target_season": target,
                "n_draws": n_draws,
                "calibration_sha256": calibration_hash,
                "p50_sha256": sha256_file(cache_path),
                "metrics": metrics,
                "provenance": provenance,
                "parity": parity,
            }
            cache_meta_path.write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")
        historical[target], build_meta = build_historical_frame(target, p50)
        fold_meta[str(target)] = {"v3": provenance, "parity": parity, **build_meta}
    conn.close()

    scored_2025, fit_weights, holdout_report = evaluate_holdout(
        historical[2024], historical[2025], n_boot=args.bootstrap
    )
    historical[2025] = scored_2025
    historical[2024] = _apply_arms(historical[2024], fit_weights)
    historical[2024]["selected_arm"] = "fit_season"
    historical[2024]["selected_pred"] = np.nan

    verdict = holdout_report["verdict"]
    selected_weights = refit_selected_weights(
        pd.concat([historical[2024], historical[2025]], ignore_index=True),
        holdout_report["final_position_selections"],
    )
    weights_payload = {
        "version": "accuracy_first_2026_v1",
        "verdict": verdict,
        "fit_season": 2024,
        "holdout_season": 2025,
        "refit_seasons": [2024, 2025] if verdict == "promote_accuracy_ensemble" else [],
        "population": {"market": "adp", "max_rank": TOP_ADP},
        "selection_rule": "MAE <= incumbent and Spearman >= incumbent, by position and overall",
        "positions": selected_weights,
        "source_hashes": {
            "v3_calibration": calibration_hash,
            "incumbent_weights": sha256_file(INCUMBENT_WEIGHTS_PATH),
            "v3_exact_p50_2024": sha256_file(args.out_dir / "v3_exact_p50_2024.parquet"),
            "v3_exact_p50_2025": sha256_file(args.out_dir / "v3_exact_p50_2025.parquet"),
        },
    }
    weights_payload["artifact_hash"] = canonical_json_hash(weights_payload)

    players_path = args.out_dir / "evaluation_players.parquet"
    report_path = args.out_dir / "report.json"
    weights_path = args.out_dir / "ensemble_weights.json"
    board_path = args.out_dir / "fantasy_points_2026.csv"
    freeze_path = args.out_dir / "freeze_manifest.json"
    pd.concat([historical[2024], historical[2025]], ignore_index=True).to_parquet(
        players_path, index=False
    )
    weights_path.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    application: dict = {"applied": False, "reason": "holdout_gate_did_not_pass"}
    if verdict == "promote_accuracy_ensemble" and not args.no_apply_2026:
        signals_2026, signals_meta = _build_2026_signals()
        signals_2026 = apply_selected_2026(signals_2026, selected_weights)
        board = write_2026_board(signals_2026, board_path)
        reproduction = verify_published_board(board, signals_2026)
        application = {
            "applied": True,
            "board_path": str(board_path.relative_to(ROOT)).replace("\\", "/"),
            "n_players": int(len(board)),
            "n_accuracy_ensemble_applied": int(board["accuracy_ensemble_applied"].fillna(False).sum()),
            "market_diagnostic": _market_rank_diagnostic(signals_2026),
            "reproduction_check": reproduction,
            **signals_meta,
        }

    selected_v3_weights = {
        position: float(spec["weights"].get("v3_p50", 0.0))
        for position, spec in selected_weights.items()
    }
    v3_selected_positions = [
        position for position, weight in selected_v3_weights.items() if weight > 0.0
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": "2026 top-120 ADP half-PPR point MAE with non-regressing rank correlation",
        "verdict": verdict,
        "chronology": {
            "market_calibration_starts": 2023,
            "ensemble_fit": 2024,
            "untouched_selection_holdout": 2025,
            "production_refit": [2024, 2025] if verdict == "promote_accuracy_ensemble" else [],
        },
        "candidate_models": {arm: list(columns) for arm, columns in ARM_MODELS.items()},
        "fit_weights_2024": fit_weights,
        "holdout_2025": holdout_report,
        "v3_marginal_value": {
            "selected": bool(v3_selected_positions),
            "selected_positions": v3_selected_positions,
            "refit_v3_weights": selected_v3_weights,
            "conclusion": (
                "v3 adds positive selected point weight"
                if v3_selected_positions
                else "v3 adds no selected marginal point accuracy; retain it as the distribution overlay"
            ),
        },
        "folds": fold_meta,
        "application_2026": application,
        "artifacts": {
            "players": str(players_path.relative_to(ROOT)).replace("\\", "/"),
            "weights": str(weights_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    freeze_sources = {
        "report.json": sha256_file(report_path),
        "evaluation_players.parquet": sha256_file(players_path),
        "ensemble_weights.json": sha256_file(weights_path),
        str(CALIBRATION_PATH.relative_to(ROOT)).replace("\\", "/"): sha256_file(CALIBRATION_PATH),
        str(INCUMBENT_WEIGHTS_PATH.relative_to(ROOT)).replace("\\", "/"): sha256_file(INCUMBENT_WEIGHTS_PATH),
    }
    if board_path.exists():
        freeze_sources["fantasy_points_2026.csv"] = sha256_file(board_path)
    for relative, digest in (application.get("sources") or {}).items():
        freeze_sources[f"source:{relative}"] = digest
    market_application = application.get("market") or {}
    if market_application.get("target_sha256"):
        freeze_sources["source:data/consensus/consensus_2026.json"] = market_application[
            "target_sha256"
        ]
    for item in market_application.get("calibration_seasons") or []:
        freeze_sources[f"source:data/consensus/consensus_{item['season']}.json"] = item["sha256"]
    for target in (2024, 2025):
        for suffix in ("json", "parquet"):
            path = args.out_dir / f"v3_exact_p50_{target}.{suffix}"
            freeze_sources[path.name] = sha256_file(path)
    freeze = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "recoverable": True,
        "canonical_v1_unchanged": True,
        "v3_distribution_unchanged": True,
        "files": freeze_sources,
    }
    freeze["manifest_hash"] = canonical_json_hash(freeze)
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    print(json.dumps({
        "verdict": verdict,
        "position_selections": holdout_report["final_position_selections"],
        "incumbent": holdout_report["overall_gate"]["incumbent"],
        "proposed": holdout_report["overall_gate"]["proposed"],
        "application_2026": application,
        "out_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
