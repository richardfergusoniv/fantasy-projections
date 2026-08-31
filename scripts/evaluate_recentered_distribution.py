"""Evaluate recentered v3 distributions on the untouched 2025 top-120 holdout."""
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
from src.projection.contracts import BACKTEST_DIR, OUTPUT_DIR
from src.projection.data_prep import get_conn
from src.projection.evaluation.accuracy_first import TOP_ADP, sha256_file
from src.projection.evaluation.calibration import coverage, pinball_loss
from src.projection.evaluation.calibration_segments import (
    build_segment_assignments,
    evaluate_all_segments,
    segment_metrics,
)
from src.projection.evaluation.finish_probability_calibration import (
    build_finish_probability_frame,
    evaluate_finish_probability_report,
)
from src.projection.evaluation.finish_probability_gate import (
    evaluate_finish_gate,
    write_finish_probability_gate,
)
from src.projection.fantasy_evaluation import build_leakage_safe_long_board
from src.projection.features import build_player_season_features
from src.projection.fantasy_points import SCORING
from src.projection.inference.recenter import (
    TRANSFORM_VERSION as RECENTER_VERSION,
    player_draw_medians,
    recenter_draws,
)
from src.projection.inference.simulate import simulate_season_distributions, slim_draw_frame
from src.projection.inference.wr_calibration import (
    ARTIFACT_PATH as WR_CALIBRATION_PATH,
    TRANSFORM_VERSION as WR_TRANSFORM_VERSION,
    fit_wr_calibration,
    load_wr_calibration,
    recenter_draws_wr_scaled,
    write_wr_calibration,
)
from src.projection.models.uncertainty import (
    AVAILABILITY_ROWS_PATH,
    PLAYER_SEASON_ROWS_PATH,
    SHARE_ROWS_PATH,
    TEAM_ROWS_PATH,
    build_joint_donors,
    fit_uncertainty_manifest,
    joint_bootstrap_draws,
)

DEFAULT_OUT = Path(OUTPUT_DIR) / "model_v3" / "recentered_holdout_2025.json"
DEFAULT_DRAWS = Path(OUTPUT_DIR) / "model_v3" / "holdout_draws_2025.parquet"
DEFAULT_DRAWS_META = Path(OUTPUT_DIR) / "model_v3" / "holdout_draws_2025.json"
EVAL_PLAYERS = Path(OUTPUT_DIR) / "accuracy_first_2026" / "evaluation_players.parquet"
CALIBRATION_PATH = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
TAIL_CONCERN_THRESHOLD = 0.20
ACCEPTANCE = {
    "coverage_min": 0.75,
    "coverage_max": 0.85,
    "wr_coverage_min": 0.70,
    "position_coverage_min": 0.70,
    "p50_mae_tolerance": 2.0,
    "p50_spearman_tolerance": 0.02,
    "interval_score_tolerance": 5.0,
}


def _acceptance(
    metrics: dict,
    baseline: dict | None = None,
    *,
    segment_summary: dict | None = None,
) -> dict:
    overall = metrics["overall"]
    coverage_val = overall.get("coverage")
    coverage_ok = (
        coverage_val is not None
        and ACCEPTANCE["coverage_min"] <= coverage_val <= ACCEPTANCE["coverage_max"]
    )
    wr_cov = metrics.get("by_position", {}).get("WR", {}).get("coverage")
    wr_coverage_ok = wr_cov is not None and wr_cov >= ACCEPTANCE["wr_coverage_min"]
    position_ok = True
    for pos, cell in metrics.get("by_position", {}).items():
        if cell.get("n", 0) >= 50 and cell.get("coverage", 0.0) < ACCEPTANCE["position_coverage_min"]:
            position_ok = False
            break
    mae_ok = True
    spearman_ok = True
    interval_ok = True
    if baseline:
        mae_ok = overall["p50_mae"] <= baseline["overall"]["p50_mae"] + ACCEPTANCE["p50_mae_tolerance"]
        spearman_ok = (
            overall["p50_spearman"]
            >= baseline["overall"]["p50_spearman"] - ACCEPTANCE["p50_spearman_tolerance"]
        )
        interval_ok = (
            overall["interval_score"]
            <= baseline["overall"]["interval_score"] + ACCEPTANCE["interval_score_tolerance"]
        )
    segment_gate_ok = True
    if segment_summary is not None:
        segment_gate_ok = bool(segment_summary.get("passes_segment_gate", False))
    return {
        "coverage_ok": coverage_ok,
        "wr_coverage_ok": wr_coverage_ok,
        "position_coverage_ok": position_ok,
        "p50_mae_ok": mae_ok,
        "p50_spearman_ok": spearman_ok,
        "interval_score_ok": interval_ok,
        "segment_gate_ok": segment_gate_ok,
        "passes": (
            coverage_ok
            and wr_coverage_ok
            and position_ok
            and mae_ok
            and spearman_ok
            and interval_ok
            and segment_gate_ok
        ),
    }


def load_holdout_frame() -> pd.DataFrame:
    frame = pd.read_parquet(EVAL_PLAYERS)
    holdout = frame[frame["season"].eq(2025)].copy()
    holdout["player_id"] = holdout["player_id"].astype(str)
    adp = pd.to_numeric(holdout["adp"], errors="coerce")
    return holdout.loc[adp.notna() & adp.le(TOP_ADP)].copy()


def _adp_tier(adp: pd.Series) -> pd.Series:
    adp = pd.to_numeric(adp, errors="coerce")
    return np.where(adp.le(36), "top36", "adp37_120")


def _tail_diagnostics(frame: pd.DataFrame) -> dict:
    actual = pd.to_numeric(frame["actual_points"], errors="coerce")
    p10 = pd.to_numeric(frame["recentered_p10"], errors="coerce")
    p50 = pd.to_numeric(frame["recentered_p50"], errors="coerce")
    p90 = pd.to_numeric(frame["recentered_p90"], errors="coerce")
    mask = actual.notna() & p10.notna() & p50.notna() & p90.notna()
    sub = frame.loc[mask]
    if sub.empty:
        return {}
    a = actual[mask].to_numpy(dtype=float)
    lo = p10[mask].to_numpy(dtype=float)
    med = p50[mask].to_numpy(dtype=float)
    hi = p90[mask].to_numpy(dtype=float)
    below = float(np.mean(a < lo))
    inside = float(np.mean((a >= lo) & (a <= hi)))
    above = float(np.mean(a > hi))
    return {
        "share_below_p10": below,
        "share_inside_80": inside,
        "share_above_p90": above,
        "mean_signed_error": float(np.mean(a - med)),
        "median_signed_error": float(np.median(a - med)),
        "p50_mae": float(np.mean(np.abs(a - med))),
        "mean_interval_width_80": float(np.mean(hi - lo)),
        "pinball_loss_q10": pinball_loss(a, 0.10, lo),
        "pinball_loss_q50": pinball_loss(a, 0.50, med),
        "pinball_loss_q90": pinball_loss(a, 0.90, hi),
        "tail_balance_concern": below > TAIL_CONCERN_THRESHOLD or above > TAIL_CONCERN_THRESHOLD,
    }


def wr_diagnostic_report(scored: pd.DataFrame) -> dict:
    wr = scored[scored["position"].eq("WR")].copy()
    overall = _tail_diagnostics(wr)
    slices: dict[str, dict] = {"overall": overall}
    wr["adp_tier"] = _adp_tier(wr["adp"])
    wr["rookie_status"] = np.where(
        wr.get("is_rookie", pd.Series(False, index=wr.index)).fillna(False).astype(bool),
        "rookie",
        "veteran",
    )
    if "projected_games" in wr.columns:
        from src.projection.evaluation.calibration_segments import _bucket_games, _bucket_percentile

        wr["projected_games_bucket"] = _bucket_games(wr["projected_games"])
        wr["point_percentile_bucket"] = _bucket_percentile(
            pd.to_numeric(wr["recentered_p50"], errors="coerce")
        )
    if "target_share" in wr.columns:
        from src.projection.evaluation.calibration_segments import build_segment_assignments

        assignments = build_segment_assignments(wr.rename(columns={"recentered_p50": "pred_p50"}))
        for name, labels in assignments.items():
            if name in ("position", "point_percentile_bucket"):
                continue
            for label, grp in wr.groupby(labels.loc[wr.index], observed=True):
                slices[f"{name}={label}"] = _tail_diagnostics(grp)
    if "depth_tier" in wr.columns:
        for label, grp in wr.groupby(wr["depth_tier"].fillna("unknown"), observed=True):
            slices[f"depth_tier={label}"] = _tail_diagnostics(grp)
    for label, grp in wr.groupby("adp_tier", observed=True):
        slices[f"adp_tier={label}"] = _tail_diagnostics(grp)
    for label, grp in wr.groupby("rookie_status", observed=True):
        slices[f"rookie_status={label}"] = _tail_diagnostics(grp)
    median_err = overall.get("median_signed_error")
    if median_err is not None and median_err > 5.0:
        center_interp = "selected_board_mean_limitation"
    elif median_err is not None and median_err < -5.0:
        center_interp = "selected_board_center_high"
    else:
        center_interp = "within_tolerance"
    return {
        "wr_n": int(len(wr)),
        "center_bias": {
            "mean_signed_error": overall.get("mean_signed_error"),
            "median_signed_error": median_err,
            "interpretation": center_interp,
        },
        "tail_balance_threshold": TAIL_CONCERN_THRESHOLD,
        "slices": slices,
    }


def generate_holdout_draws_2025(
    *,
    n_draws: int,
    out_path: Path = DEFAULT_DRAWS,
    meta_path: Path = DEFAULT_DRAWS_META,
    force: bool = False,
) -> pd.DataFrame:
    """Regenerate exact-path joint-bootstrap draws for the 2024->2025 fold."""
    calibration_hash = sha256_file(CALIBRATION_PATH)
    if (
        not force
        and out_path.exists()
        and meta_path.exists()
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            int(meta.get("n_draws", -1)) == n_draws
            and meta.get("calibration_sha256") == calibration_hash
            and meta.get("draws_sha256") == sha256_file(out_path)
        ):
            print(f"Reusing cached holdout draws: {out_path}", flush=True)
            return pd.read_parquet(out_path)

    print(f"Generating {n_draws} exact-path holdout draws for 2024->2025...", flush=True)
    uncertainty_rows = {
        "team": pd.read_parquet(TEAM_ROWS_PATH),
        "share": pd.read_parquet(SHARE_ROWS_PATH),
        "availability": pd.read_parquet(AVAILABILITY_ROWS_PATH),
        "player_season": pd.read_parquet(PLAYER_SEASON_ROWS_PATH),
        "residuals": pd.read_parquet(Path(BACKTEST_DIR) / "residuals_rolling.parquet"),
    }
    target_season = 2025
    source_season = 2024
    conn = get_conn()
    features = build_player_season_features(conn)
    board = build_leakage_safe_long_board(conn, features, source_season, target_season)
    conn.close()
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
    slim = slim_draw_frame(draws)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out_path, index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_season": source_season,
        "target_season": target_season,
        "n_draws": int(n_draws),
        "calibration_sha256": calibration_hash,
        "draws_sha256": sha256_file(out_path),
        "n_rows": int(len(slim)),
        "path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(slim)} rows)", flush=True)
    return slim


def build_scored_holdout_frame(
    draws: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    wr_scale: float = 1.0,
) -> pd.DataFrame:
    """Player-level frame with v3 and calibrated recentered percentile predictions."""
    selected = holdout.set_index("player_id")["selected_pred"].astype(float)
    v3_quantiles = (
        draws.groupby("player_id", observed=True)["fantasy_pts_season"]
        .quantile([0.10, 0.50, 0.90])
        .unstack()
    )
    v3_quantiles.columns = ["v3_p10", "v3_p50", "v3_p90"]
    recentered = recenter_draws_wr_scaled(draws, selected, wr_scale=wr_scale)
    recentered_quantiles = (
        recentered.groupby("player_id", observed=True)["fantasy_pts_season"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
    )
    recentered_quantiles.columns = [
        "recentered_p10",
        "recentered_p25",
        "recentered_p50",
        "recentered_p75",
        "recentered_p90",
    ]
    scored = v3_quantiles.reset_index().merge(
        recentered_quantiles.reset_index(),
        on="player_id",
        how="outer",
    )
    scored["player_id"] = scored["player_id"].astype(str)
    holdout = holdout.copy()
    holdout["player_id"] = holdout["player_id"].astype(str)
    scored = scored.merge(
        holdout[
            [
                c
                for c in (
                    "player_id",
                    "position",
                    "actual_points",
                    "selected_pred",
                    "adp",
                    "selected_arm",
                )
                if c in holdout.columns
            ]
        ],
        on="player_id",
        how="inner",
    )
    scored["adp_tier"] = _adp_tier(scored["adp"])
    actual = pd.to_numeric(scored["actual_points"], errors="coerce")
    scored["below_p10"] = actual < pd.to_numeric(scored["recentered_p10"], errors="coerce")
    scored["inside_80"] = actual.between(
        pd.to_numeric(scored["recentered_p10"], errors="coerce"),
        pd.to_numeric(scored["recentered_p90"], errors="coerce"),
    )
    scored["above_p90"] = actual > pd.to_numeric(scored["recentered_p90"], errors="coerce")
    scored["signed_error"] = actual - pd.to_numeric(scored["recentered_p50"], errors="coerce")
    scored["absolute_error"] = scored["signed_error"].abs()
    eval_path = Path(OUTPUT_DIR) / "fantasy_evaluation_2025.csv"
    if eval_path.exists():
        meta = pd.read_csv(eval_path)
        meta["player_id"] = meta["player_id"].astype(str)
        meta_cols = [
            c
            for c in (
                "player_id",
                "projected_games",
                "depth_tier",
                "is_rookie",
                "target_share",
                "carry_share",
                "injury_durability_rate",
                "team_target_concentration",
            )
            if c in meta.columns
        ]
        if len(meta_cols) > 1:
            scored = scored.merge(meta[meta_cols], on="player_id", how="left")
    scored["rookie_status"] = np.where(
        scored.get("is_rookie", pd.Series(False, index=scored.index)).fillna(False).astype(bool),
        "rookie",
        "veteran",
    )
    assignments = build_segment_assignments(
        scored.rename(
            columns={
                "recentered_p10": "pred_p10",
                "recentered_p25": "pred_p25",
                "recentered_p50": "pred_p50",
                "recentered_p75": "pred_p75",
                "recentered_p90": "pred_p90",
            }
        )
    )
    for name, labels in assignments.items():
        if name in scored.columns:
            continue
        scored[name] = labels.reindex(scored.index)
    return scored


def evaluate_holdout(
    draws: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    wr_scale: float = 1.0,
) -> dict:
    selected = holdout.set_index("player_id")["selected_pred"].astype(float)
    recentered_baseline = recenter_draws(draws, selected)
    recentered = recenter_draws_wr_scaled(draws, selected, wr_scale=wr_scale)
    actual = holdout[["player_id", "actual_points"]].copy()
    raw_metrics = score_draws(draws, actual)
    recentered_metrics = score_draws(recentered, actual)
    baseline_recentered_metrics = score_draws(recentered_baseline, actual)
    summary = (
        recentered.groupby("player_id", observed=True)["fantasy_pts_season"]
        .median()
        .rename("recentered_p50")
    )
    parity = selected.rename("selected_points").to_frame().join(summary, how="inner")
    parity["abs_error"] = (parity["selected_points"] - parity["recentered_p50"]).abs()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_season": 2025,
        "top120_n": int(len(holdout)),
        "recentered_p50_max_abs_error": float(parity["abs_error"].max()) if not parity.empty else None,
        "raw_metrics": raw_metrics,
        "recentered_metrics": recentered_metrics,
        "baseline_recentered_metrics": baseline_recentered_metrics,
        "wr_scale": float(wr_scale),
    }


def _segment_frame_for_gate(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.rename(
        columns={
            "recentered_p10": "pred_p10",
            "recentered_p25": "pred_p25",
            "recentered_p50": "pred_p50",
            "recentered_p75": "pred_p75",
            "recentered_p90": "pred_p90",
        }
    )


def _promotion_state(acceptance: dict) -> str:
    if acceptance.get("passes"):
        return "holdout_distribution_pass"
    return "wr_calibration_candidate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draws-parquet",
        type=Path,
        default=None,
        help="Raw v3 draw parquet for the 2025 holdout board",
    )
    parser.add_argument(
        "--generate-draws",
        action="store_true",
        help="Generate/cache exact-path 2025 holdout draws before scoring",
    )
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--write-scored-frame", action="store_true")
    parser.add_argument("--calibrate-segments", action="store_true")
    parser.add_argument(
        "--fit-wr-calibration",
        action="store_true",
        help="Fit WR residual scale on pre-2025 training folds and write artifact",
    )
    parser.add_argument(
        "--wr-scale",
        type=float,
        default=None,
        help="Override WR residual scale (default: load trained artifact or 1.0)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-finish-gate", action="store_true")
    args = parser.parse_args()

    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    n_draws = int(args.draws or calibration["n_draws"])
    draws_path = args.draws_parquet or DEFAULT_DRAWS
    if args.generate_draws or not draws_path.exists():
        draws = generate_holdout_draws_2025(
            n_draws=n_draws,
            out_path=draws_path,
            force=args.force_generate,
        )
    else:
        draws = pd.read_parquet(draws_path)

    wr_calibration = load_wr_calibration()
    if args.fit_wr_calibration:
        wr_calibration = fit_wr_calibration(n_draws=n_draws)
        write_wr_calibration(wr_calibration)
        print(
            f"Wrote WR calibration artifact: {WR_CALIBRATION_PATH} "
            f"(scale={wr_calibration['selected_wr_scale']})",
            flush=True,
        )
    wr_scale = float(
        args.wr_scale
        if args.wr_scale is not None
        else (wr_calibration or {}).get("selected_wr_scale", 1.0)
    )

    holdout = load_holdout_frame()
    report = evaluate_holdout(draws, holdout, wr_scale=wr_scale)
    report["draws_path"] = str(draws_path)
    report["n_draws"] = n_draws
    report["provenance"] = {
        "board_model_id": "accuracy_first_ensemble",
        "selected_pred_source": "evaluation_players.parquet",
        "recenter_transform_version": RECENTER_VERSION,
        "wr_transform_version": WR_TRANSFORM_VERSION if not np.isclose(wr_scale, 1.0) else None,
        "wr_residual_scale": wr_scale,
        "wr_calibration_artifact": str(WR_CALIBRATION_PATH) if wr_calibration else None,
        "holdout_population": f"2025_top_{int(TOP_ADP)}_adp",
        "draws_sha256": sha256_file(draws_path) if draws_path.exists() else None,
    }
    if wr_calibration:
        report["wr_calibration_selection"] = wr_calibration.get("selection_metric")

    scored_path = Path(OUTPUT_DIR) / "model_v3" / "holdout_scored_top120_2025.parquet"
    segment_summary = None
    if args.write_scored_frame or args.calibrate_segments:
        scored = build_scored_holdout_frame(draws, holdout, wr_scale=wr_scale)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(scored_path, index=False)
        report["scored_frame_path"] = str(scored_path)
        report["wr_diagnostics"] = wr_diagnostic_report(scored)
        print(f"Wrote scored frame: {scored_path} ({len(scored)} players)", flush=True)
    if args.calibrate_segments and scored_path.exists():
        from src.projection.evaluation.calibration_report import write_calibration_artifacts

        segment_out = write_calibration_artifacts(
            _segment_frame_for_gate(pd.read_parquet(scored_path)),
            Path(OUTPUT_DIR) / "evaluation",
            season=2025,
            run_id="holdout_top120",
            artifact_hashes={
                "draws_sha256": report["provenance"]["draws_sha256"],
                "wr_calibration": sha256_file(WR_CALIBRATION_PATH)
                if WR_CALIBRATION_PATH.exists()
                else None,
            },
        )
        segment_summary = segment_out["summary"]
        report["segment_summary"] = segment_summary
        print("Segment calibration summary:", flush=True)
        print(json.dumps({
            "passes_segment_gate": segment_summary.get("passes_segment_gate"),
            "aggregate_coverage_80": segment_summary.get("aggregate", {}).get("coverage_80"),
            "failed_eligible_segment_count": segment_summary.get("failed_eligible_segment_count"),
            "worst_by_coverage": segment_summary.get("worst_eligible_by_coverage"),
        }, indent=2), flush=True)

    baseline = report["baseline_recentered_metrics"]
    report["acceptance"] = _acceptance(
        report["recentered_metrics"],
        baseline,
        segment_summary=segment_summary,
    )
    report["promotion_state"] = _promotion_state(report["acceptance"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.write_finish_gate:
        selected = holdout.set_index("player_id")["selected_pred"].astype(float)
        recentered_draws = recenter_draws_wr_scaled(draws, selected, wr_scale=wr_scale)
        finish_scored, finish_meta = build_finish_probability_frame(
            recentered_draws,
            holdout,
            training_seasons=(2024,),
        )
        finish_calibration = evaluate_finish_probability_report(finish_scored)
        report["finish_probability_meta"] = finish_meta
        report["finish_calibration"] = finish_calibration
        finish_scored_path = Path(OUTPUT_DIR) / "model_v3" / "holdout_finish_probs_top120_2025.parquet"
        finish_scored_path.parent.mkdir(parents=True, exist_ok=True)
        finish_scored.to_parquet(finish_scored_path, index=False)
        report["finish_scored_frame_path"] = str(finish_scored_path)
        gate = evaluate_finish_gate(
            finish_scored,
            segment_summary=segment_summary,
            recentered_holdout=report["acceptance"],
            finish_calibration=finish_calibration,
            provenance=report.get("provenance"),
            holdout_season=2025,
        )
        gate_path = write_finish_probability_gate(gate)
        report["finish_gate_path"] = str(gate_path)
        if gate.get("passes"):
            report["promotion_state"] = "finish_probability_ready"
        print(f"Wrote finish gate: {gate_path}", flush=True)
        print(json.dumps({
            "finish_verdict": gate.get("verdict"),
            "finish_state": gate.get("state"),
            "publication_verdict": gate.get("publication_verdict"),
            "finish_passes": gate.get("passes"),
            "finish_reasons": gate.get("reasons"),
            "finish_metrics": gate.get("finish_calibration", {}).get("metrics"),
        }, indent=2), flush=True)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "promotion_state": report["promotion_state"],
        "wr_scale": wr_scale,
        "acceptance": report["acceptance"],
        "top120_n": report["top120_n"],
        "recentered_p50_max_abs_error": report["recentered_p50_max_abs_error"],
        "recentered_coverage": report["recentered_metrics"]["overall"].get("coverage"),
        "wr_coverage": report["recentered_metrics"]["by_position"].get("WR", {}).get("coverage"),
        "raw_coverage": report["raw_metrics"]["overall"].get("coverage"),
    }, indent=2))
    return 0 if report["acceptance"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
