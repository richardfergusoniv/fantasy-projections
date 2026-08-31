"""Exact rolling calibration of the shipped v3 season distribution.

The scored path is the production ``mode=full`` simulator and realised season
fantasy points.  Uncertainty for a target fold is fitted only on earlier OOF
fold rows.  The report is the sole interval artifact consumed by the v3 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR
from src.projection.data_prep import get_conn
from src.projection.fantasy_evaluation import attach_actual_outcomes, build_leakage_safe_long_board
from src.projection.features import build_player_season_features
from src.projection.inference.simulate import simulate_season_distributions
from src.projection.fantasy_points import SCORING
from src.projection.models.uncertainty import (
    AVAILABILITY_ROWS_PATH,
    JOINT_DONORS_PATH,
    PLAYER_SEASON_ROWS_PATH,
    SHARE_ROWS_PATH,
    TEAM_ROWS_PATH,
    UNCERTAINTY_MANIFEST_PATH,
    build_joint_donors,
    fit_uncertainty_manifest,
    joint_bootstrap_draws,
)

OUT_PATH = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
FOLDS = ((2023, 2024), (2024, 2025))
ALPHA = 0.20


def _actual_points(conn, features, season: int) -> pd.DataFrame:
    population = pd.DataFrame({"player_id": features["player_id"].astype(str).unique()})
    actual = attach_actual_outcomes(population, features, season)
    actual["player_id"] = actual["player_id"].astype(str)
    return actual[["player_id", "actual_points"]]


def _crps(samples: np.ndarray, actual: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    first = float(np.mean(np.abs(x - actual)))
    coeff = 2 * np.arange(1, n + 1) - n - 1
    half_pairwise = float(np.sum(coeff * x) / (n * n))
    return first - half_pairwise


def score_draws(draws: pd.DataFrame, actual: pd.DataFrame) -> dict:
    grouped = draws.groupby(["player_id", "position", "team"], observed=True)
    summary = grouped["fantasy_pts_season"].quantile([0.10, 0.50, 0.90]).unstack()
    summary.columns = ["p10", "p50", "p90"]
    summary = summary.reset_index()
    joined = summary.merge(actual, on="player_id", how="inner").dropna()

    crps_by_player = {}
    for player_id, grp in draws.groupby("player_id", observed=True):
        row = joined[joined["player_id"].eq(str(player_id))]
        if not row.empty:
            crps_by_player[str(player_id)] = _crps(
                grp["fantasy_pts_season"].to_numpy(), float(row["actual_points"].iloc[0]))
    joined["crps"] = joined["player_id"].map(crps_by_player)

    def metrics(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"n": 0}
        y = frame["actual_points"].to_numpy(dtype=float)
        lo = frame["p10"].to_numpy(dtype=float)
        med = frame["p50"].to_numpy(dtype=float)
        hi = frame["p90"].to_numpy(dtype=float)
        interval_score = (
            hi - lo
            + (2.0 / ALPHA) * np.maximum(lo - y, 0.0)
            + (2.0 / ALPHA) * np.maximum(y - hi, 0.0)
        )
        return {
            "n": int(len(frame)),
            "coverage": float(np.mean((lo <= y) & (y <= hi))),
            "mean_width": float(np.mean(hi - lo)),
            "interval_score": float(np.mean(interval_score)),
            "crps": float(frame["crps"].mean()),
            "p50_mae": float(np.mean(np.abs(med - y))),
            "p50_bias": float(np.mean(med - y)),
            "p50_spearman": float(pd.Series(med).corr(pd.Series(y), method="spearman")),
        }

    return {
        "overall": metrics(joined),
        "by_position": {
            str(position): metrics(grp)
            for position, grp in joined.groupby("position", observed=True)
        },
    }


def _aggregate(folds: list[dict], arm: str) -> dict:
    weighted_keys = (
        "coverage", "mean_width", "interval_score", "crps",
        "p50_mae", "p50_bias", "p50_spearman",
    )
    rows = [f[arm]["overall"] for f in folds]
    total = sum(r.get("n", 0) for r in rows)
    overall = {"n": int(total)}
    for key in weighted_keys:
        overall[key] = float(sum(r[key] * r["n"] for r in rows) / total) if total else None
    positions = {}
    for position in ("QB", "RB", "WR", "TE"):
        cells = [f[arm]["by_position"].get(position) for f in folds]
        cells = [c for c in cells if c and c.get("n")]
        n = sum(c["n"] for c in cells)
        if n:
            positions[position] = {"n": int(n), **{
                key: float(sum(c[key] * c["n"] for c in cells) / n)
                for key in weighted_keys
            }}
    return {"overall": overall, "by_position": positions}


def _acceptance(aggregate: dict, folds: list[dict], arm: str) -> dict:
    option = aggregate[arm]
    baseline = aggregate["baseline"]
    overall = option["overall"]
    fold_ok = all(0.72 <= f[arm]["overall"]["coverage"] <= 0.88 for f in folds)
    position_ok = all(
        cell["n"] < 50 or cell["coverage"] >= 0.70
        for cell in option["by_position"].values()
    )
    gates = {
        "aggregate_coverage_75_85": 0.75 <= overall["coverage"] <= 0.85,
        "fold_coverage_72_88": fold_ok,
        "position_coverage_floor_70": position_ok,
        "interval_score_improves": overall["interval_score"] < baseline["overall"]["interval_score"],
        "p50_mae_no_material_regression": overall["p50_mae"] <= baseline["overall"]["p50_mae"] + 0.5,
        "p50_spearman_no_material_regression": (
            overall["p50_spearman"] >= baseline["overall"]["p50_spearman"] - 0.01
        ),
    }
    return {"pass": all(gates.values()), "gates": gates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Reuse the existing baseline/Option-A scores and run only the fallback draws",
    )
    args = parser.parse_args()

    previous_folds = {}
    if args.fallback_only and args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        if int(previous.get("n_draws", -1)) != args.draws:
            raise SystemExit("--fallback-only requires the same --draws as the existing report")
        previous_folds = {int(f["target_season"]): f for f in previous.get("folds", [])}

    team_rows = pd.read_parquet(TEAM_ROWS_PATH)
    share_rows = pd.read_parquet(SHARE_ROWS_PATH)
    availability_rows = pd.read_parquet(AVAILABILITY_ROWS_PATH)
    player_season_rows = pd.read_parquet(PLAYER_SEASON_ROWS_PATH)
    residuals_path = Path(BACKTEST_DIR) / "residuals_rolling.parquet"
    player_residuals = pd.read_parquet(residuals_path)

    conn = get_conn()
    features = build_player_season_features(conn)
    fold_reports = []
    for source, target in FOLDS:
        print(f"exact distribution fold {source}->{target}...", flush=True)
        board = build_leakage_safe_long_board(conn, features, source, target)
        actual = _actual_points(conn, features, target)
        training = lambda frame: frame[frame["test_season"].lt(target)].copy()
        manifest = fit_uncertainty_manifest(
            training(team_rows), training(share_rows), training(availability_rows),
            training_cutoff=target - 1,
            player_residuals=training(player_residuals),
        )
        if args.fallback_only and target not in previous_folds:
            raise SystemExit(f"Missing prior fold {target} for --fallback-only")
        baseline_draws = None
        if not args.fallback_only:
            baseline_draws = simulate_season_distributions(
                board,
                n_draws=args.draws,
                seed=100 + target,
                mode="full",
                uncertainty_manifest={},
                use_projection_uncertainty=False,
            )
        option_draws = simulate_season_distributions(
            board,
            n_draws=args.draws,
            seed=100 + target,
            mode="full",
            uncertainty_manifest=manifest,
            use_projection_uncertainty=True,
        )
        donors = build_joint_donors(
            player_season_rows[player_season_rows["test_season"].lt(target)], SCORING)
        fallback_draws = joint_bootstrap_draws(
            option_draws, board, donors, rng=np.random.default_rng(900 + target))
        fold_reports.append({
            "source_season": source,
            "target_season": target,
            "training_seasons": manifest["training_seasons"],
            "uncertainty_artifact_hash": manifest["artifact_hash"],
            "baseline": (
                previous_folds[target]["baseline"]
                if args.fallback_only else score_draws(baseline_draws, actual)
            ),
            "option_a": (
                previous_folds[target]["option_a"]
                if args.fallback_only else score_draws(option_draws, actual)
            ),
            "joint_bootstrap": score_draws(fallback_draws, actual),
        })
    conn.close()

    aggregate = {
        "baseline": _aggregate(fold_reports, "baseline"),
        "option_a": _aggregate(fold_reports, "option_a"),
        "joint_bootstrap": _aggregate(fold_reports, "joint_bootstrap"),
    }
    option_a_acceptance = _acceptance(aggregate, fold_reports, "option_a")
    fallback_acceptance = _acceptance(aggregate, fold_reports, "joint_bootstrap")
    if option_a_acceptance["pass"]:
        selected_mode = "generative_projection_uncertainty"
    elif fallback_acceptance["pass"]:
        selected_mode = "joint_bootstrap"
    else:
        selected_mode = "hold"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "basis": "rolling_origin_exact_full_simulator_season_fantasy_points",
        "simulation_mode": "full",
        "distribution_mode": "generative_projection_uncertainty",
        "n_draws": args.draws,
        "target_coverage": 0.80,
        "folds": fold_reports,
        "aggregate": aggregate,
        "option_a_acceptance": option_a_acceptance,
        "fallback_acceptance": fallback_acceptance,
        "selected_distribution_mode": selected_mode,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Configure the live manifest only from a passing exact calibration.
    if UNCERTAINTY_MANIFEST_PATH.exists():
        live_manifest = json.loads(UNCERTAINTY_MANIFEST_PATH.read_text(encoding="utf-8"))
        live_manifest["selected_distribution_mode"] = selected_mode
        live_manifest["calibration_artifact"] = str(args.out)
        live_manifest["calibration_generated_at"] = report["generated_at"]
        if selected_mode == "joint_bootstrap":
            live_donors = build_joint_donors(player_season_rows, SCORING)
            JOINT_DONORS_PATH.parent.mkdir(parents=True, exist_ok=True)
            live_donors.to_parquet(JOINT_DONORS_PATH, index=False)
            donor_bytes = JOINT_DONORS_PATH.read_bytes()
            live_manifest["joint_donors"] = {
                "path": str(JOINT_DONORS_PATH),
                "sha256": hashlib.sha256(donor_bytes).hexdigest(),
                "n": int(len(live_donors)),
            }
        payload = dict(live_manifest)
        payload.pop("artifact_hash", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        live_manifest["artifact_hash"] = hashlib.sha256(encoded).hexdigest()
        UNCERTAINTY_MANIFEST_PATH.write_text(json.dumps(live_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps({
        "baseline": aggregate["baseline"]["overall"],
        "option_a": aggregate["option_a"]["overall"],
        "joint_bootstrap": aggregate["joint_bootstrap"]["overall"],
        "option_a_acceptance": option_a_acceptance,
        "fallback_acceptance": fallback_acceptance,
        "selected_distribution_mode": selected_mode,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
