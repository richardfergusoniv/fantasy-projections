"""Persist rolling-origin backtest artifacts for auditability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.backtest import (
    ROLLING_TEST_PAIRS,
    backtest_availability,
    backtest_position_stat,
    backtest_season_totals,
    backtest_team_total,
    rolling_residual_rows,
    run_rolling_origin_backtest,
    run_rookie_backtest,
    run_veteran_backtest,
)
from src.projection.contracts import BACKTEST_DIR
from src.projection.data_prep import get_conn
from src.projection.evaluation.baselines import attach_all_baselines, baseline_mae
from src.projection.evaluation.calibration import summarize_interval_calibration
from src.projection.features import TARGET_STATS, build_player_season_features
from src.projection.transitions import build_role_transition_pairs, role_rate_label


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _available_train_pairs(test_pair: tuple[int, int]) -> list[tuple[int, int]]:
    available = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    return [pair for pair in available if pair[1] <= test_pair[0]]


def run_position_stat_with_baselines(feat, test_pairs=ROLLING_TEST_PAIRS) -> pd.DataFrame:
    rows = []
    for test_pair in test_pairs:
        train_pairs = _available_train_pairs(test_pair)
        if not train_pairs:
            continue
        for position, stats in TARGET_STATS.items():
            for stat in stats:
                result = backtest_position_stat(
                    feat, position, stat, train_pairs=train_pairs, test_pair=test_pair
                )
                if not result:
                    continue
                train = build_role_transition_pairs(feat, position, stat, train_pairs)
                test = build_role_transition_pairs(feat, position, stat, [test_pair])
                if test.empty:
                    continue
                with_baselines = attach_all_baselines(train, test, position, stat)
                y_col = role_rate_label(stat)
                actual = with_baselines[y_col]
                row = {
                    **result,
                    "test_season": test_pair[1],
                    "n_train_transitions": len(train_pairs),
                }
                for col in with_baselines.columns:
                    if col.startswith("baseline_"):
                        row[f"{col}_mae"] = baseline_mae(actual, with_baselines[col])
                rows.append(row)
    return pd.DataFrame(rows)


def persist_backtests(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or BACKTEST_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    feat = build_player_season_features(conn)

    position_stat = run_position_stat_with_baselines(feat)
    position_stat_path = out_dir / "position_stat_rolling.csv"
    position_stat.to_csv(position_stat_path, index=False)

    team_rows = []
    for test_pair in ROLLING_TEST_PAIRS:
        train_pairs = _available_train_pairs(test_pair)
        if not train_pairs:
            continue
        for fn, kwargs in (
            (backtest_team_total, {}),
            (backtest_team_total, {"stat": "pass_attempts"}),
        ):
            row = fn(feat, train_pairs=train_pairs, test_pair=test_pair, **kwargs)
            if row:
                row["test_season"] = test_pair[1]
                row["n_train_transitions"] = len(train_pairs)
                team_rows.append(row)
    team_totals = pd.DataFrame(team_rows)
    team_path = out_dir / "team_totals_rolling.csv"
    team_totals.to_csv(team_path, index=False)

    residuals = rolling_residual_rows(feat)
    residuals_path = out_dir / "residuals_rolling.parquet"
    residuals.to_parquet(residuals_path, index=False)

    per_season = {}
    for season in sorted(residuals["test_season"].unique()) if not residuals.empty else []:
        sub = residuals[residuals["test_season"] == season]
        path = out_dir / f"residuals_{int(season)}.parquet"
        sub.to_parquet(path, index=False)
        per_season[int(season)] = str(path.relative_to(out_dir)).replace("\\", "/")

    availability = backtest_availability(feat, conn=conn)
    availability.to_csv(out_dir / "availability_rolling.csv", index=False)

    season_totals = backtest_season_totals(feat, conn=conn)
    season_totals.to_csv(out_dir / "season_totals_2025.csv", index=False)

    veteran = run_veteran_backtest(feat)
    veteran.to_csv(out_dir / "veteran_holdout_2025.csv", index=False)

    rolling = run_rolling_origin_backtest(feat)
    rolling.to_csv(out_dir / "position_stat_legacy_rolling.csv", index=False)

    rookies = run_rookie_backtest(conn, feat)
    rookies.to_csv(out_dir / "rookie_rolling.csv", index=False)

    calibration = summarize_interval_calibration(residuals)

    artifacts = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            artifacts.append({
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": [[a, b] for a, b in ROLLING_TEST_PAIRS],
        "residuals_by_season": per_season,
        "calibration_summary": calibration,
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    conn.close()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(BACKTEST_DIR))
    args = parser.parse_args()
    manifest = persist_backtests(args.out)
    print(f"Wrote {len(manifest['artifacts'])} artifacts to {args.out}")
    print(f"Mean interval coverage: {manifest['calibration_summary'].get('mean_coverage')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
