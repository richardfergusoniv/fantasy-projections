"""Fit v3 projection uncertainty from rolling-origin forecast rows.

Builds the same leakage-safe long boards used by evaluation, persists row-level
team/share/availability calibration data, and fits the live manifest through
the requested cutoff.  No target-season outcome is used to fit its own fold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.data_prep import get_conn
from src.projection.fantasy_evaluation import build_leakage_safe_long_board
from src.projection.features import build_player_season_features
from src.projection.models.uncertainty import (
    extract_uncertainty_rows,
    extract_player_season_rows,
    fit_uncertainty_manifest,
    write_uncertainty_artifacts,
)
from src.projection.contracts import BACKTEST_DIR

FOLDS = ((2022, 2023), (2023, 2024), (2024, 2025))


def build_rows(conn, features, folds=FOLDS):
    teams, shares, availability, player_seasons = [], [], [], []
    for source, target in folds:
        print(f"building uncertainty fold {source}->{target}...", flush=True)
        board = build_leakage_safe_long_board(conn, features, source, target)
        team, share, avail = extract_uncertainty_rows(board, features, target)
        teams.append(team); shares.append(share); availability.append(avail)
        player_seasons.append(extract_player_season_rows(board, features, target))
    return (
        pd.concat(teams, ignore_index=True),
        pd.concat(shares, ignore_index=True),
        pd.concat(availability, ignore_index=True),
        pd.concat(player_seasons, ignore_index=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2025)
    args = parser.parse_args()
    conn = get_conn()
    features = build_player_season_features(conn)
    team, share, availability, player_seasons = build_rows(conn, features)
    conn.close()
    selected = lambda frame: frame[frame["test_season"].le(args.cutoff)].copy()
    team_fit, share_fit, availability_fit = map(selected, (team, share, availability))
    residual_path = Path(BACKTEST_DIR) / "residuals_rolling.parquet"
    player_residuals = pd.read_parquet(residual_path) if residual_path.exists() else pd.DataFrame()
    if not player_residuals.empty:
        player_residuals = player_residuals[
            player_residuals["test_season"].le(args.cutoff)]
    manifest = fit_uncertainty_manifest(
        team_fit,
        share_fit,
        availability_fit,
        training_cutoff=args.cutoff,
        player_residuals=player_residuals,
    )
    path = write_uncertainty_artifacts(
        manifest, team, share, availability, player_season_rows=player_seasons)
    print(f"Wrote {path}")
    print(
        f"team rows={len(team)} share rows={len(share)} "
        f"availability rows={len(availability)} player-season rows={len(player_seasons)}"
    )
    print(f"artifact_hash={manifest['artifact_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
