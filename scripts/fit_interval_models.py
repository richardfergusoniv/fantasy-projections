"""Fit conditional interval models from rolling residuals."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.backtest import rolling_residual_rows
from src.projection.contracts import INTERVAL_MODELS_DIR
from src.projection.data_prep import get_conn
from src.projection.evaluation.interval_models import fit_conditional_intervals
from src.projection.features import build_player_season_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=INTERVAL_MODELS_DIR)
    args = parser.parse_args()
    conn = get_conn()
    feat = build_player_season_features(conn)
    conn.close()
    residuals = rolling_residual_rows(feat)
    manifest = fit_conditional_intervals(residuals, out_dir=args.out)
    print(f"Fitted {len(manifest.get('cells', {}))} interval cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
