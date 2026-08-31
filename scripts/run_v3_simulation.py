"""Run the v3 season simulation from the shipped projections board."""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.inference.predict_v3 import project_season_v3
from src.projection.inference.simulate import SIMULATION_MODE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument(
        "--mode",
        default=SIMULATION_MODE,
        choices=("full", "interim"),
        help="full = generative (default). interim is retired; comparison only.",
    )
    parser.add_argument(
        "--projections",
        default=None,
        help="Path to long projections CSV (defaults to output/projections_<season>.csv)",
    )
    args = parser.parse_args()
    proj_path = args.projections or os.path.join(OUTPUT_DIR, f"projections_{args.season}.csv")
    projections = pd.read_csv(proj_path)
    manifest = project_season_v3(
        projections, args.season, n_draws=args.draws, mode=args.mode)
    print(f"v3 simulation complete -> {MODEL_V3_DIR}")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
