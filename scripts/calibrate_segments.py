"""Run one-dimensional segment calibration for recentered distributions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.contracts import OUTPUT_DIR
from src.projection.evaluation.calibration_report import write_calibration_artifacts

DEFAULT_EVAL_DIR = Path(OUTPUT_DIR) / "evaluation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-parquet",
        type=Path,
        required=True,
        help="Player-level scored frame with actual_points and pred_p10/p50/p90",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_DIR)
    args = parser.parse_args()
    frame = pd.read_parquet(args.scored_parquet)
    result = write_calibration_artifacts(
        frame,
        args.out_dir,
        season=args.season,
        run_id=args.run_id,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
