"""Shadow 0A evaluation against incumbent target opportunity predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.shadow.contracts import gate_0a_path
from src.projection.shadow.evaluate_0a import evaluate_shadow_0a_on_long_board
from src.projection.shadow.share_model import write_shadow_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Shadow 0A target shares")
    parser.add_argument("--long-board", type=str, required=True)
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    import pandas as pd

    board = pd.read_csv(args.long_board)
    metrics = evaluate_shadow_0a_on_long_board(board)
    payload = {
        "milestone": "shadow_0a",
        "season": args.season,
        "verdict": "hold",  # research branch until leakage-safe top-120 beats accuracy-first
        **metrics,
    }
    write_shadow_artifact(payload, gate_0a_path(args.season))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
