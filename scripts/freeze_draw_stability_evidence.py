#!/usr/bin/env python3
"""Freeze intermediate v20k draw-stability evidence with hash manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.contracts import MODEL_V3_DIR
from src.projection.evaluation.evidence_freeze import freeze_draw_stability_evidence

DEFAULT_BOARD_HASH = "67f2c4b88ad370b15e2363d4f915e5ec915d1ea6280625df528edfbd75d41700"
DEFAULT_RUN_ID = "d494c516-f86a-4fc3-afdd-dd8635b72ec5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--freeze-id",
        default="draw_stability_intermediate_v20k_2026",
        help="Frozen bundle directory name under output/model_v3/frozen/",
    )
    parser.add_argument("--selected-board-hash", default=DEFAULT_BOARD_HASH)
    parser.add_argument("--canonical-projection-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--reference-draw-count", type=int, default=20000)
    parser.add_argument("--source-dir", type=Path, default=Path(MODEL_V3_DIR))
    args = parser.parse_args()

    manifest_path = freeze_draw_stability_evidence(
        freeze_id=args.freeze_id,
        season=args.season,
        source_dir=args.source_dir,
        selected_board_hash=args.selected_board_hash,
        canonical_projection_run_id=args.canonical_projection_run_id,
        reference_draw_count=args.reference_draw_count,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
