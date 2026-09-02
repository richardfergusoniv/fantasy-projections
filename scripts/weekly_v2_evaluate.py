#!/usr/bin/env python3
"""Strict leave-one-season-out preseason evaluation and promotion report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.projection.weekly.config.paths import OUTPUTS_DIR
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.evaluate.harness import (
    PreseasonEvalConfig,
    load_panel_for_eval,
    run_preseason_backtest,
    write_preseason_backtest,
)
from src.projection.weekly.models.volume_config import VolumeModelConfig


def _load_volume_options(args: argparse.Namespace) -> dict:
    if args.volume_options_json:
        return json.loads(Path(args.volume_options_json).read_text(encoding="utf-8"))
    if args.tuning_selection:
        selection_path = Path(args.tuning_selection)
        if selection_path.exists():
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
            return payload.get("volume_options") or {}
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--scoring", default="half_ppr")
    parser.add_argument("--panel", type=Path, default=None, help="Panel parquet path")
    parser.add_argument(
        "--tuning-selection",
        type=Path,
        default=None,
        help="Explicit tuning selection JSON with volume_options",
    )
    parser.add_argument(
        "--volume-options-json",
        type=Path,
        default=None,
        help="Direct volume options JSON (overrides tuning selection)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output backtest JSON path (default: OUTPUTS_DIR/preseason_backtest.json)",
    )
    args = parser.parse_args(argv)

    panel_path = args.panel
    panel = load_panel_for_eval(panel_path)
    volume_options = _load_volume_options(args)
    config = PreseasonEvalConfig(
        panel_path=panel_path or Path("data/processed/player_week_panel.parquet"),
        outer_start=args.start,
        outer_end=args.end,
        scoring=args.scoring,
        volume_options=volume_options,
    )
    scoring = ScoringConfig.from_name(args.scoring)
    result = run_preseason_backtest(panel, config=config, scoring=scoring)
    out = args.output or (OUTPUTS_DIR / "preseason_backtest.json")
    write_preseason_backtest(result, out)
    print(out)
    if volume_options:
        print(f"volume_options={VolumeModelConfig.from_options(volume_options).fingerprint()}")
    return 0 if result["promotion"]["promote"] else 2


if __name__ == "__main__":
    sys.exit(main())
