#!/usr/bin/env python3
"""Decompose weekly-v2 under-dispersion by transformation stage and segment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.projection.weekly.config.paths import OUTPUTS_DIR
from src.projection.weekly.evaluate.dispersion_diagnostics import write_dispersion_diagnostics
from src.projection.weekly.evaluate.harness import (
    PreseasonEvalConfig,
    load_panel_for_eval,
    run_preseason_backtest,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=None, help="Existing OOF parquet")
    parser.add_argument("--namespace", type=str, default="dispersion_diagnose")
    parser.add_argument("--panel", type=Path, default=None)
    args = parser.parse_args(argv)

    exp_dir = OUTPUTS_DIR / "experiments" / args.namespace
    exp_dir.mkdir(parents=True, exist_ok=True)

    code_revision = "unknown"
    try:
        code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    if args.oof and args.oof.exists():
        import polars as pl

        oof = pl.read_parquet(args.oof)
        panel_hash = None
    else:
        panel = load_panel_for_eval(args.panel)
        panel_path = args.panel or Path("data/processed/player_week_panel.parquet")
        panel_hash = sha256_file(panel_path) if panel_path.exists() else None
        config = PreseasonEvalConfig(outer_start=2022, outer_end=2025)
        result = run_preseason_backtest(panel, config=config)
        oof = result["oof"]
        if oof is None:
            print("ERROR: no OOF rows produced", file=sys.stderr)
            return 2

    out = exp_dir / "dispersion_diagnostics.json"
    write_dispersion_diagnostics(
        oof,
        out,
        extra_metadata={
            "code_revision": code_revision,
            "panel_hash": panel_hash,
            "oof_source": str(args.oof) if args.oof else "fresh_backtest",
        },
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
