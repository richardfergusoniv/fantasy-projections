#!/usr/bin/env python3
"""Fit prediction calibration from strict preseason out-of-fold rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.projection.weekly.config.paths import MODELS_DIR, OUTPUTS_DIR
from src.projection.weekly.models.calibration import fit_position_calibration, save_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(OUTPUTS_DIR / "preseason_oof.parquet"))
    parser.add_argument("--output", default=str(MODELS_DIR / "calibration.json"))
    parser.add_argument("--alpha", type=float, default=0.20)
    args = parser.parse_args()
    rows = pl.read_parquet(args.input)
    payload = fit_position_calibration(rows, alpha=args.alpha)
    if not payload["positions"]:
        raise RuntimeError("No position had enough out-of-fold rows to calibrate")
    out = save_calibration(payload, Path(args.output))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
