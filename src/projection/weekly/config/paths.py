"""Shared paths and season defaults."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# weekly/config/paths.py -> repo root is parents[4]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(os.getenv("WEEKLY_V2_DATA_DIR", os.getenv("DATA_DIR", PROJECT_ROOT / "data")))
MODELS_DIR = Path(
    os.getenv("WEEKLY_V2_MODELS_DIR", PROJECT_ROOT / "output" / "weekly_v2" / "models")
)
OUTPUTS_DIR = Path(
    os.getenv("WEEKLY_V2_OUTPUTS_DIR", PROJECT_ROOT / "output" / "weekly_v2")
)

TRAIN_START_SEASON = int(os.getenv("TRAIN_START_SEASON", "2016"))
TRAIN_END_SEASON = int(os.getenv("TRAIN_END_SEASON", "2025"))
VALIDATE_SEASON = int(os.getenv("VALIDATE_SEASON", "2025"))

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def ensure_dirs() -> None:
    for path in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, DATA_DIR / "raw", DATA_DIR / "processed"):
        path.mkdir(parents=True, exist_ok=True)
