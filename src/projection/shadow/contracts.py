"""Shadow research contracts and artifact paths."""
from __future__ import annotations

from pathlib import Path

from src.projection.contracts import OUTPUT_DIR

SHADOW_OUTPUT_DIR = Path(OUTPUT_DIR) / "shadow_opportunity_mean"
SHADOW_0A_GATE = "gate_0a_{season}.json"
SHADOW_0B_GATE = "gate_0b_{season}.json"

SHADOW_V1_RB_WR_DIR = Path(OUTPUT_DIR) / "shadow_v1_rb_wr"


def gate_0a_path(season: int) -> Path:
    return SHADOW_OUTPUT_DIR / SHADOW_0A_GATE.format(season=season)


def gate_0b_path(season: int) -> Path:
    return SHADOW_OUTPUT_DIR / SHADOW_0B_GATE.format(season=season)


def shadow_v1_rb_wr_path(*parts: str) -> Path:
    return SHADOW_V1_RB_WR_DIR.joinpath(*parts)
