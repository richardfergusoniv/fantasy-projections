"""V3 publish helpers."""
from __future__ import annotations

import json
from pathlib import Path

from src.projection.contracts import MODEL_V3_DIR


def write_v3_manifest(season: int, payload: dict) -> Path:
    out_dir = Path(MODEL_V3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_{season}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
