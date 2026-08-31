"""Model save/load registry."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.projection.weekly.config.paths import MODELS_DIR, ensure_dirs


def model_path(name: str) -> Path:
    ensure_dirs()
    return MODELS_DIR / f"{name}.joblib"


def meta_path(name: str) -> Path:
    ensure_dirs()
    return MODELS_DIR / f"{name}.meta.json"


def save_model(name: str, obj: Any, meta: dict | None = None) -> Path:
    path = model_path(name)
    joblib.dump(obj, path)
    if meta is not None:
        payload = dict(meta)
        payload.setdefault("schema_version", 1)
        payload.setdefault("created_at_utc", datetime.now(UTC).isoformat())
        meta_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_model(name: str) -> Any:
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run scripts/train.py first.")
    return joblib.load(path)


def load_meta(name: str) -> dict:
    path = meta_path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
