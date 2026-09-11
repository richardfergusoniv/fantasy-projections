"""Fail-fast guard for the production projections database.

``data/projections.db`` is often a missing or zero-byte Cloud placeholder.
Callers must never treat that as permission to skip team reconciliation.
"""
from __future__ import annotations

from pathlib import Path

from src.paths import configured_db_path

DEFAULT_DB = Path(configured_db_path())


class ProjectionsDbUnusable(RuntimeError):
    """Raised when projections.db is missing or a zero-byte placeholder."""


def projections_db_status(path: Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_DB
    exists = p.exists()
    size = int(p.stat().st_size) if exists and p.is_file() else 0
    usable = exists and p.is_file() and size > 0
    return {
        "path": str(p),
        "exists": exists,
        "size_bytes": size,
        "usable": usable,
        "placeholder": exists and size == 0,
    }


def require_usable_projections_db(path: Path | None = None) -> Path:
    """Return the DB path or raise. Never returns on a missing/empty file."""
    status = projections_db_status(path)
    if status["usable"]:
        return Path(status["path"])
    raise ProjectionsDbUnusable(
        "projections.db is missing or zero bytes (Cloud placeholder). "
        "Do not open it and do not silently skip team reconciliation. "
        "Supply a validated portable QB reconciliation fixture instead. "
        f"status={status}"
    )
