"""Immutable snapshot persistence for weekly prop ingest.

Production writes should target durable object storage. The local filesystem
layout is used for fixtures and regression artifacts only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.ingest.props.contracts import ProviderSnapshot

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "props" / "snapshots"


def snapshot_key(*, season: int, week: int, source: str, fetched_at: datetime) -> str:
    stamp = fetched_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"season={season}/week={week:02d}/source={source}/fetch={stamp}.json"


class SnapshotStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_LOCAL_ROOT

    def write(self, snapshot: ProviderSnapshot) -> Path:
        relative = snapshot_key(
            season=snapshot.season,
            week=snapshot.week,
            source=snapshot.source,
            fetched_at=snapshot.fetched_at,
        )
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        return path

    def latest(self, *, season: int, week: int, source: str) -> Path | None:
        directory = self.root / f"season={season}" / f"week={week:02d}" / f"source={source}"
        if not directory.is_dir():
            return None
        files = sorted(directory.glob("fetch=*.json"))
        return files[-1] if files else None
