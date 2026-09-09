"""Orchestrate weekly prop scrape → consensus → optional promote."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.ingest.props.contracts import ProviderSnapshot
from src.ingest.props.providers import PropProvider
from src.ingest.props.snapshot import SnapshotStore
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY, WeeklyPropsPolicy


@dataclass(frozen=True)
class IngestResult:
    season: int
    week: int
    snapshots: tuple[ProviderSnapshot, ...]
    written: tuple[str, ...]
    success_count: int
    failure_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "week": self.week,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "written": list(self.written),
            "sources": [
                {
                    "source": s.source,
                    "success": s.success,
                    "error": s.error,
                    "quote_count": len(s.quotes),
                }
                for s in self.snapshots
            ],
        }


def run_ingest(
    *,
    season: int,
    week: int,
    providers: Sequence[PropProvider],
    store: SnapshotStore | None = None,
    now: datetime | None = None,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> IngestResult:
    del policy  # reserved for future provider-level filtering
    clock = now or datetime.now(timezone.utc)
    store = store or SnapshotStore()
    snapshots: list[ProviderSnapshot] = []
    written: list[str] = []
    for provider in providers:
        snapshot = provider.fetch(season=season, week=week, now=clock)
        snapshots.append(snapshot)
        path = store.write(snapshot)
        written.append(str(path))
    successes = sum(1 for s in snapshots if s.success)
    return IngestResult(
        season=season,
        week=week,
        snapshots=tuple(snapshots),
        written=tuple(written),
        success_count=successes,
        failure_count=len(snapshots) - successes,
    )


def default_fixture_providers(fixtures_dir: Path) -> list[PropProvider]:
    from src.ingest.props.providers import (
        BettingProsProvider,
        DraftKingsProvider,
        FanDuelProvider,
        OddsCheckerProvider,
    )

    mapping = {
        "draftkings.json": DraftKingsProvider,
        "fanduel.json": FanDuelProvider,
        "bettingpros.json": BettingProsProvider,
        "oddschecker.json": OddsCheckerProvider,
    }
    providers: list[PropProvider] = []
    for name, cls in mapping.items():
        path = fixtures_dir / name
        if path.is_file():
            providers.append(cls(path))
    return providers
