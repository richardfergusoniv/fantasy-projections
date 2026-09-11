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

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES_DIR = REPO_ROOT / "data" / "props" / "fixtures" / "providers"

LIVE_CAPABLE = frozenset({"draftkings", "fanduel"})


@dataclass(frozen=True)
class IngestResult:
    season: int
    week: int
    snapshots: tuple[ProviderSnapshot, ...]
    written: tuple[str, ...]
    success_count: int
    failure_count: int
    mode: str = "fixture"

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "week": self.week,
            "mode": self.mode,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "written": list(self.written),
            "sources": [
                {
                    "source": s.source,
                    "success": s.success,
                    "error": s.error,
                    "quote_count": len(s.quotes),
                    "live": bool((s.metadata or {}).get("live")),
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
    mode: str = "fixture",
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
        mode=mode,
    )


def parse_provider_names(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return ["draftkings", "fanduel"]
    if isinstance(raw, str):
        names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip().lower() for part in raw if str(part).strip()]
    return names or ["draftkings", "fanduel"]


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


def build_providers(
    *,
    mode: str = "live",
    provider_names: str | Sequence[str] | None = None,
    fixtures_dir: Path | None = None,
) -> list[PropProvider]:
    """Build provider adapters for weekly props ingest.

    ``mode=live`` uses real DraftKings + FanDuel HTTP fetch for named live-capable
    books. Non-live names in the list are skipped (BettingPros/OddsChecker remain
    fixture-only until implemented). ``mode=fixture`` loads local JSON fixtures.
    """
    names = parse_provider_names(provider_names)
    normalized = (mode or "live").strip().lower()
    fixtures = fixtures_dir or DEFAULT_FIXTURES_DIR

    if normalized in {"fixture", "fixtures", "offline"}:
        available = default_fixture_providers(fixtures)
        by_name = {p.name: p for p in available}
        return [by_name[name] for name in names if name in by_name]

    from src.ingest.props.providers import LiveDraftKingsProvider, LiveFanDuelProvider
    from src.ingest.props.providers.base import live_fetch_stub
    from src.ingest.props.contracts import ProviderSnapshot
    from datetime import datetime as _dt

    class _DisabledProvider:
        def __init__(self, name: str, error: str) -> None:
            self.name = name
            self._error = error

        def fetch(
            self,
            *,
            season: int,
            week: int,
            now: _dt | None = None,
        ) -> ProviderSnapshot:
            del now
            return live_fetch_stub(
                self.name, season=season, week=week, error=self._error
            )

    live_map = {
        "draftkings": LiveDraftKingsProvider,
        "fanduel": LiveFanDuelProvider,
    }
    providers: list[PropProvider] = []
    for name in names:
        if name in live_map:
            providers.append(live_map[name]())  # type: ignore[arg-type]
        else:
            providers.append(
                _DisabledProvider(  # type: ignore[arg-type]
                    name, "live_provider_not_implemented"
                )
            )
    return providers
