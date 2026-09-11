"""DraftKings weekly player-prop provider (fixture or live)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.ingest.props.contracts import ProviderSnapshot
from src.ingest.props.providers.base import FixturePropProvider, live_fetch_stub


class DraftKingsProvider(FixturePropProvider):
    """Fixture-backed DraftKings provider (tests / offline)."""

    def __init__(self, fixture_path: Path) -> None:
        super().__init__("draftkings", fixture_path)


class LiveDraftKingsProvider:
    """Live DraftKings sportscontent O/U provider."""

    name = "draftkings"

    def fetch(
        self,
        *,
        season: int,
        week: int,
        now: datetime | None = None,
    ) -> ProviderSnapshot:
        try:
            from src.ingest.props.providers.draftkings_live import fetch_weekly_snapshot
        except Exception as exc:  # noqa: BLE001
            return live_fetch_stub(
                self.name,
                season=season,
                week=week,
                error=f"live_import_failed: {exc}",
            )
        try:
            return fetch_weekly_snapshot(season=season, week=week, now=now)
        except Exception as exc:  # noqa: BLE001 — provider isolation
            return live_fetch_stub(
                self.name,
                season=season,
                week=week,
                error=str(exc),
            )
