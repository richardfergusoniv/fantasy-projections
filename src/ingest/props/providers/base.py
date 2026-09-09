"""Shared provider base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import json

from src.ingest.props.contracts import ProviderSnapshot
from src.ingest.props.normalize import quotes_from_legacy_player_row


class PropProvider(ABC):
    name: str

    @abstractmethod
    def fetch(
        self,
        *,
        season: int,
        week: int,
        now: datetime | None = None,
    ) -> ProviderSnapshot:
        raise NotImplementedError


class FixturePropProvider(PropProvider):
    """Load a sanitized fixture JSON shaped like vegas_raw player dumps."""

    def __init__(self, name: str, fixture_path: Path) -> None:
        self.name = name
        self.fixture_path = fixture_path

    def fetch(
        self,
        *,
        season: int,
        week: int,
        now: datetime | None = None,
    ) -> ProviderSnapshot:
        clock = now or datetime.now(timezone.utc)
        try:
            payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — provider isolation
            return ProviderSnapshot(
                source=self.name,
                season=season,
                week=week,
                fetched_at=clock,
                urls=(),
                quotes=(),
                success=False,
                error=str(exc),
                raw_uri=self.fixture_path.as_uri(),
            )
        fetched_at = clock
        raw_fetched = payload.get("fetched_at")
        if raw_fetched:
            try:
                fetched_at = datetime.fromisoformat(str(raw_fetched).replace("Z", "+00:00"))
            except ValueError:
                fetched_at = clock
        urls = tuple(str(u) for u in (payload.get("urls") or []))
        quotes = []
        for row in payload.get("players") or []:
            if not isinstance(row, dict):
                continue
            quotes.extend(
                quotes_from_legacy_player_row(
                    source=self.name,
                    row=row,
                    fetched_at=fetched_at,
                    source_url=urls[0] if urls else None,
                    period="game",
                    default_sportsbook=self.name,
                )
            )
        return ProviderSnapshot(
            source=self.name,
            season=int(payload.get("season") or season),
            week=int(payload.get("week") or week),
            fetched_at=fetched_at,
            urls=urls,
            quotes=tuple(quotes),
            success=True,
            error=None,
            raw_uri=self.fixture_path.as_uri(),
            metadata={"fixture": True},
        )


def live_fetch_stub(
    name: str,
    *,
    season: int,
    week: int,
    error: str = "live_provider_disabled",
) -> ProviderSnapshot:
    now = datetime.now(timezone.utc)
    return ProviderSnapshot(
        source=name,
        season=season,
        week=week,
        fetched_at=now,
        urls=(),
        quotes=(),
        success=False,
        error=error,
        metadata={"live": True, "enabled": False},
    )
