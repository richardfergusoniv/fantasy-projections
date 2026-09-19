"""BettingPros weekly player-prop provider (fixture or live)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.ingest.props.contracts import ProviderSnapshot
from src.ingest.props.providers.base import FixturePropProvider, live_fetch_stub


class BettingProsProvider(FixturePropProvider):
    """Fixture-backed BettingPros provider (tests / offline).

    Fixture JSON must mirror live emission: each market carries a ``books`` map
    of real sportsbooks (e.g. Caesars, BetMGM). Blended top-level lines that
    would default to ``sportsbook=bettingpros`` — and any DK/FD / prediction-
    market / DFS names — are filtered out so ``--from-fixture`` cannot feed
    Role 1 the consensus-blend shape the live path rejects.
    """

    def __init__(self, fixture_path: Path) -> None:
        super().__init__("bettingpros", fixture_path)

    def fetch(
        self,
        *,
        season: int,
        week: int,
        now: datetime | None = None,
    ) -> ProviderSnapshot:
        snapshot = super().fetch(season=season, week=week, now=now)
        if not snapshot.success:
            return snapshot
        from src.ingest.props.providers.bettingpros_live import (
            is_eligible_role1_sportsbook,
        )

        kept = tuple(
            quote
            for quote in snapshot.quotes
            if is_eligible_role1_sportsbook(quote.sportsbook)
        )
        metadata = dict(snapshot.metadata or {})
        metadata["fixture"] = True
        metadata["emission"] = "per_book"
        metadata["filtered_ineligible"] = len(snapshot.quotes) - len(kept)
        if not kept:
            return ProviderSnapshot(
                source=snapshot.source,
                season=snapshot.season,
                week=snapshot.week,
                fetched_at=snapshot.fetched_at,
                urls=snapshot.urls,
                quotes=(),
                success=False,
                error=(
                    "no_eligible_quotes: BettingPros fixture must use per-book "
                    "`markets.*.books` (not blended sportsbook=bettingpros)"
                ),
                raw_uri=snapshot.raw_uri,
                metadata=metadata,
            )
        return ProviderSnapshot(
            source=snapshot.source,
            season=snapshot.season,
            week=snapshot.week,
            fetched_at=snapshot.fetched_at,
            urls=snapshot.urls,
            quotes=kept,
            success=True,
            error=None,
            raw_uri=snapshot.raw_uri,
            metadata=metadata,
        )


class LiveBettingProsProvider:
    """Live BettingPros weekly per-book O/U provider.

    Never emits ``book_id=0`` Consensus as an independent Role 1 book.
    """

    name = "bettingpros"

    def fetch(
        self,
        *,
        season: int,
        week: int,
        now: datetime | None = None,
    ) -> ProviderSnapshot:
        try:
            from src.ingest.props.providers.bettingpros_live import fetch_weekly_snapshot
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
