"""FanDuel weekly player-prop provider."""

from __future__ import annotations

from pathlib import Path

from src.ingest.props.providers.base import FixturePropProvider


class FanDuelProvider(FixturePropProvider):
    def __init__(self, fixture_path: Path) -> None:
        super().__init__("fanduel", fixture_path)
