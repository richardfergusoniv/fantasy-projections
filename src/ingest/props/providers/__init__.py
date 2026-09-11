"""Provider adapters for weekly NFL player props."""

from src.ingest.props.providers.base import FixturePropProvider, PropProvider, live_fetch_stub
from src.ingest.props.providers.bettingpros import BettingProsProvider
from src.ingest.props.providers.draftkings import DraftKingsProvider, LiveDraftKingsProvider
from src.ingest.props.providers.fanduel import FanDuelProvider, LiveFanDuelProvider
from src.ingest.props.providers.oddschecker import OddsCheckerProvider

__all__ = [
    "PropProvider",
    "FixturePropProvider",
    "DraftKingsProvider",
    "LiveDraftKingsProvider",
    "FanDuelProvider",
    "LiveFanDuelProvider",
    "BettingProsProvider",
    "OddsCheckerProvider",
    "live_fetch_stub",
]
