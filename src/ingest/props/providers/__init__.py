"""Provider adapters for weekly NFL player props."""

from src.ingest.props.providers.base import FixturePropProvider, PropProvider, live_fetch_stub
from src.ingest.props.providers.bettingpros import BettingProsProvider
from src.ingest.props.providers.draftkings import DraftKingsProvider
from src.ingest.props.providers.fanduel import FanDuelProvider
from src.ingest.props.providers.oddschecker import OddsCheckerProvider

__all__ = [
    "PropProvider",
    "FixturePropProvider",
    "DraftKingsProvider",
    "FanDuelProvider",
    "BettingProsProvider",
    "OddsCheckerProvider",
    "live_fetch_stub",
]
