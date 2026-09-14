"""Typed contracts for weekly NFL player-prop ingest and consensus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

MarketName = Literal[
    "pass_yards",
    "pass_tds",
    "pass_attempts",
    "pass_completions",
    "pass_ints",
    "rush_yards",
    "rush_tds",
    "rush_attempts",
    "rec_yards",
    "rec_tds",
    "receptions",
    "targets",
]

MarketCoverageKind = Literal["books", "projection", "none"]
PlayerScoringClass = Literal[
    "market_complete",
    "market_partial",
    "baseline_only",
    "rejected",
]
QuoteRejectReason = Literal[
    "missing_line",
    "skewed_odds",
    "one_sided_longshot",
    "prediction_market_only",
    "projection_conflict",
    "stale_snapshot",
    "event_started",
    "duplicate",
    "sanity_bound",
    "cross_book_outlier",
    "unresolved_identity",
    "wrong_slate",
    "odds_less_alt",
]


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text)


SCORING_MARKETS: tuple[str, ...] = (
    "pass_yards",
    "pass_tds",
    "rush_yards",
    "rush_tds",
    "rec_yards",
    "rec_tds",
    "receptions",
)

PLAYER_MARKET_ALIASES: dict[str, str] = {
    "pass_yards": "pass_yards",
    "passing_yards": "pass_yards",
    "pass_tds": "pass_tds",
    "passing_tds": "pass_tds",
    "pass_attempts": "pass_attempts",
    "passing_attempts": "pass_attempts",
    "pass_completions": "pass_completions",
    "completions": "pass_completions",
    "pass_ints": "pass_ints",
    "interceptions": "pass_ints",
    "rush_yards": "rush_yards",
    "rushing_yards": "rush_yards",
    "rush_tds": "rush_tds",
    "rushing_tds": "rush_tds",
    "rush_attempts": "rush_attempts",
    "rushing_attempts": "rush_attempts",
    "carries": "rush_attempts",
    "rec_yards": "rec_yards",
    "receiving_yards": "rec_yards",
    "rec_tds": "rec_tds",
    "receiving_tds": "rec_tds",
    "receptions": "receptions",
    "targets": "targets",
}


@dataclass(frozen=True)
class NormalizedQuote:
    source: str
    sportsbook: str
    event_id: str | None
    game_id: str | None
    player_name_raw: str
    player_id: str | None
    team: str | None
    opponent: str | None
    market: str
    period: str
    line: float
    over_odds: float | None
    under_odds: float | None
    fetched_at: datetime
    event_start: datetime | None
    source_url: str | None
    raw_record_hash: str
    over_prob_novig: float | None = None
    under_prob_novig: float | None = None
    kind: Literal["book", "projection"] = "book"
    reject_reason: QuoteRejectReason | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fetched_at"] = self.fetched_at.isoformat()
        payload["event_start"] = (
            self.event_start.isoformat() if self.event_start is not None else None
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizedQuote:
        data = dict(payload)
        fetched = _parse_datetime(data.get("fetched_at"))
        if fetched is None:
            raise ValueError("quote is missing fetched_at")
        return cls(
            source=str(data.get("source") or ""),
            sportsbook=str(data.get("sportsbook") or ""),
            event_id=data.get("event_id"),
            game_id=data.get("game_id"),
            player_name_raw=str(data.get("player_name_raw") or ""),
            player_id=data.get("player_id"),
            team=data.get("team"),
            opponent=data.get("opponent"),
            market=str(data.get("market") or ""),
            period=str(data.get("period") or "full_game"),
            line=float(data["line"]),
            over_odds=data.get("over_odds"),
            under_odds=data.get("under_odds"),
            fetched_at=fetched,
            event_start=_parse_datetime(data.get("event_start")),
            source_url=data.get("source_url"),
            raw_record_hash=str(data.get("raw_record_hash") or ""),
            over_prob_novig=data.get("over_prob_novig"),
            under_prob_novig=data.get("under_prob_novig"),
            kind=data.get("kind") or "book",
            reject_reason=data.get("reject_reason"),
        )


@dataclass(frozen=True)
class ProviderSnapshot:
    source: str
    season: int
    week: int
    fetched_at: datetime
    urls: tuple[str, ...]
    quotes: tuple[NormalizedQuote, ...]
    success: bool
    error: str | None = None
    raw_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "season": self.season,
            "week": self.week,
            "fetched_at": self.fetched_at.isoformat(),
            "urls": list(self.urls),
            "quotes": [quote.to_dict() for quote in self.quotes],
            "success": self.success,
            "error": self.error,
            "raw_uri": self.raw_uri,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProviderSnapshot:
        data = dict(payload)
        fetched = _parse_datetime(data.get("fetched_at"))
        if fetched is None:
            raise ValueError("snapshot is missing fetched_at")
        quotes = tuple(
            NormalizedQuote.from_dict(quote) for quote in (data.get("quotes") or [])
        )
        urls = data.get("urls") or ()
        return cls(
            source=str(data.get("source") or ""),
            season=int(data["season"]),
            week=int(data["week"]),
            fetched_at=fetched,
            urls=tuple(str(url) for url in urls),
            quotes=quotes,
            success=bool(data.get("success")),
            error=data.get("error"),
            raw_uri=data.get("raw_uri"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MarketCoverage:
    coverage: MarketCoverageKind
    book_count: int
    accepted_quote_count: int
    rejected_quote_count: int
    reject_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "book_count": self.book_count,
            "accepted_quote_count": self.accepted_quote_count,
            "rejected_quote_count": self.rejected_quote_count,
            "reject_reasons": list(self.reject_reasons),
        }


@dataclass(frozen=True)
class ConsensusMarket:
    market: str
    line: float | None
    kind: MarketCoverageKind
    coverage: MarketCoverage
    accepted_quotes: tuple[NormalizedQuote, ...]
    rejected_quotes: tuple[NormalizedQuote, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "line": self.line,
            "kind": self.kind,
            "coverage": self.coverage.to_dict(),
            "accepted_quotes": [quote.to_dict() for quote in self.accepted_quotes],
            "rejected_quotes": [quote.to_dict() for quote in self.rejected_quotes],
        }


@dataclass(frozen=True)
class PlayerConsensus:
    player_id: str | None
    player_name: str
    team: str | None
    opponent: str | None
    position: str | None
    markets: dict[str, ConsensusMarket]
    scoring_class: PlayerScoringClass
    on_slate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "opponent": self.opponent,
            "position": self.position,
            "markets": {key: value.to_dict() for key, value in self.markets.items()},
            "scoring_class": self.scoring_class,
            "on_slate": self.on_slate,
        }


@dataclass(frozen=True)
class ConsensusManifest:
    season: int
    week: int
    policy_version: str
    gate_version: str
    built_at: datetime
    semantic_input_hash: str
    players: tuple[PlayerConsensus, ...]
    source_snapshots: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "week": self.week,
            "policy_version": self.policy_version,
            "gate_version": self.gate_version,
            "built_at": self.built_at.isoformat(),
            "semantic_input_hash": self.semantic_input_hash,
            "players": [player.to_dict() for player in self.players],
            "source_snapshots": list(self.source_snapshots),
            "metadata": dict(self.metadata),
        }
