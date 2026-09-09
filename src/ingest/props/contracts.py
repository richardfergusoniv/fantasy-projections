"""Typed contracts for weekly NFL player-prop ingest and consensus."""

from __future__ import annotations

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
