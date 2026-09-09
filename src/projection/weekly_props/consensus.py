"""Week-scoped player-prop consensus with per-market coverage."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable

from src.ingest.props.contracts import (
    SCORING_MARKETS,
    ConsensusManifest,
    ConsensusMarket,
    MarketCoverage,
    NormalizedQuote,
    PlayerConsensus,
    ProviderSnapshot,
)
from src.projection.market_quotes import (
    conflicts_with_projection,
    is_prediction_market_book,
    odds_skewed,
    one_sided_longshot,
    robust_median,
)
from src.projection.weekly_props.config import (
    DEFAULT_WEEKLY_POLICY,
    GATE_VERSION,
    WeeklyPropsPolicy,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _with_reject(quote: NormalizedQuote, reason: str) -> NormalizedQuote:
    return replace(quote, reject_reason=reason)  # type: ignore[arg-type]


def evaluate_quote(
    quote: NormalizedQuote,
    *,
    policy: WeeklyPropsPolicy,
    now: datetime | None = None,
    projection_line: float | None = None,
    event_started: bool = False,
) -> NormalizedQuote:
    """Return quote unchanged or with reject_reason set."""
    clock = now or _utcnow()
    if quote.reject_reason:
        return quote
    if event_started or (
        quote.event_start is not None and quote.event_start <= clock
    ):
        return _with_reject(quote, "event_started")
    age_hours = (clock - quote.fetched_at).total_seconds() / 3600.0
    if age_hours > policy.max_snapshot_age_hours:
        return _with_reject(quote, "stale_snapshot")
    if quote.kind == "projection":
        # Projection quotes are retained for audit but never set scoring means.
        return quote
    if is_prediction_market_book(quote.sportsbook):
        return _with_reject(quote, "prediction_market_only")
    if one_sided_longshot(
        quote.over_odds,
        quote.under_odds,
        threshold=policy.quote.one_sided_longshot_threshold,
    ):
        return _with_reject(quote, "one_sided_longshot")
    if policy.require_two_sided and (
        quote.over_odds is None or quote.under_odds is None
    ):
        if not policy.allow_odds_less_exception:
            return _with_reject(quote, "odds_less_alt")
    if odds_skewed(
        quote.over_odds,
        quote.under_odds,
        gap=policy.quote.skewed_odds_gap,
    ):
        return _with_reject(quote, "skewed_odds")
    if projection_line is not None and conflicts_with_projection(
        quote.line, projection_line, policy=policy.quote
    ):
        return _with_reject(quote, "projection_conflict")
    return quote


def _dedupe_key(quote: NormalizedQuote) -> tuple:
    return (
        quote.sportsbook.strip().lower(),
        quote.market,
        quote.player_id or quote.player_name_raw.strip().lower(),
        round(quote.line, 3),
        None if quote.over_odds is None else round(quote.over_odds, 1),
        None if quote.under_odds is None else round(quote.under_odds, 1),
    )


def dedupe_quotes(quotes: Iterable[NormalizedQuote]) -> list[NormalizedQuote]:
    seen: set[tuple] = set()
    out: list[NormalizedQuote] = []
    for quote in quotes:
        key = _dedupe_key(quote)
        if key in seen:
            out.append(_with_reject(quote, "duplicate"))
            continue
        seen.add(key)
        out.append(quote)
    return out


def _position_sanity(
    quote: NormalizedQuote,
    *,
    position: str | None,
    policy: WeeklyPropsPolicy,
) -> NormalizedQuote:
    if quote.reject_reason or position is None:
        return quote
    caps = policy.sanity_caps.get(position.upper(), {})
    cap = caps.get(quote.market)
    if cap is not None and quote.line > cap:
        return _with_reject(quote, "sanity_bound")
    return quote


def consensus_for_market(
    quotes: list[NormalizedQuote],
    *,
    market: str,
    policy: WeeklyPropsPolicy,
    position: str | None = None,
) -> ConsensusMarket:
    projection_lines = [
        q.line for q in quotes if q.kind == "projection" and q.reject_reason is None
    ]
    projection_ref = (
        robust_median(projection_lines, policy=policy.quote) if projection_lines else None
    )
    evaluated: list[NormalizedQuote] = []
    for quote in quotes:
        current = evaluate_quote(
            quote, policy=policy, projection_line=projection_ref
        )
        current = _position_sanity(current, position=position, policy=policy)
        evaluated.append(current)

    # Apply cross-book robust median filter as soft reject.
    provisionally_accepted = [
        q for q in evaluated if q.reject_reason is None and q.kind == "book"
    ]
    if len(provisionally_accepted) >= 3:
        lines = [q.line for q in provisionally_accepted]
        center = robust_median(lines, policy=policy.quote)
        tolerance = max(
            abs(center) * policy.quote.robust_median_rel,
            policy.quote.robust_median_abs_floor,
        )
        outlier_hashes = {
            q.raw_record_hash
            for q in provisionally_accepted
            if abs(q.line - center) > tolerance
        }
        evaluated = [
            _with_reject(q, "cross_book_outlier")
            if q.raw_record_hash in outlier_hashes and q.reject_reason is None
            else q
            for q in evaluated
        ]

    accepted_books = [
        q for q in evaluated if q.reject_reason is None and q.kind == "book"
    ]
    rejected = [q for q in evaluated if q.reject_reason is not None]
    book_names = {q.sportsbook.strip().lower() for q in accepted_books}
    if accepted_books:
        line = robust_median([q.line for q in accepted_books], policy=policy.quote)
        kind = "books"
        coverage_kind = "books"
    else:
        line = None
        kind = "none"
        coverage_kind = "none"
        # Projection-only must not set scoring means.
    reasons = tuple(sorted({str(q.reject_reason) for q in rejected if q.reject_reason}))
    coverage = MarketCoverage(
        coverage=coverage_kind,  # type: ignore[arg-type]
        book_count=len(book_names),
        accepted_quote_count=len(accepted_books),
        rejected_quote_count=len(rejected),
        reject_reasons=reasons,
    )
    return ConsensusMarket(
        market=market,
        line=line,
        kind=kind,  # type: ignore[arg-type]
        coverage=coverage,
        accepted_quotes=tuple(accepted_books),
        rejected_quotes=tuple(rejected),
    )


def player_scoring_class(
    markets: dict[str, ConsensusMarket],
    *,
    scoring_markets: tuple[str, ...] = SCORING_MARKETS,
    baseline_components: set[str] | None = None,
) -> str:
    """Classify player-level coverage for scoring.

    Considers the union of observed prop markets and baseline components that
    belong to ``scoring_markets``. Partial means some of those are book-backed
    and others must fall back to baseline.
    """
    relevant_names = set(scoring_markets) & (
        set(markets) | (baseline_components or set())
    )
    if not relevant_names and not markets:
        return "baseline_only"
    if not relevant_names:
        relevant_names = set(markets) & set(scoring_markets)
    relevant = [markets[m] for m in relevant_names if m in markets]
    missing = [m for m in relevant_names if m not in markets]
    book_backed = [
        m for m in relevant if m.kind == "books" and m.line is not None
    ]
    if not book_backed:
        if any(m.coverage.rejected_quote_count for m in relevant):
            return "rejected"
        return "baseline_only"
    if missing or len(book_backed) < len(relevant_names):
        return "market_partial"
    return "market_complete"


def build_player_consensus(
    *,
    player_key: str,
    quotes: list[NormalizedQuote],
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
    player_id: str | None = None,
    player_name: str | None = None,
    team: str | None = None,
    opponent: str | None = None,
    position: str | None = None,
    on_slate: bool = True,
) -> PlayerConsensus:
    quotes = dedupe_quotes(quotes)
    by_market: dict[str, list[NormalizedQuote]] = defaultdict(list)
    for quote in quotes:
        by_market[quote.market].append(quote)
    markets = {
        market: consensus_for_market(
            rows, market=market, policy=policy, position=position
        )
        for market, rows in sorted(by_market.items())
    }
    name = player_name or next((q.player_name_raw for q in quotes), player_key)
    resolved_id = player_id or next((q.player_id for q in quotes if q.player_id), None)
    resolved_team = team or next((q.team for q in quotes if q.team), None)
    resolved_opp = opponent or next((q.opponent for q in quotes if q.opponent), None)
    return PlayerConsensus(
        player_id=resolved_id,
        player_name=name,
        team=resolved_team,
        opponent=resolved_opp,
        position=position,
        markets=markets,
        scoring_class=player_scoring_class(markets, scoring_markets=policy.scoring_markets),  # type: ignore[arg-type]
        on_slate=on_slate,
    )


def semantic_input_hash(
    *,
    season: int,
    week: int,
    policy: WeeklyPropsPolicy,
    players: list[PlayerConsensus],
    baseline_run_id: str | None,
    status_overlay_id: str | None,
    slate_version: str | None,
) -> str:
    accepted: list[dict[str, Any]] = []
    for player in sorted(
        players, key=lambda p: (p.player_id or "", p.player_name.lower())
    ):
        for market_name, market in sorted(player.markets.items()):
            if market.kind != "books" or market.line is None:
                continue
            for quote in market.accepted_quotes:
                accepted.append(
                    {
                        "player_id": player.player_id,
                        "market": market_name,
                        "line": round(float(market.line), 4),
                        "sportsbook": quote.sportsbook.lower(),
                        "over_odds": quote.over_odds,
                        "under_odds": quote.under_odds,
                    }
                )
    payload = {
        "season": season,
        "week": week,
        "policy_version": policy.version,
        "gate_version": GATE_VERSION,
        "baseline_run_id": baseline_run_id,
        "status_overlay_id": status_overlay_id,
        "slate_version": slate_version,
        "accepted": accepted,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_consensus_manifest(
    *,
    season: int,
    week: int,
    snapshots: list[ProviderSnapshot],
    players: list[PlayerConsensus],
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
    baseline_run_id: str | None = None,
    status_overlay_id: str | None = None,
    slate_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConsensusManifest:
    input_hash = semantic_input_hash(
        season=season,
        week=week,
        policy=policy,
        players=players,
        baseline_run_id=baseline_run_id,
        status_overlay_id=status_overlay_id,
        slate_version=slate_version,
    )
    source_meta = [
        {
            "source": snap.source,
            "success": snap.success,
            "error": snap.error,
            "fetched_at": snap.fetched_at.isoformat(),
            "quote_count": len(snap.quotes),
            "raw_uri": snap.raw_uri,
        }
        for snap in snapshots
    ]
    return ConsensusManifest(
        season=season,
        week=week,
        policy_version=policy.version,
        gate_version=GATE_VERSION,
        built_at=_utcnow(),
        semantic_input_hash=input_hash,
        players=tuple(players),
        source_snapshots=tuple(source_meta),
        metadata=dict(metadata or {}),
    )
