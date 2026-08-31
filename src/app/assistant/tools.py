"""Typed assistant tools backed by decision services."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.app.availability.research import (
    ResearchUnavailable,
    build_provider,
    resolve_research_mode,
)
from src.app.availability.service import AvailabilityService
from src.app.config import get_settings
from src.app.decisions.dynasty import DynastyService
from src.app.decisions.services import LineupService, TradeService, WaiverService
from src.app.decisions.trades import TradeSide
from src.app.persistence.repositories import LeagueRepository


class AssistantTools:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.leagues = LeagueRepository(session)
        self.lineup = LineupService(session)
        self.waivers = WaiverService(session)
        self.trades = TradeService(session)
        self.dynasty = DynastyService(session)
        self.availability = AvailabilityService(session)

    def get_league_context(self, league_id: str) -> dict:
        league = self.leagues.get_league(league_id)
        rules = self.leagues.latest_rules(league_id)
        if league is None:
            return {"error": "league_not_found"}
        return {
            "league_id": league_id,
            "name": league.name,
            "type": league.league_type,
            "season": league.season,
            "contract_hash": rules.contract_hash if rules else None,
        }

    def get_matchup(self, league_id: str, week: int, opponent_mode: str = "current") -> dict:
        return self.lineup.recommend(league_id, week, opponent_mode=opponent_mode)

    def recommend_lineup(self, league_id: str, week: int, opponent_mode: str = "current") -> dict:
        return self.lineup.recommend(league_id, week, opponent_mode=opponent_mode)

    def recommend_waivers(self, league_id: str, week: int, budget: float = 100) -> dict:
        return self.waivers.recommend(league_id, week, remaining_faab=budget)

    def _side(self, raw: dict | None) -> TradeSide:
        raw = raw or {}
        return TradeSide(
            roster_id=int(raw.get("roster_id", 0)),
            player_ids=list(raw.get("player_ids") or []),
            pick_assets=list(raw.get("pick_assets") or []),
        )

    def evaluate_trade(self, league_id: str, sides: dict, horizon: str = "ros") -> dict:
        result = self.trades.evaluate(
            league_id,
            self._side(sides.get("side_a")),
            self._side(sides.get("side_b")),
            horizon=horizon,
        )
        return {
            "objective": result.objective,
            "fairness": result.fairness,
            "acceptance": result.acceptance,
        }

    def get_injury_evidence(self, player_id: str) -> dict:
        rows = self.availability.repo.evidence_for_player(player_id)
        return {
            "player_id": player_id,
            "evidence": [
                {"id": r.id, "claim": r.claim_json, "confidence": r.confidence, "source_url": r.source_url}
                for r in rows
            ],
        }

    def research_injury(self, player_id: str, as_of: str | None = None) -> dict:
        """Research a player's availability using the configured provider.

        The provider is resolved per call rather than pinned to fixtures at
        construction time: in production the fixture provider must not stand in
        for live research, and an unconfigured live mode is reported as
        unavailable instead of answering with synthetic evidence.
        """
        settings = get_settings()
        try:
            mode = resolve_research_mode(settings)
            provider = build_provider(settings, mode=mode)
            result = provider.research(player_id, as_of=as_of)
        except ResearchUnavailable as exc:
            return {
                "player_id": player_id,
                "available": False,
                "reason": str(exc),
                "claims": [],
                "citations": [],
                "model_id": None,
            }
        return {
            "player_id": player_id,
            "available": True,
            "mode": result.mode,
            "synthetic": result.synthetic,
            "claims": [claim.__dict__ for claim in result.claims],
            "citations": result.citations,
            "model_id": result.model_id,
        }
