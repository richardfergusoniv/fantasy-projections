"""Assistant gateway routes user prompts to typed tools."""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from src.app.assistant.openai_service import OpenAIAssistantService
from src.app.assistant.service import AssistantService
from src.app.assistant.tools import AssistantTools
from src.app.assistant.validation import (
    ToolArgumentError,
    tool_error,
    validate_league_id,
    validate_player_id,
    validate_week,
)
from src.app.config import get_settings
from src.app.logging import get_logger
from src.app.persistence.models import AppUser

logger = get_logger(__name__)

#: Free text is not a trade. Without a structured proposal the deterministic
#: path refuses rather than scoring an invented one.
TRADE_NOT_SPECIFIED = (
    "A trade has to be built from real assets. Open the Trade Lab (or call "
    "/leagues/{league_id}/trades/evaluate) with both sides, and the evaluation "
    "will come back with objective impact, fairness, and acceptance."
)
PLAYER_NOT_SPECIFIED = (
    "Name the player as `player: <player_id>` so the evidence lookup targets "
    "the right person; there is no default player."
)


class AssistantGateway:
    TOOL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\b(lineup|start/sit|starters?|start)\b", re.I), "recommend_lineup"),
        (re.compile(r"\b(waiver|faab|pickup|add)\b", re.I), "recommend_waivers"),
        (re.compile(r"\b(trade|offer|swap)\b", re.I), "evaluate_trade"),
        (re.compile(r"\b(injury|hurt|questionable)\b", re.I), "get_injury_evidence"),
        (re.compile(r"\b(dynasty|rebuild|contender)\b", re.I), "get_league_context"),
    ]

    def __init__(self, session: Session) -> None:
        self.session = session
        self.service = AssistantService(session)
        self.tools = AssistantTools(session)
        self.settings = get_settings()

    def respond(self, user: AppUser, message: str, *, league_id: str | None = None, week: int = 1) -> dict:
        if self.settings.openai_api_key:
            try:
                return OpenAIAssistantService(self.session).respond(
                    user,
                    message,
                    league_id=league_id,
                    week=week,
                )
            except Exception as exc:  # noqa: BLE001
                # Any upstream problem (budget, timeout, transport) degrades to
                # the deterministic path rather than surfacing provider detail.
                logger.warning("assistant_openai_degraded", exception_type=type(exc).__name__)

        tool_name = self._select_tool(message)
        if tool_name and league_id:
            tool_result = self._invoke(tool_name, league_id, week, message)
            if tool_result is not None:
                return self.service.respond_with_tool_result(
                    user,
                    message,
                    tool_result,
                    tools_called=[tool_name],
                )
        return self.service.respond(user, message, league_id=league_id)

    def _select_tool(self, message: str) -> str | None:
        for pattern, tool_name in self.TOOL_PATTERNS:
            if pattern.search(message):
                return tool_name
        return None

    def _invoke(self, tool_name: str, league_id: str, week: int, message: str) -> dict | None:
        """Dispatch the deterministic tool for a matched intent.

        ``league_id`` is the server-side scope for the request; nothing parsed
        out of the user message is allowed to replace it.
        """
        try:
            scoped_league_id = validate_league_id(self.session, league_id)
            scoped_week = validate_week(week, default=1)
        except ToolArgumentError as exc:
            logger.warning("assistant_scope_rejected", tool=tool_name, code=exc.code)
            return tool_error(tool_name, exc.code, str(exc))

        try:
            if tool_name == "recommend_lineup":
                return self.tools.recommend_lineup(scoped_league_id, scoped_week)
            if tool_name == "recommend_waivers":
                return self.tools.recommend_waivers(scoped_league_id, scoped_week)
            if tool_name == "get_injury_evidence":
                player_id = self._player_id_from(message)
                if player_id is None:
                    return tool_error(
                        tool_name, "player_not_specified", PLAYER_NOT_SPECIFIED
                    )
                return self.tools.get_injury_evidence(player_id)
            if tool_name == "evaluate_trade":
                # Scoring a hard-coded trade here would return a confident
                # answer about a trade the user never proposed.
                return tool_error(
                    tool_name,
                    "trade_not_specified",
                    TRADE_NOT_SPECIFIED.format(league_id=scoped_league_id),
                )
            if tool_name == "get_league_context":
                return self.tools.get_league_context(scoped_league_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "assistant_tool_failed", tool=tool_name, exception_type=type(exc).__name__
            )
            return tool_error(tool_name, "tool_execution_failed", "Tool execution failed.")
        return None

    def _player_id_from(self, message: str) -> str | None:
        """Extract an explicit `player: <id>` reference, or nothing.

        Falling back to a fixed player would answer an injury question about
        somebody the user never asked about, which is worse than declining.
        """
        match = re.search(r"player[:\s]+(\S{1,64})", message, re.I)
        if not match:
            return None
        try:
            return validate_player_id(match.group(1))
        except ToolArgumentError:
            logger.warning("assistant_player_id_rejected")
            return None
