"""Validation for assistant tool calls.

Everything the model emits is untrusted input. Tool arguments are validated
against this module before dispatch, and validation failures become typed
result dictionaries rather than exceptions so a bad tool call degrades the
answer instead of failing the request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from src.app.persistence.models import League

MIN_WEEK = 1
MAX_WEEK = 25
MAX_LEAGUE_ID_CHARS = 64
MAX_PLAYER_ID_CHARS = 32
MAX_PLAYERS_PER_SIDE = 12
MAX_PICKS_PER_SIDE = 12
MAX_ROSTER_ID = 64

HORIZONS = frozenset({"weekly", "ros", "dynasty"})
OPPONENT_MODES = frozenset({"current", "optimized"})

LEAGUE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
PLAYER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")

#: Tool name -> the only argument keys that tool accepts.
TOOL_ARGUMENT_KEYS: dict[str, frozenset[str]] = {
    "recommend_lineup": frozenset({"league_id", "week", "opponent_mode"}),
    "recommend_waivers": frozenset({"league_id", "week", "budget"}),
    "evaluate_trade": frozenset({"league_id", "side_a", "side_b", "horizon", "week"}),
    "get_injury_evidence": frozenset({"player_id", "week"}),
    "get_league_context": frozenset({"league_id", "week"}),
}

ALLOWED_TOOLS = frozenset(TOOL_ARGUMENT_KEYS)


class ToolArgumentError(ValueError):
    """Raised internally when a tool argument fails validation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ValidatedToolCall:
    name: str
    arguments: dict[str, Any]


def tool_error(name: str, code: str, message: str) -> dict[str, Any]:
    """The typed error shape returned to the model in place of tool output."""
    return {"error": {"tool": name, "code": code, "message": message}}


def validate_week(value: Any, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    try:
        week = int(value)
    except (TypeError, ValueError):
        raise ToolArgumentError("invalid_week", "week must be an integer") from None
    if not MIN_WEEK <= week <= MAX_WEEK:
        raise ToolArgumentError("invalid_week", f"week must be between {MIN_WEEK} and {MAX_WEEK}")
    return week


def validate_player_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ToolArgumentError("invalid_player_id", "player_id must be a non-empty string")
    if len(value) > MAX_PLAYER_ID_CHARS or not PLAYER_ID_PATTERN.match(value):
        raise ToolArgumentError("invalid_player_id", "player_id is not a recognized identifier")
    return value


def validate_horizon(value: Any, *, default: str = "ros") -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value not in HORIZONS:
        raise ToolArgumentError(
            "invalid_horizon", f"horizon must be one of {sorted(HORIZONS)}"
        )
    return value


def validate_opponent_mode(value: Any, *, default: str = "current") -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value not in OPPONENT_MODES:
        raise ToolArgumentError(
            "invalid_opponent_mode", f"opponent_mode must be one of {sorted(OPPONENT_MODES)}"
        )
    return value


def validate_league_id(session: Session, value: Any) -> str:
    """Shape-check a league id and confirm the league actually exists."""
    if not isinstance(value, str) or not value:
        raise ToolArgumentError("invalid_league_id", "league_id must be a non-empty string")
    if len(value) > MAX_LEAGUE_ID_CHARS or not LEAGUE_ID_PATTERN.match(value):
        raise ToolArgumentError("invalid_league_id", "league_id is not a recognized identifier")
    exists = session.query(League.id).filter(League.league_id == value).first()
    if exists is None:
        raise ToolArgumentError("unknown_league", "league_id does not match a known league")
    return value


def _validate_trade_side(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ToolArgumentError("invalid_trade_side", f"{label} must be an object")
    unknown = set(raw) - {"roster_id", "player_ids", "pick_assets"}
    if unknown:
        raise ToolArgumentError(
            "unexpected_argument", f"{label} has unsupported keys: {sorted(unknown)}"
        )
    try:
        roster_id = int(raw["roster_id"])
    except (KeyError, TypeError, ValueError):
        raise ToolArgumentError("invalid_trade_side", f"{label}.roster_id must be an integer") from None
    if not 1 <= roster_id <= MAX_ROSTER_ID:
        raise ToolArgumentError("invalid_trade_side", f"{label}.roster_id is out of range")

    player_ids_raw = raw.get("player_ids") or []
    if not isinstance(player_ids_raw, list) or len(player_ids_raw) > MAX_PLAYERS_PER_SIDE:
        raise ToolArgumentError(
            "invalid_trade_side", f"{label}.player_ids must be a list of at most {MAX_PLAYERS_PER_SIDE}"
        )
    player_ids = [validate_player_id(item) for item in player_ids_raw]

    picks_raw = raw.get("pick_assets") or []
    if not isinstance(picks_raw, list) or len(picks_raw) > MAX_PICKS_PER_SIDE:
        raise ToolArgumentError(
            "invalid_trade_side", f"{label}.pick_assets must be a list of at most {MAX_PICKS_PER_SIDE}"
        )
    picks: list[dict[str, Any]] = []
    for pick in picks_raw:
        if not isinstance(pick, dict):
            raise ToolArgumentError("invalid_pick_asset", f"{label}.pick_assets entries must be objects")
        unknown_pick = set(pick) - {"season", "round", "original_roster_id"}
        if unknown_pick:
            # A model-supplied ``value`` is exactly what this rejects.
            raise ToolArgumentError(
                "unexpected_argument",
                f"{label}.pick_assets has unsupported keys: {sorted(unknown_pick)}",
            )
        try:
            season = int(pick["season"])
            rnd = int(pick["round"])
        except (KeyError, TypeError, ValueError):
            raise ToolArgumentError(
                "invalid_pick_asset", f"{label}.pick_assets requires integer season and round"
            ) from None
        if not 2000 <= season <= 2100 or not 1 <= rnd <= 10:
            raise ToolArgumentError("invalid_pick_asset", f"{label}.pick_assets values are out of range")
        cleaned: dict[str, Any] = {"season": season, "round": rnd}
        if pick.get("original_roster_id") is not None:
            original = int(pick["original_roster_id"])
            if not 1 <= original <= MAX_ROSTER_ID:
                raise ToolArgumentError(
                    "invalid_pick_asset", f"{label}.pick_assets original_roster_id is out of range"
                )
            cleaned["original_roster_id"] = original
        picks.append(cleaned)

    return {"roster_id": roster_id, "player_ids": player_ids, "pick_assets": picks}


def validate_tool_call(
    session: Session,
    name: str,
    raw_arguments: Any,
    *,
    authorized_league_id: str | None,
    default_week: int = 1,
) -> ValidatedToolCall:
    """Validate one model-proposed tool call.

    ``authorized_league_id`` always wins: the model cannot redirect a tool at
    a league the request was not scoped to, even if it supplies one.
    """
    if name not in ALLOWED_TOOLS:
        raise ToolArgumentError("unknown_tool", f"unknown tool: {name}")
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ToolArgumentError("invalid_arguments", "tool arguments must be an object")

    unknown = set(raw_arguments) - TOOL_ARGUMENT_KEYS[name]
    if unknown:
        raise ToolArgumentError("unexpected_argument", f"unsupported argument keys: {sorted(unknown)}")

    args: dict[str, Any] = {}

    if name in {"recommend_lineup", "recommend_waivers", "evaluate_trade", "get_league_context"}:
        if authorized_league_id is None:
            raise ToolArgumentError(
                "missing_league_scope", "this request is not scoped to a league"
            )
        # Server-side value only; any model-supplied league_id is discarded.
        args["league_id"] = validate_league_id(session, authorized_league_id)

    if name == "recommend_lineup":
        args["week"] = validate_week(raw_arguments.get("week"), default=default_week)
        args["opponent_mode"] = validate_opponent_mode(raw_arguments.get("opponent_mode"))
    elif name == "recommend_waivers":
        args["week"] = validate_week(raw_arguments.get("week"), default=default_week)
    elif name == "evaluate_trade":
        args["side_a"] = _validate_trade_side(raw_arguments.get("side_a"), "side_a")
        args["side_b"] = _validate_trade_side(raw_arguments.get("side_b"), "side_b")
        args["horizon"] = validate_horizon(raw_arguments.get("horizon"))
    elif name == "get_injury_evidence":
        args["player_id"] = validate_player_id(raw_arguments.get("player_id"))

    return ValidatedToolCall(name=name, arguments=args)
