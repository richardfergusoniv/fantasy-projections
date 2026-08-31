"""Schedule guards for release and weekly-close jobs."""

from __future__ import annotations


def weekly_close_allowed(nfl_state: dict) -> tuple[bool, str | None]:
    """Postpone weekly close when the NFL week still has unfinished games."""
    if nfl_state.get("week_has_completed") is False:
        return False, "scheduled_games_not_final"
    if nfl_state.get("season_type") == "regular" and nfl_state.get("display_week") != nfl_state.get("week"):
        return False, "display_week_mismatch"
    return True, None
