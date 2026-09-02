"""Map weekly-v2 component stats to league scoring draw keys."""

from __future__ import annotations

from typing import Any

#: Weekly model column -> ``score_stat_draw`` stat name (see ``LINEAR_KEY_MAP``).
WEEKLY_TO_DRAW_STAT: dict[str, str] = {
    "attempts": "pass_attempts",
    "completions": "pass_completions",
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "interceptions": "pass_ints",
    "carries": "rush_attempts",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
    # First-down components for PPFD (joint mixture path; legacy means may omit).
    "passing_first_downs": "pass_first_downs",
    "rushing_first_downs": "rush_first_downs",
    "receiving_first_downs": "rec_first_downs",
}


def weekly_row_to_stat_draw(record: dict[str, Any]) -> dict[str, float]:
    """Convert a weekly projection row into a single stat-level draw."""
    draw: dict[str, float] = {}
    for src, dst in WEEKLY_TO_DRAW_STAT.items():
        value = record.get(src)
        if value is None:
            continue
        draw[dst] = float(value)
    return draw
