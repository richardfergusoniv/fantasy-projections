"""Opt-in, read-only live Sleeper smoke check.

Skipped unless both `LIVE_SLEEPER_SMOKE=1` and `SLEEPER_USERNAME` are set, so it
can sit in a pipeline without ever becoming a network-dependent required test.
It performs GET requests only and prints counts, never payloads, emails, tokens,
or roster contents.

    LIVE_SLEEPER_SMOKE=1 SLEEPER_USERNAME=<name> uv run python scripts/live_sleeper_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.league.sleeper.client import (  # noqa: E402
    READ_ONLY_METHOD,
    SleeperClient,
    SleeperError,
)

OPT_IN_ENV = "LIVE_SLEEPER_SMOKE"
USERNAME_ENV = "SLEEPER_USERNAME"
SEASON_ENV = "SLEEPER_SEASON"
DEFAULT_SEASON = 2026
MAX_LABEL_LENGTH = 24


def truncate(value: str | None, limit: int = MAX_LABEL_LENGTH) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def assert_read_only() -> None:
    """Fail loudly if the client ever grows a write path."""

    if READ_ONLY_METHOD != "GET":
        raise SystemExit("refusing to run: Sleeper client is not GET-only")
    write_methods = ("post", "put", "patch", "delete")
    offenders = [name for name in dir(SleeperClient) if name.startswith(write_methods)]
    if offenders:
        raise SystemExit(f"refusing to run: Sleeper client exposes write methods {offenders}")


def main() -> int:
    opted_in = os.environ.get(OPT_IN_ENV, "").strip() in {"1", "true", "yes"}
    username = (os.environ.get(USERNAME_ENV) or "").strip()
    if not opted_in or not username:
        missing = []
        if not opted_in:
            missing.append(f"{OPT_IN_ENV}=1")
        if not username:
            missing.append(USERNAME_ENV)
        print(f"skipped live Sleeper smoke: set {' and '.join(missing)} to opt in")
        return 0

    assert_read_only()
    season = int(os.environ.get(SEASON_ENV) or DEFAULT_SEASON)
    client = SleeperClient(use_fixtures=False)
    try:
        user = client.get_user(username)
        if not isinstance(user, dict) or not user.get("user_id"):
            print("FAIL user lookup returned no user_id")
            return 1
        user_id = str(user["user_id"])
        print(f"OK   user resolved (display_name={truncate(user.get('display_name'))})")

        leagues = client.get_leagues(user_id, season)
        print(f"OK   leagues for {season}: {len(leagues)}")
        if not leagues:
            print("skipped league detail checks: no leagues for this season")
            return 0

        league_id = str(leagues[0]["league_id"])
        print(f"OK   sampling league name={truncate(leagues[0].get('name'))}")

        league = client.get_league(league_id)
        scoring_keys = len((league.get("scoring_settings") or {}))
        roster_slots = len(league.get("roster_positions") or [])
        print(f"OK   rules: {scoring_keys} scoring keys, {roster_slots} roster slots")

        rosters = client.get_rosters(league_id)
        print(f"OK   rosters: {len(rosters)}")

        state = client.get_nfl_state()
        week = int(state.get("week", 1) or 1)
        matchups = client.get_matchups(league_id, week)
        print(f"OK   matchups week {week}: {len(matchups)}")

        completeness = client.classify_completeness(f"league/{league_id}/rosters", rosters)
        print(f"OK   roster payload complete: {completeness}")
    except SleeperError as exc:
        print(f"FAIL live Sleeper request failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        client.close()

    print("live Sleeper smoke passed (read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
