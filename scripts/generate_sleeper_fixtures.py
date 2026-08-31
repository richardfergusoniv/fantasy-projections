"""Generate the Sleeper API fixture set from the seed league fixtures.

The seed fixtures under ``src/app/fixtures/seed`` are the single source of truth
for the six representative leagues (two redraft, four dynasty, two of them
Superflex, one points-per-first-down, one yardage-bonus, five with a team
defense, four with a kicker). This script derives the *Sleeper-shaped* payloads
from them so that a fixture-mode sync reproduces exactly the same six leagues,
the same rosters, and the same scoring contracts as the seed.

Why this exists: the two fixture universes used to disagree. The Sleeper set
described a single toy league whose roster held player ids that appeared nowhere
else, so running `POST /api/v1/sync` overwrote the seeded rosters and every
lineup recommendation started failing with `no_projected_players_on_roster`.

Identifier convention, mirroring the real API:

* Offensive players get an opaque Sleeper id (``sl-<n>``) and carry ``gsis_id``.
  That is how the real payload joins to nflverse-keyed projections, and it
  exercises the identity resolver's gsis path.
* Kickers and team defenses have no gsis id. Real Sleeper keys defenses by team
  abbreviation; the fixtures reuse the seed's own ``K-*``/``DEF-*`` ids as the
  Sleeper id so the resolver's sleeper-id path is exercised too.

Run: ``uv run python scripts/generate_sleeper_fixtures.py``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "src" / "app" / "fixtures" / "seed"
OUT = ROOT / "tests" / "fixtures" / "sleeper"

SEASON = 2026
WEEK = 1
OWNER_USER_ID = "fixture-user-1"
RIVAL_USER_ID = "fixture-user-2"

#: Players given a non-null Sleeper injury status, to exercise the availability
#: lifecycle without marking the whole league questionable.
INJURY_STATUS: dict[str, str] = {"00-0034857": "Questionable"}


def _load(name: str) -> Any:
    return json.loads((SEED / name).read_text(encoding="utf-8"))


def _write(name: str, payload: Any) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def build_player_index(players: list[dict]) -> tuple[dict[str, str], dict[str, Any]]:
    """Return (canonical player_id -> sleeper id, Sleeper players/nfl payload)."""
    to_sleeper: dict[str, str] = {}
    payload: dict[str, Any] = {}
    offense_counter = 0
    for player in players:
        canonical = player["player_id"]
        gsis_id = player.get("gsis_id")
        if gsis_id:
            offense_counter += 1
            sleeper_id = f"sl-{offense_counter:04d}"
        else:
            # Kickers and defenses: no gsis id exists upstream either.
            sleeper_id = player.get("sleeper_id") or canonical
        to_sleeper[canonical] = sleeper_id
        payload[sleeper_id] = {
            "player_id": sleeper_id,
            "gsis_id": gsis_id,
            "full_name": player["name"],
            "position": player["position"],
            "team": player.get("team"),
            "status": "Active",
            "injury_status": INJURY_STATUS.get(canonical),
        }
    return to_sleeper, payload


def sleeper_roster(roster: dict, owner_id: str, to_sleeper: dict[str, str]) -> dict:
    def convert(ids: list[str]) -> list[str]:
        return [to_sleeper.get(pid, pid) for pid in ids or []]

    return {
        "roster_id": roster["roster_id"],
        "owner_id": owner_id,
        "league_id": None,  # filled in by the caller
        "players": convert(roster.get("players", [])),
        "starters": convert(roster.get("starters", [])),
        "reserve": convert(roster.get("reserve", [])),
        "settings": {"wins": 0, "losses": 0, "fpts": 0, "waiver_budget_used": 0},
    }


def main() -> int:
    manifest = _load("leagues_manifest.json")
    players = _load("players.json")
    to_sleeper, players_payload = build_player_index(players)

    OUT.mkdir(parents=True, exist_ok=True)
    _write("players__nfl.json", players_payload)

    # Trending adds are real, mostly unrostered, players — that is what makes
    # them a waiver signal. What keeps them out of the forecast is that they are
    # stored outside the projection tables and tagged `projection_input: false`,
    # not that they use a different id space.
    trending_players = ["00-0030506", "00-0031381", "00-0033106"]
    _write(
        "players__nfl__trending__add.json",
        [
            {"player_id": to_sleeper[pid], "count": count}
            for pid, count in zip(trending_players, (4821, 1904, 233), strict=True)
        ],
    )
    _write(
        "user__fixture_owner.json",
        {
            "user_id": OWNER_USER_ID,
            "username": "fixture_owner",
            "display_name": "Fixture Owner",
        },
    )
    _write(
        "state__nfl.json",
        {
            "season": SEASON,
            "week": WEEK,
            "season_type": "regular",
            "week_has_completed": True,
            "display_week": WEEK,
        },
    )

    league_list: list[dict] = []
    for entry in manifest["leagues"]:
        league = _load(entry["league_file"])
        league_id = league["league_id"]
        is_dynasty = league["type"] == "dynasty"
        draft_rule = entry.get("draft_order_rule")

        league_payload = {
            "league_id": league_id,
            "name": league["name"],
            "season": str(league["season"]),
            "status": "in_season",
            "sport": "nfl",
            "previous_league_id": None,
            "draft_id": f"{league_id}-draft",
            "settings": {"type": 2 if is_dynasty else 0, "num_teams": 2},
            "scoring_settings": league["scoring_settings"],
            "roster_positions": league["roster_positions"],
            "metadata": {},
        }
        league_list.append(league_payload)
        _write(f"league__{league_id}.json", league_payload)

        _write(
            f"league__{league_id}__users.json",
            [
                {
                    "user_id": OWNER_USER_ID,
                    "display_name": "Owner",
                    "username": "fixture_owner",
                },
                {
                    "user_id": RIVAL_USER_ID,
                    "display_name": "Rival",
                    "username": "rival",
                },
            ],
        )

        rosters = []
        for owner_id, roster_key in (
            (OWNER_USER_ID, "roster_file"),
            (RIVAL_USER_ID, "opponent_roster_file"),
        ):
            if roster_key not in entry:
                continue
            row = sleeper_roster(_load(entry[roster_key]), owner_id, to_sleeper)
            row["league_id"] = league_id
            rosters.append(row)
        _write(f"league__{league_id}__rosters.json", rosters)

        _write(
            f"league__{league_id}__matchups__{WEEK}.json",
            [
                {
                    "roster_id": row["roster_id"],
                    "matchup_id": 1,
                    "points": 0,
                    "starters": row["starters"],
                    "players": row["players"],
                }
                for row in rosters
            ],
        )

        # Only dynasty leagues trade future picks, so only they have traded-pick
        # or pick-bearing trade payloads.
        _write(
            f"league__{league_id}__traded_picks.json",
            [
                {
                    "season": str(SEASON + 1),
                    "round": 1,
                    "roster_id": 2,
                    "previous_owner_id": 2,
                    "owner_id": 1,
                },
                {
                    "season": str(SEASON + 2),
                    "round": 2,
                    "roster_id": 1,
                    "previous_owner_id": 1,
                    "owner_id": 2,
                },
            ]
            if is_dynasty
            else [],
        )

        transactions: list[dict] = []
        if is_dynasty:
            owner_out = to_sleeper.get(_load(entry["roster_file"])["players"][0])
            rival_out = to_sleeper.get(
                _load(entry["opponent_roster_file"])["players"][0]
            )
            transactions.append(
                {
                    "transaction_id": f"{league_id}-txn-1",
                    "type": "trade",
                    "status": "complete",
                    "created": 1_756_000_000_000,
                    "roster_ids": [1, 2],
                    "adds": {rival_out: 1, owner_out: 2},
                    "drops": {owner_out: 1, rival_out: 2},
                    "draft_picks": [
                        {
                            "season": str(SEASON + 1),
                            "round": 1,
                            "roster_id": 2,
                            "previous_owner_id": 2,
                            "owner_id": 1,
                        }
                    ],
                }
            )
        _write(f"league__{league_id}__transactions__{WEEK}.json", transactions)

        draft = {
            "draft_id": f"{league_id}-draft",
            "league_id": league_id,
            "season": str(SEASON),
            "status": "complete",
            "type": "snake",
            "settings": {"teams": 2, "rounds": 3, "reversal_round": 0},
            "draft_order": {OWNER_USER_ID: 1, RIVAL_USER_ID: 2},
            "slot_to_roster_id": {"1": 1, "2": 2},
            "metadata": {"name": f"{league['name']} Draft"},
        }
        # Rookie-pick order is a dynasty concept; a redraft league states none,
        # and the sync must never infer one.
        if draft_rule and is_dynasty:
            draft["metadata"]["draft_order_rule"] = draft_rule
        _write(f"league__{league_id}__drafts.json", [draft])

    _write(f"user__{OWNER_USER_ID}__leagues__nfl__{SEASON}.json", league_list)
    print(f"wrote Sleeper fixtures for {len(league_list)} leagues to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
