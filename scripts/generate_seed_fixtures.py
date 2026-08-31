"""Regenerate the six representative seed-league fixtures.

The fixtures must collectively satisfy the product contract:

* 2 redraft leagues and 4 dynasty leagues
* 2 of the dynasty leagues are Superflex
* exactly 1 league scores points per first down
* exactly 1 league scores yardage bonuses
* 5 leagues start a team defense and 4 start a kicker
* among the dynasty leagues, 2 assign rookie picks by max potential points and
  2 by reverse standings

Rosters are built from the active release bundle so every seeded league has a
full, legal lineup of real projected players rather than two placeholders. Each
league gets a disjoint slice of the player pool so the six leagues are genuinely
distinct and a bug that crosses league boundaries is detectable.

Run:  uv run python scripts/generate_seed_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEED_DIR = REPO / "src" / "app" / "fixtures" / "seed"
SCORING_FIXTURE_DIR = REPO / "tests" / "fixtures" / "scoring"

SEASON = 2026
WEEK = 1

BASE_OFFENSE = {
    "pass_yd": 0.04,
    "pass_td": 4,
    "pass_int": -2,
    "rush_yd": 0.1,
    "rush_td": 6,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6,
    "fum_lost": -2,
}

FULL_PPR = {**BASE_OFFENSE, "rec": 1.0}

PPFD = {
    **BASE_OFFENSE,
    "pass_fd": 0.5,
    "rush_fd": 0.5,
    "rec_fd": 0.5,
}

YARDAGE_BONUS = {
    **BASE_OFFENSE,
    "bonus_pass_yd_300": 3,
    "bonus_pass_yd_400": 3,
    "bonus_rush_yd_100": 3,
    "bonus_rush_yd_200": 3,
    "bonus_rec_yd_100": 3,
    "bonus_rec_yd_200": 3,
}

KICKER_RULES = {
    "fgm_0_19": 3,
    "fgm_20_29": 3,
    "fgm_30_39": 3,
    "fgm_40_49": 4,
    "fgm_50p": 5,
    "xpm": 1,
    "xpmiss": -1,
}

DST_SIMPLE = {
    "def_sack": 1,
    "def_int": 2,
    "def_fr": 2,
    "def_td": 6,
    "safe": 2,
}

#: Sleeper's real tiered points-allowed scoring, exercised by one league.
DST_TIERED = {
    **DST_SIMPLE,
    "pts_allow_0": 10,
    "pts_allow_1_6": 7,
    "pts_allow_7_13": 4,
    "pts_allow_14_20": 1,
    "pts_allow_21_27": 0,
    "pts_allow_28_34": -1,
    "pts_allow_35p": -4,
}

OFFENSE_CORE = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]

LEAGUES = [
    {
        "key": "standard",
        "league_id": "fixture-standard",
        "name": "Standard Half PPR",
        "type": "redraft",
        "scoring": {**BASE_OFFENSE, **KICKER_RULES, **DST_SIMPLE},
        "roster_positions": [*OFFENSE_CORE, "K", "DEF", "BN", "BN", "BN"],
        "draft_order_rule": None,
    },
    {
        "key": "ppfd",
        "league_id": "fixture-ppfd",
        "name": "Points Per First Down",
        "type": "redraft",
        "scoring": PPFD,
        "roster_positions": [*OFFENSE_CORE, "BN", "BN", "BN"],
        "draft_order_rule": None,
    },
    {
        "key": "superflex",
        "league_id": "fixture-superflex",
        "name": "Superflex Dynasty (Max PF)",
        "type": "dynasty",
        "scoring": {**FULL_PPR, **KICKER_RULES, **DST_SIMPLE},
        "roster_positions": [
            *OFFENSE_CORE,
            "SUPER_FLEX",
            "K",
            "DEF",
            "BN",
            "BN",
            "TAXI",
        ],
        "draft_order_rule": "max_pf",
    },
    {
        "key": "dynasty",
        "league_id": "fixture-dynasty",
        "name": "Superflex Dynasty (Reverse Standings)",
        "type": "dynasty",
        "scoring": {**FULL_PPR, **KICKER_RULES, **DST_SIMPLE},
        "roster_positions": [
            *OFFENSE_CORE,
            "SUPER_FLEX",
            "K",
            "DEF",
            "BN",
            "BN",
            "IR",
        ],
        "draft_order_rule": "reverse_standings",
    },
    {
        "key": "yardage_bonus",
        "league_id": "fixture-yardage-bonus",
        "name": "Yardage Bonus Dynasty (Max PF)",
        "type": "dynasty",
        "scoring": {**YARDAGE_BONUS, **DST_SIMPLE},
        "roster_positions": [*OFFENSE_CORE, "DEF", "BN", "BN", "BN"],
        "draft_order_rule": "max_pf",
    },
    {
        "key": "k_dst",
        "league_id": "fixture-k-dst",
        "name": "Kicker and Tiered Defense Dynasty (Reverse Standings)",
        "type": "dynasty",
        "scoring": {**BASE_OFFENSE, **KICKER_RULES, **DST_TIERED},
        "roster_positions": [*OFFENSE_CORE, "K", "DEF", "BN", "BN", "BN"],
        "draft_order_rule": "reverse_standings",
    },
]

#: Special-teams identities. The release bundle projects offense only; kickers
#: and team defenses are simulated by src/projection/special_teams.
SPECIAL_TEAMS = [
    {"player_id": "K-BAL", "name": "Baltimore Kicker", "position": "K", "team": "BAL"},
    {"player_id": "K-SF", "name": "San Francisco Kicker", "position": "K", "team": "SF"},
    {"player_id": "K-DET", "name": "Detroit Kicker", "position": "K", "team": "DET"},
    {"player_id": "K-KC", "name": "Kansas City Kicker", "position": "K", "team": "KC"},
    {"player_id": "K-BUF", "name": "Buffalo Kicker", "position": "K", "team": "BUF"},
    {"player_id": "K-PHI", "name": "Philadelphia Kicker", "position": "K", "team": "PHI"},
    {"player_id": "K-DAL", "name": "Dallas Kicker", "position": "K", "team": "DAL"},
    {"player_id": "K-GB", "name": "Green Bay Kicker", "position": "K", "team": "GB"},
    {"player_id": "DEF-BUF", "name": "Buffalo Defense", "position": "DEF", "team": "BUF"},
    {"player_id": "DEF-PHI", "name": "Philadelphia Defense", "position": "DEF", "team": "PHI"},
    {"player_id": "DEF-DEN", "name": "Denver Defense", "position": "DEF", "team": "DEN"},
    {"player_id": "DEF-MIN", "name": "Minnesota Defense", "position": "DEF", "team": "MIN"},
    {"player_id": "DEF-HOU", "name": "Houston Defense", "position": "DEF", "team": "HOU"},
    {"player_id": "DEF-PIT", "name": "Pittsburgh Defense", "position": "DEF", "team": "PIT"},
    {"player_id": "DEF-BAL", "name": "Baltimore Defense", "position": "DEF", "team": "BAL"},
    {"player_id": "DEF-LA", "name": "Los Angeles Defense", "position": "DEF", "team": "LA"},
    {"player_id": "DEF-GB", "name": "Green Bay Defense", "position": "DEF", "team": "GB"},
    {"player_id": "DEF-SF", "name": "San Francisco Defense", "position": "DEF", "team": "SF"},
    {"player_id": "DEF-KC", "name": "Kansas City Defense", "position": "DEF", "team": "KC"},
    {"player_id": "DEF-DET", "name": "Detroit Defense", "position": "DEF", "team": "DET"},
]


def load_bundle_players() -> dict[str, list[dict]]:
    from src.projection.active_release import read_active_pointer
    from src.projection.contracts import REPO_ROOT

    pointer = read_active_pointer(SEASON)
    if pointer is None:
        raise SystemExit("no active release pointer; cannot build seed rosters")
    rel = (pointer.get("public_urls") or {}).get("players")
    path = Path(REPO_ROOT) / "draft_assistant" / rel
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_position: dict[str, list[dict]] = {}
    for row in payload["players"]:
        position = row.get("position")
        if position not in {"QB", "RB", "WR", "TE"}:
            continue
        by_position.setdefault(position, []).append(row)
    for rows in by_position.values():
        rows.sort(key=lambda r: -(r.get("fantasy_pts_season") or 0.0))
    return by_position


class PlayerAllocator:
    """Hands out disjoint players so leagues never share a roster spot."""

    def __init__(self, by_position: dict[str, list[dict]]) -> None:
        self._pools = {pos: list(rows) for pos, rows in by_position.items()}
        self._cursor = dict.fromkeys(self._pools, 0)
        self.used: list[dict] = []

    def take(self, position: str, count: int) -> list[dict]:
        pool = self._pools[position]
        start = self._cursor[position]
        if start + count > len(pool):
            raise SystemExit(f"exhausted {position} pool at {start}+{count}")
        chosen = pool[start : start + count]
        self._cursor[position] = start + count
        self.used.extend(chosen)
        return chosen


class SpecialTeamsAllocator:
    def __init__(self) -> None:
        self._kickers = [p for p in SPECIAL_TEAMS if p["position"] == "K"]
        self._defenses = [p for p in SPECIAL_TEAMS if p["position"] == "DEF"]
        self._k = 0
        self._d = 0
        self.used: list[dict] = []

    def take(self, position: str) -> dict:
        if position == "K":
            player = self._kickers[self._k % len(self._kickers)]
            self._k += 1
        else:
            player = self._defenses[self._d % len(self._defenses)]
            self._d += 1
        self.used.append(player)
        return player


def build_roster(
    league: dict, allocator: PlayerAllocator, st: SpecialTeamsAllocator, roster_id: int
) -> dict:
    """Build a roster that can legally fill every starting seat, plus bench."""
    slots = league["roster_positions"]
    needs = {
        "QB": slots.count("QB") + slots.count("SUPER_FLEX"),
        "RB": slots.count("RB") + 1,
        "WR": slots.count("WR") + 1,
        "TE": slots.count("TE") + 1,
    }
    starters: list[str] = []
    players: list[str] = []

    picks = {pos: allocator.take(pos, count) for pos, count in needs.items()}
    # Starters, in roster_positions order, so the fixture mirrors Sleeper output.
    pos_cursor = {pos: 0 for pos in picks}

    def next_of(position: str) -> str:
        idx = pos_cursor[position]
        pos_cursor[position] = idx + 1
        return picks[position][idx]["player_id"]

    for slot in slots:
        if slot in {"BN", "IR", "TAXI"}:
            continue
        if slot == "FLEX":
            starters.append(next_of("RB"))
        elif slot == "SUPER_FLEX":
            starters.append(next_of("QB"))
        elif slot in {"K", "DEF"}:
            starters.append(st.take(slot)["player_id"])
        else:
            starters.append(next_of(slot))

    players.extend(starters)
    # Bench: whatever remains from the allocated pool.
    for position, rows in picks.items():
        for row in rows[pos_cursor[position] :]:
            players.append(row["player_id"])

    return {
        "roster_id": roster_id,
        "players": players,
        "starters": starters,
        "reserve": [],
    }


def main() -> int:
    by_position = load_bundle_players()
    allocator = PlayerAllocator(by_position)
    st = SpecialTeamsAllocator()

    manifest_leagues = []
    identity_rows: dict[str, dict] = {}

    for league in LEAGUES:
        league_doc = {
            "league_id": league["league_id"],
            "season": SEASON,
            "name": league["name"],
            "type": league["type"],
            "roster_positions": league["roster_positions"],
            "scoring_settings": league["scoring"],
        }
        (SEED_DIR / f"league_{league['key']}.json").write_text(
            json.dumps(league_doc, indent=2) + "\n", encoding="utf-8"
        )
        # Scoring contract regression fixture used by tests/scoring.
        (SCORING_FIXTURE_DIR / f"{league['key']}.json").write_text(
            json.dumps(
                {
                    "league_id": league["league_id"],
                    "roster_positions": league["roster_positions"],
                    "scoring_settings": league["scoring"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        owner = build_roster(league, allocator, st, roster_id=1)
        rival = build_roster(league, allocator, st, roster_id=2)
        (SEED_DIR / f"roster_{league['key']}.json").write_text(
            json.dumps(owner, indent=2) + "\n", encoding="utf-8"
        )
        (SEED_DIR / f"roster_{league['key']}_opponent.json").write_text(
            json.dumps(rival, indent=2) + "\n", encoding="utf-8"
        )

        entry = {
            "league_file": f"league_{league['key']}.json",
            "roster_file": f"roster_{league['key']}.json",
            "opponent_roster_file": f"roster_{league['key']}_opponent.json",
            "members": [
                {"user_id": "u1", "roster_id": 1, "display_name": "Owner"},
                {"user_id": "u2", "roster_id": 2, "display_name": "Rival"},
            ],
        }
        if league["draft_order_rule"]:
            entry["draft_order_rule"] = league["draft_order_rule"]
        manifest_leagues.append(entry)

    for row in allocator.used:
        identity_rows[row["player_id"]] = {
            "player_id": row["player_id"],
            "gsis_id": row["player_id"],
            "name": row.get("display_name") or row["player_id"],
            "position": row["position"],
            "team": row.get("team"),
        }
    for row in st.used:
        identity_rows[row["player_id"]] = {
            "player_id": row["player_id"],
            "sleeper_id": row["player_id"],
            "name": row["name"],
            "position": row["position"],
            "team": row["team"],
        }

    (SEED_DIR / "players.json").write_text(
        json.dumps(sorted(identity_rows.values(), key=lambda r: r["player_id"]), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (SEED_DIR / "leagues_manifest.json").write_text(
        json.dumps(
            {
                "season": SEASON,
                "week": WEEK,
                "generated_by": "scripts/generate_seed_fixtures.py",
                "leagues": manifest_leagues,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Legacy single-opponent fixture retained for backwards compatibility.
    legacy = SEED_DIR / "roster_opponent.json"
    if legacy.exists():
        legacy.write_text(
            (SEED_DIR / "roster_standard_opponent.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    print(f"wrote {len(LEAGUES)} leagues, {len(identity_rows)} player identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
