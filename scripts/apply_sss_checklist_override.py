#!/usr/bin/env python3
"""Override draft checklist checks from @SUNDAYSPORTSSOCIETY screenshot boards.

Projection-derived context checks are replaced for every player who appears on
the SSS positional boards. Players not on those boards keep market order but
have checks cleared so we are not mixing sources. O-line is included for QB/RB
because the graphics publish that column.

Re-run after refreshing draft_checklist_{season}.json from checklist_prepare if
you want to re-apply this overlay on a newer market board.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "draft_assistant" / "data" / "draft_checklist_2026.json"

# (rank, name, checks...) — checks order matches CRITERIA_BASE with OL for QB/RB.
# Source: @SUNDAYSPORTSSOCIETY WR/RB/QB/TE checklist graphics (PPR boards).

WR_BOARD: list[tuple[int, str, tuple[bool, bool, bool, bool]]] = [
    # target_leader, qb_top16, offense_top16, sos_top16
    (1, "Ja'Marr Chase", (True, True, True, True)),
    (2, "Puka Nacua", (True, True, True, True)),
    (3, "Jaxon Smith-Njigba", (True, True, True, True)),
    (4, "Amon-Ra St. Brown", (True, True, True, True)),
    (5, "CeeDee Lamb", (True, True, True, True)),
    (6, "Justin Jefferson", (True, False, False, True)),
    (7, "Drake London", (True, False, False, True)),
    (8, "A.J. Brown", (True, True, True, False)),
    (9, "Nico Collins", (True, False, False, False)),
    (10, "George Pickens", (False, True, True, True)),
    (11, "Rashee Rice", (True, True, True, False)),
    (12, "Malik Nabers", (True, False, False, True)),
    (13, "Chris Olave", (True, False, False, True)),
    (14, "DeVonta Smith", (True, False, True, True)),
    (15, "Zay Flowers", (True, True, True, False)),
    (16, "Emeka Egbuka", (True, True, False, True)),
    (17, "Tee Higgins", (False, True, True, True)),
    (18, "Garrett Wilson", (True, False, False, True)),
    (19, "Tetairoa McMillan", (True, False, False, False)),
    (20, "Ladd McConkey", (True, True, True, False)),
    (21, "Jaylen Waddle", (True, True, True, True)),
    (22, "Terry McLaurin", (True, False, False, True)),
    (23, "Luther Burden", (True, True, True, False)),
    (24, "Davante Adams", (False, True, True, True)),
    (25, "DJ Moore", (True, True, True, False)),
    (26, "Jameson Williams", (False, True, True, True)),
    (27, "Rome Odunze", (False, True, True, False)),
    (28, "Mike Evans", (True, True, True, True)),
    (29, "Christian Watson", (True, True, True, True)),
    (30, "Parker Washington", (True, True, True, False)),
    (31, "Carnell Tate", (False, False, False, False)),
    (32, "Marvin Harrison Jr.", (False, False, True, False)),
    (33, "Brian Thomas Jr.", (False, True, True, False)),
    (34, "DK Metcalf", (False, False, False, False)),
    (35, "Courtland Sutton", (False, True, False, False)),
    (36, "Josh Downs", (False, False, True, True)),
    (37, "Quentin Johnston", (False, True, True, False)),
    (38, "Chris Godwin", (False, True, False, True)),
    (39, "Alec Pierce", (False, False, True, True)),
    (40, "Michael Wilson", (False, False, False, False)),
    (41, "Jordyn Tyson", (False, False, False, True)),
    (42, "Michael Pittman Jr.", (True, False, False, False)),
    (43, "Stefon Diggs", (False, False, False, True)),
    (44, "Wan'Dale Robinson", (True, False, False, True)),
    (45, "Jordan Addison", (False, False, False, True)),
    (46, "Jayden Reed", (False, True, True, True)),
    (47, "Makai Lemon", (False, False, True, True)),
    (48, "Matthew Golden", (False, True, True, True)),
    (49, "Jakobi Meyers", (False, True, True, False)),
    (50, "Xavier Worthy", (False, True, True, False)),
    (51, "KC Concepcion", (True, False, False, False)),
    (52, "Romeo Doubs", (False, True, True, False)),
    (53, "Deebo Samuel", (False, True, True, True)),
    (54, "DeZhaun Stribling", (False, True, True, True)),
    (55, "Khalil Shakir", (False, True, True, False)),
    (56, "Jalen Coker", (False, False, False, False)),
    (57, "Rashid Shaheed", (False, True, True, True)),
    (58, "Denzel Boston", (False, False, False, False)),
    (59, "Tre Tucker", (False, False, False, False)),
    (60, "Jerry Jeudy", (False, False, False, False)),
]

RB_BOARD: list[tuple[int, str, tuple[bool, bool, bool, bool, bool]]] = [
    # target_leader, rush_leader, offense, ol, sos
    (1, "Jahmyr Gibbs", (True, True, True, True, True)),
    (2, "Bijan Robinson", (True, True, False, True, True)),
    (3, "Christian McCaffrey", (True, True, True, True, False)),
    (4, "Jonathan Taylor", (True, True, True, True, False)),
    (5, "James Cook", (True, True, True, True, False)),
    (6, "Ashton Jeanty", (True, True, False, False, False)),
    (7, "De'Von Achane", (True, True, False, False, True)),
    (8, "Chase Brown", (True, True, True, False, False)),
    (9, "Saquon Barkley", (True, True, True, True, True)),
    (10, "Kenneth Walker III", (True, True, True, True, True)),
    (11, "Omarion Hampton", (True, True, True, True, False)),
    (12, "Derrick Henry", (False, True, True, False, True)),
    (13, "Jeremiyah Love", (True, True, False, False, False)),
    (14, "Kyren Williams", (True, True, True, True, True)),
    (15, "Breece Hall", (True, True, False, False, False)),
    (16, "Javonte Williams", (True, True, True, True, True)),
    (17, "Josh Jacobs", (True, True, True, False, False)),
    (18, "Cam Skattebo", (True, True, False, False, False)),
    (19, "Travis Etienne", (True, True, False, True, True)),
    (20, "D'Andre Swift", (True, True, True, True, False)),
    (21, "David Montgomery", (True, True, False, False, True)),
    (22, "Quinshon Judkins", (True, True, False, False, True)),
    (23, "Bucky Irving", (False, True, False, True, False)),
    (24, "Bhayshul Tuten", (True, True, True, False, False)),
    (25, "TreVeyon Henderson", (True, False, True, False, False)),
    (26, "Jadarian Price", (False, True, True, True, True)),
    (27, "Jaylen Warren", (True, False, False, False, False)),
    (28, "Rhamondre Stevenson", (False, True, True, False, False)),
    (29, "Tony Pollard", (False, True, False, False, True)),
    (30, "Jonathon Brooks", (True, False, False, False, False)),
    (31, "Rico Dowdle", (False, True, False, False, False)),
    (32, "RJ Harvey", (True, False, False, True, True)),
    (33, "J.K. Dobbins", (False, True, False, True, True)),
    (34, "Chuba Hubbard", (False, True, False, False, True)),
    (35, "Kenny Gainwell", (True, False, False, True, False)),
    (36, "Blake Corum", (False, False, True, True, True)),
    (37, "Jordan Mason", (False, True, False, True, True)),
    (38, "Kyle Monangai", (False, False, True, True, False)),
    (39, "Jacory Croskey-Merritt", (False, True, False, False, True)),
    (40, "Rachaad White", (True, False, False, True, True)),
    (41, "Aaron Jones", (True, False, False, True, True)),
    (42, "Tyrone Tracy Jr.", (False, False, False, False, False)),
    (43, "Keaton Mitchell", (False, False, True, True, False)),
    (44, "Zach Charbonnet", (True, False, True, True, True)),
    (45, "Tyler Allgeier", (False, False, False, True, True)),
    (46, "Tyjae Spears", (True, False, False, False, True)),
    (47, "Alvin Kamara", (False, False, False, True, True)),
    (48, "Isiah Pacheco", (False, False, True, True, True)),
]

QB_BOARD: list[tuple[int, str, tuple[bool, bool, bool, bool, bool]]] = [
    # pass_att, rush_att, offense, ol, sos
    (1, "Josh Allen", (False, True, True, True, False)),
    (2, "Lamar Jackson", (False, True, True, False, True)),
    (3, "Drake Maye", (True, True, True, False, False)),
    (4, "Joe Burrow", (True, False, True, False, True)),
    (5, "Jayden Daniels", (False, True, False, False, True)),
    (6, "Jalen Hurts", (False, True, True, True, True)),
    (7, "Caleb Williams", (True, True, True, True, False)),
    (8, "Dak Prescott", (True, False, True, True, True)),
    (9, "Trevor Lawrence", (True, True, True, False, True)),
    (10, "Justin Herbert", (True, True, True, True, False)),
    (11, "Jaxson Dart", (False, True, False, False, True)),
    (12, "Matthew Stafford", (True, False, True, True, True)),
    (13, "Brock Purdy", (False, True, True, True, True)),
    (14, "Patrick Mahomes", (True, True, True, True, False)),
    (15, "Bo Nix", (True, True, False, True, False)),
    (16, "Jared Goff", (True, False, True, True, True)),
    (17, "Kyler Murray", (True, True, False, True, True)),
    (18, "Baker Mayfield", (True, True, False, True, False)),
    (19, "Jordan Love", (True, False, True, False, False)),
    (20, "Tyler Shough", (True, True, False, True, True)),
    (21, "Malik Willis", (False, True, False, False, False)),
    (22, "Daniel Jones", (False, True, True, True, True)),
    (23, "C.J. Stroud", (True, False, False, False, True)),
    (24, "Sam Darnold", (False, False, True, True, False)),
]

TE_BOARD: list[tuple[int, str, tuple[bool, bool, bool, bool]]] = [
    # te_top2, qb, offense, sos
    (1, "Brock Bowers", (True, False, False, False)),
    (2, "Trey McBride", (True, False, False, False)),
    (3, "Colston Loveland", (True, True, True, False)),
    (4, "Tyler Warren", (True, False, True, True)),
    (5, "Sam LaPorta", (False, True, True, True)),
    (6, "Tucker Kraft", (True, True, True, False)),
    (7, "Kyle Pitts", (True, False, False, True)),
    (8, "Harold Fannin", (True, False, False, True)),
    (9, "George Kittle", (False, True, True, True)),
    (10, "Travis Kelce", (True, True, True, True)),
    (11, "Dallas Goedert", (False, False, True, True)),
    (12, "Isaiah Likely", (True, False, False, False)),
    (13, "Dalton Kincaid", (True, True, True, False)),
    (14, "Jake Ferguson", (False, True, True, False)),
    (15, "Mark Andrews", (True, True, True, True)),
    (16, "Juwan Johnson", (True, False, False, True)),
    (17, "Brenton Strange", (False, True, True, False)),
    (18, "Hunter Henry", (False, True, True, False)),
    (19, "T.J. Hockenson", (False, False, False, True)),
    (20, "Kenyon Sadiq", (False, False, False, True)),
    (21, "Dalton Schultz", (True, False, False, True)),
    (22, "AJ Barner", (False, True, True, True)),
    (23, "Oronde Gadsden", (False, True, True, False)),
    (24, "Terrance Ferguson", (False, True, True, False)),
]

NAME_ALIASES: dict[str, str] = {
    "aj brown": "a.j. brown",
    "a j brown": "a.j. brown",
    "dk metcalf": "d.k. metcalf",
    "d k metcalf": "d.k. metcalf",
    "dj moore": "d.j. moore",
    "d j moore": "d.j. moore",
    "cj stroud": "c.j. stroud",
    "c j stroud": "c.j. stroud",
    "jk dobbins": "j.k. dobbins",
    "j k dobbins": "j.k. dobbins",
    "tj hockenson": "t.j. hockenson",
    "t j hockenson": "t.j. hockenson",
    "rj harvey": "r.j. harvey",
    "r j harvey": "r.j. harvey",
    "aj barner": "a.j. barner",
    "a j barner": "a.j. barner",
    "denry henry": "derrick henry",
    "james cook iii": "james cook",
    "travis etienne jr": "travis etienne",
    "travis etienne jr.": "travis etienne",
    "chris godwin jr": "chris godwin",
    "chris godwin jr.": "chris godwin",
    "deebo samuel sr": "deebo samuel",
    "deebo samuel sr.": "deebo samuel",
    "luther burden iii": "luther burden",
    "marvin harrison jr": "marvin harrison jr.",
    "brian thomas jr": "brian thomas jr.",
    "michael pittman jr": "michael pittman jr.",
    "tyrone tracy jr": "tyrone tracy jr.",
    "kc concepcion": "k.c. concepcion",
    "sam laporta": "sam laporta",
    "harold fannin": "harold fannin jr.",
    "kenneth gainwell": "kenny gainwell",
    "isiah pacheco": "isiah pacheco",
    "bhayshul tuten": "bhayshul tuten",
    "jadarian price": "jadarian price",
    "oronde gadsden": "oronde gadsden",
    "kenyon sadiq": "kenyon sadiq",
    "tyler shough": "tyler shough",
    "dezhaun stribling": "dezhaun stribling",
    "jordyn tyson": "jordyn tyson",
    "carnell tate": "carnell tate",
    "makai lemon": "makai lemon",
    "matthew golden": "matthew golden",
    "denzel boston": "denzel boston",
    "jeremiyah love": "jeremiyah love",
    "cam skattebo": "cam skattebo",
}


def norm_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "'").replace("'", "'")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(jr|sr|iii|ii|iv)\b\.?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return NAME_ALIASES.get(text, text)


def build_name_index(players: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for player in players:
        key = norm_name(str(player.get("name") or ""))
        index.setdefault(key, []).append(player)
        # Also index without periods (A.J. -> AJ)
        compact = key.replace(".", "")
        if compact != key:
            index.setdefault(compact, []).append(player)
    return index


def resolve_player(
    name: str,
    position: str,
    index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    key = norm_name(name)
    candidates = index.get(key) or index.get(key.replace(".", "")) or []
    pos_matches = [p for p in candidates if p.get("position") == position]
    pool = pos_matches or candidates
    if len(pool) == 1:
        return pool[0]
    if len(pool) > 1:
        # Prefer exact position + non-empty ADP when ambiguous.
        ranked = sorted(
            pool,
            key=lambda p: (
                0 if p.get("position") == position else 1,
                0 if p.get("adp") is not None else 1,
                p.get("pos_market_rank") or 9999,
            ),
        )
        return ranked[0]
    # Token fallback: last-name + first-token match within position.
    tokens = key.split()
    if len(tokens) >= 2:
        last = tokens[-1]
        first = tokens[0]
        hits = []
        for players in index.values():
            for player in players:
                if player.get("position") != position:
                    continue
                pn = norm_name(str(player.get("name") or ""))
                pt = pn.split()
                if pt and pt[-1] == last and pt[0].startswith(first[:3]):
                    hits.append(player)
        if len(hits) == 1:
            return hits[0]
    return None


def empty_checks(position: str, *, ol_included: bool) -> dict[str, bool]:
    if position == "QB":
        keys = ["pass_att_top16", "rush_vol_top16", "offense_top16", "sos_top16"]
        if ol_included:
            keys = ["pass_att_top16", "rush_vol_top16", "offense_top16", "ol_top16", "sos_top16"]
    elif position == "RB":
        keys = ["target_leader_in_group", "rush_vol_leader_in_group", "offense_top16", "sos_top16"]
        if ol_included:
            keys = [
                "target_leader_in_group",
                "rush_vol_leader_in_group",
                "offense_top16",
                "ol_top16",
                "sos_top16",
            ]
    elif position == "WR":
        keys = ["target_leader_in_group", "qb_top16", "offense_top16", "sos_top16"]
    else:
        keys = ["te_top2_targets_in_group", "qb_top16", "offense_top16", "sos_top16"]
    return {key: False for key in keys}


def apply_board(
    *,
    board: list[tuple[int, str, tuple[bool, ...]]],
    position: str,
    keys: list[str],
    index: dict[str, list[dict[str, Any]]],
    touched: set[str],
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []
    for rank, name, flags in board:
        player = resolve_player(name, position, index)
        if player is None:
            missing.append(f"{position} #{rank} {name}")
            continue
        checks = {key: bool(flags[i]) for i, key in enumerate(keys)}
        player["checks"] = checks
        player["pos_market_rank"] = rank
        player["rank_tier"] = "adp" if player.get("adp") is not None else player.get("rank_tier") or "ecr"
        player["unranked_break"] = False
        player["sss_board"] = True
        touched.add(str(player["player_id"]))
        matched.append(f"{position} #{rank} {player['name']}")
    return matched, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    players: list[dict[str, Any]] = list(payload.get("players") or [])
    index = build_name_index(players)
    touched: set[str] = set()

    matched: list[str] = []
    missing: list[str] = []

    m, miss = apply_board(
        board=WR_BOARD,
        position="WR",
        keys=["target_leader_in_group", "qb_top16", "offense_top16", "sos_top16"],
        index=index,
        touched=touched,
    )
    matched.extend(m)
    missing.extend(miss)

    m, miss = apply_board(
        board=RB_BOARD,
        position="RB",
        keys=[
            "target_leader_in_group",
            "rush_vol_leader_in_group",
            "offense_top16",
            "ol_top16",
            "sos_top16",
        ],
        index=index,
        touched=touched,
    )
    matched.extend(m)
    missing.extend(miss)

    m, miss = apply_board(
        board=QB_BOARD,
        position="QB",
        keys=["pass_att_top16", "rush_vol_top16", "offense_top16", "ol_top16", "sos_top16"],
        index=index,
        touched=touched,
    )
    matched.extend(m)
    missing.extend(miss)

    m, miss = apply_board(
        board=TE_BOARD,
        position="TE",
        keys=["te_top2_targets_in_group", "qb_top16", "offense_top16", "sos_top16"],
        index=index,
        touched=touched,
    )
    matched.extend(m)
    missing.extend(miss)

    # Clear projection checks for everyone not on an SSS board so sources don't mix.
    for player in players:
        pid = str(player["player_id"])
        if pid in touched:
            continue
        player["checks"] = empty_checks(str(player.get("position") or "WR"), ol_included=True)
        player["sss_board"] = False
        # Push off-board players below the SSS board within each position.
        pos = str(player.get("position") or "")
        board_size = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}.get(pos, 60)
        current = int(player.get("pos_market_rank") or 9999)
        if current <= board_size:
            player["pos_market_rank"] = board_size + current
        player["unranked_break"] = current == 1 or player.get("unranked_break") is True

    # Stable position order then SSS/market rank.
    pos_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    players.sort(
        key=lambda row: (
            pos_order.get(str(row.get("position")), 9),
            int(row.get("pos_market_rank") or 9999),
            str(row.get("name") or ""),
        )
    )
    # Mark the first off-board row per position for the divider.
    for pos in ("QB", "RB", "WR", "TE"):
        cohort = [p for p in players if p.get("position") == pos]
        saw_board = False
        for player in cohort:
            on_board = bool(player.get("sss_board"))
            if on_board:
                saw_board = True
                player["unranked_break"] = False
            else:
                player["unranked_break"] = saw_board and not any(
                    bool(p.get("unranked_break")) for p in cohort if not p.get("sss_board") and p is not player
                )
                # Only the first off-board player gets the break flag.
                if player["unranked_break"]:
                    saw_board = False  # prevent later ones; break already placed
                    # Actually we need only first — set others false below
        first_off = next((p for p in cohort if not p.get("sss_board")), None)
        for player in cohort:
            player["unranked_break"] = first_off is not None and player is first_off

    payload["players"] = players
    payload["criteria_by_position"] = {
        "QB": ["pass_att_top16", "rush_vol_top16", "offense_top16", "ol_top16", "sos_top16"],
        "RB": [
            "target_leader_in_group",
            "rush_vol_leader_in_group",
            "offense_top16",
            "ol_top16",
            "sos_top16",
        ],
        "WR": ["target_leader_in_group", "qb_top16", "offense_top16", "sos_top16"],
        "TE": ["te_top2_targets_in_group", "qb_top16", "offense_top16", "sos_top16"],
    }
    meta = dict(payload.get("meta") or {})
    meta["ol_included"] = False
    meta["ol_source"] = "sunday_sports_society_screenshots_player_checks_only"
    meta["offense_source"] = "sunday_sports_society_screenshots"
    meta["sos_source"] = "sunday_sports_society_screenshots"
    meta["check_source"] = "sunday_sports_society_screenshots"
    meta["check_source_note"] = (
        "Context checks transcribed from @SUNDAYSPORTSSOCIETY positional "
        "checklist graphics (PPR boards). Not derived from projections.db. "
        "Ja'Marr Chase and other CIN pass-catchers use the graphic's TOP 16 QB "
        "marks until the internal QB-context repair lands. Player OL check "
        "columns are included for QB/RB; team OL unit ranks are still unavailable."
    )
    meta["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["scoring_flavor"] = "ppr"
    meta["volume_caveat"] = (
        "Checks transcribed from @SUNDAYSPORTSSOCIETY checklist graphics "
        "(team target / rush leaders, top-16 QB, offense, O-line, SOS). "
        "Not computed from our projection model."
    )
    meta["criteria_labels"] = {
        "pass_att_top16": "TOP 16 PASS ATT",
        "rush_vol_top16": "TOP 16 RUSH ATT",
        "offense_top16": "TOP 16 OFFENSE",
        "ol_top16": "TOP 16 O-LINE",
        "sos_top16": "TOP 16 SOS",
        "target_leader_in_group": "TEAM TARGET LEADER",
        "rush_vol_leader_in_group": "RUSH ATT LEADER",
        "qb_top16": "TOP 16 QB",
        "te_top2_targets_in_group": "TOP-2 IN TEAM TARGETS",
    }
    meta["sss_matched"] = len(matched)
    meta["sss_missing"] = missing
    payload["meta"] = meta

    print(f"matched {len(matched)} / {len(WR_BOARD)+len(RB_BOARD)+len(QB_BOARD)+len(TE_BOARD)}")
    if missing:
        print("MISSING:")
        for row in missing:
            print(" ", row)
    chase = next(p for p in players if p.get("player_id") == "00-0036900")
    print("Ja'Marr Chase checks:", chase["checks"])

    if args.dry_run:
        return 0 if not missing else 1

    args.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.path}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
