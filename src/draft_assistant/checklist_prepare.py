"""Build sealed draft-checklist JSON from Sunday Sports Society screenshots.

Player order and non-OL context checks come from
``draft_assistant/data/screenshot_checklist_{season}.json`` (transcribed from
the SSS checklist screenshots). QB/RB ``ol_top16`` is overwritten from
``ol_unit_ranks_{season}.json`` (the O-line unit rankings screenshot) — not
from the SSS OL column and not from ``projections.db``.

Identity fields (player_id / team) are joined from sealed ``team_stats_*`` by
name so drafted toggles still work. ADP/ECR/our offense-SOS ranks are not used
for checklist order or checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import DB_PATH
from src.team_stats.prepare import TEAM_META

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DATA_DIR = REPO_ROOT / "draft_assistant" / "data"

HISTORY_SEASON = 2025
TOP_N = 16

# Mirrors src.projection.data_prep.team_season_opponent_strength.
_FRANCHISE_CODE_FIX = {"OAK": "LV", "SD": "LAC"}

VOLUME_CAVEAT = (
    "2025 volume leader within this team's 2026 skill group; history has no "
    "team field (volume may have been earned elsewhere). Projected pool only — "
    "departed/retired players are missing. Rush volume uses rushing_yards "
    "because history has no carries column."
)

CRITERIA_BASE: dict[str, list[str]] = {
    "QB": [
        "pass_att_top16",
        "rush_vol_top16",
        "offense_top16",
        "ol_top16",
        "sos_top16",
    ],
    "RB": [
        "target_leader_in_group",
        "rush_vol_leader_in_group",
        "offense_top16",
        "ol_top16",
        "sos_top16",
    ],
    "WR": [
        "target_leader_in_group",
        "qb_top16",
        "offense_top16",
        "sos_top16",
    ],
    "TE": [
        "te_top2_targets_in_group",
        "qb_top16",
        "offense_top16",
        "sos_top16",
    ],
}

CRITERIA_LABELS: dict[str, str] = {
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

# Screenshot lettering → sealed team_stats display_name.
SCREENSHOT_NAME_ALIASES: dict[str, str] = {
    "Michael Pittman Jr.": "Michael Pittman",
    "DeZhaun Stribling": "De'Zhaun Stribling",
    "CJ Stroud": "C.J. Stroud",
    "James Cook III": "James Cook",
    "Denry Henry": "Derrick Henry",
    "Travis Etienne Jr.": "Travis Etienne",
    "Kenneth Gainwell": "Kenny Gainwell",
    "Sam Laporta": "Sam LaPorta",
    "Harold Fannin": "Harold Fannin Jr.",
    "TJ Hockenson": "T.J. Hockenson",
    "Oronde Gadsden": "Oronde Gadsden II",
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _history_2025(player: dict[str, Any]) -> dict[str, Any] | None:
    for row in player.get("history") or []:
        if int(row.get("season") or 0) == HISTORY_SEASON:
            return row
    return None


def load_team_stats(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload.get("players") or [])


def load_comparison(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def rank_descending(values: dict[str, float]) -> dict[str, int]:
    """1 = best (highest value). Ties share the minimum rank."""
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    for index, (key, _) in enumerate(ordered, start=1):
        ranks[key] = index
    return ranks


def volume_flags_for_players(
    players: list[dict[str, Any]],
    *,
    top_n: int = TOP_N,
) -> dict[str, dict[str, bool]]:
    """Compute per-player volume/QB checks from sealed history only."""
    by_id = {str(p["player_id"]): p for p in players if p.get("player_id")}

    hist: dict[str, dict[str, Any]] = {}
    for pid, player in by_id.items():
        row = _history_2025(player)
        if row is not None:
            hist[pid] = row

    # League-wide QB top-16 by 2025 pass attempts / rush yards.
    qb_pass: dict[str, float] = {}
    qb_rush: dict[str, float] = {}
    qb_fp: dict[str, float] = {}
    for pid, player in by_id.items():
        if player.get("position") != "QB":
            continue
        row = hist.get(pid)
        if not row:
            continue
        attempts = _num(row.get("attempts")) or 0.0
        rush_yds = _num(row.get("rushing_yards")) or 0.0
        fp = _num(row.get("fantasy_pts_season")) or 0.0
        qb_pass[pid] = attempts
        qb_rush[pid] = rush_yds
        qb_fp[pid] = fp

    qb_pass_ok = {pid for pid, rank in rank_descending(qb_pass).items() if rank <= top_n}
    qb_rush_ok = {pid for pid, rank in rank_descending(qb_rush).items() if rank <= top_n}
    # Team QB quality for WR/TE/RB: rank QB1 on each 2026 team by 2025 FP.
    team_qb1: dict[str, tuple[str, float]] = {}
    for pid, player in by_id.items():
        if player.get("position") != "QB":
            continue
        team = player.get("team")
        if not team:
            continue
        fp = qb_fp.get(pid, 0.0)
        prev = team_qb1.get(team)
        if prev is None or fp > prev[1]:
            team_qb1[team] = (pid, fp)
    qb1_fp = {team: fp for team, (_pid, fp) in team_qb1.items()}
    qb1_ranks = rank_descending(qb1_fp)
    teams_qb_top16 = {team for team, rank in qb1_ranks.items() if rank <= top_n}

    # Group leaders by 2026 team + position pool.
    flags: dict[str, dict[str, bool]] = {pid: {} for pid in by_id}

    # Target leaders: among WR+RB on team (WR/RB target leader) and TE top-2.
    teams = sorted({p.get("team") for p in by_id.values() if p.get("team")})
    for team in teams:
        skill = [
            pid
            for pid, p in by_id.items()
            if p.get("team") == team and p.get("position") in ("WR", "RB", "TE")
        ]
        wr_rb = [pid for pid in skill if by_id[pid].get("position") in ("WR", "RB")]
        tes = [pid for pid in skill if by_id[pid].get("position") == "TE"]
        rbs = [pid for pid in skill if by_id[pid].get("position") == "RB"]

        def _targets(pid: str) -> float:
            return _num((hist.get(pid) or {}).get("targets")) or 0.0

        def _rush_vol(pid: str) -> float:
            return _num((hist.get(pid) or {}).get("rushing_yards")) or 0.0

        if wr_rb:
            best_tgt = max(wr_rb, key=_targets)
            best_tgt_val = _targets(best_tgt)
            for pid in wr_rb:
                flags[pid]["target_leader_in_group"] = (
                    pid == best_tgt and best_tgt_val > 0
                )
        if rbs:
            best_rush = max(rbs, key=_rush_vol)
            best_rush_val = _rush_vol(best_rush)
            for pid in rbs:
                flags[pid]["rush_vol_leader_in_group"] = (
                    pid == best_rush and best_rush_val > 0
                )
        if tes:
            ordered_te = sorted(tes, key=_targets, reverse=True)
            top2 = set()
            for pid in ordered_te[:2]:
                if _targets(pid) > 0:
                    top2.add(pid)
            for pid in tes:
                flags[pid]["te_top2_targets_in_group"] = pid in top2

    for pid, player in by_id.items():
        pos = player.get("position")
        team = player.get("team")
        if pos == "QB":
            flags[pid]["pass_att_top16"] = pid in qb_pass_ok
            flags[pid]["rush_vol_top16"] = pid in qb_rush_ok
        if pos in ("WR", "TE", "RB") and team:
            flags[pid]["qb_top16"] = team in teams_qb_top16

    return flags


def assign_rank_tiers(
    players: list[dict[str, Any]],
    comparison_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach ADP/ECR/prior_pts and sort keys; return enriched player dicts."""
    enriched: list[dict[str, Any]] = []
    for player in players:
        pid = str(player["player_id"])
        pos = str(player.get("position") or "")
        if pos not in CRITERIA_BASE:
            continue
        market = comparison_by_id.get(pid) or {}
        hist = _history_2025(player) or {}
        adp = _num(market.get("adp"))
        ecr = _num(market.get("ecr"))
        prior_pts = _num(hist.get("fantasy_pts_season"))
        if adp is not None:
            rank_tier = "adp"
        elif ecr is not None:
            rank_tier = "ecr"
        elif prior_pts is not None:
            rank_tier = "prior_pts"
        else:
            rank_tier = "none"
        enriched.append(
            {
                "player_id": pid,
                "name": player.get("display_name") or player.get("name") or pid,
                "position": pos,
                "team": player.get("team"),
                "adp": adp,
                "ecr": ecr,
                "prior_pts": prior_pts,
                "rank_tier": rank_tier,
            }
        )

    # Positional market rank within ADP→ECR→prior→none.
    tier_order = {"adp": 0, "ecr": 1, "prior_pts": 2, "none": 3}
    for pos in CRITERIA_BASE:
        cohort = [row for row in enriched if row["position"] == pos]
        cohort.sort(
            key=lambda row: (
                tier_order[row["rank_tier"]],
                row["adp"] if row["rank_tier"] == "adp" else 0.0,
                row["ecr"] if row["rank_tier"] == "ecr" else 0.0,
                -(row["prior_pts"] or 0.0) if row["rank_tier"] == "prior_pts" else 0.0,
                row["name"],
            )
        )
        for index, row in enumerate(cohort, start=1):
            row["pos_market_rank"] = index
            # First player off the market board (no ADP and no ECR).
            row["unranked_break"] = row["rank_tier"] in ("prior_pts", "none") and (
                index == 1
                or cohort[index - 2]["rank_tier"] in ("adp", "ecr")
            )

    enriched.sort(
        key=lambda row: (
            {"QB": 0, "RB": 1, "WR": 2, "TE": 3}.get(row["position"], 9),
            row.get("pos_market_rank") or 9999,
        )
    )
    return enriched


def db_has_tables(db_file: Path, tables: tuple[str, ...]) -> bool:
    """True when ``db_file`` is a readable SQLite DB containing every table.

    ``projections.db`` is a zero-byte placeholder in some deploy environments;
    ``sqlite3.connect`` happily opens it, so probe for the tables the DB-backed
    rank loaders need instead of trusting the file's existence.
    """
    if not db_file.is_file() or db_file.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(str(db_file))
    except sqlite3.Error:
        return False
    try:
        present = {
            str(row[0])
            for row in conn.execute("select name from sqlite_master where type = 'table'")
        }
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return all(table in present for table in tables)


def count_2026_reg_schedules(conn: sqlite3.Connection, season: int) -> int:
    row = conn.execute(
        "select count(*) from schedules where season = ? and game_type = 'REG'",
        (season,),
    ).fetchone()
    return int(row[0] if row else 0)


def load_offense_ranks_from_db(
    conn: sqlite3.Connection, history_season: int = HISTORY_SEASON
) -> dict[str, int]:
    from src.projection.data_prep import team_season_yardage_totals

    pass_yds = team_season_yardage_totals(conn, seasons=[history_season])
    # Rush yards: same REG pbp source as yardage totals.
    rush = pd.read_sql(
        f"""
        select season, posteam as team, sum(rushing_yards) as team_rushing_yards
        from pbp
        where season = {history_season} and season_type = 'REG'
          and posteam is not null and rush_attempt = 1
        group by season, posteam
        """,
        conn,
    )
    merged = pass_yds.merge(rush, on=["season", "team"], how="outer").fillna(0.0)
    totals = {
        str(row.team): float(row.team_passing_yards) + float(row.team_rushing_yards)
        for row in merged.itertuples(index=False)
    }
    return rank_descending(totals)


def load_ol_ranks_from_db(
    conn: sqlite3.Connection, history_season: int = HISTORY_SEASON
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    from src.projection.ol_quality import team_season_ol_quality

    ol = team_season_ol_quality(conn, seasons=[history_season])
    ol = ol[ol["season"] == history_season].copy()
    if ol.empty:
        return {}, {}, {}
    pass_scores = {
        str(row.team): float(row.ol_pass_protection_score)
        for row in ol.itertuples(index=False)
        if pd.notna(row.ol_pass_protection_score)
    }
    run_scores = {
        str(row.team): float(row.ol_run_blocking_score)
        for row in ol.itertuples(index=False)
        if pd.notna(row.ol_run_blocking_score)
    }
    unit_scores = {
        team: 0.55 * pass_scores.get(team, 0.0) + 0.45 * run_scores.get(team, 0.0)
        for team in set(pass_scores) | set(run_scores)
    }
    return rank_descending(pass_scores), rank_descending(run_scores), rank_descending(unit_scores)


def sealed_ol_ranks_path(season: int = 2026) -> Path:
    return DRAFT_DATA_DIR / f"ol_unit_ranks_{season}.json"


def load_sealed_ol_unit_ranks(path: Path) -> dict[str, int]:
    """Load manual unit ranks (1 = best) from the sealed screenshot board."""
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    raw = payload.get("unit_ranks") or {}
    ranks: dict[str, int] = {}
    for abbr, rank in raw.items():
        try:
            ranks[str(abbr)] = int(rank)
        except (TypeError, ValueError):
            continue
    return ranks


def apply_ol_unit_ranks(
    payload: dict[str, Any],
    unit_ranks: dict[str, int],
    *,
    ol_source: str,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Inject / replace OL unit ranks and QB/RB ``ol_top16`` checks in place."""
    if not unit_ranks:
        return payload

    top = int(top_n if top_n is not None else payload.get("meta", {}).get("top_n") or TOP_N)
    meta = dict(payload.get("meta") or {})
    meta["ol_included"] = True
    meta["ol_source"] = ol_source
    labels = dict(meta.get("criteria_labels") or {})
    labels["ol_top16"] = CRITERIA_LABELS["ol_top16"]
    meta["criteria_labels"] = labels
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["meta"] = meta

    criteria = {
        pos: list(keys) for pos, keys in (payload.get("criteria_by_position") or {}).items()
    }
    for pos, keys in CRITERIA_BASE.items():
        if "ol_top16" not in keys:
            continue
        current = criteria.setdefault(pos, [])
        if "ol_top16" not in current:
            # Keep OL next to offense when present; otherwise append.
            if "offense_top16" in current:
                insert_at = current.index("offense_top16") + 1
                current.insert(insert_at, "ol_top16")
            else:
                current.append("ol_top16")
    payload["criteria_by_position"] = criteria

    teams_out: list[dict[str, Any]] = []
    for team in payload.get("teams") or []:
        row = dict(team)
        abbr = str(row.get("abbr") or "")
        if abbr in unit_ranks:
            row["ol_unit_rank"] = unit_ranks[abbr]
        teams_out.append(row)
    payload["teams"] = teams_out

    players_out: list[dict[str, Any]] = []
    for player in payload.get("players") or []:
        row = dict(player)
        checks = dict(row.get("checks") or {})
        pos = str(row.get("position") or "")
        if "ol_top16" in (criteria.get(pos) or []):
            team = row.get("team")
            if not team:
                checks["ol_top16"] = False
            else:
                checks["ol_top16"] = (unit_ranks.get(str(team)) or 99) <= top
        row["checks"] = checks
        players_out.append(row)
    payload["players"] = players_out
    return payload


def load_sos_ranks_from_db(
    conn: sqlite3.Connection, season: int
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    from src.projection.data_prep import team_season_opponent_strength

    opp = team_season_opponent_strength(conn, seasons=[season - 1, season])
    opp = opp[opp["season"] == season].copy()
    if opp.empty:
        return {}, {}, {}
    pass_vals = {
        str(row.team): float(row.opp_def_pass_epa_prior)
        for row in opp.itertuples(index=False)
        if pd.notna(row.opp_def_pass_epa_prior)
    }
    rush_vals = {
        str(row.team): float(row.opp_def_rush_epa_prior)
        for row in opp.itertuples(index=False)
        if pd.notna(row.opp_def_rush_epa_prior)
    }
    combined = {
        team: 0.5 * pass_vals.get(team, 0.0) + 0.5 * rush_vals.get(team, 0.0)
        for team in set(pass_vals) | set(rush_vals)
    }
    # Lower opponent EPA allowed = tougher defense = harder SOS.
    # Infographic "TOP 16 SOS" means favorable schedule → easier opponents
    # → higher opponent EPA allowed. Rank ascending on difficulty so rank 1
    # is easiest (highest EPA allowed). Invert by ranking descending on EPA.
    return (
        rank_descending(pass_vals),
        rank_descending(rush_vals),
        rank_descending(combined),
    )


def load_offense_ranks_from_nflverse(history_season: int = HISTORY_SEASON) -> dict[str, int]:
    import nflreadpy as nfl

    pbp = nfl.load_pbp([history_season])
    if hasattr(pbp, "to_pandas"):
        pbp = pbp.to_pandas()
    reg = pbp[pbp["season_type"] == "REG"]
    pass_yds = (
        reg[reg["pass_attempt"] == 1]
        .groupby("posteam")["passing_yards"]
        .sum()
    )
    rush_yds = (
        reg[reg["rush_attempt"] == 1]
        .groupby("posteam")["rushing_yards"]
        .sum()
    )
    totals = (pass_yds.fillna(0) + rush_yds.fillna(0)).to_dict()
    return rank_descending({str(k): float(v) for k, v in totals.items() if k})


def count_2026_reg_schedules_nflverse(season: int) -> int:
    import nflreadpy as nfl

    sched = nfl.load_schedules([season])
    if hasattr(sched, "to_pandas"):
        sched = sched.to_pandas()
    return int((sched["game_type"] == "REG").sum())


def load_sos_ranks_from_nflverse(season: int) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Mirror team_season_opponent_strength using nflverse schedules + prior pbp."""
    import nflreadpy as nfl

    sched = nfl.load_schedules([season])
    if hasattr(sched, "to_pandas"):
        sched = sched.to_pandas()
    sched = sched[sched["game_type"] == "REG"].copy()
    if sched.empty:
        return {}, {}, {}
    # Same normalization team_season_opponent_strength applies, so the DB and
    # nflverse paths agree on historical franchise codes.
    sched["home_team"] = sched["home_team"].replace(_FRANCHISE_CODE_FIX)
    sched["away_team"] = sched["away_team"].replace(_FRANCHISE_CODE_FIX)

    pbp = nfl.load_pbp([season - 1])
    if hasattr(pbp, "to_pandas"):
        pbp = pbp.to_pandas()
    reg = pbp[pbp["season_type"] == "REG"]
    pass_epa = (
        reg[reg["pass_attempt"] == 1]
        .groupby("defteam")["epa"]
        .mean()
        .rename("def_pass_epa_allowed")
    )
    rush_epa = (
        reg[reg["rush_attempt"] == 1]
        .groupby("defteam")["epa"]
        .mean()
        .rename("def_rush_epa_allowed")
    )
    def_epa = pd.concat([pass_epa, rush_epa], axis=1).reset_index().rename(
        columns={"defteam": "opponent"}
    )

    home = sched.rename(columns={"home_team": "team", "away_team": "opponent"})[
        ["team", "opponent"]
    ]
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent"})[
        ["team", "opponent"]
    ]
    long = pd.concat([home, away], ignore_index=True)
    merged = long.merge(def_epa, on="opponent", how="left")
    agg = (
        merged.groupby("team")[["def_pass_epa_allowed", "def_rush_epa_allowed"]]
        .mean()
        .reset_index()
    )
    pass_vals = {
        str(row.team): float(row.def_pass_epa_allowed)
        for row in agg.itertuples(index=False)
        if pd.notna(row.def_pass_epa_allowed)
    }
    rush_vals = {
        str(row.team): float(row.def_rush_epa_allowed)
        for row in agg.itertuples(index=False)
        if pd.notna(row.def_rush_epa_allowed)
    }
    combined = {
        team: 0.5 * pass_vals.get(team, 0.0) + 0.5 * rush_vals.get(team, 0.0)
        for team in set(pass_vals) | set(rush_vals)
    }
    return (
        rank_descending(pass_vals),
        rank_descending(rush_vals),
        rank_descending(combined),
    )


def criteria_for_meta(*, ol_included: bool, sos_included: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pos, keys in CRITERIA_BASE.items():
        filtered = []
        for key in keys:
            if key == "ol_top16" and not ol_included:
                continue
            if key == "sos_top16" and not sos_included:
                continue
            filtered.append(key)
        out[pos] = filtered
    return out


def screenshot_checklist_path(season: int = 2026) -> Path:
    return DRAFT_DATA_DIR / f"screenshot_checklist_{season}.json"


def load_screenshot_board(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def resolve_screenshot_name(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def index_team_stats_by_pos_name(
    players: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for player in players:
        pos = str(player.get("position") or "")
        name = str(player.get("display_name") or player.get("name") or "")
        if not pos or not name:
            continue
        out.setdefault(pos, {})[name] = player
    return out


def _history_prior_pts(player: dict[str, Any]) -> float | None:
    hist = _history_2025(player) or {}
    return _num(hist.get("fantasy_pts_season"))


def build_checklist(
    *,
    season: int = 2026,
    team_stats_path: Path | None = None,
    comparison_path: Path | None = None,
    db_path: str | Path | None = None,
    top_n: int = TOP_N,
    screenshot_path: Path | None = None,
    ol_ranks_path: Path | None = None,
) -> dict[str, Any]:
    """Build checklist from sealed SSS screenshots + O-line unit ranks.

    ``comparison_path`` / ``db_path`` remain on the signature for CLI
    compatibility but are not used for ranks or non-OL checks.
    """
    del comparison_path, db_path

    team_stats_path = team_stats_path or DRAFT_DATA_DIR / f"team_stats_{season}.json"
    board_path = screenshot_path or screenshot_checklist_path(season)
    if not board_path.is_file():
        raise FileNotFoundError(f"screenshot checklist board not found: {board_path}")

    board = load_screenshot_board(board_path)
    aliases = dict(SCREENSHOT_NAME_ALIASES)
    aliases.update({str(k): str(v) for k, v in (board.get("name_aliases") or {}).items()})

    roster = load_team_stats(team_stats_path)
    by_pos_name = index_team_stats_by_pos_name(roster)

    ranks_path = ol_ranks_path or sealed_ol_ranks_path(season)
    if not ranks_path.is_file():
        raise FileNotFoundError(f"sealed OL ranks not found: {ranks_path}")
    ol_unit_ranks = load_sealed_ol_unit_ranks(ranks_path)
    if not ol_unit_ranks:
        raise ValueError(f"sealed OL ranks empty: {ranks_path}")

    criteria = criteria_for_meta(ol_included=True, sos_included=True)

    teams_out: list[dict[str, Any]] = []
    for meta in TEAM_META:
        abbr = meta["abbr"]
        teams_out.append(
            {
                "abbr": abbr,
                "name": meta["name"],
                "offense_rank": None,
                "ol_pass_rank": None,
                "ol_run_rank": None,
                "ol_unit_rank": ol_unit_ranks.get(abbr),
                "sos_pass_rank": None,
                "sos_rush_rank": None,
                "sos_unit_rank": None,
            }
        )

    players_out: list[dict[str, Any]] = []
    missing: list[str] = []
    positions = board.get("positions") or {}
    for pos in ("QB", "RB", "WR", "TE"):
        rows = list(positions.get(pos) or [])
        rows.sort(key=lambda row: int(row.get("rank") or 9999))
        pool = by_pos_name.get(pos) or {}
        for row in rows:
            raw_name = str(row.get("name") or "")
            resolved = resolve_screenshot_name(raw_name, aliases)
            identity = pool.get(resolved)
            if identity is None:
                missing.append(f"{pos}:{raw_name}")
                continue
            team = identity.get("team")
            shot_checks = dict(row.get("checks") or {})
            checks: dict[str, bool] = {}
            for key in criteria.get(pos, []):
                if key == "ol_top16":
                    checks[key] = bool(team) and (
                        (ol_unit_ranks.get(str(team)) or 99) <= top_n
                    )
                elif key in shot_checks:
                    checks[key] = bool(shot_checks[key])
                else:
                    checks[key] = False
            players_out.append(
                {
                    "player_id": str(identity["player_id"]),
                    "name": identity.get("display_name") or resolved,
                    "position": pos,
                    "team": team,
                    "adp": None,
                    "ecr": None,
                    "prior_pts": _history_prior_pts(identity),
                    "rank_tier": "screenshot",
                    "pos_market_rank": int(row["rank"]),
                    "unranked_break": False,
                    "checks": checks,
                }
            )

    if missing:
        raise ValueError(
            "screenshot names not found in team_stats: " + ", ".join(missing)
        )

    return {
        "season": season,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "top_n": top_n,
            "rank_source": "screenshot",
            "scoring_flavor": "ppr",
            "team_count": 12,
            "market_as_of": {
                "source": board.get("source") or "sunday_sports_society_screenshots",
                "as_of": board.get("as_of"),
                "scoring": "ppr",
                "teams": 12,
            },
            "sos_included": True,
            "schedule_2026_reg_games": None,
            "ol_included": True,
            "offense_source": "screenshot_checklist",
            "sos_source": "screenshot_checklist",
            "ol_source": f"sealed:{ranks_path.name}",
            "rank_board_source": f"sealed:{board_path.name}",
            "volume_caveat": (
                "Checks and positional ranks transcribed from SSS checklist "
                "screenshots. QB/RB TOP 16 O-LINE uses the sealed O-line unit "
                "rankings screenshot, not the SSS OL column."
            ),
            "criteria_labels": {
                key: CRITERIA_LABELS[key]
                for keys in criteria.values()
                for key in keys
            },
        },
        "criteria_by_position": criteria,
        "teams": teams_out,
        "players": players_out,
    }


def export_checklist(
    season: int = 2026,
    *,
    out_path: Path | None = None,
    db_path: str | Path | None = None,
) -> Path:
    payload = build_checklist(season=season, db_path=db_path)
    destination = out_path or DRAFT_DATA_DIR / f"draft_checklist_{season}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return destination


def patch_checklist_ol(
    season: int = 2026,
    *,
    checklist_path: Path | None = None,
    ol_ranks_path: Path | None = None,
) -> Path:
    """Rewrite OL ranks/checks on an existing checklist without rebuilding offense/SOS."""
    destination = checklist_path or DRAFT_DATA_DIR / f"draft_checklist_{season}.json"
    ranks_path = ol_ranks_path or sealed_ol_ranks_path(season)
    if not destination.is_file():
        raise FileNotFoundError(f"checklist not found: {destination}")
    if not ranks_path.is_file():
        raise FileNotFoundError(f"sealed OL ranks not found: {ranks_path}")
    with destination.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    unit_ranks = load_sealed_ol_unit_ranks(ranks_path)
    apply_ol_unit_ranks(
        payload,
        unit_ranks,
        ol_source=f"sealed:{ranks_path.name}",
    )
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draft checklist JSON")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument(
        "--db-path",
        default=None,
        help="projections.db path (default: configured DB_PATH)",
    )
    parser.add_argument(
        "--patch-ol-only",
        action="store_true",
        help="Inject sealed ol_unit_ranks into the existing checklist JSON only",
    )
    args = parser.parse_args()
    if args.patch_ol_only:
        path = patch_checklist_ol(
            args.season,
            checklist_path=Path(args.out) if args.out else None,
        )
    else:
        path = export_checklist(
            args.season,
            out_path=Path(args.out) if args.out else None,
            db_path=args.db_path,
        )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
