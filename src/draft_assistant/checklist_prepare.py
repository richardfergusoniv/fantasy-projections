"""Build sealed draft-checklist JSON (market ranks + context checks).

Volume-leader flags come only from sealed ``team_stats_*`` history blocks.
OL / offense / SOS ranks require ``projections.db`` when available; offense
and SOS can fall back to the same nflverse sources ``src.db.load`` would
ingest. OL ranks have no substitute — omit them when the DB is absent.

Market order is half-PPR / 12-team ADP only (the single committed flavor).
Refresh ``comparison_{season}.json`` via ``compare_prepare`` before this
module when drafting live.
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
    "rush_vol_top16": "TOP 16 RUSH VOL",
    "offense_top16": "TOP 16 OFFENSE",
    "ol_top16": "TOP 16 O-LINE",
    "sos_top16": "TOP 16 SOS",
    "target_leader_in_group": "2025 TGT LEADER IN GROUP",
    "rush_vol_leader_in_group": "2025 RUSH VOL LEADER IN GROUP",
    "qb_top16": "TOP 16 QB",
    "te_top2_targets_in_group": "2025 TOP-2 TGT IN GROUP",
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


def rank_ascending(values: dict[str, float]) -> dict[str, int]:
    """1 = best (lowest value). Used for SOS (lower opponent EPA = easier)."""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
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


def build_checklist(
    *,
    season: int = 2026,
    team_stats_path: Path | None = None,
    comparison_path: Path | None = None,
    db_path: str | Path | None = None,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    team_stats_path = team_stats_path or DRAFT_DATA_DIR / f"team_stats_{season}.json"
    comparison_path = comparison_path or DRAFT_DATA_DIR / f"comparison_{season}.json"
    players = load_team_stats(team_stats_path)
    comparison = load_comparison(comparison_path)
    comparison_by_id = {
        str(row["player_id"]): row
        for row in (comparison.get("players") or [])
        if row.get("player_id") is not None
    }

    volume = volume_flags_for_players(players, top_n=top_n)
    enriched = assign_rank_tiers(players, comparison_by_id)

    db_file = Path(db_path or DB_PATH)
    ol_included = False
    sos_included = False
    offense_source = "missing"
    sos_source = "missing"
    ol_source = "missing"
    schedule_games = 0
    offense_ranks: dict[str, int] = {}
    ol_pass_ranks: dict[str, int] = {}
    ol_run_ranks: dict[str, int] = {}
    ol_unit_ranks: dict[str, int] = {}
    sos_pass_ranks: dict[str, int] = {}
    sos_rush_ranks: dict[str, int] = {}
    sos_unit_ranks: dict[str, int] = {}

    if db_file.is_file():
        conn = sqlite3.connect(str(db_file))
        try:
            offense_ranks = load_offense_ranks_from_db(conn, HISTORY_SEASON)
            offense_source = "projections.db:team_season_yardage_totals"
            ol_pass_ranks, ol_run_ranks, ol_unit_ranks = load_ol_ranks_from_db(
                conn, HISTORY_SEASON
            )
            if ol_unit_ranks:
                ol_included = True
                ol_source = "projections.db:ol_quality"
            schedule_games = count_2026_reg_schedules(conn, season)
            if schedule_games > 0:
                sos_pass_ranks, sos_rush_ranks, sos_unit_ranks = load_sos_ranks_from_db(
                    conn, season
                )
                if sos_unit_ranks:
                    sos_included = True
                    sos_source = "projections.db:team_season_opponent_strength"
        finally:
            conn.close()
    else:
        # Same nflverse underlying data as db.load — not a sum of the 778 pool.
        offense_ranks = load_offense_ranks_from_nflverse(HISTORY_SEASON)
        offense_source = "nflverse_pbp:team_pass_rush_yards"
        schedule_games = count_2026_reg_schedules_nflverse(season)
        if schedule_games > 0:
            sos_pass_ranks, sos_rush_ranks, sos_unit_ranks = load_sos_ranks_from_nflverse(
                season
            )
            if sos_unit_ranks:
                sos_included = True
                sos_source = "nflverse_schedules+pbp:opponent_strength"
        # OL has no substitute without ol_coefficients.
        ol_included = False
        ol_source = "unavailable_without_projections.db"

    criteria = criteria_for_meta(ol_included=ol_included, sos_included=sos_included)

    teams_out: list[dict[str, Any]] = []
    for meta in TEAM_META:
        abbr = meta["abbr"]
        row: dict[str, Any] = {
            "abbr": abbr,
            "name": meta["name"],
            "offense_rank": offense_ranks.get(abbr),
        }
        if ol_included:
            row["ol_pass_rank"] = ol_pass_ranks.get(abbr)
            row["ol_run_rank"] = ol_run_ranks.get(abbr)
            row["ol_unit_rank"] = ol_unit_ranks.get(abbr)
        if sos_included:
            row["sos_pass_rank"] = sos_pass_ranks.get(abbr)
            row["sos_rush_rank"] = sos_rush_ranks.get(abbr)
            row["sos_unit_rank"] = sos_unit_ranks.get(abbr)
        teams_out.append(row)

    players_out: list[dict[str, Any]] = []
    for row in enriched:
        pid = row["player_id"]
        team = row.get("team")
        checks: dict[str, bool] = {}
        for key in criteria.get(row["position"], []):
            if key in ("offense_top16", "ol_top16", "sos_top16"):
                if not team:
                    checks[key] = False
                    continue
                if key == "offense_top16":
                    checks[key] = (offense_ranks.get(team) or 99) <= top_n
                elif key == "ol_top16":
                    checks[key] = (ol_unit_ranks.get(team) or 99) <= top_n
                else:
                    checks[key] = (sos_unit_ranks.get(team) or 99) <= top_n
            else:
                checks[key] = bool(volume.get(pid, {}).get(key, False))
        players_out.append(
            {
                **row,
                "checks": checks,
            }
        )

    market_meta = comparison.get("meta") or {}
    adp_meta = market_meta.get("adp") or {}
    ecr_meta = market_meta.get("ecr") or {}

    return {
        "season": season,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "top_n": top_n,
            "rank_source": "adp",
            "scoring_flavor": "half-ppr",
            "team_count": int(adp_meta.get("teams") or 12),
            "market_as_of": {
                "adp_start": adp_meta.get("start_date"),
                "adp_end": adp_meta.get("end_date"),
                "ecr_scrape": ecr_meta.get("scrape_date"),
                "scoring": adp_meta.get("scoring") or "half-ppr",
                "teams": int(adp_meta.get("teams") or 12),
                "comparison_generated_at": market_meta.get("generated_at"),
                "matched_adp": market_meta.get("matched_adp"),
                "matched_ecr": market_meta.get("matched_ecr"),
            },
            "sos_included": sos_included,
            "schedule_2026_reg_games": schedule_games,
            "ol_included": ol_included,
            "offense_source": offense_source,
            "sos_source": sos_source,
            "ol_source": ol_source,
            "volume_caveat": VOLUME_CAVEAT,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draft checklist JSON")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument(
        "--db-path",
        default=None,
        help="projections.db path (default: configured DB_PATH)",
    )
    args = parser.parse_args()
    path = export_checklist(
        args.season,
        out_path=Path(args.out) if args.out else None,
        db_path=args.db_path,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
