"""Build sealed draft-checklist JSON with Vegas / Sharp SOS ranks.

Context columns are ranks (1 = best), not boolean checks.

- Player volume: Vegas season O/U yards/receptions (attempts/targets are not
  publicly posted). QB volume uses total yards (pass + rush) and rush yards;
  skill volume uses receptions and total yards (rush + receiving).
- Team offense: Vegas-implied points scored and total yards, ranked 1-32.
- O-line: sealed ol_unit_ranks_{season}.json.
- SOS: Sharp Football Analysis fantasy SOS (pass for QB/WR/TE, rush for RB).

Board covers every rostered QB/RB/WR/TE in team_stats_{season}.json.
Positional order remains market ADP/ECR average.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.draft_assistant.market_adp import (
    average_market_value,
    fetch_market_maps,
    market_components_for_player,
    normalize_player_name,
)
from src.team_stats.prepare import TEAM_META

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DATA_DIR = REPO_ROOT / "draft_assistant" / "data"

HISTORY_SEASON = 2025
TOP_N = 16

VOLUME_CAVEAT = (
    "Season-long attempt/target O/Us are not posted on public boards "
    "(DraftKings/FanDuel/BettingPros/VegasInsider/Kalshi/Polymarket checked). "
    "Volume ranks therefore use Vegas yards/receptions: QB total yards "
    "(pass + rush) and rush yards; skill receptions and total yards "
    "(rush + receiving). Team offense ranks combine Vegas-implied points "
    "scored and total yards. O-line ranks stay on the sealed screenshot board. "
    "SOS is Sharp Football Analysis fantasy SOS (passing for QB/WR/TE, "
    "rushing for RB). Board includes every rostered QB/RB/WR/TE."
)

CRITERIA_BASE: dict[str, list[str]] = {
    "QB": [
        "total_yds_rank",
        "rush_yds_rank",
        "offense_pts_rank",
        "offense_yds_rank",
        "ol_rank",
        "sos_rank",
    ],
    "RB": [
        "rec_rank",
        "total_yds_rank",
        "offense_pts_rank",
        "offense_yds_rank",
        "ol_rank",
        "sos_rank",
    ],
    "WR": [
        "rec_rank",
        "qb_rank",
        "offense_pts_rank",
        "offense_yds_rank",
        "sos_rank",
    ],
    "TE": [
        "rec_rank",
        "qb_rank",
        "offense_pts_rank",
        "offense_yds_rank",
        "sos_rank",
    ],
}

CRITERIA_LABELS: dict[str, str] = {
    "total_yds_rank": "TOTAL YDS RANK",
    "rush_yds_rank": "RUSH YDS RANK",
    "offense_pts_rank": "OFFENSE PTS RANK",
    "offense_yds_rank": "OFFENSE YDS RANK",
    "ol_rank": "O-LINE RANK",
    "sos_rank": "SOS RANK",
    "rec_rank": "RECEPTIONS RANK",
    "qb_rank": "QB RANK",
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def rank_descending(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], str(item[0])))
    return {key: index for index, (key, _) in enumerate(ordered, start=1)}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_team_stats(path: Path) -> list[dict[str, Any]]:
    return list(load_json(path).get("players") or [])


def sealed_ol_ranks_path(season: int = 2026) -> Path:
    return DRAFT_DATA_DIR / f"ol_unit_ranks_{season}.json"


def load_sealed_ol_unit_ranks(path: Path) -> dict[str, int]:
    raw = load_json(path).get("unit_ranks") or {}
    ranks: dict[str, int] = {}
    for abbr, rank in raw.items():
        try:
            ranks[str(abbr)] = int(rank)
        except (TypeError, ValueError):
            continue
    return ranks


def load_sharp_sos(path: Path) -> dict[str, dict[str, int]]:
    payload = load_json(path)
    return {
        "passing": {str(k): int(v) for k, v in (payload.get("passing_ranks") or {}).items()},
        "rushing": {str(k): int(v) for k, v in (payload.get("rushing_ranks") or {}).items()},
    }


def _history_prior_pts(player: dict[str, Any]) -> float | None:
    for row in player.get("history") or []:
        if int(row.get("season") or 0) == HISTORY_SEASON:
            return _num(
                row.get("fantasy_pts_season")
                or row.get("fantasy_pts")
                or row.get("fantasy_points")
            )
    return _num(player.get("fantasy_pts_season"))


def _qb_fantasy_from_markets(markets: dict[str, Any]) -> float:
    return (
        (_num(markets.get("pass_yards")) or 0.0) / 25.0
        + (_num(markets.get("pass_tds")) or 0.0) * 4.0
        + (_num(markets.get("rush_yards")) or 0.0) / 10.0
        + (_num(markets.get("rush_tds")) or 0.0) * 6.0
    )


def criteria_for_meta(*, ol_included: bool, sos_included: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pos, keys in CRITERIA_BASE.items():
        filtered: list[str] = []
        for key in keys:
            if key == "ol_rank" and not ol_included:
                continue
            if key == "sos_rank" and not sos_included:
                continue
            filtered.append(key)
        out[pos] = filtered
    return out


def apply_ol_unit_ranks(
    payload: dict[str, Any],
    unit_ranks: dict[str, int],
    *,
    ol_source: str,
    top_n: int | None = None,
) -> dict[str, Any]:
    if not unit_ranks:
        return payload
    top = int(top_n if top_n is not None else (payload.get("meta") or {}).get("top_n") or TOP_N)

    meta = dict(payload.get("meta") or {})
    meta["ol_included"] = True
    meta["ol_source"] = ol_source
    labels = dict(meta.get("criteria_labels") or {})
    labels["ol_rank"] = CRITERIA_LABELS["ol_rank"]
    meta["criteria_labels"] = labels
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["meta"] = meta

    criteria = {
        pos: list(keys) for pos, keys in (payload.get("criteria_by_position") or {}).items()
    }
    for pos, keys in CRITERIA_BASE.items():
        if "ol_rank" not in keys:
            continue
        current = criteria.setdefault(pos, [])
        if "ol_rank" not in current:
            if "offense_yds_rank" in current:
                current.insert(current.index("offense_yds_rank") + 1, "ol_rank")
            else:
                current.append("ol_rank")
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
        ranks = dict(row.get("ranks") or {})
        checks = dict(row.get("checks") or {})
        pos = str(row.get("position") or "")
        if "ol_rank" in (criteria.get(pos) or []):
            team = row.get("team")
            rank = unit_ranks.get(str(team)) if team else None
            ranks["ol_rank"] = rank
            checks["ol_rank"] = bool(rank is not None and rank <= top)
        row["ranks"] = ranks
        row["checks"] = checks
        players_out.append(row)
    payload["players"] = players_out
    return payload


def build_checklist(
    *,
    season: int = 2026,
    team_stats_path: Path | None = None,
    comparison_path: Path | None = None,
    db_path: str | Path | None = None,
    top_n: int = TOP_N,
    vegas_path: Path | None = None,
    sos_path: Path | None = None,
    ol_ranks_path: Path | None = None,
) -> dict[str, Any]:
    del db_path

    team_stats_path = team_stats_path or DRAFT_DATA_DIR / f"team_stats_{season}.json"
    comparison_path = comparison_path or DRAFT_DATA_DIR / f"comparison_{season}.json"
    vegas_path = vegas_path or DRAFT_DATA_DIR / f"vegas_consensus_{season}.json"
    sos_path = sos_path or DRAFT_DATA_DIR / f"sharp_fantasy_sos_{season}.json"
    ranks_path = ol_ranks_path or sealed_ol_ranks_path(season)

    for path, label in (
        (team_stats_path, "team_stats"),
        (comparison_path, "comparison"),
        (vegas_path, "vegas_consensus"),
        (sos_path, "sharp_fantasy_sos"),
        (ranks_path, "ol_unit_ranks"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    roster = load_team_stats(team_stats_path)
    vegas = load_json(vegas_path)
    sos = load_sharp_sos(sos_path)
    ol_unit_ranks = load_sealed_ol_unit_ranks(ranks_path)
    if not ol_unit_ranks:
        raise ValueError(f"sealed OL ranks empty: {ranks_path}")

    vegas_by_norm = {
        str(row.get("name_norm") or normalize_player_name(str(row.get("name") or ""))): row
        for row in vegas.get("players") or []
        if row.get("name")
    }
    team_offense = {
        str(row.get("abbr")): dict(row.get("markets") or {})
        for row in vegas.get("teams") or []
        if row.get("abbr")
    }

    espn_adp, ffc_adp, mfl_adp, ecr_by_id, market_meta = fetch_market_maps(
        comparison_path=comparison_path
    )
    criteria = criteria_for_meta(ol_included=True, sos_included=True)

    points_vals = {
        abbr: float(markets["points_scored"])
        for abbr, markets in team_offense.items()
        if _num(markets.get("points_scored")) is not None
    }
    yards_vals = {
        abbr: float(markets["total_yards"])
        for abbr, markets in team_offense.items()
        if _num(markets.get("total_yards")) is not None
    }
    offense_pts_ranks = rank_descending(points_vals)
    offense_yds_ranks = rank_descending(yards_vals)

    candidates: list[dict[str, Any]] = []
    for player in roster:
        pos = str(player.get("position") or "")
        if pos not in CRITERIA_BASE:
            continue
        team = player.get("team")
        if not team:
            continue
        display_name = str(player.get("display_name") or player.get("name") or "")
        player_id = str(player.get("player_id") or "")
        norm = normalize_player_name(display_name)
        markets = dict((vegas_by_norm.get(norm) or {}).get("markets") or {})

        season_stats = player.get("season") or {}
        if _num(markets.get("pass_yards")) is None and pos == "QB":
            fallback = _num(season_stats.get("passing_yards"))
            if fallback is not None:
                markets["pass_yards"] = fallback
        if _num(markets.get("rush_yards")) is None and pos in ("QB", "RB"):
            fallback = _num(season_stats.get("rushing_yards"))
            if fallback is not None:
                markets["rush_yards"] = fallback
        if _num(markets.get("rec_yards")) is None and pos in ("RB", "WR", "TE"):
            fallback = _num(season_stats.get("receiving_yards"))
            if fallback is not None:
                markets["rec_yards"] = fallback
        if _num(markets.get("receptions")) is None and pos in ("RB", "WR", "TE"):
            fallback = _num(season_stats.get("receptions"))
            if fallback is not None:
                markets["receptions"] = fallback

        components = market_components_for_player(
            position=pos,
            name=display_name,
            player_id=player_id,
            espn=espn_adp,
            ffc=ffc_adp,
            mfl=mfl_adp,
            ecr_by_id=ecr_by_id,
        )
        market_avg = average_market_value(components)
        candidates.append(
            {
                "player_id": player_id,
                "name": display_name,
                "name_norm": norm,
                "position": pos,
                "team": str(team),
                "adp": round(market_avg, 2) if market_avg is not None else None,
                "adp_espn": components.get("adp_espn"),
                "adp_ffc": components.get("adp_ffc"),
                "adp_mfl": components.get("adp_mfl"),
                "ecr": components.get("ecr"),
                "prior_pts": _history_prior_pts(player),
                "rank_tier": "market_avg" if market_avg is not None else "none",
                "market_avg": round(market_avg, 2) if market_avg is not None else None,
                "markets": markets,
            }
        )

    def _total_yards(row: dict[str, Any]) -> float | None:
        markets = row["markets"]
        if row["position"] == "QB":
            pass_yards = _num(markets.get("pass_yards"))
            rush_yards = _num(markets.get("rush_yards")) or 0.0
            if pass_yards is None and _num(markets.get("rush_yards")) is None:
                return None
            return (pass_yards or 0.0) + rush_yards
        rush_yards = _num(markets.get("rush_yards")) or 0.0
        rec_yards = _num(markets.get("rec_yards")) or 0.0
        if _num(markets.get("rush_yards")) is None and _num(markets.get("rec_yards")) is None:
            return None
        return rush_yards + rec_yards

    qb_total_yds = {
        row["player_id"]: float(total)
        for row in candidates
        if row["position"] == "QB" and (total := _total_yards(row)) is not None
    }
    qb_rush_yds = {
        row["player_id"]: float(row["markets"]["rush_yards"])
        for row in candidates
        if row["position"] == "QB" and _num(row["markets"].get("rush_yards")) is not None
    }
    rb_total_yds = {
        row["player_id"]: float(total)
        for row in candidates
        if row["position"] == "RB" and (total := _total_yards(row)) is not None
    }
    wr_rec = {
        row["player_id"]: float(row["markets"]["receptions"])
        for row in candidates
        if row["position"] == "WR" and _num(row["markets"].get("receptions")) is not None
    }
    rb_rec = {
        row["player_id"]: float(row["markets"]["receptions"])
        for row in candidates
        if row["position"] == "RB" and _num(row["markets"].get("receptions")) is not None
    }
    te_rec = {
        row["player_id"]: float(row["markets"]["receptions"])
        for row in candidates
        if row["position"] == "TE" and _num(row["markets"].get("receptions")) is not None
    }

    qb_total_yds_ranks = rank_descending(qb_total_yds)
    qb_rush_yds_ranks = rank_descending(qb_rush_yds)
    rb_total_yds_ranks = rank_descending(rb_total_yds)
    wr_rec_ranks = rank_descending(wr_rec)
    rb_rec_ranks = rank_descending(rb_rec)
    te_rec_ranks = rank_descending(te_rec)

    team_qb1_fp: dict[str, float] = {}
    for row in candidates:
        if row["position"] != "QB":
            continue
        fp = _num(row["markets"].get("fantasy_points"))
        if fp is None:
            fp = _qb_fantasy_from_markets(row["markets"])
        prev = team_qb1_fp.get(row["team"])
        if prev is None or fp > prev:
            team_qb1_fp[row["team"]] = float(fp)
    team_qb_ranks = rank_descending(team_qb1_fp)

    teams_out: list[dict[str, Any]] = []
    for meta in TEAM_META:
        abbr = meta["abbr"]
        teams_out.append(
            {
                "abbr": abbr,
                "name": meta["name"],
                "offense_pts_rank": offense_pts_ranks.get(abbr),
                "offense_yds_rank": offense_yds_ranks.get(abbr),
                "offense_rank": offense_pts_ranks.get(abbr),
                "ol_pass_rank": None,
                "ol_run_rank": None,
                "ol_unit_rank": ol_unit_ranks.get(abbr),
                "sos_pass_rank": sos["passing"].get(abbr),
                "sos_rush_rank": sos["rushing"].get(abbr),
                "sos_unit_rank": sos["passing"].get(abbr),
            }
        )

    players_out: list[dict[str, Any]] = []
    for pos in ("QB", "RB", "WR", "TE"):
        cohort = [row for row in candidates if row["position"] == pos]
        cohort.sort(
            key=lambda row: (
                row["market_avg"] is None,
                row["market_avg"] if row["market_avg"] is not None else 9999.0,
                row["name"],
            )
        )
        for index, row in enumerate(cohort, start=1):
            team = row["team"]
            ranks: dict[str, int | None] = {
                "offense_pts_rank": offense_pts_ranks.get(team),
                "offense_yds_rank": offense_yds_ranks.get(team),
            }
            if pos == "QB":
                ranks["total_yds_rank"] = qb_total_yds_ranks.get(row["player_id"])
                ranks["rush_yds_rank"] = qb_rush_yds_ranks.get(row["player_id"])
                ranks["ol_rank"] = ol_unit_ranks.get(team)
                ranks["sos_rank"] = sos["passing"].get(team)
            elif pos == "RB":
                ranks["rec_rank"] = rb_rec_ranks.get(row["player_id"])
                ranks["total_yds_rank"] = rb_total_yds_ranks.get(row["player_id"])
                ranks["ol_rank"] = ol_unit_ranks.get(team)
                ranks["sos_rank"] = sos["rushing"].get(team)
            elif pos == "WR":
                ranks["rec_rank"] = wr_rec_ranks.get(row["player_id"])
                ranks["qb_rank"] = team_qb_ranks.get(team)
                ranks["sos_rank"] = sos["passing"].get(team)
            else:
                ranks["rec_rank"] = te_rec_ranks.get(row["player_id"])
                ranks["qb_rank"] = team_qb_ranks.get(team)
                ranks["sos_rank"] = sos["passing"].get(team)

            allowed = list(criteria.get(pos) or [])
            ranks = {key: ranks.get(key) for key in allowed}
            checks = {
                key: bool(val is not None and val <= top_n) for key, val in ranks.items()
            }
            players_out.append(
                {
                    "player_id": row["player_id"],
                    "name": row["name"],
                    "position": pos,
                    "team": team,
                    "adp": row["adp"],
                    "adp_espn": row["adp_espn"],
                    "adp_ffc": row["adp_ffc"],
                    "adp_mfl": row["adp_mfl"],
                    "ecr": row["ecr"],
                    "prior_pts": row["prior_pts"],
                    "rank_tier": row["rank_tier"],
                    "pos_market_rank": index,
                    "market_avg": row["market_avg"],
                    "unranked_break": False,
                    "ranks": ranks,
                    "checks": checks,
                }
            )

    comparison = load_json(comparison_path)
    comparison_meta = comparison.get("meta") or {}
    ecr_meta = comparison_meta.get("ecr") or {}

    return {
        "season": season,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "top_n": top_n,
            "rank_source": "market_avg",
            "scoring_flavor": "ppr",
            "team_count": 12,
            "market_as_of": {
                "source": "espn+ffc+mfl+fantasypros_ecr",
                "scoring": "ppr",
                "teams": 12,
                "ecr_scrape": ecr_meta.get("scrape_date"),
                "fetched_at": market_meta.get("fetched_at"),
                "formula": market_meta.get("formula"),
                "sources": market_meta.get("sources"),
            },
            "sos_included": True,
            "ol_included": True,
            "offense_source": f"sealed:{vegas_path.name}",
            "sos_source": f"sealed:{sos_path.name}",
            "ol_source": f"sealed:{ranks_path.name}",
            "volume_source": f"sealed:{vegas_path.name}",
            "rank_board_source": "full_roster_team_stats",
            "checks_source": "derived_top16_from_ranks",
            "volume_caveat": VOLUME_CAVEAT,
            "criteria_labels": {
                key: CRITERIA_LABELS[key] for keys in criteria.values() for key in keys
            },
            "player_count": len(players_out),
            "vegas_method": vegas.get("method") or {},
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
        fh.write("\n")
    return destination


def patch_checklist_ol(
    season: int = 2026,
    *,
    checklist_path: Path | None = None,
    ol_ranks_path: Path | None = None,
) -> Path:
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
        fh.write("\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draft checklist JSON")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--db-path", default=None, help="projections.db path")
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
