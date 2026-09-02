"""League-specific scoring for weekly-v2 component stat projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl

from src.app.projections.weekly_stat_draw import weekly_row_to_stat_draw
from src.app.scoring.compiler import compile_sleeper_scoring, score_stat_draw
from src.app.scoring.contract import ScoringContract


@dataclass(frozen=True)
class LeagueScoringContract:
    league_id: str
    display_name: str
    scoring_settings: dict[str, Any]
    roster_positions: list[str]
    contract: ScoringContract

    @classmethod
    def from_league_json(
        cls,
        *,
        league_id: str,
        display_name: str,
        raw_json: dict[str, Any],
    ) -> "LeagueScoringContract":
        scoring = dict(raw_json.get("scoring_settings") or {})
        roster_positions = list(raw_json.get("roster_positions") or [])
        contract = compile_sleeper_scoring(scoring, roster_positions)
        return cls(
            league_id=league_id,
            display_name=display_name,
            scoring_settings=scoring,
            roster_positions=roster_positions,
            contract=contract,
        )


def score_weekly_record(
    record: dict[str, Any],
    league: LeagueScoringContract,
) -> float:
    position = str(record.get("position") or "RB")
    draw = weekly_row_to_stat_draw(record)
    return score_stat_draw(draw, league.contract, position=position)


def score_weekly_frame_for_leagues(
    frame: pl.DataFrame,
    leagues: Sequence[LeagueScoringContract],
    *,
    min_receptions: float = 3.0,
    sample_size: int = 25,
) -> dict[str, Any]:
    """Score weekly component stats under each league contract.

    Returns an artifact suitable for shadow validation: per-league contract
    hashes, sample player scores, and cross-league spread checks.
    """
    skill = frame.filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE"])
        & (pl.col("receptions").fill_null(0) >= min_receptions)
    ).sort("fantasy_points", descending=True)
    if skill.is_empty():
        skill = frame.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"])).head(sample_size)
    else:
        skill = skill.head(sample_size)

    sample_records = list(skill.iter_rows(named=True))
    per_league: list[dict[str, Any]] = []
    player_scores: dict[str, dict[str, float]] = {}

    for league in leagues:
        scored_rows: list[dict[str, Any]] = []
        for record in sample_records:
            player_id = str(record.get("gsis_id") or record.get("player_id") or "")
            if not player_id:
                continue
            points = score_weekly_record(record, league)
            scored_rows.append(
                {
                    "player_id": player_id,
                    "name": record.get("player_name"),
                    "position": record.get("position"),
                    "league_scored_points": round(points, 3),
                    "model_fantasy_points": round(float(record.get("fantasy_points") or 0.0), 3),
                }
            )
            player_scores.setdefault(player_id, {})[league.league_id] = points
        scored_rows.sort(key=lambda row: row["league_scored_points"], reverse=True)
        per_league.append(
            {
                "league_id": league.league_id,
                "display_name": league.display_name,
                "contract_hash": league.contract.contract_hash,
                "unsupported_keys": list(league.contract.unsupported_keys),
                "top_players": scored_rows[:10],
            }
        )

    spreads: list[dict[str, Any]] = []
    for player_id, by_league in player_scores.items():
        if len(by_league) < 2:
            continue
        values = list(by_league.values())
        spreads.append(
            {
                "player_id": player_id,
                "min_score": round(min(values), 3),
                "max_score": round(max(values), 3),
                "spread": round(max(values) - min(values), 3),
            }
        )
    spreads.sort(key=lambda row: row["spread"], reverse=True)

    distinct_hashes = {league.contract.contract_hash for league in leagues}
    max_spread = spreads[0]["spread"] if spreads else 0.0

    return {
        "league_count": len(leagues),
        "distinct_contract_hashes": len(distinct_hashes),
        "sample_player_count": len(sample_records),
        "max_cross_league_spread": max_spread,
        "leagues": per_league,
        "cross_league_spreads": spreads[:15],
        "validation": {
            "six_distinct_contracts": len(distinct_hashes) >= 6,
            "cross_league_scoring_differs": max_spread > 0.25,
            "all_contracts_publishable": all(
                not league.contract.blocks_publication for league in leagues
            ),
        },
    }
