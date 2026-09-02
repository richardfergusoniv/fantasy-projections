"""League-specific draft board derived from the sealed production release."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.app.projections.league_rescore import (
    load_component_projections,
    score_component_projection_set,
)
from src.app.projections.loader import ReleaseBundleLoader
from src.app.scoring.compiler import (
    compile_sleeper_scoring,
    scoring_settings_from_snapshot,
)
from src.app.scoring.contract import ScoringContract
from src.draft_assistant.tiers import OVERALL_TIER_GAP, assign_tiers
from src.projection.active_release import read_active_pointer
from src.projection.contracts import REPO_ROOT

OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")
_NEG = -1.0e12

# The sealed accuracy-first board is half-PPR with four-point passing TDs.
# League scoring is applied as a component-stat delta so the trained ensemble
# and market signal remain the point-estimate baseline.
BASELINE_HALF_PPR_SCORING: dict[str, float] = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fum_lost": -2.0,
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if np.isfinite(parsed) else default


def _draft_sort_value(row: dict) -> float:
    """Rank by VORP when present; zero VORP must not fall through."""
    vorp = row.get("vorp")
    if vorp is not None:
        try:
            return float(vorp)
        except TypeError, ValueError:
            pass
    season_pts = row.get("fantasy_pts_season")
    if season_pts is not None:
        try:
            return float(season_pts)
        except TypeError, ValueError:
            pass
    return 0.0


def _league_wide_replacement_ranks(
    rows: list[dict[str, Any]],
    contract: ScoringContract,
    team_count: int,
) -> dict[str, int]:
    """Derive replacement ranks from an exact league-wide starter assignment.

    Scoring seats are expanded across every fantasy team and assigned to the
    highest-value eligible players. Therefore 3-WR, multiple FLEX, REC_FLEX,
    and SUPER_FLEX formats change demand without hard-coded flex shares.
    """

    candidates = [row for row in rows if str(row.get("position")) in OFFENSE_POSITIONS]
    seats: list[tuple[str, ...]] = []
    for slot in contract.scoring_slots:
        eligible = tuple(
            pos for pos in slot.eligible_positions if pos in OFFENSE_POSITIONS
        )
        if eligible:
            seats.extend([eligible] * (int(slot.count) * max(1, int(team_count))))

    if not candidates or not seats:
        return {pos: 1 for pos in OFFENSE_POSITIONS}

    n_rows = len(candidates)
    n_cols = len(seats)
    size = max(n_rows, n_cols)
    matrix = np.full((size, size), _NEG, dtype=float)
    matrix[n_rows:, :] = 0.0
    matrix[:, n_cols:] = 0.0

    for row_idx, row in enumerate(candidates):
        position = str(row.get("position"))
        value = _number(row.get("league_points"))
        for col_idx, eligible in enumerate(seats):
            if position in eligible:
                matrix[row_idx, col_idx] = value

    assigned_rows, assigned_cols = linear_sum_assignment(-matrix)
    starter_counts = {pos: 0 for pos in OFFENSE_POSITIONS}
    for row_idx, col_idx in zip(assigned_rows, assigned_cols):
        if (
            row_idx >= n_rows
            or col_idx >= n_cols
            or matrix[row_idx, col_idx] <= _NEG / 2
        ):
            continue
        starter_counts[str(candidates[row_idx]["position"])] += 1

    available_counts = {
        pos: sum(str(row.get("position")) == pos for row in candidates)
        for pos in OFFENSE_POSITIONS
    }
    return {
        pos: min(max(1, starter_counts[pos] + 1), max(1, available_counts[pos]))
        for pos in OFFENSE_POSITIONS
    }


def _apply_league_vorp(
    rows: list[dict[str, Any]],
    contract: ScoringContract,
    team_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float]]:
    replacement_ranks = _league_wide_replacement_ranks(rows, contract, team_count)
    replacement_points: dict[str, float] = {}
    for position in OFFENSE_POSITIONS:
        ordered = sorted(
            (
                _number(row.get("league_points"))
                for row in rows
                if str(row.get("position")) == position
            ),
            reverse=True,
        )
        if not ordered:
            replacement_points[position] = 0.0
            continue
        index = min(max(replacement_ranks[position], 1), len(ordered)) - 1
        replacement_points[position] = ordered[index]

    enriched: list[dict[str, Any]] = []
    for row in rows:
        position = str(row.get("position"))
        if position not in OFFENSE_POSITIONS:
            continue
        item = dict(row)
        baseline = replacement_points[position]
        item["replacement_points"] = baseline
        item["replacement_rank"] = replacement_ranks[position]
        item["vorp"] = _number(item.get("league_points")) - baseline
        enriched.append(item)

    enriched.sort(
        key=lambda row: (
            _number(row.get("vorp")),
            _number(row.get("league_points")),
            str(row.get("player_id")),
        ),
        reverse=True,
    )
    if enriched:
        values = pd.Series([_number(row.get("vorp")) for row in enriched])
        tiers = assign_tiers(values, gap=OVERALL_TIER_GAP, pct_gap=0.04).tolist()
        for index, (row, tier) in enumerate(zip(enriched, tiers), start=1):
            row["rank"] = index
            row["tier"] = int(tier)
    return enriched, replacement_ranks, replacement_points


class DraftBoardService:
    def __init__(self, session=None) -> None:
        self.session = session

    def load_board(
        self,
        season: int = 2026,
        *,
        league_id: str | None = None,
        limit: int = 300,
    ) -> dict:
        pointer = read_active_pointer(season)
        players_path = self._players_path(season, pointer)
        if players_path is None or not players_path.exists():
            return {
                "entries": [],
                "source": "unavailable",
                "data_as_of": datetime.now(UTC).isoformat(),
                "projection_run_id": "unavailable",
                "league_specific": False,
            }
        payload = json.loads(players_path.read_text(encoding="utf-8"))
        if self.session is not None and league_id:
            board = self._load_league_board(
                season=season,
                league_id=league_id,
                payload=payload,
                limit=limit,
            )
            if board is not None:
                return board
        return self._load_sealed_default(payload, pointer=pointer, limit=limit)

    def _load_league_board(
        self,
        *,
        season: int,
        league_id: str,
        payload: dict[str, Any],
        limit: int,
    ) -> dict | None:
        from src.app.persistence.models import (
            League,
            LeagueMember,
            LeagueRuleSnapshot,
            RosterSnapshot,
        )

        league = (
            self.session.query(League)
            .filter(League.league_id == league_id, League.season == season)
            .one_or_none()
        )
        rules = (
            self.session.query(LeagueRuleSnapshot)
            .filter(LeagueRuleSnapshot.league_id == league_id)
            .order_by(LeagueRuleSnapshot.fetched_at.desc())
            .first()
        )
        if league is None or rules is None:
            return None

        raw = league.raw_json or {}
        roster_positions = list(raw.get("roster_positions") or [])
        contract = compile_sleeper_scoring(
            scoring_settings_from_snapshot(rules.raw_json or {}),
            roster_positions,
        )
        team_count = self._team_count(
            league_id=league_id,
            league_raw=raw,
            member_model=LeagueMember,
            roster_model=RosterSnapshot,
        )

        bundle = ReleaseBundleLoader(season=season).load_bundle()
        if bundle is None or bundle.component_projections_path is None:
            return None
        components = load_component_projections(bundle.component_projections_path)
        league_pg, fidelity, approximate = score_component_projection_set(
            components, contract
        )
        baseline_contract = compile_sleeper_scoring(BASELINE_HALF_PPR_SCORING, [])
        baseline_pg, _baseline_fidelity, _baseline_approximate = (
            score_component_projection_set(
                components,
                baseline_contract,
            )
        )

        rows: list[dict[str, Any]] = []
        covered = 0
        for player in payload.get("players", []):
            player_id = str(player.get("player_id") or "")
            position = str(player.get("position") or "")
            if not player_id or position not in OFFENSE_POSITIONS:
                continue
            selected_points = _number(player.get("fantasy_pts_season"))
            if player_id in league_pg and player_id in baseline_pg:
                games = _number(
                    components.get(player_id, {}).get("_projected_games"),
                    _number(player.get("projected_games"), 17.0),
                )
                league_points = (
                    selected_points
                    + (league_pg[player_id] - baseline_pg[player_id]) * games
                )
                covered += 1
            else:
                league_points = selected_points
            rows.append(
                {
                    "player_id": player_id,
                    "name": player.get("display_name")
                    or player.get("name")
                    or player_id,
                    "position": position,
                    "team": player.get("team"),
                    "league_points": league_points,
                    "fantasy_pts_season": selected_points,
                }
            )

        ranked, replacement_ranks, replacement_points = _apply_league_vorp(
            rows,
            contract,
            team_count,
        )
        entries = [
            {
                "player_id": row["player_id"],
                "name": row["name"],
                "position": row["position"],
                "team": row.get("team"),
                "rank": row["rank"],
                "tier": row["tier"],
                "vorp": round(_number(row.get("vorp")), 4),
                "points_mean": round(_number(row.get("league_points")), 4),
                "replacement_points": round(_number(row.get("replacement_points")), 4),
                "replacement_rank": int(row.get("replacement_rank") or 1),
            }
            for row in ranked[: max(1, int(limit))]
        ]
        meta = payload.get("meta", {})
        caveats = list(approximate)
        if contract.unsupported_keys:
            caveats.extend(
                f"unsupported_scoring:{key}"
                for key in sorted(contract.unsupported_keys)
            )
        if contract.unsupported_slots:
            caveats.extend(
                f"unsupported_slot:{slot}"
                for slot in sorted(contract.unsupported_slots)
            )
        if covered < len(rows):
            caveats.append(f"component_coverage:{covered}/{len(rows)}")
        return {
            "entries": entries,
            "source": "sealed_release_league_rescore",
            "namespace": bundle.namespace,
            "data_as_of": meta.get(
                "generated_at", bundle.generated_at or datetime.now(UTC).isoformat()
            ),
            "projection_run_id": f"preseason-{bundle.namespace}",
            "league_specific": True,
            "league_id": league_id,
            "league_name": league.name,
            "team_count": team_count,
            "roster_positions": roster_positions,
            "contract_hash": contract.contract_hash,
            "scoring_fidelity": fidelity,
            "component_coverage": {"covered": covered, "total": len(rows)},
            "replacement_ranks": replacement_ranks,
            "replacement_points": {
                key: round(value, 4) for key, value in replacement_points.items()
            },
            "caveats": sorted(set(caveats)),
        }

    def _team_count(
        self,
        *,
        league_id: str,
        league_raw: dict[str, Any],
        member_model,
        roster_model,
    ) -> int:
        for value in (
            league_raw.get("total_rosters"),
            (league_raw.get("settings") or {}).get("num_teams"),
        ):
            count = int(_number(value))
            if count > 1:
                return count
        member_count = (
            self.session.query(member_model)
            .filter(member_model.league_id == league_id)
            .count()
        )
        if member_count >= 4:
            return int(member_count)
        roster_count = (
            self.session.query(roster_model.roster_id)
            .filter(roster_model.league_id == league_id)
            .distinct()
            .count()
        )
        return int(roster_count) if roster_count >= 4 else 12

    def _load_sealed_default(
        self, payload: dict[str, Any], *, pointer: dict | None, limit: int
    ) -> dict:
        meta = payload.get("meta", {})
        ranked = sorted(payload.get("players", []), key=_draft_sort_value, reverse=True)
        entries = []
        for index, row in enumerate(ranked[: max(1, int(limit))], start=1):
            entries.append(
                {
                    "player_id": row.get("player_id"),
                    "name": row.get("display_name") or row.get("name"),
                    "position": row.get("position"),
                    "team": row.get("team"),
                    "rank": index,
                    "tier": row.get("overall_tier") or row.get("pos_tier"),
                    "vorp": row.get("vorp"),
                    "points_mean": row.get("fantasy_pts_season"),
                }
            )
        namespace = pointer.get("namespace") if pointer else "unavailable"
        return {
            "entries": entries,
            "source": "draft_assistant_release",
            "namespace": namespace,
            "data_as_of": meta.get("generated_at", datetime.now(UTC).isoformat()),
            "projection_run_id": f"preseason-{namespace}",
            "league_specific": False,
            "caveats": ["league_contract_unavailable"],
        }

    def _players_path(self, season: int, pointer: dict | None) -> Path | None:
        if pointer and pointer.get("public_urls", {}).get("players"):
            rel = pointer["public_urls"]["players"]
            if rel.startswith("data/"):
                rel = rel.removeprefix("data/")
            candidate = Path(REPO_ROOT) / "draft_assistant" / "data" / rel
            if candidate.exists():
                return candidate
        fallback = (
            Path(REPO_ROOT) / "draft_assistant" / "data" / f"players_{season}.json"
        )
        return fallback if fallback.exists() else None
