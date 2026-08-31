"""Incremental simulation — affected-team fast path for projection updates.

The impact set used to be "changed players plus their own teammates", which
silently under-recomputed: an availability change reallocates opportunity to the
*opponent* defense-adjusted context too, and every league decision that consumes
an affected player is stale afterwards. Worse, when the dependency graph could
not be established at all (unknown team, no opponent mapping, a changed scoring
contract or model version) the code still produced a confident partial refresh.

:func:`build_impact_set` now widens to a full refresh whenever dependency
certainty is unavailable, and :class:`IncrementalSimulationService` refuses to
publish a partial run in that case.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.app.persistence.models import AvailabilityEvent
from src.app.persistence.repositories import ProjectionRepository
from src.app.projections.loader import PlayerSummary, ReleaseBundleLoader
from src.app.releases.gates import validate_promotion
from src.app.releases.publication import Candidate, CandidateRow, publish


@dataclass(frozen=True)
class ImpactSet:
    changed_player_ids: frozenset[str]
    affected_player_ids: frozenset[str]
    affected_teams: frozenset[str]
    input_hash: str
    opponent_teams: frozenset[str] = frozenset()
    affected_league_ids: frozenset[str] = frozenset()
    requires_full_refresh: bool = False
    widening_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "changed_players": sorted(self.changed_player_ids),
            "affected_players": len(self.affected_player_ids),
            "affected_teams": sorted(self.affected_teams),
            "opponent_teams": sorted(self.opponent_teams),
            "affected_leagues": sorted(self.affected_league_ids),
            "requires_full_refresh": self.requires_full_refresh,
            "widening_reasons": list(self.widening_reasons),
        }


def build_impact_set(
    changed_player_ids: set[str],
    players: dict[str, PlayerSummary],
    *,
    opponent_map: Mapping[str, str] | None = None,
    league_dependencies: Mapping[str, Iterable[str]] | None = None,
    weekly_context_changed: bool = False,
    scoring_contract_hash: str | None = None,
    baseline_scoring_contract_hash: str | None = None,
    model_version: str | None = None,
    baseline_model_version: str | None = None,
) -> ImpactSet:
    """Compute the recomputation scope, widening to a full refresh when unsure.

    Widening triggers (each forces ``requires_full_refresh``):

    * a changed player is unknown to the bundle or has no team — cross-team
      reallocation cannot be bounded;
    * weekly context changed but no opponent mapping is available, or a mapping
      is available and does not cover an affected team;
    * the compiled scoring contract hash differs from the baseline;
    * the model version differs from the baseline.
    """
    reasons: list[str] = []
    affected_teams: set[str] = set()
    for player_id in sorted(changed_player_ids):
        summary = players.get(player_id)
        if summary is None:
            reasons.append(f"unknown_player:{player_id}")
            continue
        if not summary.team:
            reasons.append(f"unknown_team:{player_id}")
            continue
        affected_teams.add(summary.team)

    opponent_teams: set[str] = set()
    if weekly_context_changed:
        if opponent_map is None:
            reasons.append("missing_opponent_map")
        else:
            for team in sorted(affected_teams):
                opponent = opponent_map.get(team)
                if not opponent:
                    reasons.append(f"missing_opponent_mapping:{team}")
                else:
                    opponent_teams.add(opponent)
    elif opponent_map is not None:
        for team in sorted(affected_teams):
            opponent = opponent_map.get(team)
            if opponent:
                opponent_teams.add(opponent)

    if (
        scoring_contract_hash is not None
        and baseline_scoring_contract_hash is not None
        and scoring_contract_hash != baseline_scoring_contract_hash
    ):
        reasons.append("scoring_contract_changed")
    if (
        model_version is not None
        and baseline_model_version is not None
        and model_version != baseline_model_version
    ):
        reasons.append("model_version_changed")

    recompute_teams = affected_teams | opponent_teams
    peer_ids = {
        player_id
        for player_id, summary in players.items()
        if summary.team and summary.team in recompute_teams
    }
    affected = frozenset(changed_player_ids | peer_ids)

    affected_leagues: set[str] = set()
    for league_id, rostered in (league_dependencies or {}).items():
        if affected.intersection(set(rostered)):
            affected_leagues.add(league_id)

    fingerprint = "|".join(
        (
            ",".join(sorted(affected)),
            ",".join(sorted(recompute_teams)),
            ",".join(sorted(affected_leagues)),
        )
    )
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()
    return ImpactSet(
        changed_player_ids=frozenset(changed_player_ids),
        affected_player_ids=affected,
        affected_teams=frozenset(affected_teams),
        input_hash=digest,
        opponent_teams=frozenset(opponent_teams),
        affected_league_ids=frozenset(affected_leagues),
        requires_full_refresh=bool(reasons),
        widening_reasons=tuple(reasons),
    )


class IncrementalSimulationService:
    """Recompute projections for an impact set and publish through the gates."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.projections = ProjectionRepository(session)

    def active_availability(self) -> dict[str, float]:
        events = (
            self.session.query(AvailabilityEvent)
            .filter(AvailabilityEvent.cleared_at.is_(None))
            .all()
        )
        availability: dict[str, float] = {}
        for event in events:
            policy = event.policy_json or {}
            if "play_probability" in policy:
                availability[event.player_id] = float(policy["play_probability"])
        return availability

    def promote_affected_week(
        self,
        season: int,
        week: int,
        impact: ImpactSet,
        *,
        base_run_id: str | None = None,
        automatic: bool = True,
    ) -> str | None:
        """Publish an incremental weekly run, or ``None`` when a full refresh is required."""
        if impact.requires_full_refresh:
            return None
        players = ReleaseBundleLoader(season=season).load()
        if not players:
            return None

        active = self.projections.active_run(mode="weekly", season=season, week=week)
        if active is None and base_run_id is None:
            return None
        source_run_id = base_run_id or active.id
        run_id = f"{source_run_id}-inc-{impact.input_hash[:10]}"
        availability = self.active_availability()

        rows: list[CandidateRow] = []
        for row in self.projections.player_projections(source_run_id):
            summary = players.get(row.player_id)
            if summary is None:
                continue
            points = float((row.mean_json or {}).get("points", summary.mean_points))
            avail = availability.get(
                row.player_id, row.availability_probability or summary.availability_probability
            )
            affected = row.player_id in impact.affected_player_ids
            if affected:
                points = points * avail
            rows.append(
                CandidateRow(
                    player_id=row.player_id,
                    team=row.team,
                    opponent=row.opponent,
                    availability_probability=avail,
                    mean_json={
                        **(row.mean_json or {}),
                        "points": points,
                        "incremental": affected,
                    },
                    quantiles_json=row.quantiles_json or summary.quantiles,
                )
            )
        if not rows:
            return None

        candidate = Candidate(
            mode="weekly",
            season=season,
            week=week,
            run_id=run_id,
            model_version="weekly_incremental_fixture_v1",
            input_hash=impact.input_hash,
            manifest_uri=f"derived://weekly-incremental/{season}/{week:02d}/{source_run_id}",
            artifact_mode="derived",
            partition_mode="weekly-incremental",
            rows=tuple(rows),
            metadata={
                "derivation": "incremental_affected_team",
                "base_run_id": source_run_id,
                "automatic": automatic,
                **impact.to_dict(),
            },
        )
        gates = {"promotion": validate_promotion(mode="weekly", players=players)}
        return publish(self.session, candidate, gates=gates).run_id
