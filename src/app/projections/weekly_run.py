"""Derive weekly (and ROS/dynasty) projection runs from preseason release data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.persistence.models import League
from src.app.projections.loader import ReleaseBundleLoader
from src.app.projections.weekly_v2_bridge import (
    STATE_ABSENT,
    STATE_FALLBACK,
    STATE_FIXTURE,
    STATE_TRAINED,
    WeeklyV2Readiness,
    weekly_v2_readiness,
)
from src.app.releases.gates import (
    GateResult,
    scoring_contract_fingerprint,
    validate_artifact_readiness,
    validate_promotion,
    validate_scoring_contracts,
)
from src.app.releases.publication import Candidate, CandidateRow, publish

#: Maps the weekly-v2 artifact state onto the label stored on ``projection_run``.
#: ``derived`` means the run was scaled from the sealed preseason release bundle
#: (real data, no weekly model); it is not a fixture.
ARTIFACT_MODE_BY_STATE = {
    STATE_TRAINED: "trained",
    STATE_FALLBACK: "fallback",
    STATE_FIXTURE: "fixture",
    STATE_ABSENT: "derived",
}

DERIVATION_BY_STATE = {
    STATE_TRAINED: "weekly_v2_trained_artifacts",
    STATE_FALLBACK: "weekly_v2_fallback_outputs",
    STATE_FIXTURE: "weekly_v2_fixture_manifest",
    STATE_ABSENT: "preseason_bundle_scaled",
}


@dataclass(frozen=True)
class WeeklyManifest:
    run_id: str
    season: int
    week: int
    model_version: str
    input_hash: str
    player_count: int
    derivation: str


def _weekly_factor(player_id: str, week: int) -> float:
    digest = hashlib.sha256(f"{player_id}:{week}".encode()).hexdigest()
    offset = (int(digest[:4], 16) / 0xFFFF) - 0.5
    return 1.0 + offset * 0.12


def build_weekly_manifest(season: int, week: int, *, namespace: str = "fixture") -> WeeklyManifest:
    input_hash = hashlib.sha256(f"{season}:{week}:{namespace}".encode()).hexdigest()
    return WeeklyManifest(
        run_id=f"weekly-{season}-w{week:02d}-{input_hash[:12]}",
        season=season,
        week=week,
        model_version="weekly_fixture_v1",
        input_hash=input_hash,
        player_count=0,
        derivation="preseason_bundle_scaled",
    )


def current_weekly_model_version(season: int, week: int) -> str:
    """Model version the weekly pipeline would stamp on a run created right now."""
    readiness = weekly_v2_readiness(season, week)
    if readiness.state == STATE_ABSENT:
        return build_weekly_manifest(season, week).model_version
    return readiness.model_version


class WeeklyProjectionService:
    """Builds immutable release candidates and publishes them through the gates."""

    BYE_WEEKS: dict[str, set[int]] = {
        "BUF": {7},
        "SF": {9},
        "KC": {10},
    }

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------ gates

    def _league_ids(self, league_ids: list[str] | None) -> list[str]:
        if league_ids is not None:
            return league_ids
        return [row.league_id for row in self.session.query(League).all()]

    def _scoring_gate(self, league_ids: list[str]) -> GateResult:
        return validate_scoring_contracts(self.session, league_ids)

    def _artifact_gate(self, readiness: WeeklyV2Readiness, *, automatic: bool) -> GateResult:
        return validate_artifact_readiness(
            readiness, app_env=get_settings().app_env, automatic=automatic
        )

    # ----------------------------------------------------------------- weekly

    def promote_week(
        self,
        season: int,
        week: int,
        *,
        namespace: str = "preseason-bridge",
        automatic: bool = True,
        league_ids: list[str] | None = None,
        extra_gates: dict[str, GateResult] | None = None,
    ) -> str | None:
        players = ReleaseBundleLoader(season=season).load()
        if not players:
            return None
        readiness = weekly_v2_readiness(season, week)
        manifest = build_weekly_manifest(season, week, namespace=namespace)
        state = readiness.state
        resolved_leagues = self._league_ids(league_ids)
        candidate = Candidate(
            mode="weekly",
            season=season,
            week=week,
            run_id=manifest.run_id,
            model_version=(
                readiness.model_version if state != STATE_ABSENT else manifest.model_version
            ),
            input_hash=manifest.input_hash,
            manifest_uri=readiness.manifest_uri,
            artifact_mode=ARTIFACT_MODE_BY_STATE[state],
            partition_mode="weekly",
            rows=self._weekly_rows(players, week),
            metadata={
                "derivation": DERIVATION_BY_STATE[state],
                "week": week,
                "player_count": len(players),
                "automatic": automatic,
                "weekly_v2_state": state,
                "weekly_v2_reasons": list(readiness.reasons),
                "scoring_contract_fingerprint": scoring_contract_fingerprint(
                    self.session, resolved_leagues
                ),
            },
        )
        gates: dict[str, GateResult] = {
            "promotion": validate_promotion(mode="weekly", players=players),
            "scoring_contract": self._scoring_gate(resolved_leagues),
            "artifact_readiness": self._artifact_gate(readiness, automatic=automatic),
            **(extra_gates or {}),
        }
        return publish(self.session, candidate, gates=gates).run_id

    def _weekly_rows(self, players: dict, week: int) -> tuple[CandidateRow, ...]:
        rows: list[CandidateRow] = []
        for summary in players.values():
            factor = _weekly_factor(summary.player_id, week)
            bye = week in self.BYE_WEEKS.get(summary.team or "", set())
            availability = 0.0 if bye else summary.availability_probability
            rows.append(
                CandidateRow(
                    player_id=summary.player_id,
                    team=summary.team,
                    opponent=None,
                    availability_probability=availability,
                    mean_json={
                        "points": 0.0 if bye else summary.mean_points * factor,
                        "position": summary.position,
                        "name": summary.name,
                        "team": summary.team,
                    },
                    quantiles_json={
                        key: (0.0 if bye else float(value) * factor)
                        for key, value in summary.quantiles.items()
                    },
                )
            )
        return tuple(rows)

    # -------------------------------------------------------------------- ros

    def promote_ros(
        self,
        season: int,
        *,
        from_week: int = 1,
        automatic: bool = True,
        league_ids: list[str] | None = None,
    ) -> str | None:
        players = ReleaseBundleLoader(season=season).load()
        if not players:
            return None
        input_hash = hashlib.sha256(f"ros:{season}:{from_week}".encode()).hexdigest()
        remaining_weeks = max(1, 18 - from_week)
        rows = tuple(
            CandidateRow(
                player_id=summary.player_id,
                team=summary.team,
                opponent=None,
                availability_probability=summary.availability_probability,
                mean_json={
                    "points": summary.mean_points * remaining_weeks,
                    "position": summary.position,
                    "name": summary.name,
                    "team": summary.team,
                },
                quantiles_json={
                    key: float(value) * remaining_weeks for key, value in summary.quantiles.items()
                },
            )
            for summary in players.values()
        )
        candidate = Candidate(
            mode="ros",
            season=season,
            week=None,
            run_id=f"ros-{season}-{input_hash[:12]}",
            model_version="ros_fixture_v1",
            input_hash=input_hash,
            manifest_uri=f"derived://ros/{season}/from-week-{from_week:02d}",
            artifact_mode="derived",
            partition_mode="ros",
            rows=rows,
            metadata={
                "remaining_weeks": remaining_weeks,
                "player_count": len(players),
                "automatic": automatic,
            },
        )
        gates = {
            "promotion": validate_promotion(mode="ros", players=players),
            "scoring_contract": self._scoring_gate(self._league_ids(league_ids)),
        }
        return publish(self.session, candidate, gates=gates).run_id

    # ---------------------------------------------------------------- dynasty

    def promote_dynasty(
        self,
        season: int,
        *,
        automatic: bool = True,
        league_ids: list[str] | None = None,
    ) -> str | None:
        players = ReleaseBundleLoader(season=season).load()
        if not players:
            return None
        input_hash = hashlib.sha256(f"dynasty:{season}".encode()).hexdigest()
        years = (1.0, 0.82, 0.64, 0.48)
        rows = tuple(
            CandidateRow(
                player_id=summary.player_id,
                team=summary.team,
                opponent=None,
                availability_probability=summary.availability_probability,
                mean_json={
                    "points": sum(summary.mean_points * 17 * factor for factor in years),
                    "position": summary.position,
                    "name": summary.name,
                    "team": summary.team,
                },
                quantiles_json={
                    key: float(value) * 17 * sum(years) / 4
                    for key, value in summary.quantiles.items()
                },
            )
            for summary in players.values()
        )
        candidate = Candidate(
            mode="dynasty",
            season=season,
            week=None,
            run_id=f"dynasty-{season}-{input_hash[:12]}",
            model_version="dynasty_fixture_v1",
            input_hash=input_hash,
            manifest_uri=f"derived://dynasty/{season}",
            artifact_mode="derived",
            partition_mode="dynasty",
            rows=rows,
            metadata={
                "horizon_years": len(years),
                "player_count": len(players),
                "automatic": automatic,
            },
        )
        gates = {
            "promotion": validate_promotion(mode="dynasty", players=players),
            "scoring_contract": self._scoring_gate(self._league_ids(league_ids)),
        }
        return publish(self.session, candidate, gates=gates).run_id
