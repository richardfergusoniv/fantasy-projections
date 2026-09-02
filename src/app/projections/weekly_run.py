"""Derive weekly (and ROS/dynasty) projection runs from preseason release data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.persistence.models import League
from src.app.projections.loader import ReleaseBundleLoader
from src.app.projections.weekly_inference import (
    hash_scaled_preseason_rows,
    run_weekly_inference,
)
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
    validate_inference_provenance,
    validate_promotion,
    validate_scoring_contracts,
)
from src.app.releases.partitions import register_weekly_draw_partition
from src.app.releases.publication import Candidate, CandidateRow, publish
from src.projection.weekly.config.paths import OUTPUTS_DIR

ARTIFACT_MODE_BY_STATE = {
    STATE_TRAINED: "trained",
    STATE_FALLBACK: "fallback",
    STATE_FIXTURE: "fixture",
    STATE_ABSENT: "derived",
}

DERIVATION_BY_STATE = {
    STATE_TRAINED: "weekly_v2_trained_inference",
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

    def _build_rows(
        self,
        players: dict,
        season: int,
        week: int,
        readiness: WeeklyV2Readiness,
    ) -> tuple[CandidateRow, ...]:
        if readiness.state == STATE_TRAINED:
            inference = run_weekly_inference(season, week, persist=True)
            return inference.rows

        rows = hash_scaled_preseason_rows(players, week, factor_fn=_weekly_factor)
        adjusted: list[CandidateRow] = []
        for row in rows:
            bye = week in self.BYE_WEEKS.get(row.team or "", set())
            if not bye:
                adjusted.append(row)
                continue
            adjusted.append(
                CandidateRow(
                    player_id=row.player_id,
                    team=row.team,
                    opponent=row.opponent,
                    availability_probability=0.0,
                    mean_json={**row.mean_json, "points": 0.0},
                    quantiles_json={key: 0.0 for key in row.quantiles_json},
                )
            )
        return tuple(adjusted)

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
        state = readiness.state
        manifest = build_weekly_manifest(season, week, namespace=namespace)
        resolved_leagues = self._league_ids(league_ids)
        derivation = DERIVATION_BY_STATE[state]
        input_hash = manifest.input_hash
        output_sha256: str | None = None
        if state == STATE_TRAINED:
            inference = run_weekly_inference(season, week, persist=True)
            rows = inference.rows
            input_hash = inference.input_hash
            output_sha256 = inference.output_sha256
        else:
            rows = self._build_rows(players, season, week, readiness)
        candidate = Candidate(
            mode="weekly",
            season=season,
            week=week,
            run_id=(
                f"weekly-{season}-w{week:02d}-{input_hash[:12]}"
                if state == STATE_TRAINED
                else manifest.run_id
            ),
            model_version=(
                readiness.model_version if state != STATE_ABSENT else manifest.model_version
            ),
            input_hash=input_hash,
            manifest_uri=readiness.manifest_uri,
            artifact_mode=ARTIFACT_MODE_BY_STATE[state],
            partition_mode="weekly",
            rows=rows,
            metadata={
                "derivation": derivation,
                "week": week,
                "player_count": len(rows),
                "automatic": automatic,
                "weekly_v2_state": state,
                "weekly_v2_reasons": list(readiness.reasons),
                "output_sha256": output_sha256,
                "scoring_contract_fingerprint": scoring_contract_fingerprint(
                    self.session, resolved_leagues
                ),
            },
        )
        gates: dict[str, GateResult] = {
            "promotion": validate_promotion(mode="weekly", players=players),
            "scoring_contract": self._scoring_gate(resolved_leagues),
            "artifact_readiness": self._artifact_gate(readiness, automatic=automatic),
            "inference_provenance": validate_inference_provenance(candidate, readiness),
            **(extra_gates or {}),
        }
        result = publish(
            self.session,
            candidate,
            gates=gates,
            register_partitions=state != STATE_TRAINED,
            validate_partitions=state != STATE_TRAINED,
        )
        if result.run_id and state == STATE_TRAINED:
            partition_path = (
                OUTPUTS_DIR / f"season={season}" / f"week={week:02d}" / "stat_draw_partition.json"
            )
            if partition_path.exists():
                from src.app.projections.weekly_draws import weekly_draw_partition_from_file

                partition = weekly_draw_partition_from_file(partition_path, seed_salt=input_hash)
                register_weekly_draw_partition(
                    self.session,
                    run_id=result.run_id,
                    input_hash=input_hash,
                    partition=partition,
                )
        return result.run_id

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
