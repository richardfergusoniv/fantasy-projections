"""Dedicated weekly_props publisher — does not use weekly-v2 readiness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy.orm import Session

from src.app.logging import get_logger
from src.app.persistence.models import League
from src.app.releases.gates import (
    GateResult,
    scoring_contract_fingerprint,
    validate_promotion,
    validate_scoring_contracts,
)
from src.app.releases.publication import PublicationResult, active_pointer, publish
from src.ingest.props.contracts import ConsensusManifest, ProviderSnapshot
from src.projection.weekly_props.bridge import (
    BaselinePlayer,
    build_candidate,
    build_candidate_rows,
)
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY, WeeklyPropsPolicy
from src.projection.weekly_props.consensus import build_consensus_manifest, build_player_consensus
from src.projection.weekly_props.gates import (
    validate_baseline_movement,
    validate_candidate_sanity,
    validate_consensus_coverage,
    validate_props_provenance,
    validate_snapshots,
)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ARTIFACT_ROOT = REPO_ROOT / "data" / "props" / "fixtures"


@dataclass(frozen=True)
class WeeklyPropsPromoteResult:
    publication: PublicationResult
    manifest_uri: str
    semantic_input_hash: str
    shadow: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.publication.to_dict(),
            "manifest_uri": self.manifest_uri,
            "semantic_input_hash": self.semantic_input_hash,
            "shadow": self.shadow,
        }


class WeeklyPropsProjectionService:
    """Build market candidates and publish through lower-level ``publish()``."""

    def __init__(
        self,
        session: Session,
        *,
        policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
        artifact_root: Path | None = None,
    ) -> None:
        self.session = session
        self.policy = policy
        self.artifact_root = artifact_root or FIXTURE_ARTIFACT_ROOT

    def _league_ids(self, league_ids: list[str] | None) -> list[str]:
        if league_ids is not None:
            return league_ids
        return [row.league_id for row in self.session.query(League).all()]

    def write_manifest(self, manifest: ConsensusManifest) -> str:
        path = (
            self.artifact_root
            / f"season={manifest.season}"
            / f"week={manifest.week:02d}"
            / f"consensus_{manifest.semantic_input_hash[:12]}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return path.as_uri()

    def build_manifest_from_snapshots(
        self,
        *,
        season: int,
        week: int,
        snapshots: list[ProviderSnapshot],
        identity_map: Mapping[str, dict[str, Any]] | None = None,
        slate_teams: set[str] | None = None,
        baseline_run_id: str | None = None,
        status_overlay_id: str | None = None,
        slate_version: str | None = None,
    ) -> ConsensusManifest:
        identity_map = identity_map or {}
        by_player: dict[str, list] = {}
        meta_by_key: dict[str, dict[str, Any]] = {}
        for snap in snapshots:
            if not snap.success:
                continue
            for quote in snap.quotes:
                key = quote.player_id or quote.player_name_raw.strip().lower()
                by_player.setdefault(key, []).append(quote)
                meta_by_key.setdefault(
                    key,
                    {
                        "player_id": quote.player_id,
                        "player_name": quote.player_name_raw,
                        "team": quote.team,
                        "opponent": quote.opponent,
                    },
                )
        players = []
        for key, quotes in by_player.items():
            ident = identity_map.get(key) or identity_map.get(
                (meta_by_key[key].get("player_name") or "").lower()
            ) or {}
            team = ident.get("team") or meta_by_key[key].get("team")
            on_slate = True
            if slate_teams is not None and team:
                on_slate = str(team).upper() in slate_teams
            players.append(
                build_player_consensus(
                    player_key=key,
                    quotes=quotes,
                    policy=self.policy,
                    player_id=ident.get("player_id") or meta_by_key[key].get("player_id"),
                    player_name=ident.get("name") or meta_by_key[key].get("player_name"),
                    team=team,
                    opponent=ident.get("opponent") or meta_by_key[key].get("opponent"),
                    position=ident.get("position"),
                    on_slate=on_slate,
                )
            )
        return build_consensus_manifest(
            season=season,
            week=week,
            snapshots=snapshots,
            players=players,
            policy=self.policy,
            baseline_run_id=baseline_run_id,
            status_overlay_id=status_overlay_id,
            slate_version=slate_version,
        )

    def promote(
        self,
        *,
        season: int,
        week: int,
        snapshots: list[ProviderSnapshot],
        baselines: Mapping[str, BaselinePlayer],
        identity_map: Mapping[str, dict[str, Any]] | None = None,
        slate_teams: set[str] | None = None,
        baseline_run_id: str | None = None,
        status_overlay_id: str | None = None,
        slate_version: str | None = None,
        league_ids: list[str] | None = None,
        automatic: bool = True,
        shadow: bool = True,
        force: bool = False,
    ) -> WeeklyPropsPromoteResult:
        snapshot_gate = validate_snapshots(
            snapshots, season=season, week=week, policy=self.policy
        )
        manifest = self.build_manifest_from_snapshots(
            season=season,
            week=week,
            snapshots=snapshots,
            identity_map=identity_map,
            slate_teams=slate_teams,
            baseline_run_id=baseline_run_id,
            status_overlay_id=status_overlay_id,
            slate_version=slate_version,
        )
        coverage_gate = validate_consensus_coverage(manifest, policy=self.policy)
        rows = build_candidate_rows(manifest, baselines, policy=self.policy)
        sanity_gate = validate_candidate_sanity(rows, policy=self.policy)
        baseline_points = {
            pid: float(base.mean_json.get("points") or 0.0)
            for pid, base in baselines.items()
        }
        movement_gate = validate_baseline_movement(
            rows, baseline_points, policy=self.policy
        )

        # Idempotency: same semantic hash already active → no new run.
        pointer = active_pointer(self.session, mode="weekly", season=season, week=week)
        if (
            pointer is not None
            and not force
            and pointer.run_id.endswith(manifest.semantic_input_hash[:12])
        ):
            return WeeklyPropsPromoteResult(
                publication=PublicationResult(
                    run_id=pointer.run_id,
                    promoted=True,
                    reason="semantic_hash_unchanged",
                    gates={},
                    already_active=True,
                ),
                manifest_uri=f"active:{pointer.run_id}",
                semantic_input_hash=manifest.semantic_input_hash,
                shadow=shadow,
            )

        manifest_uri = self.write_manifest(manifest)
        candidate = build_candidate(
            manifest=manifest,
            rows=rows,
            manifest_uri=manifest_uri,
            baseline_run_id=baseline_run_id,
            metadata={
                "automatic": automatic,
                "shadow": shadow,
                "scoring_contract_fingerprint": scoring_contract_fingerprint(
                    self.session, self._league_ids(league_ids)
                ),
            },
        )
        provenance_gate = validate_props_provenance(candidate)
        players_for_promotion = {
            row.player_id: type(
                "P",
                (),
                {
                    "quantiles": row.quantiles_json,
                    "availability_probability": row.availability_probability or 0.0,
                    "mean_points": float(row.mean_json.get("points") or 0.0),
                    "position": row.mean_json.get("position"),
                },
            )()
            for row in rows
        }
        gates: dict[str, GateResult] = {
            "snapshots": snapshot_gate,
            "coverage": coverage_gate,
            "candidate_sanity": sanity_gate,
            "baseline_movement": movement_gate,
            "provenance": provenance_gate,
            "promotion": validate_promotion(
                mode="weekly",
                players=players_for_promotion,  # type: ignore[arg-type]
                min_players=self.policy.min_candidate_players,
            ),
            "scoring_contract": validate_scoring_contracts(
                self.session, self._league_ids(league_ids)
            ),
        }
        if shadow:
            # Shadow never swaps the pointer; still persist failed/candidate audit via publish
            # by forcing a gate failure after computing the full report when requested.
            # Prefer explicit shadow path: run publish only when not shadow.
            logger.info(
                "weekly_props_shadow_candidate",
                run_id=candidate.run_id,
                hash=manifest.semantic_input_hash[:12],
                gates={k: v.to_dict() for k, v in gates.items()},
            )
            passed = all(g.passed for g in gates.values())
            return WeeklyPropsPromoteResult(
                publication=PublicationResult(
                    run_id=candidate.run_id,
                    promoted=False,
                    reason="shadow_only" if passed else "shadow_gates_failed",
                    gates={k: v.to_dict() for k, v in gates.items()},
                ),
                manifest_uri=manifest_uri,
                semantic_input_hash=manifest.semantic_input_hash,
                shadow=True,
            )

        result = publish(
            self.session,
            candidate,
            gates=gates,
            register_partitions=True,
            validate_partitions=True,
        )
        return WeeklyPropsPromoteResult(
            publication=result,
            manifest_uri=manifest_uri,
            semantic_input_hash=manifest.semantic_input_hash,
            shadow=False,
        )
