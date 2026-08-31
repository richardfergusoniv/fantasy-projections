"""Release pointer integration for app projection modes."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from src.app.projections.loader import ReleaseBundleLoader
from src.app.releases.gates import GateResult, validate_promotion
from src.app.releases.publication import Candidate, CandidateRow, publish
from src.projection.active_release import read_active_pointer


class ReleaseBridge:
    """Bridge existing sealed release bundles into app projection pointers."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def _manifest_path(self, season: int, namespace: str) -> Path | None:
        candidates = (
            Path("draft_assistant/data/releases") / namespace / "release_bundle_manifest.json",
            Path("output/model_v3/release_bundles")
            / f"season={season}"
            / f"namespace={namespace}"
            / "release_bundle_manifest.json",
        )
        for path in candidates:
            if path.exists():
                return path
        return None

    def sync_preseason_pointer(
        self,
        season: int,
        *,
        automatic: bool = True,
        extra_gates: dict[str, GateResult] | None = None,
    ) -> str | None:
        """Promote the sealed preseason release bundle behind the same gates.

        This used to promote the pointer unconditionally, with no gates at all,
        and always wrote ``PromotionEvent(promoted=True)`` even when nothing was
        validated. It now runs the completeness/bounds gate against an immutable
        candidate and goes through the shared publication pipeline, so a bad
        bundle cannot silently become the active preseason release.
        """
        pointer = read_active_pointer(season)
        if pointer is None:
            return None
        namespace = pointer["namespace"]
        manifest_path = self._manifest_path(season, namespace)
        if manifest_path is None:
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        players = ReleaseBundleLoader(season=season).load()
        if not players:
            return None

        rows = tuple(
            CandidateRow(
                player_id=summary.player_id,
                team=summary.team,
                opponent=None,
                availability_probability=summary.availability_probability,
                mean_json={
                    "points": summary.mean_points,
                    "position": summary.position,
                    "name": summary.name,
                },
                quantiles_json=summary.quantiles,
            )
            for summary in players.values()
        )
        candidate = Candidate(
            mode="preseason",
            season=season,
            week=None,
            run_id=f"preseason-{namespace}",
            model_version=str(manifest.get("model_version", "v2_baseline")),
            input_hash=str(pointer.get("manifest_sha256", "unknown")),
            manifest_uri=manifest_path.resolve().as_uri(),
            artifact_mode="release_bundle",
            partition_mode="preseason",
            rows=rows,
            metadata={
                "source": "active_release_pointer",
                "namespace": namespace,
                "player_count": len(players),
                "automatic": automatic,
            },
        )
        gates: dict[str, GateResult] = {
            "promotion": validate_promotion(mode="preseason", players=players),
            **(extra_gates or {}),
        }
        # The sealed bundle carries its own provenance manifest rather than app
        # simulation partitions, so the partition gate does not apply here.
        return publish(
            self.session,
            candidate,
            gates=gates,
            register_partitions=False,
            validate_partitions=False,
        ).run_id
