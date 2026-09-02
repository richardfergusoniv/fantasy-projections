"""Conservative daily status/depth overlay on sealed production release.

The overlay is versioned, immutable, and separately gated. It may zero OUT
players and adjust questionable availability; it must not pretend to be a weekly
matchup model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.app.artifacts.store import get_artifact_store
from src.app.config import get_settings
from src.app.projections.loader import PlayerSummary, ReleaseBundleLoader
from src.app.releases.gates import GateResult
from src.projection.active_release import read_active_pointer
from src.projection.contracts import REPO_ROOT

OVERLAY_SCHEMA_VERSION = "status_overlay_pointer_v1"
OVERLAY_ALGORITHM_VERSION = "status_overlay_v1"
OVERLAY_ARTIFACT_DIR = Path(REPO_ROOT) / "output" / "app_artifacts" / "status_overlays"


@dataclass
class OverlayAdjustment:
    player_id: str
    team: str | None
    position: str
    before_points: float
    after_points: float
    before_availability: float
    after_availability: float
    reason_code: str
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "position": self.position,
            "before_points": round(self.before_points, 6),
            "after_points": round(self.after_points, 6),
            "before_availability": round(self.before_availability, 6),
            "after_availability": round(self.after_availability, 6),
            "reason_code": self.reason_code,
            "citations": list(self.citations),
        }


@dataclass
class StatusOverlayBundle:
    schema_version: str
    algorithm_version: str
    base_release_id: str
    base_manifest_sha256: str
    overlay_hash: str
    generated_at: str
    source_observations: list[dict[str, Any]]
    adjustments: list[OverlayAdjustment]
    players: dict[str, PlayerSummary]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "base_release_id": self.base_release_id,
            "base_manifest_sha256": self.base_manifest_sha256,
            "overlay_hash": self.overlay_hash,
            "generated_at": self.generated_at,
            "source_observations": self.source_observations,
            "adjustments": [adj.to_dict() for adj in self.adjustments],
            "validation": self.validation,
            "player_count": len(self.players),
        }


def _overlay_pointer_path(season: int) -> Path:
    return Path(REPO_ROOT) / "draft_assistant" / "data" / f"active_status_overlay_{season}.json"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _apply_availability_rules(
    summary: PlayerSummary,
    *,
    status: str | None,
    availability_probability: float | None,
) -> tuple[float, float, str]:
    """Apply availability exactly once. Returns (points, availability, reason)."""
    base_pts = summary.mean_points
    base_avail = summary.availability_probability
    normalized = (status or "").upper()

    if normalized in {"OUT", "IR", "PUP", "SUSP", "INACTIVE"}:
        return 0.0, 0.0, f"status_zero:{normalized}"
    if availability_probability is not None:
        avail = max(0.0, min(1.0, float(availability_probability)))
        if avail < base_avail:
            return base_pts * avail, avail, "availability_downgrade"
    if normalized in {"DOUBTFUL", "QUESTIONABLE"}:
        avail = min(base_avail, 0.5 if normalized == "DOUBTFUL" else 0.75)
        return base_pts * avail, avail, f"status_discount:{normalized}"
    return base_pts, base_avail, "unchanged"


def build_status_overlay(
    *,
    season: int,
    availability_events: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]] | None = None,
) -> StatusOverlayBundle | None:
    pointer = read_active_pointer(season)
    if pointer is None:
        return None
    loader = ReleaseBundleLoader(season=season)
    bundle = loader.load_bundle()
    if bundle is None or not bundle.players:
        return None

    base_players = dict(bundle.players)
    adjustments: list[OverlayAdjustment] = []
    observations: list[dict[str, Any]] = []
    adjusted_players: dict[str, PlayerSummary] = {}

    status_by_player = {
        str(evt.get("player_id")): evt for evt in availability_events if evt.get("player_id")
    }
    for evt in availability_events:
        observations.append(
            {
                "player_id": evt.get("player_id"),
                "status": evt.get("status"),
                "observed_at": evt.get("observed_at") or evt.get("fetched_at"),
                "source": evt.get("source", "sleeper"),
            }
        )

    for player_id, summary in base_players.items():
        evt = status_by_player.get(player_id, {})
        after_pts, after_avail, reason = _apply_availability_rules(
            summary,
            status=evt.get("status"),
            availability_probability=evt.get("availability_probability"),
        )
        if reason != "unchanged":
            citations = []
            if evidence_rows:
                for row in evidence_rows:
                    if str(row.get("player_id")) == player_id:
                        citations.append(str(row.get("summary") or row.get("source") or "evidence"))
            adjustments.append(
                OverlayAdjustment(
                    player_id=player_id,
                    team=summary.team,
                    position=summary.position,
                    before_points=summary.mean_points,
                    after_points=after_pts,
                    before_availability=summary.availability_probability,
                    after_availability=after_avail,
                    reason_code=reason,
                    citations=citations[:3],
                )
            )
        adjusted_players[player_id] = PlayerSummary(
            player_id=summary.player_id,
            name=summary.name,
            position=summary.position,
            team=summary.team,
            mean_points=after_pts,
            quantiles={
                k: float(v) * (after_avail / summary.availability_probability)
                if summary.availability_probability > 0
                else float(v)
                for k, v in summary.quantiles.items()
            }
            if summary.quantiles
            else {},
            availability_probability=after_avail,
        )

    payload_without_hash = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "algorithm_version": OVERLAY_ALGORITHM_VERSION,
        "base_release_id": bundle.release_id,
        "base_manifest_sha256": bundle.manifest_sha256,
        "generated_at": datetime.now(UTC).isoformat(),
        "adjustments": [adj.to_dict() for adj in adjustments],
        "observations": observations,
    }
    overlay_hash = _canonical_hash(payload_without_hash)

    validation = validate_overlay_gate(
        bundle_release_id=bundle.release_id,
        bundle_manifest_sha256=bundle.manifest_sha256,
        adjustments=adjustments,
        players=adjusted_players,
        overlay_hash=overlay_hash,
    )

    return StatusOverlayBundle(
        schema_version=OVERLAY_SCHEMA_VERSION,
        algorithm_version=OVERLAY_ALGORITHM_VERSION,
        base_release_id=bundle.release_id,
        base_manifest_sha256=bundle.manifest_sha256,
        overlay_hash=overlay_hash,
        generated_at=payload_without_hash["generated_at"],
        source_observations=observations,
        adjustments=adjustments,
        players=adjusted_players,
        validation=validation.to_dict(),
    )


def validate_overlay_gate(
    *,
    bundle_release_id: str,
    bundle_manifest_sha256: str,
    adjustments: list[OverlayAdjustment],
    players: dict[str, PlayerSummary],
    overlay_hash: str,
) -> GateResult:
    failures: list[str] = []
    if not bundle_release_id:
        failures.append("missing_base_release_id")
    if len(bundle_manifest_sha256) != 64:
        failures.append("invalid_base_manifest_hash")
    if len(overlay_hash) != 64:
        failures.append("invalid_overlay_hash")

    for adj in adjustments:
        if adj.after_points < 0 or adj.after_availability < 0 or adj.after_availability > 1:
            failures.append(f"invalid_adjustment:{adj.player_id}")
        if adj.reason_code.startswith("status_zero") and adj.after_points != 0:
            failures.append(f"out_not_zeroed:{adj.player_id}")
        if not adj.reason_code:
            failures.append(f"missing_reason_code:{adj.player_id}")

    for pid, summary in players.items():
        if summary.mean_points < 0:
            failures.append(f"negative_points:{pid}")
        if summary.availability_probability < 0 or summary.availability_probability > 1:
            failures.append(f"invalid_availability:{pid}")

    return GateResult(passed=not failures, failures=failures)


def write_overlay_artifact(overlay: StatusOverlayBundle) -> str:
    """Persist overlay body to the configured artifact backend; return URI."""
    store = get_artifact_store()
    return store.put_json(
        overlay.to_dict(),
        provenance={
            "kind": "status_overlay",
            "overlay_hash": overlay.overlay_hash,
            "base_release_id": overlay.base_release_id,
        },
    )


def promote_overlay_pointer(
    overlay: StatusOverlayBundle,
    *,
    season: int,
    session: Session | None = None,
) -> str | None:
    if not overlay.validation.get("passed"):
        return None
    artifact_uri = write_overlay_artifact(overlay)
    pointer = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "season": season,
        "status": "active",
        "overlay_hash": overlay.overlay_hash,
        "base_release_id": overlay.base_release_id,
        "base_manifest_sha256": overlay.base_manifest_sha256,
        "algorithm_version": overlay.algorithm_version,
        "activated_at": datetime.now(UTC).isoformat(),
        "artifact_uri": artifact_uri,
        "adjustment_count": len(overlay.adjustments),
        "previous": None,
    }

    if session is not None:
        from src.app.persistence.release_pointers import StatusOverlayPointerStore

        existing = StatusOverlayPointerStore(session).read(season)
        if existing is not None:
            pointer["previous"] = existing
        StatusOverlayPointerStore(session).write(pointer)
    else:
        _try_db_write_overlay_pointer(pointer)

    settings = get_settings()
    if settings.app_env != "production":
        _write_overlay_pointer_filesystem(pointer, season=season)
    return artifact_uri


def _try_db_write_overlay_pointer(pointer: dict[str, Any]) -> bool:
    try:
        from src.app.persistence.database import get_session
        from src.app.persistence.release_pointers import StatusOverlayPointerStore

        with get_session() as session:
            StatusOverlayPointerStore(session).write(pointer)
        return True
    except Exception:
        return False


def _write_overlay_pointer_filesystem(pointer: dict[str, Any], *, season: int) -> None:
    from src.projection.active_release import atomic_write_json

    path = _overlay_pointer_path(season)
    if path.exists():
        try:
            pointer["previous"] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pointer["previous"] = None
    atomic_write_json(path, pointer)


def read_active_overlay(season: int, *, session: Session | None = None) -> dict[str, Any] | None:
    if session is not None:
        from src.app.persistence.release_pointers import StatusOverlayPointerStore

        return StatusOverlayPointerStore(session).read(season)

    try:
        from src.app.persistence.database import get_session
        from src.app.persistence.release_pointers import StatusOverlayPointerStore

        with get_session() as db_session:
            db_pointer = StatusOverlayPointerStore(db_session).read(season)
            if db_pointer is not None:
                return db_pointer
    except Exception:
        pass

    settings = get_settings()
    if settings.app_env == "production":
        return None

    path = _overlay_pointer_path(season)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rollback_overlay_pointer(season: int, *, session: Session | None = None) -> dict[str, Any] | None:
    if session is not None:
        from src.app.persistence.release_pointers import StatusOverlayPointerStore

        return StatusOverlayPointerStore(session).rollback(season)

    try:
        from src.app.persistence.database import get_session
        from src.app.persistence.release_pointers import StatusOverlayPointerStore

        with get_session() as db_session:
            rolled = StatusOverlayPointerStore(db_session).rollback(season)
            if rolled is not None:
                return rolled
    except Exception:
        pass

    settings = get_settings()
    if settings.app_env == "production":
        return None

    path = _overlay_pointer_path(season)
    if not path.exists():
        return None
    current = json.loads(path.read_text(encoding="utf-8"))
    previous = current.get("previous")
    if not previous:
        return None
    from src.projection.active_release import atomic_write_json

    atomic_write_json(path, previous)
    return previous


def active_overlay_status(session: Session | None, *, season: int = 2026) -> dict[str, Any]:
    pointer = read_active_overlay(season, session=session)
    if pointer is None:
        return {
            "active": False,
            "detail": "no active status overlay pointer",
            "publication_allowed": False,
            "publication_detail": "no overlay has passed gate",
        }
    artifact_uri = str(pointer.get("artifact_uri") or pointer.get("artifact_path") or "")
    readable = bool(artifact_uri)
    if readable:
        try:
            from src.app.artifacts.store import get_artifact_store

            get_artifact_store().get_json(artifact_uri)
        except Exception:
            readable = False
    return {
        "active": readable,
        "detail": (
            f"overlay={str(pointer.get('overlay_hash', ''))[:12]}… "
            f"adjustments={pointer.get('adjustment_count', 0)}"
        ),
        "publication_allowed": readable,
        "publication_detail": "active overlay pointer with readable artifact",
        "pointer": pointer,
    }


def load_overlay_players(season: int, *, session: Session | None = None) -> dict[str, PlayerSummary] | None:
    pointer = read_active_overlay(season, session=session)
    if pointer is None:
        return None
    artifact_uri = str(pointer.get("artifact_uri") or pointer.get("artifact_path") or "")
    if not artifact_uri:
        return None
    loader = ReleaseBundleLoader(season=season)
    bundle = loader.load_bundle()
    if bundle is None:
        return None
    try:
        from src.app.artifacts.store import get_artifact_store

        payload = get_artifact_store().get_json(artifact_uri)
    except Exception:
        return None
    # Reconstruct from base + stored adjustments for immutability verification.
    players = dict(bundle.players)
    for adj in payload.get("adjustments", []):
        pid = str(adj.get("player_id"))
        if pid not in players:
            continue
        base = players[pid]
        players[pid] = PlayerSummary(
            player_id=base.player_id,
            name=base.name,
            position=base.position,
            team=base.team,
            mean_points=float(adj.get("after_points", base.mean_points)),
            quantiles=base.quantiles,
            availability_probability=float(
                adj.get("after_availability", base.availability_probability)
            ),
        )
    return players
