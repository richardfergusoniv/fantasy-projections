"""Database-backed release and status-overlay pointer stores."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from sqlalchemy.orm import Session

from src.app.persistence.models import (
    ReleasePointer,
    ReleasePointerHistory,
    StatusOverlayPointer,
    StatusOverlayPointerHistory,
    utcnow,
)
from src.projection.active_release import (
    ActiveReleaseError,
    build_active_pointer,
    validate_active_pointer,
)


class ReleasePointerStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def read(self, season: int) -> dict[str, Any] | None:
        row = (
            self.session.query(ReleasePointer)
            .filter(ReleasePointer.season == season, ReleasePointer.status == "active")
            .one_or_none()
        )
        if row is None:
            return None
        payload = dict(row.pointer_json or {})
        if not payload:
            payload = {
                "schema_version": "active_release_pointer_v1",
                "season": row.season,
                "status": row.status,
                "namespace": row.namespace,
                "release_id": row.release_id,
                "manifest_path": f"data/releases/{row.namespace}/release_bundle_manifest.json",
                "manifest_sha256": row.manifest_sha256,
                "activated_at": row.activated_at.isoformat(),
                "manifest_storage_uri": row.manifest_storage_uri,
            }
        return validate_active_pointer(payload, season=season)

    def write(self, payload: Mapping[str, Any], *, reason: str = "promote") -> dict[str, Any]:
        pointer = validate_active_pointer(payload)
        season = int(pointer["season"])
        existing = (
            self.session.query(ReleasePointer)
            .filter(ReleasePointer.season == season)
            .one_or_none()
        )
        if existing is not None:
            self.session.add(
                ReleasePointerHistory(
                    season=season,
                    pointer_json=dict(existing.pointer_json or {}),
                    reason=reason,
                    activated_at=existing.activated_at,
                )
            )
            existing.namespace = str(pointer["namespace"])
            existing.release_id = str(pointer["release_id"])
            existing.manifest_sha256 = str(pointer["manifest_sha256"])
            existing.manifest_storage_uri = pointer.get("manifest_storage_uri")
            existing.status = str(pointer["status"])
            existing.pointer_json = dict(pointer)
            existing.activated_at = utcnow()
            self.session.add(existing)
        else:
            self.session.add(
                ReleasePointer(
                    season=season,
                    namespace=str(pointer["namespace"]),
                    release_id=str(pointer["release_id"]),
                    manifest_sha256=str(pointer["manifest_sha256"]),
                    manifest_storage_uri=pointer.get("manifest_storage_uri"),
                    status=str(pointer["status"]),
                    pointer_json=dict(pointer),
                    activated_at=utcnow(),
                )
            )
        self.session.flush()
        return pointer

    def rollback(self, season: int, *, reason: str = "rollback") -> dict[str, Any] | None:
        history = (
            self.session.query(ReleasePointerHistory)
            .filter(ReleasePointerHistory.season == season)
            .order_by(ReleasePointerHistory.activated_at.desc())
            .first()
        )
        if history is None:
            return None
        previous = dict(history.pointer_json or {})
        if not previous:
            return None
        return self.write(previous, reason=reason)


class StatusOverlayPointerStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def read(self, season: int) -> dict[str, Any] | None:
        row = (
            self.session.query(StatusOverlayPointer)
            .filter(StatusOverlayPointer.season == season)
            .one_or_none()
        )
        if row is None:
            return None
        payload = dict(row.pointer_json or {})
        if not payload:
            payload = {
                "schema_version": "status_overlay_pointer_v1",
                "season": row.season,
                "status": "active",
                "overlay_hash": row.overlay_hash,
                "base_release_id": row.base_release_id,
                "base_manifest_sha256": row.base_manifest_sha256,
                "algorithm_version": row.algorithm_version,
                "activated_at": row.activated_at.isoformat(),
                "artifact_uri": row.artifact_uri,
                "adjustment_count": row.adjustment_count,
            }
        return payload

    def write(
        self,
        pointer: Mapping[str, Any],
        *,
        reason: str = "promote",
    ) -> dict[str, Any]:
        season = int(pointer["season"])
        snapshot = dict(pointer)
        existing = (
            self.session.query(StatusOverlayPointer)
            .filter(StatusOverlayPointer.season == season)
            .one_or_none()
        )
        if existing is not None:
            self.session.add(
                StatusOverlayPointerHistory(
                    season=season,
                    pointer_json=dict(existing.pointer_json or {}),
                    reason=reason,
                    activated_at=existing.activated_at,
                )
            )
            existing.overlay_hash = str(pointer["overlay_hash"])
            existing.base_release_id = str(pointer["base_release_id"])
            existing.base_manifest_sha256 = str(pointer["base_manifest_sha256"])
            existing.artifact_uri = str(pointer["artifact_uri"])
            existing.adjustment_count = int(pointer.get("adjustment_count") or 0)
            existing.algorithm_version = str(pointer.get("algorithm_version") or "")
            existing.pointer_json = snapshot
            existing.activated_at = utcnow()
            self.session.add(existing)
        else:
            self.session.add(
                StatusOverlayPointer(
                    season=season,
                    overlay_hash=str(pointer["overlay_hash"]),
                    base_release_id=str(pointer["base_release_id"]),
                    base_manifest_sha256=str(pointer["base_manifest_sha256"]),
                    artifact_uri=str(pointer["artifact_uri"]),
                    adjustment_count=int(pointer.get("adjustment_count") or 0),
                    algorithm_version=str(pointer.get("algorithm_version") or ""),
                    pointer_json=snapshot,
                    activated_at=utcnow(),
                )
            )
        self.session.flush()
        return snapshot

    def rollback(self, season: int, *, reason: str = "rollback") -> dict[str, Any] | None:
        history = (
            self.session.query(StatusOverlayPointerHistory)
            .filter(StatusOverlayPointerHistory.season == season)
            .order_by(StatusOverlayPointerHistory.activated_at.desc())
            .first()
        )
        if history is None:
            return None
        previous = dict(history.pointer_json or {})
        if not previous:
            return None
        return self.write(previous, reason=reason)


def pointer_payload_from_row(row: ReleasePointer) -> dict[str, Any]:
    return validate_active_pointer(
        dict(row.pointer_json or {}),
        season=row.season,
    )


def try_db_read_release_pointer(season: int) -> dict[str, Any] | None:
    """Best-effort DB read without an injected session (API/job paths)."""
    try:
        from src.app.persistence.database import get_session

        with get_session() as session:
            return ReleasePointerStore(session).read(season)
    except Exception:
        return None


def try_db_write_release_pointer(payload: Mapping[str, Any], *, reason: str = "promote") -> bool:
    try:
        from src.app.persistence.database import get_session

        with get_session() as session:
            ReleasePointerStore(session).write(payload, reason=reason)
        return True
    except Exception:
        return False
