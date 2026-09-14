"""Durable weekly_props snapshot catalog and closing-line promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from src.app.logging import get_logger
from src.app.persistence.models import (
    PlayerProjection,
    ProjectionRun,
    PromotionEvent,
    SourceSnapshot,
)
from src.app.releases.publication import (
    PublicationResult,
    activate_existing_run,
    active_pointer,
)
from src.ingest.props.contracts import ProviderSnapshot
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY
from src.projection.weekly_props.provenance import (
    WEEKLY_PROPS_POINTER_MODE,
    is_weekly_props_run,
)

logger = get_logger(__name__)

SNAPSHOT_ENDPOINT_PREFIX = "weekly_props"


def snapshot_endpoint(*, source: str, season: int, week: int) -> str:
    return f"{SNAPSHOT_ENDPOINT_PREFIX}:{source}:{season}:{week}"


def persist_provider_snapshots(session: Session, snapshots: list[ProviderSnapshot]) -> list[str]:
    """Write successful provider snapshots to the artifact store + source_snapshot.

    GitHub Actions runners discard the local ``data/props/snapshots`` tree when
    the job ends. Cataloguing here is what lets a later thin live scrape reuse
    the week's closing book.
    """
    uris: list[str] = []
    try:
        from src.app.artifacts.store import get_artifact_store

        store = get_artifact_store()
    except Exception as exc:  # noqa: BLE001 — local/fixture jobs still succeed
        logger.warning("weekly_props_snapshot_store_unavailable", error=str(exc))
        return uris

    for snap in snapshots:
        if not snap.success:
            continue
        payload = snap.to_dict()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        try:
            uri = store.put_json(
                payload,
                inputs={
                    "season": snap.season,
                    "week": snap.week,
                    "source": snap.source,
                },
                provenance={"kind": "weekly_props_provider_snapshot"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "weekly_props_snapshot_put_failed",
                source=snap.source,
                error=str(exc),
            )
            continue
        session.add(
            SourceSnapshot(
                endpoint=snapshot_endpoint(
                    source=snap.source, season=snap.season, week=snap.week
                ),
                request_params_json={
                    "season": snap.season,
                    "week": snap.week,
                    "source": snap.source,
                    "quote_count": len(snap.quotes),
                },
                fetched_at=snap.fetched_at,
                body_hash=digest,
                artifact_uri=uri,
                health_verdict="healthy" if snap.quotes else "degraded",
                is_complete=bool(snap.quotes),
            )
        )
        uris.append(uri)
    if uris:
        session.flush()
    return uris


def load_latest_provider_snapshots(
    session: Session,
    *,
    season: int,
    week: int,
    sources: tuple[str, ...] = ("draftkings", "fanduel"),
) -> list[ProviderSnapshot]:
    from src.app.artifacts.store import ArtifactError, get_artifact_store

    try:
        store = get_artifact_store()
    except Exception as exc:  # noqa: BLE001
        logger.warning("weekly_props_snapshot_store_unavailable", error=str(exc))
        return []

    loaded: list[ProviderSnapshot] = []
    for source in sources:
        row = (
            session.query(SourceSnapshot)
            .filter(SourceSnapshot.endpoint == snapshot_endpoint(source=source, season=season, week=week))
            .order_by(SourceSnapshot.fetched_at.desc())
            .first()
        )
        if row is None:
            continue
        try:
            payload = store.get_json(row.artifact_uri)
            loaded.append(ProviderSnapshot.from_dict(payload))
        except (ArtifactError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "weekly_props_snapshot_load_failed",
                source=source,
                uri=row.artifact_uri,
                error=str(exc),
            )
    return loaded


def richer_stored_snapshots(
    stored: list[ProviderSnapshot],
    live: list[ProviderSnapshot],
    *,
    min_extra_quotes: int = 50,
    min_age: timedelta = timedelta(minutes=5),
) -> list[ProviderSnapshot]:
    """Keep stored snapshots only when they are an older, fuller book than live."""
    if not stored or not live:
        return []
    live_quotes = sum(len(snap.quotes) for snap in live if snap.success)
    live_fetched = max((snap.fetched_at for snap in live if snap.success), default=None)
    stored_quotes = sum(len(snap.quotes) for snap in stored if snap.success)
    stored_fetched = max((snap.fetched_at for snap in stored if snap.success), default=None)
    if live_fetched is None or stored_fetched is None:
        return []
    if stored_fetched >= live_fetched - min_age:
        return []
    if stored_quotes < live_quotes + min_extra_quotes:
        return []
    return [snap for snap in stored if snap.success]


def _gates_all_passed(payload: dict[str, Any] | None) -> bool:
    gates = (payload or {}).get("gates") if isinstance(payload, dict) else None
    if not isinstance(gates, dict) or not gates:
        return False
    for result in gates.values():
        if isinstance(result, dict) and not result.get("passed", False):
            return False
    return True


def latest_passing_unpublished_weekly_props(
    session: Session, *, season: int, week: int
) -> ProjectionRun | None:
    """Return the newest weekly_props candidate that passed gates but is not active."""
    runs = (
        session.query(ProjectionRun)
        .filter(
            ProjectionRun.season == season,
            ProjectionRun.week == week,
            ProjectionRun.status.in_(("shadow", "candidate")),
        )
        .order_by(ProjectionRun.as_of.desc())
        .all()
    )
    min_players = DEFAULT_WEEKLY_POLICY.min_candidate_players
    for run in runs:
        if not is_weekly_props_run(run):
            continue
        row_count = (
            session.query(PlayerProjection)
            .filter(PlayerProjection.run_id == run.id)
            .count()
        )
        if row_count < min_players:
            continue
        event = (
            session.query(PromotionEvent)
            .filter(PromotionEvent.candidate_run_id == run.id)
            .order_by(PromotionEvent.created_at.desc())
            .first()
        )
        if event is None:
            continue
        validation = event.validation_json or {}
        reason = str(validation.get("reason") or "")
        if reason in {"shadow_only", "activated_closing_line"} or _gates_all_passed(validation):
            return run
    return None


def activate_last_passing_weekly_props(
    session: Session, *, season: int, week: int
) -> PublicationResult | None:
    """Promote a previously persisted passing candidate for this NFL week."""
    pointer = active_pointer(
        session, mode=WEEKLY_PROPS_POINTER_MODE, season=season, week=week
    )
    if pointer is not None:
        current = (
            session.query(ProjectionRun).filter(ProjectionRun.id == pointer.run_id).one_or_none()
        )
        if current is not None and is_weekly_props_run(current) and current.status == "active":
            return None
    run = latest_passing_unpublished_weekly_props(session, season=season, week=week)
    if run is None:
        return None
    logger.info(
        "weekly_props_activate_closing_line",
        run_id=run.id,
        season=season,
        week=week,
    )
    return activate_existing_run(
        session,
        run,
        reason="activated_closing_line",
    )
