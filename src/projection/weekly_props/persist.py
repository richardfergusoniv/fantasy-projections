"""Durable weekly_props snapshot catalog and closing-line promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
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
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY, WeeklyPropsPolicy
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

    Fail-closed **per provider**: a successful scrape is only catalogued as
    healthy/complete after the artifact backend confirms the object is
    readable (verify-after-upload). A missing or unreadable blob skips that
    provider — no healthy ``source_snapshot`` row for the void URI — without
    aborting other providers in the same call (provider isolation).
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
            # Defense in depth: put_json already verifies, but catalog rows
            # must never be written without an explicit readable check.
            store.verify_readable(uri)
        except Exception as exc:  # noqa: BLE001 — fail closed for this provider only
            logger.error(
                "weekly_props_snapshot_put_failed",
                source=snap.source,
                season=snap.season,
                week=snap.week,
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


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def age_hours(value: datetime | None, *, now: datetime | None = None) -> float | None:
    if value is None:
        return None
    clock = _aware_utc(now or datetime.now(UTC))
    try:
        return (_aware_utc(value) - clock).total_seconds() / -3600.0
    except (TypeError, ValueError):
        return None


def richer_stored_snapshots(
    stored: list[ProviderSnapshot],
    live: list[ProviderSnapshot],
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
    now: datetime | None = None,
    min_extra_quotes: int | None = None,
    min_age: timedelta | None = None,
) -> list[ProviderSnapshot]:
    """Keep stored snapshots when they are a fuller older book, or live failed.

    A total live-scrape failure (no successful snapshots) is the case that most
    needs the stored closing book. When live succeeded, stored must be older
    than ``min_age`` and have ``min_extra_quotes`` more quotes.
    """
    stored_ok = [snap for snap in stored if snap.success]
    if not stored_ok:
        return []
    extra = (
        policy.min_stored_quote_advantage if min_extra_quotes is None else min_extra_quotes
    )
    age_floor = (
        timedelta(minutes=policy.min_stored_snapshot_age_minutes)
        if min_age is None
        else min_age
    )
    clock = _aware_utc(now or datetime.now(UTC))
    fresh_stored: list[ProviderSnapshot] = []
    for snap in stored_ok:
        age_h = age_hours(snap.fetched_at, now=clock)
        if age_h is None or age_h > policy.max_closing_line_age_hours:
            continue
        fresh_stored.append(snap)
    if not fresh_stored:
        return []

    live_ok = [snap for snap in live if snap.success]
    if not live_ok:
        return fresh_stored

    live_quotes = sum(len(snap.quotes) for snap in live_ok)
    stored_quotes = sum(len(snap.quotes) for snap in fresh_stored)
    try:
        live_fetched = max(_aware_utc(snap.fetched_at) for snap in live_ok)
        stored_fetched = max(_aware_utc(snap.fetched_at) for snap in fresh_stored)
    except (TypeError, ValueError):
        return []
    if stored_fetched >= live_fetched - age_floor:
        return []
    if stored_quotes < live_quotes + extra:
        return []
    return fresh_stored


def _gates_all_passed(payload: dict[str, Any] | None) -> bool:
    gates = (payload or {}).get("gates") if isinstance(payload, dict) else None
    if not isinstance(gates, dict) or not gates:
        return False
    for result in gates.values():
        if isinstance(result, dict) and not result.get("passed", False):
            return False
    return True


def latest_passing_unpublished_weekly_props(
    session: Session,
    *,
    season: int,
    week: int,
    now: datetime | None = None,
    max_age_hours: float | None = None,
) -> ProjectionRun | None:
    """Return the newest weekly_props candidate that passed gates but is not active.

    Candidates older than ``max_age_hours`` (default
    ``policy.max_closing_line_age_hours``) are skipped so a Tuesday book cannot
    become Sunday's live Vegas board.
    """
    clock = _aware_utc(now or datetime.now(UTC))
    bound = (
        DEFAULT_WEEKLY_POLICY.max_closing_line_age_hours
        if max_age_hours is None
        else max_age_hours
    )
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
        age_h = age_hours(run.as_of, now=clock)
        if age_h is None or age_h > bound:
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
    session: Session,
    *,
    season: int,
    week: int,
    now: datetime | None = None,
) -> PublicationResult | None:
    """Promote a previously persisted passing candidate for this NFL week."""
    clock = _aware_utc(now or datetime.now(UTC))
    pointer = active_pointer(
        session, mode=WEEKLY_PROPS_POINTER_MODE, season=season, week=week
    )
    if pointer is not None:
        current = (
            session.query(ProjectionRun).filter(ProjectionRun.id == pointer.run_id).one_or_none()
        )
        if current is not None and is_weekly_props_run(current) and current.status == "active":
            return None
    run = latest_passing_unpublished_weekly_props(
        session, season=season, week=week, now=clock
    )
    if run is None:
        return None
    age_h = age_hours(run.as_of, now=clock) or 0.0
    freshness_warnings: list[str] = []
    if age_h > DEFAULT_WEEKLY_POLICY.max_snapshot_age_hours:
        freshness_warnings.append(f"stale_snapshot:closing_line:{age_h:.1f}h")
    as_of_iso = None
    if run.as_of is not None:
        as_of_iso = _aware_utc(run.as_of).isoformat()
    logger.info(
        "weekly_props_activate_closing_line",
        run_id=run.id,
        season=season,
        week=week,
        closing_line_age_hours=round(age_h, 3),
    )
    return activate_existing_run(
        session,
        run,
        reason="activated_closing_line",
        gates={
            "closing_line_freshness": {
                "passed": True,
                "failures": [],
                "warnings": freshness_warnings,
            }
        },
        extra={
            "closing_line_as_of": as_of_iso,
            "closing_line_age_hours": round(age_h, 3),
            "max_closing_line_age_hours": DEFAULT_WEEKLY_POLICY.max_closing_line_age_hours,
        },
    )
