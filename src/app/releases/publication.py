"""Candidate-first, gate-before-write projection publication.

Publication used to interleave three concerns in one transaction: it wrote
candidate rows, then ran some gates, then swapped the active pointer, all in the
caller's single commit. A gate failure therefore left orphan ``projection_run`` /
``player_projection`` rows behind, and any error during the pointer swap could
leave the pointer pointing at a half-written release.

This module separates the phases:

1. **Compute** an immutable :class:`Candidate` (no database writes at all).
2. **Gate** the candidate — schema/completeness/bounds, scoring contract and
   artifact readiness run *before* anything is written; the partition gate runs
   inside the candidate savepoint and rolls the candidate write back on failure.
3. **Promote** in an isolated savepoint that only flips run status and the
   active pointer.

Every failure path retains a ``failed`` ``projection_run`` marker plus a
``PromotionEvent`` with ``promoted=False`` so a rejected candidate is auditable
rather than silently orphaned, and the previously active pointer is untouched.

Transaction scope note: the session lifecycle is owned by the caller (request or
job), so the phases use ``SAVEPOINT``s rather than independent commits. That
gives per-phase atomicity and rollback isolation within the caller's
transaction; it does not make the pointer swap durable ahead of the caller's
commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from src.app.logging import get_logger
from src.app.persistence.models import (
    ActiveProjectionPointer,
    PlayerProjection,
    ProjectionRun,
    PromotionEvent,
    utcnow,
)
from src.app.releases.gates import GateResult, validate_simulation_partitions
from src.app.releases.partitions import register_run_partitions

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateRow:
    player_id: str
    team: str | None
    opponent: str | None
    availability_probability: float | None
    mean_json: dict
    quantiles_json: dict


@dataclass(frozen=True)
class Candidate:
    """An immutable, fully computed release candidate awaiting gates."""

    mode: str
    season: int
    week: int | None
    run_id: str
    model_version: str
    input_hash: str
    manifest_uri: str
    artifact_mode: str
    partition_mode: str
    rows: tuple[CandidateRow, ...]
    as_of: datetime = field(default_factory=utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PublicationResult:
    run_id: str | None
    promoted: bool
    reason: str
    gates: dict
    already_active: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "promoted": self.promoted,
            "reason": self.reason,
            "gates": self.gates,
            "already_active": self.already_active,
        }


def merge_gates(named_gates: dict[str, GateResult]) -> tuple[bool, dict]:
    payload = {name: gate.to_dict() for name, gate in named_gates.items()}
    passed = all(gate.passed for gate in named_gates.values())
    return passed, payload


def active_pointer(
    session: Session, *, mode: str, season: int, week: int | None
) -> ActiveProjectionPointer | None:
    return (
        session.query(ActiveProjectionPointer)
        .filter(
            ActiveProjectionPointer.mode == mode,
            ActiveProjectionPointer.season == season,
            ActiveProjectionPointer.week == week,
        )
        .one_or_none()
    )


def swap_pointer(
    session: Session, *, mode: str, season: int, week: int | None, run_id: str
) -> tuple[str | None, bool]:
    """Point ``mode/season/week`` at ``run_id``; returns ``(previous_run_id, changed)``.

    Re-promoting the currently active run is a no-op: ``previous_run_id`` keeps
    pointing at the genuinely previous release instead of being overwritten with
    ``run_id`` itself, which used to break the rollback chain.
    """
    pointer = active_pointer(session, mode=mode, season=season, week=week)
    if pointer is None:
        session.add(
            ActiveProjectionPointer(mode=mode, season=season, week=week, run_id=run_id)
        )
        session.flush()
        return None, True
    if pointer.run_id == run_id:
        pointer.activated_at = utcnow()
        session.flush()
        return pointer.previous_run_id, False
    pointer.previous_run_id = pointer.run_id
    pointer.run_id = run_id
    pointer.activated_at = utcnow()
    session.flush()
    return pointer.previous_run_id, True


def _upsert_run(session: Session, candidate: Candidate, *, status: str) -> ProjectionRun:
    run = session.query(ProjectionRun).filter(ProjectionRun.id == candidate.run_id).one_or_none()
    if run is None:
        run = ProjectionRun(
            id=candidate.run_id,
            mode=candidate.mode,
            season=candidate.season,
            week=candidate.week,
            as_of=candidate.as_of,
            model_version=candidate.model_version,
            input_hash=candidate.input_hash,
            status=status,
            manifest_uri=candidate.manifest_uri,
            artifact_mode=candidate.artifact_mode,
        )
        session.add(run)
    else:
        run.status = status
        run.model_version = candidate.model_version
        run.input_hash = candidate.input_hash
        run.manifest_uri = candidate.manifest_uri
        run.artifact_mode = candidate.artifact_mode
    session.flush()
    return run


def _write_rows(session: Session, candidate: Candidate) -> int:
    existing = {
        row.player_id: row
        for row in session.query(PlayerProjection)
        .filter(PlayerProjection.run_id == candidate.run_id)
        .all()
    }
    written = 0
    for row in candidate.rows:
        current = existing.get(row.player_id)
        if current is None:
            session.add(
                PlayerProjection(
                    run_id=candidate.run_id,
                    player_id=row.player_id,
                    team=row.team,
                    opponent=row.opponent,
                    availability_probability=row.availability_probability,
                    mean_json=row.mean_json,
                    quantiles_json=row.quantiles_json,
                )
            )
        else:
            current.team = row.team
            current.opponent = row.opponent
            current.availability_probability = row.availability_probability
            current.mean_json = row.mean_json
            current.quantiles_json = row.quantiles_json
        written += 1
    session.flush()
    return written


def record_failure(
    session: Session,
    candidate: Candidate,
    *,
    reason: str,
    gates: dict,
) -> PublicationResult:
    """Retain a rejected candidate as an auditable ``failed`` run + promotion event."""
    run = session.query(ProjectionRun).filter(ProjectionRun.id == candidate.run_id).one_or_none()
    if run is None:
        run = ProjectionRun(
            id=candidate.run_id,
            mode=candidate.mode,
            season=candidate.season,
            week=candidate.week,
            as_of=candidate.as_of,
            model_version=candidate.model_version,
            input_hash=candidate.input_hash,
            status="failed",
            manifest_uri=candidate.manifest_uri,
            artifact_mode=candidate.artifact_mode,
        )
        session.add(run)
    else:
        run.status = "failed"
    pointer = active_pointer(
        session, mode=candidate.mode, season=candidate.season, week=candidate.week
    )
    previous_run_id = pointer.run_id if pointer is not None else None
    if previous_run_id == candidate.run_id:
        previous_run_id = pointer.previous_run_id if pointer is not None else None
    session.add(
        PromotionEvent(
            mode=candidate.mode,
            candidate_run_id=candidate.run_id,
            previous_run_id=previous_run_id,
            promoted=False,
            validation_json={
                "reason": reason,
                "gates": gates,
                "artifact_mode": candidate.artifact_mode,
                "manifest_uri": candidate.manifest_uri,
                **candidate.metadata,
            },
        )
    )
    session.flush()
    logger.warning(
        "promotion_rejected",
        mode=candidate.mode,
        run_id=candidate.run_id,
        reason=reason,
    )
    return PublicationResult(run_id=None, promoted=False, reason=reason, gates=gates)


def publish(
    session: Session,
    candidate: Candidate,
    *,
    gates: dict[str, GateResult],
    register_partitions: bool = True,
    validate_partitions: bool = True,
) -> PublicationResult:
    """Gate ``candidate`` and, only if every gate passes, promote it."""
    passed, gate_payload = merge_gates(gates)
    if not passed:
        return record_failure(session, candidate, reason="gate_failed", gates=gate_payload)

    pointer = active_pointer(
        session, mode=candidate.mode, season=candidate.season, week=candidate.week
    )
    if pointer is not None and pointer.run_id == candidate.run_id:
        existing = (
            session.query(ProjectionRun).filter(ProjectionRun.id == candidate.run_id).one_or_none()
        )
        if existing is not None and existing.status == "active":
            logger.info(
                "promotion_noop", mode=candidate.mode, run_id=candidate.run_id
            )
            return PublicationResult(
                run_id=candidate.run_id,
                promoted=True,
                reason="already_active",
                gates=gate_payload,
                already_active=True,
            )

    # Phase 1 — candidate write, isolated so a failure leaves no orphan rows.
    try:
        with session.begin_nested():
            _upsert_run(session, candidate, status="candidate")
            _write_rows(session, candidate)
            if register_partitions:
                register_run_partitions(
                    session,
                    run_id=candidate.run_id,
                    input_hash=candidate.input_hash,
                    player_count=len(candidate.rows),
                    mode=candidate.partition_mode,
                )
            if validate_partitions:
                partition_gate = validate_simulation_partitions(
                    session, run_id=candidate.run_id, input_hash=candidate.input_hash
                )
                gate_payload["partitions"] = partition_gate.to_dict()
                if not partition_gate.passed:
                    raise _CandidateRejected("partition_gate_failed")
    except _CandidateRejected as rejected:
        return record_failure(session, candidate, reason=str(rejected), gates=gate_payload)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "candidate_write_failed",
            mode=candidate.mode,
            run_id=candidate.run_id,
            error=str(exc),
        )
        gate_payload["candidate_write"] = {
            "passed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }
        return record_failure(session, candidate, reason="candidate_write_failed", gates=gate_payload)

    # Phase 2 — promote-only. Nothing here writes projection data.
    try:
        with session.begin_nested():
            run = (
                session.query(ProjectionRun)
                .filter(ProjectionRun.id == candidate.run_id)
                .one()
            )
            run.status = "active"
            previous_run_id, changed = swap_pointer(
                session,
                mode=candidate.mode,
                season=candidate.season,
                week=candidate.week,
                run_id=candidate.run_id,
            )
            session.add(
                PromotionEvent(
                    mode=candidate.mode,
                    candidate_run_id=candidate.run_id,
                    previous_run_id=previous_run_id,
                    promoted=True,
                    validation_json={
                        "gates": gate_payload,
                        "artifact_mode": candidate.artifact_mode,
                        "manifest_uri": candidate.manifest_uri,
                        "pointer_changed": changed,
                        **candidate.metadata,
                    },
                )
            )
            session.flush()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "pointer_swap_failed",
            mode=candidate.mode,
            run_id=candidate.run_id,
            error=str(exc),
        )
        gate_payload["pointer_swap"] = {
            "passed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }
        return record_failure(session, candidate, reason="pointer_swap_failed", gates=gate_payload)

    return PublicationResult(
        run_id=candidate.run_id, promoted=True, reason="promoted", gates=gate_payload
    )


class _CandidateRejected(RuntimeError):
    """Internal signal used to unwind the candidate savepoint on a gate failure."""
