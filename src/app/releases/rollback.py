"""Restore the previous active projection pointer after a failed or superseded promotion."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.app.persistence.models import ActiveProjectionPointer, PromotionEvent, utcnow


class ProjectionRollbackService:
    """Pointer-only rollback.

    Neither the candidate nor the restored bundle is mutated: no
    ``projection_run``, ``player_projection`` or ``simulation_partition`` row is
    touched. Only the pointer moves, so both bundles stay byte-identical and a
    rollback can itself be rolled forward again.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def rollback(
        self,
        mode: str,
        season: int,
        week: int | None,
        *,
        reason: str = "manual_rollback",
    ) -> str | None:
        pointer = (
            self.session.query(ActiveProjectionPointer)
            .filter(
                ActiveProjectionPointer.mode == mode,
                ActiveProjectionPointer.season == season,
                ActiveProjectionPointer.week == week,
            )
            .one_or_none()
        )
        if pointer is None or not pointer.previous_run_id:
            return None
        current_run_id = pointer.run_id
        restored_run_id = pointer.previous_run_id
        if restored_run_id == current_run_id:
            # A corrupted chain (previous == current) has nothing to restore;
            # swapping would be a silent no-op reported as success.
            return None
        pointer.run_id = restored_run_id
        pointer.previous_run_id = current_run_id
        pointer.activated_at = utcnow()
        self.session.add(
            PromotionEvent(
                mode=mode,
                candidate_run_id=restored_run_id,
                previous_run_id=current_run_id,
                promoted=True,
                validation_json={
                    "derivation": "rollback",
                    "reason": reason,
                    "restored_run_id": restored_run_id,
                    "superseded_run_id": current_run_id,
                },
            )
        )
        self.session.flush()
        return restored_run_id
