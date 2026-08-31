"""Register simulation partition metadata for projection runs."""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.persistence.models import SimulationPartition


def register_run_partitions(
    session: Session,
    *,
    run_id: str,
    input_hash: str,
    player_count: int,
    mode: str,
) -> SimulationPartition:
    settings = get_settings()
    partition_key = f"{mode}:{input_hash[:16]}"
    digest = hashlib.sha256(f"{run_id}:{partition_key}:{player_count}".encode()).hexdigest()
    existing = (
        session.query(SimulationPartition)
        .filter(SimulationPartition.run_id == run_id, SimulationPartition.partition_key == partition_key)
        .one_or_none()
    )
    if existing is not None:
        return existing
    row = SimulationPartition(
        run_id=run_id,
        partition_key=partition_key,
        uri=f"fixture://partitions/{run_id}/{partition_key}",
        sha256=digest,
        draw_count=settings.simulation_draw_count,
    )
    session.add(row)
    session.flush()
    return row
