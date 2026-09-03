"""QB projection final-repair package.

Leakage-safe experimental arms for QB team-volume allocation and multi-season
rate priors. Production compose defaults are unchanged until a gated candidate
is explicitly published into a new immutable namespace.
"""

from src.projection.qb_repair.allocation import (
    estimate_starter_backup_shares,
    reconcile_qb_volume_with_allocation,
)
from src.projection.qb_repair.rate_prior import (
    apply_qb_rate_prior,
    build_qb_rate_priors,
    classify_qb_archetype,
)

__all__ = [
    "estimate_starter_backup_shares",
    "reconcile_qb_volume_with_allocation",
    "apply_qb_rate_prior",
    "build_qb_rate_priors",
    "classify_qb_archetype",
]
