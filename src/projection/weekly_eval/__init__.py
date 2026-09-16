"""Research/shadow Role 2 Vegas weekly props evaluation comparator.

Does not blend market into the model (Role 3 forbidden). Does not promote.
Does not change APP_PROJECTION_SOURCE or weekly_latent allocation.
"""

from src.projection.weekly_eval.comparator import compare_shadow_to_vegas
from src.projection.weekly_eval.errors import (
    MissingAsOfError,
    MissingKickoffError,
    OutcomeFeatureLeakageError,
    PostKickoffSnapshotError,
    Role3BlendForbiddenError,
    WeeklyEvalError,
)
from src.projection.weekly_eval.schema import (
    PropSnapshot,
    load_outcomes,
    load_prop_snapshots,
    load_shadow_board,
    prop_snapshot_from_row,
)

__all__ = [
    "MissingAsOfError",
    "MissingKickoffError",
    "OutcomeFeatureLeakageError",
    "PostKickoffSnapshotError",
    "PropSnapshot",
    "Role3BlendForbiddenError",
    "WeeklyEvalError",
    "compare_shadow_to_vegas",
    "load_outcomes",
    "load_prop_snapshots",
    "load_shadow_board",
    "prop_snapshot_from_row",
]
