"""Leakage gates for the Role 2 comparator.

Reuses the weekly inference denylist so same-week box/share/team aggregates
cannot be smuggled in as prediction features. Does not import weekly_latent.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.projection.weekly.draws.feature_outcome_split import (
    is_allowed_prediction_column,
)
from src.projection.weekly_eval.errors import (
    OutcomeFeatureLeakageError,
    Role3BlendForbiddenError,
)

# Identity / market keys used to join boards, snapshots, and labels.
JOIN_KEYS: tuple[str, ...] = ("player_id", "season", "week", "market")

# Columns that may appear on a shadow board or snapshot without being outcomes.
EVAL_PASSTHROUGH_COLUMNS: frozenset[str] = frozenset(
    {
        *JOIN_KEYS,
        "player_name",
        "position",
        "team",
        "opponent",
        "model_mean",
        "model_std",
        "model_p_over",
        "line",
        "implied_p_over",
        "implied_mean",
        "as_of",
        "kickoff_at",
        "source",
        "actual",
    }
)


def assert_prediction_frame_has_no_outcomes(columns: Sequence[str]) -> None:
    """Fail closed if same-week outcome columns appear on a prediction frame.

    Join keys, shadow-board means, and snapshot fields are passthrough. Lagged
    suffixes remain allowed via ``is_allowed_prediction_column``.
    """
    blocked = [
        str(c)
        for c in columns
        if str(c) not in EVAL_PASSTHROUGH_COLUMNS and not is_allowed_prediction_column(str(c))
    ]
    if blocked:
        raise OutcomeFeatureLeakageError(
            "prediction frame contains same-week outcome columns: "
            + ", ".join(sorted(blocked)[:20])
        )


def assert_no_role3_blend(blend_weights: Mapping[str, Any] | None) -> None:
    """Role 3 ensemble is out of scope until a market-free challenger is reported."""
    if blend_weights:
        raise Role3BlendForbiddenError(
            "Role 3 market blend is forbidden until a market-free challenger is reported; "
            f"got weights {dict(blend_weights)}"
        )
