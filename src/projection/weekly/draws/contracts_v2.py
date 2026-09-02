"""Weekly mixture contract v2: row states, denominators, and label derivation."""

from __future__ import annotations

from enum import Enum
from typing import Any

import polars as pl

CONTRACT_VERSION_V2 = "weekly_mixture_contract_v2"


class RowOutcomeState(str, Enum):
    BYE_OR_NO_SCHEDULED_GAME = "bye_or_no_scheduled_game"
    NOT_ON_RECOVERABLE_ROSTER = "not_on_recoverable_roster_at_cutoff"
    ACTIVE_STATUS_UNKNOWN = "active_status_unknown"
    INACTIVE = "inactive"
    ACTIVE_NO_OFFENSIVE_PARTICIPATION = "active_no_offensive_participation"
    PARTICIPATED_ZERO_POSITIVE_USAGE = "participated_zero_positive_usage"
    POSITIVE_USAGE = "positive_usage"
    OUTCOME_MISSING_OR_INCOMPLETE = "outcome_missing_or_source_incomplete"


NOT_ON_ROSTER_STATUSES = frozenset({"DEV", "CUT", "RET", "EXE", "TRC", "TRD", "E01"})
INACTIVE_STATUSES = frozenset({"INA"})
RESERVE_STATUSES = frozenset({"RES", "PUP", "NFI", "SUS"})
ACTIVE_STATUSES = frozenset({"ACT"})


def _col(df: pl.DataFrame, name: str, default: float = 0.0) -> pl.Expr:
    if name in df.columns:
        return pl.col(name).fill_null(default)
    return pl.lit(default)


def observed_active_from_roster_status(status: str | None) -> bool | None:
    """Ground-truth active label from roster evidence only (never play_prob)."""
    if status is None:
        return None
    s = str(status).upper()
    if s in ACTIVE_STATUSES:
        return True
    if s in INACTIVE_STATUSES or s in RESERVE_STATUSES:
        return False
    if s in NOT_ON_ROSTER_STATUSES:
        return None
    return None


def positive_usage_for_position(
    position: str,
    *,
    targets: float,
    carries: float,
    attempts: float,
) -> bool:
    """Position-appropriate positive usage given participation."""
    pos = str(position).upper()
    t, c, a = float(targets), float(carries), float(attempts)
    if pos == "QB":
        return a > 0 or c > 0
    if pos == "RB":
        return c > 0 or t > 0
    if pos in {"WR", "TE"}:
        return t > 0
    return (t + c + a) > 0


def classify_row_outcome_state(
    *,
    has_scheduled_game: bool,
    roster_status: str | None,
    active_label: bool | None,
    participated_label: bool | None,
    positive_usage_label: bool | None,
    outcome_missing: bool,
) -> RowOutcomeState:
    if not has_scheduled_game:
        return RowOutcomeState.BYE_OR_NO_SCHEDULED_GAME
    if roster_status is not None and str(roster_status).upper() in NOT_ON_ROSTER_STATUSES:
        return RowOutcomeState.NOT_ON_RECOVERABLE_ROSTER
    if outcome_missing:
        return RowOutcomeState.OUTCOME_MISSING_OR_INCOMPLETE
    if active_label is None:
        return RowOutcomeState.ACTIVE_STATUS_UNKNOWN
    if active_label is False:
        return RowOutcomeState.INACTIVE
    if participated_label is False:
        return RowOutcomeState.ACTIVE_NO_OFFENSIVE_PARTICIPATION
    if positive_usage_label is False:
        return RowOutcomeState.PARTICIPATED_ZERO_POSITIVE_USAGE
    if positive_usage_label is True:
        return RowOutcomeState.POSITIVE_USAGE
    return RowOutcomeState.ACTIVE_STATUS_UNKNOWN


def derive_event_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Derive v2 event labels with explicit denominators.

    Labels:
    - ``active_label``: scheduled + rostered + observed roster active status
    - ``participated_label``: denominator active_label == True
    - ``positive_usage_label``: denominator participated_label == True
    """
    status = (
        pl.col("roster_status").cast(pl.Utf8)
        if "roster_status" in df.columns
        else pl.lit(None).cast(pl.Utf8)
    )
    snaps = _col(df, "offense_snaps")
    targets = _col(df, "targets")
    carries = _col(df, "carries")
    attempts = _col(df, "attempts")
    has_snaps_source = "offense_snaps" in df.columns

    # Active from roster status only — never is_out/play_prob.
    active_from_status = (
        pl.when(status.is_in(list(ACTIVE_STATUSES)))
        .then(True)
        .when(status.is_in(list(INACTIVE_STATUSES | RESERVE_STATUSES)))
        .then(False)
        .otherwise(None)
        .alias("_active_from_status")
    )

    out = df.with_columns(
        [
            active_from_status,
            snaps.alias("_snaps"),
            targets.alias("_targets"),
            carries.alias("_carries"),
            attempts.alias("_attempts"),
        ]
    )

    out = out.with_columns(
        [
            pl.when(~pl.col("has_scheduled_game"))
            .then(None)
            .when(pl.col("roster_status").is_in(list(NOT_ON_ROSTER_STATUSES)))
            .then(None)
            .otherwise(pl.col("_active_from_status"))
            .alias("active_label"),
        ]
    )

    # Participation from snaps when source exists; else unknown (not negative).
    out = out.with_columns(
        pl.when(~pl.col("has_scheduled_game"))
        .then(None)
        .when(pl.col("active_label") != True)  # noqa: E712
        .then(None)
        .when(~pl.lit(has_snaps_source))
        .then(None)
        .otherwise(pl.col("_snaps") > 0)
        .alias("participated_label")
    )

    # Positive usage conditional on participation.
    pos_usage_exprs = []
    for pos in ("QB", "RB", "WR", "TE"):
        if pos == "QB":
            cond = (pl.col("_attempts") > 0) | (pl.col("_carries") > 0)
        elif pos == "RB":
            cond = (pl.col("_carries") > 0) | (pl.col("_targets") > 0)
        else:
            cond = pl.col("_targets") > 0
        pos_usage_exprs.append(
            pl.when(pl.col("position") == pos).then(cond).otherwise(pl.lit(False))
        )

    usage_cond = pos_usage_exprs[0]
    for expr in pos_usage_exprs[1:]:
        usage_cond = usage_cond | expr

    out = out.with_columns(
        pl.when(~pl.col("has_scheduled_game"))
        .then(None)
        .when(pl.col("participated_label") != True)  # noqa: E712
        .then(None)
        .when(pl.col("participated_label").is_null())
        .then(None)
        .otherwise(usage_cond)
        .alias("positive_usage_label")
    )

    missing_outcome = (
        pl.col("has_scheduled_game")
        & pl.col("active_label").is_not_null()
        & pl.lit(has_snaps_source)
        & pl.col("offense_snaps").is_null()
        & pl.col("targets").is_null()
        & pl.col("carries").is_null()
        & pl.col("attempts").is_null()
    )
    out = out.with_columns(missing_outcome.alias("outcome_missing"))

    states: list[str] = []
    for row in out.select(
        [
            "has_scheduled_game",
            "roster_status",
            "active_label",
            "participated_label",
            "positive_usage_label",
            "outcome_missing",
        ]
    ).iter_rows(named=True):
        states.append(
            classify_row_outcome_state(
                has_scheduled_game=bool(row["has_scheduled_game"]),
                roster_status=row["roster_status"],
                active_label=row["active_label"],
                participated_label=row["participated_label"],
                positive_usage_label=row["positive_usage_label"],
                outcome_missing=bool(row["outcome_missing"]),
            ).value
        )
    out = out.with_columns(pl.Series("row_outcome_state", states)).drop(
        ["_active_from_status", "_snaps", "_targets", "_carries", "_attempts"]
    )
    return out


def event_denominator_mask(event: str, frame: pl.DataFrame) -> pl.Series:
    """Boolean mask for rows in the declared event denominator."""
    if event == "active_label":
        return frame["has_scheduled_game"] & frame["active_label"].is_not_null()
    if event == "participated_label":
        return frame["active_label"] == True  # noqa: E712
    if event == "positive_usage_label":
        return frame["participated_label"] == True  # noqa: E712
    raise ValueError(f"unknown event: {event}")
