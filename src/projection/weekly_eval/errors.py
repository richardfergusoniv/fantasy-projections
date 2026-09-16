"""Fail-closed errors for the Role 2 Vegas weekly props comparator."""
from __future__ import annotations


class WeeklyEvalError(ValueError):
    """Base error for the shadow evaluation comparator."""


class MissingAsOfError(WeeklyEvalError):
    """A prop snapshot row is missing a usable as_of timestamp."""


class MissingKickoffError(WeeklyEvalError):
    """A prop snapshot row is missing kickoff_at, so later-season leak cannot be ruled out."""


class PostKickoffSnapshotError(WeeklyEvalError):
    """as_of is after kickoff_at — later-season / post-kickoff market leaked into the row."""


class OutcomeFeatureLeakageError(WeeklyEvalError):
    """Same-week realized outcome columns were present on a prediction frame."""


class Role3BlendForbiddenError(WeeklyEvalError):
    """Role 3 market blend is forbidden until a market-free challenger is reported."""
