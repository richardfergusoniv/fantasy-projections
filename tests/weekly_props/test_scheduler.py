"""Scheduler registration for weekly props jobs."""

from src.app.jobs.handlers import JOB_HANDLERS
from src.app.jobs.scheduler import LONG_RUNNING_JOBS, SCHEDULE_SLOTS


def test_weekly_props_jobs_registered():
    for name in (
        "weekly-props-open",
        "weekly-props-market-close",
        "weekly-props-refresh-thu",
        "weekly-props-refresh-sat",
        "weekly-props-refresh-sun",
    ):
        assert name in SCHEDULE_SLOTS
        assert name in LONG_RUNNING_JOBS
        assert name in JOB_HANDLERS


def test_market_close_is_daily_2300_pt():
    slot = SCHEDULE_SLOTS["weekly-props-market-close"]
    assert slot.hour == 23
    assert slot.minute == 0
    assert slot.days_of_week == frozenset(range(7))
    assert "23:00" in slot.description
