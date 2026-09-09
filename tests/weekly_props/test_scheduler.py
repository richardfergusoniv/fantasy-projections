"""Scheduler registration for weekly props jobs."""

from src.app.jobs.handlers import JOB_HANDLERS
from src.app.jobs.scheduler import LONG_RUNNING_JOBS, SCHEDULE_SLOTS


def test_weekly_props_jobs_registered():
    for name in (
        "weekly-props-open",
        "weekly-props-refresh-thu",
        "weekly-props-refresh-sat",
        "weekly-props-refresh-sun",
    ):
        assert name in SCHEDULE_SLOTS
        assert name in LONG_RUNNING_JOBS
        assert name in JOB_HANDLERS
