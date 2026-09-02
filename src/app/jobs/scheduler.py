"""Scheduled worker entrypoints and the authoritative America/Los_Angeles schedule.

The schedule is expressed in local (America/Los_Angeles) wall-clock time because
that is how the operating calendar is defined, but every function in this module
returns *UTC instants*. Conversion goes through :func:`local_wall_time_to_utc`,
which uses ``zoneinfo`` so the two annual DST transitions are handled explicitly
rather than by assuming a fixed UTC offset.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time as wall_time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from src.app.jobs.handlers import JOB_HANDLERS
from src.app.jobs.runner import JobRunner
from src.app.logging import bind_correlation_id, configure_logging, get_logger
from src.app.config import get_settings
from src.app.ops.alerts import send_ops_alert
from src.app.persistence.database import assert_expected_revision, get_job_session, get_session
from src.app.persistence.job_outbox import JobOutboxService

LONG_RUNNING_JOBS = frozenset(
    {
        "daily-refresh",
        "sunday-early",
        "sunday-afternoon",
        "sunday-night",
        "monday-night",
        "weekly-close-preliminary",
        "weekly-correction",
        "full-release",
    }
)

TZ = ZoneInfo("America/Los_Angeles")
logger = get_logger(__name__)

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)
EVERY_DAY = frozenset(range(7))
EVERY_DAY_EXCEPT_SUNDAY = frozenset(EVERY_DAY - {SUNDAY})

#: How far past a scheduled instant a slot is still considered runnable. A cron
#: that fires late (host restart, queue backlog) still picks the slot up; a slot
#: older than this is skipped rather than run at a meaningless time.
DEFAULT_GRACE = timedelta(hours=6)

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class ScheduleSlot:
    """One recurring job slot, defined in America/Los_Angeles wall-clock time."""

    job_name: str
    days_of_week: frozenset[int]
    hour: int
    minute: int
    on_demand: bool = False

    @property
    def description(self) -> str:
        if self.on_demand:
            return "On demand only (never auto-scheduled)"
        if self.days_of_week == EVERY_DAY:
            days = "Daily"
        elif self.days_of_week == EVERY_DAY_EXCEPT_SUNDAY:
            days = "Daily except Sunday"
        else:
            days = "/".join(_DAY_NAMES[day] for day in sorted(self.days_of_week))
        return f"{days} {self.hour:02d}:{self.minute:02d} America/Los_Angeles"


SCHEDULE_SLOTS: dict[str, ScheduleSlot] = {
    slot.job_name: slot
    for slot in (
        ScheduleSlot("daily-refresh", EVERY_DAY_EXCEPT_SUNDAY, 17, 0),
        ScheduleSlot("sunday-early", frozenset({SUNDAY}), 8, 45),
        ScheduleSlot("sunday-afternoon", frozenset({SUNDAY}), 11, 45),
        ScheduleSlot("sunday-night", frozenset({SUNDAY}), 16, 0),
        ScheduleSlot("monday-night", frozenset({MONDAY}), 16, 0),
        ScheduleSlot("weekly-close-preliminary", frozenset({TUESDAY}), 5, 0),
        ScheduleSlot("weekly-correction", frozenset({WEDNESDAY}), 17, 0),
        ScheduleSlot("full-release", frozenset(), 0, 0, on_demand=True),
    )
}

#: Human-readable rendering, kept for the ``list`` CLI command and log records.
SCHEDULES: dict[str, str] = {name: slot.description for name, slot in SCHEDULE_SLOTS.items()}


@dataclass(frozen=True)
class DueSlot:
    """A schedule occurrence that has come due and not yet been satisfied."""

    job_name: str
    scheduled_at: datetime
    slot_key: str


def slot_key(job_name: str, scheduled_at: datetime) -> str:
    """Stable idempotency key for one occurrence of one job."""
    instant = scheduled_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"schedule:{job_name}:{instant}"


def local_wall_time_to_utc(local_naive: datetime) -> datetime:
    """Convert an America/Los_Angeles wall-clock time to a UTC instant.

    Ambiguous wall times (the repeated hour on fall-back) resolve to the *first*
    occurrence, so a daily slot never fires twice for one calendar date.
    Nonexistent wall times (the skipped hour on spring-forward) resolve to the
    instant at which the gap closes, so a slot is never silently dropped.
    """
    aware = local_naive.replace(tzinfo=TZ, fold=0)
    return aware.astimezone(UTC)


def scheduled_instant(slot: ScheduleSlot, local_date) -> datetime:
    """UTC instant at which ``slot`` fires on ``local_date`` (local calendar date)."""
    return local_wall_time_to_utc(datetime.combine(local_date, wall_time(slot.hour, slot.minute)))


def next_run_utc(slot: ScheduleSlot, now_utc: datetime) -> datetime | None:
    """First UTC instant strictly after ``now_utc`` at which ``slot`` fires."""
    if slot.on_demand or not slot.days_of_week:
        return None
    now_utc = _as_utc(now_utc)
    today_local = now_utc.astimezone(TZ).date()
    for offset in range(0, 9):
        day = today_local + timedelta(days=offset)
        if day.weekday() not in slot.days_of_week:
            continue
        candidate = scheduled_instant(slot, day)
        if candidate > now_utc:
            return candidate
    return None


def previous_run_utc(slot: ScheduleSlot, now_utc: datetime) -> datetime | None:
    """Most recent UTC instant at or before ``now_utc`` at which ``slot`` fired."""
    if slot.on_demand or not slot.days_of_week:
        return None
    now_utc = _as_utc(now_utc)
    today_local = now_utc.astimezone(TZ).date()
    for offset in range(0, 9):
        day = today_local - timedelta(days=offset)
        if day.weekday() not in slot.days_of_week:
            continue
        candidate = scheduled_instant(slot, day)
        if candidate <= now_utc:
            return candidate
    return None


def next_due_job(now_utc: datetime) -> DueSlot | None:
    """The soonest upcoming scheduled occurrence across every recurring slot."""
    upcoming: list[DueSlot] = []
    for slot in SCHEDULE_SLOTS.values():
        instant = next_run_utc(slot, now_utc)
        if instant is not None:
            upcoming.append(DueSlot(slot.job_name, instant, slot_key(slot.job_name, instant)))
    if not upcoming:
        return None
    return min(upcoming, key=lambda item: (item.scheduled_at, item.job_name))


def due_slots(
    now_utc: datetime,
    last_runs: Mapping[str, datetime | None],
    *,
    grace: timedelta = DEFAULT_GRACE,
) -> list[DueSlot]:
    """Slots whose latest occurrence is due and not yet satisfied by ``last_runs``.

    ``last_runs`` maps job name to the completion time of its last *successful*
    run. A slot is due when its most recent occurrence is within ``grace`` of now
    and no successful run has happened at or after that occurrence, which makes
    it impossible both to double-run a slot and to silently skip one.
    """
    now_utc = _as_utc(now_utc)
    due: list[DueSlot] = []
    for slot in SCHEDULE_SLOTS.values():
        scheduled = previous_run_utc(slot, now_utc)
        if scheduled is None:
            continue
        if now_utc - scheduled > grace:
            continue
        last = last_runs.get(slot.job_name)
        if last is not None and _as_utc(last) >= scheduled:
            continue
        due.append(DueSlot(slot.job_name, scheduled, slot_key(slot.job_name, scheduled)))
    return sorted(due, key=lambda item: (item.scheduled_at, item.job_name))


def last_successful_runs(session: Session) -> dict[str, datetime]:
    """Completion time of the most recent successful run per job name."""
    from src.app.persistence.models import JobRun

    rows = (
        session.query(JobRun)
        .filter(JobRun.status == "succeeded")
        .order_by(JobRun.started_at.asc())
        .all()
    )
    latest: dict[str, datetime] = {}
    for row in rows:
        completed = row.finished_at or row.started_at
        if completed is None:
            continue
        completed = _as_utc(completed)
        current = latest.get(row.job_name)
        if current is None or completed > current:
            latest[row.job_name] = completed
    return latest


def last_completed_slot(session: Session, now_utc: datetime | None = None) -> dict | None:
    """The most recently satisfied schedule occurrence, for operational display."""
    now_utc = _as_utc(now_utc or datetime.now(UTC))
    completions = last_successful_runs(session)
    best: dict | None = None
    for job_name, completed in completions.items():
        slot = SCHEDULE_SLOTS.get(job_name)
        if slot is None or slot.on_demand:
            continue
        scheduled = previous_run_utc(slot, completed)
        if scheduled is None:
            continue
        candidate = {
            "job_name": job_name,
            "scheduled_at": scheduled.isoformat(),
            "completed_at": completed.isoformat(),
        }
        if best is None or completed > datetime.fromisoformat(best["completed_at"]):
            best = candidate
    return best


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def run_job(job_name: str, *, idempotency_key: str | None = None) -> dict:
    bind_correlation_id()
    assert_expected_revision()
    with get_job_session() as session:
        runner = JobRunner(session)

        handler = JOB_HANDLERS.get(job_name)

        def _body() -> dict:
            logger.info("job_execute", job_name=job_name, schedule=SCHEDULES.get(job_name))
            if handler is None:
                return {"job_name": job_name, "status": "unknown_job"}
            return handler(session)

        job = runner.run(job_name, _body, idempotency_key=idempotency_key)
        return {
            "id": job.id,
            "job_name": job.job_name,
            "status": job.status,
            "attempt": job.attempt,
            "idempotency_key": job.idempotency_key,
            "error": job.error,
            "metadata_json": dict(job.metadata_json or {}),
        }


def enqueue_due_slots(
    now_utc: datetime | None = None,
    *,
    grace: timedelta = DEFAULT_GRACE,
) -> list[dict]:
    now_utc = _as_utc(now_utc or datetime.now(UTC))
    with get_session() as session:
        last_runs = last_successful_runs(session)
        pending = due_slots(now_utc, last_runs, grace=grace)
        outbox = JobOutboxService(session)
        enqueued: list[dict] = []
        for slot in pending:
            row = outbox.enqueue(
                slot.job_name,
                idempotency_key=slot.slot_key,
                scheduled_at=slot.scheduled_at,
            )
            enqueued.append(
                {
                    "job_name": slot.job_name,
                    "idempotency_key": slot.slot_key,
                    "outbox_id": row.id,
                    "status": row.status,
                }
            )
    return enqueued


def process_outbox(*, max_jobs: int = 5) -> list[dict]:
    results: list[dict] = []
    with get_job_session() as session:
        outbox = JobOutboxService(session)
        outbox.recover_stale_running(stale_after=timedelta(hours=2))
        for _ in range(max_jobs):
            row = outbox.claim_next()
            if row is None:
                break
            outbox.mark_running(row)
            try:
                payload = run_job(row.job_name, idempotency_key=row.idempotency_key)
                if payload.get("status") == "succeeded":
                    outbox.mark_succeeded(row, metadata=payload)
                else:
                    error = str(payload.get("error") or payload.get("status"))
                    outbox.mark_failed(row, error)
                    send_ops_alert(
                        f"Job failed: {row.job_name}",
                        f"outbox_id={row.id}\nidempotency_key={row.idempotency_key}\nerror={error}",
                    )
                results.append(payload)
            except Exception as exc:  # noqa: BLE001
                outbox.mark_failed(row, f"{type(exc).__name__}: {exc}")
                send_ops_alert(
                    f"Job exception: {row.job_name}",
                    f"outbox_id={row.id}\nerror={type(exc).__name__}: {exc}",
                )
                results.append(
                    {
                        "outbox_id": row.id,
                        "job_name": row.job_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    return results


def run_due(now_utc: datetime | None = None, *, grace: timedelta = DEFAULT_GRACE) -> list[dict]:
    """Enqueue due slots, then execute inline unless external executor is configured."""
    enqueued = enqueue_due_slots(now_utc, grace=grace)
    if not enqueued:
        return []
    settings = get_settings()
    if settings.long_jobs_external:
        return enqueued
    return process_outbox(max_jobs=len(enqueued) + 2)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fantasy app worker")
    parser.add_argument("command", choices=["run-once", "run-due", "list"])
    parser.add_argument("job_name", nargs="?")
    args = parser.parse_args()
    if args.command == "list":
        for name, desc in SCHEDULES.items():
            print(f"{name}: {desc}")
        return
    if args.command == "run-due":
        for result in run_due():
            print(result)
        return
    if not args.job_name:
        raise SystemExit("job_name required")
    result = run_job(args.job_name)
    print(result)


if __name__ == "__main__":
    main()
