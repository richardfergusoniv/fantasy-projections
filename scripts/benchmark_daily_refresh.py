#!/usr/bin/env python3
"""Benchmark daily-refresh against the Vercel deployment gate."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--p95-limit", type=float, default=180.0)
    parser.add_argument("--max-limit", type=float, default=220.0)
    args = parser.parse_args()

    from src.app.jobs.handlers import run_daily_refresh
    from src.app.persistence.database import get_job_session, reset_engine

    durations: list[float] = []
    failures: list[str] = []
    for run in range(args.runs):
        if args.cold_start:
            reset_engine()
        started = time.perf_counter()
        try:
            with get_job_session() as session:
                run_daily_refresh(session, automatic=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"run_{run + 1}:{type(exc).__name__}:{exc}")
        else:
            durations.append(time.perf_counter() - started)

    report = {
        "runs_requested": args.runs,
        "runs_completed": len(durations),
        "failures": failures,
        "p50_seconds": round(statistics.median(durations), 3) if durations else None,
        "p95_seconds": round(percentile(durations, 95), 3) if durations else None,
        "max_seconds": round(max(durations), 3) if durations else None,
        "gate": {
            "p95_below": args.p95_limit,
            "max_below": args.max_limit,
            "zero_failures": not failures,
        },
    }
    passed = (
        report["p95_seconds"] is not None
        and report["p95_seconds"] < args.p95_limit
        and report["max_seconds"] is not None
        and report["max_seconds"] < args.max_limit
        and not failures
        and len(durations) >= args.runs
    )
    report["passed"] = passed
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
