#!/usr/bin/env bash
set -euo pipefail

export LONG_JOBS_EXTERNAL=false
exec uv run python -m src.app.jobs.scheduler run-due
