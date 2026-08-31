"""Verify fixture-ready MVP artifacts and smoke paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "src/app/factory.py",
    "src/app/cli.py",
    "src/app/projections/weekly_v2_bridge.py",
    "src/app/releases/partitions.py",
    "src/app/releases/rollback.py",
    "src/app/releases/incremental.py",
    "web/src/App.tsx",
    "web/vite.config.ts",
    "docker-compose.yml",
    "migrations/versions/e53ebac3a6e5_initial_app_schema.py",
    "tests/fixtures/weekly_v2/season=2026/manifest.json",
    "docs/APP_IMPLEMENTATION_BLUEPRINT.md",
    "docs/APP_OPERATIONS_RUNBOOK.md",
    "scripts/vertical_smoke.py",
]


def check_paths() -> list[str]:
    missing = []
    for rel in REQUIRED_PATHS:
        if not (ROOT / rel).exists():
            missing.append(rel)
    return missing


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def main() -> int:
    failures: list[str] = []

    missing = check_paths()
    if missing:
        failures.append(f"missing paths: {', '.join(missing)}")
    else:
        print("OK   required paths")

    code, out = run([sys.executable, "scripts/validate_compose_config.py"])
    if code != 0:
        failures.append(f"compose config validation failed:\n{out}")
    else:
        print("OK   docker compose config")

    code, out = run([sys.executable, "-m", "pytest", "tests/app/test_job_rehearsal.py", "-q"])
    if code != 0:
        failures.append(f"job rehearsal tests failed:\n{out}")
    else:
        print("OK   job rehearsal tests")

    code, out = run([sys.executable, "-m", "pytest", "tests/app/test_incremental.py", "-q"])
    if code != 0:
        failures.append(f"incremental simulation tests failed:\n{out}")
    else:
        print("OK   incremental simulation tests")

    code, out = run([sys.executable, "-m", "pytest", "tests/app/test_weekly_v2_research.py", "-q"])
    if code != 0:
        failures.append(f"weekly v2 research tests failed:\n{out}")
    else:
        print("OK   weekly v2 bridge tests")

    code, out = run([sys.executable, "-m", "pytest", "tests/app/test_weekly_runs.py", "-q"])
    if code != 0:
        failures.append(f"weekly run tests failed:\n{out}")
    else:
        print("OK   weekly run tests")

    code, out = run([sys.executable, "scripts/vertical_smoke.py"])
    if code != 0:
        failures.append(f"vertical smoke failed:\n{out}")
    else:
        print("OK   vertical smoke")

    from src.app.projections.weekly_v2_bridge import (
        load_weekly_v2_manifest,
        weekly_v2_artifacts_available,
        weekly_v2_model_version,
    )

    if not weekly_v2_artifacts_available(2026, 1):
        failures.append("weekly v2 fixture manifest not detected for season 2026")
    else:
        print("OK   weekly v2 fixture manifest")
    manifest = load_weekly_v2_manifest(2026)
    if not manifest or manifest.get("model_version") != "weekly_v2_fixture":
        failures.append("weekly v2 fixture manifest content invalid")
    elif weekly_v2_model_version(2026) != "weekly_v2_fixture":
        failures.append("weekly v2 model version mismatch")
    else:
        print("OK   weekly v2 model version")

    if failures:
        print("\nMVP verification failed:")
        for item in failures:
            print(f" - {item}")
        return 1
    print("\nMVP verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
