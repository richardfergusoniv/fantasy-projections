"""Cold-start guard: session + league list must not import the decision engine.

The Vercel Python function previously imported LineupService (numpy/draws) while
booting, so the PWA's first ``GET /me`` waited ~10s on a cold isolate. Keep the
ASGI import graph light; lineup/waivers may load the engine on first use.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_factory_import_does_not_load_numpy_or_lineup_engine():
    env = os.environ.copy()
    env.setdefault("APP_ENV", "test")
    env.setdefault("APP_ENABLE_DEV_AUTH", "true")
    env.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")
    script = """
import os
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")

from src.app.factory import create_app

create_app()

import sys
heavy = [
    name
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "src.app.decisions.services",
        "src.app.decisions.lineup",
        "src.app.decisions.draws",
        "src.app.assistant.gateway",
        "src.app.jobs.handlers",
    )
    if name in sys.modules
]
assert not heavy, f"cold import loaded heavy modules: {heavy}"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
