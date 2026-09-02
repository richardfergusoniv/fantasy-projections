"""Blueprint MVP completion audit — prints pass/fail per deliverable."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("TRUSTED_HOSTS", "testserver,localhost,127.0.0.1,*")
os.environ.setdefault("APP_ALLOWED_EMAIL", "owner@example.com")
os.environ.setdefault("EMAIL_PROVIDER", "development")
os.environ.setdefault("SLEEPER_USE_FIXTURES", "true")
os.environ.setdefault("INJURY_RESEARCH_MODE", "fixture")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _exists(*parts: str) -> bool:
    return (ROOT / Path(*parts)).exists()


def _module(path: str) -> bool:
    try:
        importlib.import_module(path)
        return True
    except Exception:
        return False


def audit_paths() -> list[Check]:
    required = [
        ("FastAPI app", "src/app/factory.py"),
        ("CLI", "src/app/cli.py"),
        ("Alembic migration", "migrations/versions/e53ebac3a6e5_initial_app_schema.py"),
        ("Scoring compiler", "src/app/scoring/compiler.py"),
        ("Decision engines", "src/app/decisions/services.py"),
        ("Weekly v2 bridge", "src/app/projections/weekly_v2_bridge.py"),
        ("Incremental simulation", "src/app/releases/incremental.py"),
        ("Rollback service", "src/app/releases/rollback.py"),
        ("Artifact store", "src/app/artifacts/store.py"),
        ("Job scheduler", "src/app/jobs/scheduler.py"),
        ("React PWA", "web/src/App.tsx"),
        ("Docker Compose", "docker-compose.yml"),
        ("CI workflow", ".github/workflows/ci.yml"),
        ("Operations runbook", "docs/APP_OPERATIONS_RUNBOOK.md"),
        ("Data contracts", "docs/APP_DATA_CONTRACTS.md"),
        ("Security doc", "docs/APP_SECURITY.md"),
        ("Weekly v2 port provenance", "docs/WEEKLY_V2_PORT_PROVENANCE.md"),
        ("Vertical smoke", "scripts/vertical_smoke.py"),
        ("MVP verifier", "scripts/verify_mvp.py"),
    ]
    return [Check(name, _exists(*Path(path).parts), path) for name, path in required]


def audit_modules() -> list[Check]:
    modules = [
        ("Auth service", "src.app.auth.service"),
        ("Sleeper sync", "src.app.league.sleeper.sync"),
        ("Availability lifecycle", "src.app.availability.service"),
        ("Assistant gateway", "src.app.assistant.gateway"),
        ("Projection weekly run", "src.app.projections.weekly_run"),
        ("Weekly v2 pipeline", "src.projection.weekly.pipeline.season_projector"),
    ]
    return [Check(name, _module(path), path) for name, path in modules]


def audit_api_routes() -> list[Check]:
    from src.app.factory import create_app

    app = create_app()
    paths = set(app.openapi().get("paths", {}).keys())
    expected = [
        "/api/v1/auth/magic-link",
        "/api/v1/me",
        "/api/v1/leagues",
        "/api/v1/leagues/{league_id}/lineup/{week}",
        "/api/v1/leagues/{league_id}/waivers/{week}",
        "/api/v1/leagues/{league_id}/trades/evaluate",
        "/api/v1/assistant/responses",
        "/api/v1/operations/status",
        "/api/v1/operations/projections/rollback",
        "/api/v1/sync",
        "/health/live",
        "/health/ready",
    ]
    return [Check(f"route {path}", path in paths) for path in expected]


def audit_pwa_screens() -> list[Check]:
    screens = [
        "Home",
        "Lineup",
        "Waivers",
        "TradeLab",
        "Dynasty",
        "Draft",
        "Assistant",
        "Operations",
    ]
    app_src = (ROOT / "web/src/App.tsx").read_text(encoding="utf-8")
    return [Check(f"PWA screen {name}", f"{name}Screen" in app_src or name.lower() in app_src.lower()) for name in screens]


def audit_fixture_leagues() -> list[Check]:
    manifest = ROOT / "src/app/fixtures/seed/leagues_manifest.json"
    if not manifest.exists():
        return [Check("seed leagues manifest", False)]
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    leagues = data.get("leagues") or data
    count = len(leagues) if isinstance(leagues, list) else 0
    return [Check("six seed leagues", count >= 6, f"found {count}")]


def audit_scoring_fixtures() -> list[Check]:
    fixtures = list((ROOT / "tests/fixtures/scoring").glob("*.json"))
    return [Check("scoring compiler fixtures", len(fixtures) >= 6, f"found {len(fixtures)}")]


def run_pytest_subset() -> Check:
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/app", "tests/scoring", "-q", "--tb=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    return Check("app + scoring tests", proc.returncode == 0, proc.stdout[-200:] if proc.stdout else proc.stderr[-200:])


def _run_script(rel: str) -> bool:
    proc = subprocess.run([sys.executable, rel], cwd=ROOT, capture_output=True, text=True)
    return proc.returncode == 0


def main() -> int:
    checks: list[Check] = []
    checks.extend(audit_paths())
    checks.extend(audit_modules())
    checks.extend(audit_fixture_leagues())
    checks.extend(audit_scoring_fixtures())
    checks.extend(audit_pwa_screens())
    try:
        checks.extend(audit_api_routes())
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("API route audit", False, str(exc)))
    checks.append(Check("Docker Compose config", _run_script("scripts/validate_compose_config.py")))
    checks.append(run_pytest_subset())

    failed = [c for c in checks if not c.passed]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        suffix = f" — {check.detail}" if check.detail and not check.passed else ""
        print(f"{status}  {check.name}{suffix}")

    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("\nRemaining gaps (fixture MVP may still be runnable):")
        for check in failed:
            print(f" - {check.name}: {check.detail or 'see above'}")
        return 1
    print("\nBlueprint fixture MVP audit: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
