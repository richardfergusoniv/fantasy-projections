"""Static checks on the container stack, for when Docker is not available.

This is *static* validation and says so: it parses the YAML and the Dockerfiles
and asserts properties that are worth asserting. It cannot prove that an image
builds, that a container starts, or that a healthcheck passes — only a real
`docker compose up` does that, and the readiness audit records the difference.

The earlier version only checked that certain strings appeared somewhere in the
files, so it passed while the runtime image ran as root, installed the dev
dependency group, shipped the test suite, and the stack had no scheduler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "docker-compose.yml",
    "Dockerfile",
    "Dockerfile.web",
    "docker/nginx.conf",
    ".env.example",
)

REQUIRED_SERVICES = ("db", "migrate", "seed", "api", "scheduler", "worker", "web")

#: Services that must declare a healthcheck, because something depends on them
#: being *ready* rather than merely started.
SERVICES_NEEDING_HEALTHCHECK = ("db", "api")

#: Never in the production runtime image.
FORBIDDEN_IN_RUNTIME_IMAGE = (
    ("--all-extras", "installs optional extras into the production image"),
    ("--dev", "installs the dev dependency group into the production image"),
    ("COPY tests", "ships the test suite in the production image"),
)


def _load_compose() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8")) or {}


def check_files() -> list[str]:
    return [f"missing {rel}" for rel in REQUIRED_FILES if not (ROOT / rel).exists()]


def check_services(compose: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    services = compose.get("services") or {}
    for name in REQUIRED_SERVICES:
        if name not in services:
            failures.append(f"docker-compose missing service: {name}")
    for name in SERVICES_NEEDING_HEALTHCHECK:
        service = services.get(name) or {}
        if name in services and "healthcheck" not in service:
            failures.append(f"service {name} has no healthcheck")

    # Anything that depends on readiness must say so, not just on start order.
    for name in ("migrate", "seed", "api", "scheduler", "worker"):
        service = services.get(name) or {}
        depends = service.get("depends_on") or {}
        db_dep = depends.get("db") if isinstance(depends, dict) else None
        if isinstance(db_dep, dict) and db_dep.get("condition") != "service_healthy":
            failures.append(f"service {name} does not wait for a healthy database")

    # Exactly one scheduler, or two replicas contend for the same slots.
    scheduler = services.get("scheduler") or {}
    replicas = ((scheduler.get("deploy") or {}).get("replicas"))
    if replicas not in (None, 1):
        failures.append(f"scheduler must run exactly one replica (got {replicas})")

    # The database must not be published to every interface.
    for port in (services.get("db") or {}).get("ports") or []:
        if isinstance(port, str) and not port.startswith("127.0.0.1:"):
            failures.append(f"db port {port!r} is not bound to loopback")

    if not compose.get("name"):
        failures.append("docker-compose has no project name, so it can collide with other stacks")

    volumes = compose.get("volumes") or {}
    if "fantasy_app_pg" not in volumes:
        failures.append("no named volume for PostgreSQL data")
    if "fantasy_app_artifacts" not in volumes:
        failures.append("no named volume for the artifact store")
    return failures


def check_runtime_image() -> list[str]:
    failures: list[str] = []
    raw = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    # Comments explain *why* a flag is absent, so they must not be mistaken for
    # the flag being present.
    dockerfile = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )

    for fragment in ("python:3.14", "alembic.ini", "draft_assistant/data", "src.app.cli"):
        if fragment not in dockerfile:
            failures.append(f"Dockerfile missing: {fragment}")
    if "uv sync --frozen" not in dockerfile:
        failures.append("Dockerfile must install from the lockfile (uv sync --frozen)")
    if "USER appuser" not in dockerfile:
        failures.append("Dockerfile must drop to a non-root user")
    for fragment, why in FORBIDDEN_IN_RUNTIME_IMAGE:
        if fragment in dockerfile:
            failures.append(f"Dockerfile {fragment!r}: {why}")
    return failures


def check_web_image() -> list[str]:
    failures: list[str] = []
    web_docker = (ROOT / "Dockerfile.web").read_text(encoding="utf-8")
    if "npm run build" not in web_docker:
        failures.append("Dockerfile.web missing production build step")
    if "npm ci" not in web_docker:
        failures.append("Dockerfile.web must install from the lockfile (npm ci)")
    if "HEALTHCHECK" not in web_docker:
        failures.append("Dockerfile.web has no healthcheck")

    nginx = (ROOT / "docker/nginx.conf").read_text(encoding="utf-8")
    for fragment in ("proxy_pass http://api:8000", "try_files"):
        if fragment not in nginx:
            failures.append(f"nginx.conf missing: {fragment}")
    return failures


def main() -> int:
    failures = check_files()
    if not failures:
        compose = _load_compose()
        failures += check_services(compose)
    failures += check_runtime_image()
    failures += check_web_image()

    if failures:
        print("Docker Compose config validation failed:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("Docker Compose config validation passed (static only — not a runtime check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
