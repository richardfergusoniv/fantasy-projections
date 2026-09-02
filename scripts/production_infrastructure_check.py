#!/usr/bin/env python3
"""Audit production infrastructure for phone-access deployment.

Checks configuration safety, PostgreSQL reachability, backup tooling, and
optional live API health endpoints. Writes a machine-readable report and exits
non-zero when blockers remain.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "output" / "live_shadow" / "production_infrastructure_audit.json"


def _parse_database_url(url: str) -> dict[str, str | int]:
    """Best-effort parse of sqlalchemy postgres URLs for pg_isready."""
    lowered = url.lower()
    if "postgresql" not in lowered:
        return {}
    # postgresql+psycopg://user:pass@host:port/dbname
    without_scheme = url.split("://", 1)[-1]
    creds_host, _, dbname = without_scheme.partition("/")
    user = "fantasy"
    password = ""
    hostport = creds_host
    if "@" in creds_host:
        userinfo, hostport = creds_host.rsplit("@", 1)
        if ":" in userinfo:
            user, password = userinfo.split(":", 1)
        else:
            user = userinfo
    host = hostport
    port = 5432
    if ":" in hostport:
        host, port_str = hostport.rsplit(":", 1)
        port = int(port_str)
    return {
        "user": user,
        "password": password,
        "host": host or "localhost",
        "port": port,
        "dbname": dbname.split("?")[0] if dbname else "fantasy_app",
    }


def _pg_isready(database_url: str) -> dict:
    info = _parse_database_url(database_url)
    if not info:
        return {"status": "skipped", "reason": "not_postgresql"}
    pg_isready = shutil.which("pg_isready")
    if not pg_isready:
        win_pg = Path(r"C:\Program Files\PostgreSQL\16\bin\pg_isready.exe")
        pg_isready = str(win_pg) if win_pg.is_file() else None
    if not pg_isready:
        return {"status": "unknown", "reason": "pg_isready_not_found"}
    proc = subprocess.run(
        [
            pg_isready,
            "-h",
            str(info["host"]),
            "-p",
            str(info["port"]),
            "-U",
            str(info["user"]),
            "-d",
            str(info["dbname"]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "unreachable",
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip() or None,
        "host": info["host"],
        "port": info["port"],
        "database": info["dbname"],
    }


def _http_probe(url: str, *, method: str = "GET", timeout: float = 3.0) -> dict:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return {"status": "ok", "http_status": resp.status, "body_preview": body[:200]}
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "http_status": exc.code}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}


def run_audit(
    *,
    api_base_url: str | None = None,
    database_url: str | None = None,
    env_file: Path | None = None,
    report_path: Path = DEFAULT_REPORT,
) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.app.config import Settings
    from src.app.env_file import settings_from_env_file

    if env_file and env_file.is_file():
        prod = settings_from_env_file(env_file)
        database_url = database_url or prod.database_url
    else:
        database_url = database_url or os.environ.get("DATABASE_URL", "")
        prod = Settings(app_env="production")
        if database_url:
            prod = prod.model_copy(update={"database_url": database_url})
    problems = prod.production_config_problems()

    audit: dict = {
        "started_at": datetime.now(UTC).isoformat(),
        "database_url": database_url or None,
        "api_base_url": api_base_url,
        "env_file": str(env_file) if env_file else None,
    }

    audit["configuration"] = {
        "production_config_problems": problems,
        "production_ready": not problems,
        "email_provider": prod.email_provider,
        "app_public_url_https": str(prod.app_public_url or "").startswith("https://"),
        "artifact_backend": prod.artifact_backend,
        "trusted_hosts": prod.trusted_host_list,
    }

    backups_dir = ROOT / "output" / "backups"
    latest_backup = None
    if backups_dir.is_dir():
        candidates = sorted(backups_dir.glob("fantasy_app*.sql"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            latest = candidates[0]
            latest_backup = {
                "path": str(latest),
                "bytes": latest.stat().st_size,
                "age_hours": round((datetime.now(UTC).timestamp() - latest.stat().st_mtime) / 3600, 2),
            }

    audit["runtime"] = {
        "docker_available": shutil.which("docker") is not None,
        "postgresql": _pg_isready(database_url) if database_url else {"status": "skipped"},
        "backup_script": (ROOT / "scripts" / "pg_backup.ps1").is_file(),
        "monitor_script": (ROOT / "scripts" / "monitor_health.ps1").is_file(),
        "latest_backup": latest_backup,
        "compose_config_valid": _compose_config_valid(),
        "nginx_tls_example": (ROOT / "docker" / "nginx.tls.conf.example").is_file(),
        "production_env_example": (ROOT / ".env.production.example").is_file(),
    }

    if api_base_url:
        base = api_base_url.rstrip("/")
        audit["health"] = {
            "live": _http_probe(f"{base}/health/live"),
            "ready": _http_probe(f"{base}/health/ready"),
        }
    else:
        audit["health"] = {"status": "skipped", "reason": "no_api_base_url"}

    config_blockers: list[str] = []
    if problems:
        config_blockers.extend(f"config:{p}" for p in problems)

    runtime_blockers: list[str] = []
    pg = audit["runtime"]["postgresql"]
    if database_url and "postgresql" in database_url.lower() and pg.get("status") not in {"ok", "skipped"}:
        runtime_blockers.append(f"postgresql:{pg.get('status')}")
    if api_base_url:
        for name, probe in (audit["health"] or {}).items():
            if isinstance(probe, dict) and probe.get("status") not in {"ok", None}:
                if name in {"live", "ready"}:
                    runtime_blockers.append(f"health_{name}:{probe.get('status')}")

    cloud_blockers = list(config_blockers)
    if not audit["runtime"]["production_env_example"]:
        cloud_blockers.append("missing_production_env_example")

    audit["phone_access_requirements"] = {
        "https_public_url": "Set APP_PUBLIC_URL=https://your-domain",
        "email_magic_link": "Set EMAIL_PROVIDER=resend or smtp with credentials",
        "trusted_hosts": "Set TRUSTED_HOSTS to your API hostname",
        "cors_origins": "Set APP_CORS_ORIGINS=https://your-domain",
        "tls_termination": "Terminate TLS at reverse proxy; see docker/nginx.tls.conf.example",
        "backups": "Schedule scripts/pg_backup.ps1 daily; rehearse restore before go-live",
        "monitoring": "Poll GET /health/ready every 60s; alert on non-200",
        "supabase_roles": "Create fantasy_app_runtime and fantasy_app_migrator via supabase/roles.sql",
        "release_pointer": "Set release_pointer.manifest_storage_uri for the sealed baseline",
        "cron_secret": "Store CRON_SECRET in Supabase Vault and install supabase/cron/run_due.sql",
    }

    audit["verdict"] = {
        "code_ready": audit["runtime"]["backup_script"] and audit["runtime"]["compose_config_valid"],
        "cloud_configuration_ready": not cloud_blockers,
        "runtime_verified": not runtime_blockers,
        "phone_access_ready": not config_blockers and not runtime_blockers,
        "config_blockers": config_blockers,
        "cloud_blockers": cloud_blockers,
        "runtime_blockers": runtime_blockers,
        "blockers": config_blockers + runtime_blockers,
    }
    audit["finished_at"] = datetime.now(UTC).isoformat()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _compose_config_valid() -> bool:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_compose_config.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--api-base-url", default=os.environ.get("APP_PUBLIC_URL"))
    parser.add_argument("--env-file", type=Path, default=None, help="Production .env to validate")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    audit = run_audit(
        api_base_url=args.api_base_url,
        database_url=args.database_url,
        env_file=args.env_file,
        report_path=args.report,
    )
    print(json.dumps(audit["verdict"], indent=2))
    return 0 if audit["verdict"]["phone_access_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
