#!/usr/bin/env python3
"""Preflight checks before phone-access bootstrap (step 8)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config" / "phone_access.secrets.example.json"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def _production_env_ready(base: Path) -> tuple[bool, list[str], list[str]]:
    """Return (ready, blockers, warnings) for Supabase/Vercel production access."""
    blockers: list[str] = []
    warnings: list[str] = []
    required_vars = (
        "APP_SECRET_KEY",
        "APP_ALLOWED_EMAIL",
        "APP_PUBLIC_URL",
        "APP_CORS_ORIGINS",
        "TRUSTED_HOSTS",
        "DATABASE_URL",
        "CRON_SECRET",
    )
    env_path = base / ".env"
    values: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    missing = [name for name in required_vars if not values.get(name)]
    if missing:
        blockers.extend(f"missing_env:{name}" for name in missing)
    public_url = values.get("APP_PUBLIC_URL", "")
    if public_url and not public_url.startswith("https://"):
        blockers.append("app_public_url_not_https")
    return not blockers, blockers, warnings


def run_preflight(*, root: Path | None = None) -> dict:
    base = root or ROOT
    secrets = base / "config" / "phone_access.secrets.json"
    example = base / "config" / "phone_access.secrets.example.json"
    checks: list[dict] = []

    prod_ready, prod_blockers, prod_warnings = _production_env_ready(base)
    checks.append(
        _check(
            "production_env",
            prod_ready,
            "ready" if prod_ready else ", ".join(prod_blockers),
        )
    )

    checks.append(_check("web_node_modules", (base / "web" / "node_modules").is_dir()))
    checks.append(_check("sleeper_owner_config", (base / "config" / "sleeper_owner.json").is_file()))
    checks.append(_check("secrets_example", example.is_file()))

    secrets_ok = False
    secrets_detail = "missing"
    if secrets.is_file():
        payload = json.loads(secrets.read_text(encoding="utf-8"))
        email = (payload.get("allowed_email") or "").strip()
        key = (payload.get("resend_api_key") or "").strip()
        placeholder = email.endswith("@example.com") or key.startswith("re_REPLACE")
        secrets_ok = bool(email and key and not placeholder)
        secrets_detail = "ready" if secrets_ok else "placeholder values — update email and resend_api_key"
    checks.append(_check("phone_access_secrets", secrets_ok, secrets_detail))

    legacy_checks = {
        "postgresql_service": shutil.which("pg_isready") is not None
        or Path(r"C:\Program Files\PostgreSQL\16\bin\pg_isready.exe").is_file(),
        "cloudflared": (base / "tools" / "cloudflared.exe").is_file()
        or shutil.which("cloudflared") is not None,
        "node_npm": Path(r"C:\Program Files\nodejs\npm.cmd").is_file()
        or shutil.which("npm") is not None,
    }
    for name, ok in legacy_checks.items():
        checks.append(_check(name, ok, "legacy_local_stack"))

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "live_go_live_gate.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    gate_summary = {}
    try:
        gate_path = base / "output" / "live_pg" / "go_live_gate.json"
        if gate_path.is_file():
            gate_summary = json.loads(gate_path.read_text(encoding="utf-8")).get("summary", {})
    except json.JSONDecodeError:
        pass
    checks.append(
        _check(
            "go_live_gate_objectives_1_7",
            gate_summary.get("passed_count", 0) >= 7,
            f"passed {gate_summary.get('passed_count', '?')}/8 (legacy local stack)",
        )
    )

    production_blockers = list(prod_blockers)
    if not secrets_ok:
        production_blockers.append("phone_access_secrets")
    legacy_blockers = [
        c["name"]
        for c in checks
        if c["name"] in legacy_checks and not c["ok"]
    ]
    stack_blockers = list(legacy_blockers)
    if not secrets_ok:
        stack_blockers.append("phone_access_secrets")
    return {
        "ready_for_production": not production_blockers,
        "ready_for_stack": not stack_blockers,
        "production_blockers": production_blockers,
        "legacy_stack_blockers": legacy_blockers,
        "blockers": stack_blockers,
        "warnings": prod_warnings,
        "checks": checks,
        "next_command": "powershell -File scripts/start_phone_access_stack.ps1 -NonInteractive"
        if secrets_ok and not legacy_blockers
        else "copy config\\phone_access.secrets.example.json config\\phone_access.secrets.json",
    }


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, indent=2))
    if report["blockers"] == ["phone_access_secrets"]:
        return 2
    return 0 if report["ready_for_stack"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
