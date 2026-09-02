#!/usr/bin/env python3
"""Preflight checks before phone-access bootstrap (step 8)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / "config" / "phone_access.secrets.json"
EXAMPLE = ROOT / "config" / "phone_access.secrets.example.json"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def run_preflight(*, root: Path | None = None) -> dict:
    base = root or ROOT
    secrets = base / "config" / "phone_access.secrets.json"
    example = base / "config" / "phone_access.secrets.example.json"
    checks: list[dict] = []

    checks.append(_check("postgresql_service", shutil.which("pg_isready") is not None or Path(r"C:\Program Files\PostgreSQL\16\bin\pg_isready.exe").is_file()))
    checks.append(_check("cloudflared", (base / "tools" / "cloudflared.exe").is_file() or shutil.which("cloudflared") is not None))
    checks.append(_check("node_npm", Path(r"C:\Program Files\nodejs\npm.cmd").is_file() or shutil.which("npm") is not None))
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

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "live_go_live_gate.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    gate_ok = proc.returncode == 0
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
            f"passed {gate_summary.get('passed_count', '?')}/8",
        )
    )

    blockers = [c["name"] for c in checks if not c["ok"]]
    return {
        "ready_for_stack": not blockers or blockers == ["phone_access_secrets"],
        "blockers": blockers,
        "checks": checks,
        "next_command": "powershell -File scripts/start_phone_access_stack.ps1 -NonInteractive"
        if secrets_ok
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
