#!/usr/bin/env python3
"""Unified go-live gate: all eight live-readiness objectives in one report."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_URL = "postgresql+psycopg://fantasy:fantasy@localhost:5432/fantasy_app"
DEFAULT_REPORT = ROOT / "output" / "live_pg" / "go_live_gate.json"


def _run(cmd: list[str], *, env: dict | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_gate(
    *,
    database_url: str = DEFAULT_PG_URL,
    sleeper_report: Path = ROOT / "output" / "live_pg" / "sleeper_sync_report.json",
    readiness_report: Path = ROOT / "output" / "live_pg" / "live_readiness_audit_promoted.json",
    infra_report: Path = ROOT / "output" / "live_pg" / "production_infrastructure_audit.json",
    backup_dir: Path = ROOT / "output" / "backups",
    report_path: Path = DEFAULT_REPORT,
    api_base_url: str | None = None,
) -> dict:
    gate: dict = {"started_at": datetime.now(UTC).isoformat(), "objectives": {}}

    # 1 — Sealed bundle (no mutable fallback)
    loader_code, _ = _run(
        [sys.executable, "-m", "pytest", "tests/app/test_projection_reanchor.py::test_loader_uses_sealed_component_projections_not_output_fallback", "-q"]
    )
    readiness = _load_json(readiness_report) or {}
    bundle = readiness.get("sealed_bundle") or {}
    caveats = bundle.get("caveats") or []
    gate["objectives"]["1_sealed_bundle"] = {
        "passed": loader_code == 0 and not caveats,
        "namespace": bundle.get("namespace"),
        "caveats": caveats,
    }

    # 2 — PostgreSQL + Sleeper sync (6 leagues)
    sleeper = _load_json(sleeper_report) or {}
    selection = sleeper.get("league_selection") or {}
    gate["objectives"]["2_postgresql_sleeper_sync"] = {
        "passed": sleeper.get("status") == "passed" and selection.get("imported_count") == 6,
        "imported_count": selection.get("imported_count"),
        "database_url": database_url,
        "go_no_go": sleeper.get("go_no_go"),
    }

    # 3 — Live scoring contracts
    scoring = readiness.get("scoring_summary") or {}
    gate["objectives"]["3_scoring_contracts"] = {
        "passed": scoring.get("league_count") == 6 and scoring.get("all_publishable"),
        "league_count": scoring.get("league_count"),
        "fidelity_counts": scoring.get("fidelity_counts"),
    }

    # 4 — Daily refresh candidate
    verdict = readiness.get("verdict") or {}
    gate["objectives"]["4_daily_refresh_candidate"] = {
        "passed": verdict.get("first_live_overlay_promoted") or verdict.get("ready_for_overlay_promotion"),
        "verdict": verdict,
    }

    # 5 — Identity, injury, deltas, conservation
    identity = readiness.get("identity") or {}
    injury = readiness.get("injury_research") or {}
    conservation = readiness.get("conservation") or {}
    gate["objectives"]["5_inspections"] = {
        "passed": (
            identity.get("unresolved_count", 1) == 0
            and not injury.get("synthetic")
            and injury.get("mode") != "fixture"
            and conservation.get("exit_code") == 0
        ),
        "unresolved_count": identity.get("unresolved_count"),
        "injury_mode": injury.get("mode"),
        "conservation_exit_code": conservation.get("exit_code"),
        "projection_deltas": readiness.get("projection_deltas", {}).get("adjustment_count"),
    }

    # 6 — Live overlay promoted
    overlay = readiness.get("overlay_candidate") or {}
    promotion = readiness.get("overlay_promotion") or {}
    gate["objectives"]["6_live_overlay_promoted"] = {
        "passed": (
            overlay.get("promoted")
            and overlay.get("from_live_data")
            and promotion.get("fixture_citations_in_artifact", 1) == 0
        ),
        "overlay_hash": overlay.get("overlay_hash"),
        "fixture_citations": promotion.get("fixture_citations_in_artifact"),
    }

    # 7 — Decision flows per league
    recs = sleeper.get("recommendations") or []
    completeness = sleeper.get("completeness") or {}
    gate["objectives"]["7_decision_flows"] = {
        "passed": sleeper.get("go_no_go") == "go",
        "recommendation_count": len(recs),
        "rostered_player_ids": completeness.get("rostered_player_ids"),
    }

    # 8 — Phone-access infrastructure
    infra = _load_json(infra_report)
    if infra is None or not infra.get("verdict", {}).get("phone_access_ready"):
        env_file = ROOT / ".env"
        env_args = ["--env-file", str(env_file)] if env_file.is_file() else []
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "production_infrastructure_check.py"),
                "--database-url",
                database_url,
                "--report",
                str(infra_report),
            ]
            + env_args
            + (["--api-base-url", api_base_url] if api_base_url else []),
        )
        infra = _load_json(infra_report) or {}

    backups = sorted(backup_dir.glob("fantasy_app*.sql"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_backup = backups[0] if backups else None
    runtime = infra.get("runtime") or {}
    gate["objectives"]["8_phone_access_infrastructure"] = {
        "passed": infra.get("verdict", {}).get("phone_access_ready", False),
        "blockers": infra.get("verdict", {}).get("blockers", []),
        "postgresql_ok": (runtime.get("postgresql") or {}).get("status") == "ok",
        "backup_script": runtime.get("backup_script"),
        "latest_backup": str(latest_backup) if latest_backup else None,
        "latest_backup_bytes": latest_backup.stat().st_size if latest_backup else None,
        "nginx_tls_example": runtime.get("nginx_tls_example"),
        "production_env_example": runtime.get("production_env_example"),
        "operator_actions": infra.get("phone_access_requirements"),
    }

    passed = [k for k, v in gate["objectives"].items() if v.get("passed")]
    failed = [k for k, v in gate["objectives"].items() if not v.get("passed")]
    gate["summary"] = {
        "passed_count": len(passed),
        "total": 8,
        "passed_objectives": passed,
        "failed_objectives": failed,
        "go_live_ready": len(failed) == 0,
    }
    gate["finished_at"] = datetime.now(UTC).isoformat()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def main() -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_PG_URL))
    parser.add_argument("--api-base-url", default=os.environ.get("APP_PUBLIC_URL"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    gate = run_gate(
        database_url=args.database_url,
        report_path=args.report,
        api_base_url=args.api_base_url,
    )
    print(json.dumps(gate["summary"], indent=2))
    return 0 if gate["summary"]["go_live_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
