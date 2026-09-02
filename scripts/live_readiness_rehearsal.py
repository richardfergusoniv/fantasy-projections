#!/usr/bin/env python3
"""Live-readiness rehearsal: daily refresh, scoring audit, optional overlay promotion.

Uses the isolated shadow database populated by ``sleeper-shadow-sync``. Candidate
mode (default) builds but does not promote overlays. ``--promote-overlay`` promotes
the first live availability overlay with injury research disabled so fixture
citations never attach.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SHADOW_DB = "sqlite+pysqlite:///output/live_shadow/shadow_app.db"
DEFAULT_REPORT = ROOT / "output" / "live_shadow" / "live_readiness_audit.json"


def _configure(
    *,
    database_url: str,
    artifact_root: str,
    sleeper_user_id: str | None = None,
    promote_overlay: bool = False,
) -> None:
    os.environ["APP_ENV"] = "development"
    os.environ["DATABASE_URL"] = database_url
    os.environ["ARTIFACT_LOCAL_ROOT"] = artifact_root
    os.environ["SLEEPER_USE_FIXTURES"] = "false"
    os.environ["WEEKLY_RND_ENABLED"] = "false"
    os.environ["STATUS_OVERLAY_AUTO_PUBLISH"] = "true"
    os.environ.setdefault("LIVE_SLEEPER_SHADOW", "1")
    os.environ["INJURY_RESEARCH_MODE"] = "sleeper"
    if sleeper_user_id:
        os.environ["SLEEPER_USER_ID"] = sleeper_user_id
    from src.app.config import get_settings
    from src.app.persistence.database import reset_engine

    get_settings.cache_clear()
    reset_engine()


def run_rehearsal(
    *,
    database_url: str = DEFAULT_SHADOW_DB,
    artifact_root: str = "output/live_shadow/artifacts",
    report_path: Path = DEFAULT_REPORT,
    owner_config: Path | None = None,
    promote_overlay: bool = False,
) -> dict:
    owner_config = owner_config or (ROOT / "config" / "sleeper_owner.json")
    sleeper_user_id = _resolve_sleeper_user_id(owner_config)
    _configure(
        database_url=database_url,
        artifact_root=artifact_root,
        sleeper_user_id=sleeper_user_id,
        promote_overlay=promote_overlay,
    )
    from src.app.jobs.handlers import run_daily_refresh
    from src.app.persistence.database import get_session, init_db
    from src.app.projections.league_rescore import rescore_configured_leagues
    from src.app.projections.loader import ReleaseBundleLoader, invalidate_bundle_loader_cache
    from src.projection.release_bundle import sha256_file

    init_db()
    started = datetime.now(UTC).isoformat()
    audit: dict = {"started_at": started, "database_url": database_url}

    invalidate_bundle_loader_cache(2026)
    bundle = ReleaseBundleLoader(2026).load_bundle()
    if bundle is None:
        audit["error"] = "no_active_sealed_bundle"
        return audit

    audit["sealed_bundle"] = {
        "namespace": bundle.namespace,
        "release_id": bundle.release_id,
        "manifest_sha256": bundle.manifest_sha256,
        "caveats": list(bundle.caveats),
        "component_projections": str(bundle.component_projections_path),
        "component_hash": sha256_file(bundle.component_projections_path)
        if bundle.component_projections_path
        else None,
    }
    assert "component_projections_from_output_fallback" not in bundle.caveats

    with get_session() as session:
        refresh = run_daily_refresh(session, automatic=promote_overlay)
        session.commit()
        rescored = rescore_configured_leagues(
            session,
            components_path=bundle.component_projections_path,
            season=refresh.get("season", 2026),
        )

    audit["infrastructure"] = _infrastructure_audit()

    scoring_audit = []
    for result in rescored:
        scoring_audit.append(
            {
                **result.to_dict(),
                "special_rules": {
                    "has_ppfd_approximation": any("ppfd" in r for r in result.approximate_rules),
                    "approximate_rules": list(result.approximate_rules),
                },
            }
        )

    audit["daily_refresh"] = refresh
    audit["league_rescore"] = scoring_audit
    audit["scoring_summary"] = {
        "league_count": len(scoring_audit),
        "all_publishable": all(
            row.get("scoring_fidelity") != "unsupported_rule" for row in scoring_audit
        ),
        "distinct_contracts": len({row["contract_hash"] for row in scoring_audit}),
        "fidelity_counts": _count_fidelity(scoring_audit),
    }

    if bundle.component_projections_path:
        audit["conservation"] = _conservation_summary(bundle.component_projections_path)

    overlay = refresh.get("status_overlay") or {}
    audit["overlay_candidate"] = {
        "status": overlay.get("status"),
        "overlay_hash": overlay.get("overlay_hash"),
        "adjustment_count": overlay.get("adjustment_count"),
        "artifact_path": overlay.get("artifact_path"),
        "promoted": overlay.get("status") == "promoted",
        "from_live_data": refresh.get("sleeper_source") == "live",
    }
    audit["projection_deltas"] = _overlay_projection_deltas(overlay.get("artifact_path"))
    audit["overlay_promotion"] = _overlay_promotion_audit(promote_overlay, overlay)
    audit["identity"] = {
        "unresolved_player_ids": refresh.get("unresolved_player_ids", []),
        "unresolved_count": len(refresh.get("unresolved_player_ids") or []),
    }
    audit["injury_research"] = refresh.get("injury_research")
    audit["finished_at"] = datetime.now(UTC).isoformat()
    audit["verdict"] = _verdict(audit, promote_overlay=promote_overlay)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _resolve_sleeper_user_id(config_path: Path) -> str:
    from src.app.league.sleeper.client import SleeperClient
    from src.app.league.sleeper.owner_config import load_owner_config

    config = load_owner_config(config_path)
    user = SleeperClient(use_fixtures=False).get_user(config.username)
    user_id = str(user.get("user_id") or "")
    if not user_id:
        raise RuntimeError(f"Sleeper user lookup failed for {config.username!r}")
    return user_id


def _overlay_promotion_audit(promote_overlay: bool, overlay: dict) -> dict:
    from src.app.projections.status_overlay import read_active_overlay

    pointer = read_active_overlay(2026)
    artifact_path = overlay.get("artifact_path")
    fixture_citations = 0
    if artifact_path and Path(artifact_path).is_file():
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        for adj in payload.get("adjustments") or []:
            for cite in adj.get("citations") or []:
                if str(cite).startswith("fixture://") or "SYNTHETIC" in str(cite).upper():
                    fixture_citations += 1
    return {
        "requested": promote_overlay,
        "pointer_active": pointer is not None,
        "pointer_hash": pointer.get("overlay_hash") if pointer else None,
        "matches_candidate": pointer is not None and pointer.get("overlay_hash") == overlay.get("overlay_hash"),
        "fixture_citations_in_artifact": fixture_citations,
    }


def _infrastructure_audit() -> dict:
    import shutil

    from src.app.config import Settings

    prod = Settings(app_env="production")
    problems = prod.production_config_problems()
    return {
        "docker_available": shutil.which("docker") is not None,
        "postgresql_url_configured": "postgresql" in (os.environ.get("DATABASE_URL") or ""),
        "production_config_problems": problems,
        "production_ready": not problems,
        "email_provider_dev_default": prod.email_provider == "development",
        "app_public_url_https": str(prod.app_public_url or "").startswith("https://"),
    }


def _overlay_projection_deltas(artifact_path: str | None) -> dict:
    if not artifact_path:
        return {"status": "missing_artifact"}
    path = Path(artifact_path)
    if not path.is_file():
        return {"status": "artifact_not_found", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    adjustments = payload.get("adjustments") or []
    deltas = []
    for adj in adjustments:
        before = float(adj.get("before_points") or 0.0)
        after = float(adj.get("after_points") or 0.0)
        deltas.append(
            {
                "player_id": adj.get("player_id"),
                "position": adj.get("position"),
                "reason_code": adj.get("reason_code"),
                "before_points": before,
                "after_points": after,
                "delta": round(after - before, 4),
                "citation_count": len(adj.get("citations") or []),
            }
        )
    deltas.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return {
        "status": "ok",
        "adjustment_count": len(deltas),
        "max_abs_delta": round(max((abs(d["delta"]) for d in deltas), default=0.0), 4),
        "zeroed_players": sum(1 for d in deltas if d["after_points"] == 0.0),
        "top_adjustments": deltas[:10],
    }


def _count_fidelity(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        fid = str(row.get("scoring_fidelity") or "unknown")
        counts[fid] = counts.get(fid, 0) + 1
    return counts


def _conservation_summary(path: Path) -> dict:
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "conservation_check.py"), "--projections", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout.strip().splitlines()[-12:] if proc.stdout else [],
        "stderr": proc.stderr.strip() or None,
    }


def _verdict(audit: dict, *, promote_overlay: bool = False) -> dict:
    blockers: list[str] = []
    bundle = audit.get("sealed_bundle") or {}
    if bundle.get("caveats"):
        blockers.append(f"sealed_bundle_caveats:{bundle['caveats']}")
    if audit.get("identity", {}).get("unresolved_count", 0) > 0:
        unresolved = audit.get("identity", {}).get("unresolved_player_ids") or []
        real = [pid for pid in unresolved if pid not in {"0"}]
        if real:
            blockers.append("unresolved_player_ids")
    injury = audit.get("injury_research") or {}
    if injury.get("mode") == "fixture" or injury.get("synthetic"):
        blockers.append("injury_research_fixture_mode")
    overlay = audit.get("overlay_candidate") or {}
    promotion = audit.get("overlay_promotion") or {}
    if not overlay.get("from_live_data"):
        blockers.append("not_live_sleeper_source")
    scoring = audit.get("scoring_summary") or {}
    if scoring.get("league_count", 0) < 6:
        blockers.append(f"league_rescore_incomplete:{scoring.get('league_count')}")
    if promotion.get("fixture_citations_in_artifact", 0) > 0:
        blockers.append("fixture_citations_in_overlay")

    if promote_overlay:
        if overlay.get("status") != "promoted":
            blockers.append(f"overlay_not_promoted:{overlay.get('status')}")
        if not promotion.get("matches_candidate"):
            blockers.append("overlay_pointer_mismatch")
        return {
            "first_live_overlay_promoted": not blockers,
            "blockers": blockers,
        }

    if overlay.get("promoted"):
        blockers.append("overlay_promoted_in_candidate_mode")
    return {
        "ready_for_overlay_promotion": not blockers and overlay.get("status") == "built_not_promoted",
        "blockers": blockers,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=DEFAULT_SHADOW_DB)
    parser.add_argument("--artifact-root", default="output/live_shadow/artifacts")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "sleeper_owner.json")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--promote-overlay",
        action="store_true",
        help="Promote the live availability overlay (injury research disabled; no fixture citations)",
    )
    args = parser.parse_args()
    audit = run_rehearsal(
        database_url=args.database_url,
        artifact_root=args.artifact_root,
        report_path=args.report,
        owner_config=args.config,
        promote_overlay=args.promote_overlay,
    )
    print(json.dumps(audit.get("verdict", audit), indent=2))
    return 0 if not audit.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
