#!/usr/bin/env python3
"""Verify all draft_assistant surfaces resolve the active release namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.active_release import read_active_pointer

SURFACES = [
    ("draft", "/index.html", "data", ["players", "team_stats"]),
    ("teams", "/teams/index.html", "../data", ["players", "team_stats"]),
    ("totals", "/totals/index.html", "../data", ["players", "team_stats"]),
    ("compare", "/compare/index.html", "../data", ["comparison", "team_stats"]),
    ("sleepers", "/sleepers/index.html", "../data", ["players", "comparison", "deep_band_accuracy"]),
]


def fetch(url: str) -> tuple[int, bytes]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_map(season: int) -> dict[str, str]:
    return {
        "players": f"players_{season}.json",
        "team_stats": f"team_stats_{season}.json",
        "comparison": f"comparison_{season}.json",
        "deep_band_accuracy": "deep_band_accuracy.json",
    }


def verify(base_url: str, season: int, *, expect_namespace: str | None, expect_legacy: bool) -> dict:
    checked_in = None if expect_legacy else read_active_pointer(season)
    results: dict = {
        "base_url": base_url,
        "checked_in_pointer": checked_in,
        "expect_legacy": expect_legacy,
        "expect_namespace": expect_namespace,
        "frozen_identity": None,
        "checks": [],
    }
    if expect_legacy:
        if checked_in is not None:
            results["checks"].append({"check": "no_active_pointer", "passed": False})
        else:
            results["checks"].append({"check": "no_active_pointer", "passed": True})
    elif expect_namespace:
        passed = checked_in is not None and checked_in.get("namespace") == expect_namespace
        results["checks"].append(
            {
                "check": "active_pointer_namespace",
                "passed": passed,
                "namespace": (checked_in or {}).get("namespace"),
                "expected": expect_namespace,
            }
        )
    elif checked_in is None:
        results["checks"].append(
            {
                "check": "active_pointer_present",
                "passed": False,
                "error": "checked-in active pointer missing; pass --namespace or --legacy",
            }
        )

    pointer_url = f"{base_url.rstrip('/')}/data/active_release_{season}.json"
    status, body = fetch(pointer_url)
    if expect_legacy:
        results["checks"].append({"check": "pointer_missing_404", "passed": status == 404, "status": status})
    else:
        results["checks"].append({"check": "pointer_present", "passed": status == 200, "status": status})
        if status == 200:
            served = json.loads(body.decode("utf-8"))
            served_ns = served.get("namespace")
            served_release = served.get("release_id")
            served_hash = str(served.get("manifest_sha256") or "").lower()

            if checked_in is not None:
                drift_ok = (
                    served_ns == checked_in.get("namespace")
                    and str(served_release) == str(checked_in.get("release_id"))
                    and served_hash == str(checked_in.get("manifest_sha256") or "").lower()
                )
                results["checks"].append(
                    {
                        "check": "served_pointer_matches_checked_in",
                        "passed": drift_ok,
                        "served": {
                            "namespace": served_ns,
                            "release_id": served_release,
                            "manifest_sha256": served_hash,
                        },
                        "checked_in": {
                            "namespace": checked_in.get("namespace"),
                            "release_id": checked_in.get("release_id"),
                            "manifest_sha256": checked_in.get("manifest_sha256"),
                        },
                    }
                )

            if expect_namespace is not None:
                results["checks"].append(
                    {
                        "check": "pointer_namespace",
                        "passed": served_ns == expect_namespace,
                        "namespace": served_ns,
                        "expected": expect_namespace,
                    }
                )
                frozen_ns = expect_namespace
            else:
                frozen_ns = served_ns

            frozen = {
                "namespace": frozen_ns,
                "release_id": served_release,
                "manifest_sha256": served_hash,
            }
            results["frozen_identity"] = frozen

            manifest_path = served.get("manifest_path", "")
            manifest_url = f"{base_url.rstrip('/')}/{manifest_path.lstrip('/')}"
            m_status, m_body = fetch(manifest_url)
            digest = sha256(m_body) if m_status == 200 else None
            results["checks"].append(
                {
                    "check": "manifest_hash",
                    "passed": digest == served_hash,
                    "status": m_status,
                    "digest": digest,
                }
            )
            if m_status == 200:
                try:
                    manifest = json.loads(m_body.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    results["checks"].append(
                        {
                            "check": "manifest_json",
                            "passed": False,
                            "error": str(exc),
                        }
                    )
                    manifest = None
                if isinstance(manifest, dict):
                    ns = frozen["namespace"]
                    for entry in manifest.get("artifacts") or []:
                        if not entry.get("browser_consumed"):
                            continue
                        rel = entry["path"]
                        url = f"{base_url.rstrip('/')}/data/releases/{ns}/{rel}"
                        a_status, a_body = fetch(url)
                        file_digest = sha256(a_body) if a_status == 200 else None
                        results["checks"].append(
                            {
                                "check": f"artifact:{entry['role']}",
                                "passed": a_status == 200 and file_digest == entry["sha256"],
                                "status": a_status,
                                "url": url,
                            }
                        )
        else:
            frozen = None

    frozen_ns = None
    if not expect_legacy:
        frozen = results.get("frozen_identity") or {}
        frozen_ns = frozen.get("namespace")

    for name, page, data_root, roles in SURFACES:
        status, _ = fetch(f"{base_url.rstrip('/')}{page}")
        results["checks"].append({"check": f"page:{name}", "passed": status == 200, "status": status})
        for role in roles:
            if expect_legacy:
                legacy_map = {
                    "players": f"{data_root}/players_{season}.json",
                    "team_stats": f"{data_root}/team_stats_{season}.json",
                    "comparison": f"{data_root}/comparison_{season}.json",
                    "deep_band_accuracy": f"{data_root}/deep_band_accuracy.json",
                }
                url = f"{base_url.rstrip('/')}/{legacy_map[role].lstrip('/')}"
            else:
                if not frozen_ns:
                    results["checks"].append(
                        {
                            "check": f"asset:{name}:{role}",
                            "passed": False,
                            "error": "frozen namespace unavailable",
                        }
                    )
                    continue
                file_map = _file_map(season)
                url = f"{base_url.rstrip('/')}/data/releases/{frozen_ns}/{file_map[role]}"
            status, body = fetch(url)
            results["checks"].append(
                {
                    "check": f"asset:{name}:{role}",
                    "passed": status == 200 and len(body) > 0,
                    "status": status,
                    "url": url,
                    "frozen_namespace": frozen_ns,
                }
            )
    results["verdict"] = "pass" if all(c.get("passed") for c in results["checks"]) else "fail"
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--namespace",
        help="Optional explicit namespace assertion against the checked-in pointer",
    )
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    report = verify(
        args.base_url,
        args.season,
        expect_namespace=args.namespace,
        expect_legacy=args.legacy,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
