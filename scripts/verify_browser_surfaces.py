#!/usr/bin/env python3
"""Verify all draft_assistant surfaces resolve the active release namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
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
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(base_url: str, season: int, *, expect_namespace: str | None, expect_legacy: bool) -> dict:
    pointer = read_active_pointer(season)
    results: dict = {
        "base_url": base_url,
        "pointer": pointer,
        "expect_legacy": expect_legacy,
        "expect_namespace": expect_namespace,
        "checks": [],
    }
    if expect_legacy:
        if pointer is not None:
            results["checks"].append({"check": "no_active_pointer", "passed": False})
        else:
            results["checks"].append({"check": "no_active_pointer", "passed": True})
    elif expect_namespace:
        passed = pointer is not None and pointer.get("namespace") == expect_namespace
        results["checks"].append(
            {
                "check": "active_pointer_namespace",
                "passed": passed,
                "namespace": (pointer or {}).get("namespace"),
                "expected": expect_namespace,
            }
        )
    pointer_url = f"{base_url.rstrip('/')}/data/active_release_{season}.json"
    status, body = fetch(pointer_url)
    if expect_legacy:
        results["checks"].append({"check": "pointer_missing_404", "passed": status == 404, "status": status})
    else:
        results["checks"].append({"check": "pointer_present", "passed": status == 200, "status": status})
        if status == 200:
            doc = json.loads(body.decode("utf-8"))
            results["checks"].append(
                {
                    "check": "pointer_namespace",
                    "passed": doc.get("namespace") == expect_namespace,
                    "namespace": doc.get("namespace"),
                }
            )
            manifest_path = doc.get("manifest_path", "")
            manifest_url = f"{base_url.rstrip('/')}/{manifest_path.lstrip('/')}"
            m_status, m_body = fetch(manifest_url)
            digest = sha256(m_body) if m_status == 200 else None
            results["checks"].append(
                {
                    "check": "manifest_hash",
                    "passed": digest == doc.get("manifest_sha256"),
                    "status": m_status,
                }
            )
            if m_status == 200:
                manifest = json.loads(m_body.decode("utf-8"))
                ns = doc.get("namespace")
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
                url = f"{base_url.rstrip('/')}/data/releases/{expect_namespace}/{role.replace('players', f'players_{season}.json').replace('team_stats', f'team_stats_{season}.json').replace('comparison', f'comparison_{season}.json').replace('deep_band_accuracy', 'deep_band_accuracy.json')}"
                # fix mapping
                file_map = {
                    "players": f"players_{season}.json",
                    "team_stats": f"team_stats_{season}.json",
                    "comparison": f"comparison_{season}.json",
                    "deep_band_accuracy": "deep_band_accuracy.json",
                }
                url = f"{base_url.rstrip('/')}/data/releases/{expect_namespace}/{file_map[role]}"
            status, body = fetch(url)
            results["checks"].append(
                {
                    "check": f"asset:{name}:{role}",
                    "passed": status == 200 and len(body) > 0,
                    "status": status,
                    "url": url,
                }
            )
    results["verdict"] = "pass" if all(c.get("passed") for c in results["checks"]) else "fail"
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--namespace")
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
