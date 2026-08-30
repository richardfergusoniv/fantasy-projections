"""Canonical overlay coverage counts from the final serialized player payload."""
from __future__ import annotations

from typing import Any, Mapping

# Published overlay fields that must be present on every simulated player.
OVERLAY_COVERAGE_FIELDS = (
    "fantasy_pts_p10",
    "fantasy_pts_p25",
    "fantasy_pts_p50",
    "fantasy_pts_p75",
    "fantasy_pts_p90",
    "volatility_flag",
    "p_finish_top6",
    "p_finish_top12",
    "p_finish_top24",
    "p_finish_top36",
    "p_finish_top48",
    "sim_vorp_p10",
    "sim_vorp_p50",
    "sim_vorp_p90",
    "p_vorp_positive",
    "expected_pos_rank",
    "median_pos_rank",
)


def _is_non_null(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        import math

        return math.isfinite(value)
    if isinstance(value, str) and not value.strip():
        return False
    return True


def compute_overlay_coverage(players_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Count non-null overlay fields directly from the final player payload."""
    players = players_doc.get("players")
    if players is None:
        raise ValueError("players payload missing players[]")
    total = len(players)
    fields: dict[str, dict[str, int]] = {}
    for field in OVERLAY_COVERAGE_FIELDS:
        non_null = sum(1 for row in players if _is_non_null((row or {}).get(field)))
        fields[field] = {"non_null": non_null}
    return {"total_players": total, "fields": fields}


def _coverage_mismatches(
    label: str,
    coverage: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> list[str]:
    if coverage is None:
        return [f"{label} overlay_coverage missing"]
    mismatches: list[str] = []
    if "total_players" not in coverage:
        mismatches.append(f"{label} missing total_players")
    elif int(coverage["total_players"]) != int(expected["total_players"]):
        mismatches.append(
            f"{label}.total_players={coverage['total_players']} computed={expected['total_players']}"
        )
    fields = coverage.get("fields")
    if not isinstance(fields, Mapping):
        return mismatches + [f"{label} missing fields block"]
    expected_fields = expected.get("fields") or {}
    for field in OVERLAY_COVERAGE_FIELDS:
        block = fields.get(field)
        if not isinstance(block, Mapping) or "non_null" not in block:
            mismatches.append(f"{label} missing {field}.non_null")
            continue
        expected_count = int((expected_fields.get(field) or {}).get("non_null", -1))
        actual_count = int(block["non_null"])
        if actual_count != expected_count:
            mismatches.append(
                f"{label}.{field}.non_null={actual_count} computed={expected_count}"
            )
    return mismatches


def overlay_coverage_alignment(
    *,
    players_doc: Mapping[str, Any],
    manifest_coverage: Mapping[str, Any] | None,
    report_coverage: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    """Require exact equality among computed, manifest, and report coverage."""
    computed = compute_overlay_coverage(players_doc)
    mismatches: list[str] = []
    mismatches.extend(_coverage_mismatches("manifest", manifest_coverage, computed))
    mismatches.extend(_coverage_mismatches("report", report_coverage, computed))
    return not mismatches, {"computed": computed, "mismatches": mismatches}
