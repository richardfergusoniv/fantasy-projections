"""Produce the sealed 2024 WR sentiment feasibility report."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_sentiment_spike_population import EXPECTED_WR_COUNT, SPIKE_CUTOFF_2024
from scripts.validate_spike_claims import validate_claims_file

STOP_COVERAGE = 0.40
VERDICT_INFEASIBLE = "historical_collection_infeasible"
VERDICT_FEASIBLE = "feasibility_passed_stop_after_report"
MANIFEST_PATH = ROOT / "models" / "sentiment_manifest.json"


def _load_attempts(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["player_id"]): row for row in payload.get("attempts") or []}


def _load_verified_claims(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    claims: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return [c for c in claims if c.get("evidence_tier") == "verified" and c.get("training_eligible")]


def build_report(
    *,
    population_path: Path,
    claims_path: Path | None,
    attempts_path: Path | None,
    reviewer_hours: float = 0.0,
) -> dict:
    population = json.loads(population_path.read_text(encoding="utf-8"))
    denominator = int(population.get("denominator") or EXPECTED_WR_COUNT)
    if denominator != EXPECTED_WR_COUNT:
        raise ValueError(f"Expected denominator {EXPECTED_WR_COUNT}, found {denominator}")

    claims = _load_verified_claims(claims_path)
    validation = (
        validate_claims_file(claims_path, population_path)
        if claims_path and claims_path.exists()
        else {"valid": 0, "invalid": [], "invalid_count": 0}
    )
    if validation["invalid_count"]:
        raise ValueError(f"invalid verified claims: {validation['invalid']}")
    verified_players = {str(c["player_id"]) for c in claims}
    numerator = len(verified_players)
    coverage = numerator / denominator if denominator else 0.0

    attempts = _load_attempts(attempts_path)
    attempted_ids = set(attempts) | verified_players
    attempted_count = len(attempted_ids)
    unattempted = denominator - attempted_count

    by_band = Counter()
    player_band = {str(p["player_id"]): p.get("band") for p in population.get("players") or []}
    for player_id in verified_players:
        by_band[player_band.get(player_id, "unknown")] += 1

    miss_reasons = Counter(
        row.get("miss_reason", "not_attempted")
        for row in attempts.values()
        if not row.get("verified")
    )
    for _ in range(max(0, unattempted)):
        miss_reasons["not_attempted"] += 1

    source_classes = Counter(c.get("source_class", "unknown") for c in claims)
    unverifiable = sum(
        1 for row in attempts.values() if row.get("outcome") == "unverifiable_discovery"
    )

    hours_per_covered = reviewer_hours / numerator if numerator else None
    full_backfill_estimate_hours = None
    if hours_per_covered is not None and numerator:
        # Rough RB/WR/TE deep-band scale from measured pace.
        full_backfill_estimate_hours = round(hours_per_covered * 3 * 250, 1)

    if attempted_count < denominator:
        verdict = VERDICT_INFEASIBLE
        stop_rule = "incomplete_run_counts_as_uncovered"
    elif coverage < STOP_COVERAGE:
        verdict = VERDICT_INFEASIBLE
        stop_rule = "below_40_percent_verified_player_coverage"
    else:
        verdict = VERDICT_FEASIBLE
        stop_rule = "at_or_above_40_percent_stop_after_report"

    report = {
        "spike": "2024_WR",
        "cutoff": SPIKE_CUTOFF_2024,
        "frozen_denominator": denominator,
        "population_hash": population.get("content_hash"),
        "verified_numerator": numerator,
        "coverage_rate": round(coverage, 4),
        "coverage_by_band": dict(by_band),
        "players_attempted": attempted_count,
        "players_unattempted": unattempted,
        "verified_over_attempted": round(numerator / attempted_count, 4) if attempted_count else 0.0,
        "verified_over_attempted_note": (
            "Estimate only — gate uses verified/frozen_denominator (105), not this ratio."
        ),
        "reviewer_hours": reviewer_hours,
        "hours_per_covered_player": hours_per_covered,
        "full_backfill_labor_estimate_hours": full_backfill_estimate_hours,
        "source_class_distribution": dict(source_classes),
        "unverifiable_discoveries": unverifiable,
        "miss_reasons": dict(miss_reasons),
        "validation_invalid_count": validation["invalid_count"],
        "stop_rule": stop_rule,
        "verdict": verdict,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authorization": (
            "Passing does not authorize backfill, modeling, ranking changes, or overlays."
        ),
    }
    report["report_hash"] = json.dumps(report, sort_keys=True)
    import hashlib

    report["report_hash"] = hashlib.sha256(report["report_hash"].encode("utf-8")).hexdigest()
    return report


def write_manifest_verdict(report: dict) -> None:
    if report["verdict"] != VERDICT_INFEASIBLE:
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["historical_collection_infeasible"] = {
        "verdict": report["verdict"],
        "report_hash": report["report_hash"],
        "coverage_rate": report["coverage_rate"],
        "recorded_at": report["generated_at"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        type=Path,
        default=ROOT / "data" / "sentiment" / "spike" / "population_2024_wr.json",
    )
    parser.add_argument("--claims", type=Path, default=None)
    parser.add_argument("--attempts", type=Path, default=None)
    parser.add_argument("--reviewer-hours", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sentiment" / "spike" / "report_2024_wr.json")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    report = build_report(
        population_path=args.population,
        claims_path=args.claims,
        attempts_path=args.attempts,
        reviewer_hours=args.reviewer_hours,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.write_manifest:
        write_manifest_verdict(report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
