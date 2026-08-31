"""Validate verified claims collected during the 2024 WR sentiment spike."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_sentiment_spike_population import SPIKE_CUTOFF_2024

REQUIRED_FIELDS = (
    "source_url",
    "excerpt",
    "publication_timestamp",
    "captured_content_hash",
    "reviewer",
)
RETROSPECTIVE_PATTERNS = (
    r"\bfinal stats\b",
    r"\bfinished the season\b",
    r"\bseason total\b",
    r"\bended the year\b",
    r"\b2024 season stats\b",
)


def _parse_ts(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def load_population(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["player_id"]) for row in payload.get("players") or []}


def validate_claim(
    claim: dict,
    *,
    allowed_players: set[str],
    cutoff: datetime,
) -> list[str]:
    errors: list[str] = []
    if claim.get("evidence_tier") != "verified":
        errors.append("evidence_tier must be verified")
    if not claim.get("training_eligible"):
        errors.append("training_eligible must be true for verified spike claims")
    for field in REQUIRED_FIELDS:
        if not claim.get(field):
            errors.append(f"missing required field: {field}")
    player_id = str(claim.get("player_id") or "")
    if player_id not in allowed_players:
        errors.append(f"player_id {player_id!r} outside frozen population")
    excerpt = str(claim.get("excerpt") or "")
    if not excerpt.strip():
        errors.append("excerpt must be non-empty")
    ts_raw = claim.get("publication_timestamp")
    if ts_raw:
        try:
            published = _parse_ts(ts_raw)
        except ValueError:
            errors.append("publication_timestamp is not unambiguous ISO-8601")
        else:
            if published > cutoff:
                errors.append("publication_timestamp is after spike cutoff")
    topic = f"{claim.get('parsed_label', '')} {excerpt}".lower()
    if not any(
        term in topic
        for term in (
            "role",
            "usage",
            "development",
            "coach",
            "front office",
            "sentiment",
            "opportunity",
            "depth chart",
        )
    ):
        errors.append("claim is not player-specific to role/usage/development/sentiment")
    for pattern in RETROSPECTIVE_PATTERNS:
        if re.search(pattern, excerpt, flags=re.IGNORECASE):
            errors.append(f"excerpt contains retrospective outcome language ({pattern})")
            break
    return errors


def validate_claims_file(
    claims_path: Path,
    population_path: Path,
    *,
    cutoff: str = SPIKE_CUTOFF_2024,
) -> dict:
    allowed = load_population(population_path)
    cutoff_dt = _parse_ts(cutoff)
    invalid: list[dict] = []
    valid = 0
    with claims_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            errors = validate_claim(claim, allowed_players=allowed, cutoff=cutoff_dt)
            if errors:
                invalid.append({"line": line_no, "claim_id": claim.get("claim_id"), "errors": errors})
            else:
                valid += 1
    return {"valid": valid, "invalid": invalid, "invalid_count": len(invalid)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claims", type=Path)
    parser.add_argument(
        "--population",
        type=Path,
        default=ROOT / "data" / "sentiment" / "spike" / "population_2024_wr.json",
    )
    args = parser.parse_args()
    report = validate_claims_file(args.claims, args.population)
    print(json.dumps(report, indent=2))
    return 1 if report["invalid_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
