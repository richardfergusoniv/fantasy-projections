"""Audit whether sentiment history is sufficient to attempt model activation.

This command deliberately does not activate a position. It verifies the
minimum data prerequisites before an end-to-end rolling fantasy ablation is
meaningful; activation still requires the performance thresholds documented
in the sentiment manifest/report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.sentiment.snapshot import DEFAULT_SENTIMENT_DIR


MIN_SEASONS = 3
MIN_COVERED_PLAYER_SEASONS = 200
MIN_POSITION_COVERAGE = 0.40
VERIFIED_EVIDENCE_TIER = "verified"


def _infer_season(path: Path) -> int:
    match = path.stem.split("_")
    season = next((int(part) for part in match if part.isdigit() and len(part) == 4), None)
    if season is None:
        raise ValueError(f"Could not infer season from {path.name}")
    return season


def _load_canonical_snapshots(paths: list[str | Path]) -> dict[int, pd.DataFrame]:
    """Keep one canonical preseason snapshot per season; reject duplicates."""
    by_season: dict[int, pd.DataFrame] = {}
    for path_value in paths:
        path = Path(path_value)
        season = _infer_season(path)
        if season in by_season:
            raise ValueError(
                f"Duplicate snapshot for season {season}: {path} and prior path"
            )
        frame = pd.read_csv(path)
        required = {"player_id", "position", "sentiment_as_of"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing sentiment columns: {sorted(missing)}")
        frame = frame.copy()
        frame["snapshot_season"] = season
        by_season[season] = frame
    return by_season


def _load_verified_claims(paths: list[str | Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "position",
                "player_id",
                "training_eligible",
                "evidence_tier",
            ]
        )
    return pd.DataFrame(rows)


def _covered_player_keys(claims: pd.DataFrame) -> set[tuple[int, str, str]]:
    """Unique player-seasons with at least one training-eligible verified claim."""
    if claims.empty:
        return set()
    verified = claims[
        claims["training_eligible"].astype(bool)
        & claims["evidence_tier"].astype(str).eq(VERIFIED_EVIDENCE_TIER)
    ]
    if verified.empty:
        return set()
    return {
        (int(season), str(position), str(player_id))
        for season, position, player_id in verified[
            ["season", "position", "player_id"]
        ].itertuples(index=False, name=None)
    }


def audit_snapshots(
    paths: list[str | Path],
    *,
    claim_paths: list[str | Path] | None = None,
) -> dict:
    """Audit readiness using unique player-seasons and verified-claim coverage."""
    snapshots = _load_canonical_snapshots(paths)
    combined = (
        pd.concat(snapshots.values(), ignore_index=True) if snapshots else pd.DataFrame()
    )
    claims = _load_verified_claims(claim_paths or [])
    covered_keys = _covered_player_keys(claims)

    by_position: dict[str, dict] = {}
    for position in ("QB", "RB", "WR", "TE"):
        subset = combined[combined["position"].eq(position)] if not combined.empty else combined
        position_seasons = sorted(
            subset.get("snapshot_season", pd.Series(dtype=int)).dropna().unique().tolist()
        )
        eligible_keys = {
            (int(season), str(position), str(player_id))
            for season, player_id in subset[["snapshot_season", "player_id"]].itertuples(
                index=False, name=None
            )
        } if not subset.empty else set()
        covered = eligible_keys & covered_keys
        eligible_count = len(eligible_keys)
        covered_count = len(covered)
        coverage = (covered_count / eligible_count) if eligible_count else 0.0
        prerequisites = (
            len(position_seasons) >= MIN_SEASONS
            and covered_count >= MIN_COVERED_PLAYER_SEASONS
            and coverage >= MIN_POSITION_COVERAGE
        )
        by_position[position] = {
            "rows": int(len(subset)),
            "seasons": position_seasons,
            "eligible_player_seasons": eligible_count,
            "covered_player_seasons": covered_count,
            "coverage": coverage,
            "prerequisites_met": prerequisites,
            "model_active": False,
        }
    return {
        "snapshot_count": len(paths),
        "canonical_seasons": sorted(snapshots.keys()),
        "minimums": {
            "seasons": MIN_SEASONS,
            "covered_player_seasons": MIN_COVERED_PLAYER_SEASONS,
            "position_coverage": MIN_POSITION_COVERAGE,
        },
        "by_position": by_position,
        "ready_for_ablation": any(v["prerequisites_met"] for v in by_position.values()),
        "note": "Passing prerequisites permits an ablation; it does not activate sentiment.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Sentiment snapshot CSVs")
    parser.add_argument("--claims", nargs="*", default=None, help="Verified claim JSONL files")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    paths = args.paths or sorted(DEFAULT_SENTIMENT_DIR.glob("sentiment_*.csv"))
    report = audit_snapshots(paths, claim_paths=args.claims)
    rendered = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
