"""Refresh diagnostic sentiment fields without rerunning projection models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.projection.fantasy_points import compute_fantasy_points
from src.projection.publish import sha256_file, validate_projection_contract
from src.sentiment.snapshot import REPO_ROOT, SENTIMENT_OUTPUT_COLS, attach_sentiment


def refresh_outputs(
    *,
    season: int,
    as_of: str,
    projections_path: str | Path | None = None,
    fantasy_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    projections_path = Path(projections_path) if projections_path else (
        REPO_ROOT / "output" / f"projections_{season}.csv"
    )
    fantasy_path = Path(fantasy_path) if fantasy_path else (
        REPO_ROOT / "output" / f"fantasy_points_{season}.csv"
    )
    manifest_path = Path(manifest_path) if manifest_path else (
        projections_path.parent / f"projection_run_{season}.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing projection run manifest: {manifest_path}. "
            "Run `python -m src.projection.publish` before a sentiment-only refresh."
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    long = pd.read_csv(projections_path)
    validate_projection_contract(
        long, season, manifest=manifest, projection_path=projections_path
    )
    baseline_cols = [c for c in long.columns if c not in SENTIMENT_OUTPUT_COLS]
    long = long.drop(columns=SENTIMENT_OUTPUT_COLS, errors="ignore")
    long = attach_sentiment(long, season=season, as_of=as_of)
    long = long[[*baseline_cols, *SENTIMENT_OUTPUT_COLS]]
    fantasy = compute_fantasy_points(long)
    projections_temp = projections_path.with_suffix(projections_path.suffix + ".tmp")
    fantasy_temp = fantasy_path.with_suffix(fantasy_path.suffix + ".tmp")
    manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    long.to_csv(projections_temp, index=False)
    fantasy.to_csv(fantasy_temp, index=False)
    manifest["sentiment_refreshed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["sentiment_as_of"] = as_of
    manifest["files"]["projections"]["sha256"] = sha256_file(projections_temp)
    manifest["files"]["fantasy_points"]["sha256"] = sha256_file(fantasy_temp)
    with open(manifest_temp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    os.replace(projections_temp, projections_path)
    os.replace(fantasy_temp, fantasy_path)
    os.replace(manifest_temp, manifest_path)
    return projections_path, fantasy_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--projections-path", default=None)
    parser.add_argument("--fantasy-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    args = parser.parse_args()
    projections, fantasy = refresh_outputs(
        season=args.season,
        as_of=args.as_of,
        projections_path=args.projections_path,
        fantasy_path=args.fantasy_path,
        manifest_path=args.manifest_path,
    )
    print(f"Refreshed {projections}")
    print(f"Refreshed {fantasy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
