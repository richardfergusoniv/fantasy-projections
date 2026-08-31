"""Freeze the 2024 WR deep-band population for the sentiment feasibility spike."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.contracts import OUTPUT_DIR

# Day before 2024 NFL Week 1 kickoff (Thursday 2024-09-05, Ravens @ Chiefs).
# Source: https://www.nfl.com/schedules/2024/REG1/
SPIKE_CUTOFF_2024 = "2024-09-04T00:00:00Z"

BAND_LOW = 61
BAND_HIGH = 300
EXPECTED_WR_COUNT = 105
SPIKE_SEED = 20240830


def freeze_population(season: int, position: str) -> dict:
    path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing evaluation frame: {path}")
    frame = pd.read_csv(path)
    frame["season"] = season
    scored = frame[frame["model_forecast_points"].notna()].copy()
    scored["proj_points"] = pd.to_numeric(scored["model_forecast_points"], errors="coerce")
    scored["proj_rank"] = scored.groupby("season")["proj_points"].rank(
        ascending=False, method="first"
    )
    band = scored[
        scored["proj_rank"].between(BAND_LOW, BAND_HIGH)
        & scored["preseason_position"].eq(position)
    ].copy()
    if len(band) != EXPECTED_WR_COUNT and position == "WR" and season == 2024:
        raise ValueError(
            f"Expected {EXPECTED_WR_COUNT} {position} player-seasons, found {len(band)}"
        )
    records = []
    for row in band.itertuples(index=False):
        records.append(
            {
                "player_id": str(row.player_id),
                "display_name": row.display_name,
                "preseason_team": row.preseason_team,
                "model_forecast_points": float(row.proj_points),
                "proj_rank": int(row.proj_rank),
                "band": "deep_core"
                if BAND_LOW <= row.proj_rank <= 120
                else "deep_primary"
                if row.proj_rank <= 200
                else "deep_speculative",
            }
        )
    records.sort(key=lambda item: item["player_id"])
    shuffled = records[:]
    rng = random.Random(SPIKE_SEED)
    rng.shuffle(shuffled)
    collection_order = [item["player_id"] for item in shuffled]
    content = json.dumps(records, sort_keys=True)
    return {
        "season": season,
        "position": position,
        "cutoff": SPIKE_CUTOFF_2024,
        "cutoff_note": (
            "Upper bound on evidence availability for the Week 1 roster population. "
            "Real drafts occur weeks earlier; coverage here does not transfer to "
            "draft-time overlays without re-measurement at an earlier cutoff."
        ),
        "population_rule": (
            f"output/fantasy_evaluation_{season}.csv; rank model_forecast_points "
            f"descending within season; keep proj_rank {BAND_LOW}-{BAND_HIGH}; "
            f"preseason_position == {position!r}; forecast presence only"
        ),
        "denominator": len(records),
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "collection_order_seed": SPIKE_SEED,
        "collection_order": collection_order,
        "players": records,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--position", default="WR")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "sentiment" / "spike" / "population_2024_wr.json"),
    )
    args = parser.parse_args()
    payload = freeze_population(args.season, args.position)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"denominator": payload["denominator"], "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
