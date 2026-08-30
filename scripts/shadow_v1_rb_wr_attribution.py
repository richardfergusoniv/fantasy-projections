"""Non-mutating rolling-origin attribution for v1 RB/WR repair investigation.

Produces read-only diagnostics under output/shadow_v1_rb_wr/. Does not change
production ensemble weights or publish paths.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.projection.contracts import OUTPUT_DIR, REPO_ROOT

SHADOW_DIR = Path(OUTPUT_DIR) / "shadow_v1_rb_wr"
SEASONS = (2023, 2024, 2025)


def _load_board(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["player_id"] = frame["player_id"].astype(str)
    return frame


def _population_mask(frame: pd.DataFrame, *, top120: bool) -> pd.Series:
    if not top120:
        return pd.Series(True, index=frame.index)
    if "adp" in frame.columns:
        adp = pd.to_numeric(frame["adp"], errors="coerce")
        return adp.notna() & (adp <= 120)
    return pd.Series(False, index=frame.index)


def run_attribution(*, season: int, top120: bool = False) -> dict:
    """Placeholder rolling-origin attribution scaffold.

    Compares available mean columns when present. Extend with games-vs-rate
  decomposition and feature-level attribution as frozen evaluation fixtures land.
    """
    v1_path = Path(OUTPUT_DIR) / f"fantasy_points_{season}.csv"
    v2_path = Path(OUTPUT_DIR) / "model_v2" / f"fantasy_points_{season}.csv"
    selected_path = Path(OUTPUT_DIR) / f"fantasy_points_{season}.csv"
    v1 = _load_board(v1_path)
    mask = _population_mask(v1, top120=top120)
    subset = v1.loc[mask].copy()
    result = {
        "season": season,
        "population": "top120" if top120 else "all_eligible",
        "player_count": int(mask.sum()),
        "signals": {},
    }
    for label, col in (("v1", "fantasy_pts_season"),):
        if col in subset.columns:
            result["signals"][label] = {
                "mean": float(pd.to_numeric(subset[col], errors="coerce").mean()),
                "count": int(subset[col].notna().sum()),
            }
    if v2_path.exists():
        v2 = _load_board(v2_path)
        merged = subset.merge(v2[["player_id", "fantasy_pts_season"]], on="player_id", how="left", suffixes=("", "_v2"))
        result["signals"]["v2"] = {
            "mean": float(pd.to_numeric(merged["fantasy_pts_season_v2"], errors="coerce").mean()),
            "count": int(merged["fantasy_pts_season_v2"].notna().sum()),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, action="append", dest="seasons")
    args = parser.parse_args()
    seasons = tuple(args.seasons or SEASONS)
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "shadow_v1_rb_wr_attribution_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "production_weights_frozen": True,
        "seasons": [],
    }
    for season in seasons:
        for top120 in (False, True):
            payload["seasons"].append(run_attribution(season=season, top120=top120))
    out = SHADOW_DIR / "attribution_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(out), "seasons": len(payload["seasons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
