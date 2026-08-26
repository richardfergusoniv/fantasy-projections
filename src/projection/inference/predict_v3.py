"""V3 prediction orchestration (parallel to v1 predict.project_season)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.projection.contracts import MODEL_V3_DIR, V3_MODELS_DIR
from src.projection.inference.simulate import write_simulation_outputs


def project_season_v3(projections: pd.DataFrame, season: int, *, n_draws: int = 1000) -> dict:
    """Run interim/full simulation on an existing long projection board."""
    manifest = write_simulation_outputs(projections, season, n_draws=n_draws)
    out_dir = Path(MODEL_V3_DIR)
    summary = pd.read_csv(manifest["summary_path"])
    summary.to_csv(out_dir / f"fantasy_points_{season}.csv", index=False)
    (out_dir / f"manifest_{season}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
