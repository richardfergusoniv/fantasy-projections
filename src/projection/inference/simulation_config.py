"""Load simulation configuration from config/simulation.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.projection.contracts import REPO_ROOT

CONFIG_PATH = Path(REPO_ROOT) / "config" / "simulation.json"


def load_simulation_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return {
            "random_seed": 2026,
            "profiles": {
                "dev": {"draws": 1000, "chunk_size": 250},
                "publish": {"draws": None, "chunk_size": 250},
                "release_candidate": {"draws": 10000, "chunk_size": 250},
            },
            "stability_tolerances": {
                "p50_abs": 0.25,
                "p10_p90_abs": 1.0,
                "sim_vorp_p50_abs": 0.5,
                "probability_abs": 0.015,
                "expected_pos_rank_abs": 0.25,
            },
            "stability_draw_candidates": [1000, 2000, 5000, 10000],
            "decision_thresholds": {
                "p_finish_top12": 0.5,
                "p_finish_top24": 0.5,
                "p_vorp_positive": 0.5,
            },
            "core_drafted_adp_max": 36,
        }
    return json.loads(config_path.read_text(encoding="utf-8"))


def profile_draws(config: dict, profile: str) -> int | None:
    profiles = config.get("profiles") or {}
    profile_cfg = profiles.get(profile) or profiles.get("dev") or {}
    return profile_cfg.get("draws")


def profile_chunk_size(config: dict, profile: str) -> int:
    profiles = config.get("profiles") or {}
    profile_cfg = profiles.get(profile) or profiles.get("dev") or {}
    return int(profile_cfg.get("chunk_size") or 250)


def deterministic_simulation_seed(
    *,
    season: int,
    board_hash: str,
    calibration_hash: str,
    configured_seed: int,
    transform_version: str | None = None,
    wr_calibration_hash: str | None = None,
) -> int:
    """Stable seed from season, board identity, and calibration artifacts.

    Random publish run IDs are intentionally excluded so reruns reproduce.
    """
    payload = "|".join(
        str(part)
        for part in (
            season,
            board_hash,
            calibration_hash,
            configured_seed,
            transform_version or "",
            wr_calibration_hash or "",
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def stability_simulation_seed(
    *,
    season: int,
    board_hash: str,
    calibration_hash: str,
    configured_seed: int,
    canonical_projection_run_id: str,
    transform_version: str,
    wr_calibration_hash: str,
) -> int:
    """Seed for draw-stability sweeps with frozen projection-run identity."""
    payload = "|".join(
        str(part)
        for part in (
            season,
            board_hash,
            calibration_hash,
            configured_seed,
            canonical_projection_run_id,
            transform_version,
            wr_calibration_hash,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def rng_for_draw(master_seed: int, draw_id: int):
    """Per-draw RNG stream invariant to chunk boundaries."""
    import numpy as np

    return np.random.default_rng((master_seed + int(draw_id) * 1_000_003) % (2**32 - 1))
