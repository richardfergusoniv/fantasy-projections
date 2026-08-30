"""Simulated VORP from recentered fantasy-point draws and fixed replacement."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.draft_assistant.positional_ranks import rank_positional_draws
from src.draft_assistant.replacement_contract import replacement_points_map
from src.projection.inference.simulate import load_partitioned_draws

POINTS_COL = "fantasy_pts_season"
SUMMARY_COLS = (
    "sim_vorp_p10",
    "sim_vorp_p50",
    "sim_vorp_p90",
    "p_vorp_positive",
    "expected_pos_rank",
    "median_pos_rank",
)


def _quantile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q, method="linear"))


def aggregate_player_metrics(
    vorp_values: list[float],
    rank_values: list[float],
) -> dict[str, float]:
    v = np.asarray(vorp_values, dtype=float)
    r = np.asarray(rank_values, dtype=float)
    return {
        "sim_vorp_p10": _quantile(v, 0.10),
        "sim_vorp_p50": _quantile(v, 0.50),
        "sim_vorp_p90": _quantile(v, 0.90),
        "p_vorp_positive": float(np.mean(v > 0.0)) if len(v) else float("nan"),
        "expected_pos_rank": float(np.mean(r)) if len(r) else float("nan"),
        "median_pos_rank": _quantile(r, 0.50),
    }


def process_draw_partition(
    partition: pd.DataFrame,
    *,
    replacement_points: dict[str, float],
    points_col: str = POINTS_COL,
) -> pd.DataFrame:
    """Compute draw-level simulated VORP and positional ranks for one partition."""
    if partition.empty:
        return partition.iloc[0:0].copy()
    needed = {"player_id", "position", "draw", points_col}
    missing = needed - set(partition.columns)
    if missing:
        raise ValueError(f"partition missing columns: {sorted(missing)}")
    frame = partition[list(needed)].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame[points_col] = pd.to_numeric(frame[points_col], errors="coerce")
    frame["replacement_points"] = frame["position"].astype(str).map(replacement_points)
    frame["sim_vorp_draw"] = frame[points_col] - frame["replacement_points"]
    frame["pos_rank_draw"] = rank_positional_draws(frame, points_col=points_col)
    return frame


def stream_simulated_vorp_summary(
    partitions: Iterable[pd.DataFrame],
    *,
    replacement_contract: dict,
    points_col: str = POINTS_COL,
) -> pd.DataFrame:
    """Aggregate player-level simulated VORP metrics across draw partitions."""
    replacement_points = replacement_points_map(replacement_contract)
    vorp_acc: dict[str, list[float]] = defaultdict(list)
    rank_acc: dict[str, list[float]] = defaultdict(list)
    for partition in partitions:
        enriched = process_draw_partition(
            partition,
            replacement_points=replacement_points,
            points_col=points_col,
        )
        for row in enriched.itertuples(index=False):
            pid = str(row.player_id)
            vorp_acc[pid].append(float(row.sim_vorp_draw))
            rank_acc[pid].append(float(row.pos_rank_draw))
    rows = []
    for player_id in sorted(vorp_acc):
        metrics = aggregate_player_metrics(vorp_acc[player_id], rank_acc[player_id])
        rows.append({"player_id": player_id, **metrics})
    return pd.DataFrame(rows)


def load_manifest_partitions(
    manifest: dict,
    *,
    season: int,
) -> list[pd.DataFrame]:
    run_id = manifest.get("canonical_projection_run_id") or manifest.get(
        "source_projection_run_id"
    )
    if run_id:
        partitioned = load_partitioned_draws(season, str(run_id))
        if not partitioned.empty:
            partition_dir = manifest.get("partition_dir")
            if partition_dir:
                root = Path(partition_dir)
                parts = sorted(root.glob("part-*.parquet"))
                return [pd.read_parquet(path) for path in parts]
            draw_ids = sorted(partitioned["draw"].unique())
            chunk = int(manifest.get("partition_count") or 1)
            if chunk <= 1:
                return [partitioned]
            size = max(1, len(draw_ids) // chunk)
            frames = []
            for start in range(0, len(draw_ids), size):
                chunk_ids = set(draw_ids[start : start + size])
                frames.append(partitioned[partitioned["draw"].isin(chunk_ids)].copy())
            return frames
    recentered_path = manifest.get("recentered_draws_path")
    if recentered_path and Path(recentered_path).exists():
        return [pd.read_parquet(recentered_path)]
    return []


def manifest_hash(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("runtime_seconds", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
