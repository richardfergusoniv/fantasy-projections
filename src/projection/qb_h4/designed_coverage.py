"""Build portable pre-2023+ designed/scramble coverage for H4.

Uses existing weekly_qb_repair_cache PBP parquet (2022–2025). Seasons 2018–2021
have no scramble/designed flags in this repository — they remain uncovered
(NaN), never silently zeroed or classified as pocket.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.qb_rush_features import compute_qb_rush_splits_from_pbp

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / "data" / "raw" / "weekly_qb_repair_cache"
OUT_DIR = REPO_ROOT / "output" / "qb_h4" / "infra"
SPLITS_PATH = OUT_DIR / "designed_scramble_coverage.parquet"
MANIFEST_PATH = OUT_DIR / "designed_scramble_coverage_manifest.json"

PBP_SOURCES = (
    CACHE / "pbp_rush_2022.parquet",
    CACHE / "pbp_rush_2023_2024.parquet",
    CACHE / "pbp_rush_2025.parquet",
    CACHE / "pbp_qb_rush_features_2022_2025.parquet",
)


def _sha256(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_available_pbp() -> tuple[pd.DataFrame, dict]:
    """Concatenate available PBP sources; prefer richer columns when duplicating."""
    frames = []
    meta = {"sources_used": [], "sources_missing": [], "source_hashes": {}}
    for path in PBP_SOURCES:
        if not path.exists() or path.stat().st_size == 0:
            meta["sources_missing"].append(str(path.relative_to(REPO_ROOT)))
            continue
        df = pd.read_parquet(path)
        df["__source"] = path.name
        frames.append(df)
        rel = str(path.relative_to(REPO_ROOT))
        meta["sources_used"].append(rel)
        meta["source_hashes"][rel] = _sha256(path)
    if not frames:
        return pd.DataFrame(), meta
    # Prefer the dedicated seasonal files; drop duplicates from the combined blob
    # by (season, week, rusher, scramble, yards) when present.
    raw = pd.concat(frames, ignore_index=True, sort=False)
    # Keep one row preference: seasonal files over combined.
    raw["__pref"] = raw["__source"].map(
        lambda s: 0 if s.startswith("pbp_rush_") else 1
    )
    key_cols = [c for c in ("season", "week", "rusher_player_id", "qb_scramble", "rushing_yards", "rush_attempt") if c in raw.columns]
    if key_cols:
        raw = raw.sort_values("__pref").drop_duplicates(subset=key_cols, keep="first")
    return raw.drop(columns=["__pref"], errors="ignore"), meta


def build_coverage_table() -> tuple[pd.DataFrame, dict]:
    pbp, meta = load_available_pbp()
    if pbp.empty:
        raise RuntimeError(
            "No PBP rush sources available under data/raw/weekly_qb_repair_cache. "
            "Cannot extend designed/scramble coverage."
        )
    splits = compute_qb_rush_splits_from_pbp(pbp)
    splits["player_id"] = splits["player_id"].astype(str)
    splits["coverage_status"] = "observed"
    splits["source"] = "weekly_qb_repair_cache_pbp"
    # Coverage report by season
    seasons_present = sorted(int(s) for s in splits["season"].unique())
    unresolved = int(splits["player_id"].isna().sum()) if "player_id" in splits.columns else 0
    # Explicit uncovered seasons in the lookback window
    uncovered = [s for s in range(2018, 2026) if s not in seasons_present]
    manifest = {
        **meta,
        "seasons_with_coverage": seasons_present,
        "seasons_uncovered_no_pbp_in_repo": uncovered,
        "n_player_seasons": int(len(splits)),
        "n_unresolved_player_id": unresolved,
        "transformations": [
            "compute_qb_rush_splits_from_pbp: designed = rush_attempt & qb_scramble!=1",
            "scramble = rush_attempt & qb_scramble==1",
            "null never filled with 0; uncovered seasons omitted",
        ],
        "identity": "nflverse-style GSIS ids (00-00xxxxxx) as in weekly cache",
        "null_policy": "missing designed/scramble remains NaN; never pocket default",
    }
    return splits, manifest


def write_coverage_fixture() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits, manifest = build_coverage_table()
    splits.to_parquet(SPLITS_PATH, index=False)
    payload = splits[["player_id", "season", "designed_carries", "scramble_carries"]].sort_values(
        ["season", "player_id"]
    )
    content_hash = hashlib.sha256(payload.to_csv(index=False).encode()).hexdigest()
    manifest["content_hash"] = content_hash
    manifest["artifact"] = str(SPLITS_PATH.relative_to(REPO_ROOT))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_coverage() -> pd.DataFrame:
    if not SPLITS_PATH.exists() or SPLITS_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"H4 designed/scramble fixture missing: {SPLITS_PATH}. "
            "Run scripts/qb_h4_build_designed_coverage.py"
        )
    return pd.read_parquet(SPLITS_PATH)


def merge_coverage_into_history(history: pd.DataFrame) -> pd.DataFrame:
    """Attach designed/scramble per-active where coverage exists; leave NaN otherwise."""
    cov = load_coverage()
    out = history.copy()
    out["player_id"] = out["player_id"].astype(str)
    # Drop prior designed columns to avoid _x/_y, then re-merge coverage.
    drop = [
        c
        for c in (
            "designed_carries",
            "scramble_carries",
            "designed_rushing_yards",
            "scramble_rushing_yards",
            "designed_carries_per_active",
            "scramble_carries_per_active",
            "scramble_per_dropback",
            "designed_ypc",
            "scramble_ypa",
            "designed_rushing_yards_per_active",
            "scramble_rushing_yards_per_active",
        )
        if c in out.columns
    ]
    if drop:
        out = out.drop(columns=drop)
    keep = [c for c in cov.columns if c in (
        "player_id", "season", "designed_carries", "scramble_carries",
        "designed_rushing_yards", "scramble_rushing_yards", "dropbacks",
        "coverage_status", "source",
    )]
    out = out.merge(cov[keep], on=["player_id", "season"], how="left")
    act = pd.to_numeric(out.get("active_starts"), errors="coerce").replace(0, np.nan)
    des = pd.to_numeric(out.get("designed_carries"), errors="coerce")
    scr = pd.to_numeric(out.get("scramble_carries"), errors="coerce")
    des_yds = pd.to_numeric(out.get("designed_rushing_yards"), errors="coerce")
    scr_yds = pd.to_numeric(out.get("scramble_rushing_yards"), errors="coerce")
    out["designed_carries_per_active"] = des / act
    out["scramble_carries_per_active"] = scr / act
    out["designed_rushing_yards_per_active"] = des_yds / act
    out["scramble_rushing_yards_per_active"] = scr_yds / act
    att = pd.to_numeric(out.get("attempts_per_active"), errors="coerce")
    out["scramble_per_dropback"] = out["scramble_carries_per_active"] / att.replace(0, np.nan)
    out["designed_ypc"] = out["designed_rushing_yards_per_active"] / out[
        "designed_carries_per_active"
    ].replace(0, np.nan)
    out["scramble_ypa"] = out["scramble_rushing_yards_per_active"] / out[
        "scramble_carries_per_active"
    ].replace(0, np.nan)
    out["designed_coverage_status"] = np.where(
        des.notna() | scr.notna(), "observed", "uncovered"
    )
    return out
