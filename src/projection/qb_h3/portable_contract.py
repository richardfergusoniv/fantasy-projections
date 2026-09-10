"""Portable QB team-reconciliation contract (evaluation fixture).

Prediction-side columns contain only information known before the predicted
season. Actual starts and season outcomes live in explicit label columns and
must never enter feature construction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.projection.qb_h3.projections_db import (
    ProjectionsDbUnusable,
    projections_db_status,
    require_usable_projections_db,
)
from src.projection.transitions import SEASON_GAMES

SCHEMA_VERSION = "qb_h3_reconcile_contract_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "output" / "qb_h3" / "infra"
FIXTURE_PARQUET = FIXTURE_DIR / "portable_qb_reconcile_fixture.parquet"
FIXTURE_MANIFEST = FIXTURE_DIR / "portable_qb_reconcile_manifest.json"

# Columns that may be used to construct predictions / allocation / reconcile.
PREDICTION_COLUMNS = (
    "prediction_season",
    "prediction_cutoff",
    "team",
    "player_id",
    "display_name",
    "preseason_depth_tier",
    "preseason_role",
    "is_rookie_at_cutoff",
    "prior_active_starts_sum",
    "prior_active_starts_mean",
    "prior_partial_exits_sum",
    "prior_player_attempts_per_active",
    "prior_player_carries_per_active",
    "prior_team_pass_attempts",
    "prior_team_qb_carries",
    "pred_team_pass_attempts_pg",
    "pred_team_qb_carries_pg",
    "destination_team_at_cutoff",
)

# Evaluation labels only — never features.
LABEL_COLUMNS = (
    "actual_starts",
    "actual_attempts",
    "actual_carries",
    "actual_passing_yards",
    "actual_rushing_yards",
    "actual_passing_tds",
    "actual_rushing_tds",
    "actual_points",
    "sealed_model_points_end_to_end",
    "sealed_projected_games",
)

PROVENANCE_COLUMNS = (
    "source_weekly",
    "source_eval",
    "source_active_rates",
    "schema_version",
)

ROW_COUNT_EXPECTATIONS = {
    "min_rows_per_eval_season": 60,
    "min_teams_per_eval_season": 32,
    "min_depth1_per_eval_season": 28,
    "eval_seasons": (2023, 2024, 2025),
}


class ReconciliationSkipped(RuntimeError):
    """Raised if evaluation would proceed without team reconciliation."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(frame: pd.DataFrame) -> str:
    """Deterministic hash of prediction-side + key columns (not labels)."""
    cols = [c for c in ("prediction_season", "player_id", "team") if c in frame.columns]
    cols += [c for c in PREDICTION_COLUMNS if c in frame.columns and c not in cols]
    slim = frame[cols].copy()
    slim["player_id"] = slim["player_id"].astype(str)
    slim = slim.sort_values(["prediction_season", "player_id", "team"]).reset_index(drop=True)
    payload = slim.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def leakage_audit(frame: pd.DataFrame, *, feature_columns: list[str] | None = None) -> dict:
    """Fail if any actual/label column is used as a prediction feature."""
    feats = list(feature_columns or PREDICTION_COLUMNS)
    leaked = [c for c in feats if c in LABEL_COLUMNS or str(c).startswith("actual_")]
    for c in feats:
        if c in frame.columns and c in LABEL_COLUMNS:
            leaked.append(c)
    leaked = sorted(set(leaked))
    ok = not leaked
    return {
        "ok": ok,
        "leaked_columns": leaked,
        "prediction_columns": [c for c in feats if c in frame.columns],
        "label_columns_present": [c for c in LABEL_COLUMNS if c in frame.columns],
        "season_games": SEASON_GAMES,
    }


def assert_no_label_leakage(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    audit = leakage_audit(frame, feature_columns=feature_columns)
    if not audit["ok"]:
        raise AssertionError(f"label leakage into features: {audit['leaked_columns']}")


def load_portable_fixture(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else FIXTURE_PARQUET
    if not p.exists() or p.stat().st_size == 0:
        raise ReconciliationSkipped(
            f"Portable QB reconciliation fixture missing or empty: {p}. "
            "Build it with scripts/qb_h3_build_portable_fixture.py. "
            "Team reconciliation cannot be skipped."
        )
    df = pd.read_parquet(p)
    if df.empty:
        raise ReconciliationSkipped("Portable fixture loaded but has 0 rows.")
    return df


def resolve_reconciliation_source(*, require_reconciliation: bool = True) -> dict:
    """Choose DB (if usable) or portable fixture. Never skip reconciliation."""
    db = projections_db_status()
    db_error = None
    if not db["usable"]:
        try:
            require_usable_projections_db()
        except ProjectionsDbUnusable as exc:
            db_error = str(exc)
    fixture_ok = FIXTURE_PARQUET.exists() and FIXTURE_PARQUET.stat().st_size > 0
    if db["usable"]:
        source = "projections_db"
    elif fixture_ok:
        source = "portable_fixture"
    else:
        source = None
    if require_reconciliation and source is None:
        raise ReconciliationSkipped(
            "Cannot run QB team reconciliation: projections.db is missing or "
            "zero bytes AND the portable fixture is absent. Refusing to skip "
            "reconciliation. "
            f"db={db} fixture={FIXTURE_PARQUET}"
        )
    return {
        "source": source,
        "projections_db": db,
        "projections_db_error": db_error,
        "fixture_path": str(FIXTURE_PARQUET),
        "fixture_present": fixture_ok,
        "reconciliation_will_run": source is not None,
        "schema_version": SCHEMA_VERSION,
    }


def write_manifest(*, frame: pd.DataFrame, sources: dict, extra: dict | None = None) -> dict:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    seasons = sorted(int(s) for s in frame["prediction_season"].unique())
    counts = {
        int(s): int((frame.prediction_season == s).sum()) for s in seasons
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_cutoff": "prediction_side uses only seasons < prediction_season",
        "content_hash": content_hash(frame),
        "n_rows": int(len(frame)),
        "seasons": seasons,
        "rows_per_season": counts,
        "row_count_expectations": ROW_COUNT_EXPECTATIONS,
        "source_hashes": {k: v for k, v in sources.items()},
        "prediction_columns": list(PREDICTION_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "leakage_audit": leakage_audit(frame),
        "committed_as": "versioned_evaluation_fixture",
        "does_not_include": [
            "projections.db",
            "player_week_panel.parquet",
            "sleeper league data",
            "secrets",
            "database URLs",
        ],
    }
    if extra:
        manifest.update(extra)
    FIXTURE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
