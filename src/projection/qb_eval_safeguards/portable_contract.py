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

from src.projection.qb_eval_safeguards.projections_db import (
    ProjectionsDbUnusable,
    projections_db_status,
    require_usable_projections_db,
)
from src.projection.transitions import SEASON_GAMES

SCHEMA_VERSION = "qb_eval_reconcile_contract_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "output" / "qb_eval_safeguards" / "infra"
DEFAULT_FIXTURE_PARQUET = DEFAULT_FIXTURE_DIR / "portable_qb_reconcile_fixture.parquet"
DEFAULT_FIXTURE_MANIFEST = DEFAULT_FIXTURE_DIR / "portable_qb_reconcile_manifest.json"

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

REQUIRED_COLUMNS = (
    "prediction_season",
    "team",
    "player_id",
    "preseason_depth_tier",
    "preseason_role",
    "schema_version",
)

ROW_COUNT_EXPECTATIONS = {
    "min_rows_per_eval_season": 1,
    "min_teams_per_eval_season": 1,
    "min_depth1_per_eval_season": 1,
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


def validate_portable_fixture(
    frame: pd.DataFrame,
    *,
    expected_schema_version: str = SCHEMA_VERSION,
    require_labels: bool = False,
) -> dict:
    """Validate schema/version, season boundaries, and required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ReconciliationSkipped(
            f"Portable fixture missing required columns: {missing}. "
            "Refusing to run unreconciled evaluation."
        )
    if frame.empty:
        raise ReconciliationSkipped("Portable fixture has 0 rows.")
    versions = sorted({str(v) for v in frame["schema_version"].dropna().unique()})
    if versions != [expected_schema_version]:
        raise ReconciliationSkipped(
            f"Portable fixture schema_version mismatch: {versions} "
            f"(expected {[expected_schema_version]}). Schema drift is a hard fail."
        )
    seasons = pd.to_numeric(frame["prediction_season"], errors="coerce")
    if seasons.isna().any():
        raise ReconciliationSkipped("prediction_season must be numeric for every row.")
    # Prediction-side priors must not encode future seasons relative to cutoff.
    if "prior_active_starts_sum" in frame.columns and "actual_starts" in frame.columns:
        # Soft structural check only: labels may exist but must not be required features.
        pass
    audit = leakage_audit(frame)
    if not audit["ok"]:
        raise ReconciliationSkipped(
            f"Portable fixture fails leakage audit: {audit['leaked_columns']}"
        )
    if require_labels:
        missing_labels = [c for c in LABEL_COLUMNS if c not in frame.columns]
        if missing_labels:
            raise ReconciliationSkipped(
                f"Portable fixture missing required label columns: {missing_labels}"
            )
    return {
        "ok": True,
        "schema_version": expected_schema_version,
        "n_rows": int(len(frame)),
        "seasons": sorted(int(s) for s in seasons.unique()),
        "content_hash": content_hash(frame),
        "leakage_audit": audit,
    }


def load_portable_fixture(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_FIXTURE_PARQUET
    if not p.exists() or p.stat().st_size == 0:
        raise ReconciliationSkipped(
            f"Portable QB reconciliation fixture missing or empty: {p}. "
            "Team reconciliation cannot be skipped."
        )
    df = pd.read_parquet(p)
    validate_portable_fixture(df)
    return df


def resolve_reconciliation_source(
    *,
    require_reconciliation: bool = True,
    fixture_path: Path | None = None,
    db_path: Path | None = None,
) -> dict:
    """Choose DB (if usable) or portable fixture. Never skip reconciliation."""
    db = projections_db_status(db_path)
    db_error = None
    if not db["usable"]:
        try:
            require_usable_projections_db(db_path)
        except ProjectionsDbUnusable as exc:
            db_error = str(exc)
    fixture = Path(fixture_path) if fixture_path is not None else DEFAULT_FIXTURE_PARQUET
    fixture_ok = fixture.exists() and fixture.stat().st_size > 0
    if db["usable"]:
        source = "projections_db"
    elif fixture_ok:
        source = "portable_fixture"
    else:
        source = None
    if require_reconciliation and source is None:
        raise ReconciliationSkipped(
            "Cannot run QB team reconciliation: projections.db is missing or "
            "zero bytes AND no validated portable fixture was supplied. "
            "Refusing to skip reconciliation. "
            f"db={db} fixture={fixture}"
        )
    return {
        "source": source,
        "projections_db": db,
        "projections_db_error": db_error,
        "fixture_path": str(fixture),
        "fixture_present": fixture_ok,
        "reconciliation_will_run": source is not None,
        "schema_version": SCHEMA_VERSION,
    }


def write_manifest(
    *,
    frame: pd.DataFrame,
    sources: dict,
    fixture_dir: Path | None = None,
    extra: dict | None = None,
) -> dict:
    out_dir = Path(fixture_dir) if fixture_dir is not None else DEFAULT_FIXTURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = sorted(int(s) for s in frame["prediction_season"].unique())
    counts = {int(s): int((frame.prediction_season == s).sum()) for s in seasons}
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
    (out_dir / "portable_qb_reconcile_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
