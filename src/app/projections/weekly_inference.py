"""Execute verified weekly-v2 inference and persist provenance-linked outputs."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.app.projections.weekly_manifest import (
    OutputProvenance,
    REQUIRED_MODEL_ARTIFACTS,
    output_provenance_path,
    sha256_file,
    validate_manifest,
)
from src.app.projections.weekly_draws import write_weekly_draw_partition
from src.app.config import get_settings
from src.app.releases.publication import CandidateRow
from src.projection.weekly.config.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_dirs
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.features.panel import load_panel
import joblib

from src.projection.weekly.pipeline.rookie_projector import project_week_with_rookies
from src.projection.weekly.pipeline.season_projector import build_outlook_panel

logger = logging.getLogger(__name__)

STAT_COMPONENTS = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)


@dataclass(frozen=True)
class WeeklyInferenceResult:
    season: int
    week: int
    model_version: str
    input_hash: str
    output_path: Path
    output_sha256: str
    model_hashes: dict[str, str]
    frame: pl.DataFrame
    rows: tuple[CandidateRow, ...]

    @property
    def player_count(self) -> int:
        return len(self.rows)


def _prepare_panel(panel: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    week_rows = panel.filter((pl.col("season") == season) & (pl.col("week") == week))
    if not week_rows.is_empty():
        return panel
    history = panel.filter(pl.col("season") < season)
    outlook = build_outlook_panel(history, target_season=season)
    outlook_week = outlook.filter(pl.col("week") == week)
    if outlook_week.is_empty():
        raise ValueError(f"No outlook rows for season={season} week={week}")
    return pl.concat([history, outlook], how="diagonal_relaxed")


def _load_model_file(models_root: Path, stem: str) -> object:
    path = models_root / f"{stem}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _models_search_root(season: int) -> Path:
    season_dir = MODELS_DIR / f"season={season}"
    return season_dir if season_dir.exists() else MODELS_DIR


def run_weekly_inference(
    season: int,
    week: int,
    *,
    scoring: ScoringConfig | None = None,
    panel: pl.DataFrame | None = None,
    persist: bool = True,
) -> WeeklyInferenceResult:
    """Run weekly-v2 inference for one season/week using verified manifest artifacts."""
    validation = validate_manifest(season)
    if not validation.valid:
        raise ValueError(f"weekly_v2 manifest invalid: {validation.failures}")

    scoring = scoring or ScoringConfig()
    panel = _prepare_panel(panel if panel is not None else load_panel(), season, week)
    train_seasons = list(range(2016, season))
    models_root = _models_search_root(season)

    team_totals = _load_model_file(models_root, "team_totals")
    volume_models = {pos: _load_model_file(models_root, f"volume_{pos}") for pos in ("QB", "RB", "WR", "TE")}
    efficiency_models = {
        pos: _load_model_file(models_root, f"efficiency_{pos}") for pos in ("QB", "RB", "WR", "TE")
    }
    rookie_models = {}
    for pos in ("QB", "RB", "WR", "TE"):
        if (models_root / f"rookie_{pos}.joblib").exists():
            rookie_models[pos] = _load_model_file(models_root, f"rookie_{pos}")

    projected = project_week_with_rookies(
        panel,
        season=season,
        week=week,
        scoring=scoring,
        train_seasons=train_seasons,
        team_totals_model=team_totals,
        volume_models=volume_models,
        efficiency_models=efficiency_models,
        rookie_models=rookie_models or None,
    )

    rows = projections_to_candidate_rows(projected)
    input_hash = hashlib.sha256(
        f"{season}:{week}:{validation.model_version}:{len(rows)}".encode()
    ).hexdigest()

    output_dir = OUTPUTS_DIR / f"season={season}" / f"week={week:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weekly_projections.parquet"
    projected.write_parquet(output_path)
    output_sha256 = sha256_file(output_path)
    partition_sha256: str | None = None

    if persist:
        settings = get_settings()
        partition = write_weekly_draw_partition(
            projected,
            output_dir,
            draw_count=settings.simulation_draw_count,
            seed_salt=input_hash,
        )
        partition_sha256 = partition.sha256
        provenance = OutputProvenance(
            season=season,
            week=week,
            path=output_path,
            sha256=output_sha256,
            model_hashes=dict(validation.artifact_hashes),
            observed_at=datetime.now(UTC).isoformat(),
        )
        provenance_path = output_provenance_path(season, week)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(
                {
                    **provenance.to_dict(),
                    "output_path": str(output_path),
                    "model_version": validation.model_version,
                    "input_hash": input_hash,
                    "derivation": "weekly_v2_trained_inference",
                    "partition_sha256": partition_sha256,
                    "partition_path": str(output_dir / "stat_draw_partition.json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return WeeklyInferenceResult(
        season=season,
        week=week,
        model_version=str(validation.model_version),
        input_hash=input_hash,
        output_path=output_path,
        output_sha256=output_sha256,
        model_hashes=dict(validation.artifact_hashes),
        frame=projected,
        rows=rows,
    )


def projections_to_candidate_rows(frame: pl.DataFrame) -> tuple[CandidateRow, ...]:
    rows: list[CandidateRow] = []
    for record in frame.iter_rows(named=True):
        player_id = str(record.get("gsis_id") or record.get("player_id") or "")
        if not player_id:
            continue
        fp = float(record.get("fantasy_points") or 0.0)
        floor = float(record.get("floor") or max(0.0, fp * 0.7))
        ceiling = float(record.get("ceiling") or max(fp, fp * 1.3))
        play_prob = record.get("play_prob")
        availability = float(play_prob) if play_prob is not None else 1.0
        mean_json: dict = {
            "points": fp,
            "position": record.get("position"),
            "name": record.get("player_name"),
            "team": record.get("team"),
        }
        for stat in STAT_COMPONENTS:
            if stat in record and record[stat] is not None:
                mean_json[stat] = float(record[stat])
        rows.append(
            CandidateRow(
                player_id=player_id,
                team=record.get("team"),
                opponent=None,
                availability_probability=min(1.0, max(0.0, availability)),
                mean_json=mean_json,
                quantiles_json={
                    "0.1": floor,
                    "0.5": fp,
                    "0.9": ceiling,
                },
            )
        )
    return tuple(rows)


def hash_scaled_preseason_rows(
    players: dict,
    week: int,
    *,
    factor_fn,
) -> tuple[CandidateRow, ...]:
    """Explicit fallback path: deterministic preseason scaling (never ``trained``)."""
    rows: list[CandidateRow] = []
    for summary in players.values():
        factor = factor_fn(summary.player_id, week)
        rows.append(
            CandidateRow(
                player_id=summary.player_id,
                team=summary.team,
                opponent=None,
                availability_probability=summary.availability_probability,
                mean_json={
                    "points": summary.mean_points * factor,
                    "position": summary.position,
                    "name": summary.name,
                    "team": summary.team,
                    "derivation": "preseason_bundle_scaled",
                },
                quantiles_json={
                    key: float(value) * factor for key, value in summary.quantiles.items()
                },
            )
        )
    return tuple(rows)
