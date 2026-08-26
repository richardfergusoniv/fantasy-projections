"""Transactional publisher for the canonical projection artifact set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.draft_assistant.prepare import DRAFT_DATA_DIR, export_draft_data
from src.paths import DB_PATH
from src.projection.contracts import (
    COMPOSITION_VERSION,
    CONCENTRATION_PATH,
    MODELS_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_DIR,
    REPO_ROOT,
)
from src.projection.data_prep import get_conn
from src.projection.fantasy_points import compute_fantasy_points
from src.projection.inference.simulate import write_simulation_outputs
from src.projection.predict import project_season, with_display_names
from src.sentiment.snapshot import attach_sentiment
from src.team_stats.prepare import export_team_stats


SIMULATION_DRAWS = 1000
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> dict:
    """Commit and working-tree state of the CODE that built this board.

    Must be called before the staging directory exists. tempfile puts that
    directory inside REPO_ROOT (it has to, so the final os.replace stays on
    one filesystem), and once files are staged into it `git status` reports
    it as untracked. Called from inside the staging block this returned
    ``dirty: true`` on every publish without exception -- a provenance flag
    that can never say "clean" is worse than none, because it trains the
    reader to ignore the one field that would have flagged a board built
    from uncommitted code.
    """
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def _artifact_hashes() -> dict[str, str]:
    paths = [
        *sorted(Path(MODELS_DIR).glob("*.joblib")),
        *sorted(Path(MODELS_DIR).glob("*.csv")),
        Path(CONCENTRATION_PATH),
    ]
    return {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in paths if path.exists()
    }


def _data_snapshot() -> dict:
    path = Path(DB_PATH)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def validate_projection_contract(
    frame: pd.DataFrame,
    season: int,
    *,
    manifest: dict | None = None,
    projection_path: str | Path | None = None,
) -> None:
    """Reject stale/raw boards before any canonical or sentiment publish."""
    missing = sorted(set(OUTPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"projection schema is stale; missing columns: {missing}")
    seasons = set(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int))
    if seasons != {int(season)}:
        raise ValueError(f"projection season mismatch: expected {season}, found {sorted(seasons)}")
    if frame["projection_run_id"].nunique(dropna=False) != 1:
        raise ValueError("projection_run_id must be populated and identical on every row")
    if frame["composition_version"].nunique(dropna=False) != 1:
        raise ValueError("composition_version must be populated and identical on every row")
    healthy = ~frame["status_override_applied"].fillna(False).astype(bool)
    games = pd.to_numeric(frame.loc[healthy, "projected_games"], errors="coerce")
    volume_games = pd.to_numeric(frame.loc[healthy, "projected_volume_games"], errors="coerce")
    if not games.eq(17.0).all() or not volume_games.eq(17.0).all():
        raise ValueError(
            "stale exposure artifact: healthy rows must use 17 projected and volume games"
        )
    raw = pd.to_numeric(frame["projected_games_raw"], errors="coerce")
    if raw.notna().sum() == 0:
        raise ValueError("projected_games_raw is missing Gate-A diagnostics")

    if manifest is not None:
        if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError("projection run manifest schema is unsupported")
        if int(manifest.get("season", -1)) != int(season):
            raise ValueError("projection run manifest season does not match the board")
        run_id = str(frame["projection_run_id"].iloc[0])
        if manifest.get("run_id") != run_id:
            raise ValueError("projection run manifest does not match the board run_id")
        if projection_path is not None:
            expected = manifest.get("files", {}).get("projections", {}).get("sha256")
            if expected and sha256_file(projection_path) != expected:
                raise ValueError("projection CSV hash differs from its run manifest")


def _write_manifest(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def publish(
    season: int,
    *,
    as_of: str | None = None,
    simulate: bool = True,
    simulation_draws: int = SIMULATION_DRAWS,
) -> dict:
    """Build, validate, and commit projections plus all downstream artifacts.

    The season simulation runs HERE, between the projections being finalised
    and the draft board being exported. It cannot run outside: every publish
    mints a fresh ``projection_run_id``, so a simulation generated beforehand
    is stale against the board by construction and the percentile overlay's
    provenance guard refuses it forever.

    ``simulate=False`` skips it, which is only for a fast rebuild -- the board
    then ships without percentile columns rather than with stale ones.
    """
    run_id = str(uuid.uuid4())
    conn = get_conn()
    try:
        projections = project_season(conn, season, as_of=as_of)
        projections = with_display_names(conn, projections, season)
    finally:
        conn.close()
    projections = attach_sentiment(projections, season=season, as_of=as_of)
    projections["projection_run_id"] = run_id
    projections["composition_version"] = COMPOSITION_VERSION
    projections = projections[OUTPUT_COLUMNS].sort_values(
        ["position", "team", "player_id", "stat"]
    )
    validate_projection_contract(projections, season)
    fantasy = compute_fantasy_points(projections)

    simulation_manifest = None
    if simulate:
        # Runs against the board just built, so the percentiles carry this
        # run's id and the overlay's provenance guard accepts them.
        simulation_manifest = write_simulation_outputs(
            projections, season, n_draws=simulation_draws)

    final_paths = {
        "projections": Path(OUTPUT_DIR) / f"projections_{season}.csv",
        "fantasy_points": Path(OUTPUT_DIR) / f"fantasy_points_{season}.csv",
        "team_stats": Path(DRAFT_DATA_DIR) / f"team_stats_{season}.json",
        "draft_data": Path(DRAFT_DATA_DIR) / f"players_{season}.json",
        "manifest": Path(OUTPUT_DIR) / f"projection_run_{season}.json",
    }
    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Captured before staging exists - see _git_revision.
    code_revision = _git_revision()

    with tempfile.TemporaryDirectory(prefix=f"publish-{season}-", dir=REPO_ROOT) as temp_dir:
        stage = Path(temp_dir)
        staged = {
            "projections": stage / final_paths["projections"].name,
            "fantasy_points": stage / final_paths["fantasy_points"].name,
            "team_stats": stage / final_paths["team_stats"].name,
            "draft_data": stage / final_paths["draft_data"].name,
            "manifest": stage / final_paths["manifest"].name,
        }
        projections.to_csv(staged["projections"], index=False)
        fantasy.to_csv(staged["fantasy_points"], index=False)
        export_team_stats(
            season,
            projections_path=str(staged["projections"]),
            fantasy_path=str(staged["fantasy_points"]),
            out_path=str(staged["team_stats"]),
        )
        export_draft_data(
            season,
            fantasy_path=str(staged["fantasy_points"]),
            out_path=str(staged["draft_data"]),
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "season": int(season),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "as_of": as_of,
            "code_revision": code_revision,
            "artifact_hashes": _artifact_hashes(),
            "data_snapshot": _data_snapshot(),
            "exposure_policy": "healthy_active_17_games; explicit IR/PUP/suspension overrides only",
            "composition_version": COMPOSITION_VERSION,
            "simulation": simulation_manifest,
            "concentration_version": str(
                projections["concentration_calibration_version"].replace(
                    "not_applicable", pd.NA
                ).dropna().iloc[0]
            ),
            "files": {
                key: {
                    "path": str(final_paths[key].relative_to(REPO_ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(staged[key]),
                }
                for key in ("projections", "fantasy_points", "team_stats", "draft_data")
            },
        }
        _write_manifest(staged["manifest"], manifest)
        # The manifest is the commit marker and moves last. Consumers validate
        # its run_id/hash, so an interrupted multi-file replacement is rejected
        # rather than treated as a coherent publish.
        for key in ("projections", "fantasy_points", "team_stats", "draft_data", "manifest"):
            os.replace(staged[key], final_paths[key])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help=(
            "Skip the season simulation. The board then ships WITHOUT "
            "percentile columns; it never ships with stale ones."
        ),
    )
    parser.add_argument(
        "--simulation-draws", type=int, default=SIMULATION_DRAWS)
    args = parser.parse_args()
    manifest = publish(
        args.season,
        as_of=args.as_of,
        simulate=not args.no_simulate,
        simulation_draws=args.simulation_draws,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
