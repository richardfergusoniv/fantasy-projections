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
from src.projection.evaluation.release_report import (
    build_release_report_simulation,
    write_release_report_simulation,
)
from src.projection.fantasy_points import compute_fantasy_points
from src.projection.inference.simulate import write_simulation_outputs
from src.projection.inference.simulation_config import load_simulation_config, profile_draws
from src.projection.predict import project_season, with_display_names
from src.sentiment.snapshot import attach_sentiment
from src.team_stats.prepare import export_team_stats


SIMULATION_DRAWS = 10000
ACCURACY_FIRST_BOARD = Path(OUTPUT_DIR) / "accuracy_first_2026"
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

    # Captured before ANYTHING this function writes into the working tree.
    # The simulation below writes tracked artifacts under output/model_v3, so
    # capturing after it reintroduces exactly the contamination _git_revision
    # documents: a dirty flag describing the publish's own output rather than
    # the code that produced it.
    code_revision = _git_revision()

    simulation_manifest = None
    if simulate:
        selected_board = None
        selected_board_hash = None
        selected_board_model_id = None
        accuracy_board_path = ACCURACY_FIRST_BOARD / f"fantasy_points_{season}.csv"
        if accuracy_board_path.exists():
            selected_board = pd.read_csv(accuracy_board_path)
            selected_board_hash = sha256_file(accuracy_board_path)
            selected_board_model_id = "accuracy_first_ensemble"
        simulation_manifest = write_simulation_outputs(
            projections,
            season,
            n_draws=simulation_draws,
            selected_board=selected_board,
            selected_board_hash=selected_board_hash,
            selected_board_model_id=selected_board_model_id,
            simulation_profile="dev",
        )

    final_paths = {
        "projections": Path(OUTPUT_DIR) / f"projections_{season}.csv",
        "fantasy_points": Path(OUTPUT_DIR) / f"fantasy_points_{season}.csv",
        "team_stats": Path(DRAFT_DATA_DIR) / f"team_stats_{season}.json",
        "draft_data": Path(DRAFT_DATA_DIR) / f"players_{season}.json",
        "manifest": Path(OUTPUT_DIR) / f"projection_run_{season}.json",
    }
    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

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
    sim_report = build_release_report_simulation(
        season=season,
        projection_run=manifest,
        simulation_manifest=simulation_manifest,
    )
    write_release_report_simulation(sim_report, season=season)
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
    parser.add_argument(
        "--simulation-profile",
        default="dev",
        help="Simulation profile from config/simulation.json (dev, publish, release_candidate).",
    )
    parser.add_argument(
        "--artifact-namespace",
        default=None,
        help="Required for publish and release_candidate profiles; namespaced output root.",
    )
    parser.add_argument(
        "--rollout-label",
        default=None,
        help="Human-readable rollout label stored on RC manifests and validation.",
    )
    args = parser.parse_args()

    sim_config = load_simulation_config()
    profile = args.simulation_profile
    if profile != "dev" and not args.artifact_namespace:
        raise SystemExit(
            f"--artifact-namespace is required for non-default simulation profile {profile!r}"
        )
    if profile == "release_candidate":
        from src.projection.release_candidate import publish_release_candidate

        if not args.artifact_namespace:
            raise SystemExit("--artifact-namespace is required for release_candidate publish")
        if not args.rollout_label:
            raise SystemExit("--rollout-label is required for release_candidate publish")
        profile_draws_value = profile_draws(sim_config, profile)
        draws = args.simulation_draws
        if profile_draws_value is not None:
            draws = int(profile_draws_value)
        result = publish_release_candidate(
            args.season,
            artifact_namespace=args.artifact_namespace,
            simulation_draws=draws,
            simulation_profile=profile,
            rollout_label=args.rollout_label,
            as_of=args.as_of,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if profile == "publish":
        from src.projection.release_bundle_publish import publish_release_bundle

        profile_draws_value = profile_draws(sim_config, profile)
        draws = args.simulation_draws
        if profile_draws_value is not None:
            draws = int(profile_draws_value)
        result = publish_release_bundle(
            args.season,
            artifact_namespace=args.artifact_namespace,
            simulation_draws=draws,
            simulation_profile=profile,
            as_of=args.as_of,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

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
