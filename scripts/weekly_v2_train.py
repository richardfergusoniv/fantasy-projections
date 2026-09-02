#!/usr/bin/env python3
"""Train team totals, volume, efficiency, and rookie models."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src.app.projections.weekly_manifest import write_manifest_v2
from src.projection.weekly.config.paths import DATA_DIR, MODELS_DIR, TRAIN_END_SEASON, TRAIN_START_SEASON
from src.projection.weekly.data.cfbd_loader import load_college_features_for_drafted
from src.projection.weekly.data.nflverse_loader import load_combine, load_draft_picks
from src.projection.weekly.evaluate.harness import sha256_file
from src.projection.weekly.features.panel import load_panel
from src.projection.weekly.models.efficiency import train_efficiency_models
from src.projection.weekly.models.registry import set_registry_dir
from src.projection.weekly.models.rookie import train_rookie_model
from src.projection.weekly.models.team_totals import train_team_totals
from src.projection.weekly.models.volume import train_volume_models
from src.projection.weekly.models.volume_config import VolumeModelConfig


def _load_volume_options(tuning_selection: Path | None) -> tuple[dict, dict | None]:
    """Load volume options from an explicit tuning selection path only."""
    if tuning_selection is None or not tuning_selection.exists():
        return {}, None
    tuning = json.loads(tuning_selection.read_text(encoding="utf-8"))
    if not tuning.get("promote"):
        logging.warning("Tuning selection promote=false; training with default volume config")
        return {}, tuning
    return tuning.get("volume_options") or {}, tuning


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train projection models")
    parser.add_argument("--train-start", type=int, default=TRAIN_START_SEASON)
    parser.add_argument("--train-end", type=int, default=TRAIN_END_SEASON)
    parser.add_argument("--skip-rookie", action="store_true")
    parser.add_argument("--skip-cfbd", action="store_true", help="Skip CFBD college pull")
    parser.add_argument("--target-season", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=None, help="Candidate models directory")
    parser.add_argument(
        "--tuning-selection",
        type=Path,
        default=None,
        help="Explicit tuning selection JSON (required for tuned volume config)",
    )
    parser.add_argument(
        "--volume-options-json",
        type=Path,
        default=None,
        help="Direct volume options JSON (overrides tuning selection)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    models_dir = args.output_dir or (MODELS_DIR / f"season={args.target_season}")
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_registry_dir(models_dir)

    panel = load_panel()
    train_seasons = list(range(args.train_start, args.train_end + 1))
    if args.target_season in train_seasons:
        print(f"ERROR: target season {args.target_season} must not appear in train seasons", file=sys.stderr)
        return 2

    if args.volume_options_json:
        volume_options = json.loads(args.volume_options_json.read_text(encoding="utf-8"))
        tuning_artifact = None
    else:
        volume_options, tuning_artifact = _load_volume_options(args.tuning_selection)

    volume_config = VolumeModelConfig.from_options(volume_options)
    print(
        f"Training on seasons {train_seasons[0]}-{train_seasons[-1]} "
        f"for target {args.target_season} ({panel.height} panel rows) "
        f"volume_config={volume_config.fingerprint()}"
    )

    train_team_totals(panel, train_seasons=train_seasons, model_type="ridge")
    train_volume_models(
        panel,
        train_seasons=train_seasons,
        model_type="hgb",
        **volume_config.to_options(),
    )
    train_efficiency_models(panel, train_seasons=train_seasons, model_type="ridge")

    if not args.skip_rookie:
        college = None
        if not args.skip_cfbd:
            try:
                draft = load_draft_picks(train_seasons)
                college = load_college_features_for_drafted(draft, seasons=train_seasons)
            except Exception as exc:
                logging.warning("CFBD college features unavailable: %s", exc)
                college = None
        try:
            combine = load_combine(train_seasons)
        except Exception:
            combine = None
        train_rookie_model(
            panel,
            train_seasons=train_seasons,
            college=college,
            combine=combine,
            model_type="ridge",
        )

    import joblib as _joblib
    import sklearn

    panel_path = DATA_DIR / "processed" / "player_week_panel.parquet"
    code_revision = None
    try:
        code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        code_revision = "unknown"

    tuning_provenance = None
    if args.tuning_selection and args.tuning_selection.exists():
        tuning_provenance = {
            "path": str(args.tuning_selection.resolve()),
            "sha256": sha256_file(args.tuning_selection),
            "selected": (tuning_artifact or {}).get("selected"),
            "experiment_id": (tuning_artifact or {}).get("experiment_id"),
        }
        shutil.copy2(args.tuning_selection, models_dir / "tuning_selection.json")

    manifest_path = write_manifest_v2(
        target_season=args.target_season,
        train_seasons=train_seasons,
        models_dir=models_dir.parent,
        data_inputs=[("data/processed/player_week_panel.parquet", panel_path)],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_revision"] = code_revision
    manifest["library_versions"] = {
        "scikit-learn": sklearn.__version__,
        "joblib": _joblib.__version__,
    }
    manifest["volume_config"] = volume_config.to_options()
    manifest["volume_config_fingerprint"] = volume_config.fingerprint()
    if tuning_provenance:
        manifest["tuning_provenance"] = tuning_provenance
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Training complete. Models saved under {models_dir}")
    print(f"Manifest v2: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
