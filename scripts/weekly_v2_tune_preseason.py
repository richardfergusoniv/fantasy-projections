#!/usr/bin/env python3
"""Nested expanding-window volume architecture selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.projection.weekly.config.paths import OUTPUTS_DIR
from src.projection.weekly.evaluate.harness import load_panel_for_eval, sha256_file
from src.projection.weekly.evaluate.nested_selection import (
    NestedSelectionConfig,
    build_tuning_selection_payload,
    run_nested_selection,
    write_selection_artifact,
)
from src.projection.weekly.models.volume_config import DEFAULT_CANDIDATE_GRID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--scoring", default="half_ppr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Experiment namespace under output/weekly_v2/experiments/",
    )
    parser.add_argument(
        "--candidate-spec",
        type=Path,
        default=None,
        help="JSON file with candidate grid (default: frozen DEFAULT_CANDIDATE_GRID)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip if result exists")
    args = parser.parse_args(argv)

    panel_path = args.panel
    panel = load_panel_for_eval(panel_path)
    resolved_panel = panel_path or Path("data/processed/player_week_panel.parquet")
    if resolved_panel.exists():
        panel_hash = sha256_file(resolved_panel)
    else:
        from src.projection.weekly.evaluate.harness import default_panel_path

        panel_hash = sha256_file(default_panel_path())

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    namespace = args.namespace or f"volume_tune_{ts}"
    exp_dir = OUTPUTS_DIR / "experiments" / namespace
    exp_dir.mkdir(parents=True, exist_ok=True)

    result_path = exp_dir / "nested_selection.json"
    if args.resume and result_path.exists():
        print(result_path)
        return 0

    candidates = DEFAULT_CANDIDATE_GRID
    if args.candidate_spec:
        spec = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
        candidates = tuple(spec.get("candidates") or spec)

    config = NestedSelectionConfig(
        outer_start=args.start,
        outer_end=args.end,
        scoring=args.scoring,
        random_seed=args.seed,
        candidates=candidates,
    )
    result = run_nested_selection(
        panel,
        config=config,
        panel_path=resolved_panel,
        cache_dir=exp_dir / "candidate_cache",
    )
    write_selection_artifact(result, result_path)

    code_revision = None
    try:
        code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        code_revision = "unknown"

    selection_payload = build_tuning_selection_payload(
        result,
        experiment_id=namespace,
        panel_hash=panel_hash,
        code_revision=code_revision,
    )
    selection_payload["nested_result_path"] = str(result_path)
    selection_path = exp_dir / "tuning_selection.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2, default=str), encoding="utf-8")

    # Persist frozen grid and protocol for audit
    protocol_path = exp_dir / "selection_protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "candidate_grid": [c["name"] for c in candidates],
                "selection_rule": selection_payload["selection_rule"],
                "promotion_policy": config.promotion_policy,
                "outer_range": [args.start, args.end],
                "seed": args.seed,
                "panel_hash": panel_hash,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(result_path)
    print(selection_path)
    promote = selection_payload.get("promote", False)
    return 0 if promote else 2


if __name__ == "__main__":
    sys.exit(main())
