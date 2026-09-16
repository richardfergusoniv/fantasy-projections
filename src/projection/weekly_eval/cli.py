"""CLI for the Role 2 Vegas weekly props evaluation comparator.

Research / shadow only. Does not promote, reseal, or change APP_PROJECTION_SOURCE.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.projection.contracts import OUTPUT_DIR, REPO_ROOT
from src.projection.weekly_eval.comparator import compare_shadow_to_vegas

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURE_DIR = PACKAGE_DIR / "fixtures"
DEFAULT_OUTPUT_DIR = Path(OUTPUT_DIR) / "shadow_vegas_props_compare"


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path(REPO_ROOT).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Role 2 shadow comparator: score a weekly board CSV against "
            "timestamped Vegas prop snapshots. Does not promote."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use committed weekly_eval synthetic fixtures (no live scrape, no production pointer).",
    )
    parser.add_argument(
        "--m3-dry-run",
        action="store_true",
        help=(
            "Produce an M3 dry-run Role 2 board and score it against the "
            "timestamped M3 props fixture (as_of <= kickoff). Still not promoting."
        ),
    )
    parser.add_argument("--props", default=None, help="Prop snapshot CSV.")
    parser.add_argument("--board", default=None, help="Shadow weekly board CSV (player-week means).")
    parser.add_argument("--outcomes", default=None, help="Optional realized outcomes CSV.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for summary.json (default: output/shadow_vegas_props_compare).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dry_run and args.m3_dry_run:
        parser.error("choose one of --dry-run or --m3-dry-run")

    source = "supplied"
    if args.m3_dry_run:
        from src.projection.weekly_latent.constants import DEFAULT_VEGAS_PROPS_M3_REL
        from src.projection.weekly_latent.run import run_milestone3

        m3_dir = Path(args.output).resolve().parent / "_m3_dry_run_board"
        m3 = run_milestone3(dry_run=True, output_dir=m3_dir, run_backtest=False)
        board = Path(m3.output_paths.get("shadow_board_role2.csv") or (m3_dir / "shadow_board_role2.csv"))
        props = Path(REPO_ROOT) / DEFAULT_VEGAS_PROPS_M3_REL
        outcomes = Path(args.outcomes) if args.outcomes else None
        source = "m3_dry_run"
    elif args.dry_run:
        props = DEFAULT_FIXTURE_DIR / "prop_snapshots.csv"
        board = DEFAULT_FIXTURE_DIR / "shadow_board.csv"
        outcomes = DEFAULT_FIXTURE_DIR / "outcomes.csv"
        source = "weekly_eval_fixture"
    else:
        if not args.props or not args.board:
            parser.error("--props and --board are required unless --dry-run or --m3-dry-run")
        props = Path(args.props)
        board = Path(args.board)
        outcomes = Path(args.outcomes) if args.outcomes else None

    summary = compare_shadow_to_vegas(
        board=board,
        snapshots=props,
        outcomes=outcomes,
    )
    summary["dry_run"] = bool(args.dry_run)
    summary["source"] = source
    if args.m3_dry_run:
        summary["harness"] = "weekly_eval"
        summary["snapshot_source"] = "m3_synthetic_fixture"
    summary["inputs"] = {
        "props": _repo_relative(props),
        "board": _repo_relative(board),
        "outcomes": _repo_relative(outcomes) if outcomes is not None else None,
    }
    out_dir = Path(args.output)
    summary["output"] = _repo_relative(out_dir / "summary.json")
    write_summary(summary, out_dir)
    print(json.dumps(_json_ready({k: summary[k] for k in (
        "role",
        "promoting",
        "gate_verdict",
        "dry_run",
        "source",
        "n_matched",
        "n_with_actuals",
        "metrics",
        "output",
    ) if k in summary}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
