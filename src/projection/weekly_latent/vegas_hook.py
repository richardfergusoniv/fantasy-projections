"""Role 2 Vegas weekly-props hook for Milestone 3 (compare, do not replace).

Does not implement Role 3 blend. Uses ``weekly_eval`` when importable; otherwise
a local fixture scorer. Live weeks need timestamped snapshots — the synthetic
M3 fixture is only for dry-run / harness demonstration. ADP / season-long
Vegas market-sanity bands are stubbed.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import REPO_ROOT
from src.projection.weekly_latent.constants import DEFAULT_VEGAS_PROPS_M3_REL
from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns

LIVE_PROPS_ENV = "WEEKLY_EVAL_PROPS_PATH"
DEFAULT_LIVE_PROPS_REL = "output/shadow_vegas_props_compare/live_snapshots.csv"
_JOIN_KEYS = ("player_id", "season", "week", "market")
M3_DRY_RUN_METRICS_CAVEAT = (
    "Harness / scale-mismatch dry-run check, not weekly accuracy. "
    "M3 dry-run keeps full-season mass across weeks 1–3, so MAE compares "
    "season-scale model means to weekly lines. n_matched>0 proves the join; "
    "it is not a promotion argument."
)
ROLE2_MARKET_MAP: dict[str, str] = {
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "attempts": "pass_attempts",
    "completions": "pass_completions",
    "interceptions": "pass_ints",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "carries": "rush_attempts",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
    "receptions": "receptions",
    "targets": "targets",
}


def export_role2_board(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Long Role-2 board: player_id, season, week, market, model_mean.

    This CSV is the later input to ``compare_shadow_vegas_props.py --board``
    once PR #83 lands. It is not a production pointer.
    """
    refuse_forbidden_m2_columns(player_weeks, where="M3 Role 2 board export")
    need = {"player_id", "week"}
    missing = need - set(player_weeks.columns)
    if missing:
        raise ValueError(f"player-weeks missing columns for Role 2 export: {sorted(missing)}")
    season = (
        pd.to_numeric(player_weeks["season"], errors="coerce").fillna(2026).astype(int)
        if "season" in player_weeks.columns
        else pd.Series(2026, index=player_weeks.index)
    )
    rows: list[dict[str, Any]] = []
    for idx in player_weeks.index:
        rec = player_weeks.loc[idx]
        for src_col, market in ROLE2_MARKET_MAP.items():
            if src_col not in player_weeks.columns:
                continue
            rows.append(
                {
                    "player_id": str(rec["player_id"]),
                    "season": int(season.loc[idx]),
                    "week": int(rec["week"]),
                    "market": market,
                    "model_mean": float(pd.to_numeric(rec[src_col], errors="coerce") or 0.0),
                }
            )
    board = pd.DataFrame(rows)
    if board.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", "market", "model_mean"])
    refuse_forbidden_m2_columns(board, where="M3 Role 2 board")
    return board


def market_sanity_bands() -> dict[str, Any]:
    """Optional empirical ADP / season-Vegas bands — deferred, not a driver."""
    return {
        "status": "deferred",
        "used_as_driver": False,
        "role": "empirical_bands_only",
        "note": (
            "ADP and season-long Vegas may later form outside market-sanity "
            "bands (widen when those two disagree). They are not training "
            "targets, weekly drivers, or a promotion argument. Stubbed in M3."
        ),
    }


def _resolve_snapshots_path(
    *,
    snapshots_path: str | Path | None,
    dry_run: bool,
    repo_root: Path,
) -> tuple[Path | None, str]:
    """Prefer a supplied path; dry-run uses the M3 fixture; else live or missing."""
    if snapshots_path is not None:
        return Path(snapshots_path), "supplied"
    if dry_run:
        return repo_root / DEFAULT_VEGAS_PROPS_M3_REL, "m3_synthetic_fixture"
    env = os.environ.get(LIVE_PROPS_ENV, "").strip()
    if env:
        env_path = Path(env)
        if not env_path.is_file():
            raise FileNotFoundError(
                f"{LIVE_PROPS_ENV} is set to {env_path} but that path is not a "
                "file; refusing to fall back to live_snapshots.csv (fail closed)"
            )
        return env_path, "live_timestamped"
    live = repo_root / DEFAULT_LIVE_PROPS_REL
    if live.is_file():
        return live, "live_timestamped"
    return None, "missing_live_snapshots"


def _live_data_blocker(*, n_board: int) -> dict[str, Any]:
    return {
        "status": "blocked",
        "n_board": int(n_board),
        "must_supply": [
            (
                "CSV of timestamped Vegas weekly prop snapshots at "
                "WEEKLY_EVAL_PROPS_PATH or output/shadow_vegas_props_compare/live_snapshots.csv"
            ),
            (
                "player_id matching the M3 / sealed board (gsis-style ids such as "
                "00-0034857), not dry-run names (p-qb / p-wr)"
            ),
            "season, week, market using Role 2 names (pass_yards, rec_yards, receptions, ...)",
            "as_of (ISO-8601 UTC) and kickoff_at with as_of <= kickoff_at",
            "line or implied_mean; implied_p_over when de-vig is available",
            "optional outcomes CSV after the week: player_id, season, week, market, actual",
        ],
        "note": (
            "Role 1 weekly_props snapshots live in the DB / artifact store, not "
            "as committed as_of CSVs. Export a leakage-safe Role 2 frame before "
            "kickoff; do not reuse a later closing line as the historical as-of."
        ),
    }


def _require_join_keys(frame: pd.DataFrame, *, label: str) -> None:
    if frame is None or frame.empty:
        return
    missing = [k for k in _JOIN_KEYS if k not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing join keys: {missing}")


def _match_keys(
    frame: pd.DataFrame | None,
    *,
    label: str,
) -> set[tuple[str, int, int, str]]:
    if frame is None or frame.empty:
        return set()
    _require_join_keys(frame, label=label)
    return set(
        zip(
            frame["player_id"].astype(str),
            pd.to_numeric(frame["season"], errors="coerce").fillna(0).astype(int),
            pd.to_numeric(frame["week"], errors="coerce").fillna(0).astype(int),
            frame["market"].astype(str),
        )
    )


def _empty_match(
    board: pd.DataFrame | None = None,
    snapshots: pd.DataFrame | None = None,
) -> dict[str, Any]:
    board_keys = _match_keys(board, label="board")
    snap_keys = _match_keys(snapshots, label="snapshots")
    return {
        "join_keys": list(_JOIN_KEYS),
        "n_board_unmatched": len(board_keys),
        "n_snapshot_unmatched": len(snap_keys),
        "unmatched_board_ids_sample": sorted({row[0] for row in board_keys})[:8],
        "unmatched_snapshot_ids_sample": sorted({row[0] for row in snap_keys})[:8],
    }


def _describe_match(
    board: pd.DataFrame,
    snapshots: pd.DataFrame,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    board_keys = _match_keys(board, label="board")
    snap_keys = _match_keys(snapshots, label="snapshots")
    joined_keys = _match_keys(joined, label="joined")
    unmatched_board = board_keys - joined_keys
    unmatched_snap = snap_keys - joined_keys
    return {
        "join_keys": list(_JOIN_KEYS),
        "n_board_unmatched": len(unmatched_board),
        "n_snapshot_unmatched": len(unmatched_snap),
        "unmatched_board_ids_sample": sorted({row[0] for row in unmatched_board})[:8],
        "unmatched_snapshot_ids_sample": sorted({row[0] for row in unmatched_snap})[:8],
    }


def _parse_stamp(value: object):
    from src.projection.weekly_latent.constants import parse_available_at

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    return parse_available_at(text)


def _local_compare(
    board: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> dict[str, Any]:
    """Fixture scorer used when weekly_eval (PR #83) is not on the branch."""
    need_snap = {"player_id", "season", "week", "market", "as_of", "kickoff_at"}
    missing = need_snap - set(snapshots.columns)
    if "as_of" in missing:
        raise ValueError("prop snapshot is missing as_of")
    if "kickoff_at" in missing:
        raise ValueError("prop snapshot is missing kickoff_at")
    if missing:
        raise ValueError(f"prop snapshots missing columns: {sorted(missing)}")
    for rec in snapshots.to_dict(orient="records"):
        as_of = _parse_stamp(rec.get("as_of"))
        kickoff = _parse_stamp(rec.get("kickoff_at"))
        if as_of is None:
            raise ValueError("prop snapshot is missing as_of")
        if kickoff is None:
            raise ValueError("prop snapshot is missing kickoff_at")
        if as_of > kickoff:
            raise ValueError(f"as_of {as_of.isoformat()} is after kickoff_at {kickoff.isoformat()}")
    snap = snapshots.copy()
    snap["player_id"] = snap["player_id"].astype(str)
    snap["season"] = snap["season"].astype(int)
    snap["week"] = snap["week"].astype(int)
    snap["market"] = snap["market"].astype(str)
    if "implied_mean" in snap.columns:
        snap["market_mean"] = pd.to_numeric(snap["implied_mean"], errors="coerce")
    elif "line" in snap.columns:
        snap["market_mean"] = pd.to_numeric(snap["line"], errors="coerce")
    else:
        raise ValueError("prop snapshot needs line or implied_mean")
    keys = ["player_id", "season", "week", "market"]
    joined = board.merge(snap[keys + ["market_mean", "as_of", "kickoff_at"]], on=keys, how="inner")
    mae = None
    if len(joined):
        delta = (
            pd.to_numeric(joined["model_mean"], errors="coerce")
            - pd.to_numeric(joined["market_mean"], errors="coerce")
        ).abs()
        mae = float(delta.mean())
    return {
        "schema_version": "weekly_latent_m3_vegas_compare_v1",
        "role": "evaluation_comparator",
        "role3_blend": False,
        "promoting": False,
        "gate_verdict": "not_promoting",
        "harness": "local_fixture_until_weekly_eval",
        "n_board": int(len(board)),
        "n_snapshots": int(len(snap)),
        "n_matched": int(len(joined)),
        "match": _describe_match(board, snap, joined),
        "metrics": {"n": int(len(joined)), "mae_model_vs_market": mae},
        "caveat": (
            "Role 2 measurement only. Compare, do not replace. Not a promotion. "
            "Do not optimize toward market agreement. Feed "
            "output/shadow_weekly_schedule_m3/shadow_board_role2.csv to "
            "scripts/compare_shadow_vegas_props.py --board."
        ),
        "pr83_hook": (
            "src.projection.weekly_eval.compare_shadow_to_vegas("
            "board=shadow_board_role2.csv, snapshots=..., outcomes=...)"
        ),
    }


def compare_m3_to_vegas_props(
    *,
    board: pd.DataFrame | Path | str,
    snapshots_path: str | Path | None = None,
    dry_run: bool = False,
    blend_weights: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Score an M3 Role-2 board against timestamped props. Never blends."""
    if blend_weights:
        raise ValueError(
            "Role 3 blend is forbidden until a market-free challenger is reported; "
            f"got blend_weights={dict(blend_weights)}"
        )
    if isinstance(board, (str, Path)):
        board_df = pd.read_csv(board)
    else:
        board_df = board.copy()
    refuse_forbidden_m2_columns(board_df, where="M3 vegas compare board")
    root = Path(repo_root or REPO_ROOT)
    snap_path, snap_source = _resolve_snapshots_path(
        snapshots_path=snapshots_path,
        dry_run=dry_run,
        repo_root=root,
    )
    if snap_path is None or not snap_path.is_file():
        missing = (
            "missing_live_snapshots"
            if snap_source == "missing_live_snapshots"
            else "missing_fixture"
        )
        blocker = _live_data_blocker(n_board=len(board_df))
        return {
            "schema_version": "weekly_latent_m3_vegas_compare_v1",
            "role": "evaluation_comparator",
            "role3_blend": False,
            "promoting": False,
            "gate_verdict": "not_promoting",
            "harness": missing,
            "snapshot_source": snap_source,
            "n_board": int(len(board_df)),
            "n_snapshots": 0,
            "n_matched": 0,
            "match": _empty_match(board=board_df),
            "metrics": {"n": 0, "mae_model_vs_market": None},
            "live_data_blocker": blocker,
            "caveat": (
                "no live timestamped Vegas snapshots; Role 2 compare skipped. "
                "Do not score a sealed 2026 board against the M3 dry-run fixture."
            ),
            "pr83_hook": (
                "src.projection.weekly_eval.compare_shadow_to_vegas("
                "board=shadow_board_role2.csv, snapshots=..., outcomes=...)"
            ),
        }
    snapshots = pd.read_csv(snap_path)
    try:
        # Dynamic so AST import-graph walking cannot follow weekly_eval into
        # feature_outcome_split (M1/M2 guards on weekly_latent.run).
        comparator = importlib.import_module("src.projection.weekly_eval.comparator")
        compare_shadow_to_vegas = getattr(comparator, "compare_shadow_to_vegas", None)
        if compare_shadow_to_vegas is None:
            local = _local_compare(board_df, snapshots)
            local["snapshot_source"] = snap_source
            if snap_source == "m3_synthetic_fixture":
                local["metrics_caveat"] = M3_DRY_RUN_METRICS_CAVEAT
            return local

        result = compare_shadow_to_vegas(board=board_df, snapshots=snapshots)
        result["harness"] = "weekly_eval"
        result["snapshot_source"] = snap_source
        result["pr83_hook"] = "src.projection.weekly_eval.compare_shadow_to_vegas"
        result.setdefault("role3_blend", False)
        result.setdefault("promoting", False)
        result.setdefault("gate_verdict", "not_promoting")
        result.setdefault("role", "evaluation_comparator")
        if snap_source == "m3_synthetic_fixture":
            result["metrics_caveat"] = M3_DRY_RUN_METRICS_CAVEAT
        return result
    except ImportError:
        local = _local_compare(board_df, snapshots)
        local["snapshot_source"] = snap_source
        if snap_source == "m3_synthetic_fixture":
            local["metrics_caveat"] = M3_DRY_RUN_METRICS_CAVEAT
        return local
