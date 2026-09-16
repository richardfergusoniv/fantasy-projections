"""Role 2 Vegas weekly-props hook for Milestone 3 (compare, do not replace).

Does not implement Role 3 blend. Does not depend on PR #83 merging: if
``src.projection.weekly_eval`` is importable it is used; otherwise a local
fixture scorer runs. ADP / season-long Vegas market-sanity bands are stubbed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.projection.contracts import REPO_ROOT
from src.projection.weekly_latent.constants import DEFAULT_VEGAS_PROPS_M3_REL
from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns

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
        "metrics": {"n": int(len(joined)), "mae_model_vs_market": mae},
        "caveat": (
            "Role 2 measurement only. Compare, do not replace. Not a promotion. "
            "Do not optimize toward market agreement. Feed "
            "output/shadow_weekly_schedule_m3/shadow_board_role2.csv to "
            "scripts/compare_shadow_vegas_props.py --board once PR #83 merges."
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
    if snapshots_path is None or dry_run:
        snap_path = root / DEFAULT_VEGAS_PROPS_M3_REL
    else:
        snap_path = Path(snapshots_path)
    if not snap_path.is_file():
        return {
            "schema_version": "weekly_latent_m3_vegas_compare_v1",
            "role": "evaluation_comparator",
            "role3_blend": False,
            "promoting": False,
            "gate_verdict": "not_promoting",
            "harness": "missing_fixture",
            "n_board": int(len(board_df)),
            "n_snapshots": 0,
            "n_matched": 0,
            "metrics": {"n": 0, "mae_model_vs_market": None},
            "caveat": f"no snapshots at {snap_path}; Role 2 compare skipped",
            "pr83_hook": (
                "src.projection.weekly_eval.compare_shadow_to_vegas("
                "board=shadow_board_role2.csv, snapshots=..., outcomes=...)"
            ),
        }
    snapshots = pd.read_csv(snap_path)
    try:
        from src.projection.weekly_eval.comparator import compare_shadow_to_vegas

        result = compare_shadow_to_vegas(board=board_df, snapshots=snapshots)
        result["harness"] = "weekly_eval"
        result["pr83_hook"] = "src.projection.weekly_eval.compare_shadow_to_vegas"
        result.setdefault("role3_blend", False)
        result.setdefault("promoting", False)
        result.setdefault("gate_verdict", "not_promoting")
        result.setdefault("role", "evaluation_comparator")
        return result
    except ImportError:
        return _local_compare(board_df, snapshots)
