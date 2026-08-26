"""Sync fantasy-projections-2 season outputs into draft-assistant CSVs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.projection.team_reconcile import (
    RECEIVING_POSITIONS,
    TARGETS_PER_ATTEMPT,
    TEAM_IDENTITY_PAIRS,
    reconcile_team_season_identities,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output"
# Never overwrite this repo's native rate-forecast board. Cross-repo syncs land
# in a namespaced folder so v1 and v2 projections stay distinguishable.
MODEL_V2_OUTPUT_DIR = OUTPUT_DIR / "model_v2"
DEFAULT_V2_ROOT = Path(os.environ.get("FANTASY_PROJECTIONS_V2", REPO_ROOT.parent / "fantasy-projections-2"))

STAT_COLS = [
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
]

FANTASY_POINTS_COLS = [
    "team_identity_scale",
    "player_id",
    "display_name",
    "position",
    "team",
    "fantasy_pts",
    "fantasy_pts_low",
    "fantasy_pts_high",
    "fantasy_pts_season",
    "projected_games",
    "source",
    "low_confidence",
    "role",
    "depth_chart_status",
    "depth_rank",
    "season",
]


def _v2_season_csv(v2_root: Path, season: int) -> Path:
    return v2_root / "outputs" / f"season_projections_{season}.csv"


def run_v2_season_project(
    *,
    season: int,
    v2_root: Path,
    scoring: str = "half_ppr",
    train_end: int | None = None,
) -> Path:
    """Invoke fantasy-projections-2 project_season.py and return season CSV path."""
    script = v2_root / "scripts" / "project_season.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing v2 season script: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--season",
        str(season),
        "--scoring",
        scoring,
    ]
    if train_end is not None:
        cmd.extend(["--train-end", str(train_end)])

    env = os.environ.copy()
    # Ensure v2 package is importable when run as script
    src = str(v2_root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(v2_root), env=env)

    out = _v2_season_csv(v2_root, season)
    if not out.exists():
        raise FileNotFoundError(f"v2 did not write {out}")
    return out


def map_season_df_to_fantasy_points(df: pd.DataFrame, *, season: int) -> pd.DataFrame:
    """Map v2 season_projections columns onto draft fantasy_points schema."""
    out = pd.DataFrame()
    out["player_id"] = df.get("gsis_id", df.get("player_id")).astype(str)
    out["display_name"] = df.get("player_name", df.get("display_name"))
    out["position"] = df["position"]
    out["team"] = df["team"]
    out["fantasy_pts"] = pd.to_numeric(df.get("fantasy_pts", df.get("fantasy_points")), errors="coerce")
    low = None
    if "fantasy_pts_low" in df.columns:
        low = pd.to_numeric(df["fantasy_pts_low"], errors="coerce")
    if "floor" in df.columns:
        floor = pd.to_numeric(df["floor"], errors="coerce")
        low = floor if low is None else low.fillna(floor)
    out["fantasy_pts_low"] = low

    high = None
    if "fantasy_pts_high" in df.columns:
        high = pd.to_numeric(df["fantasy_pts_high"], errors="coerce")
    if "ceiling" in df.columns:
        ceiling = pd.to_numeric(df["ceiling"], errors="coerce")
        high = ceiling if high is None else high.fillna(ceiling)
    out["fantasy_pts_high"] = high
    out["projected_games"] = pd.to_numeric(df.get("projected_games"), errors="coerce").fillna(17.0)
    if "fantasy_pts_season" in df.columns:
        out["fantasy_pts_season"] = pd.to_numeric(df["fantasy_pts_season"], errors="coerce")
    else:
        out["fantasy_pts_season"] = out["fantasy_pts"] * out["projected_games"]
    out["source"] = df["source"] if "source" in df.columns else "v2_team_first"
    out["source"] = out["source"].fillna("v2_team_first")
    if "low_confidence" in df.columns:
        out["low_confidence"] = df["low_confidence"].fillna(False).astype(bool)
    else:
        out["low_confidence"] = False
    out["role"] = df["role"] if "role" in df.columns else None
    out["depth_chart_status"] = (
        df["depth_chart_status"] if "depth_chart_status" in df.columns else None
    )
    out["depth_rank"] = (
        pd.to_numeric(df["depth_rank"], errors="coerce") if "depth_rank" in df.columns else None
    )
    out["season"] = season
    # Identity until reconcile_team_season_identities has something to say, so
    # the column is always present for downstream consumers.
    out["team_identity_scale"] = 1.0

    out = out[out["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    out = out.dropna(subset=["player_id", "display_name", "fantasy_pts"])
    out = out.sort_values("fantasy_pts", ascending=False).reset_index(drop=True)
    return out


# Pass/catch season identities live in src.projection.team_reconcile\n# (reconcile_team_season_identities); re-exported via the import above.\n\n# Half-PPR, 4-point passing TD -- matches src/projection/fantasy_points.SCORING
# and v2's own half_ppr ScoringConfig. Used only to measure how much the team
# identity reconciliation moved a player, never to replace v2's fantasy points
# (which carry a fitted per-position calibration on top of the scored line).
_SCORING = {
    "passing_yards": 1 / 25, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 1 / 10, "rushing_tds": 6,
    "receiving_yards": 1 / 10, "receiving_tds": 6, "receptions": 0.5,
}


def _scored(frame: pd.DataFrame, value_col: str) -> pd.Series:
    wide = frame.pivot_table(
        index="player_id", columns="stat", values=value_col, aggfunc="first"
    )
    total = pd.Series(0.0, index=wide.index)
    for stat, weight in _SCORING.items():
        if stat in wide.columns:
            total = total + wide[stat].fillna(0.0) * weight
    return total


def apply_identity_scale_to_points(
    fantasy: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    """Carry the team reconciliation through to season fantasy points.

    ``reconcile_team_season_identities`` fixes the stat lines, but the board
    ranks ``fantasy_pts_season``, which comes straight from v2 and would still
    carry the un-reconciled games artifact. Rather than rescoring from the
    reconciled line -- which would throw away v2's fitted per-position points
    calibration, an offset of +1.4 to +3.6 points per game at draft-relevant
    depth -- this measures each player's own before/after ratio on the scored
    line and applies it multiplicatively. A player whose points are mostly
    rushing is barely moved; a receiver on a team whose quarterback room did not
    cover the season is moved by close to the full team factor.

    Per-game rates stay as v2 produced them, so ``fantasy_pts_season`` is no
    longer exactly ``fantasy_pts * projected_games``; the ratio is published as
    ``team_identity_scale`` so the gap is visible rather than mysterious.
    """
    ratio = (_scored(after, "pred_season") / _scored(before, "pred_season")).replace(
        [float("inf"), -float("inf")], pd.NA
    )
    out = fantasy.copy()
    scale = out["player_id"].map(ratio).astype(float).fillna(1.0).clip(0.5, 2.0)
    out["team_identity_scale"] = scale
    for col in ("fantasy_pts_season",):
        if col in out.columns:
            out[col] = out[col] * scale
    return out


def map_season_df_to_long_projections(df: pd.DataFrame, *, season: int) -> pd.DataFrame:
    """Wide v2 season rows → long projections_<season>.csv for team_stats.prepare."""
    base = map_season_df_to_fantasy_points(df, season=season)
    rows: list[dict] = []
    for _, r in df.iterrows():
        pid = str(r.get("gsis_id", r.get("player_id", "")))
        pos = r.get("position")
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        games = float(r.get("projected_games") or 17.0)
        meta = {
            "player_id": pid,
            "display_name": r.get("player_name", r.get("display_name")),
            "team": r.get("team"),
            "position": pos,
            "projected_games": games,
            "role": r.get("role"),
            "depth_rank": r.get("depth_rank"),
            "depth_chart_status": r.get("depth_chart_status"),
            "source": r.get("source", "v2_team_first"),
            "low_confidence": bool(r.get("low_confidence") or False),
            "season": season,
        }
        for stat in STAT_COLS:
            pg = r.get(stat)
            if pg is None or (isinstance(pg, float) and pd.isna(pg)):
                pg_val = None
                season_val = None
            else:
                pg_val = float(pg)
                season_val = pg_val * games
            rows.append(
                {
                    **meta,
                    "stat": stat,
                    "pred_pg": pg_val,
                    "pred_pg_low": None,
                    "pred_pg_high": None,
                    "pred_season": season_val,
                    "pred_season_low": None,
                    "pred_season_high": None,
                }
            )
    long = pd.DataFrame(rows)
    long_raw = long.copy()
    long = reconcile_team_season_identities(long)
    # Attach fantasy points for convenience (team_stats merges fantasy file separately)
    if not long.empty and not base.empty:
        long = long.merge(
            base[["player_id", "fantasy_pts", "fantasy_pts_season", "fantasy_pts_low", "fantasy_pts_high"]],
            on="player_id",
            how="left",
        )
    long.attrs["unreconciled"] = long_raw
    return long


def sync_from_v2(
    *,
    season: int,
    v2_root: Path | None = None,
    run_project: bool = True,
    scoring: str = "half_ppr",
    train_end: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write team-first (v2) boards under output/model_v2/ only.

    Does not touch this repo's canonical ``output/fantasy_points_*.csv`` /
    ``output/projections_*.csv``, which belong to the native rate-forecast (v1)
    pipeline.
    """
    v2_root = Path(v2_root or DEFAULT_V2_ROOT)
    out_dir = Path(output_dir or MODEL_V2_OUTPUT_DIR)
    if run_project:
        season_csv = run_v2_season_project(
            season=season, v2_root=v2_root, scoring=scoring, train_end=train_end
        )
    else:
        season_csv = _v2_season_csv(v2_root, season)
        if not season_csv.exists():
            raise FileNotFoundError(
                f"Missing {season_csv}. Run with --run-project or "
                f"`python scripts/project_season.py --season {season}` in v2."
            )

    raw = pd.read_csv(season_csv)
    fantasy = map_season_df_to_fantasy_points(raw, season=season)
    projections = map_season_df_to_long_projections(raw, season=season)
    unreconciled = projections.attrs.get("unreconciled")
    if unreconciled is not None:
        fantasy = apply_identity_scale_to_points(fantasy, unreconciled, projections)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = out_dir / f"fantasy_points_{season}.csv"
    proj_path = out_dir / f"projections_{season}.csv"
    fantasy.to_csv(fp_path, index=False)
    projections.to_csv(proj_path, index=False)
    return {"fantasy_points": fp_path, "projections": proj_path, "v2_source": season_csv}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync fantasy-projections-2 season board into output/model_v2/ "
            "(does not overwrite native v1 draft CSVs)"
        )
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=DEFAULT_V2_ROOT,
        help="Path to fantasy-projections-2 repo",
    )
    parser.add_argument(
        "--no-run-project",
        action="store_true",
        help="Use existing v2 outputs/season_projections_*.csv only",
    )
    parser.add_argument("--scoring", default="half_ppr")
    parser.add_argument("--train-end", type=int, default=None)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help=(
            "Rejected: preparing the draft UI from v2 would retangle boards. "
            "Serve v2 via fantasy-projections-2 on port 8765 instead."
        ),
    )
    args = parser.parse_args(argv)

    if args.prepare:
        raise SystemExit(
            "--prepare is disabled after model detangle. "
            "Native draft UI uses output/fantasy_points_*.csv (v1). "
            "For the team-first board, run projections.draft_assistant in "
            "fantasy-projections-2 (port 8765). Sync only writes output/model_v2/."
        )

    paths = sync_from_v2(
        season=args.season,
        v2_root=args.v2_root,
        run_project=not args.no_run_project,
        scoring=args.scoring,
        train_end=args.train_end,
    )
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
