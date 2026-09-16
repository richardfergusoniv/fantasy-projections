"""Orchestrate Milestone 1 deterministic weekly schedule allocation.

Shadow/research only. Does not promote, reseal, train, or touch the PWA.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import REPO_ROOT
from src.paths import DB_PATH
from src.projection.weekly_latent.allocate import (
    AllocationTables,
    allocate_players,
    allocate_team_weeks,
    season_box_from_players,
)
from src.projection.weekly_latent.artifacts import write_shadow_outputs
from src.projection.weekly_latent.conservation import evaluate_conservation
from src.projection.weekly_latent.constants import (
    DEFAULT_OUTPUT_REL,
    DEFAULT_SCHEDULE_FIXTURE_REL,
    DEFAULT_SEALED_NAMESPACE,
    DEFAULT_SEALED_PROJECTIONS_REL,
    GAMES_PER_SEASON,
    LEAGUE_VALUE_ROLE,
    MARKET_SANITY_ROLE,
    MILESTONE,
    PRODUCTION_HASH_PATHS,
    SCHEMA_VERSION,
    SEASON_DEFAULT,
    VEGAS_ROLE,
)
from src.projection.weekly_latent.schedule import explode_team_weeks, load_schedule_csv
from src.projection.weekly_latent.season_board import (
    load_long_projections,
    player_season_table,
    role_shares,
    team_season_volume,
)


@dataclass
class Milestone1Run:
    tables: AllocationTables
    conservation: dict[str, Any]
    summary: dict[str, Any]
    output_paths: dict[str, str]


def hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_fingerprint(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    files = {}
    for rel in PRODUCTION_HASH_PATHS:
        path = root / rel
        files[rel.replace("\\", "/")] = {
            "exists": path.is_file(),
            "sha256": hash_file(path),
        }
    return files


def synthetic_dry_run_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tiny 2-team, 3-week board used when sealed artifacts are unavailable."""
    schedule = pd.DataFrame(
        {
            "game_id": ["S1", "S3"],
            "season": [SEASON_DEFAULT, SEASON_DEFAULT],
            "week": [1, 3],
            "gameday": ["2026-09-10", "2026-09-24"],
            "weekday": ["Thursday", "Thursday"],
            "gametime": ["20:20", "20:20"],
            "away_team": ["AAA", "BBB"],
            "home_team": ["BBB", "AAA"],
            "location": ["Home", "Home"],
            "away_rest": [7, 14],
            "home_rest": [7, 14],
            "div_game": [0, 0],
            "roof": ["outdoors", "dome"],
            "surface": ["grass", "turf"],
            "stadium_id": ["BBB00", "AAA00"],
            "stadium": ["BBB Park", "AAA Dome"],
        }
    )
    # Week 2 is a bye for both; explode_team_weeks fills REG_WEEKS 1–18, so
    # tests that need a 3-week grid should call allocate on a trimmed grid.
    long_board = pd.DataFrame(
        [
            {
                "player_id": "p-qb",
                "display_name": "Alpha QB",
                "team": "AAA",
                "position": "QB",
                "stat": stat,
                "pred_season": value,
                "team_pass_attempts_pg_pred": 34.0,
                "team_carries_pg_pred": 26.0,
                "team_passing_yards_pg_pred": 240.0,
                "team_rushing_yards_pg_pred": 110.0,
            }
            for stat, value in {
                "attempts": 500.0,
                "completions": 320.0,
                "passing_yards": 3800.0,
                "passing_tds": 28.0,
                "interceptions": 10.0,
                "carries": 40.0,
                "rushing_yards": 200.0,
                "rushing_tds": 3.0,
                "targets": 0.0,
                "receptions": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
            }.items()
        ]
        + [
            {
                "player_id": "p-rb",
                "display_name": "Alpha RB",
                "team": "AAA",
                "position": "RB",
                "stat": stat,
                "pred_season": value,
                "team_pass_attempts_pg_pred": 34.0,
                "team_carries_pg_pred": 26.0,
                "team_passing_yards_pg_pred": 240.0,
                "team_rushing_yards_pg_pred": 110.0,
            }
            for stat, value in {
                "attempts": 0.0,
                "completions": 0.0,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "carries": 250.0,
                "rushing_yards": 1100.0,
                "rushing_tds": 8.0,
                "targets": 50.0,
                "receptions": 40.0,
                "receiving_yards": 350.0,
                "receiving_tds": 2.0,
            }.items()
        ]
        + [
            {
                "player_id": "q-qb",
                "display_name": "Beta QB",
                "team": "BBB",
                "position": "QB",
                "stat": stat,
                "pred_season": value,
                "team_pass_attempts_pg_pred": 32.0,
                "team_carries_pg_pred": 28.0,
                "team_passing_yards_pg_pred": 220.0,
                "team_rushing_yards_pg_pred": 120.0,
            }
            for stat, value in {
                "attempts": 480.0,
                "completions": 300.0,
                "passing_yards": 3500.0,
                "passing_tds": 24.0,
                "interceptions": 12.0,
                "carries": 30.0,
                "rushing_yards": 150.0,
                "rushing_tds": 2.0,
                "targets": 0.0,
                "receptions": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
            }.items()
        ]
    )
    priors = pd.DataFrame(
        {
            "opponent": ["AAA", "BBB"],
            "pass_factor": [0.80, 1.20],
            "rush_factor": [1.10, 0.90],
        }
    )
    return schedule, long_board, priors


def _maybe_load_db_opponent_priors(db_path: Path) -> tuple[pd.DataFrame | None, str]:
    """Optional lagged opponent priors. Never reads same-week team_attempts."""
    if not db_path.is_file():
        return None, f"no projections.db at {db_path} (priors stay 1.0)"
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "select name from sqlite_master where type='table'"
                ).fetchall()
            }
            if "weekly" not in tables:
                return None, "projections.db has no weekly table; priors stay 1.0"
            # Prior-season (2025) team pass/rush allowed from opponents' offense.
            # This is a lagged season aggregate, not a same-week join.
            weekly = pd.read_sql_query(
                "select season, recent_team as offense, "
                "coalesce(attempts,0) as attempts, coalesce(carries,0) as carries "
                "from weekly where season = 2025",
                conn,
            )
        finally:
            conn.close()
        if weekly.empty or "offense" not in weekly.columns:
            return None, "weekly 2025 empty; priors stay 1.0"
        # Without opponent on weekly we cannot form true allowed-rates here.
        # Do not invent a same-week join. Record skip.
        return (
            None,
            "weekly table present but has no opponent column in this query; "
            "M1 keeps opponent factors at 1.0 rather than risk a same-week join",
        )
    except Exception as exc:
        return None, f"db opponent prior skipped: {exc}"


def run_milestone1(
    *,
    season: int = SEASON_DEFAULT,
    projections_path: str | Path | None = None,
    schedule_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    opponent_priors: pd.DataFrame | None = None,
    dry_run: bool = False,
    repo_root: str | Path | None = None,
) -> Milestone1Run:
    root = Path(repo_root or REPO_ROOT)
    before = production_fingerprint(root)
    db_note = f"FANTASY_PROJECTIONS_DB_PATH / default = {DB_PATH}"
    db_priors_note = ""

    if dry_run:
        schedule, long_board, priors = synthetic_dry_run_inputs()
        opponent_priors = priors if opponent_priors is None else opponent_priors
        source = "synthetic_dry_run"
        proj_used = None
        sched_used = "synthetic"
    else:
        proj_used = Path(projections_path or (root / DEFAULT_SEALED_PROJECTIONS_REL))
        sched_used = Path(schedule_path or (root / DEFAULT_SCHEDULE_FIXTURE_REL))
        if not proj_used.is_file():
            raise FileNotFoundError(
                f"sealed projections not found at {proj_used}. "
                "Pass --projections or run --dry-run. On Windows the usual "
                "board CSV is next to the DB under D:\\fantasy-projections-data "
                "or the repo path "
                "draft_assistant\\data\\releases\\v2_baseline_20260830\\projections_2026.csv"
            )
        if not sched_used.is_file():
            raise FileNotFoundError(
                f"schedule fixture not found at {sched_used}. "
                "Pass --schedule or use the committed "
                "src/projection/weekly_latent/fixtures/nfl_schedules_2026_reg.csv"
            )
        long_board = load_long_projections(proj_used)
        schedule = load_schedule_csv(sched_used)
        schedule = schedule[schedule["season"].eq(season)].copy()
        source = "sealed_board_plus_schedule_fixture"
        if opponent_priors is None:
            loaded, db_priors_note = _maybe_load_db_opponent_priors(Path(DB_PATH))
            opponent_priors = loaded

    team_weeks = explode_team_weeks(
        schedule, require_full_season=not dry_run
    )
    if dry_run:
        team_weeks = team_weeks[team_weeks["week"].isin([1, 2, 3])].copy()
    players = player_season_table(long_board)
    team_volume = team_season_volume(long_board)
    shares = role_shares(players, team_volume)
    allocated_teams = allocate_team_weeks(
        team_weeks, team_volume, opponent_priors=opponent_priors
    )
    player_weeks = allocate_players(allocated_teams, players, shares)
    players = season_box_from_players(players)
    conservation = evaluate_conservation(
        allocated_teams, team_volume, shares, player_weeks, players
    )
    after = production_fingerprint(root)
    drift = {
        rel: {"before": before[rel], "after": after[rel]}
        for rel in before
        if before[rel] != after[rel]
    }
    summary = {
        "milestone": MILESTONE,
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "dry_run": dry_run,
        "source": source,
        "projections_path": None if proj_used is None else str(proj_used).replace("\\", "/"),
        "schedule_path": None if sched_used is None else str(sched_used).replace("\\", "/"),
        "sealed_namespace_read": None if dry_run else DEFAULT_SEALED_NAMESPACE,
        "n_teams": int(allocated_teams["team"].nunique()),
        "n_team_weeks": int(len(allocated_teams)),
        "n_players": int(player_weeks["player_id"].nunique()) if len(player_weeks) else 0,
        "n_player_weeks": int(len(player_weeks)),
        "n_bye_team_weeks": int(allocated_teams["is_bye"].sum()),
        "conservation_passes": conservation["passes"],
        "failing_checks": conservation["failing_checks"],
        "share_overflow_team_stats": _overflow_counts(shares),
        "product_split": {
            "vegas": VEGAS_ROLE,
            "league_value": LEAGUE_VALUE_ROLE,
            "adp_and_season_vegas": MARKET_SANITY_ROLE,
        },
        "does_not": [
            "replace Vegas weekly props",
            "promote or reseal League Value / v2_baseline_20260830",
            "change freeze knobs or production defaults",
            "train a model",
            "use ADP or season-long Vegas as drivers",
            "join same-week team_attempts/team_carries from add_team_pass_rate",
            "wire into the PWA",
        ],
        "db": db_note,
        "db_opponent_priors": db_priors_note,
        "production_hash_drift": drift,
        "games_per_season": GAMES_PER_SEASON,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if drift:
        raise RuntimeError(
            "Milestone 1 is research/shadow only but production files changed: "
            + json.dumps(drift)
        )
    out_dir = Path(output_dir or (root / DEFAULT_OUTPUT_REL))
    paths = write_shadow_outputs(
        out_dir,
        team_weeks=allocated_teams,
        player_weeks=player_weeks,
        shares=shares,
        conservation=conservation,
        summary=summary,
    )
    return Milestone1Run(
        tables=AllocationTables(
            team_weeks=allocated_teams,
            player_weeks=player_weeks,
            shares=shares,
            players=players,
            team_volume=team_volume,
        ),
        conservation=conservation,
        summary=summary,
        output_paths=paths,
    )


def _overflow_counts(shares: pd.DataFrame) -> dict[str, int]:
    modes = shares.groupby(["team", "stat"])["share_mode"].first().reset_index()
    return {
        str(stat): int((grp["share_mode"] == "rescaled_overflow").sum())
        for stat, grp in modes.groupby("stat")
    }
