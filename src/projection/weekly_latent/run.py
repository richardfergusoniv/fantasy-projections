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
    allocate_players_m3,
    allocate_team_weeks,
    allocate_team_weeks_m2,
    season_box_from_players,
)
from src.projection.weekly_latent.artifacts import write_shadow_outputs
from src.projection.weekly_latent.backtest import run_m2_backtests
from src.projection.weekly_latent.backtest_m3 import run_m3_backtests
from src.projection.weekly_latent.conservation import evaluate_conservation, evaluate_m2, evaluate_m3
from src.projection.weekly_latent.constants import (
    DEFAULT_OPP_EPA_PRIOR_REL,
    DEFAULT_OUTPUT_REL,
    DEFAULT_OUTPUT_REL_M2,
    DEFAULT_OUTPUT_REL_M3,
    DEFAULT_SCHEDULE_FIXTURE_REL,
    DEFAULT_SEALED_NAMESPACE,
    DEFAULT_SEALED_PROJECTIONS_REL,
    GAMES_PER_SEASON,
    LEAGUE_VALUE_ROLE,
    M1_AVAILABLE_AT,
    MARKET_SANITY_ROLE,
    MILESTONE,
    MILESTONE_M2,
    MILESTONE_M3,
    PRODUCTION_HASH_PATHS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_M2,
    SCHEMA_VERSION_M3,
    SEASON_DEFAULT,
    VEGAS_ROLE,
)
from src.projection.weekly_latent.priors import load_m2_opponent_priors
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


@dataclass
class Milestone2Run:
    tables: AllocationTables
    m1_tables: AllocationTables
    conservation: dict[str, Any]
    summary: dict[str, Any]
    backtest: dict[str, Any]
    output_paths: dict[str, str]


@dataclass
class Milestone3Run:
    tables: AllocationTables
    m1_tables: AllocationTables
    conservation: dict[str, Any]
    summary: dict[str, Any]
    backtest: dict[str, Any]
    vegas_compare: dict[str, Any]
    market_sanity: dict[str, Any]
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


def _season_mass_delta(
    m2_weeks: pd.DataFrame,
    m1_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
) -> dict[str, Any]:
    from src.projection.weekly_latent.constants import TEAM_VOLUME_PG_COLUMNS

    m2_sum = m2_weeks.groupby("team")[list(TEAM_VOLUME_PG_COLUMNS)].sum()
    m1_sum = m1_weeks.groupby("team")[list(TEAM_VOLUME_PG_COLUMNS)].sum()
    sealed = team_volume.set_index("team")[list(TEAM_VOLUME_PG_COLUMNS)]
    out: dict[str, Any] = {}
    for name in TEAM_VOLUME_PG_COLUMNS:
        vs_sealed = (m2_sum[name] - sealed[name]).abs()
        vs_m1 = (m2_sum[name] - m1_sum[name]).abs()
        ratio = m2_sum[name] / sealed[name].replace(0.0, pd.NA)
        out[name] = {
            "max_abs_vs_sealed": float(vs_sealed.max()),
            "max_abs_vs_m1": float(vs_m1.max()),
            "ratio_vs_sealed_minmax": [float(ratio.min()), float(ratio.max())],
        }
    return out


def run_milestone2(
    *,
    season: int = SEASON_DEFAULT,
    projections_path: str | Path | None = None,
    schedule_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    opponent_priors: pd.DataFrame | None = None,
    opp_epa_priors_path: str | Path | None = None,
    dry_run: bool = False,
    repo_root: str | Path | None = None,
    run_backtest: bool = True,
) -> Milestone2Run:
    """Shadow Milestone 2: environment may move season mass. Not a promotion."""
    root = Path(repo_root or REPO_ROOT)
    before = production_fingerprint(root)
    db_note = f"FANTASY_PROJECTIONS_DB_PATH / default = {DB_PATH}"

    if dry_run:
        schedule, long_board, priors = synthetic_dry_run_inputs()
        if opponent_priors is None:
            opponent_priors = priors.copy()
            if "available_at" not in opponent_priors.columns:
                opponent_priors["available_at"] = "2026-01-05T00:00:00+00:00"
        source = "synthetic_dry_run"
        proj_used = None
        sched_used = "synthetic"
        prior_note = "synthetic dry-run opponent factors"
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
        source = "sealed_board_plus_schedule_fixture_plus_lagged_epa"
        if opponent_priors is None:
            opponent_priors, prior_note = load_m2_opponent_priors(
                repo_root=root,
                db_path=Path(DB_PATH),
                fixture_path=opp_epa_priors_path or (root / DEFAULT_OPP_EPA_PRIOR_REL),
                season=season,
            )
        else:
            prior_note = "caller-supplied opponent priors"

    team_weeks = explode_team_weeks(schedule, require_full_season=not dry_run)
    if dry_run:
        team_weeks = team_weeks[team_weeks["week"].isin([1, 2, 3])].copy()
    players = player_season_table(long_board)
    team_volume = team_season_volume(long_board)
    shares = role_shares(players, team_volume)
    m1_teams = allocate_team_weeks(
        team_weeks, team_volume, opponent_priors=opponent_priors
    )
    m2_teams = allocate_team_weeks_m2(
        team_weeks,
        team_volume,
        opponent_priors=opponent_priors,
        board_available_at=M1_AVAILABLE_AT,
    )
    player_weeks = allocate_players(m2_teams, players, shares)
    players = season_box_from_players(players)
    conservation = evaluate_m2(
        m2_teams,
        team_volume,
        shares,
        player_weeks,
        players,
        m1_team_weeks=m1_teams,
    )
    m1_conservation = evaluate_conservation(
        m1_teams, team_volume, shares, allocate_players(m1_teams, players, shares), players
    )
    backtest = run_m2_backtests() if run_backtest else {"skipped": True}
    after = production_fingerprint(root)
    drift = {
        rel: {"before": before[rel], "after": after[rel]}
        for rel in before
        if before[rel] != after[rel]
    }
    summary = {
        "milestone": MILESTONE_M2,
        "schema_version": SCHEMA_VERSION_M2,
        "season": season,
        "dry_run": dry_run,
        "source": source,
        "projections_path": None if proj_used is None else str(proj_used).replace("\\", "/"),
        "schedule_path": None if sched_used is None else str(sched_used).replace("\\", "/"),
        "sealed_namespace_read": None if dry_run else DEFAULT_SEALED_NAMESPACE,
        "n_teams": int(m2_teams["team"].nunique()),
        "n_team_weeks": int(len(m2_teams)),
        "n_players": int(player_weeks["player_id"].nunique()) if len(player_weeks) else 0,
        "n_player_weeks": int(len(player_weeks)),
        "n_bye_team_weeks": int(m2_teams["is_bye"].sum()),
        "conservation_passes": conservation["passes"],
        "failing_checks": conservation["failing_checks"],
        "m1_comparison_conservation_passes": m1_conservation["passes"],
        "season_mass_delta_vs_m1_and_sealed": _season_mass_delta(
            m2_teams, m1_teams, team_volume
        ),
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
            "train a hierarchical Monte Carlo / M3 availability+conversion layer",
            "use ADP or season-long Vegas as drivers",
            "join same-week team_attempts/team_carries/team_targets/team_air_yards",
            "wire into the PWA",
            "flip APP_PROJECTION_SOURCE",
        ],
        "identity": (
            "V_w = A_w * (V_sealed / n_active) * m_HA * m_opp * m_env "
            "(no renormalize). Shares + other = 1. Bye = 0."
        ),
        "db": db_note,
        "opponent_priors": prior_note,
        "production_hash_drift": drift,
        "games_per_season": GAMES_PER_SEASON,
        "m1_schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_verdict": "not promoting",
        "still_shadow": True,
        "backtest_passes": backtest.get("passes"),
    }
    if drift:
        raise RuntimeError(
            "Milestone 2 is research/shadow only but production files changed: "
            + json.dumps(drift)
        )
    out_dir = Path(output_dir or (root / DEFAULT_OUTPUT_REL_M2))
    paths = write_shadow_outputs(
        out_dir,
        team_weeks=m2_teams,
        player_weeks=player_weeks,
        shares=shares,
        conservation=conservation,
        summary=summary,
        readme_title="Shadow weekly schedule allocation (Milestone 2)",
        extra_json={
            "backtest_synthetic.json": backtest.get("synthetic", backtest),
            "backtest_historical.json": backtest.get("historical", {}),
        },
        cli_name="scripts/run_weekly_schedule_m2.py",
    )
    return Milestone2Run(
        tables=AllocationTables(
            team_weeks=m2_teams,
            player_weeks=player_weeks,
            shares=shares,
            players=players,
            team_volume=team_volume,
        ),
        m1_tables=AllocationTables(
            team_weeks=m1_teams,
            player_weeks=allocate_players(m1_teams, players, shares),
            shares=shares,
            players=players,
            team_volume=team_volume,
        ),
        conservation=conservation,
        summary=summary,
        backtest=backtest,
        output_paths=paths,
    )


def run_milestone3(
    *,
    season: int = SEASON_DEFAULT,
    projections_path: str | Path | None = None,
    schedule_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    opponent_priors: pd.DataFrame | None = None,
    opp_epa_priors_path: str | Path | None = None,
    availability: pd.DataFrame | None = None,
    dry_run: bool = False,
    repo_root: str | Path | None = None,
    run_backtest: bool = True,
) -> Milestone3Run:
    """Shadow Milestone 3: week-varying availability + conversions. Not a promotion."""
    from src.projection.weekly_latent.vegas_hook import (
        compare_m3_to_vegas_props,
        export_role2_board,
        market_sanity_bands,
    )

    root = Path(repo_root or REPO_ROOT)
    before = production_fingerprint(root)
    db_note = f"FANTASY_PROJECTIONS_DB_PATH / default = {DB_PATH}"

    if dry_run:
        schedule, long_board, priors = synthetic_dry_run_inputs()
        if opponent_priors is None:
            opponent_priors = priors.copy()
            if "available_at" not in opponent_priors.columns:
                opponent_priors["available_at"] = "2026-01-05T00:00:00+00:00"
        source = "synthetic_dry_run"
        proj_used = None
        sched_used = "synthetic"
        prior_note = "synthetic dry-run opponent factors"
    else:
        proj_used = Path(projections_path or (root / DEFAULT_SEALED_PROJECTIONS_REL))
        sched_used = Path(schedule_path or (root / DEFAULT_SCHEDULE_FIXTURE_REL))
        if not proj_used.is_file():
            raise FileNotFoundError(
                f"sealed projections not found at {proj_used}. "
                "Pass --projections or run --dry-run."
            )
        if not sched_used.is_file():
            raise FileNotFoundError(f"schedule fixture not found at {sched_used}.")
        long_board = load_long_projections(proj_used)
        schedule = load_schedule_csv(sched_used)
        schedule = schedule[schedule["season"].eq(season)].copy()
        source = "sealed_board_plus_schedule_fixture_plus_lagged_epa_m3"
        if opponent_priors is None:
            opponent_priors, prior_note = load_m2_opponent_priors(
                repo_root=root,
                db_path=Path(DB_PATH),
                fixture_path=opp_epa_priors_path or (root / DEFAULT_OPP_EPA_PRIOR_REL),
                season=season,
            )
        else:
            prior_note = "caller-supplied opponent priors"

    team_weeks = explode_team_weeks(schedule, require_full_season=not dry_run)
    if dry_run:
        team_weeks = team_weeks[team_weeks["week"].isin([1, 2, 3])].copy()
    players = player_season_table(long_board)
    team_volume = team_season_volume(long_board)
    shares = role_shares(players, team_volume)
    m1_teams = allocate_team_weeks(
        team_weeks, team_volume, opponent_priors=opponent_priors
    )
    m2_teams = allocate_team_weeks_m2(
        team_weeks,
        team_volume,
        opponent_priors=opponent_priors,
        board_available_at=M1_AVAILABLE_AT,
    )
    player_weeks = allocate_players_m3(
        m2_teams, players, shares, availability=availability
    )
    players = season_box_from_players(players)
    conservation = evaluate_m3(
        m2_teams,
        team_volume,
        shares,
        player_weeks,
        players,
        m1_team_weeks=m1_teams,
    )
    backtest = run_m3_backtests() if run_backtest else {"skipped": True}
    role2_board = export_role2_board(player_weeks)
    vegas_compare = compare_m3_to_vegas_props(
        board=role2_board,
        dry_run=True,
        repo_root=root,
    )
    sanity = market_sanity_bands()
    after = production_fingerprint(root)
    drift = {
        rel: {"before": before[rel], "after": after[rel]}
        for rel in before
        if before[rel] != after[rel]
    }
    summary = {
        "milestone": MILESTONE_M3,
        "schema_version": SCHEMA_VERSION_M3,
        "season": season,
        "dry_run": dry_run,
        "source": source,
        "projections_path": None if proj_used is None else str(proj_used).replace("\\", "/"),
        "schedule_path": None if sched_used is None else str(sched_used).replace("\\", "/"),
        "sealed_namespace_read": None if dry_run else DEFAULT_SEALED_NAMESPACE,
        "n_teams": int(m2_teams["team"].nunique()),
        "n_team_weeks": int(len(m2_teams)),
        "n_players": int(player_weeks["player_id"].nunique()) if len(player_weeks) else 0,
        "n_player_weeks": int(len(player_weeks)),
        "n_bye_team_weeks": int(m2_teams["is_bye"].sum()),
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
            "use ADP or season-long Vegas as drivers",
            "join same-week team_attempts/team_carries/team_targets/team_air_yards",
            "wire into the PWA",
            "flip APP_PROJECTION_SOURCE",
            "implement Role 3 market blend",
        ],
        "identity": (
            "A_i,w week-varying (bye=0); opportunity = A * share * V_m2; "
            "box = opportunity × season rates × opponent-only conversion "
            "multipliers (clipped). Season = sum of weeks."
        ),
        "db": db_note,
        "opponent_priors": prior_note,
        "production_hash_drift": drift,
        "games_per_season": GAMES_PER_SEASON,
        "m1_schema_version": SCHEMA_VERSION,
        "m2_schema_version": SCHEMA_VERSION_M2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_verdict": "not promoting",
        "still_shadow": True,
        "backtest_passes": backtest.get("passes"),
        "vegas_compare_role": vegas_compare.get("role"),
        "market_sanity_status": sanity.get("status"),
    }
    if drift:
        raise RuntimeError(
            "Milestone 3 is research/shadow only but production files changed: "
            + json.dumps(drift)
        )
    out_dir = Path(output_dir or (root / DEFAULT_OUTPUT_REL_M3))
    paths = write_shadow_outputs(
        out_dir,
        team_weeks=m2_teams,
        player_weeks=player_weeks,
        shares=shares,
        conservation=conservation,
        summary=summary,
        readme_title="Shadow weekly schedule allocation (Milestone 3)",
        extra_json={
            "backtest_synthetic.json": backtest.get("synthetic", backtest),
            "backtest_historical.json": backtest.get("historical", {}),
            "vegas_props_compare.json": vegas_compare,
            "market_sanity.json": sanity,
        },
        extra_csv={"shadow_board_role2.csv": role2_board},
        cli_name="scripts/run_weekly_schedule_m3.py",
    )
    return Milestone3Run(
        tables=AllocationTables(
            team_weeks=m2_teams,
            player_weeks=player_weeks,
            shares=shares,
            players=players,
            team_volume=team_volume,
        ),
        m1_tables=AllocationTables(
            team_weeks=m1_teams,
            player_weeks=allocate_players(m1_teams, players, shares),
            shares=shares,
            players=players,
            team_volume=team_volume,
        ),
        conservation=conservation,
        summary=summary,
        backtest=backtest,
        vegas_compare=vegas_compare,
        market_sanity=sanity,
        output_paths=paths,
    )

