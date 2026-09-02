#!/usr/bin/env python3
"""Exercise lineup, waiver, trade, and draft flows across all six fixture leagues.

Writes a JSON report suitable for post-fix verification. With
``APP_PROJECTION_SOURCE=sealed_release``, every league must cite a preseason
release id — never an active weekly DB run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPORT = ROOT / "output" / "decision_league_exercise.json"
DEFAULT_LIVE_REPORT = ROOT / "output" / "live_pg" / "decision_league_exercise.json"


def _redact_database_url(url: str | None) -> str | None:
    if not url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    host = rest.split("@", 1)[-1]
    return f"{scheme}://***@{host}"


def _configure(*, database_url: str | None = None, app_env: str = "test") -> None:
    os.environ["APP_ENV"] = app_env
    os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
    os.environ.setdefault("APP_PROJECTION_SOURCE", "sealed_release")
    os.environ.setdefault("WEEKLY_RND_ENABLED", "false")
    if database_url:
        os.environ["DATABASE_URL"] = database_url
        os.environ.pop("TEST_DATABASE_URL", None)
    else:
        os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")
    from src.app.config import get_settings
    from src.app.persistence.database import reset_engine

    get_settings.cache_clear()
    reset_engine()


def _load_live_league_ids(config_path: Path) -> list[str]:
    from src.app.league.sleeper.owner_config import load_owner_config

    config = load_owner_config(config_path)
    return [entry.league_id for entry in config.leagues]


def _owner_roster_id(session, league_id: str, username: str) -> int:
    from src.app.league.sleeper.client import SleeperClient
    from src.app.league.sleeper.owner_config import load_owner_config
    from src.app.persistence.models import LeagueMember

    config = load_owner_config(Path("config/sleeper_owner.json"))
    user = SleeperClient(use_fixtures=False).get_user(username)
    user_id = str(user.get("user_id") or "")
    member = (
        session.query(LeagueMember)
        .filter(LeagueMember.league_id == league_id, LeagueMember.user_id == user_id)
        .one_or_none()
    )
    if member is not None:
        return int(member.roster_id)
    roster = (
        session.query(LeagueMember)
        .filter(LeagueMember.league_id == league_id)
        .order_by(LeagueMember.roster_id.asc())
        .first()
    )
    if roster is None:
        raise RuntimeError(f"no_league_members:{league_id}")
    return int(roster.roster_id)


def exercise_league(
    session,
    league_id: str,
    *,
    week: int = 1,
    user_roster_id: int | None = None,
) -> dict:
    from src.app.decisions.draft_board import DraftBoardService
    from src.app.decisions.services import LeagueContextError, LineupService, WaiverService
    from src.app.decisions.trades import RedraftPickNotTradeable, TradeSide
    from src.app.decisions.services import TradeService
    from src.app.persistence.models import League, RosterSnapshot

    league = session.query(League).filter(League.league_id == league_id).one()
    roster_rows = (
        session.query(RosterSnapshot)
        .filter(RosterSnapshot.league_id == league_id, RosterSnapshot.week == week)
        .all()
    )
    rostered = {pid for row in roster_rows for pid in (row.players or []) if pid}
    league_status = str((league.raw_json or {}).get("status") or "").lower()
    pre_draft = league_status in {"pre_draft", "drafting"} or not rostered

    result: dict = {
        "league_id": league_id,
        "display_name": league.name,
        "league_type": league.league_type,
        "league_status": league_status or None,
        "ok": True,
        "errors": [],
        "checks": {},
    }
    if pre_draft:
        result["checks"]["skipped"] = "pre_draft_or_empty_roster"
        return result
    started = time.time()

    try:
        current = LineupService(session).recommend(league_id, week, opponent_mode="current")
        optimized = LineupService(session).recommend(league_id, week, opponent_mode="optimized")
        run_id = str(current.get("projection_run_id") or "")
        result["checks"]["projection_run_id"] = run_id
        result["checks"]["sealed_source"] = run_id.startswith("preseason-")
        result["checks"]["not_weekly_db_run"] = not run_id.startswith("weekly-")
        result["checks"]["starter_count"] = len(current.get("starters") or [])
        result["checks"]["scoring_fidelity"] = current.get("scoring_fidelity")
        result["checks"]["matchup_win_probability_available"] = current.get(
            "matchup_win_probability_available"
        )
        result["checks"]["win_probability"] = current.get("win_probability")
        result["checks"]["opponent_modes_distinct"] = (
            current.get("opponent_lineup_source") != optimized.get("opponent_lineup_source")
            or current.get("opponent_starters") != optimized.get("opponent_starters")
        )
        if not result["checks"]["sealed_source"]:
            result["errors"].append(f"projection_source_not_sealed:{run_id}")
    except LeagueContextError as exc:
        result["errors"].append(f"lineup:{exc}")
        result["ok"] = False

    try:
        waivers = WaiverService(session).recommend(
            league_id, week, user_roster_id=user_roster_id or 1
        )
        adds = waivers.get("adds") or waivers.get("recommendations") or []
        add_ids = {str(row.get("player_id")) for row in adds if row.get("player_id")}
        overlap = sorted(add_ids & rostered)
        result["checks"]["waiver_add_count"] = len(adds)
        result["checks"]["waiver_rostered_overlap"] = overlap
        if overlap:
            result["errors"].append(f"waiver_recommends_rostered_players:{overlap[:5]}")
            result["ok"] = False
    except LeagueContextError as exc:
        result["errors"].append(f"waivers:{exc}")
        result["ok"] = False

    try:
        user_roster = next((r for r in roster_rows if r.roster_id == 1), None)
        opponent = next((r for r in roster_rows if r.roster_id != 1), None)
        if user_roster and opponent and user_roster.players and opponent.players:
            trade = TradeService(session).evaluate(
                league_id,
                TradeSide(roster_id=1, player_ids=[str(user_roster.players[0])]),
                TradeSide(roster_id=opponent.roster_id, player_ids=[str(opponent.players[0])]),
                horizon="ros",
            )
            result["checks"]["trade_fair"] = trade.fairness.get("fair")
        else:
            result["checks"]["trade_skipped"] = "missing_roster_players"
    except LeagueContextError as exc:
        result["errors"].append(f"trade:{exc}")
        result["ok"] = False
    except RedraftPickNotTradeable:
        result["checks"]["trade_skipped"] = "redraft_pick_not_tradeable"

    board = DraftBoardService().load_board(league.season, limit=5)
    entries = board.get("entries") or []
    result["checks"]["draft_board_top"] = [
        {"name": row.get("name"), "vorp": row.get("vorp"), "rank": row.get("rank")}
        for row in entries[:3]
    ]
    if entries and entries[0].get("vorp") == 0.0:
        result["errors"].append("draft_board_zero_vorp_ranked_first")
        result["ok"] = False

    result["elapsed_seconds"] = round(time.time() - started, 2)
    if result["errors"]:
        result["ok"] = False
    return result


def run_exercise(
    *,
    report_path: Path = DEFAULT_REPORT,
    database_url: str | None = None,
    owner_config: Path | None = None,
    week: int = 1,
) -> dict:
    live_mode = database_url is not None
    _configure(database_url=database_url, app_env="production" if live_mode else "test")
    from src.app.persistence.database import get_session, init_db

    init_db()
    league_ids: list[str]
    roster_by_league: dict[str, int] = {}
    if live_mode:
        config_path = owner_config or (ROOT / "config" / "sleeper_owner.json")
        from src.app.league.sleeper.owner_config import load_owner_config

        owner = load_owner_config(config_path)
        league_ids = [entry.league_id for entry in owner.leagues]
        with get_session() as session:
            for league_id in league_ids:
                roster_by_league[league_id] = _owner_roster_id(session, league_id, owner.username)
    else:
        from src.app.seed import seed_development_data

        with get_session() as session:
            seed = seed_development_data(session, email="owner@example.com")
            league_ids = list(seed["leagues"])

    report: dict = {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "live_postgresql" if live_mode else "fixture_seed",
        "database_url": _redact_database_url(database_url)
        if live_mode
        else os.environ.get("TEST_DATABASE_URL"),
        "projection_source": os.environ.get("APP_PROJECTION_SOURCE", "sealed_release"),
        "week": week,
        "league_count": len(league_ids),
        "leagues": [],
        "summary": {},
    }

    with get_session() as session:
        for league_id in league_ids:
            report["leagues"].append(
                exercise_league(
                    session,
                    league_id,
                    week=week,
                    user_roster_id=roster_by_league.get(league_id),
                )
            )

    exercised = [row for row in report["leagues"] if row.get("checks", {}).get("skipped") is None]
    ok_count = sum(1 for row in report["leagues"] if row.get("ok"))
    report["summary"] = {
        "passed": ok_count,
        "failed": len(report["leagues"]) - ok_count,
        "exercised": len(exercised),
        "skipped": len(report["leagues"]) - len(exercised),
        "all_passed": ok_count == len(report["leagues"]),
        "all_sealed_source": all(
            row.get("checks", {}).get("sealed_source") for row in exercised
        )
        if exercised
        else True,
    }
    report["finished_at"] = datetime.now(UTC).isoformat()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--database-url",
        default=None,
        help="Use live PostgreSQL instead of in-memory fixture seed",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "sleeper_owner.json",
        help="Owner league config for live mode",
    )
    parser.add_argument("--week", type=int, default=1)
    args = parser.parse_args()
    report_path = args.report or (
        DEFAULT_LIVE_REPORT if args.database_url else DEFAULT_REPORT
    )
    report = run_exercise(
        report_path=report_path,
        database_url=args.database_url,
        owner_config=args.config,
        week=args.week,
    )
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"].get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
