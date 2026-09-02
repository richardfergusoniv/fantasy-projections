"""Isolated live Sleeper shadow sync — read-only, opt-in, non-promoting.

Runs a complete application sync against owner-configured leagues in a
throwaway database and artifact prefix. Never touches production pointers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.app.decisions.draft_board import DraftBoardService
from src.app.decisions.dynasty import DynastyService
from src.app.decisions.services import LeagueContextError, LineupService, TradeService, WaiverService
from src.app.decisions.trades import TradeSide
from src.app.jobs.handlers import resolve_season_week
from src.app.league.sleeper.client import READ_ONLY_METHOD, SleeperClient
from src.app.league.sleeper.identity_gate import IdentityReconciliationGate
from src.app.league.sleeper.owner_config import (
    LeagueSelectionError,
    SleeperOwnerConfig,
    load_owner_config,
    validate_league_selection,
)
from src.app.league.sleeper.sync import SleeperSyncService
from src.app.persistence.models import (
    ActiveProjectionPointer,
    LeagueDraftRule,
    LeagueMember,
    LeagueRuleSnapshot,
    LeagueTransaction,
    RosterSnapshot,
    SourceSnapshot,
    TradedPick,
)
from src.app.projections.weekly_v2_bridge import weekly_v2_readiness
from src.app.releases.bridge import ReleaseBridge
from src.app.releases.gates import GateResult
from src.app.releases.publication import Candidate, CandidateRow, active_pointer, publish
from src.app.scoring.compiler import compile_sleeper_scoring, require_publishable

OPT_IN_ENV = "LIVE_SLEEPER_SHADOW"
DEFAULT_SHADOW_DB = "output/live_shadow/shadow_app.db"
DEFAULT_ARTIFACT_ROOT = "output/live_shadow/artifacts"
DEFAULT_REPORT_JSON = "output/live_shadow/sleeper_sync_report.json"
PRODUCTION_DB_MARKERS = ("local_app.db", "fantasy_app", "postgresql")


@dataclass
class ShadowSyncOptions:
    config_path: Path
    season: int | None = None
    database_url: str = f"sqlite+pysqlite:///{DEFAULT_SHADOW_DB}"
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_path: Path = Path(DEFAULT_REPORT_JSON)
    allow_production_database: bool = False
    inject_failure: bool = False
    skip_second_run: bool = False


@dataclass
class ShadowSyncReport:
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    exit_code: int = 0
    environment: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    league_selection: dict[str, Any] = field(default_factory=dict)
    draft_rules: dict[str, str] = field(default_factory=dict)
    scoring: list[dict[str, Any]] = field(default_factory=list)
    identity: dict[str, Any] = field(default_factory=dict)
    completeness: dict[str, Any] = field(default_factory=dict)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    idempotency: dict[str, Any] = field(default_factory=dict)
    failure_injection: dict[str, Any] = field(default_factory=dict)
    projections: dict[str, Any] = field(default_factory=dict)
    league_scoring_shadow: dict[str, Any] = field(default_factory=dict)
    defects: list[dict[str, str]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    go_no_go: str = "pending"
    commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "exit_code": self.exit_code,
            "environment": self.environment,
            "safety": self.safety,
            "league_selection": self.league_selection,
            "draft_rules": self.draft_rules,
            "scoring": self.scoring,
            "identity": self.identity,
            "completeness": self.completeness,
            "recommendations": self.recommendations,
            "idempotency": self.idempotency,
            "failure_injection": self.failure_injection,
            "projections": self.projections,
            "league_scoring_shadow": self.league_scoring_shadow,
            "defects": self.defects,
            "blockers": self.blockers,
            "go_no_go": self.go_no_go,
            "commands": self.commands,
        }


def assert_read_only_client() -> None:
    if READ_ONLY_METHOD != "GET":
        raise RuntimeError("Sleeper client is not GET-only")
    write_methods = ("post", "put", "patch", "delete")
    offenders = [name for name in dir(SleeperClient) if name.startswith(write_methods)]
    if offenders:
        raise RuntimeError(f"Sleeper client exposes write methods: {offenders}")


def assert_opt_in() -> None:
    if os.environ.get(OPT_IN_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError(f"refusing shadow sync: set {OPT_IN_ENV}=1 to opt in")


def assert_database_target(database_url: str, *, allow_production: bool) -> None:
    lowered = database_url.lower()
    looks_production = any(marker in lowered for marker in PRODUCTION_DB_MARKERS)
    if looks_production and not allow_production:
        raise RuntimeError(
            "refusing to run against a production-looking DATABASE_URL; "
            "pass --allow-production-database only if you intend to overwrite it"
        )


def _configure_shadow_environment(options: ShadowSyncOptions) -> None:
    os.environ["APP_ENV"] = "development"
    os.environ["DATABASE_URL"] = options.database_url
    os.environ["ARTIFACT_LOCAL_ROOT"] = options.artifact_root
    os.environ["SLEEPER_USE_FIXTURES"] = "false"
    os.environ.setdefault(OPT_IN_ENV, "1")
    from src.app.config import get_settings
    from src.app.persistence.database import reset_engine

    get_settings.cache_clear()
    reset_engine()


def _sqlite_db_path(database_url: str) -> Path | None:
    if not database_url.lower().startswith("sqlite"):
        return None
    db_path = database_url.split("///")[-1]
    if not db_path or db_path.startswith(":"):
        return None
    return Path(db_path)


def _migrate_database(database_url: str) -> None:
    sqlite_path = _sqlite_db_path(database_url)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        from src.app.persistence.database import init_db

        init_db()


def _count_rows(session: Session, model) -> int:
    return session.query(model).count()


def _weekly_league_scoring_shadow(
    session: Session,
    config: SleeperOwnerConfig,
    *,
    season: int,
    week: int,
) -> dict[str, Any]:
    """Score trained weekly component stats under each configured league contract."""
    from src.app.persistence.models import League
    from src.app.projections.weekly_league_scoring import (
        LeagueScoringContract,
        score_weekly_frame_for_leagues,
    )
    from src.projection.weekly.config.paths import OUTPUTS_DIR

    parquet = OUTPUTS_DIR / f"season={season}" / f"week={week:02d}" / "weekly_projections.parquet"
    if not parquet.exists():
        return {"status": "skipped", "reason": "weekly_projections_missing", "path": str(parquet)}

    import polars as pl

    contracts: list[LeagueScoringContract] = []
    for entry in config.leagues:
        league = session.query(League).filter(League.league_id == entry.league_id).one_or_none()
        if league is None or not league.raw_json:
            return {
                "status": "failed",
                "reason": f"league_missing:{entry.league_id}",
            }
        contracts.append(
            LeagueScoringContract.from_league_json(
                league_id=entry.league_id,
                display_name=entry.display_name,
                raw_json=league.raw_json,
            )
        )
    artifact = score_weekly_frame_for_leagues(pl.read_parquet(parquet), contracts)
    artifact["status"] = "ok"
    artifact["source_parquet"] = str(parquet)
    return artifact


def _league_scoring_summary(session: Session, league_id: str, display_name: str) -> dict[str, Any]:
    from src.app.persistence.models import League

    league = session.query(League).filter(League.league_id == league_id).one_or_none()
    snapshot = (
        session.query(LeagueRuleSnapshot)
        .filter(LeagueRuleSnapshot.league_id == league_id)
        .order_by(LeagueRuleSnapshot.fetched_at.desc())
        .first()
    )
    raw = (league.raw_json or {}) if league else {}
    scoring = raw.get("scoring_settings") or {}
    roster_positions = raw.get("roster_positions") or []
    contract = compile_sleeper_scoring(scoring, roster_positions)
    nonzero_keys = sorted(key for key, value in scoring.items() if value not in (0, 0.0, None, ""))
    waiver_type = (raw.get("settings") or {}).get("waiver_type")
    waiver_budget = (raw.get("settings") or {}).get("waiver_budget")
    return {
        "league_id": league_id,
        "display_name": display_name,
        "season": league.season if league else None,
        "status": league.status if league else None,
        "league_type": league.league_type if league else None,
        "team_count": len(raw.get("roster_positions") or []),
        "roster_slot_count": len(roster_positions),
        "roster_positions": roster_positions,
        "nonzero_scoring_keys": nonzero_keys,
        "contract_hash": contract.contract_hash,
        "unsupported_keys": list(contract.unsupported_keys),
        "unsupported_slots": list(contract.unsupported_slots),
        "waiver_type": waiver_type,
        "waiver_budget": waiver_budget,
        "playoff_week_start": (raw.get("settings") or {}).get("playoff_week_start"),
        "publishable": not contract.blocks_publication,
    }


def _find_owner_roster(session: Session, league_id: str, sleeper_user_id: str) -> int | None:
    member = (
        session.query(LeagueMember)
        .filter(LeagueMember.league_id == league_id, LeagueMember.user_id == str(sleeper_user_id))
        .one_or_none()
    )
    return member.roster_id if member is not None else None


def _run_recommendation_smoke(
    session: Session,
    *,
    league_id: str,
    display_name: str,
    roster_id: int,
    week: int,
    league_type: str,
    league_status: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "league_id": league_id,
        "display_name": display_name,
        "owner_roster_id": roster_id,
        "league_status": league_status,
        "checks": {},
        "errors": [],
    }
    pre_draft = (league_status or "").lower() in {"pre_draft", "drafting"}
    rosters = (
        session.query(RosterSnapshot)
        .filter(RosterSnapshot.league_id == league_id, RosterSnapshot.week == week)
        .all()
    )
    user_roster = next((r for r in rosters if r.roster_id == roster_id), None)
    if pre_draft and (user_roster is None or not (user_roster.players or [])):
        result["checks"]["skipped"] = "pre_draft_empty_roster"
        result["ok"] = True
        return result
    lineup = LineupService(session)
    waivers = WaiverService(session)
    trades = TradeService(session)
    dynasty = DynastyService(session)
    draft_board = DraftBoardService()

    try:
        current = lineup.recommend(
            league_id, week, opponent_mode="current", user_roster_id=roster_id
        )
        optimized = lineup.recommend(
            league_id, week, opponent_mode="optimized", user_roster_id=roster_id
        )
        result["checks"]["lineup_current"] = "ok"
        result["checks"]["lineup_optimized"] = "ok"
        result["checks"]["matchup_modes_distinct"] = (
            current.get("matchup") != optimized.get("matchup")
        )
    except LeagueContextError as exc:
        result["errors"].append(f"lineup:{exc}")

    try:
        waiver_payload = waivers.recommend(league_id, week, user_roster_id=roster_id)
        result["checks"]["waivers"] = len(waiver_payload.get("recommendations", []))
    except LeagueContextError as exc:
        result["errors"].append(f"waivers:{exc}")

    try:
        rosters = (
            session.query(RosterSnapshot)
            .filter(RosterSnapshot.league_id == league_id, RosterSnapshot.week == week)
            .all()
        )
        opponent = next((r for r in rosters if r.roster_id != roster_id), None)
        user_roster = next((r for r in rosters if r.roster_id == roster_id), None)
        if opponent and user_roster and user_roster.players and opponent.players:
            side_a = TradeSide(
                roster_id=roster_id,
                player_ids=[str(user_roster.players[0])],
                pick_assets=[],
            )
            side_b = TradeSide(
                roster_id=opponent.roster_id,
                player_ids=[str(opponent.players[0])],
                pick_assets=[],
            )
            trade_result = trades.evaluate(league_id, side_a, side_b, horizon="ros")
            result["checks"]["trade_evaluate"] = trade_result.fairness.get("label", "ok")
        else:
            result["errors"].append("trade:insufficient_roster_players")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"trade:{type(exc).__name__}")

    if league_type == "dynasty":
        try:
            state = dynasty.evaluate_roster(league_id, roster_id, week=week)
            pick = dynasty.project_rookie_pick_for_roster(league_id, roster_id, week=week)
            result["checks"]["dynasty_state"] = state.label
            result["checks"]["rookie_pick_rule"] = pick.get("rule")
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"dynasty:{type(exc).__name__}")
    else:
        try:
            trades.evaluate(
                league_id,
                TradeSide(roster_id=roster_id, player_ids=[], pick_assets=[{"season": 2027, "round": 1}]),
                TradeSide(roster_id=2, player_ids=[], pick_assets=[]),
                horizon="ros",
            )
            result["errors"].append("redraft_pick_trade_should_fail")
        except Exception:
            result["checks"]["redraft_rejects_future_pick"] = True

    try:
        board = draft_board.load_board(season=2026)
        result["checks"]["draft_board_players"] = len(board.get("entries", []))
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"draft_board:{type(exc).__name__}")

    result["ok"] = not result["errors"]
    return result


def _inject_publication_failure(session: Session, *, season: int, week: int) -> dict[str, Any]:
    pointer_before = active_pointer(session, mode="weekly", season=season, week=week)
    before_run_id = pointer_before.run_id if pointer_before else None
    candidate = Candidate(
        mode="weekly",
        season=season,
        week=week,
        run_id="shadow-failure-injection",
        model_version="shadow-injected-failure",
        input_hash="deadbeef",
        manifest_uri="shadow://injected-failure",
        artifact_mode="fixture",
        partition_mode="weekly",
        rows=(
            CandidateRow(
                player_id="00-test",
                team="TST",
                opponent=None,
                availability_probability=1.0,
                mean_json={"points": 1.0},
                quantiles_json={"p50": 1.0},
            ),
        ),
        metadata={"derivation": "shadow_failure_injection"},
    )
    result = publish(
        session,
        candidate,
        gates={"promotion": GateResult(passed=False, failures=["injected_shadow_failure"])},
        register_partitions=False,
        validate_partitions=False,
    )
    pointer_after = active_pointer(session, mode="weekly", season=season, week=week)
    after_run_id = pointer_after.run_id if pointer_after else None
    return {
        "pointer_before": before_run_id,
        "pointer_after": after_run_id,
        "promoted": result.promoted,
        "reason": result.reason,
        "pointer_unchanged": before_run_id == after_run_id,
        "passed": not result.promoted and before_run_id == after_run_id,
    }


class SleeperShadowSyncRunner:
    def __init__(self, options: ShadowSyncOptions) -> None:
        self.options = options
        self.report = ShadowSyncReport(started_at=datetime.now(UTC).isoformat())

    def run(self) -> int:
        try:
            assert_opt_in()
            assert_read_only_client()
            assert_database_target(
                self.options.database_url,
                allow_production=self.options.allow_production_database,
            )
            config = load_owner_config(self.options.config_path)
            season = self.options.season or config.season
            _configure_shadow_environment(self.options)
            from src.app.config import get_settings

            settings = get_settings()
            self.report.environment = {
                "database_url": self.options.database_url,
                "artifact_root": self.options.artifact_root,
                "sleeper_mode": settings.sleeper_mode,
                "config_path": str(self.options.config_path),
                "season": season,
            }
            self.report.safety = {
                "opt_in_env": OPT_IN_ENV,
                "read_only_method": READ_ONLY_METHOD,
                "auto_publish_allowed": weekly_v2_readiness(season, 1).auto_publish_allowed,
            }
            self.report.commands.append(
                "uv run python -m src.app.cli sleeper-shadow-sync "
                f"--season {season} --config {self.options.config_path}"
            )

            sqlite_path = _sqlite_db_path(self.options.database_url)
            if sqlite_path is not None:
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            Path(self.options.artifact_root).mkdir(parents=True, exist_ok=True)
            self.options.report_path.parent.mkdir(parents=True, exist_ok=True)

            _migrate_database(self.options.database_url)
            from src.app.persistence.database import get_session

            with get_session() as session:
                exit_code = self._run_sync_body(session, config, season)
        except Exception as exc:  # noqa: BLE001
            self.report.status = "failed"
            self.report.exit_code = 1
            self.report.defects.append(
                {"severity": "fatal", "detail": f"{type(exc).__name__}: {exc}"}
            )
            exit_code = 1
        finally:
            self.report.finished_at = datetime.now(UTC).isoformat()
            self._write_report()
        return exit_code

    def _run_sync_body(self, session: Session, config: SleeperOwnerConfig, season: int) -> int:
        sync = SleeperSyncService(session, use_fixtures=False)
        user = sync.connect_user(config.username)
        user_id = str(user.get("user_id", ""))
        if not user_id:
            raise RuntimeError("Sleeper user lookup returned no user_id")

        discovered = sync.client.get_leagues(user_id, season)
        sync._record_last_source()
        self.report.league_selection = validate_league_selection(config, discovered)
        self.report.safety["sleeper_connectivity"] = "ok"
        self.report.safety["resolved_user_id"] = user_id

        availability = sync.sync_player_availability()
        synced = sync.sync_configured_leagues(
            user_id,
            season,
            config.allowed_league_ids,
            include_history=True,
        )
        if sorted(synced) != sorted(config.allowed_league_ids):
            missing = sorted(set(config.allowed_league_ids) - set(synced))
            raise RuntimeError(f"sync incomplete; missing leagues: {', '.join(missing)}")

        for entry in config.leagues:
            if entry.league_type == "dynasty" and entry.rookie_pick_rule:
                sync.persist_owner_confirmed_draft_rule(entry.league_id, entry.rookie_pick_rule)

        self.report.draft_rules = {
            row.league_id: row.rule
            for row in session.query(LeagueDraftRule).order_by(LeagueDraftRule.league_id).all()
        }

        season_week = resolve_season_week(session, nfl_state=sync.client.get_nfl_state())
        week = season_week.week

        scoring_failures: list[str] = []
        for entry in config.leagues:
            summary = _league_scoring_summary(session, entry.league_id, entry.display_name)
            self.report.scoring.append(summary)
            if summary["unsupported_keys"]:
                scoring_failures.append(
                    f"{entry.league_id}:unsupported={summary['unsupported_keys']}"
                )
            else:
                try:
                    require_publishable(
                        compile_sleeper_scoring(
                            (session.query(LeagueRuleSnapshot)
                             .filter(LeagueRuleSnapshot.league_id == entry.league_id)
                             .order_by(LeagueRuleSnapshot.fetched_at.desc())
                             .first()
                             .raw_json or {}),
                            summary["roster_positions"],
                        ),
                        entry.league_id,
                    )
                except ValueError as exc:
                    scoring_failures.append(f"{entry.league_id}:{exc}")

        bridge = ReleaseBridge(session)
        bridge.sync_preseason_pointer(season, automatic=False)
        from src.app.projections.weekly_run import WeeklyProjectionService

        WeeklyProjectionService(session).promote_week(
            season,
            week,
            automatic=False,
            league_ids=list(config.allowed_league_ids),
        )

        readiness = weekly_v2_readiness(season, week)
        self.report.projections = {
            "weekly_v2_state": readiness.state,
            "auto_publish_allowed": readiness.auto_publish_allowed,
            "manifest_uri": readiness.manifest_uri,
            "reasons": list(readiness.reasons),
            "shadow_mode_label": "fixture/fallback — not production-quality football advice",
        }
        self.report.league_scoring_shadow = _weekly_league_scoring_shadow(
            session, config, season=season, week=week
        )

        identity_gate = IdentityReconciliationGate(session, season=season, week=week)
        identity_report = identity_gate.run(
            [(entry.league_id, entry.display_name) for entry in config.leagues],
            week=week,
        )
        self.report.identity = identity_report.to_dict()
        unresolved_path = self.options.report_path.parent / "unresolved_player_ids.json"
        unresolved_path.write_text(
            json.dumps(identity_report.unresolved_artifact, indent=2),
            encoding="utf-8",
        )

        self.report.completeness = {
            "source_snapshots": _count_rows(session, SourceSnapshot),
            "roster_snapshots": _count_rows(session, RosterSnapshot),
            "transactions": _count_rows(session, LeagueTransaction),
            "traded_picks": _count_rows(session, TradedPick),
            "nfl_week": week,
            "season_week_source": season_week.source,
        }

        for entry in config.leagues:
            roster_id = _find_owner_roster(session, entry.league_id, user_id)
            if roster_id is None:
                self.report.defects.append(
                    {
                        "severity": "S1",
                        "detail": f"owner roster not found for league {entry.league_id}",
                    }
                )
                continue
            from src.app.persistence.models import League

            league_row = (
                session.query(League).filter(League.league_id == entry.league_id).one_or_none()
            )
            self.report.recommendations.append(
                _run_recommendation_smoke(
                    session,
                    league_id=entry.league_id,
                    display_name=entry.display_name,
                    roster_id=roster_id,
                    week=week,
                    league_type=entry.league_type,
                    league_status=league_row.status if league_row else None,
                )
            )

        counts_before = self._snapshot_counts(session)
        if not self.options.skip_second_run:
            sync.sync_configured_leagues(
                user_id,
                season,
                config.allowed_league_ids,
                include_history=False,
            )
            counts_after = self._snapshot_counts(session)
            self.report.idempotency = {
                "first_run": counts_before,
                "second_run": counts_after,
                "stable": counts_before == counts_after,
            }
        else:
            self.report.idempotency = {"skipped": True}

        if self.options.inject_failure:
            self.report.failure_injection = _inject_publication_failure(
                session, season=season, week=week
            )

        failures: list[str] = []
        failures.extend(scoring_failures)
        if identity_report.recommendation_gate_failed:
            failures.extend(identity_report.gate_failures)
        if any(not row.get("ok") for row in self.report.recommendations):
            failures.append("recommendation_smoke_incomplete")
        if self.report.idempotency.get("stable") is False:
            failures.append("idempotency_regression")
        if self.report.failure_injection and not self.report.failure_injection.get("passed", True):
            failures.append("failure_injection_regression")

        self.report.blockers = [
            "automatic weekly publication blocked until evaluation promotion passes",
            "PostgreSQL runtime unverified on this machine",
            "Docker compose runtime unverified on this machine",
            "email delivery unverified",
            "OpenAI assistant path unverified",
            "internet deployment unverified",
        ]
        if failures:
            self.report.status = "failed"
            self.report.exit_code = 1
            self.report.go_no_go = "no-go"
            self.report.defects.extend(
                {"severity": "gate", "detail": failure} for failure in failures
            )
        else:
            self.report.status = "passed"
            self.report.exit_code = 0
            self.report.go_no_go = "go"
        return self.report.exit_code

    @staticmethod
    def _snapshot_counts(session: Session) -> dict[str, int]:
        return {
            "draft_rules": _count_rows(session, LeagueDraftRule),
            "rosters": _count_rows(session, RosterSnapshot),
            "transactions": _count_rows(session, LeagueTransaction),
            "traded_picks": _count_rows(session, TradedPick),
            "source_snapshots": _count_rows(session, SourceSnapshot),
        }

    def _write_report(self) -> None:
        payload = self.report.to_dict()
        self.options.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path = self.options.report_path.with_suffix(".md")
        markdown_path.write_text(_render_markdown(self.report), encoding="utf-8")


def _render_markdown(report: ShadowSyncReport) -> str:
    lines = [
        "# Live Sleeper shadow sync report",
        "",
        f"- **Started:** {report.started_at}",
        f"- **Finished:** {report.finished_at}",
        f"- **Status:** {report.status}",
        f"- **Go / no-go:** {report.go_no_go}",
        "",
        "## Environment",
        f"- Database: `{report.environment.get('database_url', '')}`",
        f"- Artifact root: `{report.environment.get('artifact_root', '')}`",
        f"- Sleeper mode: `{report.environment.get('sleeper_mode', '')}`",
        "",
        "## Safety",
        f"- GET-only: `{report.safety.get('read_only_method')}`",
        f"- Connectivity: `{report.safety.get('sleeper_connectivity', 'unknown')}`",
        f"- Auto publish allowed: `{report.safety.get('auto_publish_allowed')}`",
        "",
        "## League selection",
        json.dumps(report.league_selection, indent=2),
        "",
        "## Draft rules",
        json.dumps(report.draft_rules, indent=2),
        "",
        "## Identity reconciliation",
        json.dumps(report.identity, indent=2),
        "",
        "## Projection mode",
        json.dumps(report.projections, indent=2),
        "",
        "## League scoring shadow",
        json.dumps(report.league_scoring_shadow, indent=2),
        "",
        "## Blockers",
    ]
    lines.extend(f"- {blocker}" for blocker in report.blockers)
    if report.defects:
        lines.extend(["", "## Defects"])
        lines.extend(f"- {row['severity']}: {row['detail']}" for row in report.defects)
    return "\n".join(lines) + "\n"


def run_shadow_sync(options: ShadowSyncOptions) -> int:
    return SleeperShadowSyncRunner(options).run()
