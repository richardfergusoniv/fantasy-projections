"""Identity reconciliation for live Sleeper shadow sync.

Reports how rostered Sleeper player ids map onto canonical identities without
printing roster contents or private member data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from src.app.availability.identity import PlayerIdentityResolver
from src.app.persistence.models import League, PlayerIdentity, RosterSnapshot
from src.app.persistence.repositories import ProjectionRepository


@dataclass
class PlayerDisposition:
    sleeper_id: str
    category: str
    position: str | None = None
    team: str | None = None
    reason: str | None = None


@dataclass
class LeagueIdentityReport:
    league_id: str
    display_name: str
    total_distinct_rostered_ids: int = 0
    resolved_canonical: int = 0
    unresolved_ids: int = 0
    ambiguous_ids: int = 0
    resolved_starters: int = 0
    unresolved_starters: int = 0
    missing_weekly_projections: int = 0
    missing_season_projections: int = 0
    outside_projection_universe: dict[str, int] = field(default_factory=dict)
    unresolved_starter_ids: list[str] = field(default_factory=list)
    unresolved_skill_ids: list[str] = field(default_factory=list)
    disposition_samples: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_id": self.league_id,
            "display_name": self.display_name,
            "total_distinct_rostered_ids": self.total_distinct_rostered_ids,
            "resolved_canonical": self.resolved_canonical,
            "unresolved_ids": self.unresolved_ids,
            "ambiguous_ids": self.ambiguous_ids,
            "resolved_starters": self.resolved_starters,
            "unresolved_starters": self.unresolved_starters,
            "missing_weekly_projections": self.missing_weekly_projections,
            "missing_season_projections": self.missing_season_projections,
            "outside_projection_universe": dict(self.outside_projection_universe),
            "unresolved_starter_ids": list(self.unresolved_starter_ids),
            "unresolved_skill_ids": list(self.unresolved_skill_ids),
        }


@dataclass
class IdentityReconciliationReport:
    by_league: list[LeagueIdentityReport] = field(default_factory=list)
    aggregate: dict[str, int] = field(default_factory=dict)
    unresolved_artifact: list[dict[str, str]] = field(default_factory=list)
    recommendation_gate_failed: bool = False
    gate_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_league": [row.to_dict() for row in self.by_league],
            "aggregate": dict(self.aggregate),
            "unresolved_artifact_count": len(self.unresolved_artifact),
            "recommendation_gate_failed": self.recommendation_gate_failed,
            "gate_failures": list(self.gate_failures),
        }


SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


class IdentityReconciliationGate:
    def __init__(self, session: Session, *, season: int, week: int) -> None:
        self.session = session
        self.season = season
        self.week = week
        self.resolver = PlayerIdentityResolver(session)
        self.projections = ProjectionRepository(session)
        self._weekly_run = self.projections.active_run(mode="weekly", season=season, week=week)
        if self._weekly_run is None:
            self._weekly_run = self.projections.active_run(mode="preseason", season=season, week=None)
        self._weekly_projected: set[str] | None = None
        self._season_projected: set[str] | None = None

    def _projected_ids(self, mode: str, week: int | None) -> set[str]:
        run = self.projections.active_run(mode=mode, season=self.season, week=week)
        if run is None:
            return set()
        return {
            row.player_id
            for row in self.projections.player_projections(run.id)
        }

    @property
    def weekly_projected(self) -> set[str]:
        if self._weekly_projected is None:
            self._weekly_projected = self._projected_ids("weekly", self.week)
            if not self._weekly_projected and self._weekly_run is not None:
                self._weekly_projected = {
                    row.player_id
                    for row in self.projections.player_projections(self._weekly_run.id)
                }
        return self._weekly_projected

    @property
    def season_projected(self) -> set[str]:
        if self._season_projected is None:
            self._season_projected = self._projected_ids("preseason", None)
        return self._season_projected

    def _categorize_unresolved(
        self,
        sleeper_id: str,
        identity_row: PlayerIdentity | None,
    ) -> str:
        if identity_row is None:
            return "unknown"
        position = (identity_row.position or "").upper()
        team = (identity_row.team or "").upper()
        if position in {"DEF", "DST"}:
            return "team_defense"
        if position == "K":
            return "kicker"
        if position == "IDP" or position.startswith("LB") or position.startswith("DB"):
            return "idp"
        if team in {"", "FA", "FREE AGENT"}:
            return "free_agent"
        if team == "RET":
            return "retired"
        return "unknown"

    def _resolve_status(self, sleeper_id: str) -> tuple[str, str | None]:
        resolution = self.resolver.resolve(sleeper_id=sleeper_id, player_id=sleeper_id)
        if resolution.status == "resolved" and resolution.player_id:
            return "resolved", resolution.player_id
        if resolution.status == "ambiguous":
            return "ambiguous", None
        return "unresolved", None

    def analyze_league(
        self,
        league_id: str,
        *,
        display_name: str,
        week: int,
    ) -> LeagueIdentityReport:
        report = LeagueIdentityReport(league_id=league_id, display_name=display_name)
        rosters = (
            self.session.query(RosterSnapshot)
            .filter(RosterSnapshot.league_id == league_id, RosterSnapshot.week == week)
            .all()
        )
        rostered: set[str] = set()
        starters: set[str] = set()
        for row in rosters:
            rostered.update(str(pid) for pid in (row.players or []) if pid)
            starters.update(
                str(pid)
                for pid in (row.starters or [])
                if pid not in (None, "", "0", 0)
            )
        report.total_distinct_rostered_ids = len(rostered)

        identity_by_sleeper = {
            row.sleeper_id: row
            for row in self.session.query(PlayerIdentity).all()
            if row.sleeper_id
        }
        identity_by_canonical = {
            row.player_id: row for row in self.session.query(PlayerIdentity).all()
        }

        for sleeper_id in sorted(rostered):
            status, canonical = self._resolve_status(sleeper_id)
            identity_row = identity_by_sleeper.get(sleeper_id) or identity_by_canonical.get(sleeper_id)
            if status == "resolved":
                report.resolved_canonical += 1
                if canonical and canonical not in self.weekly_projected:
                    report.missing_weekly_projections += 1
                if canonical and canonical not in self.season_projected:
                    report.missing_season_projections += 1
            elif status == "ambiguous":
                report.ambiguous_ids += 1
            else:
                report.unresolved_ids += 1
                category = self._categorize_unresolved(sleeper_id, identity_row)
                report.outside_projection_universe[category] = (
                    report.outside_projection_universe.get(category, 0) + 1
                )
                if len(report.disposition_samples) < 25:
                    report.disposition_samples.append(
                        {
                            "sleeper_id": sleeper_id,
                            "category": category,
                            "position": identity_row.position if identity_row else "",
                            "team": identity_row.team if identity_row else "",
                        }
                    )
                position = (identity_row.position or "").upper() if identity_row else ""
                if sleeper_id in starters:
                    report.unresolved_starter_ids.append(sleeper_id)
                elif position in SKILL_POSITIONS:
                    report.unresolved_skill_ids.append(sleeper_id)

        for starter_id in starters:
            status, _ = self._resolve_status(starter_id)
            if status == "resolved":
                report.resolved_starters += 1
            else:
                report.unresolved_starters += 1

        return report

    def run(
        self,
        league_entries: list[tuple[str, str]],
        *,
        week: int,
    ) -> IdentityReconciliationReport:
        result = IdentityReconciliationReport()
        totals = {
            "total_distinct_rostered_ids": 0,
            "resolved_canonical": 0,
            "unresolved_ids": 0,
            "ambiguous_ids": 0,
            "resolved_starters": 0,
            "unresolved_starters": 0,
            "missing_weekly_projections": 0,
            "missing_season_projections": 0,
        }
        for league_id, display_name in league_entries:
            league_report = self.analyze_league(league_id, display_name=display_name, week=week)
            result.by_league.append(league_report)
            for key in totals:
                totals[key] += getattr(league_report, key)
            if league_report.unresolved_starters > 0:
                result.gate_failures.append(
                    f"unresolved_starters:{league_id}:{league_report.unresolved_starters}"
                )
            for sample in league_report.disposition_samples:
                result.unresolved_artifact.append(
                    {
                        "league_id": league_id,
                        **sample,
                    }
                )
        result.aggregate = totals
        result.recommendation_gate_failed = bool(result.gate_failures)
        return result
