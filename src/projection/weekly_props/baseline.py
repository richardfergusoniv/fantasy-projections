"""Week-level baselines for weekly_props merge and movement gates.

Season-long point summaries are not a valid baseline: the scorer rebuilds points
from components, so uncovered players collapse to zero and movement gates compare
weekly projections against season totals. This module builds a week-scoped
component baseline (sealed per-game components + slate/bye + availability).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.app.projections.league_rescore import load_component_projections
from src.app.projections.loader import PlayerSummary, ReleaseBundleLoader
from src.app.projections.status_overlay import load_overlay_players, read_active_overlay
from src.projection.weekly_props.bridge import BaselinePlayer
from src.projection.weekly_props.scoring import COMPONENT_KEYS, half_ppr_parity_points


class WeeklyBaselineError(RuntimeError):
    """Raised when a trustworthy week-level baseline cannot be constructed."""


@dataclass(frozen=True)
class WeekSlate:
    season: int
    week: int
    teams: frozenset[str]
    opponents: Mapping[str, str]

    def on_slate(self, team: str | None) -> bool:
        if not team:
            return False
        return str(team).upper() in self.teams


@dataclass(frozen=True)
class WeeklyBaselineBundle:
    season: int
    week: int
    baselines: dict[str, BaselinePlayer]
    slate: WeekSlate
    baseline_source: str
    baseline_run_id: str | None
    status_overlay_id: str | None
    component_player_count: int

    @property
    def slate_teams(self) -> set[str]:
        return set(self.slate.teams)


def load_week_slate(season: int, week: int) -> WeekSlate:
    """Load regular-season slate teams and opponents for ``season``/``week``."""
    try:
        from src.projection.weekly.data.nflverse_loader import load_schedules
        from src.projection.weekly.features.team_context import (
            explode_schedules_to_team_weeks,
        )
    except Exception as exc:
        raise WeeklyBaselineError(f"schedule_loader_unavailable:{exc}") from exc

    try:
        schedules = load_schedules([season], force=False)
        team_weeks = explode_schedules_to_team_weeks(schedules)
        week_rows = team_weeks.filter(
            (team_weeks["season"] == season) & (team_weeks["week"] == week)
        )
    except Exception as exc:
        raise WeeklyBaselineError(f"schedule_load_failed:{exc}") from exc

    if week_rows.is_empty():
        raise WeeklyBaselineError(f"empty_slate:season={season}:week={week}")

    teams: set[str] = set()
    opponents: dict[str, str] = {}
    for record in week_rows.iter_rows(named=True):
        team = str(record.get("team") or "").upper()
        opponent = str(record.get("opponent") or "").upper() or None
        if not team:
            continue
        teams.add(team)
        if opponent:
            opponents[team] = opponent

    if len(teams) < 2:
        raise WeeklyBaselineError(f"incomplete_slate:season={season}:week={week}:teams={len(teams)}")
    return WeekSlate(
        season=season,
        week=week,
        teams=frozenset(teams),
        opponents=opponents,
    )


def _availability(summary: PlayerSummary) -> float:
    """Preserve a real 0.0 availability; only default when the value is missing."""
    value = summary.availability_probability
    if value is None:
        return 1.0
    return float(value)


def _weekly_quantiles(summary: PlayerSummary, *, points: float, on_slate: bool) -> dict[str, float]:
    if not on_slate:
        return {str(k): 0.0 for k in (summary.quantiles or {"0.1": 0.0, "0.5": 0.0, "0.9": 0.0})}
    if summary.quantiles:
        return {str(k): float(v) for k, v in summary.quantiles.items()}
    return {
        "0.1": max(0.0, points * 0.7),
        "0.5": max(0.0, points),
        "0.9": max(0.0, points * 1.3),
    }


def _component_mean_json(
    *,
    summary: PlayerSummary,
    components: Mapping[str, Any],
    on_slate: bool,
) -> dict[str, Any]:
    mean: dict[str, Any] = {
        "position": summary.position,
        "name": summary.name,
        "team": summary.team,
        "derivation": "weekly_props_baseline_v1",
    }
    for key in COMPONENT_KEYS:
        raw = components.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        mean[key] = value if on_slate else 0.0

    if on_slate:
        points = half_ppr_parity_points(mean)
        # Prefer sealed per-game points when present so movement gates stay aligned
        # with the published weekly baseline, not a parity helper drift.
        if summary.mean_points is not None:
            points = float(summary.mean_points)
        mean["points"] = points
        mean["points_source"] = "sealed_per_game_baseline"
    else:
        mean["points"] = 0.0
        mean["points_source"] = "bye_or_inactive"
    return mean


def build_weekly_baselines(
    *,
    season: int,
    week: int,
    slate: WeekSlate | None = None,
    players: Mapping[str, PlayerSummary] | None = None,
    components_by_player: Mapping[str, Mapping[str, Any]] | None = None,
    status_overlay_id: str | None = None,
) -> WeeklyBaselineBundle:
    """Construct week-scoped component baselines or raise ``WeeklyBaselineError``."""
    slate = slate or load_week_slate(season, week)

    loader = ReleaseBundleLoader(season=season)
    bundle = loader.load_bundle()
    if bundle is None or not bundle.players:
        raise WeeklyBaselineError(f"missing_sealed_bundle:season={season}")

    if components_by_player is None:
        if bundle.component_projections_path is None:
            raise WeeklyBaselineError("missing_component_projections_path")
        components_by_player = load_component_projections(bundle.component_projections_path)
    if not components_by_player:
        raise WeeklyBaselineError("empty_component_projections")

    summaries: dict[str, PlayerSummary] = dict(players) if players is not None else dict(bundle.players)
    overlay_id = status_overlay_id
    if players is None:
        overlay_players = load_overlay_players(season)
        if overlay_players:
            summaries = overlay_players
            pointer = read_active_overlay(season)
            if pointer is not None:
                overlay_id = overlay_id or str(
                    pointer.get("overlay_hash") or pointer.get("overlay_id") or ""
                ) or None

    baselines: dict[str, BaselinePlayer] = {}
    component_hits = 0
    for player_id, summary in summaries.items():
        comps = components_by_player.get(player_id)
        if not comps:
            continue
        component_hits += 1
        team = summary.team
        on_slate = slate.on_slate(team)
        availability = 0.0 if not on_slate else _availability(summary)
        mean_json = _component_mean_json(summary=summary, components=comps, on_slate=on_slate)
        points = float(mean_json.get("points") or 0.0)
        baselines[player_id] = BaselinePlayer(
            player_id=player_id,
            team=team,
            opponent=slate.opponents.get(str(team or "").upper()),
            position=summary.position,
            name=summary.name,
            availability_probability=availability,
            mean_json=mean_json,
            quantiles_json=_weekly_quantiles(summary, points=points, on_slate=on_slate),
            on_slate=on_slate,
        )

    if component_hits < 50 or len(baselines) < 50:
        raise WeeklyBaselineError(
            f"insufficient_component_baseline:players={len(baselines)}:components={component_hits}"
        )

    return WeeklyBaselineBundle(
        season=season,
        week=week,
        baselines=baselines,
        slate=slate,
        baseline_source="status_adjusted_components"
        if overlay_id
        else "sealed_components_week",
        baseline_run_id=bundle.release_id,
        status_overlay_id=overlay_id,
        component_player_count=component_hits,
    )


def build_identity_map(
    baselines: Mapping[str, BaselinePlayer],
) -> dict[str, dict[str, Any]]:
    """Index baselines by canonical id and unambiguous name keys."""
    from src.app.availability.identity import normalize_name
    from src.draft_assistant.market_adp import canonicalize_player_name

    identity_map: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[str]] = {}
    for pid, baseline in baselines.items():
        payload = {
            "player_id": pid,
            "name": baseline.name,
            "team": baseline.team,
            "position": baseline.position,
            "opponent": baseline.opponent,
        }
        identity_map[pid] = payload
        canon = canonicalize_player_name(baseline.name or "")
        norm = normalize_name(baseline.name or "")
        if canon:
            by_name.setdefault(canon, []).append(pid)
        if norm and norm != canon:
            by_name.setdefault(norm, []).append(pid)
        team = str(baseline.team or "").upper()
        pos = str(baseline.position or "").upper()
        if canon and team:
            identity_map[f"{canon}|{team}"] = payload
            if pos:
                identity_map[f"{canon}|{team}|{pos}"] = payload

    for name, ids in by_name.items():
        unique = sorted(set(ids))
        if len(unique) == 1:
            identity_map[name] = identity_map[unique[0]]
    return identity_map
