"""Decision services — lineup, waivers, trades wired to repositories.

Every response carries provenance: the projection run it came from, the league
scoring contract hash, the draw-source fidelity, and any player on the roster
that had no projection. There are no fixture player fallbacks in this path: if a
rostered player cannot be projected, that fact is reported rather than replaced
with invented data.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.app.decisions.draws import (
    DEFAULT_DRAW_COUNT,
    DrawSet,
    build_draw_set,
    stable_seed,
)
from src.app.decisions.lineup import (
    matchup_probabilities,
    optimize_lineup,
    swap_recommendations,
)
from src.app.decisions.tendencies import ManagerTendencyService
from src.app.decisions.trades import (
    RedraftPickNotTradeable,
    TradeEvaluationResult,
    TradeSide,
    evaluate_trade,
)
from src.app.decisions.waivers import WaiverPlayer, recommend_waivers
from src.app.persistence.repositories import LeagueRepository, ProjectionRepository
from src.app.projections.loader import PlayerSummary, get_bundle_loader
from src.app.releases.gates import validate_matchup_probabilities
from src.app.scoring.compiler import compile_sleeper_scoring, require_publishable
from src.app.scoring.contract import ScoringContract
from src.projection.special_teams.models import (
    KickerContext,
    TeamContext,
    simulate_dst_draw,
    simulate_kicker_draw,
)

#: Regular-season length used for horizon scaling when a league does not say.
REGULAR_SEASON_WEEKS = 17
DEFAULT_PLAYOFF_START_WEEK = 15
#: Dynasty horizon: current season plus three future seasons, discounted.
DYNASTY_SEASON_WEIGHTS = (1.0, 0.85, 0.70, 0.55)


class LeagueContextError(ValueError):
    """Raised when a league cannot be evaluated."""


class _LeagueContext:
    """Resolved league, contract, projection run, and draw source."""

    def __init__(
        self,
        session: Session,
        league_id: str,
        week: int | None,
        *,
        draw_count: int = DEFAULT_DRAW_COUNT,
    ) -> None:
        self.leagues = LeagueRepository(session)
        self.projections = ProjectionRepository(session)
        self.league = self.leagues.get_league(league_id)
        rules = self.leagues.latest_rules(league_id)
        if self.league is None or rules is None:
            raise LeagueContextError(f"league_or_rules_not_found:{league_id}")
        self.raw_scoring = rules.raw_json or {}
        self.roster_positions = list(
            (self.league.raw_json or {}).get("roster_positions", [])
        )
        self.contract: ScoringContract = compile_sleeper_scoring(
            self.raw_scoring, self.roster_positions
        )
        # Fail closed: an unmapped nonzero rule means we cannot reproduce this
        # league's scoring, so we must not publish a recommendation for it.
        require_publishable(self.contract, league_id)

        self.week = week
        self.season = self.league.season
        self.league_type = (self.league.league_type or "redraft").lower()
        self.run = self._resolve_run(week)
        self.projection_run_id = self.run.id if self.run else None
        self.bundle = get_bundle_loader(self.season)
        self.draw_count = draw_count

    def _resolve_run(self, week: int | None):
        if week is not None:
            run = self.projections.active_run(
                mode="weekly", season=self.season, week=week
            )
            if run is not None:
                return run
        return self.projections.active_run(
            mode="preseason", season=self.season, week=None
        )

    # ------------------------------------------------------------------ players
    def summaries_for(self, player_ids: list[str]) -> tuple[list[PlayerSummary], list[str]]:
        """Resolve projections for player ids, preferring the active run."""
        wanted = [pid for pid in player_ids if pid]
        found: dict[str, PlayerSummary] = {}

        if self.run is not None:
            for row in self.projections.player_projections(self.run.id, wanted):
                mean = row.mean_json or {}
                quantiles = row.quantiles_json or {}
                found[row.player_id] = PlayerSummary(
                    player_id=row.player_id,
                    name=str(mean.get("name") or row.player_id),
                    position=str(mean.get("position") or "RB"),
                    team=mean.get("team") or row.team,
                    mean_points=float(mean.get("points") or 0.0),
                    quantiles={str(k): float(v) for k, v in quantiles.items()},
                    availability_probability=float(row.availability_probability or 0.0),
                )

        for pid in wanted:
            if pid in found:
                continue
            summary = self.bundle.get(pid)
            if summary is not None:
                found[pid] = summary

        still_missing = [pid for pid in wanted if pid not in found]
        if still_missing:
            found.update(self._special_teams_identities(still_missing))

        missing = [pid for pid in wanted if pid not in found]
        return [found[pid] for pid in wanted if pid in found], missing

    def _special_teams_identities(self, player_ids: list[str]) -> dict[str, PlayerSummary]:
        """Resolve kickers and team defenses from the identity table.

        The projection release publishes offense only, so a kicker or defense has
        no point summary. Their draws come from the special-teams simulator
        instead, so an identity record is sufficient and no point value is
        invented here.
        """
        from src.app.persistence.models import PlayerIdentity

        rows = (
            self.leagues.session.query(PlayerIdentity)
            .filter(
                PlayerIdentity.player_id.in_(player_ids),
                PlayerIdentity.position.in_(["K", "DEF", "DST"]),
            )
            .all()
        )
        resolved: dict[str, PlayerSummary] = {}
        for row in rows:
            resolved[row.player_id] = PlayerSummary(
                player_id=row.player_id,
                name=row.name or row.player_id,
                position="K" if row.position == "K" else "DEF",
                team=row.team,
                # Deliberately zero: the value comes from the stat-level draws.
                mean_points=0.0,
                quantiles={},
                availability_probability=1.0,
            )
        return resolved

    def special_teams_draws(
        self, summaries: list[PlayerSummary]
    ) -> tuple[dict[str, list[dict[str, float]]], dict[str, str]]:
        """Stat-level draws for kickers and team defenses.

        These are the documented intentionally-simple models. Because they are
        stat-level, tiered points-allowed and field-goal-distance rules score
        exactly. Their uncertainty is wide and reported as such.
        """
        stat_draws: dict[str, list[dict[str, float]]] = {}
        positions: dict[str, str] = {}
        for summary in summaries:
            if summary.position not in {"K", "DEF", "DST"}:
                continue
            positions[summary.player_id] = "DEF" if summary.position != "K" else "K"
            # Seed per player so two defenses are not perfectly correlated, and
            # stably so the same run reproduces the same draws.
            base = stable_seed(summary.player_id, self.projection_run_id, self.week) % (
                2**31
            )
            draws = []
            for i in range(self.draw_count):
                seed = (base + i) % (2**31)
                if summary.position == "K":
                    draws.append(simulate_kicker_draw(KickerContext(), seed=seed))
                else:
                    draws.append(simulate_dst_draw(TeamContext(), seed=seed))
            stat_draws[summary.player_id] = draws
        return stat_draws, positions

    def build_draws(
        self,
        player_ids: list[str],
        *,
        locked_player_ids: list[str] | None = None,
        actual_points: dict[str, float] | None = None,
    ) -> tuple[DrawSet, list[str]]:
        summaries, missing = self.summaries_for(player_ids)
        offense = [s for s in summaries if s.position not in {"K", "DEF", "DST"}]
        stat_draws, st_positions = self.special_teams_draws(summaries)
        draw_set = build_draw_set(
            contract=self.contract,
            run_id=self.projection_run_id or "no-active-run",
            week=self.week,
            draw_count=self.draw_count,
            point_summaries=offense,
            stat_draws_by_player=stat_draws,
            positions=st_positions,
            locked_player_ids=locked_player_ids or (),
            actual_points=actual_points,
        )
        return draw_set, missing

    def data_as_of(self) -> str:
        if self.run is not None and getattr(self.run, "as_of", None):
            as_of = self.run.as_of
            if isinstance(as_of, datetime):
                return as_of.astimezone(UTC).isoformat()
            return str(as_of)
        if self.bundle.load():
            return self.bundle.as_of()
        return datetime.now(UTC).isoformat()

    def provenance(self, draw_set: DrawSet, missing: list[str]) -> dict:
        return {
            "projection_run_id": self.projection_run_id,
            "projection_available": self.projection_run_id is not None,
            "contract_hash": self.contract.contract_hash,
            "league_type": self.league_type,
            "data_as_of": self.data_as_of(),
            "draw_count": draw_set.draw_count,
            "scoring_fidelity": draw_set.mode,
            "scoring_fidelity_note": draw_set.fidelity_note(),
            "unapplied_scoring_rules": draw_set.unapplied_rules,
            "players_without_projection": missing,
            "baseline_scoring": self.bundle.meta.get("scoring"),
            # The active release publishes means and uncertainty bands from
            # different models; these players' bands were recentred on the
            # promoted mean so the overlay cannot override the point forecast.
            "recentred_uncertainty_players": draw_set.recentred_players,
        }


class LineupService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.leagues = LeagueRepository(session)
        self.projections = ProjectionRepository(session)

    def recommend(
        self,
        league_id: str,
        week: int,
        *,
        opponent_mode: str = "current",
        user_roster_id: int = 1,
        opponent_roster_id: int | None = None,
    ) -> dict:
        if opponent_mode not in {"current", "optimized"}:
            raise LeagueContextError(f"invalid_opponent_mode:{opponent_mode}")
        ctx = _LeagueContext(self.session, league_id, week)

        rosters = self.leagues.latest_rosters(league_id, week)
        user_roster = next((r for r in rosters if r.roster_id == user_roster_id), None)
        if user_roster is None:
            raise LeagueContextError(
                f"no_roster_snapshot:league={league_id},roster={user_roster_id},week={week}"
            )
        if opponent_roster_id is None:
            opponent = next((r for r in rosters if r.roster_id != user_roster_id), None)
        else:
            opponent = next(
                (r for r in rosters if r.roster_id == opponent_roster_id), None
            )

        user_ids = [pid for pid in (user_roster.players or []) if pid]
        opp_ids = [pid for pid in ((opponent.players if opponent else []) or []) if pid]
        draw_set, missing = ctx.build_draws(user_ids + opp_ids)

        user_candidates = [pid for pid in user_ids if pid in draw_set.players]
        opp_candidates = [pid for pid in opp_ids if pid in draw_set.players]
        submitted_user = [
            pid for pid in (user_roster.starters or []) if pid in draw_set.players
        ]
        submitted_opp = [
            pid
            for pid in ((opponent.starters if opponent else []) or [])
            if pid in draw_set.players
        ]

        if not user_candidates:
            raise LeagueContextError(
                f"no_projected_players_on_roster:league={league_id},week={week}"
            )

        evaluation = matchup_probabilities(
            draw_set,
            ctx.contract,
            user_candidate_ids=user_candidates,
            opponent_candidate_ids=opp_candidates,
            user_starters=submitted_user,
            opponent_mode=opponent_mode,
            opponent_submitted_starters=submitted_opp,
        )
        recommended = evaluation["recommended"]
        probs = evaluation["recommended_probabilities"]

        gate = validate_matchup_probabilities(probs)
        if not gate.passed:
            raise LeagueContextError(
                f"matchup_probability_gate_failed:{gate.failures}"
            )

        opponent_totals = draw_set.totals_for(evaluation["opponent_starters"])
        swaps = swap_recommendations(
            draw_set,
            ctx.contract,
            current_starters=submitted_user,
            recommended=recommended,
            opponent_totals=opponent_totals,
        )

        starter_details = []
        for pid in recommended.starters:
            player = draw_set.players[pid]
            summary = ctx.bundle.get(pid)
            starter_details.append(
                {
                    "player_id": pid,
                    "name": summary.name if summary else pid,
                    "position": player.position,
                    "team": summary.team if summary else None,
                    "slot": recommended.assignments.get(pid),
                    "expected_points": round(player.mean, 3),
                    "points_p10": round(player.percentile(0.1), 3),
                    "points_p50": round(player.percentile(0.5), 3),
                    "points_p90": round(player.percentile(0.9), 3),
                    "availability_probability": round(player.availability_probability, 4),
                    "draw_mode": player.mode,
                    "locked": player.locked,
                }
            )

        provenance = ctx.provenance(draw_set, missing)
        return {
            "week": week,
            "opponent_mode": opponent_mode,
            "opponent_lineup_source": evaluation["opponent_lineup_source"],
            "recommended_starters": recommended.starters,
            "starters": starter_details,
            "slot_assignments": recommended.assignments,
            "unfilled_seats": recommended.unfilled_seats,
            "expected_points": round(recommended.expected_points, 4),
            "quantiles": {k: round(v, 4) for k, v in recommended.quantiles.items()},
            "objective": recommended.objective,
            "matchup_probabilities": probs,
            "win_probability": probs.get("win", 0.0),
            "current_lineup_probabilities": evaluation["current_probabilities"],
            "current_starters": evaluation["current_starters"],
            "current_expected_points": round(evaluation["current_expected_points"], 4),
            "opponent_starters": evaluation["opponent_starters"],
            "opponent_expected_points": round(evaluation["opponent_expected_points"], 4),
            "recommended_swaps": swaps,
            "swaps": swaps,
            **provenance,
            "meta": provenance,
        }


class WaiverService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.leagues = LeagueRepository(session)

    def recommend(
        self,
        league_id: str,
        week: int,
        *,
        remaining_faab: float = 100.0,
        user_roster_id: int = 1,
    ) -> dict:
        ctx = _LeagueContext(self.session, league_id, week)
        rosters = self.leagues.latest_rosters(league_id, week)
        rostered = {pid for roster in rosters for pid in (roster.players or []) if pid}
        user_roster = next((r for r in rosters if r.roster_id == user_roster_id), None)
        user_ids = [pid for pid in ((user_roster.players if user_roster else []) or []) if pid]

        startable = ctx.contract.eligible_positions()
        pool_summaries = [
            s
            for s in ctx.bundle.available_pool(rostered)
            if s.position in startable
        ]
        roster_summaries, _missing_roster = ctx.summaries_for(user_ids)

        pool = [_to_waiver_player(s) for s in pool_summaries]
        roster_players = [_to_waiver_player(s) for s in roster_summaries]

        # Positional counts across every roster in the league: scarcity signal.
        league_counts: dict[str, int] = {}
        all_rostered_summaries, _ = ctx.summaries_for(sorted(rostered))
        for summary in all_rostered_summaries:
            league_counts[summary.position] = league_counts.get(summary.position, 0) + 1

        trending = self._trending_adds(league_id)
        weeks_remaining = max(0, REGULAR_SEASON_WEEKS - week)

        recs = recommend_waivers(
            pool,
            contract=ctx.contract,
            roster=roster_players,
            remaining_faab=remaining_faab,
            week=week,
            weeks_remaining=weeks_remaining,
            playoff_start_week=DEFAULT_PLAYOFF_START_WEEK,
            league_position_counts=league_counts,
            trending_adds=trending,
        )

        draw_set, missing = ctx.build_draws(user_ids)
        provenance = ctx.provenance(draw_set, missing)
        adds = [
            {
                "player_id": r.player_id,
                "name": r.name,
                "position": r.position,
                "faab_min": r.faab_low,
                "faab_max": r.faab_high,
                "confidence": r.confidence,
                "start_probability": r.start_probability,
                "replacement_level": r.replacement_level,
                "incremental_utility": r.incremental_utility,
                "reason": " ".join(r.rationale),
                "rationale": r.rationale,
            }
            for r in recs
        ]
        return {
            "week": week,
            "remaining_faab": remaining_faab,
            "recommendations": [asdict(r) for r in recs],
            "adds": adds,
            "trending_adds_considered": sorted(trending),
            **provenance,
            "meta": provenance,
        }

    def _trending_adds(self, league_id: str) -> dict[str, int]:
        """Sleeper trending adds recorded by sync for *this* league, if any.

        Returns an empty mapping when no trending snapshot exists. It is never
        synthesised from the projection pool, because that would turn our own
        forecast into a fake market signal.

        Three things have to line up for the signal to be usable, and each was
        previously wrong: the payload lives in the artifact store (the snapshot
        row only holds a URI), the market-signal envelope keys players as
        ``sleeper_player_id``/``add_count``, and those are Sleeper ids that must
        be resolved to canonical player ids before they can match a waiver
        candidate. A snapshot from a different league is not this league's
        market, so the lookup is scoped by ``league_id``.
        """
        from src.app.artifacts.store import ArtifactError, get_artifact_store
        from src.app.availability.identity import PlayerIdentityResolver
        from src.app.league.sleeper.sync import MARKET_SIGNAL_ENDPOINT
        from src.app.persistence.models import SourceSnapshot

        rows = (
            self.session.query(SourceSnapshot)
            .filter(SourceSnapshot.endpoint == MARKET_SIGNAL_ENDPOINT)
            .order_by(SourceSnapshot.fetched_at.desc())
            .limit(50)
            .all()
        )
        payload: dict | None = None
        store = get_artifact_store()
        for row in rows:
            if str((row.request_params_json or {}).get("league_id")) != str(league_id):
                continue
            try:
                candidate = store.get_json(row.artifact_uri)
            except (ArtifactError, OSError, ValueError):
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            return {}
        entries = payload.get("players")
        if not isinstance(entries, list):
            return {}
        resolver = PlayerIdentityResolver(self.session)
        counts: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("sleeper_player_id") or entry.get("player_id")
            count = entry.get("add_count", entry.get("count"))
            if not raw_id or not isinstance(count, (int, float)):
                continue
            resolution = resolver.resolve(player_id=str(raw_id), sleeper_id=str(raw_id))
            if resolution.status != "resolved" or not resolution.player_id:
                # An unknown trending player is not on any roster we score, and
                # guessing an identity would attach urgency to the wrong player.
                continue
            counts[resolution.player_id] = int(count)
        return counts


class TradeService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.tendencies = ManagerTendencyService(session)

    def evaluate(
        self,
        league_id: str,
        side_a: TradeSide,
        side_b: TradeSide,
        *,
        horizon: str = "ros",
        week: int = 1,
    ) -> TradeEvaluationResult:
        if horizon not in {"weekly", "ros", "dynasty"}:
            raise LeagueContextError(f"invalid_horizon:{horizon}")
        ctx = _LeagueContext(self.session, league_id, week)

        player_ids = list(side_a.player_ids) + list(side_b.player_ids)
        summaries, missing = ctx.summaries_for(player_ids)
        multiplier = _horizon_multiplier(horizon, week)

        player_values = {
            s.player_id: s.mean_points * multiplier * max(s.availability_probability, 0.0)
            for s in summaries
        }

        tendency = self.tendencies.tendency_adjustment(
            league_id,
            side_b.roster_id,
            package_size=len(player_ids)
            + len(side_a.pick_assets)
            + len(side_b.pick_assets),
        )
        sample_size = self._tendency_sample_size(league_id, side_b.roster_id)

        try:
            result = evaluate_trade(
                side_a,
                side_b,
                player_values=player_values,
                league_type=ctx.league_type,
                current_season=ctx.season,
                tendency_adjustment=tendency,
                tendency_sample_size=sample_size,
                roster_context=self._roster_context(league_id, side_a, side_b),
                horizon=horizon,
            )
        except RedraftPickNotTradeable:
            raise
        for pid in missing:
            result.warnings.append(f"no_projection_for_asset:{pid}")
        result.objective["provenance"] = {
            "projection_run_id": ctx.projection_run_id,
            "contract_hash": ctx.contract.contract_hash,
            "data_as_of": ctx.data_as_of(),
            "horizon_multiplier": multiplier,
            "baseline_scoring": ctx.bundle.meta.get("scoring"),
        }
        return result

    def _tendency_sample_size(self, league_id: str, roster_id: int) -> int:
        try:
            features = self.tendencies.get(league_id, roster_id)
        except Exception:  # pragma: no cover - defensive
            return 0
        return int(getattr(features, "sample_size", 0) or 0)

    def _roster_context(
        self, league_id: str, side_a: TradeSide, side_b: TradeSide
    ) -> dict:
        from src.app.persistence.models import ManagerState

        def label_for(roster_id: int) -> str | None:
            row = (
                self.session.query(ManagerState)
                .filter(
                    ManagerState.league_id == league_id,
                    ManagerState.roster_id == roster_id,
                )
                .order_by(ManagerState.as_of.desc())
                .first()
            )
            if row is None:
                return None
            return row.overridden_label or row.label

        return {
            "side_a_state": label_for(side_a.roster_id),
            "side_b_state": label_for(side_b.roster_id),
        }


def _horizon_multiplier(horizon: str, week: int) -> float:
    """Convert per-week projected points into horizon value."""
    if horizon == "weekly":
        return 1.0
    if horizon == "ros":
        return float(max(1, REGULAR_SEASON_WEEKS - week + 1))
    # Dynasty: this season's remaining weeks plus three discounted future seasons.
    remaining = float(max(1, REGULAR_SEASON_WEEKS - week + 1))
    future = sum(
        REGULAR_SEASON_WEEKS * weight for weight in DYNASTY_SEASON_WEIGHTS[1:]
    )
    return remaining + future


def _to_waiver_player(summary: PlayerSummary) -> WaiverPlayer:
    quantiles = summary.quantiles or {}

    def q(key: str, fallback: float) -> float:
        for candidate in (key, f"{float(key):g}"):
            if candidate in quantiles:
                try:
                    return float(quantiles[candidate])
                except (TypeError, ValueError):
                    continue
        return fallback

    return WaiverPlayer(
        player_id=summary.player_id,
        name=summary.name,
        position=summary.position,
        mean_points=float(summary.mean_points),
        p10=q("0.1", summary.mean_points * 0.6),
        p90=q("0.9", summary.mean_points * 1.4),
        team=summary.team,
        availability_probability=float(summary.availability_probability or 0.0),
    )
