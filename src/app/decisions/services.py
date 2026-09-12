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

from src.app.availability.identity import PlayerIdentityResolver, normalize_name
from src.app.decisions.draws import (
    DEFAULT_DRAW_COUNT,
    DrawSet,
    build_draw_set,
    stable_seed,
)
from src.app.decisions.lineup import (
    assign_submitted_seat_labels,
    matchup_probabilities,
    optimize_lineup,
    swap_recommendations,
)
from src.app.decisions.matchup_cache import (
    board_source_for_projection_source,
    load_matchup_board,
    mode_payload_from_board,
    projection_run_exists,
    roster_input_fingerprint,
    store_matchup_board,
)
from src.app.decisions.tendencies import ManagerTendencyService
from src.app.decisions.trades import (
    RedraftPickNotTradeable,
    TradeEvaluationResult,
    TradeSide,
    evaluate_trade,
)
from src.app.decisions.waivers import WaiverPlayer, recommend_waivers
from src.app.persistence.models import LeagueMember, MatchupSnapshot
from src.app.persistence.repositories import LeagueRepository, ProjectionRepository
from src.app.projections.loader import PlayerSummary, get_bundle_loader
from src.app.projections.service import ProjectionService
from src.app.projections.source import (
    ProjectionSource,
    configured_projection_source,
    weekly_rnd_enabled,
)
from src.app.releases.gates import validate_matchup_probabilities
from src.app.scoring.compiler import compile_sleeper_scoring, require_publishable
from src.app.scoring.contract import ScoringContract
from src.draft_assistant.market_adp import canonicalize_player_name
from src.projection.special_teams.models import (
    KickerContext,
    TeamContext,
    simulate_dst_draw,
    simulate_kicker_draw,
)
from src.projection.weekly_props.provenance import (
    decision_provenance,
    is_weekly_props_run,
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
        requested_source: ProjectionSource | str | None = None,
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
        # Client/query override mirrors APP_PROJECTION_SOURCE when omitted.
        if requested_source is None:
            self.requested_source = configured_projection_source()
        elif isinstance(requested_source, ProjectionSource):
            self.requested_source = requested_source
        else:
            self.requested_source = ProjectionSource.parse(str(requested_source))
        self.projection_source = self.requested_source
        self.source_fallback_reason: str | None = None
        self.run = self._resolve_run(week)
        self.projection_service = ProjectionService(session, season=self.season)
        self.bundle = get_bundle_loader(self.season)
        self.projection_run_id = self._projection_run_id()
        # Provenance must reflect the run decisions actually consume (including
        # weekly_props → sealed/status fallback), not a sealed-rescore default.
        self.projection_context = self.projection_service.context(
            requested_source=self.requested_source,
            league_id=league_id,
            week=week,
            effective_source=self.projection_source,
            projection_run=self.run,
            fallback_reason=self.source_fallback_reason,
        )
        self.draw_count = draw_count
        self._identity = PlayerIdentityResolver(session)
        self._props_name_index: dict[str, str] | None = None
        self._props_player_ids: set[str] | None = None

    def board_source(self) -> str:
        """UI flip seam: vegas_props vs league_value for the active decision path."""
        if self.projection_source == ProjectionSource.WEEKLY_PROPS:
            return "vegas_props"
        return "league_value"

    def _uses_weekly_props_run(self) -> bool:
        return (
            self.projection_source == ProjectionSource.WEEKLY_PROPS
            and self.run is not None
            and is_weekly_props_run(self.run)
        )

    def _ensure_props_indexes(self) -> None:
        """Index the active weekly_props run by player id and normalized name/team."""
        if self._props_name_index is not None and self._props_player_ids is not None:
            return
        name_index: dict[str, str] = {}
        player_ids: set[str] = set()
        if self._uses_weekly_props_run():
            assert self.run is not None
            for row in self.projections.player_projections(self.run.id):
                player_ids.add(row.player_id)
                mean = row.mean_json or {}
                raw_name = str(mean.get("name") or "")
                canon = canonicalize_player_name(raw_name)
                norm = normalize_name(raw_name)
                team = str(mean.get("team") or row.team or "").upper()
                for key in {canon, norm} - {""}:
                    name_index.setdefault(key, row.player_id)
                    if team:
                        name_index.setdefault(f"{key}|{team}", row.player_id)
        self._props_name_index = name_index
        self._props_player_ids = player_ids

    def _props_player_id_set(self) -> set[str]:
        self._ensure_props_indexes()
        assert self._props_player_ids is not None
        return self._props_player_ids

    def _match_props_by_name(
        self, *, name: str | None, team: str | None = None
    ) -> str | None:
        """Map a Sleeper identity name onto a weekly_props projection id."""
        if not name or not self._uses_weekly_props_run():
            return None
        self._ensure_props_indexes()
        assert self._props_name_index is not None
        canon = canonicalize_player_name(name)
        norm = normalize_name(name)
        team_key = str(team or "").upper()
        for key in (canon, norm):
            if not key:
                continue
            if team_key:
                hit = self._props_name_index.get(f"{key}|{team_key}")
                if hit:
                    return hit
            hit = self._props_name_index.get(key)
            if hit:
                return hit
        return None

    def _uses_weekly_db_run(self) -> bool:
        if self.requested_source == ProjectionSource.WEEKLY_PROPS:
            return True
        return (
            self.requested_source == ProjectionSource.WEEKLY_V2_RND
            and weekly_rnd_enabled()
        )

    def _resolve_preseason_run(self):
        status_run = self.projections.active_run(
            mode="preseason", season=self.season, week=None
        )
        # Prefer status-adjusted when available and configured/fallback.
        if status_run is not None and (
            self.requested_source
            in {
                ProjectionSource.STATUS_ADJUSTED_RELEASE,
                ProjectionSource.WEEKLY_PROPS,
            }
            or str(getattr(status_run, "model_version", "")).startswith("status")
        ):
            return status_run
        return status_run

    def _resolve_run(self, week: int | None):
        if self.requested_source == ProjectionSource.WEEKLY_PROPS and week is not None:
            run = self.projections.active_run(
                mode="weekly", season=self.season, week=week
            )
            if run is not None and is_weekly_props_run(run):
                self.projection_source = ProjectionSource.WEEKLY_PROPS
                self.source_fallback_reason = None
                return run
            # Missing week pointer: fall back with explicit provenance.
            fallback = self._resolve_preseason_run()
            if fallback is not None and str(
                getattr(fallback, "model_version", "")
            ).startswith("status"):
                self.projection_source = ProjectionSource.STATUS_ADJUSTED_RELEASE
            else:
                self.projection_source = ProjectionSource.SEALED_RELEASE
            self.source_fallback_reason = "missing_weekly_props_pointer"
            return fallback

        if (
            self.requested_source == ProjectionSource.WEEKLY_V2_RND
            and weekly_rnd_enabled()
            and week is not None
        ):
            run = self.projections.active_run(
                mode="weekly", season=self.season, week=week
            )
            if run is not None:
                self.projection_source = ProjectionSource.WEEKLY_V2_RND
                return run

        self.projection_source = (
            self.requested_source
            if self.requested_source
            != ProjectionSource.WEEKLY_V2_RND
            else ProjectionSource.SEALED_RELEASE
        )
        return self.projections.active_run(
            mode="preseason", season=self.season, week=None
        )

    def _projection_run_id(self) -> str | None:
        if self.projection_source in {
            ProjectionSource.SEALED_RELEASE,
            ProjectionSource.STATUS_ADJUSTED_RELEASE,
        }:
            bundle = self.bundle.load_bundle()
            if bundle is not None and self.run is None:
                return f"preseason-{bundle.namespace}"
        if self.run is not None:
            return self.run.id
        return None

    def resolve_player_id(self, raw_id: str) -> str | None:
        """Map a roster id (Sleeper or canonical) onto the projection id space.

        For ``weekly_props``, prefer ids that exist on the props run. When GSIS /
        sealed links are missing, fall back to name (+ team) against the run so
        Sleeper roster identity alone is enough for the primary Vegas view.
        """
        if not raw_id:
            return None
        key = str(raw_id)
        from src.app.availability.gsis_link import (
            build_release_gsis_index,
            match_gsis_for_identity,
            projection_id_for_identity,
        )
        from src.app.persistence.models import PlayerIdentity

        resolution = self._identity.resolve(player_id=key, sleeper_id=key)
        row: PlayerIdentity | None = None
        if resolution.status == "resolved" and resolution.player_id:
            row = self.leagues.session.get(PlayerIdentity, resolution.player_id)
        if row is None:
            row = (
                self.leagues.session.query(PlayerIdentity)
                .filter(
                    (PlayerIdentity.player_id == key) | (PlayerIdentity.sleeper_id == key)
                )
                .order_by(PlayerIdentity.player_id.asc())
                .first()
            )

        props_ids = self._props_player_id_set() if self._uses_weekly_props_run() else set()

        if row is not None:
            projected = projection_id_for_identity(row)
            # Vegas-primary: accept the identity projection id when it is on the
            # props run, even if the sealed bundle is cold.
            if props_ids:
                if projected in props_ids:
                    return projected
                if row.player_id in props_ids:
                    return row.player_id
                if row.gsis_id and row.gsis_id in props_ids:
                    return row.gsis_id
                by_name = self._match_props_by_name(name=row.name, team=row.team)
                if by_name:
                    return by_name
                # Special-teams still use identity ids (simulator path).
                if row.position in {"K", "DEF", "DST"}:
                    return row.player_id
                return None

            if self.bundle.get(projected) is not None or projected != row.player_id:
                # Prefer an explicit GSIS link even before the bundle is warm.
                if row.gsis_id or projected.startswith("00-"):
                    return projected
            if self.bundle.get(projected) is not None:
                return projected
            players = self.bundle.load() or {}
            by_name_pos, by_name_pos_team = build_release_gsis_index(players)
            matched = match_gsis_for_identity(
                row, by_name_pos=by_name_pos, by_name_pos_team=by_name_pos_team
            )
            if matched:
                if not row.gsis_id:
                    row.gsis_id = matched
                    self.leagues.session.flush()
                return matched
            # Special-teams identities often share the roster id space.
            if row.position in {"K", "DEF", "DST"}:
                return row.player_id
            return None

        if props_ids and key in props_ids:
            return key
        if self.bundle.get(key) is not None:
            return key
        return None

    def resolve_roster_ids(
        self, raw_ids: list[str]
    ) -> tuple[dict[str, str], list[str]]:
        """Resolve roster ids to canonical projection ids for ownership checks."""
        mapping: dict[str, str] = {}
        unresolved: list[str] = []
        for raw_id in raw_ids:
            if not raw_id:
                continue
            canonical = self.resolve_player_id(str(raw_id))
            if canonical is None:
                unresolved.append(str(raw_id))
            else:
                mapping[str(raw_id)] = canonical
        return mapping, unresolved

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

        # Vegas-primary view: do not silently fill gaps from sealed_release.
        if not self._uses_weekly_props_run():
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
        base = {
            "projection_run_id": self.projection_run_id,
            "projection_available": self.projection_run_id is not None,
            "contract_hash": self.contract.contract_hash,
            "league_type": self.league_type,
            "data_as_of": self.data_as_of(),
            "draw_count": draw_set.draw_count,
            "scoring_fidelity": self.projection_context.scoring_fidelity,
            "scoring_fidelity_note": draw_set.fidelity_note(),
            "unapplied_scoring_rules": draw_set.unapplied_rules,
            "players_without_projection": missing,
            "baseline_scoring": self.bundle.meta.get("scoring"),
            "recentred_uncertainty_players": draw_set.recentred_players,
            # Seam for League Value ↔ Vegas Props UI flip (in-season later).
            "board_source": self.board_source(),
        }
        base.update(self.projection_context.to_dict())
        base.update(
            decision_provenance(
                requested_source=self.requested_source.value,
                effective_source=self.projection_source.value,
                run_id=self.projection_run_id,
                season=self.season,
                week=self.week,
                fallback_reason=self.source_fallback_reason,
            )
        )
        return base


_USERNAME_TO_USER_ID_CACHE: dict[str, str | None] = {}


def _configured_sleeper_username(settings) -> str | None:
    username = (settings.sleeper_username or "").strip() or None
    if username:
        return username
    if settings.sleeper_owner_config or settings.sleeper_owner_json:
        try:
            from src.app.league.sleeper.owner_config import load_owner_config

            loaded = (load_owner_config().username or "").strip()
            return loaded or None
        except (FileNotFoundError, OSError, ValueError):
            return None
    return None


def _lookup_user_id_for_username(username: str, *, use_fixtures: bool) -> str | None:
    """Resolve a Sleeper username to user_id (cached; read-only GET)."""
    cache_key = username.strip().lower()
    if cache_key in _USERNAME_TO_USER_ID_CACHE:
        return _USERNAME_TO_USER_ID_CACHE[cache_key]
    try:
        from src.app.league.sleeper.client import SleeperClient

        payload = SleeperClient(use_fixtures=use_fixtures).get_user(username.strip())
        user_id = str(payload.get("user_id") or "").strip() or None
    except Exception:
        user_id = None
    _USERNAME_TO_USER_ID_CACHE[cache_key] = user_id
    return user_id


def _owner_user_id_candidates(settings) -> list[str]:
    """Ordered Sleeper user ids that may identify the configured owner.

    Prefer the configured ``SLEEPER_USER_ID``. When that is missing or does not
    match league memberships (quoted/stale id, etc.), also try resolving
    ``SLEEPER_USERNAME`` / owner-config username via the Sleeper user endpoint.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        if not value:
            return
        key = str(value).strip()
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(key)

    _add(settings.sleeper_user_id)
    username = _configured_sleeper_username(settings)
    if username:
        _add(_lookup_user_id_for_username(username, use_fixtures=settings.use_sleeper_fixtures))
    return candidates


def _members_for_owner_keys(session: Session, league_id: str, owner_keys: list[str]):
    if not owner_keys:
        return []
    return (
        session.query(LeagueMember)
        .filter(
            LeagueMember.league_id == league_id,
            LeagueMember.user_id.in_(owner_keys),
        )
        .order_by(LeagueMember.roster_id.asc())
        .all()
    )


def _resolve_owner_roster_id(
    session: Session, league_id: str, *, explicit_roster_id: int | None
) -> int:
    """Resolve the configured Sleeper owner to their league roster.

    Roster ids are league-specific and are not guaranteed to be ``1``. Explicit
    callers (such as shadow validation) remain authoritative; normal API and
    assistant requests use the stable Sleeper user id imported into
    ``league_member``.

    When ``SLEEPER_USER_ID`` does not match a membership, fall back to
    ``SLEEPER_USERNAME`` (or owner-config username) resolved through Sleeper so
    a quoted/stale id cannot strand an otherwise correct deployment.

    Production fails closed when the owner identity is missing or ambiguous so
    recommendations cannot silently target another manager. Fixture and local
    development retain the legacy roster-``1`` fallback only when no owner
    identity is configured.
    """
    if explicit_roster_id is not None:
        return explicit_roster_id

    from src.app.config import get_settings

    settings = get_settings()
    owner_keys = _owner_user_id_candidates(settings)
    username = _configured_sleeper_username(settings)

    members = _members_for_owner_keys(session, league_id, owner_keys)
    if not members and username:
        # Last-resort in-DB match: some syncs store the Sleeper username as the
        # member display name even when the configured user id is wrong.
        members = (
            session.query(LeagueMember)
            .filter(
                LeagueMember.league_id == league_id,
                LeagueMember.display_name.ilike(username),
            )
            .order_by(LeagueMember.roster_id.asc())
            .all()
        )

    if owner_keys or username:
        if not members:
            league_member_count = (
                session.query(LeagueMember)
                .filter(LeagueMember.league_id == league_id)
                .count()
            )
            if league_member_count == 0:
                # Historical seasons are often imported without users/rosters.
                # That is a sync/coverage gap, not a wrong SLEEPER_USER_ID.
                raise LeagueContextError(
                    f"league_membership_not_synced:league={league_id}"
                )
            raise LeagueContextError(f"owner_roster_not_found:league={league_id}")
        roster_ids = sorted({member.roster_id for member in members})
        if len(roster_ids) > 1:
            raise LeagueContextError(
                f"owner_roster_ambiguous:league={league_id},count={len(roster_ids)}"
            )
        return roster_ids[0]

    if settings.app_env == "production":
        raise LeagueContextError(f"owner_identity_unconfigured:league={league_id}")
    return 1



def _public_decision_error(exc: Exception) -> tuple[str, str]:
    """Map internal LeagueContextError reasons to safe public codes."""
    reason = str(exc)
    mapping = (
        (
            "owner_identity_unconfigured",
            "owner_roster_unavailable",
            "Owner roster is not configured for this deployment. Set SLEEPER_USER_ID and re-sync.",
        ),
        (
            "league_membership_not_synced",
            "league_membership_unavailable",
            "This league has no synced memberships. Select an active configured league or run sync.",
        ),
        (
            "owner_roster_not_found",
            "owner_roster_unavailable",
            "The configured owner is not a member of this league. Fix owner configuration and re-sync.",
        ),
        (
            "owner_roster_ambiguous",
            "owner_roster_unavailable",
            "The configured owner matches multiple rosters in this league. Resolve membership and re-sync.",
        ),
        (
            "no_roster_snapshot",
            "roster_snapshot_unavailable",
            "No roster snapshot is available for the selected week. Run sync or choose another week.",
        ),
        (
            "no_projected_players_on_roster",
            "identity_resolution_incomplete",
            "Roster players could not be linked to the active projection release. Run sync after identity repair.",
        ),
        (
            "waiver_ownership_incomplete",
            "identity_resolution_incomplete",
            "Some rostered players could not be linked to the projection release. Run sync after identity repair.",
        ),
        (
            "matchup_incomplete",
            "matchup_snapshot_incomplete",
            "Matchup pairing is incomplete for this week. Lineup optimization may still be available without win probability.",
        ),
        (
            "matchup_probability_gate_failed",
            "matchup_snapshot_incomplete",
            "Matchup win probability failed validation for this week.",
        ),
        (
            "missing_weekly_props_pointer",
            "weekly_props_unavailable",
            "Weekly Vegas props are not promoted for this week. Switch to League Value, or promote a weekly_props run (set WEEKLY_PROPS_SHADOW_ONLY=false and publish).",
        ),
        (
            "league_or_rules_not_found",
            "projection_release_unavailable",
            "League rules are missing. Run sync for this league.",
        ),
        (
            "unsupported_scoring",
            "projection_release_unavailable",
            "League scoring rules are not fully supported for recommendations.",
        ),
    )
    for prefix, code, message in mapping:
        if reason.startswith(prefix) or f":{prefix}" in reason or prefix in reason:
            return code, message
    return (
        "service_temporarily_unavailable",
        "Recommendation is unavailable for this league and week.",
    )


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
        user_roster_id: int | None = None,
        opponent_roster_id: int | None = None,
        projection_source: str | ProjectionSource | None = None,
        bypass_cache: bool = False,
    ) -> dict:
        if opponent_mode not in {"current", "optimized"}:
            raise LeagueContextError(f"invalid_opponent_mode:{opponent_mode}")
        board = self.get_or_compute_matchup_board(
            league_id,
            week,
            user_roster_id=user_roster_id,
            opponent_roster_id=opponent_roster_id,
            projection_source=projection_source,
            bypass_cache=bypass_cache,
        )
        return mode_payload_from_board(board, opponent_mode)

    def get_or_compute_matchup_board(
        self,
        league_id: str,
        week: int,
        *,
        user_roster_id: int | None = None,
        opponent_roster_id: int | None = None,
        projection_source: str | ProjectionSource | None = None,
        bypass_cache: bool = False,
    ) -> dict:
        """Compute-with-cache: both opponent modes + actual/optimized lineups."""
        ctx = _LeagueContext(
            self.session,
            league_id,
            week,
            requested_source=projection_source,
        )
        # Fail fast: uncached sealed fallback under a Vegas request is expensive
        # on Vercel and used to hang until Workbox reported opaque no-response.
        # Until a weekly_props run is promoted, ask the client to use League Value.
        if (
            ctx.requested_source == ProjectionSource.WEEKLY_PROPS
            and ctx.source_fallback_reason == "missing_weekly_props_pointer"
        ):
            raise LeagueContextError(
                "missing_weekly_props_pointer:no promoted weekly_props run for "
                f"season={ctx.season},week={week}"
            )
        user_roster_id = _resolve_owner_roster_id(
            self.session, league_id, explicit_roster_id=user_roster_id
        )
        rosters = self.leagues.latest_rosters(league_id, week)
        user_roster = next((r for r in rosters if r.roster_id == user_roster_id), None)
        if user_roster is None:
            raise LeagueContextError(
                f"no_roster_snapshot:league={league_id},roster={user_roster_id},week={week}"
            )

        matchup_incomplete = False
        matchup_degraded_reason: str | None = None
        if opponent_roster_id is None:
            paired_opponent_id = self._matchup_opponent_roster_id(
                league_id, week, user_roster_id
            )
            if paired_opponent_id is None:
                matchup_incomplete = True
                matchup_degraded_reason = "matchup_pairing_unavailable"
                opponent = None
            else:
                opponent = next(
                    (r for r in rosters if r.roster_id == paired_opponent_id), None
                )
                if opponent is None:
                    matchup_incomplete = True
                    matchup_degraded_reason = "matchup_opponent_roster_missing"
        else:
            opponent = next(
                (r for r in rosters if r.roster_id == opponent_roster_id), None
            )

        fingerprint = roster_input_fingerprint(
            owner_roster_id=user_roster_id,
            owner_snapshot_id=user_roster.id,
            opponent_roster_id=opponent.roster_id if opponent else None,
            opponent_snapshot_id=opponent.id if opponent else None,
            owner_starters=list(user_roster.starters or []),
            opponent_starters=list((opponent.starters if opponent else []) or []),
            owner_players=list(user_roster.players or []),
            opponent_players=list((opponent.players if opponent else []) or []),
        )
        # Cache by *requested* UI source so weekly_props→sealed fallback cannot
        # share / overwrite the league_value slot.
        requested_board_source = board_source_for_projection_source(
            ctx.requested_source.value
        )
        effective_board_source = ctx.board_source()
        cache_run_id = ctx.projection_run_id
        # Never serve/persist a board built under a source fallback — sticky
        # sealed numbers under a Vegas request are the accuracy bug Richard hit.
        can_cache = (
            not bypass_cache
            and not ctx.source_fallback_reason
            and cache_run_id
            and projection_run_exists(self.session, cache_run_id)
        )
        if can_cache:
            cached = load_matchup_board(
                self.session,
                league_id=league_id,
                week=week,
                board_source=requested_board_source,
                projection_run_id=cache_run_id,
                input_fingerprint=fingerprint,
            )
            if cached is not None:
                cached["_cache_hit"] = True
                return cached

        board = self._compute_matchup_board(
            ctx=ctx,
            week=week,
            user_roster=user_roster,
            opponent=opponent,
            matchup_incomplete=matchup_incomplete,
            matchup_degraded_reason=matchup_degraded_reason,
            input_fingerprint=fingerprint,
        )
        board["_cache_hit"] = False
        board["requested_board_source"] = requested_board_source
        if (
            not ctx.source_fallback_reason
            and cache_run_id
            and projection_run_exists(self.session, cache_run_id)
        ):
            stored = store_matchup_board(
                self.session,
                league_id=league_id,
                week=week,
                board_source=requested_board_source,
                projection_run_id=cache_run_id,
                roster_snapshot_id=user_roster.id,
                input_fingerprint=fingerprint,
                effective_board_source=effective_board_source,
                board={k: v for k, v in board.items() if not k.startswith("_")},
            )
            if stored is not None:
                board["decision_snapshot_id"] = stored.id
                board["cached_at"] = stored.created_at.isoformat()
        return board

    def _compute_matchup_board(
        self,
        *,
        ctx: _LeagueContext,
        week: int,
        user_roster,
        opponent,
        matchup_incomplete: bool,
        matchup_degraded_reason: str | None,
        input_fingerprint: str,
    ) -> dict:
        user_raw = [pid for pid in (user_roster.players or []) if pid]
        opp_raw = [pid for pid in ((opponent.players if opponent else []) or []) if pid]
        user_map, unresolved_user = ctx.resolve_roster_ids(user_raw)
        opp_map, _unresolved_opp = ctx.resolve_roster_ids(opp_raw)
        user_ids = list(user_map.values())
        opp_ids = list(opp_map.values())
        draw_set, missing = ctx.build_draws(user_ids + opp_ids)

        user_candidates = [pid for pid in user_ids if pid in draw_set.players]
        opp_candidates = [pid for pid in opp_ids if pid in draw_set.players]
        submitted_user = [
            user_map[str(pid)]
            for pid in (user_roster.starters or [])
            if pid and str(pid) in user_map and user_map[str(pid)] in draw_set.players
        ]
        submitted_opp = [
            opp_map[str(pid)]
            for pid in ((opponent.starters if opponent else []) or [])
            if pid and str(pid) in opp_map and opp_map[str(pid)] in draw_set.players
        ]

        if not user_candidates:
            detail = "no_projected_players_on_roster"
            if unresolved_user:
                detail = f"no_projected_players_on_roster:unresolved={len(unresolved_user)}"
            raise LeagueContextError(f"{detail}:league={ctx.league_id},week={week}")

        matchup_allowed = ctx.projection_service.matchup_win_probability_allowed(
            season=ctx.season, week=week, source=ctx.projection_source
        )
        if matchup_incomplete or not opp_candidates:
            matchup_allowed = False
            if matchup_degraded_reason is None:
                matchup_degraded_reason = "matchup_opponent_unprojected"

        # Prefer props-run / identity names over sealed bundle when on Vegas path
        # (#53), while still emitting Matchup board bench/opponent rows (#56).
        display_by_id: dict[str, tuple[str | None, str | None]] = {}
        board_ids = list(dict.fromkeys([*user_candidates, *opp_candidates]))
        if ctx._uses_weekly_props_run() and ctx.run is not None and board_ids:
            for row in ctx.projections.player_projections(ctx.run.id, board_ids):
                mean = row.mean_json or {}
                display_by_id[row.player_id] = (
                    str(mean.get("name") or "") or None,
                    mean.get("team") or row.team,
                )
        props_path = ctx._uses_weekly_props_run()

        def _player_row(
            pid: str, *, slot: str | None, on_bench: bool = False
        ) -> dict:
            player = draw_set.players[pid]
            summary = None if props_path else ctx.bundle.get(pid)
            props_name, props_team = display_by_id.get(pid, (None, None))
            return {
                "player_id": pid,
                "name": props_name or (summary.name if summary else None) or pid,
                "position": player.position,
                "team": props_team or (summary.team if summary else None),
                "slot": "BN" if on_bench else slot,
                "expected_points": round(player.mean, 3),
                "points_p10": round(player.percentile(0.1), 3),
                "points_p50": round(player.percentile(0.5), 3),
                "points_p90": round(player.percentile(0.9), 3),
                "availability_probability": round(player.availability_probability, 4),
                "draw_mode": player.mode,
                "locked": player.locked,
            }

        def _side_details(
            starter_ids: list[str],
            candidate_ids: list[str],
            assignments: dict[str, str],
        ) -> tuple[list[dict], list[dict], float]:
            starters = [
                _player_row(pid, slot=assignments.get(pid))
                for pid in starter_ids
                if pid in draw_set.players
            ]
            starter_set = set(starter_ids)
            bench = [
                _player_row(pid, slot=None, on_bench=True)
                for pid in candidate_ids
                if pid not in starter_set
            ]
            bench.sort(
                key=lambda row: float(row["expected_points"] or 0.0), reverse=True
            )
            totals = draw_set.totals_for(
                [pid for pid in starter_ids if pid in draw_set.players]
            )
            expected = float(totals.mean()) if totals.size else 0.0
            return starters, bench, round(expected, 4)

        # Owner actual = submitted Sleeper starters (slot labels for display).
        # Matchup "You" must mirror this set — not the win%-optimal lineup —
        # so swaps Apply against the same board Richard sees in Sleeper.
        # Seat labels follow Sleeper starter *order* (not re-optimized), or You
        # vs Opp rows scramble when the FE aligns by slot.
        owner_actual_lineup = optimize_lineup(
            draw_set,
            ctx.contract,
            candidate_ids=user_candidates,
            objective="points",
        )
        if submitted_user:
            owner_actual_ids = submitted_user
            owner_actual_assignments = assign_submitted_seat_labels(
                ctx.contract, submitted_user
            )
        else:
            owner_actual_ids = owner_actual_lineup.starters
            owner_actual_assignments = dict(owner_actual_lineup.assignments)
        owner_actual_starters, owner_actual_bench, owner_actual_points = _side_details(
            owner_actual_ids, user_candidates, owner_actual_assignments
        )

        by_opponent_mode: dict[str, dict] = {}
        owner_optimized_side: dict | None = None
        opponent_actual_side: dict | None = None
        opponent_optimized_side: dict | None = None

        for mode in ("current", "optimized"):
            evaluation = matchup_probabilities(
                draw_set,
                ctx.contract,
                user_candidate_ids=user_candidates,
                opponent_candidate_ids=opp_candidates,
                user_starters=submitted_user,
                opponent_mode=mode,
                opponent_submitted_starters=submitted_opp,
            )
            recommended = evaluation["recommended"]
            probs = evaluation["recommended_probabilities"]
            if matchup_allowed:
                gate = validate_matchup_probabilities(probs)
                if not gate.passed:
                    raise LeagueContextError(
                        f"matchup_probability_gate_failed:{gate.failures}"
                    )
            else:
                probs = {"win": None, "tie": None, "loss": None}

            opponent_totals = draw_set.totals_for(evaluation["opponent_starters"])
            swaps = swap_recommendations(
                draw_set,
                ctx.contract,
                current_starters=submitted_user,
                recommended=recommended,
                opponent_totals=opponent_totals,
            )

            # Primary board rows = actual Sleeper starters.
            starter_details = owner_actual_starters
            bench_details = owner_actual_bench
            rec_starter_details, rec_bench_details, rec_points = _side_details(
                recommended.starters, user_candidates, recommended.assignments
            )
            opp_assignments = evaluation.get("opponent_assignments") or {}
            opponent_starter_details, opponent_bench_details, _ = _side_details(
                evaluation["opponent_starters"], opp_candidates, opp_assignments
            )

            provenance = ctx.provenance(draw_set, missing)
            if matchup_degraded_reason:
                provenance = {
                    **provenance,
                    "matchup_degraded": True,
                    "matchup_degraded_reason": matchup_degraded_reason,
                }
            provenance = {**provenance, "decision_cache": "miss"}
            resolved_opp_roster_id = (
                opponent.roster_id if opponent is not None else None
            )
            current_probs = (
                evaluation["current_probabilities"] if matchup_allowed else None
            )
            # Win% on the payload matches the lineup shown in `starters` (actual).
            shown_probs = current_probs if current_probs is not None else probs
            payload = {
                "week": week,
                "opponent_mode": mode,
                "opponent_lineup_source": evaluation["opponent_lineup_source"],
                "recommended_starters": recommended.starters,
                "recommended_starter_details": rec_starter_details,
                "recommended_bench": rec_bench_details,
                "recommended_expected_points": rec_points,
                "recommended_slot_assignments": recommended.assignments,
                "recommended_win_probability": probs.get("win")
                if matchup_allowed
                else None,
                "starters": starter_details,
                "bench": bench_details,
                "slot_assignments": owner_actual_assignments,
                "unfilled_seats": recommended.unfilled_seats,
                "expected_points": owner_actual_points,
                "quantiles": {
                    k: round(v, 4) for k, v in recommended.quantiles.items()
                },
                "objective": recommended.objective,
                "matchup_probabilities": shown_probs
                if shown_probs is not None
                else {"win": None, "tie": None, "loss": None},
                "win_probability": (shown_probs or {}).get("win")
                if matchup_allowed
                else None,
                "matchup_win_probability_available": matchup_allowed,
                "capability_mode": ctx.projection_context.capability_mode,
                "board_source": ctx.board_source(),
                "requested_board_source": board_source_for_projection_source(
                    ctx.requested_source.value
                ),
                "current_lineup_probabilities": current_probs,
                "current_starters": evaluation["current_starters"],
                "current_expected_points": round(
                    evaluation["current_expected_points"], 4
                ),
                "opponent_roster_id": resolved_opp_roster_id,
                "opponent_starters": evaluation["opponent_starters"],
                "opponent_starter_details": opponent_starter_details,
                "opponent_bench": opponent_bench_details,
                "opponent_expected_points": round(
                    evaluation["opponent_expected_points"], 4
                ),
                "recommended_swaps": swaps,
                "swaps": swaps,
                **provenance,
                "meta": provenance,
            }
            by_opponent_mode[mode] = payload

            if owner_optimized_side is None:
                owner_optimized_side = {
                    "starters": rec_starter_details,
                    "bench": rec_bench_details,
                    "expected_points": rec_points,
                    "slot_assignments": recommended.assignments,
                }
            if mode == "current":
                opponent_actual_side = {
                    "starters": opponent_starter_details,
                    "bench": opponent_bench_details,
                    "expected_points": payload["opponent_expected_points"],
                    "lineup_source": evaluation["opponent_lineup_source"],
                }
            else:
                opponent_optimized_side = {
                    "starters": opponent_starter_details,
                    "bench": opponent_bench_details,
                    "expected_points": payload["opponent_expected_points"],
                    "lineup_source": evaluation["opponent_lineup_source"],
                }

        return {
            "week": week,
            "board_source": ctx.board_source(),
            "requested_board_source": board_source_for_projection_source(
                ctx.requested_source.value
            ),
            "projection_run_id": ctx.projection_run_id,
            "input_fingerprint": input_fingerprint,
            "owner_actual": {
                "starters": owner_actual_starters,
                "bench": owner_actual_bench,
                "expected_points": owner_actual_points,
                "slot_assignments": owner_actual_assignments,
            },
            "owner_optimized": owner_optimized_side,
            "opponent_actual": opponent_actual_side,
            "opponent_optimized": opponent_optimized_side,
            "by_opponent_mode": by_opponent_mode,
        }

    def _matchup_opponent_roster_id(
        self, league_id: str, week: int, user_roster_id: int
    ) -> int | None:
        """Resolve the opponent from the newest coherent weekly matchup pairing.

        Prefer a sync generation where owner and opponent share ``fetched_at``.
        Matchup upserts used to skip unchanged rows, so the owner's latest row
        can be days older than peers — walking older owner rows finds the last
        coherent pair before falling back to matchup_id-only matching.
        """
        owner_rows = (
            self.session.query(MatchupSnapshot)
            .filter(
                MatchupSnapshot.league_id == league_id,
                MatchupSnapshot.week == week,
                MatchupSnapshot.roster_id == user_roster_id,
                MatchupSnapshot.matchup_id > 0,
            )
            .order_by(
                MatchupSnapshot.fetched_at.desc(),
                MatchupSnapshot.id.desc(),
            )
            .all()
        )
        if not owner_rows:
            return None

        for owner in owner_rows:
            opponent = (
                self.session.query(MatchupSnapshot)
                .filter(
                    MatchupSnapshot.league_id == league_id,
                    MatchupSnapshot.week == week,
                    MatchupSnapshot.matchup_id == owner.matchup_id,
                    MatchupSnapshot.roster_id != user_roster_id,
                    MatchupSnapshot.fetched_at == owner.fetched_at,
                )
                .order_by(
                    MatchupSnapshot.fetched_at.desc(),
                    MatchupSnapshot.id.desc(),
                )
                .first()
            )
            if opponent is not None:
                return opponent.roster_id

        # Last resort: newest peer sharing the newest owner's matchup_id.
        owner = owner_rows[0]
        opponent = (
            self.session.query(MatchupSnapshot)
            .filter(
                MatchupSnapshot.league_id == league_id,
                MatchupSnapshot.week == week,
                MatchupSnapshot.matchup_id == owner.matchup_id,
                MatchupSnapshot.roster_id != user_roster_id,
            )
            .order_by(
                MatchupSnapshot.fetched_at.desc(),
                MatchupSnapshot.id.desc(),
            )
            .first()
        )
        return opponent.roster_id if opponent is not None else None


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
        user_roster_id: int | None = None,
    ) -> dict:
        ctx = _LeagueContext(self.session, league_id, week)
        user_roster_id = _resolve_owner_roster_id(
            self.session, league_id, explicit_roster_id=user_roster_id
        )
        rosters = self.leagues.latest_rosters(league_id, week)
        raw_rostered = [
            pid for roster in rosters for pid in (roster.players or []) if pid
        ]
        id_mapping, unresolved_rostered = ctx.resolve_roster_ids(raw_rostered)
        if unresolved_rostered:
            raise LeagueContextError(
                f"waiver_ownership_incomplete:{len(unresolved_rostered)}"
            )
        rostered = set(id_mapping.values())
        user_roster = next((r for r in rosters if r.roster_id == user_roster_id), None)
        user_ids = [
            id_mapping.get(str(pid), str(pid))
            for pid in ((user_roster.players if user_roster else []) or [])
            if pid
        ]

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
