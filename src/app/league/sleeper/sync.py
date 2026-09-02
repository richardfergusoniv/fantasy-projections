"""Sleeper sync — persist raw snapshots before normalization.

Every write path is idempotent: re-running a sync against unchanged upstream
content must not create a second row. Because the ORM models are owned
elsewhere, idempotency is enforced here by natural-key lookups and content
hashes rather than database constraints.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.app.availability.identity import PlayerIdentityResolver
from src.app.league.sleeper.client import SleeperClient, SleeperError
from src.app.persistence.models import (
    League,
    LeagueDraftRule,
    LeagueRuleSnapshot,
    LeagueTransaction,
    MatchupSnapshot,
    PlayerIdentity,
    RosterSnapshot,
    SourceSnapshot,
    TradedPick,
)
from src.app.persistence.repositories import LeagueRepository, SourceRepository
from src.app.scoring.compiler import compile_sleeper_scoring

logger = logging.getLogger(__name__)

MAX_SEASON_CHAIN_DEPTH = 10
MARKET_SIGNAL_ENDPOINT = "market_signal/sleeper/trending_add"
KNOWN_DRAFT_ORDER_RULES = {"max_pf", "reverse_standings"}


class SleeperSyncService:
    def __init__(self, session: Session, *, use_fixtures: bool = False, client: SleeperClient | None = None) -> None:
        self.session = session
        self.client = client or SleeperClient(use_fixtures=use_fixtures)
        self.leagues = LeagueRepository(session)
        self.sources = SourceRepository(session)
        self.identity = PlayerIdentityResolver(session)
        self._identity_cache: dict[str, str | None] = {}
        #: Raw Sleeper ids this sync could not map onto a canonical player.
        self.unresolved_player_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _record_source(self, meta: dict) -> SourceSnapshot:
        """Content-addressed source snapshot; identical payloads reuse one row."""

        existing = (
            self.session.query(SourceSnapshot)
            .filter(
                SourceSnapshot.endpoint == meta["endpoint"],
                SourceSnapshot.body_hash == meta["body_hash"],
                SourceSnapshot.health_verdict == meta["health_verdict"],
                SourceSnapshot.is_complete == bool(meta.get("is_complete", True)),
            )
            .first()
        )
        if existing is not None:
            return existing
        snapshot = SourceSnapshot(
            endpoint=meta["endpoint"],
            request_params_json=meta.get("request_params", {}),
            fetched_at=datetime.fromisoformat(meta["fetched_at"]),
            body_hash=meta["body_hash"],
            artifact_uri=meta["artifact_uri"],
            health_verdict=meta["health_verdict"],
            is_complete=meta.get("is_complete", True),
        )
        return self.sources.add_snapshot(snapshot)

    def _record_last_source(self) -> SourceSnapshot | None:
        """Record the snapshot the client already produced for its last fetch."""

        meta = self.client.last_snapshot_meta
        if not meta:
            return None
        return self._record_source(meta)

    # ------------------------------------------------------------------
    # Users and leagues
    # ------------------------------------------------------------------

    def connect_user(self, username: str) -> dict:
        user = self.client.get_user(username)
        self._record_last_source()
        return user

    def sync_leagues(self, user_id: str, season: int, *, include_history: bool = True) -> list[str]:
        payload = self.client.get_leagues(user_id, season)
        self._record_last_source()
        synced: list[str] = []
        for league_data in payload:
            league = self._upsert_league(league_data, season)
            synced.append(league.league_id)
            self._sync_league_details(league.league_id, int(league.season))
            if include_history:
                self.sync_season_history(league_data)
        return synced

    def _upsert_league(self, league_data: dict, season: int) -> League:
        league = self.leagues.upsert_league(
            league_id=str(league_data["league_id"]),
            season=int(league_data.get("season", season)),
            name=league_data.get("name", "League"),
            league_type="dynasty" if league_data.get("settings", {}).get("type") == 2 else "redraft",
            raw_json=league_data,
        )
        previous_id = league_data.get("previous_league_id")
        league.previous_league_id = str(previous_id) if previous_id else None
        status = league_data.get("status")
        if status:
            league.status = str(status)
        self.session.flush()
        return league

    def sync_season_history(self, league_data: dict) -> list[str]:
        """Walk `previous_league_id` backwards, bounded and cycle-safe.

        Sleeper chains occasionally point back at a league already seen; the
        visited set plus the depth bound guarantee termination either way.
        """

        head_id = str(league_data.get("league_id", ""))
        visited: set[str] = {head_id} if head_id else set()
        chain: list[str] = []
        previous_id = league_data.get("previous_league_id")
        depth = 0
        while previous_id:
            previous_id = str(previous_id)
            if previous_id in visited:
                logger.warning("sleeper_league_chain_cycle", extra={"depth": depth})
                break
            if depth >= MAX_SEASON_CHAIN_DEPTH:
                logger.warning("sleeper_league_chain_truncated", extra={"depth": depth})
                break
            visited.add(previous_id)
            try:
                payload = self.client.get_league(previous_id)
            except SleeperError as exc:
                logger.warning("sleeper_league_chain_fetch_failed", extra={"error": type(exc).__name__})
                break
            if not isinstance(payload, dict) or not payload.get("league_id"):
                break
            self._record_last_source()
            season = int(payload.get("season", 0) or 0)
            self._upsert_league(payload, season)
            chain.append(str(payload["league_id"]))
            depth += 1
            previous_id = payload.get("previous_league_id")
        return chain

    # ------------------------------------------------------------------
    # League details
    # ------------------------------------------------------------------

    def _sync_league_details(self, league_id: str, season: int) -> None:
        league_data = self.client.get_league(league_id)
        self._record_last_source()
        scoring = league_data.get("scoring_settings", {})
        roster_positions = league_data.get("roster_positions", [])
        contract = compile_sleeper_scoring(scoring, roster_positions)
        self._upsert_rule_snapshot(league_id, scoring, contract)

        users = self.client.get_users(league_id)
        self._record_last_source()
        rosters = self.client.get_rosters(league_id)
        self._record_last_source()
        self._link_members(league_id, users, rosters)

        nfl_state = self.client.get_nfl_state()
        self._record_last_source()
        week = int(nfl_state.get("week", 1) or 1)

        self._upsert_roster_snapshots(league_id, rosters, week)

        matchups = self.client.get_matchups(league_id, week)
        self._record_last_source()
        self._upsert_matchup_snapshots(league_id, matchups, week)

        self._sync_transactions(league_id, week)

        picks = self.client.get_traded_picks(league_id)
        self._record_last_source()
        self._upsert_traded_picks(league_id, picks, season)

        self.sync_drafts(league_id)
        self.sync_market_signals(league_id, week)

    def _upsert_rule_snapshot(self, league_id: str, scoring: dict, contract: Any) -> LeagueRuleSnapshot:
        existing = (
            self.session.query(LeagueRuleSnapshot)
            .filter(
                LeagueRuleSnapshot.league_id == league_id,
                LeagueRuleSnapshot.contract_hash == contract.contract_hash,
            )
            .first()
        )
        if existing is not None:
            return existing
        return self.leagues.add_rule_snapshot(
            LeagueRuleSnapshot(
                league_id=league_id,
                fetched_at=datetime.now(UTC),
                raw_json=scoring,
                normalized_json=contract.to_dict(),
                contract_hash=contract.contract_hash,
            )
        )

    def _link_members(self, league_id: str, users: list[dict], rosters: list[dict]) -> None:
        """Link by roster `owner_id`; `user.metadata.roster_id` is often absent."""

        users_by_id = {str(user.get("user_id")): user for user in users if user.get("user_id")}
        linked_rosters: set[int] = set()
        for roster in rosters:
            owner_id = roster.get("owner_id")
            roster_id = roster.get("roster_id")
            if owner_id is None or roster_id is None:
                continue
            user = users_by_id.get(str(owner_id), {})
            display_name = user.get("display_name") or user.get("username") or str(owner_id)
            self.leagues.upsert_member(
                league_id=league_id,
                user_id=str(owner_id),
                roster_id=int(roster_id),
                display_name=display_name,
            )
            linked_rosters.add(int(roster_id))
        for user in users:
            roster_id = (user.get("metadata") or {}).get("roster_id")
            if roster_id is None or int(roster_id) in linked_rosters:
                continue
            self.leagues.upsert_member(
                league_id=league_id,
                user_id=str(user["user_id"]),
                roster_id=int(roster_id),
                display_name=user.get("display_name") or user.get("username", "Manager"),
            )
            linked_rosters.add(int(roster_id))

    # ------------------------------------------------------------------
    # Player identity
    # ------------------------------------------------------------------

    def resolve_player_id(self, raw_id: Any) -> str | None:
        """Map one Sleeper player id onto this app's canonical player id.

        Sleeper rosters carry Sleeper's own player ids, while every projection,
        draw, and decision in this app is keyed by the canonical (GSIS) id.
        Persisting the raw id would leave every rostered player unprojectable,
        so resolution happens here at the ingest boundary. The unmodified
        payload is still retained verbatim in the content-addressed source
        snapshot, so provenance is not lost.

        Returns ``None`` when the id cannot be resolved; callers keep the raw id
        and record it as unresolved rather than guessing a player.
        """
        if raw_id in (None, "", "0", 0):
            return None
        key = str(raw_id)
        if key in self._identity_cache:
            return self._identity_cache[key]
        # `player_id` is tried first so an already-canonical id (a re-sync, or a
        # seeded roster) resolves to itself instead of being treated as unknown.
        resolution = self.identity.resolve(player_id=key, sleeper_id=key)
        resolved = resolution.player_id if resolution.status == "resolved" else None
        if resolution.status == "ambiguous":
            logger.warning(
                "sleeper_player_identity_ambiguous",
                extra={"candidate_count": len(resolution.candidates)},
            )
        self._identity_cache[key] = resolved
        return resolved

    def resolve_player_ids(self, raw_ids: list[Any] | None) -> list[str]:
        """Resolve a roster list, preserving order and dropping empty slots."""
        resolved: list[str] = []
        for raw_id in raw_ids or []:
            if raw_id in (None, "", "0", 0):
                continue
            key = str(raw_id)
            mapped = self.resolve_player_id(key)
            if mapped is None:
                # Keep the raw id so the roster stays complete and the player is
                # reported as unprojectable, instead of silently vanishing.
                self.unresolved_player_ids.add(key)
                resolved.append(key)
            else:
                resolved.append(mapped)
        return resolved

    def upsert_player_identities(self, players: dict[str, Any]) -> dict[str, int]:
        """Build the identity registry from the Sleeper player payload.

        Without this, `player_identity` is only ever populated by the local seed,
        so a live sync would have nothing to resolve Sleeper roster ids against.
        An existing canonical id is never rewritten: the row is matched by
        identifier and only enriched.
        """
        created = 0
        updated = 0
        seen_canonical: set[str] = set()
        for sleeper_id, payload in (players or {}).items():
            if not isinstance(payload, dict):
                continue
            sid = str(sleeper_id)
            gsis_id = payload.get("gsis_id")
            if gsis_id is not None:
                gsis_id = str(gsis_id).strip() or None
            resolution = self.identity.resolve(gsis_id=gsis_id, sleeper_id=sid)
            if resolution.status == "ambiguous":
                logger.warning(
                    "sleeper_identity_upsert_ambiguous",
                    extra={"candidate_count": len(resolution.candidates)},
                )
                continue
            name = payload.get("full_name") or payload.get("last_name") or sid
            position = str(payload.get("position") or "UNK")
            team = payload.get("team")
            row: PlayerIdentity | None = None
            if resolution.status == "resolved" and resolution.player_id:
                row = self.session.get(PlayerIdentity, resolution.player_id)
            if row is None and gsis_id:
                row = self.session.get(PlayerIdentity, gsis_id)
            if row is None:
                row = (
                    self.session.query(PlayerIdentity)
                    .filter(PlayerIdentity.sleeper_id == sid)
                    .one_or_none()
                )
            if row is None:
                canonical = str(gsis_id or sid).strip()
                if canonical in seen_canonical:
                    pending = self.session.get(PlayerIdentity, canonical)
                    if pending is not None:
                        row = pending
                if row is None:
                    row = PlayerIdentity(
                        player_id=canonical,
                        sleeper_id=sid,
                        gsis_id=gsis_id,
                        name=str(name),
                        position=position,
                        team=str(team) if team else None,
                    )
                    self.session.add(row)
                    seen_canonical.add(canonical)
                    created += 1
                else:
                    row.sleeper_id = sid
                    if gsis_id:
                        row.gsis_id = gsis_id
                    if name:
                        row.name = str(name)
                    row.position = position
                    row.team = str(team) if team else row.team
                    updated += 1
            else:
                row.sleeper_id = sid
                if gsis_id:
                    row.gsis_id = gsis_id
                if name:
                    row.name = str(name)
                row.position = position
                row.team = str(team) if team else row.team
                updated += 1
            self.session.flush()
        self.session.flush()
        self._identity_cache.clear()
        return {"identities_created": created, "identities_updated": updated}

    def _upsert_roster_snapshots(self, league_id: str, rosters: list[dict], week: int) -> int:
        inserted = 0
        for roster in rosters:
            roster_id = int(roster["roster_id"])
            players = self.resolve_player_ids(roster.get("players"))
            starters = self.resolve_player_ids(roster.get("starters"))
            reserve = self.resolve_player_ids(roster.get("reserve"))
            existing = (
                self.session.query(RosterSnapshot)
                .filter(
                    RosterSnapshot.league_id == league_id,
                    RosterSnapshot.week == week,
                    RosterSnapshot.roster_id == roster_id,
                )
                .all()
            )
            if any(
                (row.players or []) == players and (row.starters or []) == starters and (row.reserve or []) == reserve
                for row in existing
            ):
                continue
            self.leagues.add_roster_snapshot(
                RosterSnapshot(
                    league_id=league_id,
                    week=week,
                    roster_id=roster_id,
                    fetched_at=datetime.now(UTC),
                    players=players,
                    starters=starters,
                    reserve=reserve,
                )
            )
            inserted += 1
        return inserted

    def _upsert_matchup_snapshots(self, league_id: str, matchups: list[dict], week: int) -> int:
        inserted = 0
        for matchup in matchups:
            roster_id = int(matchup["roster_id"])
            matchup_id = int(matchup.get("matchup_id", 0) or 0)
            points = matchup.get("points")
            existing = (
                self.session.query(MatchupSnapshot)
                .filter(
                    MatchupSnapshot.league_id == league_id,
                    MatchupSnapshot.week == week,
                    MatchupSnapshot.roster_id == roster_id,
                    MatchupSnapshot.matchup_id == matchup_id,
                )
                .all()
            )
            if any(row.points == points for row in existing):
                continue
            self.session.add(
                MatchupSnapshot(
                    league_id=league_id,
                    week=week,
                    roster_id=roster_id,
                    matchup_id=matchup_id,
                    fetched_at=datetime.now(UTC),
                    points=points,
                )
            )
            inserted += 1
        self.session.flush()
        return inserted

    def _sync_transactions(self, league_id: str, week: int) -> None:
        for current_week in range(1, week + 1):
            txns = self.client.get_transactions(league_id, current_week)
            self._record_last_source()
            for txn in txns:
                if txn.get("type") != "trade" or txn.get("status") != "complete":
                    continue
                existing = (
                    self.session.query(LeagueTransaction)
                    .filter(LeagueTransaction.transaction_id == str(txn["transaction_id"]))
                    .one_or_none()
                )
                if existing is not None:
                    continue
                self.session.add(
                    LeagueTransaction(
                        league_id=league_id,
                        transaction_id=str(txn["transaction_id"]),
                        txn_type=str(txn.get("type", "trade")),
                        status=str(txn.get("status", "complete")),
                        created_at=datetime.fromtimestamp(txn.get("created", 0) / 1000, tz=UTC),
                        payload=txn,
                    )
                )
        self.session.flush()

    def _upsert_traded_picks(self, league_id: str, picks: list[dict], season: int) -> int:
        inserted = 0
        for pick in picks:
            row_season = int(pick.get("season", season) or season)
            round_number = int(pick.get("round", 1) or 1)
            original_roster_id = int(pick.get("roster_id", 0) or 0)
            owner_roster_id = int(pick.get("owner_id", 0) or 0)
            existing = (
                self.session.query(TradedPick)
                .filter(
                    TradedPick.league_id == league_id,
                    TradedPick.season == row_season,
                    TradedPick.round == round_number,
                    TradedPick.original_roster_id == original_roster_id,
                    TradedPick.owner_roster_id == owner_roster_id,
                )
                .first()
            )
            if existing is not None:
                continue
            self.session.add(
                TradedPick(
                    league_id=league_id,
                    season=row_season,
                    round=round_number,
                    original_roster_id=original_roster_id,
                    owner_roster_id=owner_roster_id,
                )
            )
            inserted += 1
        self.session.flush()
        return inserted

    # ------------------------------------------------------------------
    # Drafts and market signals
    # ------------------------------------------------------------------

    def sync_drafts(self, league_id: str) -> list[dict]:
        """Import drafts and any explicitly stated draft-order rule.

        Sleeper does not expose the dynasty rookie draft-order rule, so a rule is
        only recorded when the payload states one; it is never inferred, because
        a guess would silently override the manager's confirmed setting.
        """

        try:
            drafts = self.client.get_drafts(league_id)
        except SleeperError as exc:
            logger.warning("sleeper_drafts_unavailable", extra={"error": type(exc).__name__})
            return []
        self._record_last_source()
        summaries: list[dict] = []
        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            settings = draft.get("settings") or {}
            summaries.append(
                {
                    "draft_id": str(draft.get("draft_id", "")),
                    "type": draft.get("type"),
                    "status": draft.get("status"),
                    "season": draft.get("season"),
                    "rounds": settings.get("rounds"),
                    "reversal_round": settings.get("reversal_round"),
                    "teams": settings.get("teams"),
                    "has_draft_order": bool(draft.get("draft_order")),
                }
            )
            rule = str((draft.get("metadata") or {}).get("draft_order_rule", "")).strip().lower()
            if rule in KNOWN_DRAFT_ORDER_RULES:
                self._record_draft_rule(league_id, rule)
        return summaries

    def _record_draft_rule(self, league_id: str, rule: str) -> LeagueDraftRule:
        return self.persist_owner_confirmed_draft_rule(league_id, rule)

    def persist_owner_confirmed_draft_rule(self, league_id: str, rule: str) -> LeagueDraftRule:
        """Persist an owner-confirmed rookie-pick rule with auditable updates.

        Re-syncing the same rule is idempotent. A changed rule updates the row
        and stamps a fresh ``confirmed_at`` rather than appending a duplicate.
        """
        normalized = str(rule).strip().lower()
        if normalized not in KNOWN_DRAFT_ORDER_RULES:
            raise ValueError(f"unknown_draft_order_rule:{normalized}")
        existing = (
            self.session.query(LeagueDraftRule)
            .filter(LeagueDraftRule.league_id == league_id)
            .order_by(LeagueDraftRule.confirmed_at.desc())
            .first()
        )
        now = datetime.now(UTC)
        if existing is None:
            row = LeagueDraftRule(league_id=league_id, rule=normalized, confirmed_at=now)
            self.session.add(row)
            self.session.flush()
            return row
        if existing.rule != normalized:
            existing.rule = normalized
            existing.confirmed_at = now
            self.session.flush()
        return existing

    def sync_configured_leagues(
        self,
        user_id: str,
        season: int,
        allowed_league_ids: frozenset[str] | set[str],
        *,
        include_history: bool = True,
    ) -> list[str]:
        """Sync only explicitly allowed leagues; never import extras."""
        payload = self.client.get_leagues(user_id, season)
        self._record_last_source()
        allowed = {str(league_id) for league_id in allowed_league_ids}
        synced: list[str] = []
        for league_data in payload:
            league_id = str(league_data.get("league_id", ""))
            if league_id not in allowed:
                continue
            league = self._upsert_league(league_data, season)
            synced.append(league.league_id)
            self._sync_league_details(league.league_id, int(league.season))
            if include_history:
                self.sync_season_history(league_data)
        return synced

    def sync_market_signals(self, league_id: str, week: int) -> dict | None:
        """Store trending adds as a market/urgency signal.

        The payload is tagged `projection_input: false` and lives outside the
        projection tables: waiver urgency must never leak into a forecast.
        """

        try:
            trending = self.client.get_trending_players()
        except SleeperError as exc:
            logger.warning("sleeper_trending_unavailable", extra={"error": type(exc).__name__})
            return None
        self._record_last_source()
        payload = {
            "signal_type": "market_urgency",
            "source": "sleeper_trending_add",
            "projection_input": False,
            "league_id": league_id,
            "week": week,
            "players": [
                {
                    "sleeper_player_id": str(row.get("player_id")),
                    "add_count": int(row.get("count", 0) or 0),
                }
                for row in trending
                if isinstance(row, dict) and row.get("player_id")
            ],
        }
        meta = self.client.persist_snapshot(
            MARKET_SIGNAL_ENDPOINT,
            payload,
            {"league_id": league_id, "week": week},
        )
        self._record_source(meta)
        return payload

    def latest_market_signal(self, league_id: str | None = None) -> dict | None:
        """Most recent trending-add signal, scoped to a league when given.

        Trending adds are a per-league market read. With several leagues synced,
        an unscoped lookup would hand one league's urgency to another, so
        callers that care about a league must pass it.
        """
        if league_id is None:
            snapshot = self.sources.latest_for_endpoint(MARKET_SIGNAL_ENDPOINT)
            if snapshot is None:
                return None
            return self.client.store.get_json(snapshot.artifact_uri)
        rows = (
            self.session.query(SourceSnapshot)
            .filter(SourceSnapshot.endpoint == MARKET_SIGNAL_ENDPOINT)
            .order_by(SourceSnapshot.fetched_at.desc())
            .all()
        )
        for row in rows:
            if str((row.request_params_json or {}).get("league_id")) == str(league_id):
                return self.client.store.get_json(row.artifact_uri)
        return None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def sync_player_availability(self) -> dict:
        from src.app.availability.sync import AvailabilitySyncService

        payload = self.client.get_players_with_metadata()
        snapshot = self._record_last_source()
        if snapshot is None:
            meta = self.client.persist_snapshot(
                "players/nfl",
                payload.data,
                stale=payload.stale,
                fetched_at=payload.fetched_at,
            )
            snapshot = self._record_source(meta)
        # Register identities before availability so roster resolution during the
        # same refresh has a populated registry to resolve against.
        identities = self.upsert_player_identities(payload.data)
        result = AvailabilitySyncService(self.session).sync_from_players_payload(payload.data, snapshot)
        result.update(identities)
        result["payload_stale"] = payload.stale
        result["payload_fetched_at"] = payload.fetched_at.isoformat()
        return result
