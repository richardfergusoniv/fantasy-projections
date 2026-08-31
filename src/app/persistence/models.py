"""ORM models for the decision application."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    text,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.app.persistence.base import Base

#: Projection run identifiers are composed (``weekly-2026-w01-<hash>-inc-<hash>``)
#: and routinely exceed 36 characters. SQLite silently ignores VARCHAR limits but
#: PostgreSQL rejects the insert, so every run-id column uses this width.
RUN_ID_LEN = 128

_JSON_OBJECT_DEFAULT = text("'{}'")
_JSON_ARRAY_DEFAULT = text("'[]'")


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Timezone-aware UTC now for ``DateTime(timezone=True)`` column defaults."""
    return datetime.now(UTC)


class AppUser(Base):
    __tablename__ = "app_user"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


class MagicLinkToken(Base):
    __tablename__ = "magic_link_token"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


class SessionRecord(Base):
    __tablename__ = "session_record"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("app_user.id"), nullable=False)
    session_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    csrf_token: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


class SleeperAccount(Base):
    __tablename__ = "sleeper_account"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(128), nullable=False)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class League(Base):
    __tablename__ = "league"
    __table_args__ = (UniqueConstraint("league_id", "season", name="uq_league_season"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    league_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", server_default="active"
    )
    previous_league_id: Mapped[str | None] = mapped_column(String(64))
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT)


class LeagueDraftRule(Base):
    __tablename__ = "league_draft_rule"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rule: Mapped[str] = mapped_column(String(32), nullable=False)
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


class LeagueRuleSnapshot(Base):
    __tablename__ = "league_rule_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT)
    normalized_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    contract_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class LeagueMember(Base):
    __tablename__ = "league_member"
    __table_args__ = (UniqueConstraint("league_id", "roster_id", name="uq_league_roster"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)


class RosterSnapshot(Base):
    __tablename__ = "roster_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    players: Mapped[list] = mapped_column(JSON, default=list, server_default=_JSON_ARRAY_DEFAULT)
    starters: Mapped[list] = mapped_column(JSON, default=list, server_default=_JSON_ARRAY_DEFAULT)
    reserve: Mapped[list] = mapped_column(JSON, default=list, server_default=_JSON_ARRAY_DEFAULT)


class MatchupSnapshot(Base):
    __tablename__ = "matchup_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    matchup_id: Mapped[int] = mapped_column(Integer, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    points: Mapped[float | None] = mapped_column(Float)


class LeagueTransaction(Base):
    __tablename__ = "league_transaction"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    transaction_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    txn_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT)


class TradedPick(Base):
    __tablename__ = "traded_pick"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    original_roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_roster_id: Mapped[int] = mapped_column(Integer, nullable=False)


class PlayerIdentity(Base):
    __tablename__ = "player_identity"

    player_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sleeper_id: Mapped[str | None] = mapped_column(String(64), index=True)
    gsis_id: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    position: Mapped[str] = mapped_column(String(8), nullable=False)
    team: Mapped[str | None] = mapped_column(String(8))


class PlayerStatusSnapshot(Base):
    __tablename__ = "player_status_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str | None] = mapped_column(String(32))
    injury_status: Mapped[str | None] = mapped_column(String(32))
    practice: Mapped[str | None] = mapped_column(String(32))
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT)


class InjuryEvidence(Base):
    __tablename__ = "injury_evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_title: Mapped[str] = mapped_column(String(512), nullable=False)
    claim_json: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, server_default=text("0.5"))


class AvailabilityEvent(Base):
    __tablename__ = "availability_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    active_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_snapshot_id: Mapped[str | None] = mapped_column(String(36))
    evidence_ids: Mapped[list] = mapped_column(
        JSON, default=list, server_default=_JSON_ARRAY_DEFAULT
    )
    policy_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )


class DepthSnapshot(Base):
    __tablename__ = "depth_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ProjectionRun(Base):
    __tablename__ = "projection_run"

    id: Mapped[str] = mapped_column(String(RUN_ID_LEN), primary_key=True, default=_uuid)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int | None] = mapped_column(Integer)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="candidate", server_default="candidate"
    )
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    #: ``trained`` | ``fallback`` | ``fixture`` | ``derived`` — how the underlying
    #: model artifacts were sourced. Never ``NULL`` for runs written by this app.
    artifact_mode: Mapped[str | None] = mapped_column(
        String(32), default="derived", server_default="derived"
    )


class PlayerProjection(Base):
    __tablename__ = "player_projection"
    __table_args__ = (
        # One projection per player per run. Declared as a unique index rather
        # than a table constraint so the ORM and the Alembic migration produce
        # byte-identical DDL on both SQLite and PostgreSQL.
        Index("uq_player_projection_run_player", "run_id", "player_id", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("projection_run.id"), nullable=False, index=True)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    team: Mapped[str | None] = mapped_column(String(8))
    opponent: Mapped[str | None] = mapped_column(String(8))
    availability_probability: Mapped[float | None] = mapped_column(Float)
    mean_json: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT)
    quantiles_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )


class SimulationPartition(Base):
    __tablename__ = "simulation_partition"
    __table_args__ = (
        Index("uq_simulation_partition_run_key", "run_id", "partition_key", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("projection_run.id"), nullable=False, index=True)
    partition_key: Mapped[str] = mapped_column(String(128), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    draw_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ActiveProjectionPointer(Base):
    __tablename__ = "active_projection_pointer"
    __table_args__ = (
        UniqueConstraint("mode", "season", "week", name="uq_active_pointer"),
        # ``uq_active_pointer`` cannot constrain season-long horizons because
        # SQL treats NULL weeks as distinct. Both PostgreSQL and SQLite support
        # partial unique indexes, so a NULL-week partial index closes the hole
        # without introducing a sentinel week value that every ``week IS NULL``
        # query in the codebase would have to learn about.
        Index(
            "uq_active_pointer_season_long",
            "mode",
            "season",
            unique=True,
            sqlite_where=text("week IS NULL"),
            postgresql_where=text("week IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int | None] = mapped_column(Integer)
    run_id: Mapped[str] = mapped_column(ForeignKey("projection_run.id"), nullable=False)
    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    previous_run_id: Mapped[str | None] = mapped_column(ForeignKey("projection_run.id"))


class DecisionSnapshot(Base):
    __tablename__ = "decision_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    week: Mapped[int | None] = mapped_column(Integer)
    projection_run_id: Mapped[str] = mapped_column(
        ForeignKey("projection_run.id"), nullable=False
    )
    roster_snapshot_id: Mapped[str | None] = mapped_column(String(36))
    result_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


class ManagerState(Base):
    __tablename__ = "manager_state"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    probabilities_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    features_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    overridden_label: Mapped[str | None] = mapped_column(String(32))


class TradeProposal(Base):
    __tablename__ = "trade_proposal"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    created_by_roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    sides_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="offered", server_default="offered"
    )
    countered_by_id: Mapped[str | None] = mapped_column(String(36))


class TradeEvaluation(Base):
    __tablename__ = "trade_evaluation"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("trade_proposal.id"), nullable=False)
    projection_run_id: Mapped[str] = mapped_column(
        ForeignKey("projection_run.id"), nullable=False
    )
    objective_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    fairness_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    acceptance_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )


class ManagerTendency(Base):
    __tablename__ = "manager_tendency"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    league_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    roster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    features_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)


class JobRun(Base):
    __tablename__ = "job_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running", server_default="running"
    )
    attempt: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )


class SourceSnapshot(Base):
    __tablename__ = "source_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    endpoint: Mapped[str] = mapped_column(String(256), nullable=False)
    request_params_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    health_verdict: Mapped[str] = mapped_column(
        String(32), nullable=False, default="healthy", server_default="healthy"
    )
    is_complete: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())


class PromotionEvent(Base):
    __tablename__ = "promotion_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    candidate_run_id: Mapped[str] = mapped_column(
        ForeignKey("projection_run.id"), nullable=False
    )
    previous_run_id: Mapped[str | None] = mapped_column(ForeignKey("projection_run.id"))
    promoted: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    validation_json: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )


class AssistantAudit(Base):
    __tablename__ = "assistant_audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_class: Mapped[str] = mapped_column(String(64), nullable=False)
    tools_called: Mapped[list] = mapped_column(
        JSON, default=list, server_default=_JSON_ARRAY_DEFAULT
    )
    source_ids: Mapped[list] = mapped_column(
        JSON, default=list, server_default=_JSON_ARRAY_DEFAULT
    )
    model_id: Mapped[str | None] = mapped_column(String(64))
    token_usage: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=_JSON_OBJECT_DEFAULT
    )
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
