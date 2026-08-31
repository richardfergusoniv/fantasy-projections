"""Availability lifecycle — evidence intake, clearing rules, and pregame freeze."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from src.app.persistence.models import AvailabilityEvent, InjuryEvidence, SourceSnapshot
from src.app.persistence.repositories import AvailabilityRepository

logger = logging.getLogger(__name__)

FIXTURE_URL_SCHEME = "fixture"
WEB_URL_SCHEMES = {"http", "https"}

#: Hosts and suffixes reserved for documentation/testing. A citation pointing at
#: one of these is fabricated by definition and may only be stored when the claim
#: is explicitly flagged synthetic.
NON_CITABLE_HOSTS = {
    "example.com",
    "www.example.com",
    "example.net",
    "example.org",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
}
NON_CITABLE_SUFFIXES = (".example", ".invalid", ".test", ".local", ".localhost")

#: Evidence rows parked here failed identity resolution and are never applied to
#: a real player. Keyed off a reserved id because there is no quarantine table.
QUARANTINE_PLAYER_ID = "__quarantine__"

RESEARCH_MODE_LIVE = "live"
RESEARCH_MODE_FIXTURE = "fixture"

STATUS_PLAY_PROBABILITY = {
    "healthy": 1.0,
    "active": 1.0,
    "probable": 0.85,
    "questionable": 0.65,
    "doubtful": 0.25,
    "out": 0.0,
    "ir": 0.0,
    "pup": 0.0,
    "suspended": 0.0,
}

#: Endpoints trusted to clear an availability event, with the smallest record
#: count that is plausible for a complete payload from that endpoint.
PRIMARY_STATUS_ENDPOINTS = {"players/nfl": 10_000}

#: Bookkeeping events that must never influence the live availability policy.
NON_POLICY_EVENT_TYPES = {"pregame_freeze"}

CONTRADICTION_CONFIDENCE_PENALTY = 0.5
DEFAULT_SOURCE_RELIABILITY = 0.5


class EvidenceRejected(ValueError):
    """Base class for evidence that must not enter the lifecycle."""


class UncitedClaimError(EvidenceRejected):
    """A claim with no usable source URL."""


class FabricatedCitationError(EvidenceRejected):
    """A citation that is fabricated, or synthetic evidence posing as real."""


class AmbiguousPlayerError(EvidenceRejected):
    """A name matched more than one player identity."""


@dataclass
class EvidenceClaim:
    player_id: str
    status: str
    reported_injury: str | None
    expected_return_min: str | None
    expected_return_max: str | None
    claim_confidence: float
    sources: list[dict[str, str]]
    mode: str = RESEARCH_MODE_LIVE
    synthetic: bool = False
    publisher: str | None = None
    source_reliability: float | None = None
    published_at: str | None = None
    retrieved_at: str | None = None


@dataclass(frozen=True)
class ClearanceDecision:
    allowed: bool
    reason: str
    endpoint: str
    record_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "endpoint": self.endpoint,
            "record_count": self.record_count,
        }


@dataclass(frozen=True)
class PregameEvaluation:
    player_id: str
    season: int
    week: int
    kickoff_at: datetime
    play_probability: float
    evidence_ids: list[str]
    excluded_post_kickoff_evidence_ids: list[str]
    frozen: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "season": self.season,
            "week": self.week,
            "kickoff_at": self.kickoff_at.isoformat(),
            "play_probability": self.play_probability,
            "evidence_ids": list(self.evidence_ids),
            "excluded_post_kickoff_evidence_ids": list(self.excluded_post_kickoff_evidence_ids),
            "frozen": self.frozen,
        }


@dataclass(frozen=True)
class EvidenceSubmission:
    status: str
    evidence: InjuryEvidence | None
    player_id: str | None
    reason: str | None = None
    candidates: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return self.status == "applied"


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _normalized_status(status: str | None) -> str | None:
    if not status:
        return None
    return str(status).strip().lower().replace(" ", "_")


def is_non_citable_url(url: str) -> bool:
    """True when the URL can only be a placeholder, never a real report."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return True
    if host in NON_CITABLE_HOSTS:
        return True
    return host.endswith(NON_CITABLE_SUFFIXES)


class AvailabilityService:
    MIN_HEALTHY_PLAYER_PAYLOAD = PRIMARY_STATUS_ENDPOINTS["players/nfl"]

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = AvailabilityRepository(session)

    # ------------------------------------------------------------------
    # Source health and clearing
    # ------------------------------------------------------------------

    def clearance_check(
        self,
        snapshot: SourceSnapshot,
        *,
        record_count: int | None = None,
    ) -> ClearanceDecision:
        endpoint = (snapshot.endpoint or "").strip("/")
        if snapshot.health_verdict != "healthy":
            return ClearanceDecision(False, f"source_{snapshot.health_verdict or 'unknown'}", endpoint, record_count)
        if not snapshot.is_complete:
            return ClearanceDecision(False, "incomplete_payload", endpoint, record_count)
        if endpoint not in PRIMARY_STATUS_ENDPOINTS:
            return ClearanceDecision(False, "unrecognized_primary_source", endpoint, record_count)
        count = record_count
        if count is None:
            params = snapshot.request_params_json or {}
            raw_count = params.get("record_count")
            count = int(raw_count) if isinstance(raw_count, (int, float, str)) and str(raw_count).isdigit() else None
        if count is None:
            return ClearanceDecision(False, "record_count_unknown", endpoint, None)
        minimum = PRIMARY_STATUS_ENDPOINTS[endpoint]
        if count < minimum:
            return ClearanceDecision(False, "implausibly_small_payload", endpoint, count)
        return ClearanceDecision(True, "healthy_primary_source", endpoint, count)

    def validate_source_health(self, snapshot: SourceSnapshot, *, record_count: int | None = None) -> bool:
        return self.clearance_check(snapshot, record_count=record_count).allowed

    def can_clear_on_snapshot(self, snapshot: SourceSnapshot, player_count: int | None = None) -> bool:
        return self.clearance_check(snapshot, record_count=player_count).allowed

    def try_clear_for_player(
        self,
        player_id: str,
        snapshot: SourceSnapshot,
        *,
        player_count: int | None = None,
    ) -> int:
        decision = self.clearance_check(snapshot, record_count=player_count)
        if not decision.allowed:
            logger.debug("clear_blocked", extra={"reason": decision.reason, "endpoint": decision.endpoint})
            return 0
        cleared = 0
        for event in self._policy_events(player_id):
            self.repo.clear_event(event.id)
            cleared += 1
        return cleared

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def activate_event(
        self,
        *,
        player_id: str,
        event_type: str,
        source_snapshot_id: str | None,
        evidence_ids: list[str],
        policy: dict[str, Any],
        active_from: datetime | None = None,
    ) -> AvailabilityEvent:
        event = AvailabilityEvent(
            player_id=player_id,
            event_type=event_type,
            active_from=_as_utc(active_from) or datetime.now(UTC),
            source_snapshot_id=source_snapshot_id,
            evidence_ids=evidence_ids,
            policy_json=policy,
        )
        return self.repo.add_event(event)

    def _policy_events(self, player_id: str | None = None) -> list[AvailabilityEvent]:
        return [
            event
            for event in self.repo.active_events(player_id=player_id)
            if event.event_type not in NON_POLICY_EVENT_TYPES
        ]

    def active_policy_for_player(self, player_id: str) -> dict[str, Any]:
        events = self._policy_events(player_id)
        if not events:
            return {"play_probability": 1.0, "active_events": []}
        lowest = min(events, key=lambda e: e.policy_json.get("play_probability", 1.0))
        return {
            "play_probability": lowest.policy_json.get("play_probability", 0.5),
            "active_events": [e.id for e in events],
        }

    # ------------------------------------------------------------------
    # Evidence intake
    # ------------------------------------------------------------------

    def add_evidence(self, claim: EvidenceClaim, *, kickoff_at: datetime | None = None) -> InjuryEvidence:
        self._validate_claim(claim)
        source = claim.sources[0]
        source_url = source["url"]
        published_at = parse_timestamp(claim.published_at or source.get("published_at"))
        ingested_at = datetime.now(UTC)
        # When the source was read can predate when we wrote the row.
        retrieved_at = parse_timestamp(claim.retrieved_at) or ingested_at
        kickoff_at = _as_utc(kickoff_at)
        signature = self._claim_signature(claim, source_url)

        duplicate = self._find_duplicate(claim.player_id, source_url, signature)
        if duplicate is not None:
            return duplicate

        basis = self.evidence_basis(claim.player_id)
        superseded = bool(published_at and basis and published_at < basis)
        contradictions = [] if superseded else self._contradicting_evidence(claim.player_id, claim.status)
        confidence = float(claim.claim_confidence)
        if contradictions:
            confidence = round(confidence * CONTRADICTION_CONFIDENCE_PENALTY, 6)

        reliability = claim.source_reliability
        if reliability is None:
            reliability = 0.0 if claim.synthetic else DEFAULT_SOURCE_RELIABILITY

        row = InjuryEvidence(
            player_id=claim.player_id,
            published_at=published_at,
            fetched_at=ingested_at,
            source_url=source_url,
            source_title=source.get("title", "Injury report"),
            claim_json={
                "status": claim.status,
                "normalized_status": _normalized_status(claim.status),
                "reported_injury": claim.reported_injury,
                "expected_return_min": claim.expected_return_min,
                "expected_return_max": claim.expected_return_max,
                "claim_confidence": float(claim.claim_confidence),
                "effective_confidence": confidence,
                "sources": claim.sources,
                "mode": claim.mode,
                "synthetic": bool(claim.synthetic),
                "publisher": claim.publisher or self._publisher_from_url(source_url),
                "source_reliability": reliability,
                "published_at": published_at.isoformat() if published_at else None,
                "retrieved_at": retrieved_at.isoformat(),
                "signature": signature,
                "applied": not superseded,
                "superseded_by_newer_basis": superseded,
                "basis_at": basis.isoformat() if basis else None,
                "post_kickoff": bool(kickoff_at and published_at and published_at > kickoff_at),
                "kickoff_at": kickoff_at.isoformat() if kickoff_at else None,
                "contradicts": [other.id for other in contradictions],
            },
            confidence=confidence,
        )
        self.session.add(row)
        self.session.flush()
        for other in contradictions:
            self._mark_contradicted(other, row.id, confidence_penalty=CONTRADICTION_CONFIDENCE_PENALTY)
        self.session.flush()
        return row

    def submit_named_evidence(
        self,
        claim: EvidenceClaim,
        *,
        name: str,
        team: str | None = None,
        position: str | None = None,
        kickoff_at: datetime | None = None,
    ) -> EvidenceSubmission:
        """Attach evidence found by player name only after identity resolves uniquely."""

        from src.app.availability.identity import PlayerIdentityResolver

        resolution = PlayerIdentityResolver(self.session).resolve(name=name, team=team, position=position)
        if resolution.status != "resolved":
            quarantined = self._quarantine(claim, name=name, team=team, position=position, resolution=resolution)
            return EvidenceSubmission(
                status=f"quarantined_{resolution.status}",
                evidence=quarantined,
                player_id=None,
                reason=resolution.reason,
                candidates=list(resolution.candidates),
            )
        claim.player_id = resolution.player_id or claim.player_id
        evidence = self.add_evidence(claim, kickoff_at=kickoff_at)
        return EvidenceSubmission(status="applied", evidence=evidence, player_id=claim.player_id)

    def _quarantine(
        self,
        claim: EvidenceClaim,
        *,
        name: str,
        team: str | None,
        position: str | None,
        resolution: Any,
    ) -> InjuryEvidence:
        source = claim.sources[0] if claim.sources else {"url": f"{FIXTURE_URL_SCHEME}://unresolved", "title": "unresolved"}
        row = InjuryEvidence(
            player_id=QUARANTINE_PLAYER_ID,
            published_at=parse_timestamp(claim.published_at or source.get("published_at")),
            fetched_at=datetime.now(UTC),
            source_url=source.get("url", f"{FIXTURE_URL_SCHEME}://unresolved"),
            source_title=source.get("title", "Injury report"),
            claim_json={
                "quarantine": {
                    "reason": resolution.reason,
                    "status": resolution.status,
                    "queried_name": name,
                    "queried_team": team,
                    "queried_position": position,
                    "candidate_player_ids": list(resolution.candidates),
                },
                "status": claim.status,
                "applied": False,
                "mode": claim.mode,
                "synthetic": bool(claim.synthetic),
            },
            confidence=0.0,
        )
        self.session.add(row)
        self.session.flush()
        logger.warning(
            "evidence_quarantined",
            extra={"reason": resolution.reason, "candidate_count": len(resolution.candidates)},
        )
        return row

    def _validate_claim(self, claim: EvidenceClaim) -> None:
        asserts_return = bool(claim.expected_return_min or claim.expected_return_max)
        usable = [source for source in claim.sources or [] if source.get("url")]
        if not usable:
            if asserts_return:
                raise UncitedClaimError("Uncited return-date claims are rejected")
            raise UncitedClaimError("Evidence source must include url")
        if len(usable) != len(claim.sources or []):
            raise UncitedClaimError("Evidence source must include url")
        for source in usable:
            self._validate_citation(source["url"], synthetic=bool(claim.synthetic), mode=claim.mode)

    def _validate_citation(self, url: str, *, synthetic: bool, mode: str) -> None:
        scheme = urlparse(url).scheme.lower()
        if synthetic or mode == RESEARCH_MODE_FIXTURE:
            if scheme != FIXTURE_URL_SCHEME:
                raise FabricatedCitationError(
                    "Synthetic evidence must use a fixture:// citation so it cannot be mistaken for reporting"
                )
            if not synthetic or mode != RESEARCH_MODE_FIXTURE:
                raise FabricatedCitationError("Synthetic evidence must set both synthetic=True and mode='fixture'")
            return
        if scheme == FIXTURE_URL_SCHEME:
            raise FabricatedCitationError("fixture:// citations require synthetic=True and mode='fixture'")
        if scheme not in WEB_URL_SCHEMES:
            raise FabricatedCitationError(f"Unsupported citation scheme: {scheme or 'none'}")
        if is_non_citable_url(url):
            raise FabricatedCitationError("Fabricated citation host rejected for non-synthetic evidence")

    def _claim_signature(self, claim: EvidenceClaim, source_url: str) -> str:
        payload = json.dumps(
            {
                "player_id": claim.player_id,
                "status": _normalized_status(claim.status),
                "reported_injury": claim.reported_injury,
                "expected_return_min": claim.expected_return_min,
                "expected_return_max": claim.expected_return_max,
                "source_url": source_url,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _find_duplicate(self, player_id: str, source_url: str, signature: str) -> InjuryEvidence | None:
        rows = (
            self.session.query(InjuryEvidence)
            .filter(InjuryEvidence.player_id == player_id, InjuryEvidence.source_url == source_url)
            .all()
        )
        for row in rows:
            if (row.claim_json or {}).get("signature") == signature:
                return row
        return None

    def _contradicting_evidence(self, player_id: str, status: str) -> list[InjuryEvidence]:
        normalized = _normalized_status(status)
        contradicting = []
        for row in self.applied_evidence(player_id):
            other = (row.claim_json or {}).get("normalized_status")
            if other and normalized and other != normalized:
                contradicting.append(row)
        return contradicting

    def _mark_contradicted(self, row: InjuryEvidence, other_id: str, *, confidence_penalty: float) -> None:
        claim = dict(row.claim_json or {})
        contradicted = list(claim.get("contradicted_by", []))
        if other_id not in contradicted:
            contradicted.append(other_id)
        claim["contradicted_by"] = contradicted
        claim["effective_confidence"] = round(float(row.confidence) * confidence_penalty, 6)
        row.claim_json = claim
        row.confidence = claim["effective_confidence"]

    @staticmethod
    def _publisher_from_url(url: str) -> str:
        host = (urlparse(url).hostname or "").lower()
        return host.removeprefix("www.") or "unknown"

    # ------------------------------------------------------------------
    # Evidence reads and derived availability
    # ------------------------------------------------------------------

    def applied_evidence(self, player_id: str) -> list[InjuryEvidence]:
        return [
            row
            for row in self.repo.evidence_for_player(player_id)
            if (row.claim_json or {}).get("applied", True)
        ]

    def evidence_basis(self, player_id: str) -> datetime | None:
        """Newest timestamp that later evidence must beat to take effect."""

        candidates: list[datetime] = []
        for event in self._policy_events(player_id):
            active_from = _as_utc(event.active_from)
            if active_from:
                candidates.append(active_from)
        for row in self.applied_evidence(player_id):
            published = _as_utc(row.published_at)
            if published:
                candidates.append(published)
        return max(candidates) if candidates else None

    def evidence_probability(
        self,
        player_id: str,
        *,
        published_before: datetime | None = None,
    ) -> tuple[float | None, list[str]]:
        published_before = _as_utc(published_before)
        weighted = 0.0
        total_weight = 0.0
        used: list[str] = []
        for row in self.applied_evidence(player_id):
            claim = row.claim_json or {}
            status = claim.get("normalized_status") or _normalized_status(claim.get("status"))
            if status not in STATUS_PLAY_PROBABILITY:
                continue
            published = _as_utc(row.published_at)
            if published_before is not None and (published is None or published > published_before):
                continue
            reliability = claim.get("source_reliability")
            reliability = DEFAULT_SOURCE_RELIABILITY if reliability is None else float(reliability)
            weight = max(float(row.confidence), 0.01) * max(reliability, 0.01)
            weighted += weight * STATUS_PLAY_PROBABILITY[status]
            total_weight += weight
            used.append(row.id)
        if total_weight == 0.0:
            return None, []
        return round(weighted / total_weight, 6), used

    # ------------------------------------------------------------------
    # Pregame freeze
    # ------------------------------------------------------------------

    def evaluate_pregame(
        self,
        player_id: str,
        *,
        season: int,
        week: int,
        kickoff_at: datetime,
    ) -> PregameEvaluation:
        """Availability for a scored week, using only evidence published before kickoff."""

        kickoff = _as_utc(kickoff_at)
        assert kickoff is not None
        frozen = self.frozen_pregame_evaluation(player_id, season=season, week=week)
        if frozen is not None:
            return frozen

        probability = 1.0
        events = [
            event
            for event in self._policy_events(player_id)
            if (_as_utc(event.active_from) or kickoff) <= kickoff
        ]
        if events:
            probability = min(float(e.policy_json.get("play_probability", 1.0)) for e in events)
        evidence_probability, used = self.evidence_probability(player_id, published_before=kickoff)
        if evidence_probability is not None:
            probability = min(probability, evidence_probability) if events else evidence_probability

        excluded = [
            row.id
            for row in self.repo.evidence_for_player(player_id)
            if (_as_utc(row.published_at) or kickoff) > kickoff
        ]
        return PregameEvaluation(
            player_id=player_id,
            season=season,
            week=week,
            kickoff_at=kickoff,
            play_probability=round(probability, 6),
            evidence_ids=used,
            excluded_post_kickoff_evidence_ids=excluded,
            frozen=False,
        )

    def freeze_pregame_evaluation(
        self,
        player_id: str,
        *,
        season: int,
        week: int,
        kickoff_at: datetime,
    ) -> PregameEvaluation:
        evaluation = self.evaluate_pregame(player_id, season=season, week=week, kickoff_at=kickoff_at)
        if evaluation.frozen:
            return evaluation
        frozen = PregameEvaluation(
            player_id=evaluation.player_id,
            season=evaluation.season,
            week=evaluation.week,
            kickoff_at=evaluation.kickoff_at,
            play_probability=evaluation.play_probability,
            evidence_ids=evaluation.evidence_ids,
            excluded_post_kickoff_evidence_ids=evaluation.excluded_post_kickoff_evidence_ids,
            frozen=True,
        )
        key = self._freeze_key(season, week)
        events = self._policy_events(player_id)
        if not events:
            # Nothing to hang the freeze on; the evaluation stays reproducible from
            # the kickoff filter alone.
            return frozen
        for event in events:
            policy = dict(event.policy_json or {})
            frozen_map = dict(policy.get("frozen_pregame", {}))
            frozen_map[key] = frozen.to_dict()
            policy["frozen_pregame"] = frozen_map
            event.policy_json = policy
        self.session.flush()
        return frozen

    def frozen_pregame_evaluation(
        self,
        player_id: str,
        *,
        season: int,
        week: int,
    ) -> PregameEvaluation | None:
        key = self._freeze_key(season, week)
        for event in self.repo.active_events(player_id=player_id):
            stored = (event.policy_json or {}).get("frozen_pregame", {}).get(key)
            if not stored:
                continue
            kickoff = parse_timestamp(stored.get("kickoff_at"))
            if kickoff is None:
                continue
            return PregameEvaluation(
                player_id=player_id,
                season=season,
                week=week,
                kickoff_at=kickoff,
                play_probability=float(stored.get("play_probability", 1.0)),
                evidence_ids=list(stored.get("evidence_ids", [])),
                excluded_post_kickoff_evidence_ids=list(stored.get("excluded_post_kickoff_evidence_ids", [])),
                frozen=True,
            )
        return None

    def rest_of_season_probability(self, player_id: str) -> float:
        """Forward-looking availability; unlike pregame it uses all applied evidence."""

        probability, _ = self.evidence_probability(player_id)
        events = self._policy_events(player_id)
        event_probability = (
            min(float(e.policy_json.get("play_probability", 1.0)) for e in events) if events else 1.0
        )
        if probability is None:
            return event_probability
        return round(min(event_probability, probability), 6)

    @staticmethod
    def _freeze_key(season: int, week: int) -> str:
        return f"{season}-{week}"


def summarize_evidence(rows: Iterable[InjuryEvidence]) -> list[dict[str, Any]]:
    """Consumer-facing view that always exposes provenance and synthetic status."""

    summary = []
    for row in rows:
        claim = row.claim_json or {}
        summary.append(
            {
                "id": row.id,
                "player_id": row.player_id,
                "source_url": row.source_url,
                "source_title": row.source_title,
                "publisher": claim.get("publisher"),
                "published_at": row.published_at.isoformat() if row.published_at else None,
                "retrieved_at": claim.get("retrieved_at"),
                "mode": claim.get("mode"),
                "synthetic": bool(claim.get("synthetic", False)),
                "applied": bool(claim.get("applied", True)),
                "confidence": row.confidence,
                "source_reliability": claim.get("source_reliability"),
                "contradicts": claim.get("contradicts", []),
                "contradicted_by": claim.get("contradicted_by", []),
            }
        )
    return summary
