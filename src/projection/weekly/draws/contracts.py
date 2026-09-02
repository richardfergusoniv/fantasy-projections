"""Versioned forecast grains and probability contracts for weekly mixtures.

Authoritative event ownership
-----------------------------
``play_prob`` is the sole pre-kickoff availability probability for
``P(active_for_game)`` when derived from injury/status evidence. It must not
also silently multiply already-unconditional usage shares unless a caller
explicitly documents a residual haircut that is mathematically required after
the active event has been sampled.

Mixture identity (hand-checked in tests)::

    E[stat] = P(active) * P(participates | active) * E[stat | participates]

Bye / no-game rows are not participation failures. Locked actuals replace the
predictive mixture for already-played players without resampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping

CONTRACT_VERSION = "weekly_mixture_contract_v1"

ZeroDefinition = Literal[
    "true_dnp",
    "active_zero_usage",
    "bye_no_game",
    "preseason_roster_churn",
    "missing_data",
    "locked_actual_zero",
]


class EventLayer(str, Enum):
    ACTIVE_FOR_GAME = "active_for_game"
    OFFENSIVE_PARTICIPATION = "offensive_participation"
    POSITIVE_USAGE = "positive_usage"
    CONDITIONAL_USAGE = "conditional_usage"
    CONDITIONAL_EFFICIENCY = "conditional_efficiency"


class DrawModeLabel(str, Enum):
    """Visible draw-source labels for API/UI/ops (Phase 10)."""

    LEGACY_POINTS_INDEPENDENT = "legacy_points_independent"
    LEGACY_SCALED_COMPONENTS = "legacy_scaled_components"
    JOINT_STAT_MIXTURE_CANDIDATE = "joint_stat_mixture_candidate"
    JOINT_STAT_MIXTURE_VALIDATED = "joint_stat_mixture_validated"


@dataclass(frozen=True)
class EventProbabilitySpec:
    """One discrete event in the mixture."""

    layer: EventLayer
    name: str
    owner: str
    grain: str
    source_timestamp: str
    definition: str
    hard_rules: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ForecastGrainContract:
    """Versioned contract distinguishing availability, participation, and draws."""

    version: str = CONTRACT_VERSION
    play_prob_authority: str = (
        "play_prob is the sole pre-kickoff P(active_for_game) from injury/status; "
        "do not also multiply unconditional shares by play_prob after sampling active."
    )
    zero_definitions: tuple[ZeroDefinition, ...] = (
        "true_dnp",
        "active_zero_usage",
        "bye_no_game",
        "preseason_roster_churn",
        "missing_data",
        "locked_actual_zero",
    )
    events: tuple[EventProbabilitySpec, ...] = field(
        default_factory=lambda: (
            EventProbabilitySpec(
                layer=EventLayer.ACTIVE_FOR_GAME,
                name="p_active",
                owner="play_prob",
                grain="player-week",
                source_timestamp="pre-kickoff injury/status cutoff",
                definition="Probability the player is active for the scheduled game.",
                hard_rules=("bye_no_game -> 0", "locked inactive -> 0", "locked active -> 1"),
            ),
            EventProbabilitySpec(
                layer=EventLayer.OFFENSIVE_PARTICIPATION,
                name="p_participates",
                owner="event_model.offensive_participation",
                grain="player-week | active",
                source_timestamp="pre-kickoff features only",
                definition="P(records offensive snaps or dropbacks | active).",
                hard_rules=("inactive -> 0", "bye_no_game -> undefined/skipped"),
            ),
            EventProbabilitySpec(
                layer=EventLayer.POSITIVE_USAGE,
                name="p_positive_usage",
                owner="event_model.positive_usage",
                grain="player-week | participates",
                source_timestamp="pre-kickoff features only",
                definition=(
                    "P(positive target/carry/dropback/red-zone usage | participates)."
                ),
            ),
            EventProbabilitySpec(
                layer=EventLayer.CONDITIONAL_USAGE,
                name="usage_given_positive",
                owner="volume.conditional_share",
                grain="player-week | positive_usage",
                source_timestamp="pre-kickoff features only",
                definition="Conditional opportunity shares given positive usage.",
            ),
            EventProbabilitySpec(
                layer=EventLayer.CONDITIONAL_EFFICIENCY,
                name="efficiency_given_usage",
                owner="efficiency.conditional",
                grain="player-week | realized_usage",
                source_timestamp="pre-kickoff features only",
                definition="Yards/TD/completion rates conditional on realized opportunities.",
            ),
        )
    )
    deterministic_point_forecast: str = (
        "Expectation of the full mixture over active/participation/usage/efficiency."
    )
    predictive_component_distribution: str = (
        "Joint team/game-correlated component-stat draw set."
    )
    league_scored_distribution: str = (
        "Scoring contract applied independently to each joint draw index."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [
            {
                **asdict(event),
                "layer": event.layer.value,
            }
            for event in self.events
        ]
        return payload


DEFAULT_CONTRACT = ForecastGrainContract()


@dataclass(frozen=True)
class MixtureExpectation:
    """Hand-checkable mixture expectation without double-counting probabilities."""

    p_active: float
    p_participates_given_active: float
    expected_stat_given_participates: float

    @property
    def unconditional_expectation(self) -> float:
        return (
            float(self.p_active)
            * float(self.p_participates_given_active)
            * float(self.expected_stat_given_participates)
        )


def mixture_expectation(
    *,
    p_active: float,
    p_participates_given_active: float,
    expected_stat_given_participates: float,
) -> MixtureExpectation:
    """E[stat] = P(active) × P(participates|active) × E[stat|participates]."""
    if not (0.0 <= p_active <= 1.0):
        raise ValueError("p_active must be in [0, 1]")
    if not (0.0 <= p_participates_given_active <= 1.0):
        raise ValueError("p_participates_given_active must be in [0, 1]")
    if expected_stat_given_participates < 0:
        raise ValueError("expected_stat_given_participates must be non-negative")
    return MixtureExpectation(
        p_active=float(p_active),
        p_participates_given_active=float(p_participates_given_active),
        expected_stat_given_participates=float(expected_stat_given_participates),
    )


def apply_active_once(
    *,
    unconditional_share: float,
    p_active: float,
    already_conditioned_on_active: bool,
) -> float:
    """Prevent double application of availability to shares.

    If the share is already an expectation conditional on being active, do not
    multiply by ``p_active`` again. If it is unconditional on availability,
    multiply exactly once.
    """
    share = max(0.0, float(unconditional_share))
    p = float(np_clip_prob(p_active))
    if already_conditioned_on_active:
        return share
    return share * p


def np_clip_prob(p: float, *, eps: float = 1e-6) -> float:
    """Soft-clip probabilities away from exact 0/1 except hard-rule callers."""
    return float(min(1.0 - eps, max(eps, float(p))))


def classify_zero_row(
    *,
    has_scheduled_game: bool,
    is_active: bool | None,
    offense_snaps: float,
    targets: float,
    carries: float,
    attempts: float,
    locked_actual: bool = False,
) -> ZeroDefinition | None:
    """Classify why a player-week has zero counting stats (or None if positive)."""
    usage = float(offense_snaps) + float(targets) + float(carries) + float(attempts)
    if usage > 0:
        return None
    if locked_actual:
        return "locked_actual_zero"
    if not has_scheduled_game:
        return "bye_no_game"
    if is_active is False:
        return "true_dnp"
    if is_active is True:
        return "active_zero_usage"
    if is_active is None and not has_scheduled_game:
        return "bye_no_game"
    return "missing_data"


def contract_fingerprint(contract: ForecastGrainContract | None = None) -> str:
    import hashlib
    import json

    payload = (contract or DEFAULT_CONTRACT).to_dict()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_probability_owners(owners: Mapping[str, str]) -> None:
    """Require a single authoritative owner for play_prob."""
    if owners.get("play_prob") != "p_active":
        raise ValueError(
            "play_prob must map exclusively to p_active; refuse silent dual ownership"
        )
