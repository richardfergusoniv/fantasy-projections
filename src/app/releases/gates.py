"""Promotion gate checks before activating projection pointers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.app.projections.loader import PlayerSummary


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "failures": self.failures, "warnings": self.warnings}


def validate_promotion(
    *,
    mode: str,
    players: dict[str, PlayerSummary],
    min_players: int = 100,
) -> GateResult:
    failures: list[str] = []
    warnings: list[str] = []

    if len(players) < min_players:
        failures.append(f"player_count_below_minimum:{len(players)}<{min_players}")

    missing_quantiles = [pid for pid, summary in players.items() if not summary.quantiles]
    if missing_quantiles:
        failures.append(f"missing_quantiles:{len(missing_quantiles)}")

    invalid_probs = [
        pid
        for pid, summary in players.items()
        if summary.availability_probability < 0 or summary.availability_probability > 1
    ]
    if invalid_probs:
        failures.append(f"invalid_availability_probability:{len(invalid_probs)}")

    negative_points = [pid for pid, summary in players.items() if summary.mean_points < 0]
    if negative_points:
        failures.append(f"negative_mean_points:{len(negative_points)}")

    if mode == "weekly":
        zero_week = [pid for pid, summary in players.items() if summary.mean_points == 0 and summary.position not in {"DST", "K"}]
        if len(zero_week) > len(players) * 0.5:
            warnings.append(f"high_zero_projection_rate:{len(zero_week)}")

    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def validate_matchup_probabilities(probabilities: dict[str, float]) -> GateResult:
    failures: list[str] = []
    for key, value in probabilities.items():
        if value < 0 or value > 1:
            failures.append(f"matchup_probability_out_of_range:{key}={value}")
    total = sum(probabilities.values())
    if probabilities and abs(total - 1.0) > 0.01:
        failures.append(f"matchup_probabilities_not_normalized:{total:.4f}")
    return GateResult(passed=not failures, failures=failures)


def validate_simulation_partitions(session, *, run_id: str, input_hash: str) -> GateResult:
    """Ensure simulation partitions exist and are well-formed."""
    from src.app.persistence.models import SimulationPartition

    failures: list[str] = []
    partitions = session.query(SimulationPartition).filter(SimulationPartition.run_id == run_id).all()
    if not partitions:
        failures.append("missing_simulation_partitions")
    for partition in partitions:
        if not partition.sha256 or len(partition.sha256) != 64:
            failures.append(f"invalid_partition_hash:{partition.partition_key}")
        if partition.draw_count <= 0:
            failures.append(f"invalid_draw_count:{partition.partition_key}")
    if input_hash and len(input_hash) < 16:
        failures.append("invalid_input_hash")
    return GateResult(passed=not failures, failures=failures)


def validate_artifact_readiness(readiness, *, app_env: str, automatic: bool) -> GateResult:
    """Block automatic production publication of untrained weekly-v2 artifacts.

    ``readiness`` is a :class:`~src.app.projections.weekly_v2_bridge.WeeklyV2Readiness`.
    Anything other than the ``trained`` state is a hard failure for an
    automatically scheduled run in production. Manual or non-production runs are
    allowed through with a warning; the run itself still carries the state label
    so a fixture-backed release can never be mistaken for a trained one.
    """
    failures: list[str] = []
    warnings: list[str] = []
    state = getattr(readiness, "state", "absent")
    if state != "trained":
        detail = f"weekly_v2_artifacts_not_trained:{state}"
        missing = tuple(getattr(readiness, "missing_artifacts", ()) or ())
        if missing:
            detail = f"{detail}:missing={len(missing)}"
        if automatic and app_env == "production":
            failures.append(detail)
        else:
            warnings.append(detail)
    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def validate_scoring_contracts(session, league_ids: list[str]) -> GateResult:
    """Fail promotion when seeded leagues have unsupported nonzero scoring keys."""
    from src.app.persistence.models import LeagueRuleSnapshot

    failures: list[str] = []
    warnings: list[str] = []
    for league_id in league_ids:
        snapshot = (
            session.query(LeagueRuleSnapshot)
            .filter(LeagueRuleSnapshot.league_id == league_id)
            .order_by(LeagueRuleSnapshot.fetched_at.desc())
            .first()
        )
        if snapshot is None:
            warnings.append(f"missing_rule_snapshot:{league_id}")
            continue
        unsupported = (snapshot.normalized_json or {}).get("unsupported_keys") or []
        if unsupported:
            failures.append(f"unsupported_scoring_keys:{league_id}:{len(unsupported)}")
    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def scoring_contract_fingerprint(session, league_ids: list[str]) -> str:
    """Order-independent digest of the compiled scoring contracts in force.

    Recorded on each promoted run so a later incremental refresh can tell whether
    scoring changed underneath it and widen to a full recompute if so.
    """
    from src.app.persistence.models import LeagueRuleSnapshot

    parts: list[str] = []
    for league_id in sorted(set(league_ids)):
        snapshot = (
            session.query(LeagueRuleSnapshot)
            .filter(LeagueRuleSnapshot.league_id == league_id)
            .order_by(LeagueRuleSnapshot.fetched_at.desc())
            .first()
        )
        parts.append(f"{league_id}:{snapshot.contract_hash if snapshot else 'missing'}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()
