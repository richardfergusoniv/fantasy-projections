"""Separate readiness gates for point vs joint-draw publication paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GateStatus:
    name: str
    passed: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_hash: str = ""
    evidence_path: str = ""

    def has_evidence(self) -> bool:
        return bool(self.evidence_hash or self.evidence_path or self.evidence)


@dataclass
class JointReadinessReport:
    point_model_classification: GateStatus
    event_probability_calibration: GateStatus
    joint_draw_proper_scores: GateStatus
    per_draw_conservation: GateStatus
    ppfd_component_readiness: GateStatus
    kicker_readiness: GateStatus
    dst_readiness: GateStatus
    league_scoring_completeness: GateStatus
    decision_lineup_matchup: GateStatus
    artifact_integrity: GateStatus
    # High-level decisions
    joint_draw_classification: str = "NO-GO"
    manual_trained_shadow_publication: str = "NO-GO"
    automatic_weekly_publication: str = "NO-GO"
    start_sit_use: str = "NO-GO"
    public_internet_deployment: str = "NO-GO"
    auto_publish_allowed: bool = False
    notes: list[str] = field(default_factory=list)

    def production_gates(self) -> list[GateStatus]:
        return [
            self.point_model_classification,
            self.event_probability_calibration,
            self.joint_draw_proper_scores,
            self.per_draw_conservation,
            self.ppfd_component_readiness,
            self.kicker_readiness,
            self.dst_readiness,
            self.league_scoring_completeness,
            self.decision_lineup_matchup,
            self.artifact_integrity,
        ]

    def recompute_decisions(
        self,
        *,
        point_dispersion_passes: bool,
        external_blockers: bool = True,
    ) -> None:
        """Derive go/no-go labels. Point dispersion failure blocks auto-publish."""
        joint_core = [
            self.event_probability_calibration,
            self.joint_draw_proper_scores,
            self.per_draw_conservation,
            self.artifact_integrity,
        ]
        joint_ok = all(g.passed and g.has_evidence() for g in joint_core)
        self.joint_draw_classification = "GO" if joint_ok else "NO-GO"

        decision_ok = (
            joint_ok
            and self.decision_lineup_matchup.passed
            and self.decision_lineup_matchup.has_evidence()
            and self.league_scoring_completeness.passed
            and self.league_scoring_completeness.has_evidence()
            and self.ppfd_component_readiness.passed
            and self.ppfd_component_readiness.has_evidence()
            and self.kicker_readiness.passed
            and self.kicker_readiness.has_evidence()
            and self.dst_readiness.passed
            and self.dst_readiness.has_evidence()
        )
        # PPFD/K/DST only block leagues that need them; global start/sit requires
        # them when league_scoring_completeness says so.
        self.start_sit_use = "GO" if decision_ok else "NO-GO"
        self.manual_trained_shadow_publication = (
            "GO" if joint_ok and self.point_model_classification.passed else "NO-GO"
        )
        self.auto_publish_allowed = bool(
            point_dispersion_passes
            and all(g.passed for g in self.production_gates())
        )
        self.automatic_weekly_publication = "GO" if self.auto_publish_allowed else "NO-GO"
        self.public_internet_deployment = "NO-GO" if external_blockers else (
            "GO" if self.auto_publish_allowed else "NO-GO"
        )
        if not point_dispersion_passes:
            self.notes.append(
                "auto_publish_allowed forced false: unchanged point-dispersion gate failed"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": {g.name: asdict(g) for g in self.production_gates()},
            "point_model_classification": self.point_model_classification.passed,
            "joint_draw_classification": self.joint_draw_classification,
            "manual_trained_shadow_publication": self.manual_trained_shadow_publication,
            "automatic_weekly_publication": self.automatic_weekly_publication,
            "start_sit_use": self.start_sit_use,
            "public_internet_deployment": self.public_internet_deployment,
            "auto_publish_allowed": self.auto_publish_allowed,
            "notes": list(self.notes),
        }


def default_no_go_report(
    *,
    point_go_with_caveats: bool = True,
    point_dispersion_passes: bool = False,
) -> JointReadinessReport:
    """Baseline report preserving current no-gos until gates pass with evidence."""
    def g(name: str, passed: bool, detail: str) -> GateStatus:
        return GateStatus(name=name, passed=passed, detail=detail)

    report = JointReadinessReport(
        point_model_classification=g(
            "point_model_classification",
            point_go_with_caveats,
            "Existing trained artifact GO with caveats; volume tune promote=false",
        ),
        event_probability_calibration=g(
            "event_probability_calibration", False, "pending/failed evidence"
        ),
        joint_draw_proper_scores=g(
            "joint_draw_proper_scores", False, "pending/failed evidence"
        ),
        per_draw_conservation=g(
            "per_draw_conservation", False, "pending/failed evidence"
        ),
        ppfd_component_readiness=g(
            "ppfd_component_readiness", False, "pending/failed evidence"
        ),
        kicker_readiness=g("kicker_readiness", False, "pending/failed evidence"),
        dst_readiness=g("dst_readiness", False, "pending/failed evidence"),
        league_scoring_completeness=g(
            "league_scoring_completeness", False, "pending/failed evidence"
        ),
        decision_lineup_matchup=g(
            "decision_lineup_matchup", False, "pending/failed evidence"
        ),
        artifact_integrity=g(
            "artifact_integrity", False, "pending/failed evidence"
        ),
    )
    report.recompute_decisions(point_dispersion_passes=point_dispersion_passes)
    return report
