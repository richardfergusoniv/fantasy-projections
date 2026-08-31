"""Weekly feature as-of contract registry and audit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.projection.contracts import OUTPUT_DIR

AUDIT_SCHEMA_VERSION = "weekly_feature_contract_v1"


@dataclass(frozen=True)
class FeatureContract:
    name: str
    source_table: str
    aggregation: str
    max_observation_rule: str
    as_of_cutoff: str
    notes: str = ""


DEFAULT_CONTRACTS: tuple[FeatureContract, ...] = (
    FeatureContract(
        name="targets_share_roll3",
        source_table="weekly",
        aggregation="3-week rolling mean of target share",
        max_observation_rule="week < target_week",
        as_of_cutoff="kickoff of target week",
        notes="Leakage-safe: shift(1) then 3-week rolling mean (prior weeks only).",
    ),
    FeatureContract(
        name="carries_share_roll3",
        source_table="weekly",
        aggregation="3-week rolling mean of carry share",
        max_observation_rule="week < target_week",
        as_of_cutoff="kickoff of target week",
        notes="Leakage-safe: shift(1) then 3-week rolling mean (prior weeks only).",
    ),
    FeatureContract(
        name="snap_pct",
        source_table="snap_counts",
        aggregation="season aggregate offensive snap percentage",
        max_observation_rule="season N complete before preseason projection for N+1",
        as_of_cutoff="preseason",
        notes="Preseason-safe at season grain.",
    ),
    FeatureContract(
        name="injury_durability_rate",
        source_table="injuries",
        aggregation="trailing season injury report",
        max_observation_rule="season N only for projection of N+1",
        as_of_cutoff="preseason",
    ),
    FeatureContract(
        name="opp_def_pass_epa_prior",
        source_table="pbp",
        aggregation="prior-season opponent defensive EPA",
        max_observation_rule="season < target_season",
        as_of_cutoff="preseason",
    ),
    FeatureContract(
        name="depth_chart_status",
        source_table="depth_chart",
        aggregation="curated chart with dated overrides",
        max_observation_rule="as_of_date <= projection_as_of",
        as_of_cutoff="projection publish time",
    ),
)


def contracts_to_records() -> list[dict]:
    return [
        {
            "name": c.name,
            "source_table": c.source_table,
            "aggregation": c.aggregation,
            "max_observation_rule": c.max_observation_rule,
            "as_of_cutoff": c.as_of_cutoff,
            "notes": c.notes,
        }
        for c in DEFAULT_CONTRACTS
    ]


def audit_feature_contracts() -> dict:
    """Static audit of registered weekly/in-season feature contracts."""
    rows = []
    for contract in DEFAULT_CONTRACTS:
        # week < target_week features pass when notes document a prior-week shift.
        notes_lower = contract.notes.lower()
        leakage_safe = (
            "shift" in notes_lower
            or "shifted" in notes_lower
            or "leakage-safe" in notes_lower
            or "prior week" in notes_lower
        )
        if contract.max_observation_rule.startswith("week < target_week"):
            passes = leakage_safe
        else:
            passes = True
        rows.append(
            {
                "feature": contract.name,
                "passes": passes,
                "max_observation_rule": contract.max_observation_rule,
                "as_of_cutoff": contract.as_of_cutoff,
                "notes": contract.notes,
            }
        )
    failing = [row["feature"] for row in rows if not row["passes"]]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": rows,
        "passes": len(failing) == 0,
        "failing_features": failing,
    }


def write_audit_artifacts(audit_report: dict) -> dict[str, str]:
    out_dir = Path(OUTPUT_DIR) / "weekly_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = out_dir / "feature_contract.json"
    report_path = out_dir / "audit_report.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "features": contracts_to_records(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    return {
        "feature_contract": str(contract_path),
        "audit_report": str(report_path),
    }


def feature_at_week(
    weekly_values: pd.Series,
    *,
    target_week: int,
    builder: Callable[[pd.Series, int], float],
) -> float:
    """Compute a feature for target_week using only prior-week observations."""
    prior = weekly_values[weekly_values.index < target_week]
    return float(builder(prior, target_week))


def target_week_unchanged_when_outcomes_mutated(
    *,
    feature_before: float,
    feature_after: float,
    tolerance: float = 1e-9,
) -> bool:
    return abs(feature_before - feature_after) <= tolerance


def future_week_may_change_after_target_week_mutation(
    *,
    feature_before: float,
    feature_after: float,
    tolerance: float = 1e-9,
) -> bool:
    return abs(feature_before - feature_after) > tolerance
