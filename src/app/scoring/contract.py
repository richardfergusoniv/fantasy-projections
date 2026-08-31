"""League scoring contract types.

A compiled contract is the deterministic bridge between universal football stat
draws and league-specific fantasy points. It is intentionally explicit about what
it could *not* map: `unsupported_keys` and `unsupported_slots` block publication
rather than letting the app silently approximate a league's real rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Position = Literal["QB", "RB", "WR", "TE", "K", "DEF"]
SlotType = Literal[
    "QB",
    "RB",
    "WR",
    "TE",
    "FLEX",
    "REC_FLEX",
    "SUPER_FLEX",
    "K",
    "DEF",
    "BN",
    "IR",
    "TAXI",
]

#: Slots that hold players but never score in a weekly lineup.
NON_SCORING_SLOTS: frozenset[str] = frozenset({"BN", "IR", "TAXI"})


@dataclass(frozen=True)
class LinearRule:
    """Points per unit of a stat, optionally restricted to certain positions.

    `positions=()` means the rule applies to every position. Sleeper's
    position-conditional premiums (for example `bonus_rec_te`) compile to a
    restricted rule rather than being folded into the base receptions rule.
    """

    stat: str
    points_per_unit: float
    positions: tuple[str, ...] = ()

    def applies_to(self, position: str | None) -> bool:
        if not self.positions:
            return True
        if position is None:
            return False
        return position in self.positions


@dataclass(frozen=True)
class ThresholdRule:
    """A yardage/volume milestone bonus. Must be evaluated on each draw."""

    stat: str
    threshold: float
    bonus_points: float
    comparison: Literal[">=", ">"] = ">="

    def is_met(self, value: float) -> bool:
        if self.comparison == ">":
            return value > self.threshold
        return value >= self.threshold


@dataclass(frozen=True)
class BracketRule:
    """A mutually exclusive tier, e.g. Sleeper's tiered points/yards allowed.

    `lower` is inclusive and `upper` is inclusive; `upper=None` means unbounded.
    Exactly one bracket in a group may score for a given draw value.
    """

    stat: str
    lower: float
    upper: float | None
    points: float
    group: str = "default"

    def contains(self, value: float) -> bool:
        if value < self.lower:
            return False
        if self.upper is None:
            return True
        return value <= self.upper


@dataclass(frozen=True)
class DefenseRule:
    stat: str
    points_per_unit: float


@dataclass(frozen=True)
class RosterSlot:
    slot: SlotType
    count: int
    eligible_positions: tuple[Position, ...]

    @property
    def is_scoring(self) -> bool:
        return self.slot not in NON_SCORING_SLOTS


@dataclass
class ScoringContract:
    linear_rules: list[LinearRule] = field(default_factory=list)
    threshold_rules: list[ThresholdRule] = field(default_factory=list)
    bracket_rules: list[BracketRule] = field(default_factory=list)
    dst_rules: list[DefenseRule] = field(default_factory=list)
    roster_slots: list[RosterSlot] = field(default_factory=list)
    unsupported_keys: list[str] = field(default_factory=list)
    unsupported_slots: list[str] = field(default_factory=list)
    contract_hash: str = ""

    # ------------------------------------------------------------------ payload
    @staticmethod
    def _canonical(rules: list[Any]) -> list[dict[str, Any]]:
        """Serialize a rule list in an order that does not depend on input order.

        Rules are appended in Sleeper payload iteration order, so two fetches of
        the same league that serialize `scoring_settings` differently would
        otherwise produce different `contract_hash` values and look like a rule
        change. Rule order is semantically irrelevant (linear/DST rules sum,
        thresholds are independent, brackets are matched within a group), so the
        hashed payload sorts them canonically.
        """
        return sorted(
            (asdict(rule) for rule in rules),
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "linear_rules": self._canonical(self.linear_rules),
            "threshold_rules": self._canonical(self.threshold_rules),
            "bracket_rules": self._canonical(self.bracket_rules),
            "dst_rules": self._canonical(self.dst_rules),
            # Roster slots come from an ordered Sleeper list, not a mapping, so
            # their order is already deterministic and is meaningful to display.
            "roster_slots": [asdict(r) for r in self.roster_slots],
            "unsupported_keys": sorted(self.unsupported_keys),
            "unsupported_slots": sorted(self.unsupported_slots),
        }

    def finalize(self) -> ScoringContract:
        digest = hashlib.sha256(
            json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.contract_hash = digest
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["unsupported_keys"] = list(self.unsupported_keys)
        payload["unsupported_slots"] = list(self.unsupported_slots)
        payload["contract_hash"] = self.contract_hash
        return payload

    # ------------------------------------------------------------- publication
    @property
    def blocks_publication(self) -> bool:
        """True when the league's real rules cannot be reproduced exactly."""
        return bool(self.unsupported_keys) or bool(self.unsupported_slots)

    def publication_block_reason(self) -> str | None:
        if not self.blocks_publication:
            return None
        parts = []
        if self.unsupported_keys:
            parts.append(f"unsupported_scoring_keys={sorted(self.unsupported_keys)}")
        if self.unsupported_slots:
            parts.append(f"unsupported_roster_slots={sorted(self.unsupported_slots)}")
        return "; ".join(parts)

    # ------------------------------------------------------------------- slots
    @property
    def scoring_slots(self) -> list[RosterSlot]:
        return [slot for slot in self.roster_slots if slot.is_scoring]

    @property
    def starting_slot_count(self) -> int:
        return sum(slot.count for slot in self.scoring_slots)

    def eligible_positions(self) -> set[str]:
        positions: set[str] = set()
        for slot in self.scoring_slots:
            positions.update(slot.eligible_positions)
        return positions

    # ------------------------------------------------------------------- stats
    def required_stat_keys(self) -> set[str]:
        """Every stat this contract needs in order to score a draw exactly."""
        keys = {rule.stat for rule in self.linear_rules}
        keys |= {rule.stat for rule in self.threshold_rules}
        keys |= {rule.stat for rule in self.bracket_rules}
        keys |= {rule.stat for rule in self.dst_rules}
        return keys

    @property
    def has_nonlinear_rules(self) -> bool:
        """Threshold/bracket rules are nonlinear and invalid on mean stats."""
        return bool(self.threshold_rules) or bool(self.bracket_rules)

    def nonlinear_rule_labels(self) -> list[str]:
        labels = [
            f"threshold:{r.stat}{r.comparison}{r.threshold:g}" for r in self.threshold_rules
        ]
        labels += sorted({f"bracket:{r.stat}:{r.group}" for r in self.bracket_rules})
        return labels

    def first_down_rules(self) -> list[LinearRule]:
        return [rule for rule in self.linear_rules if rule.stat.endswith("first_downs")]
