"""Configurable fantasy scoring systems."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoringConfig:
    """Linear fantasy scoring rules.

    Defaults match standard PPR. Use presets for half-PPR / standard.
    """

    reception_points: float = 1.0
    pass_td_points: float = 4.0
    rush_rec_td_points: float = 6.0
    pass_yard_points: float = 0.04  # 1 pt / 25 yards
    rush_rec_yard_points: float = 0.1  # 1 pt / 10 yards
    interception_points: float = -2.0
    fumble_lost_points: float = -2.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_name(cls, name: str) -> ScoringConfig:
        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        presets = {
            "ppr": ppr,
            "half_ppr": half_ppr,
            "half": half_ppr,
            "standard": standard,
            "std": standard,
            "non_ppr": standard,
        }
        if key not in presets:
            raise ValueError(
                f"Unknown scoring system {name!r}. "
                f"Choose from: {', '.join(sorted(presets))}"
            )
        return presets[key]()


def ppr() -> ScoringConfig:
    """Full PPR (1 point per reception)."""
    return ScoringConfig(reception_points=1.0)


def half_ppr() -> ScoringConfig:
    """Half-PPR (0.5 points per reception)."""
    return ScoringConfig(reception_points=0.5)


def standard() -> ScoringConfig:
    """Standard scoring (no reception points)."""
    return ScoringConfig(reception_points=0.0)
