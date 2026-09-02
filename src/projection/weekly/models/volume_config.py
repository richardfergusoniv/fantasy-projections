"""Typed, serializable volume-model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VolumeModelConfig:
    """Hyperparameters passed to :func:`train_volume_models`."""

    two_stage: bool = True
    model_type: str = "hgb"
    participation_model_type: str = "hgb"
    participation_threshold: float = 1e-6
    min_participation_rows: int = 20
    recency_half_life_seasons: float | None = None

    def to_options(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_options(cls, options: dict[str, Any] | None) -> VolumeModelConfig:
        if not options:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in options.items() if k in known})

    def fingerprint(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.to_options(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


# Frozen candidate grid — established before reviewing experiment results.
DEFAULT_CANDIDATE_GRID: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline_two_stage",
        "options": VolumeModelConfig().to_options(),
        "description": "Current production two-stage architecture, all-history weighting",
    },
    {
        "name": "two_stage_half_life_6",
        "options": VolumeModelConfig(recency_half_life_seasons=6.0).to_options(),
    },
    {
        "name": "two_stage_half_life_4",
        "options": VolumeModelConfig(recency_half_life_seasons=4.0).to_options(),
    },
    {
        "name": "two_stage_half_life_2",
        "options": VolumeModelConfig(recency_half_life_seasons=2.0).to_options(),
    },
    {
        "name": "two_stage_participation_ridge",
        "options": VolumeModelConfig(participation_model_type="ridge").to_options(),
    },
    {
        "name": "two_stage_conditional_ridge",
        "options": VolumeModelConfig(model_type="ridge").to_options(),
    },
    {
        "name": "two_stage_conservative_hgb",
        "options": VolumeModelConfig(
            model_type="hgb",
            participation_model_type="hgb",
        ).to_options(),
    },
    {
        "name": "legacy_direct",
        "options": VolumeModelConfig(two_stage=False).to_options(),
    },
)

BASELINE_CANDIDATE_NAME = "baseline_two_stage"
