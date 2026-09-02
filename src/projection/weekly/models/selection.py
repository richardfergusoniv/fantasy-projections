"""Shared out-of-sample model and hyperparameter promotion rules."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations


def candidate_clears_baseline(
    candidate_score: float | None,
    baseline_score: float,
    *,
    min_improvement: float = 0.0,
) -> bool:
    """Return true only when a lower-is-better candidate clears the margin."""
    return (
        candidate_score is not None
        and candidate_score < baseline_score - max(0.0, float(min_improvement))
    )
