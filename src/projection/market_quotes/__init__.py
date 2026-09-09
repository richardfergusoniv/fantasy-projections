"""Shared sportsbook quote primitives for season and weekly consensus.

Season-long thresholds (Pierce 999.5-yard calibration) live in the draft
assistant consensus module. Weekly thresholds are configured separately under
``src.projection.weekly_props.config``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PREDICTION_MARKET_BOOKS = frozenset(
    {
        "kalshi",
        "polymarket",
        "polymarket us",
        "polymarket.com",
    }
)

DEFAULT_SKEWED_ODDS_GAP = 0.35
DEFAULT_ONE_SIDED_LONGSHOT = 100.0


@dataclass(frozen=True)
class QuotePolicy:
    """Configurable acceptance thresholds for a consensus grain."""

    skewed_odds_gap: float = DEFAULT_SKEWED_ODDS_GAP
    one_sided_longshot_threshold: float = DEFAULT_ONE_SIDED_LONGSHOT
    # Relative / absolute conflict vs same-source projection (count markets).
    count_conflict_rel: float = 0.15
    count_conflict_abs: float = 1.0
    count_scale_ceiling: float = 25.0
    # Yardage conflict vs same-source projection.
    yard_conflict_rel_hard: float = 0.5
    yard_conflict_abs_floor: float = 1.5
    yard_upward_rel: float = 0.12
    yard_upward_abs: float = 80.0
    # Robust median outlier band.
    robust_median_rel: float = 0.35
    robust_median_abs_floor: float = 100.0


SEASON_QUOTE_POLICY = QuotePolicy()


def american_implied_prob(odds: Any) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def no_vig_probs(over_odds: Any, under_odds: Any) -> tuple[float, float] | None:
    """Return normalized over/under probabilities when both sides exist."""
    over_p = american_implied_prob(over_odds)
    under_p = american_implied_prob(under_odds)
    if over_p is None or under_p is None:
        return None
    total = over_p + under_p
    if total <= 0:
        return None
    return over_p / total, under_p / total


def odds_skewed(
    over: Any,
    under: Any,
    *,
    gap: float = DEFAULT_SKEWED_ODDS_GAP,
) -> bool:
    over_p = american_implied_prob(over)
    under_p = american_implied_prob(under)
    if over_p is None or under_p is None:
        return False
    return abs(over_p - under_p) > gap


def one_sided_longshot(
    over: Any,
    under: Any,
    *,
    threshold: float = DEFAULT_ONE_SIDED_LONGSHOT,
) -> bool:
    """Reject threshold props posted as plus-money on only one side."""

    def _is_longshot(odds: Any) -> bool:
        try:
            return float(odds) >= threshold
        except (TypeError, ValueError):
            return False

    if over is not None and under is None and _is_longshot(over):
        return True
    if under is not None and over is None and _is_longshot(under):
        return True
    return False


def is_prediction_market_book(name: str) -> bool:
    key = str(name or "").strip().lower()
    if key in PREDICTION_MARKET_BOOKS:
        return True
    return "kalshi" in key or "polymarket" in key


def conflicts_with_projection(
    line: float,
    projection: float,
    *,
    policy: QuotePolicy = SEASON_QUOTE_POLICY,
) -> bool:
    """True when a book line is juiced upward vs the same-source projection."""
    scale = max(abs(projection), 1.0)
    delta = abs(line - projection)
    rel = delta / scale
    if scale <= policy.count_scale_ceiling:
        return (
            line > projection
            and delta >= policy.count_conflict_abs
            and rel >= policy.count_conflict_rel
        )
    return (rel > policy.yard_conflict_rel_hard and delta > max(policy.yard_conflict_abs_floor, 0.05 * scale)) or (
        line > projection
        and rel > policy.yard_upward_rel
        and delta > policy.yard_upward_abs
    )


def scalar_line(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def robust_median(
    values: list[float],
    *,
    policy: QuotePolicy = SEASON_QUOTE_POLICY,
) -> float:
    """Median after dropping far outliers when 3+ quotes exist."""
    from statistics import median

    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return float(median(values))
    center = float(median(values))
    tolerance = max(abs(center) * policy.robust_median_rel, policy.robust_median_abs_floor)
    kept = [value for value in values if abs(value - center) <= tolerance]
    if len(kept) >= 2:
        return float(median(kept))
    return center


def classify_kind(kind: str | None) -> str:
    if kind == "book":
        return "book"
    if kind == "projection":
        return "projection"
    return "none"
