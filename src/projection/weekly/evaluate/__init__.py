"""Weekly-v2 evaluation metrics and promotion gates."""

from src.projection.weekly.evaluate.harness import (
    PreseasonEvalConfig,
    evaluate_season,
    run_preseason_backtest,
)
from src.projection.weekly.evaluate.nested_selection import (
    NestedSelectionConfig,
    run_nested_selection,
)
from src.projection.weekly.evaluate.metrics import (
    build_last5_baseline,
    build_prior_season_ppg_baseline,
    evaluate_projections,
    evaluate_season_level,
    format_report,
)
from src.projection.weekly.evaluate.preseason import (
    PromotionPolicy,
    assert_strict_preseason_asof,
    complete_roster_week_outcomes,
    evaluate_complete_preseason,
    file_fingerprint,
    promotion_gate,
    roster_week_cohort,
    write_freshness_manifest,
)

__all__ = [
    "PromotionPolicy",
    "assert_strict_preseason_asof",
    "build_last5_baseline",
    "build_prior_season_ppg_baseline",
    "complete_roster_week_outcomes",
    "evaluate_complete_preseason",
    "evaluate_projections",
    "evaluate_season_level",
    "file_fingerprint",
    "format_report",
    "promotion_gate",
    "roster_week_cohort",
    "write_freshness_manifest",
    "evaluate_season",
    "PreseasonEvalConfig",
    "run_preseason_backtest",
    "NestedSelectionConfig",
    "run_nested_selection",
]
