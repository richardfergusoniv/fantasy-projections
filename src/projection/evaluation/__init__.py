"""Rolling backtest, calibration, and baseline reference models."""

from src.projection.evaluation.baselines import (
    BASELINE_NAMES,
    attach_all_baselines,
    empirical_bayes_shrunk_rate,
    prior_year_rate,
    ridge_elastic_baseline,
    team_share_times_volume,
    weighted_3y_average,
)
from src.projection.evaluation.calibration import (
    coverage_by_group,
    crps_gaussian,
    crps_sample,
    pinball_loss,
    reliability_table,
    summarize_interval_calibration,
)

__all__ = [
    "BASELINE_NAMES",
    "attach_all_baselines",
    "empirical_bayes_shrunk_rate",
    "prior_year_rate",
    "ridge_elastic_baseline",
    "team_share_times_volume",
    "weighted_3y_average",
    "coverage_by_group",
    "crps_gaussian",
    "crps_sample",
    "pinball_loss",
    "reliability_table",
    "summarize_interval_calibration",
]
