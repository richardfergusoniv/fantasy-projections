"""Shared projection contracts: paths, calibrated constants, output schema.

Leaf module — do not import predict, rookies, or reconcile stages from here.
Callers may re-export these names for backward-compatible imports.
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
INTERVAL_RESIDUALS_PATH = os.path.join(MODELS_DIR, "interval_residuals.csv")
CORRECTIONS_PATH = os.path.join(MODELS_DIR, "corrections.joblib")
DEPTH_CHART_PATH = os.path.join(REPO_ROOT, "src", "depth_chart", "starters_2026.csv")
LIVE_DEPTH_CHART_PATH = os.path.join(REPO_ROOT, "src", "depth_chart", "live_depth_2026.csv")
STATUS_OVERRIDES_PATH = os.path.join(
    REPO_ROOT, "src", "depth_chart", "status_overrides_2026.csv"
)

# Researched curated depths: QB1-2, RB1-2, WR1-3, TE1-2.
CURATED_RESEARCH_DEPTH = {"QB": 2, "RB": 2, "WR": 3, "TE": 2}
DEEP_BENCH_GAMES_CAP = 6.0
ROOKIE_RATIO_FALLBACK = (0.2, 3.0)

# Same band as historical vacated-opportunity clip (rookies + team changers).
VACATED_CLIP = (0.3, 2.5)
TEAM_CHANGE_SHARE_CLIP = VACATED_CLIP

# Gate B conditional-rate ladder (fit on veteran transition pairs).
DEPTH_RATE_LADDER = {
    "QB": {1: 1.00, 2: 0.77},
    "RB": {1: 1.00, 2: 0.98, 3: 0.73},
    "WR": {1: 1.00, 2: 1.00, 3: 0.97, 4: 1.00, 5: 0.86},
    "TE": {1: 1.00, 2: 0.90, 3: 0.83},
}
DEPTH_RATE_DEEP = {"QB": 0.84, "RB": 0.70, "WR": 0.94, "TE": 1.00}
DEPTH_RATE_OFF_CHART = {"QB": 1.00, "RB": 0.86, "WR": 0.79, "TE": 0.77}

BOOST_ELIGIBLE_ROLES = {"starter", "committee"}

# Vacancy damping. Carry alpha returned to 0.0 (neutral) on 2026-08-15.
#
# Provenance, not measurement: carry alpha=1.0 was re-enabled on 2026-08-14
# citing RB_CARRY_VACANCY_2026-08-14.md, but four of that diagnostic's five
# rows are Sleeper comparisons and scripts/diag_rb_carry_vacancy.py reads
# only output/sleeper_comparison_2026.csv. It was disabled on Sleeper
# evidence (9d5e533) and re-enabled on Sleeper evidence - never scored on
# fantasy points in either direction. PROVENANCE_AUDIT.md classes it C.
#
# It also cannot be settled by the current harness: fantasy_evaluation does
# not import roster_moves, so apply_incumbent_vacancy_boost runs upstream of
# compose_board and outside every leakage-safe fold (ABLATION_RESULTS.md).
# 0.0 is the pre-Sleeper-evidence default; the model's own learned rates
# carry the signal instead. Restore to 1.0 only against a held-out outcome
# fit, which requires extending the harness past compose_board first.
INCUMBENT_VACANCY_ALPHA = {"target": 0.5, "carry": 0.0}
TEAM_CHANGE_VACANCY_ALPHA = {"target": 0.35, "carry": 0.25}
INCUMBENT_VACANCY_NET_CLIP = 0.75
INCUMBENT_VACANCY_SCALE_CAP = 2.0

REPLACEMENT_POSITIONS = ("RB", "WR", "TE")
REPLACEMENT_MIN_CELL = 15
REPLACEMENT_DEPTH_BANDS = ((1, "rank_1"), (2, "rank_2"), (99, "rank_3_plus"))

# Board-level tripwire ceilings (warn only; not composition caps).
RUSH_ATTEMPTS_PER_APPEARANCE_MAX = {"QB": 12.0, "RB": 25.0, "WR": 5.0, "TE": 3.0}
RUSH_YARDS_PER_CARRY_MAX = {"QB": 10.0, "RB": 7.0, "WR": 15.0, "TE": 15.0}

# Live-only OL feature smoothing. 0 = exact-season only (historical/backtest
# default). 3 = snap-weighted trailing average over the source season and the
# two prior seasons, applied only to the live predict source season.
# Ablation (OL score persistence MAE) favored trailing — see ol_quality note.
OL_TRAILING_SEASONS = 3

TEAM_ANCHOR_OUTPUT_COLS = [
    "team_passing_yards_pg_pred", "team_pass_attempts_pg_pred",
    "team_carries_pg_pred", "team_rushing_yards_pg_pred",
    "team_anchor_source_season", "team_anchor_lag_team",
    "team_anchor_provenance",
]

# --- Team volume reconciliation -------------------------------------------
#
# Player predictions are made independently, so nothing stops a team's summed
# volume from exceeding what that team will actually run. It did: before this,
# 22 of 32 teams projected more QB pass attempts than any NFL team recorded in
# 2021-2024. Every documented industry projection process is top-down for
# exactly this reason.
#
# scale = (team_target / summed_prediction) ** TEAM_RECONCILE_ALPHA
#
# ALPHA is measured, not asserted. Rolling origin 2022-2025, refitting both
# the player and team models per fold, scored on player-level season-total MAE
# (never on the team sum, which alpha=1 fixes by construction):
#
#                    all team-folds      coverage >= 90%
#     alpha=0.25         -1.01%              -2.91%
#     alpha=0.50         -0.94%              -4.98%
#     alpha=0.75         +0.14%              -6.58%
#     alpha=1.00         +2.93%              -7.31%
#
# The two columns differ because the backtest population is only ~85% of a
# real team (transition pairs only; p10 coverage 47%) while the shipped board
# projects veterans + rookies + replacement rows and covers 96%/90%. Scaling a
# partial population to a whole-team target inflates whoever is present, which
# is why full reconciliation is harmful on the left and best on the right.
#
# 0.5 rather than 1.0 deliberately: it is the strongest setting that improves
# in BOTH regimes, so a team whose coverage degrades (players dropped for
# having no roster row) cannot be made worse by it. On the production-like
# slice it captures about two thirds of the available gain, improving all four
# position/stat combos and all four seasons. Raise it only against a measured
# improvement on the shipped path.
TEAM_RECONCILE_ALPHA = 0.5

# Ratio clip, so a team whose summed prediction is near zero cannot produce an
# unbounded rescale.
TEAM_RECONCILE_CLIP = (0.25, 4.0)

# Measured median share of each team total that the position actually takes,
# 2016-2025. RB do NOT take every team carry - scoring that ratio against 1.00
# was a real error in an earlier pass of this work.
#   (position, stat) -> (team anchor column, share of it)
TEAM_VOLUME_SHARES = {
    ("QB", "attempts"): ("team_pass_attempts_pg_pred", 0.941),
    ("QB", "passing_yards"): ("team_passing_yards_pg_pred", 0.942),
    ("RB", "carries"): ("team_carries_pg_pred", 0.810),
    ("RB", "rushing_yards"): ("team_rushing_yards_pg_pred", 0.806),
}

# Counting stats with no anchor of their own take the scale of the stat they
# ride along with, so a reconciled line stays internally consistent instead of
# gaining completions the reconciled attempts no longer support.
TEAM_VOLUME_SIBLINGS = {
    ("QB", "attempts"): ("completions", "passing_tds", "interceptions"),
    ("RB", "carries"): ("rushing_tds",),
}

OUTPUT_COLUMNS = [
    "player_id", "display_name", "team", "position", "stat",
    "pred_pg", "pred_pg_low", "pred_pg_high",
    "pred_season", "pred_season_low", "pred_season_high",
    "source", "low_confidence", "rookie_tier", "interval_low_n_flag", "season",
    "team_changed", "roster_status",
    "depth_rank", "role", "formation_role", "depth_chart_status",
    "role_discount_applied",
    "nfl_depth_rank",
    # The tier the model actually consumed, and where it came from. These are
    # the audit surface that role_discount_factor used to be: depth reaches
    # the projection as a model INPUT now, so the input has to be visible or
    # nobody can tell why a player was projected the way he was.
    # depth_tier_source = 'curated' means the hand chart overrode the feed.
    "depth_tier", "depth_tier_source",
    # What the top-down team reconciliation did to this row. 1.0 = untouched.
    # Same rationale as depth_tier: the adjustment has to be visible or nobody
    # can tell why a player's number differs from what his model produced.
    "team_volume_scale",
    "role_discount_factor",
    "athletic_tier",
    "receiving_share_capped",
    "receiving_share_normalized",
    "elite_correction_pg",
    "projected_games",
    "projected_games_raw",
    "projected_volume_games",
    "team_pass_attempts_pg_pred", "team_passing_yards_pg_pred",
    "team_carries_pg_pred", "team_rushing_yards_pg_pred",
    "team_anchor_source_season", "team_anchor_lag_team", "team_anchor_provenance",
    "target_depth_rank", "rookie_depth_band",
    "rookie_availability_cell_n", "rookie_availability_fallback_used",
    "rookie_vacancy_scale",
    "rookie_id_unresolved",
    "stat_constraint_applied",
]
