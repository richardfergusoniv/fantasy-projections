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

PASS_CATCH_COHERENCE_BAND = (0.8, 1.35)
NAMED_REC_YARDS_COVERAGE = 0.98
NAMED_REC_RECEPTIONS_COVERAGE = 0.98
NAMED_REC_TDS_COVERAGE = 0.96

QB_ATTEMPTS_PER_VOLUME_GAME_MAX = 42.0
RUSH_ATTEMPTS_PER_APPEARANCE_MAX = {"QB": 12.0, "RB": 25.0, "WR": 5.0, "TE": 3.0}
RUSH_YARDS_PER_CARRY_MAX = {"QB": 10.0, "RB": 7.0, "WR": 15.0, "TE": 15.0}
NAMED_RUSH_COVERAGE = 0.814

# Live-only OL feature smoothing. 0 = exact-season only (historical/backtest
# default). 3 = snap-weighted trailing average over the source season and the
# two prior seasons, applied only to the live predict source season.
# Ablation (OL score persistence MAE) favored trailing — see ol_quality note.
OL_TRAILING_SEASONS = 3

USAGE_SHARE_BLEND_W = 0.0
USAGE_SHARE_CURATED_W = 0.5

# Preseason WR formation columns (Ourlads LWR/RWR/SWR). Relative priors match
# the curated WR_USAGE_SLOTS defaults and are renormalized within present roles
# at L3. Blend pulls within-WR budget toward those columns so a vacated LWR
# leaves volume in the LWR bucket instead of scaling every WR equally.
WR_FORMATION_ROLES = ("LWR", "RWR", "SWR")
WR_FORMATION_ROLE_PRIORS = {"LWR": 0.1554, "RWR": 0.0667, "SWR": 0.0386}
FORMATION_ROLE_BLEND_W = 0.5
DEPTH_RANK_TO_WR_FORMATION_ROLE = {1: "LWR", 2: "RWR", 3: "SWR"}

USAGE_SHARE_FAMILIES = {
    "receiving": {
        "positions": ("WR", "TE", "RB"),
        "stats": ("targets", "receptions", "receiving_yards", "receiving_tds"),
        "prior": "target_share",
    },
    "rushing": {
        "positions": ("RB",),
        "stats": ("carries", "rushing_yards", "rushing_tds"),
        "prior": "carry_share",
    },
}
USAGE_SHARE_MAX_RANK = 5

TEAM_ANCHOR_OUTPUT_COLS = [
    "team_passing_yards_pg_pred", "team_pass_attempts_pg_pred",
    "team_carries_pg_pred", "team_rushing_yards_pg_pred",
    "team_anchor_source_season", "team_anchor_lag_team",
    "team_anchor_provenance",
]

OUTPUT_COLUMNS = [
    "player_id", "display_name", "team", "position", "stat",
    "pred_pg", "pred_pg_low", "pred_pg_high",
    "pred_season", "pred_season_low", "pred_season_high",
    "source", "low_confidence", "rookie_tier", "interval_low_n_flag", "season",
    "team_changed", "roster_status",
    "depth_rank", "role", "formation_role", "depth_chart_status",
    "role_discount_applied",
    "nfl_depth_rank",
    "role_discount_factor",
    "athletic_tier",
    "team_pass_catch_ratio_pre_normalization", "team_pass_catch_pre_normalization_flag",
    "team_pass_catch_ratio", "team_pass_catch_coherence_flag",
    "receiving_share_capped",
    "receiving_share_normalized",
    "elite_correction_pg",
    "projected_games",
    "projected_games_raw",
    "projected_volume_games",
    "team_qb_raw_appearance_games", "team_qb_volume_allocation_direction",
    "team_qb_roster_resolved", "qb_volume_games_scale", "qb_volume_allocation_adjusted",
    "team_pass_attempts_pg_pred", "team_passing_yards_pg_pred",
    "team_carries_pg_pred", "team_rushing_yards_pg_pred",
    "team_anchor_source_season", "team_anchor_lag_team", "team_anchor_provenance",
    "wr_target_share", "te_target_share", "rb_target_share", "mix_source",
    "hierarchical_pass_scale", "within_group_target_share",
    "rb_carry_share", "qb_carry_share", "other_carry_share", "rush_mix_source",
    "hierarchical_rush_scale", "within_group_carry_share",
    "team_passing_volume_scale",
    "team_qb_attempt_anchor_fully_allocated",
    "team_rushing_volume_scale", "team_pass_receive_count_scale",
    "usage_share_blend_factor",
    "team_unmodeled_qb_volume_games", "team_unmodeled_qb_attempts_season",
    "team_unmodeled_qb_completions_season", "team_unmodeled_qb_passing_yards_season",
    "team_unmodeled_qb_passing_tds_season", "team_unmodeled_receiving_yards_season",
    "team_unmodeled_receptions_season", "team_unmodeled_receiving_tds_season",
    "team_unmodeled_carries_season", "team_unmodeled_rushing_yards_season",
    "coherence_receiver_exposure_basis",
    "target_depth_rank", "rookie_depth_band",
    "rookie_availability_cell_n", "rookie_availability_fallback_used",
    "rookie_vacancy_scale",
    "rookie_id_unresolved",
    "stat_constraint_applied",
]
