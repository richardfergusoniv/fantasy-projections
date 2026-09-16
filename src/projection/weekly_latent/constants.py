"""Milestone 1 constants for deterministic weekly schedule allocation.

Research/shadow only. Not a production default, freeze knob, or sealed-board input.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

SCHEMA_VERSION = "weekly_latent_m1_v1"
SCHEMA_VERSION_M2 = "weekly_latent_m2_v1"
MILESTONE = 1
MILESTONE_M2 = 2
SEASON_DEFAULT = 2026
REG_WEEKS = tuple(range(1, 19))
GAMES_PER_SEASON = 17

# Product split — do not wire these into M1 drivers.
VEGAS_ROLE = "external_weekly_benchmark"
LEAGUE_VALUE_ROLE = "season_scaffolding_prior"
MARKET_SANITY_ROLE = "later_empirical_bands_only"

# Home/away volume tilt before renormalization. Neutral/international games
# do not get the listed-home boost.
HOME_MULT = 1.03
AWAY_MULT = 0.97
NEUTRAL_MULT = 1.00

# Optional opponent priors are centered at 1.0. Clip to this band so a
# non-positive factor cannot drive raw weekly weights negative (λ_w=1).
OPPONENT_FACTOR_MIN = 0.5
OPPONENT_FACTOR_MAX = 1.5

# Preseason opponent-strength shrinkage toward 1.0. Early weeks keep more of
# the prior; weeks 15–18 are heavily shrunk because that far-out opponent
# form is not knowable in August.
SHRINKAGE_BY_WEEK_MAX: tuple[tuple[int, float], ...] = (
    (4, 1.00),
    (8, 0.70),
    (11, 0.45),
    (14, 0.25),
    (18, 0.10),
)

CONSERVATION_ATOL = 1e-6
CONSERVATION_RTOL = 1e-9

# Team volume stats allocated across weeks. Keys are output names; values are
# the sealed-board per-game column used to form the season total (pg * 17).
TEAM_VOLUME_PG_COLUMNS: Mapping[str, str] = {
    "team_pass_attempts": "team_pass_attempts_pg_pred",
    "team_rush_attempts": "team_carries_pg_pred",
    "team_passing_yards": "team_passing_yards_pg_pred",
    "team_rushing_yards": "team_rushing_yards_pg_pred",
}

# Player opportunity/yardage stats allocated as a share of a team pool.
# Completions/TDs/receptions are conversions from the player's own season rates.
PLAYER_SHARE_POOLS: Mapping[str, str] = {
    "attempts": "team_pass_attempts",
    "targets": "team_pass_attempts",
    "passing_yards": "team_passing_yards",
    "receiving_yards": "team_passing_yards",
    "carries": "team_rush_attempts",
    "rushing_yards": "team_rushing_yards",
}

CONVERSION_RATES: Mapping[str, tuple[str, str]] = {
    # stat: (numerator season col, opportunity season col)
    "completions": ("completions", "attempts"),
    "passing_tds": ("passing_tds", "attempts"),
    "interceptions": ("interceptions", "attempts"),
    "receptions": ("receptions", "targets"),
    "receiving_tds": ("receiving_tds", "targets"),
    "rushing_tds": ("rushing_tds", "carries"),
}

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# Same aliases as src/projection/weekly/data/teams.py (avoid importing polars).
TEAM_ABBR_ALIASES: Mapping[str, str] = {
    "AZ": "ARI",
    "ARZ": "ARI",
    "WSH": "WAS",
    "JAC": "JAX",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}

# Conference/division from src/team_stats/prepare.py TEAM_META.
TEAM_DIVISIONS: Mapping[str, tuple[str, str]] = {
    "ARI": ("NFC", "West"),
    "ATL": ("NFC", "South"),
    "BAL": ("AFC", "North"),
    "BUF": ("AFC", "East"),
    "CAR": ("NFC", "South"),
    "CHI": ("NFC", "North"),
    "CIN": ("AFC", "North"),
    "CLE": ("AFC", "North"),
    "DAL": ("NFC", "East"),
    "DEN": ("AFC", "West"),
    "DET": ("NFC", "North"),
    "GB": ("NFC", "North"),
    "HOU": ("AFC", "South"),
    "IND": ("AFC", "South"),
    "JAX": ("AFC", "South"),
    "KC": ("AFC", "West"),
    "LA": ("NFC", "West"),
    "LAC": ("AFC", "West"),
    "LV": ("AFC", "West"),
    "MIA": ("AFC", "East"),
    "MIN": ("NFC", "North"),
    "NE": ("AFC", "East"),
    "NO": ("NFC", "South"),
    "NYG": ("NFC", "East"),
    "NYJ": ("AFC", "East"),
    "PHI": ("NFC", "East"),
    "PIT": ("AFC", "North"),
    "SEA": ("NFC", "West"),
    "SF": ("NFC", "West"),
    "TB": ("NFC", "South"),
    "TEN": ("AFC", "South"),
    "WAS": ("NFC", "East"),
}

# Hand-maintained copy of team-prefixed SAME_WEEK_OUTCOME_DENYLIST names.
# Kept as a literal so this shadow module does not import
# feature_outcome_split (polars). tests/test_weekly_latent_m1.py asserts
# this set matches those canonical team aggregates (plus the PR #70 hole
# columns if that denylist has not landed on the branch yet).
# Lagged team_pass_rate_l5 is a different, safe column. M1 outputs use
# team_pass_attempts as allocated projections, not these realized names.
FORBIDDEN_SAME_WEEK_TRAINING_FEATURES = frozenset(
    {
        "team_targets",
        "team_carries",
        "team_attempts",
        "team_air_yards",
    }
)

# M1 rows only carry schedule / home-away / bye / opponent scaffolding plus
# allocated season volume. That package is knowable at the sealed preseason
# board snapshot, not at kickoff. M2 must overwrite `available_at` when it
# attaches opponent priors or in-season updates (Tuesday vs 90-min-pre-kickoff
# are different vintages; M1 does not distinguish them).
M1_AVAILABLE_AT = "2026-08-30T00:00:00+00:00"

# 2025 REG week 18 finished 2026-01-04/05. Prior-season defensive EPA from
# that slate is knowable then — earlier than the sealed board, not later.
# In-season lagged EPA must stamp a later cutoff; never reuse M1_AVAILABLE_AT
# blindly once a later prior is attached (design lock §6).
PRIOR_SEASON_EPA_AVAILABLE_AT = "2026-01-05T00:00:00+00:00"

# Schedule rest / international / home-away are on the 2026 REG fixture
# before kickoff. Treat them as the same preseason vintage as the sealed board.
M2_SCHEDULE_ENV_AVAILABLE_AT = M1_AVAILABLE_AT

# Game-level nflverse passing_epa / rushing_epa (team-week totals, not per
# play). z-score × kappa → volume factor. ±1 SD ≈ ±6% team volume.
EPA_VOLUME_KAPPA = 0.06
ENV_FACTOR_MIN = 0.85
ENV_FACTOR_MAX = 1.15

# Rest and international are small environment tilts, not Vegas lines.
REST_SHORT_DAYS = 6
REST_LONG_DAYS = 10
REST_SHORT_MULT = 0.97
REST_NORMAL_MULT = 1.00
REST_LONG_MULT = 1.02
INTL_TRAVEL_MULT = 0.98

# Columns that must never drive M2 volume. Same-week box is in
# FORBIDDEN_SAME_WEEK_TRAINING_FEATURES; this set is market / ADP.
FORBIDDEN_MARKET_DRIVERS = frozenset(
    {
        "adp",
        "adp_points",
        "consensus_adp",
        "vegas_total",
        "season_win_total",
        "implied_team_total",
        "implied_opp_total",
        "spread_line",
        "total_line",
    }
)

DEFAULT_SEALED_NAMESPACE = "v2_baseline_20260830"
DEFAULT_SEALED_PROJECTIONS_REL = (
    "draft_assistant/data/releases/v2_baseline_20260830/projections_2026.csv"
)
DEFAULT_SCHEDULE_FIXTURE_REL = (
    "src/projection/weekly_latent/fixtures/nfl_schedules_2026_reg.csv"
)
DEFAULT_OPP_EPA_PRIOR_REL = (
    "src/projection/weekly_latent/fixtures/opp_def_epa_prior_2025.csv"
)
DEFAULT_OUTPUT_REL = "output/shadow_weekly_schedule_m1"
DEFAULT_OUTPUT_REL_M2 = "output/shadow_weekly_schedule_m2"

PRODUCTION_HASH_PATHS = (
    "draft_assistant/data/active_release_2026.json",
    "draft_assistant/data/releases/v2_baseline_20260830/release_bundle_manifest.json",
    "output/accuracy_first_2026/freeze_manifest.json",
    "output/accuracy_first_2026/ensemble_weights.json",
    "output/accuracy_first_2026/application_contract.json",
)


def opponent_shrinkage_lambda(week: int) -> float:
    """Return preseason opponent-strength weight λ_w ∈ (0, 1]."""
    w = int(week)
    for max_week, lam in SHRINKAGE_BY_WEEK_MAX:
        if w <= max_week:
            return float(lam)
    return 0.10


def parse_available_at(stamp: str) -> datetime:
    """Parse an ISO-8601 cutoff; naive stamps are treated as UTC."""
    text = str(stamp).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_available_at(stamp: datetime) -> str:
    utc = stamp.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def max_available_at(*stamps: str | None) -> str:
    """Latest cutoff among feature vintages. Fails if nothing was provided."""
    present = [parse_available_at(s) for s in stamps if s]
    if not present:
        raise ValueError("max_available_at requires at least one cutoff")
    return format_available_at(max(present))


def normalize_team_abbr(team: str | None) -> str | None:
    if team is None:
        return None
    text = str(team).strip().upper()
    if not text:
        return None
    return str(TEAM_ABBR_ALIASES.get(text, text))
