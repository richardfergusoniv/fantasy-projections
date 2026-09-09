"""Versioned policy and gate thresholds for weekly player props."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.projection.market_quotes import QuotePolicy

POLICY_VERSION = "weekly_props_policy_v1"
GATE_VERSION = "weekly_props_gates_v1"
MODEL_VERSION = "weekly_props_v1"
ARTIFACT_MODE = "market"


@dataclass(frozen=True)
class WeeklyPropsPolicy:
    version: str = POLICY_VERSION
    quote: QuotePolicy = field(
        default_factory=lambda: QuotePolicy(
            # Weekly yardage is ~1/17 of season; tighten absolute bands.
            skewed_odds_gap=0.35,
            one_sided_longshot_threshold=100.0,
            count_conflict_rel=0.20,
            count_conflict_abs=0.5,
            count_scale_ceiling=10.0,
            yard_conflict_rel_hard=0.45,
            yard_conflict_abs_floor=1.0,
            yard_upward_rel=0.20,
            yard_upward_abs=25.0,
            robust_median_rel=0.30,
            robust_median_abs_floor=15.0,
        )
    )
    max_snapshot_age_hours: float = 36.0
    min_source_success_count: int = 1
    min_distinct_books_per_market: int = 1
    require_two_sided: bool = True
    allow_odds_less_exception: bool = False
    min_identity_join_rate: float = 0.70
    min_candidate_players: int = 50
    max_aggregate_baseline_movement: float = 0.35
    max_top_n_displacement: int = 8
    top_n_for_displacement: int = 24
    # Per-position absolute sanity caps for a single game.
    sanity_caps: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "QB": {
                "pass_yards": 450.0,
                "pass_tds": 6.0,
                "rush_yards": 150.0,
                "rush_tds": 3.0,
            },
            "RB": {
                "rush_yards": 220.0,
                "rush_tds": 4.0,
                "rec_yards": 150.0,
                "rec_tds": 3.0,
                "receptions": 15.0,
            },
            "WR": {
                "rec_yards": 220.0,
                "rec_tds": 4.0,
                "receptions": 15.0,
                "rush_yards": 80.0,
            },
            "TE": {
                "rec_yards": 160.0,
                "rec_tds": 3.0,
                "receptions": 12.0,
            },
        }
    )
    scoring_markets: tuple[str, ...] = (
        "pass_yards",
        "pass_tds",
        "rush_yards",
        "rush_tds",
        "rec_yards",
        "rec_tds",
        "receptions",
    )
    # Components that may come from baseline when props omit them.
    baseline_only_components: tuple[str, ...] = (
        "pass_ints",
        "fumbles",
        "pass_attempts",
        "pass_completions",
        "rush_attempts",
        "targets",
    )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["quote"] = asdict(self.quote)
        return payload


DEFAULT_WEEKLY_POLICY = WeeklyPropsPolicy()
