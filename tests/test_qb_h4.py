"""H4 unit / invariant tests (research-only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.qb_h3.archetype import classify_archetype_h3
from src.projection.qb_h4.decision_policy import H4_GATES, MODEL_ID, decision_policy_dict
from src.projection.qb_h4.designed_coverage import (
    load_coverage,
    merge_coverage_into_history,
)
from src.projection.qb_h4.experience import classify_experience
from src.projection.qb_h4.priors import blend_rates, h4_active_rates

REPO = Path(__file__).resolve().parents[1]


def test_model_id_isolated():
    assert MODEL_ID == "h4_insufficient_history_prior"
    assert decision_policy_dict()["predeclared_before_final_eval"] is True


def test_bootstrap_ci_policy_is_upper_bound_strictly_negative():
    assert H4_GATES.holdout_bootstrap_ci_must_exclude_zero is True
    text = decision_policy_dict()["paired_bootstrap_ci_policy"]
    assert "upper bound" in text and "< 0" in text


def test_experience_taxonomy_leakage_safe():
    hist = pd.DataFrame(
        [
            {"player_id": "v", "season": 2020, "active_starts": 16},
            {"player_id": "v", "season": 2021, "active_starts": 15},
            {"player_id": "v", "season": 2022, "active_starts": 14},
            {"player_id": "lim", "season": 2022, "active_starts": 5},
        ]
    )
    assert (
        classify_experience(
            player_id="v", target_season=2023, history=hist, is_rookie_at_cutoff=False
        )["experience_class"]
        == "established_veteran"
    )
    assert (
        classify_experience(
            player_id="lim", target_season=2023, history=hist, is_rookie_at_cutoff=False
        )["experience_class"]
        == "limited_history"
    )
    assert (
        classify_experience(
            player_id="new", target_season=2023, history=hist, is_rookie_at_cutoff=True
        )["experience_class"]
        == "rookie"
    )
    assert (
        classify_experience(
            player_id="ghost", target_season=2023, history=hist, is_rookie_at_cutoff=False
        )["experience_class"]
        == "insufficient_history"
    )
    assert (
        classify_experience(
            player_id="", target_season=2023, history=hist, is_rookie_at_cutoff=False
        )["experience_class"]
        == "missing_identity"
    )


def test_null_designed_never_pocket_with_coverage_merge():
    hist = pd.DataFrame(
        [
            {
                "player_id": "m",
                "season": 2021,
                "active_starts": 15,
                "attempts_per_active": 30.0,
                "carries_per_active": 10.0,
                "designed_carries_per_active": None,
                "scramble_per_dropback": None,
                "rushing_yards_per_active": 50.0,
                "rushing_tds_per_active": 0.3,
            },
            {
                "player_id": "m",
                "season": 2022,
                "active_starts": 15,
                "attempts_per_active": 30.0,
                "carries_per_active": 9.5,
                "designed_carries_per_active": None,
                "scramble_per_dropback": None,
                "rushing_yards_per_active": 48.0,
                "rushing_tds_per_active": 0.3,
            },
        ]
    )
    meta = classify_archetype_h3(hist, player_id="m", target_season=2023)
    assert meta["archetype"] != "pocket_passer"


def test_coverage_fixture_exists_and_has_2022():
    cov = load_coverage()
    assert not cov.empty
    assert 2022 in set(cov.season.astype(int))
    # Null policy: no silent zeros for missing players — only observed rows.
    assert cov["designed_carries"].notna().any() or cov["scramble_carries"].notna().any()


def test_merge_coverage_leaves_uncovered_nan():
    hist = pd.DataFrame(
        [
            {
                "player_id": "00-0034796",
                "season": 2019,
                "active_starts": 15,
                "attempts_per_active": 28.0,
                "carries_per_active": 11.0,
            }
        ]
    )
    merged = merge_coverage_into_history(hist)
    # 2019 has no PBP coverage in repo → remains uncovered / NaN designed.
    row = merged.iloc[0]
    assert row["designed_coverage_status"] == "uncovered"
    assert pd.isna(row["designed_carries_per_active"])


def test_rookie_prior_dominates_empty_player_rates():
    hist = pd.DataFrame(
        [
            {
                "player_id": "peer",
                "season": 2021,
                "active_starts": 14,
                "attempts_per_active": 33.0,
                "completions_per_active": 21.0,
                "passing_yards_per_active": 230.0,
                "passing_tds_per_active": 1.4,
                "interceptions_per_active": 0.6,
                "carries_per_active": 4.0,
                "rushing_yards_per_active": 18.0,
                "rushing_tds_per_active": 0.2,
            },
            {
                "player_id": "peer",
                "season": 2022,
                "active_starts": 15,
                "attempts_per_active": 34.0,
                "completions_per_active": 22.0,
                "passing_yards_per_active": 235.0,
                "passing_tds_per_active": 1.5,
                "interceptions_per_active": 0.7,
                "carries_per_active": 3.5,
                "rushing_yards_per_active": 15.0,
                "rushing_tds_per_active": 0.15,
            },
        ]
    )
    out = h4_active_rates(
        hist,
        player_id="rookieX",
        target_season=2023,
        experience_class="rookie",
        preseason_role="rookie_starter",
    )
    assert out["rates"]["attempts"] is not None
    assert out["shrink"] <= 0.25
    assert out["method"].startswith("experience_prior")


def test_blend_rates_shrink_capped_for_rookies():
    blended = blend_rates(
        player_rates={"attempts": 40.0, "carries": 2.0},
        peer_means={"attempts_per_active": 32.0, "carries_per_active": 3.0},
        sample_starts=100.0,
        experience_class="rookie",
    )
    assert blended["_shrink"] <= 0.25
