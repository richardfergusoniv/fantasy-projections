"""Hand-calculated league scoring examples.

Every expected value below is computed by hand in the comment above the
assertion. These are contract tests: if the compiler or the draw scorer changes
behaviour for a real Sleeper rule set, one of these fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app.scoring.compiler import (
    UnsupportedScoringRule,
    compile_sleeper_scoring,
    require_publishable,
    score_stat_draw,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scoring"


def contract_for(name: str):
    payload = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return compile_sleeper_scoring(payload["scoring_settings"], payload["roster_positions"])


# --------------------------------------------------------------------- offense


def test_standard_half_ppr_quarterback():
    contract = contract_for("standard")
    # 300 pass yds * 0.04 = 12.0
    # 2 pass TD    * 4    =  8.0
    # 1 INT        * -2   = -2.0
    # 25 rush yds  * 0.1  =  2.5
    #                       -----
    #                        20.5
    points = score_stat_draw(
        {
            "pass_yards": 300,
            "pass_tds": 2,
            "pass_ints": 1,
            "rush_yards": 25,
        },
        contract,
        position="QB",
    )
    assert points == pytest.approx(20.5)


def test_standard_half_ppr_running_back_with_fumble():
    contract = contract_for("standard")
    # 95 rush yds * 0.1 =  9.5
    # 1 rush TD   * 6   =  6.0
    # 4 rec       * 0.5 =  2.0
    # 32 rec yds  * 0.1 =  3.2
    # 1 fum lost  * -2  = -2.0
    #                     -----
    #                      18.7
    points = score_stat_draw(
        {
            "rush_yards": 95,
            "rush_tds": 1,
            "receptions": 4,
            "rec_yards": 32,
            "fumbles_lost": 1,
        },
        contract,
        position="RB",
    )
    assert points == pytest.approx(18.7)


def test_full_ppr_differs_from_half_ppr_by_reception_count():
    half = contract_for("standard")
    full = contract_for("superflex")  # full PPR
    draw = {"rush_yards": 95, "rush_tds": 1, "receptions": 4, "rec_yards": 32, "fumbles_lost": 1}
    # Only the reception rule differs: 4 receptions * (1.0 - 0.5) = +2.0
    assert score_stat_draw(draw, full, position="RB") - score_stat_draw(
        draw, half, position="RB"
    ) == pytest.approx(2.0)


def test_points_per_first_down_receiver():
    contract = contract_for("ppfd")
    # 7 rec        * 0.5 =  3.5
    # 88 rec yds   * 0.1 =  8.8
    # 1 rec TD     * 6   =  6.0
    # 5 rec 1st Ds * 0.5 =  2.5
    #                      -----
    #                       20.8
    points = score_stat_draw(
        {
            "receptions": 7,
            "rec_yards": 88,
            "rec_tds": 1,
            "rec_first_downs": 5,
        },
        contract,
        position="WR",
    )
    assert points == pytest.approx(20.8)


def test_ppfd_scores_all_three_first_down_families():
    contract = contract_for("ppfd")
    stats = {"pass_first_downs": 12, "rush_first_downs": 3, "rec_first_downs": 4}
    # (12 + 3 + 4) first downs * 0.5 = 9.5
    assert score_stat_draw(stats, contract, position="QB") == pytest.approx(9.5)


def test_first_downs_are_ignored_by_a_league_that_does_not_score_them():
    contract = contract_for("standard")
    assert score_stat_draw({"rec_first_downs": 9}, contract, position="WR") == 0.0


# ----------------------------------------------------------- yardage milestones


def test_yardage_bonus_applies_cumulatively_at_each_milestone():
    contract = contract_for("yardage_bonus")
    # 205 rush yds * 0.1                     = 20.5
    # 1 rush TD    * 6                       =  6.0
    # 2 rec        * 0.5                     =  1.0
    # 15 rec yds   * 0.1                     =  1.5
    # bonus_rush_yd_100 (205 >= 100)         =  3.0
    # bonus_rush_yd_200 (205 >= 200)         =  3.0
    #                                          -----
    #                                           35.0
    points = score_stat_draw(
        {"rush_yards": 205, "rush_tds": 1, "receptions": 2, "rec_yards": 15},
        contract,
        position="RB",
    )
    assert points == pytest.approx(35.0)


def test_yardage_bonus_boundary_is_inclusive():
    contract = contract_for("yardage_bonus")
    below = score_stat_draw({"rush_yards": 99}, contract, position="RB")
    at = score_stat_draw({"rush_yards": 100}, contract, position="RB")
    # 1 extra yard * 0.1 = 0.1, plus the 3.0 milestone bonus.
    assert at - below == pytest.approx(3.1)


def test_bonus_expectation_must_be_computed_per_draw_not_on_the_mean():
    """The core reason milestone rules cannot be applied to a mean.

    Two equally likely draws of 99 and 101 rush yards have a mean of 100, which
    would earn the bonus every week if the rule were applied to the mean. The
    true expectation is half the bonus.
    """
    contract = contract_for("yardage_bonus")
    draws = [{"rush_yards": 99.0}, {"rush_yards": 101.0}]
    per_draw_expectation = sum(
        score_stat_draw(d, contract, position="RB") for d in draws
    ) / len(draws)
    mean_stats = {"rush_yards": 100.0}
    naive_mean_scoring = score_stat_draw(mean_stats, contract, position="RB")

    # Per draw: (9.9 + 0) + (10.1 + 3) = 23.0 -> 11.5
    assert per_draw_expectation == pytest.approx(11.5)
    # Naive: 10.0 + 3.0 = 13.0 -- overstated by the full bonus half the time.
    assert naive_mean_scoring == pytest.approx(13.0)
    assert naive_mean_scoring > per_draw_expectation


# ----------------------------------------------------------------------- kicker


def test_kicker_distance_bands():
    contract = contract_for("k_dst")
    # 2 FG 30-39 * 3 =  6.0
    # 1 FG 40-49 * 4 =  4.0
    # 1 FG 50+   * 5 =  5.0
    # 3 XP made  * 1 =  3.0
    # 1 XP miss  * -1= -1.0
    #                  -----
    #                   17.0
    points = score_stat_draw(
        {
            "fgm_30_39": 2,
            "fgm_40_49": 1,
            "fgm_50p": 1,
            "xpm": 3,
            "xpmiss": 1,
        },
        contract,
        position="K",
    )
    assert points == pytest.approx(17.0)


# ------------------------------------------------------------------ team defense


@pytest.mark.parametrize(
    ("points_allowed", "bracket_points", "expected_total"),
    [
        # 4 sacks*1 + 2 INT*2 + 1 FR*2 = 10, plus exactly one points-allowed tier.
        (0, 10, 20.0),
        (3, 7, 17.0),
        (10, 4, 14.0),
        (17, 1, 11.0),
        (24, 0, 10.0),
        (30, -1, 9.0),
        (38, -4, 6.0),
    ],
)
def test_tiered_points_allowed_awards_exactly_one_bracket(
    points_allowed, bracket_points, expected_total
):
    contract = contract_for("k_dst")
    points = score_stat_draw(
        {
            "sacks": 4,
            "interceptions": 2,
            "fumble_recoveries": 1,
            "points_allowed": points_allowed,
        },
        contract,
        position="DEF",
    )
    assert points == pytest.approx(expected_total)
    assert points - 10.0 == pytest.approx(bracket_points)


def test_absent_points_allowed_is_not_treated_as_a_shutout():
    """A missing stat must contribute nothing.

    Reading an absent ``points_allowed`` as 0.0 would silently award the
    shutout bracket to every defense whose draw omits the field.
    """
    contract = contract_for("k_dst")
    points = score_stat_draw({"sacks": 1}, contract, position="DEF")
    assert points == pytest.approx(1.0)


def test_simple_defense_league_has_no_points_allowed_bracket():
    contract = contract_for("standard")
    assert contract.bracket_rules == []
    # 3 sacks*1 + 1 INT*2 + 1 def TD*6 + 1 safety*2 = 13
    points = score_stat_draw(
        {"sacks": 3, "interceptions": 1, "def_tds": 1, "safeties": 1},
        contract,
        position="DEF",
    )
    assert points == pytest.approx(13.0)


# ------------------------------------------------------------- negative scoring


def test_total_can_be_negative():
    contract = contract_for("standard")
    # 100 pass yds * 0.04 =  4.0
    # 4 INT        * -2   = -8.0
    # 2 fum lost   * -2   = -4.0
    #                       -----
    #                        -8.0
    points = score_stat_draw(
        {"pass_yards": 100, "pass_ints": 4, "fumbles_lost": 2}, contract, position="QB"
    )
    assert points == pytest.approx(-8.0)


# ------------------------------------------------- position-conditional premiums


def test_tight_end_reception_premium_applies_only_to_tight_ends():
    contract = compile_sleeper_scoring(
        {"rec": 0.5, "bonus_rec_te": 0.5, "rec_yd": 0.1},
        ["QB", "RB", "WR", "TE", "FLEX", "BN"],
    )
    draw = {"receptions": 6, "rec_yards": 70}
    # WR: 6*0.5 + 7.0 = 10.0 ; TE: 6*(0.5+0.5) + 7.0 = 13.0
    assert score_stat_draw(draw, contract, position="WR") == pytest.approx(10.0)
    assert score_stat_draw(draw, contract, position="TE") == pytest.approx(13.0)
    # With no position supplied a restricted rule must not be applied.
    assert score_stat_draw(draw, contract) == pytest.approx(10.0)


# --------------------------------------------------------------- roster slots


def test_superflex_slot_accepts_a_quarterback_and_flex_does_not():
    contract = contract_for("superflex")
    slots = {slot.slot: slot for slot in contract.scoring_slots}
    assert "SUPER_FLEX" in slots
    assert "QB" in slots["SUPER_FLEX"].eligible_positions
    assert "QB" not in slots["FLEX"].eligible_positions


def test_non_scoring_slots_are_excluded_from_the_starting_lineup():
    contract = contract_for("superflex")
    assert any(slot.slot == "TAXI" for slot in contract.roster_slots)
    assert all(slot.slot != "TAXI" for slot in contract.scoring_slots)
    # QB, RB, RB, WR, WR, TE, FLEX, SUPER_FLEX, K, DEF = 10 starting seats.
    assert contract.starting_slot_count == 10


def test_leagues_without_kickers_do_not_expose_a_kicker_slot():
    contract = contract_for("ppfd")
    assert "K" not in contract.eligible_positions()
    assert "DEF" not in contract.eligible_positions()


# ----------------------------------------------------------------- fail closed


def test_unsupported_nonzero_key_blocks_publication_with_a_precise_error():
    contract = compile_sleeper_scoring(
        {"rec": 1.0, "idp_tkl_solo": 1.5, "def_forced_punt": 2.0},
        ["QB", "RB", "WR", "TE", "BN"],
    )
    assert sorted(contract.unsupported_keys) == ["def_forced_punt", "idp_tkl_solo"]
    assert contract.blocks_publication is True
    with pytest.raises(UnsupportedScoringRule) as exc:
        require_publishable(contract, "league-x")
    message = str(exc.value)
    assert "league-x" in message
    assert "idp_tkl_solo" in message
    assert "def_forced_punt" in message


def test_zero_valued_unsupported_key_does_not_block_publication():
    contract = compile_sleeper_scoring(
        {"rec": 1.0, "idp_tkl_solo": 0}, ["QB", "RB", "WR", "TE", "BN"]
    )
    assert contract.unsupported_keys == []
    require_publishable(contract)


def test_unmappable_roster_slot_blocks_publication_instead_of_being_dropped():
    contract = compile_sleeper_scoring(
        {"rec": 1.0}, ["QB", "RB", "WR", "TE", "IDP_FLEX", "LB", "BN"]
    )
    assert sorted(contract.unsupported_slots) == ["IDP_FLEX", "LB"]
    assert contract.blocks_publication is True
    with pytest.raises(UnsupportedScoringRule):
        require_publishable(contract)


def test_boolean_scoring_value_is_not_silently_read_as_zero():
    contract = compile_sleeper_scoring({"rec": True}, ["QB", "BN"])
    assert contract.unsupported_keys == ["rec"]


def test_threshold_comparison_is_honoured():
    from src.app.scoring.contract import ScoringContract, ThresholdRule

    strict = ScoringContract(
        threshold_rules=[
            ThresholdRule(stat="rush_yards", threshold=100, bonus_points=3, comparison=">")
        ]
    ).finalize()
    inclusive = ScoringContract(
        threshold_rules=[
            ThresholdRule(stat="rush_yards", threshold=100, bonus_points=3, comparison=">=")
        ]
    ).finalize()
    assert score_stat_draw({"rush_yards": 100}, strict) == 0.0
    assert score_stat_draw({"rush_yards": 100}, inclusive) == pytest.approx(3.0)
    assert score_stat_draw({"rush_yards": 101}, strict) == pytest.approx(3.0)


def test_contract_hash_is_deterministic_and_rule_sensitive():
    a = compile_sleeper_scoring({"rec": 1.0, "rec_yd": 0.1}, ["QB", "BN"])
    b = compile_sleeper_scoring({"rec_yd": 0.1, "rec": 1.0}, ["QB", "BN"])
    c = compile_sleeper_scoring({"rec": 0.5, "rec_yd": 0.1}, ["QB", "BN"])
    assert a.contract_hash == b.contract_hash
    assert a.contract_hash != c.contract_hash
