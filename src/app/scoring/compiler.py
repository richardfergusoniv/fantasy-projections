"""Compile Sleeper scoring payloads into deterministic contracts.

Design rules:

* Ordinary offensive, kicker, and team-defense rules are mapped exactly. They are
  never approximated.
* Nonlinear rules (yardage milestones, tiered points/yards allowed) compile to
  threshold/bracket rules so they can be evaluated on each simulation draw. They
  are never applied to mean statistics.
* Anything nonzero that cannot be mapped lands in `unsupported_keys`, and any
  roster slot that cannot be mapped lands in `unsupported_slots`. Both block
  publication for that league. Failing closed is deliberate: silently dropping a
  scoring rule would produce confidently wrong recommendations.
"""

from __future__ import annotations

from typing import Any

from src.app.scoring.contract import (
    BracketRule,
    DefenseRule,
    LinearRule,
    RosterSlot,
    ScoringContract,
    ThresholdRule,
)

# Keys that identify a flat Sleeper scoring_settings dict stored at snapshot root.
_FLAT_SCORING_MARKERS = frozenset(
    {"rec", "rush_yd", "pass_yd", "pts_allow_0", "fgm_0_19", "def_td", "rush_fd", "rec_fd"}
)


def scoring_settings_from_snapshot(raw_json: dict[str, Any] | None) -> dict[str, Any]:
    """Return Sleeper scoring_settings from a rule snapshot or league payload.

    Live sync persists the flat scoring dict on ``LeagueRuleSnapshot.raw_json``.
    Seed data and ``League.raw_json`` nest the same dict under ``scoring_settings``.
    """
    raw = dict(raw_json or {})
    nested = raw.get("scoring_settings")
    if isinstance(nested, dict) and nested:
        return dict(nested)
    if _FLAT_SCORING_MARKERS & raw.keys():
        return raw
    return dict(nested) if isinstance(nested, dict) else {}

# --------------------------------------------------------------------- linear
# Sleeper scoring key -> canonical stat name.
LINEAR_KEY_MAP: dict[str, str] = {
    # Passing
    "pass_yd": "pass_yards",
    "pass_td": "pass_tds",
    "pass_int": "pass_ints",
    "pass_fd": "pass_first_downs",
    "pass_2pt": "pass_two_pt",
    "pass_cmp": "pass_completions",
    "pass_att": "pass_attempts",
    "pass_inc": "pass_incompletions",
    "pass_sack": "pass_sacks_taken",
    "pass_td_40p": "pass_tds_40p",
    "pass_td_50p": "pass_tds_50p",
    # Rushing
    "rush_yd": "rush_yards",
    "rush_td": "rush_tds",
    "rush_fd": "rush_first_downs",
    "rush_2pt": "rush_two_pt",
    "rush_att": "rush_attempts",
    "rush_td_40p": "rush_tds_40p",
    "rush_td_50p": "rush_tds_50p",
    # Receiving
    "rec": "receptions",
    "rec_yd": "rec_yards",
    "rec_td": "rec_tds",
    "rec_fd": "rec_first_downs",
    "rec_2pt": "rec_two_pt",
    "rec_tgt": "targets",
    "rec_td_40p": "rec_tds_40p",
    "rec_td_50p": "rec_tds_50p",
    # Fumbles
    "fum": "fumbles",
    "fum_lost": "fumbles_lost",
    "fum_rec": "fumble_recoveries_offense",
    "fum_rec_td": "fumble_recovery_tds",
    # Kicking
    "fgm": "fgm",
    "fgm_0_19": "fgm_0_19",
    "fgm_20_29": "fgm_20_29",
    "fgm_30_39": "fgm_30_39",
    "fgm_40_49": "fgm_40_49",
    "fgm_50p": "fgm_50p",
    "fgm_50_59": "fgm_50_59",
    "fgm_60p": "fgm_60p",
    "fgmiss": "fgmiss",
    "fgmiss_0_19": "fgmiss_0_19",
    "fgmiss_20_29": "fgmiss_20_29",
    "fgmiss_30_39": "fgmiss_30_39",
    "fgmiss_40_49": "fgmiss_40_49",
    "fgmiss_50p": "fgmiss_50p",
    "xpm": "xpm",
    "xpmiss": "xpmiss",
    # Special teams credited to a skill player
    "st_td": "st_tds",
    "st_ff": "st_forced_fumbles",
    "st_fum_rec": "st_fumble_recoveries",
}

#: Sleeper position-premium keys -> (stat, eligible positions).
POSITION_BONUS_KEY_MAP: dict[str, tuple[str, tuple[str, ...]]] = {
    "bonus_rec_te": ("receptions", ("TE",)),
    "bonus_rec_rb": ("receptions", ("RB",)),
    "bonus_rec_wr": ("receptions", ("WR",)),
}

# ------------------------------------------------------------------ thresholds
#: Sleeper milestone key -> (stat, threshold).
THRESHOLD_KEY_MAP: dict[str, tuple[str, float]] = {
    "bonus_pass_yd_300": ("pass_yards", 300.0),
    "bonus_pass_yd_400": ("pass_yards", 400.0),
    "bonus_rush_yd_100": ("rush_yards", 100.0),
    "bonus_rush_yd_200": ("rush_yards", 200.0),
    "bonus_rec_yd_100": ("rec_yards", 100.0),
    "bonus_rec_yd_200": ("rec_yards", 200.0),
    "bonus_rush_rec_yd_100": ("rush_rec_yards", 100.0),
    "bonus_rush_rec_yd_200": ("rush_rec_yards", 200.0),
    "bonus_pass_cmp_25": ("pass_completions", 25.0),
    "bonus_rush_att_20": ("rush_attempts", 20.0),
    "bonus_tkl_10p": ("tackles", 10.0),
}

# -------------------------------------------------------------------- brackets
#: Tiered team-defense keys -> (stat, group, lower, upper). Upper None = open.
BRACKET_KEY_MAP: dict[str, tuple[str, str, float, float | None]] = {
    "pts_allow_0": ("points_allowed", "points_allowed", 0.0, 0.0),
    "pts_allow_1_6": ("points_allowed", "points_allowed", 1.0, 6.0),
    "pts_allow_7_13": ("points_allowed", "points_allowed", 7.0, 13.0),
    "pts_allow_14_20": ("points_allowed", "points_allowed", 14.0, 20.0),
    "pts_allow_21_27": ("points_allowed", "points_allowed", 21.0, 27.0),
    "pts_allow_28_34": ("points_allowed", "points_allowed", 28.0, 34.0),
    "pts_allow_35p": ("points_allowed", "points_allowed", 35.0, None),
    "yds_allow_0_100": ("yards_allowed", "yards_allowed", 0.0, 99.0),
    "yds_allow_100_199": ("yards_allowed", "yards_allowed", 100.0, 199.0),
    "yds_allow_200_299": ("yards_allowed", "yards_allowed", 200.0, 299.0),
    "yds_allow_300_349": ("yards_allowed", "yards_allowed", 300.0, 349.0),
    "yds_allow_350_399": ("yards_allowed", "yards_allowed", 350.0, 399.0),
    "yds_allow_400_449": ("yards_allowed", "yards_allowed", 400.0, 449.0),
    "yds_allow_450_499": ("yards_allowed", "yards_allowed", 450.0, 499.0),
    "yds_allow_500_549": ("yards_allowed", "yards_allowed", 500.0, 549.0),
    "yds_allow_550p": ("yards_allowed", "yards_allowed", 550.0, None),
}

# ------------------------------------------------------------------------- DST
#: Team-defense per-unit keys. Sleeper uses both bare and `def_`-prefixed names.
DST_KEY_MAP: dict[str, str] = {
    "def_sack": "sacks",
    "sack": "sacks",
    "def_int": "interceptions",
    "int": "interceptions",
    "def_fr": "fumble_recoveries",
    "fum_rec_def": "fumble_recoveries",
    "def_ff": "forced_fumbles",
    "ff": "forced_fumbles",
    "def_td": "def_tds",
    "def_st_td": "st_tds",
    "def_st_ff": "st_forced_fumbles",
    "def_st_fum_rec": "st_fumble_recoveries",
    "def_2pt": "def_two_pt",
    "def_pr_td": "punt_return_tds",
    "def_kr_td": "kick_return_tds",
    "safe": "safeties",
    "def_safe": "safeties",
    "blk_kick": "blocked_kicks",
    "def_blk_kick": "blocked_kicks",
    "blk_kick_ret_yd": "blocked_kick_return_yards",
    "def_pts_allowed": "points_allowed",
    "def_yds_allowed": "yards_allowed",
    "def_4_and_stop": "fourth_down_stops",
    "def_3_and_out": "three_and_outs",
    "def_pass_def": "passes_defended",
    "tkl_loss": "tackles_for_loss",
}

# ----------------------------------------------------------------------- slots
#: Sleeper roster slot -> (canonical slot, eligible positions).
SLOT_MAP: dict[str, tuple[str, tuple[str, ...]]] = {
    "QB": ("QB", ("QB",)),
    "RB": ("RB", ("RB",)),
    "WR": ("WR", ("WR",)),
    "TE": ("TE", ("TE",)),
    "FLEX": ("FLEX", ("RB", "WR", "TE")),
    "WRRB_FLEX": ("FLEX", ("RB", "WR")),
    "WRRB_WRT": ("FLEX", ("RB", "WR", "TE")),
    "REC_FLEX": ("REC_FLEX", ("WR", "TE")),
    "SUPER_FLEX": ("SUPER_FLEX", ("QB", "RB", "WR", "TE")),
    "K": ("K", ("K",)),
    "DEF": ("DEF", ("DEF",)),
    "BN": ("BN", ("QB", "RB", "WR", "TE", "K", "DEF")),
    "IR": ("IR", ("QB", "RB", "WR", "TE", "K", "DEF")),
    "TAXI": ("TAXI", ("QB", "RB", "WR", "TE", "K", "DEF")),
}


def _is_zero(value: Any) -> bool:
    """True when a Sleeper scoring value carries no rule.

    `value in (0, 0.0, None)` would also swallow `False`, and a bool is not a
    valid scoring weight, so booleans are treated as unmapped rather than zero.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compile_sleeper_scoring(
    scoring_settings: dict[str, Any],
    roster_positions: list[str] | None = None,
) -> ScoringContract:
    contract = ScoringContract()
    roster_positions = roster_positions or []

    for key, value in (scoring_settings or {}).items():
        if _is_zero(value):
            continue
        points = _numeric(value)
        if points is None:
            contract.unsupported_keys.append(key)
            continue

        if key in LINEAR_KEY_MAP:
            contract.linear_rules.append(
                LinearRule(stat=LINEAR_KEY_MAP[key], points_per_unit=points)
            )
            continue
        if key in POSITION_BONUS_KEY_MAP:
            stat, positions = POSITION_BONUS_KEY_MAP[key]
            contract.linear_rules.append(
                LinearRule(stat=stat, points_per_unit=points, positions=positions)
            )
            continue
        if key in THRESHOLD_KEY_MAP:
            stat, threshold = THRESHOLD_KEY_MAP[key]
            contract.threshold_rules.append(
                ThresholdRule(stat=stat, threshold=threshold, bonus_points=points)
            )
            continue
        if key in BRACKET_KEY_MAP:
            stat, group, lower, upper = BRACKET_KEY_MAP[key]
            contract.bracket_rules.append(
                BracketRule(stat=stat, lower=lower, upper=upper, points=points, group=group)
            )
            continue
        if key in DST_KEY_MAP:
            contract.dst_rules.append(
                DefenseRule(stat=DST_KEY_MAP[key], points_per_unit=points)
            )
            continue
        contract.unsupported_keys.append(key)

    slot_counts: dict[str, int] = {}
    slot_order: list[str] = []
    for slot in roster_positions:
        if slot not in slot_counts:
            slot_order.append(slot)
        slot_counts[slot] = slot_counts.get(slot, 0) + 1

    for slot_name in slot_order:
        count = slot_counts[slot_name]
        mapped = SLOT_MAP.get(slot_name)
        if mapped is None:
            contract.unsupported_slots.append(slot_name)
            continue
        slot_type, eligible = mapped
        contract.roster_slots.append(
            RosterSlot(
                slot=slot_type,  # type: ignore[arg-type]
                count=count,
                eligible_positions=eligible,  # type: ignore[arg-type]
            )
        )

    return contract.finalize()


class UnsupportedScoringRule(ValueError):
    """Raised when a league's rules cannot be reproduced exactly."""


def require_publishable(contract: ScoringContract, league_id: str | None = None) -> None:
    """Fail closed with a precise message when a contract is incomplete."""
    reason = contract.publication_block_reason()
    if reason is None:
        return
    prefix = f"league {league_id}: " if league_id else ""
    raise UnsupportedScoringRule(
        f"{prefix}cannot publish recommendations because the Sleeper rules are not "
        f"fully mapped ({reason}). Map the rule or explicitly waive it before publishing."
    )


def score_stat_draw(
    draw: dict[str, float],
    contract: ScoringContract,
    position: str | None = None,
) -> float:
    """Score a single simulation draw under a league contract.

    Threshold and bracket rules are evaluated against this draw's realized
    values, never against a mean, so milestone bonuses and tiered defensive
    scoring have correct expectations. A rule whose stat is absent from the draw
    contributes nothing — an absent `points_allowed` must not be read as a
    shutout.
    """
    total = 0.0

    for rule in contract.linear_rules:
        if not rule.applies_to(position):
            continue
        value = draw.get(rule.stat)
        if value is None:
            continue
        total += float(value) * rule.points_per_unit

    for rule in contract.threshold_rules:
        value = draw.get(rule.stat)
        if value is None and rule.stat == "rush_rec_yards":
            rush = draw.get("rush_yards")
            rec = draw.get("rec_yards")
            if rush is None and rec is None:
                continue
            value = float(rush or 0.0) + float(rec or 0.0)
        if value is None:
            continue
        if rule.is_met(float(value)):
            total += rule.bonus_points

    # Brackets are mutually exclusive within a group: at most one tier scores.
    groups: dict[tuple[str, str], list[BracketRule]] = {}
    for rule in contract.bracket_rules:
        groups.setdefault((rule.stat, rule.group), []).append(rule)
    for (stat, _group), rules in groups.items():
        value = draw.get(stat)
        if value is None:
            continue
        for rule in rules:
            if rule.contains(float(value)):
                total += rule.points
                break

    for rule in contract.dst_rules:
        value = draw.get(rule.stat)
        if value is None:
            continue
        total += float(value) * rule.points_per_unit

    return total
