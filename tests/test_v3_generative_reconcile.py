"""Guards on the v3 generative composer.

Nothing covered reconcile_v3_generative or the conversion draws, which is how
three defects reached a recorded backtest and were read as an architecture
verdict:

1. QBs were selected by position alone on a board carrying one row per
   (player, stat), so each QB emitted one full passing line PER STAT -- eight
   for a QB with eight target stats.
2. Every QB in a room received the whole team's attempts, so backups were
   drawn as starters.
3. Draws were made at a per-game scale and summed straight into
   fantasy_pts_season, so a receiver's "season" was one game.

Together those produced QB Spearman of -0.037 while receiver ranks stayed
competitive -- a uniform scale error preserves rank and destroys MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.inference.reconcile import (
    reconcile_v3_generative,
    team_environment_from_board,
)
from src.projection.transitions import SEASON_GAMES

QB_STATS = [
    "attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "carries", "rushing_yards", "rushing_tds",
]


def _board():
    """One team: two QBs on the real 8-row grain, a WR and an RB."""
    rows = []
    for pid, scale in (("qb_starter", 1.0), ("qb_backup", 0.1)):
        base = {
            "attempts": 35.0, "completions": 23.0, "passing_yards": 250.0,
            "passing_tds": 1.7, "interceptions": 0.7, "carries": 3.0,
            "rushing_yards": 15.0, "rushing_tds": 0.2,
        }
        for stat in QB_STATS:
            rows.append({
                "player_id": pid, "position": "QB", "team": "KC",
                "stat": stat, "pred_pg": base[stat] * scale,
            })
    for stat, value in (("targets", 9.0), ("receptions", 6.0),
                        ("receiving_yards", 80.0), ("receiving_tds", 0.5)):
        rows.append({"player_id": "wr1", "position": "WR", "team": "KC",
                     "stat": stat, "pred_pg": value})
    for stat, value in (("carries", 15.0), ("rushing_yards", 65.0),
                        ("rushing_tds", 0.4)):
        rows.append({"player_id": "rb1", "position": "RB", "team": "KC",
                     "stat": stat, "pred_pg": value})
    return pd.DataFrame(rows)


def _env(pass_attempts=600.0, carries=400.0):
    return pd.DataFrame([{
        "team": "KC",
        "team_pass_attempts_mean": pass_attempts,
        "team_carries_mean": carries,
    }])


def _run(board=None, env=None, seed=0):
    return reconcile_v3_generative(
        board if board is not None else _board(),
        env if env is not None else _env(),
        rng=np.random.default_rng(seed),
        share_manifest={},
    )


def test_each_player_emits_exactly_one_line():
    """The multiplication bug: one line per stat row instead of per player."""
    out = _run()
    counts = out["player_id"].value_counts()
    assert set(counts.index) == {"qb_starter", "qb_backup", "wr1", "rb1"}
    assert counts.max() == 1, f"a player emitted more than one line: {counts.to_dict()}"


def test_qb_room_shares_the_team_attempts_rather_than_each_taking_them_all():
    """Each QB drawing the full team volume doubles the room.

    NOTE this assertion alone cannot catch the shipped composer, because its
    8x line multiplication and its per-game scale very nearly cancel (8/17 per
    QB x 2 QBs landed at 572 against a 600-attempt team). It guards the
    take-all bug in isolation; test_each_player_emits_exactly_one_line and
    test_lines_are_season_totals_not_per_game catch the pair.
    """
    out = _run()
    qb = out[out["position"].eq("QB")]
    assert len(qb) == 2
    total = qb["pass_attempts"].sum()
    assert 450 < total < 780, f"QB room attempts {total} is not one team's worth"
    # No single QB may hold a whole team's volume when the room is shared.
    assert qb["pass_attempts"].max() < 560


def test_backup_does_not_outdraw_the_starter():
    """Volume follows the board's own prediction, not an equal split."""
    starter_wins = 0
    for seed in range(12):
        out = _run(seed=seed)
        qb = out[out["position"].eq("QB")].set_index("player_id")
        if qb.loc["qb_starter", "pass_attempts"] > qb.loc["qb_backup", "pass_attempts"]:
            starter_wins += 1
    assert starter_wins >= 10, f"starter led only {starter_wins}/12 draws"


def test_lines_are_season_totals_not_per_game():
    """The scale bug: per-game draws summed as a season.

    A monotonic 1/17 scaling leaves Spearman untouched and destroys MAE,
    which is exactly the signature the recorded backtest showed.

    Thresholds are set well clear of what the buggy path produced, not just
    clear of one game: the shipped composer gave wr1 41 targets, so a ">40"
    assertion would have passed with the bug present.
    """
    out = _run()
    qb = out[out["player_id"].eq("qb_starter")].iloc[0]
    assert qb["passing_yards"] > 2500, (
        f"{qb['passing_yards']:.0f} passing yards is not a season "
        f"(one game ~250, the buggy eight-game sum ~2000)")
    # wr1 is the only receiver, so a season is ~600 targets; the buggy path
    # produced 41.
    wr = out[out["player_id"].eq("wr1")].iloc[0]
    assert wr["targets"] > 250, f"{wr['targets']:.0f} targets is not a season"
    # rb1 is the only back, so ~400 carries; the buggy path produced ~23.
    rb = out[out["player_id"].eq("rb1")].iloc[0]
    assert rb["carries"] > 150, f"{rb['carries']:.0f} carries is not a season"


def test_conversion_rates_are_player_specific():
    """Two QBs on equal volume must not produce identical efficiency.

    draw_passing_line took league constants for every QB, so the generative
    arm had almost no cross-QB signal regardless of volume.
    """
    board = _board()
    # Give the backup a very different yards-per-completion profile.
    mask = board["player_id"].eq("qb_backup") & board["stat"].eq("passing_yards")
    board.loc[mask, "pred_pg"] = 35.0 * 0.1 * 0.64 * 15.5
    ypc_starter, ypc_backup = [], []
    for seed in range(8):
        out = reconcile_v3_generative(
            board, _env(), rng=np.random.default_rng(seed), share_manifest={})
        qb = out[out["position"].eq("QB")].set_index("player_id")
        for pid, sink in (("qb_starter", ypc_starter), ("qb_backup", ypc_backup)):
            comp = qb.loc[pid, "completions"]
            if comp > 0:
                sink.append(qb.loc[pid, "passing_yards"] / comp)
    assert np.mean(ypc_backup) > np.mean(ypc_starter) + 1.0, (
        f"backup {np.mean(ypc_backup):.2f} vs starter {np.mean(ypc_starter):.2f} "
        "yards/completion - rates are not player-specific")


def test_team_environment_comes_from_the_board_anchors():
    board = _board()
    board["team_pass_attempts_pg_pred"] = 33.0
    board["team_carries_pg_pred"] = 26.0
    env = team_environment_from_board(board)
    assert env.loc[0, "team_pass_attempts_mean"] == pytest.approx(33.0 * SEASON_GAMES)
    assert env.loc[0, "team_carries_mean"] == pytest.approx(26.0 * SEASON_GAMES)


def test_team_environment_falls_back_when_anchors_absent():
    env = team_environment_from_board(_board())
    assert env.loc[0, "team_pass_attempts_mean"] == pytest.approx(600.0)
    assert env.loc[0, "team_carries_mean"] == pytest.approx(400.0)


def test_teams_get_different_volume():
    """The hardcoded 600/400 gave every team the same draw."""
    board = pd.concat([_board(), _board().assign(team="BUF")], ignore_index=True)
    env = pd.DataFrame([
        {"team": "KC", "team_pass_attempts_mean": 700.0, "team_carries_mean": 350.0},
        {"team": "BUF", "team_pass_attempts_mean": 480.0, "team_carries_mean": 520.0},
    ])
    out = reconcile_v3_generative(
        board, env, rng=np.random.default_rng(3), share_manifest={})
    by_team = out.groupby("team")["pass_attempts"].sum()
    assert by_team["KC"] > by_team["BUF"] + 100


def test_receiving_pool_is_allocated_once_across_positions():
    """WR/TE/RB share one pool of team targets.

    allocate_opportunities keys rooms by position by default, which is right
    for QB attempts and RB carries but wrong for targets: passing all three
    receiving positions in one call gave each group a full team's worth and
    allocated the team 3x over. The per-game scale used to hide this
    (3/17 reads as under-projection); fixing the scale exposed it.
    """
    from src.projection.models.opportunity_shares import allocate_opportunities

    room = pd.DataFrame([
        {"player_id": "wr1", "position": "WR", "team": "KC", "stat": "targets", "pred_pg": 9.0},
        {"player_id": "te1", "position": "TE", "team": "KC", "stat": "targets", "pred_pg": 5.0},
        {"player_id": "rb1", "position": "RB", "team": "KC", "stat": "targets", "pred_pg": 4.0},
    ])
    out = allocate_opportunities(
        room, 600.0, rng=np.random.default_rng(0), manifest={},
        group_cols=["team", "stat"])
    assert out["allocated_volume"].sum() == pytest.approx(600.0)

    # Default grouping still splits by position, for single-position rooms.
    per_position = allocate_opportunities(
        room, 600.0, rng=np.random.default_rng(0), manifest={})
    assert per_position["allocated_volume"].sum() == pytest.approx(1800.0)


def test_team_receiving_volume_is_not_multiplied_by_position_count():
    """End to end: a team's emitted targets are one team's worth."""
    board = _board()
    # Add a TE and a second WR so the room spans three positions.
    extra = []
    for pid, pos in (("te1", "TE"), ("wr2", "WR")):
        for stat, value in (("targets", 5.0), ("receptions", 3.5),
                            ("receiving_yards", 40.0), ("receiving_tds", 0.3)):
            extra.append({"player_id": pid, "position": pos, "team": "KC",
                          "stat": stat, "pred_pg": value})
    board = pd.concat([board, pd.DataFrame(extra)], ignore_index=True)

    out = reconcile_v3_generative(
        board, _env(pass_attempts=600.0), rng=np.random.default_rng(1),
        share_manifest={})
    total_targets = out["targets"].sum()
    # ~600 attempts x 0.952 targets/attempt, with Poisson noise. The failure
    # this catches is ~1700 from a three-times allocation.
    assert 450 < total_targets < 750, (
        f"team emitted {total_targets:.0f} targets; one team's worth is ~571")


def test_rooms_claim_only_the_share_they_own():
    """A position room does not get the whole team anchor.

    Measured contracts put QB at 0.941 of team pass attempts and RB at 0.810
    of team carries; the rest is scrambles, sweeps and the like. Allocating
    100% put RB 23% over and left the simulated p50 disagreeing with the board
    by +8.6 at RB and -16.9 at QB.
    """
    from src.projection.contracts import TEAM_VOLUME_SHARES

    out = _run(env=_env(pass_attempts=600.0, carries=400.0))
    qb_share = TEAM_VOLUME_SHARES[("QB", "attempts")][1]
    rb_share = TEAM_VOLUME_SHARES[("RB", "carries")][1]

    qb_total = out[out["position"].eq("QB")]["pass_attempts"].sum()
    expected_qb = 600.0 * qb_share
    assert abs(qb_total - expected_qb) < 0.15 * expected_qb, (
        f"QB room drew {qb_total:.0f} against an owned {expected_qb:.0f}")

    rb_total = out[out["position"].eq("RB")]["carries"].sum()
    expected_rb = 400.0 * rb_share
    assert abs(rb_total - expected_rb) < 0.20 * expected_rb, (
        f"RB room drew {rb_total:.0f} against an owned {expected_rb:.0f}")


def test_quarterbacks_get_a_rushing_line():
    """QB rushing was dropped entirely, costing 18.4 points per QB.

    The rush room filters on RB, so QBs never received a rushing draw. It is
    not part of the RB carry pool -- RB owns 0.810 of team carries and
    scrambles sit in the remainder -- so it is drawn from the QB's own
    projected carries.
    """
    board = _board()
    board["pred_season"] = board["pred_pg"] * 17.0
    out = reconcile_v3_generative(
        board, _env(), rng=np.random.default_rng(5), share_manifest={})
    qb = out[out["player_id"].eq("qb_starter")].iloc[0]
    assert qb["carries"] > 0, "QB emitted no rushing volume"
    assert qb["rushing_yards"] > 0, "QB emitted no rushing yards"
    # And it must not come out of the RB pool.
    rb = out[out["player_id"].eq("rb1")].iloc[0]
    assert rb["carries"] > 100, "RB carries were consumed by the QB draw"


def test_qb_rushing_scales_with_the_board_projection():
    """A running QB must out-rush a pocket passer, not draw a league mean."""
    board = _board()
    board["pred_season"] = board["pred_pg"] * 17.0
    runner = board.copy()
    mask = runner["player_id"].eq("qb_starter") & runner["stat"].eq("carries")
    runner.loc[mask, "pred_season"] = 130.0
    pocket = board.copy()
    mask2 = pocket["player_id"].eq("qb_starter") & pocket["stat"].eq("carries")
    pocket.loc[mask2, "pred_season"] = 20.0

    def qb_carries(frame, seed):
        out = reconcile_v3_generative(
            frame, _env(), rng=np.random.default_rng(seed), share_manifest={})
        return out[out["player_id"].eq("qb_starter")].iloc[0]["carries"]

    hi = np.mean([qb_carries(runner, s) for s in range(6)])
    lo = np.mean([qb_carries(pocket, s) for s in range(6)])
    assert hi > lo + 50, f"running QB {hi:.0f} vs pocket {lo:.0f} carries"
