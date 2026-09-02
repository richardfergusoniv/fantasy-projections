"""Focused tests for weekly joint usage-mixture architecture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.app.projections.weekly_draws import write_joint_weekly_draw_partition
from src.app.scoring.compiler import compile_sleeper_scoring, score_stat_draw
from src.projection.weekly.draws.conservation import (
    validate_partition_draws,
    validate_team_draw,
)
from src.projection.weekly.draws.contracts import (
    apply_active_once,
    classify_zero_row,
    mixture_expectation,
)
from src.projection.weekly.draws.decision_backtest import (
    DEFAULT_DECISION_THRESHOLDS,
    evaluate_decision_gates,
    matchup_win_probability,
    optimal_lineup_regret,
    recommendation_stability,
)
from src.projection.weekly.draws.event_models import (
    evaluate_event_predictions,
    fit_event_models,
)
from src.projection.weekly.draws.first_downs import sample_first_downs
from src.projection.weekly.draws.game_engine import (
    PlayerGameInput,
    ScheduledGameInput,
    TeamGameInput,
    generate_game_draws,
    player_means_from_game_draws,
)
from src.projection.weekly.draws.mixture_panel import build_mixture_panel, summarize_event_rates
from src.projection.weekly.draws.partition_schema import (
    JOINT_PARTITION_SCHEMA_VERSION,
    detect_partial_or_corrupt,
    load_joint_partition,
    verify_joint_partition,
)
from src.projection.weekly.draws.readiness import default_no_go_report
from src.projection.weekly.draws.special_teams_game import (
    DstGameContext,
    GameOffenseState,
    assert_k_dst_game_coherence,
    simulate_dst_from_game,
    simulate_kicker_from_game,
)


def test_mixture_expectation_no_double_counting():
    mix = mixture_expectation(
        p_active=0.8,
        p_participates_given_active=0.5,
        expected_stat_given_participates=10.0,
    )
    assert mix.unconditional_expectation == pytest.approx(4.0)
    # Availability applied once when share is unconditional.
    assert apply_active_once(
        unconditional_share=10.0, p_active=0.8, already_conditioned_on_active=False
    ) == pytest.approx(8.0)
    # Not applied again when already conditioned on active.
    assert apply_active_once(
        unconditional_share=10.0, p_active=0.8, already_conditioned_on_active=True
    ) == pytest.approx(10.0)


def test_zero_class_semantics():
    assert classify_zero_row(
        has_scheduled_game=False,
        is_active=None,
        offense_snaps=0,
        targets=0,
        carries=0,
        attempts=0,
    ) == "bye_no_game"
    assert classify_zero_row(
        has_scheduled_game=True,
        is_active=False,
        offense_snaps=0,
        targets=0,
        carries=0,
        attempts=0,
    ) == "true_dnp"
    assert classify_zero_row(
        has_scheduled_game=True,
        is_active=True,
        offense_snaps=12,
        targets=0,
        carries=0,
        attempts=0,
    ) is None  # participated with snaps; not a zero row
    assert classify_zero_row(
        has_scheduled_game=True,
        is_active=True,
        offense_snaps=0,
        targets=0,
        carries=0,
        attempts=0,
    ) == "active_zero_usage"


def _toy_game(seed: int = 7, draw_count: int = 40) -> dict:
    home = TeamGameInput(
        team="KC",
        opponent="BUF",
        home=True,
        mean_pass_attempts=35,
        mean_rush_attempts=26,
        players=[
            PlayerGameInput(
                "qb-kc", "QB", "KC", 0.98, 0.99, 0.99, dropback_share=0.97, carry_share=0.08
            ),
            PlayerGameInput(
                "wr1-kc", "WR", "KC", 0.95, 0.9, 0.85, target_share=0.28, catch_rate=0.65
            ),
            PlayerGameInput(
                "wr2-kc", "WR", "KC", 0.9, 0.85, 0.7, target_share=0.18, catch_rate=0.6
            ),
            PlayerGameInput(
                "rb-kc", "RB", "KC", 0.95, 0.9, 0.85, target_share=0.12, carry_share=0.55
            ),
        ],
    )
    away = TeamGameInput(
        team="BUF",
        opponent="KC",
        home=False,
        mean_pass_attempts=34,
        mean_rush_attempts=25,
        players=[
            PlayerGameInput(
                "qb-buf", "QB", "BUF", 0.98, 0.99, 0.99, dropback_share=0.97, carry_share=0.1
            ),
            PlayerGameInput(
                "wr1-buf", "WR", "BUF", 0.92, 0.88, 0.8, target_share=0.26, catch_rate=0.64
            ),
            PlayerGameInput(
                "rb-buf", "RB", "BUF", 0.94, 0.9, 0.85, target_share=0.1, carry_share=0.5
            ),
        ],
    )
    game = ScheduledGameInput(
        game_id="2024_01_KC_BUF", season=2024, week=1, home=home, away=away
    )
    return generate_game_draws(game, draw_count=draw_count, seed=seed)


def test_joint_draws_correlate_teammates_and_conserve():
    payload = _toy_game(seed=11, draw_count=80)
    kc = next(t for t in payload["teams"] if t["team"] == "KC")
    qb = next(p for p in kc["players"] if p["player_id"] == "qb-kc")
    wr = next(p for p in kc["players"] if p["player_id"] == "wr1-kc")
    qb_yds = np.array([d["pass_yards"] for d in qb["draws"]])
    wr_yds = np.array([d["rec_yards"] for d in wr["draws"]])
    # Shared game environment induces non-trivial positive association vs legacy ~0.
    corr = float(np.corrcoef(qb_yds, wr_yds)[0, 1])
    assert corr > 0.05 or np.std(wr_yds) < 1e-6

    report = validate_partition_draws([payload], tol=2.0)
    # Allow modest integer/poisson residuals but require finite checks mostly clean.
    hard = [v for v in report.violations if v.rule in {"inactive_nonzero", "receptions_gt_targets", "negative"}]
    assert not hard

    # Components must not move in perfect proportional lockstep across all stats.
    ratios = [d["pass_yards"] / d["pass_attempts"] for d in qb["draws"] if d["pass_attempts"] > 0]
    assert max(ratios) - min(ratios) > 0.05


def test_inactive_and_bye_and_locked_semantics():
    home = TeamGameInput(
        team="SEA",
        opponent="BYE",
        players=[
            PlayerGameInput(
                "out-wr",
                "WR",
                "SEA",
                p_active=1e-9,
                p_participates=0.9,
                p_positive_usage=0.9,
                target_share=0.25,
            ),
            PlayerGameInput(
                "locked-rb",
                "RB",
                "SEA",
                1.0,
                1.0,
                1.0,
                carry_share=0.5,
                locked=True,
                locked_stats={"rush_attempts": 18.0, "rush_yards": 75.0, "rush_tds": 1.0},
            ),
        ],
    )
    away = TeamGameInput(
        team="ARI",
        opponent="SEA",
        players=[
            PlayerGameInput("qb-ari", "QB", "ARI", 1.0, 1.0, 1.0, dropback_share=0.97),
        ],
    )
    payload = generate_game_draws(
        ScheduledGameInput("g1", 2024, 1, home=home, away=away),
        draw_count=20,
        seed=3,
    )
    sea = next(t for t in payload["teams"] if t["team"] == "SEA")
    out_wr = next(p for p in sea["players"] if p["player_id"] == "out-wr")
    locked = next(p for p in sea["players"] if p["player_id"] == "locked-rb")
    assert sum(1 for a in out_wr["active_by_draw"] if a) <= 2
    assert all(d["rush_attempts"] == 18.0 for d in locked["draws"])


def test_first_down_bounds_and_ppfd_scoring():
    rng = np.random.default_rng(0)
    fd = sample_first_downs(rng, position="WR", completions=0, carries=2, receptions=5)
    assert fd["rec_first_downs"] <= 5
    assert fd["rush_first_downs"] <= 2
    contract = compile_sleeper_scoring({"rec": 1.0, "rec_yd": 0.1, "rec_fd": 0.5}, ["WR", "FLEX"])
    draw = {"receptions": 5.0, "rec_yards": 60.0, "rec_first_downs": 3.0}
    pts = score_stat_draw(draw, contract, position="WR")
    assert pts == pytest.approx(5.0 + 6.0 + 1.5)


def test_kicker_dst_shared_game_coherence():
    rng = np.random.default_rng(5)
    offense = GameOffenseState(
        team_touchdowns=3,
        scoring_drives=6,
        points_scored=27,
        pass_attempts=34,
        rush_attempts=25,
        turnovers=1,
    )
    opp = GameOffenseState(
        team_touchdowns=2,
        scoring_drives=5,
        points_scored=20,
        pass_attempts=30,
        rush_attempts=28,
        turnovers=2,
    )
    kick = simulate_kicker_from_game(rng, offense)
    dst = simulate_dst_from_game(
        rng, DstGameContext(opponent_points_scored=opp.points_scored, opponent_yards=350)
    )
    assert assert_k_dst_game_coherence(kick, dst, offense, opp) == []
    assert kick["xp_attempts"] == 3
    assert dst["points_allowed"] == 20


def test_seed_reproducibility_and_independence():
    a = _toy_game(seed=99, draw_count=10)
    b = _toy_game(seed=99, draw_count=10)
    c = _toy_game(seed=100, draw_count=10)
    assert a["teams"][0]["players"][0]["draws"] == b["teams"][0]["players"][0]["draws"]
    assert a["teams"][0]["players"][0]["draws"] != c["teams"][0]["players"][0]["draws"]


def test_joint_partition_schema_and_tamper(tmp_path: Path):
    frame = pl.DataFrame(
        {
            "gsis_id": ["qb1", "wr1", "qb2", "wr2"],
            "position": ["QB", "WR", "QB", "WR"],
            "team": ["AAA", "AAA", "BBB", "BBB"],
            "opponent": ["BBB", "BBB", "AAA", "AAA"],
            "game_id": ["g-1", "g-1", "g-1", "g-1"],
            "play_prob": [1.0, 0.9, 1.0, 0.9],
            "attempts": [34.0, 0.0, 32.0, 0.0],
            "carries": [3.0, 0.0, 4.0, 1.0],
            "targets": [0.0, 8.0, 0.0, 7.0],
            "pred_target_share": [0.0, 0.25, 0.0, 0.22],
            "pred_carry_share": [0.1, 0.0, 0.12, 0.05],
        }
    )
    path, digest, manifest = write_joint_weekly_draw_partition(
        frame,
        tmp_path,
        draw_count=8,
        seed_salt="unit",
        season=2026,
        week=1,
    )
    ok, got = verify_joint_partition(path, expected_hash=digest)
    assert ok and got == digest
    assert manifest.schema_version == JOINT_PARTITION_SCHEMA_VERSION
    payload = load_joint_partition(path)
    payload["draw_count"] = 999
    assert "draw_count" in detect_partial_or_corrupt(payload) or "mismatched_draws" in str(
        detect_partial_or_corrupt(payload)
    )
    # Tamper file hash
    path.write_text(json.dumps({**payload, "partition_hash": "deadbeef"}), encoding="utf-8")
    ok2, reason = verify_joint_partition(path)
    assert not ok2


def test_point_vs_draw_gate_separation():
    report = default_no_go_report(point_go_with_caveats=True, point_dispersion_passes=False)
    assert report.point_model_classification.passed is True
    assert report.auto_publish_allowed is False
    assert report.automatic_weekly_publication == "NO-GO"
    assert report.start_sit_use == "NO-GO"
    # Even if joint conservation later passes, point dispersion still blocks auto-publish.
    report.per_draw_conservation.passed = True
    report.per_draw_conservation.evidence_hash = "test"
    report.event_probability_calibration.passed = True
    report.event_probability_calibration.evidence_hash = "test"
    report.joint_draw_proper_scores.passed = True
    report.joint_draw_proper_scores.evidence_hash = "test"
    report.artifact_integrity.passed = True
    report.artifact_integrity.evidence_hash = "test"
    report.decision_lineup_matchup.passed = True
    report.decision_lineup_matchup.evidence_hash = "test"
    report.league_scoring_completeness.passed = True
    report.league_scoring_completeness.evidence_hash = "test"
    report.ppfd_component_readiness.passed = True
    report.ppfd_component_readiness.evidence_hash = "test"
    report.kicker_readiness.passed = True
    report.kicker_readiness.evidence_hash = "test"
    report.dst_readiness.passed = True
    report.dst_readiness.evidence_hash = "test"
    report.recompute_decisions(point_dispersion_passes=False)
    assert report.joint_draw_classification == "GO"
    assert report.auto_publish_allowed is False
    assert report.start_sit_use == "GO"


def test_decision_stability_and_synthetic_matchup():
    draws = {
        "a": np.array([10.0, 12.0, 11.0, 13.0]),
        "b": np.array([8.0, 9.0, 7.0, 10.0]),
        "c": np.array([6.0, 5.0, 6.0, 5.0]),
    }
    regret = optimal_lineup_regret(
        candidate_lineups=[["a", "b"], ["a", "c"], ["b", "c"]],
        draws=draws,
        actual_points={"a": 12.0, "b": 9.0, "c": 20.0},
    )
    assert regret["regret"] >= 0
    wp = matchup_win_probability(["a", "b"], ["c"], draws)
    assert 0.0 <= wp["win_prob"] <= 1.0
    stab = recommendation_stability({50: ["a"], 200: ["a"], 1000: ["b"]})
    gate = evaluate_decision_gates(
        regret=regret["regret"],
        matchup_brier_score=0.3,
        stability=float(stab["stability"]),
        mc_stderr=0.05,
        thresholds=DEFAULT_DECISION_THRESHOLDS,
    )
    assert gate["pass"] is False


def test_mixture_panel_and_event_model_smoke():
    panel = pl.DataFrame(
        {
            "season": [2023] * 80,
            "week": list(range(1, 17)) * 5,
            "gsis_id": [f"p{i}" for i in range(80)],
            "position": (["WR"] * 40) + (["RB"] * 40),
            "team": ["X"] * 80,
            "game_id": [f"g{i%16}" for i in range(80)],
            "offense_snaps": [20 if i % 3 else 0 for i in range(80)],
            "targets": [5 if i % 3 else 0 for i in range(80)],
            "carries": [8 if i % 4 else 0 for i in range(80)],
            "attempts": [0] * 80,
            "roster_status": ["ACT"] * 80,
            "play_prob": [1.0] * 80,
            "is_out": [False] * 80,
            "depth_rank": [1.0] * 80,
            "age": [25.0] * 80,
            "is_rookie": [False] * 80,
            "games_played_prior": [10] * 80,
            "target_share": [0.2 if i % 3 else 0.0 for i in range(80)],
            "carry_share": [0.3 if i % 4 else 0.0 for i in range(80)],
            "snap_share": [0.5] * 80,
            "passing_first_downs": [0.0] * 80,
            "rushing_first_downs": [1.0] * 80,
            "receiving_first_downs": [2.0 if i % 3 else 0.0 for i in range(80)],
        }
    )
    # Force some byes
    panel = panel.with_columns(
        pl.when(pl.col("week") == 1).then(None).otherwise(pl.col("game_id")).alias("game_id")
    )
    mix = build_mixture_panel(panel)
    mix = mix.with_columns(
        pl.col("is_active_label").alias("active_label")
    ) if "active_label" not in mix.columns and "is_active_label" in mix.columns else mix
    rates = summarize_event_rates(mix)
    assert rates["scheduled_rows"] < mix.height
    assert "bye_no_game" in (rates.get("zero_class_counts") or {})
    bundle = fit_event_models(mix, min_positive=5, positions=("WR", "RB"))
    assert bundle.specs
    # Synthetic evaluation metrics path
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    p = np.array([0.2, 0.7, 0.6, 0.3, 0.8, 0.4, 0.7, 0.9])
    metrics = evaluate_event_predictions(y, p, baseline_rate=0.5)
    assert "brier" in metrics and metrics["n"] == 8


def test_means_from_draws_finite():
    payload = _toy_game(seed=1, draw_count=25)
    means = player_means_from_game_draws(payload)
    assert "qb-kc" in means
    assert means["qb-kc"]["pass_attempts"] >= 0
