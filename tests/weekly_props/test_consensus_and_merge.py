"""Tests for weekly props consensus, juice rejection, and component merge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.ingest.props.normalize import build_normalized_quote
from src.projection.weekly_props.consensus import (
    build_player_consensus,
    consensus_for_market,
    evaluate_quote,
    semantic_input_hash,
)
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY, WeeklyPropsPolicy
from src.projection.weekly_props.scoring import merge_player_components


def _q(**kwargs):
    base = dict(
        source="draftkings",
        sportsbook="DraftKings",
        player_name_raw="Alec Pierce",
        market="rec_yards",
        line=48.5,
        fetched_at=datetime.now(timezone.utc),
        over_odds=-110,
        under_odds=-110,
        event_start=datetime.now(timezone.utc) + timedelta(days=2),
        player_id="00-0038120",
        team="IND",
        opponent="MIA",
        period="game",
    )
    base.update(kwargs)
    return build_normalized_quote(**base)


def test_mainline_two_sided_accepted():
    quote = evaluate_quote(_q(), policy=DEFAULT_WEEKLY_POLICY)
    assert quote.reject_reason is None


def test_one_sided_plus_money_ladder_rejected():
    quote = evaluate_quote(
        _q(line=99.5, over_odds=125, under_odds=None),
        policy=DEFAULT_WEEKLY_POLICY,
    )
    assert quote.reject_reason == "one_sided_longshot"


def test_skewed_two_sided_rejected():
    quote = evaluate_quote(
        _q(over_odds=355, under_odds=-567),
        policy=DEFAULT_WEEKLY_POLICY,
    )
    assert quote.reject_reason == "skewed_odds"


def test_stale_and_started_rejected():
    stale = evaluate_quote(
        _q(fetched_at=datetime.now(timezone.utc) - timedelta(hours=48)),
        policy=DEFAULT_WEEKLY_POLICY,
    )
    assert stale.reject_reason == "stale_snapshot"
    started = evaluate_quote(
        _q(event_start=datetime.now(timezone.utc) - timedelta(minutes=5)),
        policy=DEFAULT_WEEKLY_POLICY,
    )
    assert started.reject_reason == "event_started"


def test_pierce_juice_cannot_replace_baseline_component():
    juiced = _q(line=99.5, over_odds=125, under_odds=None, sportsbook="DraftKings")
    clean = _q(
        source="fanduel",
        sportsbook="FanDuel",
        line=48.5,
        over_odds=-110,
        under_odds=-110,
    )
    market = consensus_for_market(
        [juiced, clean],
        market="rec_yards",
        policy=DEFAULT_WEEKLY_POLICY,
        position="WR",
    )
    assert any(q.reject_reason == "one_sided_longshot" for q in market.rejected_quotes)
    assert market.line == 48.5
    assert market.kind == "books"

    player = build_player_consensus(
        player_key="pierce",
        quotes=[juiced, clean],
        player_id="00-0038120",
        player_name="Alec Pierce",
        team="IND",
        position="WR",
    )
    components, sources, scoring_class = merge_player_components(
        player,
        {"rec_yards": 45.0, "rec_tds": 0.4, "receptions": 3.0, "points": 8.0},
    )
    assert components["rec_yards"] == 48.5
    assert sources["rec_yards"] == "weekly_props_consensus"
    assert sources["rec_tds"] == "status_adjusted_baseline"
    assert scoring_class in {"market_partial", "market_complete"}


def test_projection_only_cannot_set_scoring_mean():
    proj = _q(kind="projection", line=90.0, over_odds=None, under_odds=None)
    market = consensus_for_market(
        [proj], market="rec_yards", policy=DEFAULT_WEEKLY_POLICY, position="WR"
    )
    assert market.line is None
    assert market.kind == "none"


def test_partial_coverage_merges_by_component():
    quotes = [
        _q(market="receptions", line=4.5),
        _q(market="rec_yards", line=55.5),
    ]
    player = build_player_consensus(
        player_key="p",
        quotes=quotes,
        player_id="pid",
        player_name="Test",
        position="WR",
        team="NE",
    )
    components, sources, scoring_class = merge_player_components(
        player,
        {"receptions": 3.0, "rec_yards": 40.0, "rec_tds": 0.5, "points": 7.0},
    )
    assert scoring_class == "market_partial"
    assert sources["receptions"] == "weekly_props_consensus"
    assert sources["rec_yards"] == "weekly_props_consensus"
    assert sources["rec_tds"] == "status_adjusted_baseline"
    assert components["rec_tds"] == 0.5


def test_bye_player_does_not_inherit_baseline_points():
    player = build_player_consensus(
        player_key="bye",
        quotes=[],
        player_id="bye1",
        player_name="Bye Guy",
        position="WR",
        team="BUF",
        on_slate=False,
    )
    components, sources, scoring_class = merge_player_components(
        player, {"rec_yards": 60.0, "points": 12.0}
    )
    assert components["rec_yards"] == 0.0
    assert sources["rec_yards"] == "bye_or_inactive"
    assert scoring_class == "baseline_only"


def test_semantic_hash_ignores_raw_timestamps():
    policy = WeeklyPropsPolicy()
    player = build_player_consensus(
        player_key="p",
        quotes=[_q()],
        player_id="pid",
        player_name="Alec Pierce",
        position="WR",
    )
    h1 = semantic_input_hash(
        season=2026,
        week=1,
        policy=policy,
        players=[player],
        baseline_run_id="base",
        status_overlay_id=None,
        slate_version="slate-a",
    )
    # Different fetch time on quote should not matter once consensus line accepted.
    player2 = build_player_consensus(
        player_key="p",
        quotes=[_q(fetched_at=datetime.now(timezone.utc) - timedelta(minutes=3))],
        player_id="pid",
        player_name="Alec Pierce",
        position="WR",
    )
    h2 = semantic_input_hash(
        season=2026,
        week=1,
        policy=policy,
        players=[player2],
        baseline_run_id="base",
        status_overlay_id=None,
        slate_version="slate-a",
    )
    assert h1 == h2
