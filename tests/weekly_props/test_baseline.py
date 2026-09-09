"""Week-level baseline construction for weekly_props."""

from __future__ import annotations

from datetime import UTC

from src.app.projections.loader import PlayerSummary
from src.ingest.props.contracts import PlayerConsensus
from src.projection.weekly_props.baseline import (
    WeeklyBaselineError,
    WeekSlate,
    build_identity_map,
    build_weekly_baselines,
)
from src.projection.weekly_props.gates import validate_baseline_movement
from src.projection.weekly_props.scoring import (
    half_ppr_parity_points,
    merge_player_components,
)


def _summary(**kwargs) -> PlayerSummary:
    base = {
        "player_id": "p1",
        "name": "Test Player",
        "position": "WR",
        "team": "BUF",
        "mean_points": 12.0,
        "quantiles": {"0.1": 8.0, "0.5": 12.0, "0.9": 16.0},
        "availability_probability": 1.0,
    }
    base.update(kwargs)
    return PlayerSummary(**base)


def _slate(*teams: str, opponents: dict[str, str] | None = None) -> WeekSlate:
    return WeekSlate(
        season=2026,
        week=7,
        teams=frozenset(t.upper() for t in teams),
        opponents=opponents or {},
    )


def test_uncovered_player_retains_weekly_baseline_components():
    players = {
        "p1": _summary(player_id="p1", team="KC", mean_points=14.5),
    }
    components = {
        "p1": {
            "rec_yards": 55.0,
            "rec_tds": 0.4,
            "receptions": 4.0,
            "_position": "WR",
        }
    }
    # Pad to satisfy the minimum trustworthy baseline size.
    for i in range(50):
        pid = f"pad-{i}"
        team = "KC" if i % 2 == 0 else "NE"
        players[pid] = _summary(player_id=pid, name=f"Pad {i}", team=team)
        components[pid] = {"rec_yards": 10.0, "receptions": 1.0, "_position": "WR"}

    bundle = build_weekly_baselines(
        season=2026,
        week=1,
        slate=_slate("KC", "NE", opponents={"KC": "NE", "NE": "KC"}),
        players=players,
        components_by_player=components,
    )
    baseline = bundle.baselines["p1"]
    assert baseline.on_slate is True
    assert baseline.mean_json["rec_yards"] == 55.0
    assert baseline.mean_json["rec_tds"] == 0.4
    assert baseline.mean_json["points"] == 14.5

    stub = PlayerConsensus(
        player_id="p1",
        player_name="Test Player",
        team="KC",
        opponent="NE",
        position="WR",
        markets={},
        scoring_class="baseline_only",
        on_slate=True,
    )
    merged, sources, scoring_class = merge_player_components(stub, baseline.mean_json)
    assert scoring_class == "baseline_only"
    assert merged["rec_yards"] == 55.0
    assert merged["rec_tds"] == 0.4
    assert sources["rec_yards"] == "status_adjusted_baseline"
    assert half_ppr_parity_points(merged) > 0


def test_partial_coverage_combines_quotes_with_baseline_components():
    from datetime import datetime, timedelta

    from src.ingest.props.normalize import build_normalized_quote
    from src.projection.weekly_props.consensus import build_player_consensus

    baseline_mean = {
        "rec_yards": 40.0,
        "rec_tds": 0.5,
        "receptions": 3.0,
        "points": 7.0,
    }
    quotes = [
        build_normalized_quote(
            source="draftkings",
            sportsbook="DraftKings",
            player_name_raw="Test",
            market="receptions",
            line=4.5,
            fetched_at=datetime.now(UTC),
            over_odds=-110,
            under_odds=-110,
            event_start=datetime.now(UTC) + timedelta(days=1),
            player_id="pid",
            team="NE",
        ),
        build_normalized_quote(
            source="draftkings",
            sportsbook="DraftKings",
            player_name_raw="Test",
            market="rec_yards",
            line=55.5,
            fetched_at=datetime.now(UTC),
            over_odds=-110,
            under_odds=-110,
            event_start=datetime.now(UTC) + timedelta(days=1),
            player_id="pid",
            team="NE",
        ),
    ]
    player = build_player_consensus(
        player_key="pid",
        quotes=quotes,
        player_id="pid",
        player_name="Test",
        position="WR",
        team="NE",
    )
    components, sources, scoring_class = merge_player_components(player, baseline_mean)
    assert scoring_class == "market_partial"
    assert components["receptions"] == 4.5
    assert components["rec_yards"] == 55.5
    assert components["rec_tds"] == 0.5
    assert sources["rec_tds"] == "status_adjusted_baseline"


def test_availability_probability_zero_preserved():
    players = {
        f"p{i}": _summary(
            player_id=f"p{i}",
            team="KC" if i else "NE",
            availability_probability=0.0 if i == 0 else 1.0,
            name=f"P{i}",
        )
        for i in range(50)
    }
    components = {
        pid: {"rec_yards": 20.0, "receptions": 2.0, "_position": "WR"} for pid in players
    }
    bundle = build_weekly_baselines(
        season=2026,
        week=1,
        slate=_slate("KC", "NE"),
        players=players,
        components_by_player=components,
    )
    assert bundle.baselines["p0"].availability_probability == 0.0
    assert bundle.baselines["p0"].on_slate is True


def test_bye_and_off_slate_not_marked_active():
    players = {
        "bye": _summary(player_id="bye", team="BUF", name="Bye Guy", mean_points=12.0),
        "active": _summary(player_id="active", team="KC", name="Active Guy", mean_points=11.0),
    }
    components = {
        "bye": {"rec_yards": 60.0, "receptions": 4.0, "_position": "WR"},
        "active": {"rec_yards": 50.0, "receptions": 3.0, "_position": "WR"},
    }
    for i in range(50):
        pid = f"pad-{i}"
        players[pid] = _summary(player_id=pid, team="KC", name=f"Pad {i}")
        components[pid] = {"rec_yards": 10.0, "_position": "WR"}

    # BUF is on bye this week; KC is playing.
    bundle = build_weekly_baselines(
        season=2026,
        week=7,
        slate=_slate("KC", "NE", opponents={"KC": "NE"}),
        players=players,
        components_by_player=components,
    )
    bye = bundle.baselines["bye"]
    active = bundle.baselines["active"]
    assert bye.on_slate is False
    assert bye.availability_probability == 0.0
    assert bye.mean_json["points"] == 0.0
    assert bye.mean_json["rec_yards"] == 0.0
    assert active.on_slate is True
    assert active.mean_json["points"] == 11.0


def test_movement_gate_compares_weekly_to_weekly_baseline():
    from src.app.releases.publication import CandidateRow

    baseline_points = {"p1": 10.0, "p2": 8.0}
    rows = (
        CandidateRow(
            player_id="p1",
            team="KC",
            opponent="NE",
            availability_probability=1.0,
            mean_json={"points": 11.0, "position": "WR"},
            quantiles_json={"0.5": 11.0},
        ),
        CandidateRow(
            player_id="p2",
            team="NE",
            opponent="KC",
            availability_probability=1.0,
            mean_json={"points": 7.5, "position": "RB"},
            quantiles_json={"0.5": 7.5},
        ),
    )
    gate = validate_baseline_movement(rows, baseline_points)
    assert gate.passed


def test_missing_components_fail_safely():
    try:
        build_weekly_baselines(
            season=2026,
            week=1,
            slate=_slate("KC", "NE"),
            players={"p1": _summary()},
            components_by_player={},
        )
        assert False, "expected WeeklyBaselineError"
    except WeeklyBaselineError as exc:
        assert "empty_component_projections" in str(exc)


def test_identity_map_unique_names_only():
    from src.projection.weekly_props.bridge import BaselinePlayer

    baselines = {
        "a": BaselinePlayer(
            player_id="a",
            team="BUF",
            opponent="NE",
            position="QB",
            name="Josh Allen",
            availability_probability=1.0,
            mean_json={"points": 20.0},
            quantiles_json={"0.5": 20.0},
        ),
        "b": BaselinePlayer(
            player_id="b",
            team="LAR",
            opponent="SEA",
            position="WR",
            name="Josh Allen",
            availability_probability=1.0,
            mean_json={"points": 5.0},
            quantiles_json={"0.5": 5.0},
        ),
        "c": BaselinePlayer(
            player_id="c",
            team="PHI",
            opponent="DAL",
            position="WR",
            name="A.J. Brown",
            availability_probability=1.0,
            mean_json={"points": 12.0},
            quantiles_json={"0.5": 12.0},
        ),
    }
    identity_map = build_identity_map(baselines)
    assert "josh allen" not in identity_map  # ambiguous bare name omitted
    assert identity_map["josh allen|BUF"]["player_id"] == "a"
    assert identity_map["josh allen|LAR"]["player_id"] == "b"
    assert identity_map["aj brown"]["player_id"] == "c"
