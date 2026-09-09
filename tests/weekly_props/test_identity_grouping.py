"""Identity resolution before weekly_props quote grouping."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.ingest.props.contracts import ProviderSnapshot
from src.ingest.props.normalize import build_normalized_quote
from src.projection.weekly_props.identity import resolve_quote_group_key
from src.projection.weekly_props.publisher import WeeklyPropsProjectionService


def _quote(**kwargs):
    base = {
        "source": "draftkings",
        "sportsbook": "DraftKings",
        "player_name_raw": "A.J. Brown",
        "market": "rec_yards",
        "line": 70.5,
        "fetched_at": datetime.now(UTC),
        "over_odds": -110,
        "under_odds": -110,
        "event_start": datetime.now(UTC) + timedelta(days=2),
        "team": "PHI",
        "opponent": "DAL",
    }
    base.update(kwargs)
    return build_normalized_quote(**base)


def test_id_and_name_only_quotes_merge_without_losing_markets(tmp_path):
    identity_map = {
        "00-0035676": {
            "player_id": "00-0035676",
            "name": "A.J. Brown",
            "team": "PHI",
            "position": "WR",
        },
        "aj brown": {
            "player_id": "00-0035676",
            "name": "A.J. Brown",
            "team": "PHI",
            "position": "WR",
        },
        "aj brown|PHI": {
            "player_id": "00-0035676",
            "name": "A.J. Brown",
            "team": "PHI",
            "position": "WR",
        },
    }
    id_quote = _quote(
        source="draftkings",
        sportsbook="DraftKings",
        player_id="00-0035676",
        market="rec_yards",
        line=70.5,
    )
    name_quote = _quote(
        source="fanduel",
        sportsbook="FanDuel",
        player_id=None,
        player_name_raw="AJ Brown",
        market="receptions",
        line=5.5,
    )
    key_id, _ = resolve_quote_group_key(id_quote, identity_map)
    key_name, _ = resolve_quote_group_key(name_quote, identity_map)
    assert key_id == key_name == "00-0035676"

    service = WeeklyPropsProjectionService(session=None, artifact_root=tmp_path)  # type: ignore[arg-type]
    snap_a = ProviderSnapshot(
        source="draftkings",
        season=2026,
        week=1,
        fetched_at=datetime.now(UTC),
        urls=("fixture://draftkings",),
        success=True,
        quotes=(id_quote,),
    )
    snap_b = ProviderSnapshot(
        source="fanduel",
        season=2026,
        week=1,
        fetched_at=datetime.now(UTC),
        urls=("fixture://fanduel",),
        success=True,
        quotes=(name_quote,),
    )
    manifest = service.build_manifest_from_snapshots(
        season=2026,
        week=1,
        snapshots=[snap_a, snap_b],
        identity_map=identity_map,
        slate_teams={"PHI", "DAL"},
    )
    assert len(manifest.players) == 1
    player = manifest.players[0]
    assert player.player_id == "00-0035676"
    assert "rec_yards" in player.markets
    assert "receptions" in player.markets


def test_distinct_same_name_players_are_not_merged(tmp_path):
    identity_map = {
        "buf-josh": {
            "player_id": "buf-josh",
            "name": "Josh Allen",
            "team": "BUF",
            "position": "QB",
        },
        "lar-josh": {
            "player_id": "lar-josh",
            "name": "Josh Allen",
            "team": "LAR",
            "position": "WR",
        },
        "josh allen|BUF": {
            "player_id": "buf-josh",
            "name": "Josh Allen",
            "team": "BUF",
            "position": "QB",
        },
        "josh allen|LAR": {
            "player_id": "lar-josh",
            "name": "Josh Allen",
            "team": "LAR",
            "position": "WR",
        },
    }
    qb = _quote(
        player_id=None,
        player_name_raw="Josh Allen",
        team="BUF",
        market="pass_yards",
        line=265.5,
    )
    wr = _quote(
        player_id=None,
        player_name_raw="Josh Allen",
        team="LAR",
        market="rec_yards",
        line=45.5,
    )
    key_qb, _ = resolve_quote_group_key(qb, identity_map)
    key_wr, _ = resolve_quote_group_key(wr, identity_map)
    assert key_qb != key_wr
    assert key_qb == "buf-josh"
    assert key_wr == "lar-josh"

    service = WeeklyPropsProjectionService(session=None, artifact_root=tmp_path)  # type: ignore[arg-type]
    snap = ProviderSnapshot(
        source="draftkings",
        season=2026,
        week=1,
        fetched_at=datetime.now(UTC),
        urls=("fixture://draftkings",),
        success=True,
        quotes=(qb, wr),
    )
    manifest = service.build_manifest_from_snapshots(
        season=2026,
        week=1,
        snapshots=[snap],
        identity_map=identity_map,
        slate_teams={"BUF", "LAR", "NE", "SEA"},
    )
    assert len(manifest.players) == 2
    by_id = {p.player_id: p for p in manifest.players}
    assert "pass_yards" in by_id["buf-josh"].markets
    assert "rec_yards" in by_id["lar-josh"].markets
