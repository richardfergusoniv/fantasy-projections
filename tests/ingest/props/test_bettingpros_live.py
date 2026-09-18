"""Unit tests for live BettingPros weekly parser (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.ingest.props.providers.bettingpros_live import (
    WEEKLY_MARKET_IDS,
    WEEKLY_MARKET_ID_MAP,
    parse_offer,
    parse_offers,
)
from src.ingest.props.providers.http import LiveFetchError
from src.ingest.props.service import build_providers, run_ingest


def _ou_offer(
    *,
    market_id: int,
    player: str,
    team: str,
    event_id: int,
    line: float,
    over: float = -110,
    under: float = -110,
    under_line: float | None = None,
    active: bool = True,
) -> dict:
    under_l = line if under_line is None else under_line
    return {
        "id": f"offer-{market_id}-{player}",
        "market_id": market_id,
        "event_id": event_id,
        "player_id": 1000 + market_id,
        "active": active,
        "participants": [
            {
                "id": "1000",
                "name": player,
                "player": {"team": team, "position": "WR"},
            }
        ],
        "selections": [
            {
                "selection": "over",
                "active": True,
                "books": [
                    {
                        "id": 0,
                        "lines": [
                            {
                                "main": True,
                                "active": True,
                                "is_off": False,
                                "line": line,
                                "cost": over,
                            }
                        ],
                    }
                ],
            },
            {
                "selection": "under",
                "active": True,
                "books": [
                    {
                        "id": 0,
                        "lines": [
                            {
                                "main": True,
                                "active": True,
                                "is_off": False,
                                "line": under_l,
                                "cost": under,
                            }
                        ],
                    }
                ],
            },
        ],
    }


def test_weekly_market_map_covers_scoring_volume_not_combined_tds():
    keys = {market for _, market in WEEKLY_MARKET_IDS}
    assert "pass_yards" in keys
    assert "rec_yards" in keys
    assert "receptions" in keys
    assert "pass_tds" in keys
    assert "rush_tds" not in keys
    assert "rec_tds" not in keys
    assert 334 not in WEEKLY_MARKET_ID_MAP


def test_parse_offer_maps_market_and_opponent():
    event = {
        "id": 21932,
        "home": "ATL",
        "visitor": "CAR",
        "scheduled": "2026-09-20 17:00:00",
        "status": "scheduled",
    }
    offer = _ou_offer(
        market_id=105,
        player="Drake London",
        team="ATL",
        event_id=21932,
        line=56.5,
        over=-115,
        under=-112,
    )
    quote = parse_offer(
        offer,
        event=event,
        fetched_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        source_url="https://api.bettingpros.com/v3/offers",
    )
    assert quote is not None
    assert quote.market == "rec_yards"
    assert quote.player_name_raw == "Drake London"
    assert quote.sportsbook == "bettingpros"
    assert quote.source == "bettingpros"
    assert quote.line == 56.5
    assert quote.over_odds == -115
    assert quote.under_odds == -112
    assert quote.team == "ATL"
    assert quote.opponent == "CAR"
    assert quote.period == "game"


def test_parse_offer_skips_mismatched_over_under_lines():
    offer = _ou_offer(
        market_id=104,
        player="Bijan Robinson",
        team="ATL",
        event_id=21932,
        line=5.5,
        under_line=4.5,
    )
    assert (
        parse_offer(
            offer,
            event={"id": 21932, "home": "ATL", "visitor": "CAR"},
            fetched_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        )
        is None
    )


def test_parse_offers_empty_and_partial_boards():
    fetched = datetime(2026, 9, 18, tzinfo=timezone.utc)
    assert parse_offers([], events_by_id={}, fetched_at=fetched) == []

    # Unknown market id + inactive offer + good offer → one quote.
    offers = [
        _ou_offer(
            market_id=999,
            player="Skip Me",
            team="ATL",
            event_id=1,
            line=1.5,
        ),
        _ou_offer(
            market_id=103,
            player="Inactive QB",
            team="ATL",
            event_id=1,
            line=200.5,
            active=False,
        ),
        _ou_offer(
            market_id=103,
            player="Cooper Rush",
            team="ATL",
            event_id=1,
            line=183.5,
        ),
    ]
    quotes = parse_offers(
        offers,
        events_by_id={"1": {"id": 1, "home": "ATL", "visitor": "CAR"}},
        fetched_at=fetched,
    )
    assert len(quotes) == 1
    assert quotes[0].player_name_raw == "Cooper Rush"
    assert quotes[0].market == "pass_yards"


def test_build_providers_live_includes_bettingpros():
    providers = build_providers(
        mode="live", provider_names="draftkings,fanduel,bettingpros"
    )
    assert [p.name for p in providers] == ["draftkings", "fanduel", "bettingpros"]
    assert providers[2].__class__.__name__ == "LiveBettingProsProvider"


def test_bettingpros_failure_does_not_break_dk_fd_persist(tmp_path: Path, monkeypatch):
    """Provider isolation: BP LiveFetchError must not abort DK/FD snapshots."""

    def _boom(*_args, **_kwargs):
        raise LiveFetchError("HTTP 403 for https://api.bettingpros.com/v3/events")

    monkeypatch.setattr(
        "src.ingest.props.providers.bettingpros_live.fetch_weekly_snapshot",
        _boom,
    )

    class _OkProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        def fetch(self, *, season: int, week: int, now=None):
            from src.ingest.props.contracts import ProviderSnapshot

            return ProviderSnapshot(
                source=self.name,
                season=season,
                week=week,
                fetched_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
                urls=(),
                quotes=(),
                success=True,
                metadata={"live": True, "fixture_ok": True},
            )

    from src.ingest.props.providers.bettingpros import LiveBettingProsProvider
    from src.ingest.props.snapshot import SnapshotStore

    providers = [
        _OkProvider("draftkings"),
        _OkProvider("fanduel"),
        LiveBettingProsProvider(),
    ]
    result = run_ingest(
        season=2026,
        week=2,
        providers=providers,
        store=SnapshotStore(tmp_path),
        mode="live",
    )
    assert result.success_count == 2
    assert result.failure_count == 1
    by_source = {s.source: s for s in result.snapshots}
    assert by_source["draftkings"].success is True
    assert by_source["fanduel"].success is True
    assert by_source["bettingpros"].success is False
    assert "403" in (by_source["bettingpros"].error or "")
    # Persist paths written for every provider, including the failed stub.
    assert len(result.written) == 3


def test_fetch_weekly_snapshot_empty_events(monkeypatch):
    from src.ingest.props.providers import bettingpros_live as bp

    monkeypatch.setenv(bp.BP_API_KEY_ENV, "test-bettingpros-client-key")
    monkeypatch.setattr(bp, "fetch_events", lambda **_kwargs: [])
    snap = bp.fetch_weekly_snapshot(season=2026, week=3)
    assert snap.success is False
    assert snap.error == "no_quotes"
    assert snap.quotes == ()


def test_missing_api_key_skips_provider_without_http(monkeypatch):
    from src.ingest.props.providers import bettingpros_live as bp

    monkeypatch.delenv(bp.BP_API_KEY_ENV, raising=False)

    def _must_not_fetch(*_args, **_kwargs):
        raise AssertionError("HTTP must not run when API key is missing")

    monkeypatch.setattr(bp, "fetch_json", _must_not_fetch)
    snap = bp.fetch_weekly_snapshot(season=2026, week=2)
    assert snap.success is False
    assert bp.BP_API_KEY_ENV in (snap.error or "")
    assert snap.metadata.get("missing_api_key") is True
    assert snap.quotes == ()


def test_resolve_api_key_rejects_blank(monkeypatch):
    from src.ingest.props.providers import bettingpros_live as bp
    import pytest

    monkeypatch.setenv(bp.BP_API_KEY_ENV, "   ")
    with pytest.raises(LiveFetchError, match=bp.BP_API_KEY_ENV):
        bp.resolve_bettingpros_api_key()
