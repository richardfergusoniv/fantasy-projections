"""Unit tests for live BettingPros weekly parser (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.ingest.props.providers.bettingpros_live import (
    WEEKLY_MARKET_IDS,
    WEEKLY_MARKET_ID_MAP,
    is_eligible_role1_sportsbook,
    parse_offer,
    parse_offers,
)
from src.ingest.props.providers.http import LiveFetchError
from src.ingest.props.service import build_providers, run_ingest
from src.projection.weekly_props.consensus import consensus_for_market
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY
from src.ingest.props.normalize import build_normalized_quote


BOOKS_BY_ID = {
    0: "BettingPros Consensus",
    10: "FanDuel",
    12: "DraftKings",
    13: "Caesars",
    19: "BetMGM",
    68: "Kalshi",
    73: "Polymarket",
    37: "PrizePicks",
}


def _book_entry(book_id: int, line: float, cost: float = -110) -> dict:
    return {
        "id": book_id,
        "lines": [
            {
                "main": True,
                "active": True,
                "is_off": False,
                "line": line,
                "cost": cost,
            }
        ],
    }


def _ou_offer(
    *,
    market_id: int,
    player: str,
    team: str,
    event_id: int,
    books: dict[int, tuple[float, float, float]] | None = None,
    # books: book_id -> (line, over_odds, under_odds)
    line: float | None = None,
    over: float = -110,
    under: float = -110,
    under_line: float | None = None,
    active: bool = True,
    consensus_only: bool = False,
) -> dict:
    if books is None:
        if consensus_only or line is not None:
            use_line = 50.5 if line is None else line
            under_l = use_line if under_line is None else under_line
            book_map = {0: (use_line, over, under)} if under_line is None else None
            if under_line is not None:
                # Special mismatched case uses consensus book with different sides.
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
                            "books": [_book_entry(0, use_line, over)],
                        },
                        {
                            "selection": "under",
                            "active": True,
                            "books": [_book_entry(0, under_l, under)],
                        },
                    ],
                }
            books = {0: (use_line, over, under)}
        else:
            books = {
                0: (56.5, -115, -112),
                10: (56.5, -114, -114),
                12: (57.5, -110, -110),
                13: (55.5, -115, -105),
                19: (55.5, -110, -110),
                68: (60.5, 120, -150),
                73: (58.5, -105, -105),
                37: (55.5, -110, -110),
            }

    over_books = []
    under_books = []
    for bid, (bk_line, ov, un) in books.items():
        over_books.append(_book_entry(bid, bk_line, ov))
        under_books.append(_book_entry(bid, bk_line, un))

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
            {"selection": "over", "active": True, "books": over_books},
            {"selection": "under", "active": True, "books": under_books},
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


def test_eligible_sportsbook_filters():
    assert is_eligible_role1_sportsbook("Caesars") is True
    assert is_eligible_role1_sportsbook("BetMGM") is True
    assert is_eligible_role1_sportsbook("DraftKings") is False
    assert is_eligible_role1_sportsbook("FanDuel") is False
    assert is_eligible_role1_sportsbook("BettingPros Consensus") is False
    assert is_eligible_role1_sportsbook("Kalshi") is False
    assert is_eligible_role1_sportsbook("Polymarket") is False
    assert is_eligible_role1_sportsbook("PrizePicks") is False


def test_parse_offer_emits_per_book_not_consensus():
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
    )
    quotes = parse_offer(
        offer,
        event=event,
        fetched_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        books_by_id=BOOKS_BY_ID,
        source_url="https://api.bettingpros.com/v3/offers",
    )
    names = sorted(q.sportsbook.lower() for q in quotes)
    assert names == ["betmgm", "caesars"]
    assert all(q.source == "bettingpros" for q in quotes)
    assert all(q.market == "rec_yards" for q in quotes)
    assert all(q.team == "ATL" and q.opponent == "CAR" for q in quotes)
    # Consensus / DK / FD / PMs / DFS must not appear.
    assert "bettingpros" not in names
    assert "draftkings" not in names
    assert "fanduel" not in names
    assert "kalshi" not in names
    assert "polymarket" not in names
    assert "prizepicks" not in names


def test_parse_offer_consensus_only_yields_empty():
    offer = _ou_offer(
        market_id=105,
        player="Drake London",
        team="ATL",
        event_id=21932,
        consensus_only=True,
        line=56.5,
    )
    quotes = parse_offer(
        offer,
        event={"id": 21932, "home": "ATL", "visitor": "CAR"},
        fetched_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        books_by_id=BOOKS_BY_ID,
    )
    assert quotes == []


def test_parse_offer_skips_mismatched_over_under_lines():
    offer = _ou_offer(
        market_id=104,
        player="Bijan Robinson",
        team="ATL",
        event_id=21932,
        books={13: (5.5, -110, -110)},
    )
    # Force mismatched sides on Caesars.
    offer["selections"][1]["books"][0]["lines"][0]["line"] = 4.5
    assert (
        parse_offer(
            offer,
            event={"id": 21932, "home": "ATL", "visitor": "CAR"},
            fetched_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
            books_by_id=BOOKS_BY_ID,
        )
        == []
    )


def test_parse_offers_empty_and_partial_boards():
    fetched = datetime(2026, 9, 18, tzinfo=timezone.utc)
    assert parse_offers([], events_by_id={}, fetched_at=fetched) == []

    offers = [
        _ou_offer(
            market_id=999,
            player="Skip Me",
            team="ATL",
            event_id=1,
            books={13: (1.5, -110, -110)},
        ),
        _ou_offer(
            market_id=103,
            player="Inactive QB",
            team="ATL",
            event_id=1,
            books={13: (200.5, -110, -110)},
            active=False,
        ),
        _ou_offer(
            market_id=103,
            player="Cooper Rush",
            team="ATL",
            event_id=1,
            books={13: (183.5, -110, -110), 0: (180.5, -110, -110)},
        ),
    ]
    quotes = parse_offers(
        offers,
        events_by_id={"1": {"id": 1, "home": "ATL", "visitor": "CAR"}},
        fetched_at=fetched,
        books_by_id=BOOKS_BY_ID,
    )
    assert len(quotes) == 1
    assert quotes[0].player_name_raw == "Cooper Rush"
    assert quotes[0].market == "pass_yards"
    assert quotes[0].sportsbook == "Caesars"


def test_bp_per_book_does_not_double_count_dk_fd_in_role1():
    """Regression: consensus sportsbook=bettingpros must not inflate book_count.

    Emitting BP consensus as an independent book next to live DK/FD would make
    book_count=3 and bias robust_median. Per-book Caesars is a real third book.
    """
    fetched = datetime(2026, 9, 18, tzinfo=timezone.utc)
    event_start = datetime(2099, 9, 20, tzinfo=timezone.utc)

    def _q(source: str, sportsbook: str, line: float):
        return build_normalized_quote(
            source=source,
            sportsbook=sportsbook,
            player_name_raw="Drake London",
            market="rec_yards",
            line=line,
            fetched_at=fetched,
            over_odds=-110,
            under_odds=-110,
            event_start=event_start,
            player_id="00-0037238",
            team="ATL",
            period="game",
        )

    # Bad shape (what the old consensus emission did): DK + FD + blended BP.
    bad = consensus_for_market(
        [
            _q("draftkings", "DraftKings", 50.5),
            _q("fanduel", "FanDuel", 60.5),
            _q("bettingpros", "bettingpros", 55.5),  # blended aggregate
        ],
        market="rec_yards",
        policy=DEFAULT_WEEKLY_POLICY,
        position="WR",
    )
    assert bad.coverage.book_count == 3  # documents the defect if reintroduced

    # Correct shape: live DK + FD + BP-sourced Caesars only.
    offer = _ou_offer(
        market_id=105,
        player="Drake London",
        team="ATL",
        event_id=1,
        books={
            0: (55.5, -110, -110),
            10: (60.5, -110, -110),
            12: (50.5, -110, -110),
            13: (55.5, -110, -110),
            68: (70.5, 150, -180),
        },
    )
    bp_quotes = parse_offer(
        offer,
        event={"id": 1, "home": "ATL", "visitor": "CAR"},
        fetched_at=fetched,
        books_by_id=BOOKS_BY_ID,
    )
    assert [q.sportsbook.lower() for q in bp_quotes] == ["caesars"]

    good = consensus_for_market(
        [
            _q("draftkings", "DraftKings", 50.5),
            _q("fanduel", "FanDuel", 60.5),
            *bp_quotes,
        ],
        market="rec_yards",
        policy=DEFAULT_WEEKLY_POLICY,
        position="WR",
    )
    assert good.coverage.book_count == 3
    assert good.coverage.accepted_books == ("caesars", "draftkings", "fanduel")
    assert good.coverage.line_basis == "robust_median"
    assert good.line == 55.5
    # Median must not double-weight DK/FD via a blended bettingpros row.
    assert "bettingpros" not in good.coverage.accepted_books


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
    assert len(result.written) == 3


def test_fetch_weekly_snapshot_empty_events(monkeypatch):
    from src.ingest.props.providers import bettingpros_live as bp

    monkeypatch.setenv(bp.BP_API_KEY_ENV, "test-bettingpros-client-key")
    monkeypatch.setattr(bp, "fetch_books_catalog", lambda: {13: "Caesars"})
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
