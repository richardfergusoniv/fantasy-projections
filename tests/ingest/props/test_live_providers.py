"""Unit tests for live DraftKings / FanDuel parsers (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingest.props.providers.draftkings_live import parse_season_board, parse_weekly_board
from src.ingest.props.providers.fanduel_live import parse_season_markets, parse_weekly_markets
from src.ingest.props.service import build_providers, parse_provider_names


def test_parse_provider_names_defaults():
    assert parse_provider_names("") == ["draftkings", "fanduel"]
    assert parse_provider_names("draftkings, fanduel") == ["draftkings", "fanduel"]


def test_build_providers_fixture_mode():
    providers = build_providers(mode="fixture", provider_names="draftkings,fanduel")
    assert [p.name for p in providers] == ["draftkings", "fanduel"]


def test_build_providers_live_mode_constructs_live_adapters():
    providers = build_providers(mode="live", provider_names="draftkings,fanduel")
    assert [p.name for p in providers] == ["draftkings", "fanduel"]
    assert providers[0].__class__.__name__ == "LiveDraftKingsProvider"
    assert providers[1].__class__.__name__ == "LiveFanDuelProvider"


def test_draftkings_weekly_parser():
    payload = {
        "events": [
            {
                "id": "e1",
                "name": "BUF Bills @ MIA Dolphins",
                "startEventDate": "2099-09-14T17:00:00.0000000Z",
                "participants": [
                    {
                        "name": "MIA Dolphins",
                        "venueRole": "Home",
                        "metadata": {"shortName": "MIA"},
                    },
                    {
                        "name": "BUF Bills",
                        "venueRole": "Away",
                        "metadata": {"shortName": "BUF"},
                    },
                ],
            }
        ],
        "markets": [
            {
                "id": "m1",
                "eventId": "e1",
                "name": "Josh Allen Passing Yards O/U",
                "marketType": {"name": "Passing Yards O/U"},
            }
        ],
        "selections": [
            {
                "marketId": "m1",
                "label": "Over",
                "outcomeType": "Over",
                "points": 265.5,
                "displayOdds": {"american": "−110"},
                "participants": [
                    {"name": "Josh Allen", "venueRole": "AwayPlayer", "type": "Player"}
                ],
            },
            {
                "marketId": "m1",
                "label": "Under",
                "outcomeType": "Under",
                "points": 265.5,
                "displayOdds": {"american": "−110"},
                "participants": [
                    {"name": "Josh Allen", "venueRole": "AwayPlayer", "type": "Player"}
                ],
            },
        ],
    }
    quotes = parse_weekly_board(
        payload,
        market="pass_yards",
        fetched_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    )
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.player_name_raw == "Josh Allen"
    assert quote.line == 265.5
    assert quote.over_odds == -110
    assert quote.under_odds == -110
    assert quote.team == "BUF"
    assert quote.opponent == "MIA"


def test_draftkings_season_parser():
    payload = {
        "events": [
            {
                "id": "e1",
                "participants": [
                    {"name": "C.J. Stroud", "type": "Team"},
                    {
                        "name": "HOU Texans",
                        "type": "Team",
                        "metadata": {"shortName": "HOU"},
                    },
                ],
            }
        ],
        "markets": [
            {
                "id": "m1",
                "eventId": "e1",
                "name": "NFL 2026/27 - C.J. Stroud Regular Season Passing Yards",
            }
        ],
        "selections": [
            {
                "marketId": "m1",
                "label": "Over 3549.5",
                "outcomeType": "Over",
                "displayOdds": {"american": "−110"},
            },
            {
                "marketId": "m1",
                "label": "Under 3549.5",
                "outcomeType": "Under",
                "displayOdds": {"american": "−110"},
            },
        ],
    }
    rows = parse_season_board(payload, market="pass_yards")
    assert len(rows) == 1
    assert rows[0]["name"] == "C.J. Stroud"
    assert rows[0]["markets"]["pass_yards"]["line"] == 3549.5


def test_fanduel_weekly_parser():
    payload = {
        "attachments": {
            "events": {
                "35610167": {
                    "eventId": 35610167,
                    "name": "Buffalo Bills @ Houston Texans",
                    "openDate": "2099-09-13T17:00:00.000Z",
                }
            },
            "markets": {
                "1": {
                    "eventId": 35610167,
                    "marketName": "Dalton Kincaid - Total Receptions",
                    "marketType": "PLAYER_X_RECEPTIONS_MEDIUM",
                    "marketStatus": "OPEN",
                    "runners": [
                        {
                            "handicap": 3.5,
                            "runnerName": "Dalton Kincaid Over",
                            "result": {"type": "OVER"},
                            "logo": "https://assets.sportsbook.fanduel.com/images/team/nfl/buffalo_bills_jersey.png",
                            "winRunnerOdds": {
                                "americanDisplayOdds": {"americanOdds": -102}
                            },
                        },
                        {
                            "handicap": 3.5,
                            "runnerName": "Dalton Kincaid Under",
                            "result": {"type": "UNDER"},
                            "logo": "https://assets.sportsbook.fanduel.com/images/team/nfl/buffalo_bills_jersey.png",
                            "winRunnerOdds": {
                                "americanDisplayOdds": {"americanOdds": -130}
                            },
                        },
                    ],
                }
            },
        }
    }
    quotes = parse_weekly_markets(
        payload, fetched_at=datetime(2026, 9, 11, tzinfo=timezone.utc)
    )
    assert len(quotes) == 1
    assert quotes[0].player_name_raw == "Dalton Kincaid"
    assert quotes[0].market == "receptions"
    assert quotes[0].line == 3.5
    assert quotes[0].team == "BUF"


def test_fanduel_season_parser():
    payload = {
        "attachments": {
            "markets": {
                "1": {
                    "marketName": "Aaron Rodgers Regular Season Passing Yards 2026-27",
                    "marketType": "REGULAR_SEASON_PROPS_-_QUARTERBACKS",
                    "runners": [
                        {
                            "runnerName": "Aaron Rodgers Over 3050.5",
                            "winRunnerOdds": {
                                "americanDisplayOdds": {"americanOdds": -114}
                            },
                        },
                        {
                            "runnerName": "Aaron Rodgers Under 3050.5",
                            "winRunnerOdds": {
                                "americanDisplayOdds": {"americanOdds": -114}
                            },
                        },
                    ],
                }
            }
        }
    }
    rows = parse_season_markets(payload)
    assert len(rows) == 1
    assert rows[0]["markets"]["pass_yards"]["line"] == 3050.5
