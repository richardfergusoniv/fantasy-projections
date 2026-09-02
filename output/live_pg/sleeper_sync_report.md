# Live Sleeper shadow sync report

- **Started:** 2026-09-01T05:23:17.332035+00:00
- **Finished:** 2026-09-01T05:24:46.977373+00:00
- **Status:** passed
- **Go / no-go:** go

## Environment
- Database: `postgresql+psycopg://fantasy:fantasy@localhost:5432/fantasy_app`
- Artifact root: `output/live_pg/artifacts`
- Sleeper mode: `live`

## Safety
- GET-only: `GET`
- Connectivity: `ok`
- Auto publish allowed: `False`

## League selection
{
  "configured_count": 6,
  "discovered_count": 6,
  "imported_count": 6,
  "extra_league_ids": [],
  "extra_leagues_ignored": 0,
  "leagues": [
    {
      "league_id": "1389344450517430272",
      "display_name": "The NY Sack Exchange I",
      "league_type": "redraft",
      "rookie_pick_rule": ""
    },
    {
      "league_id": "1355920300633513984",
      "display_name": "Tits Out for The Ladz XII (TWELVE\ud83d\ude24)",
      "league_type": "redraft",
      "rookie_pick_rule": ""
    },
    {
      "league_id": "1317270144682070016",
      "display_name": "C2C. The real superconference",
      "league_type": "dynasty",
      "rookie_pick_rule": "reverse_standings"
    },
    {
      "league_id": "1306489414548979712",
      "display_name": "Dynastical Cucks",
      "league_type": "dynasty",
      "rookie_pick_rule": "max_pf"
    },
    {
      "league_id": "1312127020972404736",
      "display_name": "Hoe Ass Dynasty League",
      "league_type": "dynasty",
      "rookie_pick_rule": "max_pf"
    },
    {
      "league_id": "1311470531635052544",
      "display_name": "Tainticklers",
      "league_type": "dynasty",
      "rookie_pick_rule": "reverse_standings"
    }
  ]
}

## Draft rules
{
  "1306489414548979712": "max_pf",
  "1311470531635052544": "reverse_standings",
  "1312127020972404736": "max_pf",
  "1317270144682070016": "reverse_standings"
}

## Identity reconciliation
{
  "by_league": [
    {
      "league_id": "1389344450517430272",
      "display_name": "The NY Sack Exchange I",
      "total_distinct_rostered_ids": 0,
      "resolved_canonical": 0,
      "unresolved_ids": 0,
      "ambiguous_ids": 0,
      "resolved_starters": 0,
      "unresolved_starters": 0,
      "missing_weekly_projections": 0,
      "missing_season_projections": 0,
      "outside_projection_universe": {},
      "unresolved_starter_ids": [],
      "unresolved_skill_ids": []
    },
    {
      "league_id": "1355920300633513984",
      "display_name": "Tits Out for The Ladz XII (TWELVE\ud83d\ude24)",
      "total_distinct_rostered_ids": 0,
      "resolved_canonical": 0,
      "unresolved_ids": 0,
      "ambiguous_ids": 0,
      "resolved_starters": 0,
      "unresolved_starters": 0,
      "missing_weekly_projections": 0,
      "missing_season_projections": 0,
      "outside_projection_universe": {},
      "unresolved_starter_ids": [],
      "unresolved_skill_ids": []
    },
    {
      "league_id": "1317270144682070016",
      "display_name": "C2C. The real superconference",
      "total_distinct_rostered_ids": 213,
      "resolved_canonical": 213,
      "unresolved_ids": 0,
      "ambiguous_ids": 0,
      "resolved_starters": 88,
      "unresolved_starters": 0,
      "missing_weekly_projections": 173,
      "missing_season_projections": 173,
      "outside_projection_universe": {},
      "unresolved_starter_ids": [],
      "unresolved_skill_ids": []
    },
    {
      "league_id": "1306489414548979712",
      "display_name": "Dynastical Cucks",
      "total_distinct_rostered_ids": 342,
      "resolved_canonical": 342,
      "unresolved_ids": 0,
      "ambiguous_ids": 0,
      "resolved_starters": 120,
      "unresolved_starters": 0,
      "missing_weekly_projections": 285,
      "missing_season_projections": 289,
      "outside_projection_universe": {},
      "unresolved_starter_ids": [],
      "unresolved_skill_ids": []
    },
    {
      "league_id": "1312127020972404736",
      "display_name": "Hoe Ass Dynasty League",
      "total_distinct_rostered_ids": 308,
      "resolved_canonical": 308,
      "unresolved_ids": 0,
      "ambiguous_ids": 0,
      "resolved_starters": 120,
      "unresolved_starters": 0,
      "missing_weekly_projections": 262,
      "missing_season_projections": 264,
      "outside_projection_universe": {},
      "unresolved_starter_ids": [],
      "unresolved_skill_ids": []
    },
    {
      "league_id": "1311470531635052544",
      "display_name": "Tainticklers",
      "total_distinct_rostered_ids": 457,
      "resolved_canonical": 457,
      "unresolved_ids": 0,
      "ambiguous_ids": 0,
      "resolved_starters": 161,
      "unresolved_starters": 0,
      "missing_weekly_projections": 385,
      "missing_season_projections": 390,
      "outside_projection_universe": {},
      "unresolved_starter_ids": [],
      "unresolved_skill_ids": []
    }
  ],
  "aggregate": {
    "total_distinct_rostered_ids": 1320,
    "resolved_canonical": 1320,
    "unresolved_ids": 0,
    "ambiguous_ids": 0,
    "resolved_starters": 489,
    "unresolved_starters": 0,
    "missing_weekly_projections": 1105,
    "missing_season_projections": 1116
  },
  "unresolved_artifact_count": 0,
  "recommendation_gate_failed": false,
  "gate_failures": []
}

## Projection mode
{
  "weekly_v2_state": "trained",
  "auto_publish_allowed": false,
  "manifest_uri": "file:///C:/Users/rdfer/Projects/fantasy-projections/output/weekly_v2/models/season%3D2026/manifest.json",
  "reasons": [
    "evaluation_promotion_failed:2023: dispersion outside policy",
    "evaluation_promotion_failed:2024: dispersion outside policy"
  ],
  "shadow_mode_label": "fixture/fallback \u2014 not production-quality football advice"
}

## League scoring shadow
{
  "league_count": 6,
  "distinct_contract_hashes": 6,
  "sample_player_count": 25,
  "max_cross_league_spread": 3.694,
  "leagues": [
    {
      "league_id": "1389344450517430272",
      "display_name": "The NY Sack Exchange I",
      "contract_hash": "861a9a66a4afed806d8579bd9c1946b95f9e8fdfa4bc67e653fe8b20c0a2c69a",
      "unsupported_keys": [],
      "top_players": [
        {
          "player_id": "00-0038542",
          "name": "Bijan Robinson",
          "position": "RB",
          "league_scored_points": 16.255,
          "model_fantasy_points": 23.149
        },
        {
          "player_id": "00-0036900",
          "name": "Ja'Marr Chase",
          "position": "WR",
          "league_scored_points": 15.389,
          "model_fantasy_points": 27.079
        },
        {
          "player_id": "00-0033288",
          "name": "George Kittle",
          "position": "TE",
          "league_scored_points": 13.412,
          "model_fantasy_points": 23.808
        },
        {
          "player_id": "00-0037744",
          "name": "Trey McBride",
          "position": "TE",
          "league_scored_points": 13.312,
          "model_fantasy_points": 24.536
        },
        {
          "player_id": "00-0039075",
          "name": "Puka Nacua",
          "position": "WR",
          "league_scored_points": 13.116,
          "model_fantasy_points": 22.982
        },
        {
          "player_id": "00-0036963",
          "name": "Amon-Ra St. Brown",
          "position": "WR",
          "league_scored_points": 12.572,
          "model_fantasy_points": 21.677
        },
        {
          "player_id": "00-0039040",
          "name": "De'Von Achane",
          "position": "RB",
          "league_scored_points": 12.465,
          "model_fantasy_points": 17.682
        },
        {
          "player_id": "00-0040126",
          "name": "Colston Loveland",
          "position": "TE",
          "league_scored_points": 11.915,
          "model_fantasy_points": 21.194
        },
        {
          "player_id": "00-0034351",
          "name": "Dallas Goedert",
          "position": "TE",
          "league_scored_points": 11.603,
          "model_fantasy_points": 20.5
        },
        {
          "player_id": "00-0036970",
          "name": "Kyle Pitts",
          "position": "TE",
          "league_scored_points": 10.852,
          "model_fantasy_points": 19.438
        }
      ]
    },
    {
      "league_id": "1355920300633513984",
      "display_name": "Tits Out for The Ladz XII (TWELVE\ud83d\ude24)",
      "contract_hash": "8bce4bb6aedd594065642deefced022ac18bde1655b45e56b7fc510ca86c772a",
      "unsupported_keys": [],
      "top_players": [
        {
          "player_id": "00-0038542",
          "name": "Bijan Robinson",
          "position": "RB",
          "league_scored_points": 16.255,
          "model_fantasy_points": 23.149
        },
        {
          "player_id": "00-0036900",
          "name": "Ja'Marr Chase",
          "position": "WR",
          "league_scored_points": 15.389,
          "model_fantasy_points": 27.079
        },
        {
          "player_id": "00-0033288",
          "name": "George Kittle",
          "position": "TE",
          "league_scored_points": 13.412,
          "model_fantasy_points": 23.808
        },
        {
          "player_id": "00-0037744",
          "name": "Trey McBride",
          "position": "TE",
          "league_scored_points": 13.312,
          "model_fantasy_points": 24.536
        },
        {
          "player_id": "00-0039075",
          "name": "Puka Nacua",
          "position": "WR",
          "league_scored_points": 13.116,
          "model_fantasy_points": 22.982
        },
        {
          "player_id": "00-0036963",
          "name": "Amon-Ra St. Brown",
          "position": "WR",
          "league_scored_points": 12.572,
          "model_fantasy_points": 21.677
        },
        {
          "player_id": "00-0039040",
          "name": "De'Von Achane",
          "position": "RB",
          "league_scored_points": 12.465,
          "model_fantasy_points": 17.682
        },
        {
          "player_id": "00-0040126",
          "name": "Colston Loveland",
          "position": "TE",
          "league_scored_points": 11.915,
          "model_fantasy_points": 21.194
        },
        {
          "player_id": "00-0034351",
          "name": "Dallas Goedert",
          "position": "TE",
          "league_scored_points": 11.603,
          "model_fantasy_points": 20.5
        },
        {
          "player_id": "00-0036970",
          "name": "Kyle Pitts",
          "position": "TE",
          "league_scored_points": 10.852,
          "model_fantasy_points": 19.438
        }
      ]
    },
    {
      "league_id": "1317270144682070016",
      "display_name": "C2C. The real superconference",
      "contract_hash": "fd3dbb1981f20b64ae616df5c5ab69f238649afdcd91e91d448c85f336637475",
      "unsupported_keys": [],
      "top_players": [
        {
          "player_id": "00-0038542",
          "name": "Bijan Robinson",
          "position": "RB",
          "league_scored_points": 16.255,
          "model_fantasy_points": 23.149
        },
        {
          "player_id": "00-0036900",
          "name": "Ja'Marr Chase",
          "position": "WR",
          "league_scored_points": 15.389,
          "model_fantasy_points": 27.079
        },
        {
          "player_id": "00-0033288",
          "name": "George Kittle",
          "position": "TE",
          "league_scored_points": 13.412,
          "model_fantasy_points": 23.808
        },
        {
          "player_id": "00-0037744",
          "name": "Trey McBride",
          "position": "TE",
          "league_scored_points": 13.312,
          "model_fantasy_points": 24.536
        },
        {
          "player_id": "00-0039075",
          "name": "Puka Nacua",
          "position": "WR",
          "league_scored_points": 13.116,
          "model_fantasy_points": 22.982
        },
        {
          "player_id": "00-0036963",
          "name": "Amon-Ra St. Brown",
          "position": "WR",
          "league_scored_points": 12.572,
          "model_fantasy_points": 21.677
        },
        {
          "player_id": "00-0039040",
          "name": "De'Von Achane",
          "position": "RB",
          "league_scored_points": 12.465,
          "model_fantasy_points": 17.682
        },
        {
          "player_id": "00-0040126",
          "name": "Colston Loveland",
          "position": "TE",
          "league_scored_points": 11.915,
          "model_fantasy_points": 21.194
        },
        {
          "player_id": "00-0034351",
          "name": "Dallas Goedert",
          "position": "TE",
          "league_scored_points": 11.603,
          "model_fantasy_points": 20.5
        },
        {
          "player_id": "00-0036970",
          "name": "Kyle Pitts",
          "position": "TE",
          "league_scored_points": 10.852,
          "model_fantasy_points": 19.438
        }
      ]
    },
    {
      "league_id": "1306489414548979712",
      "display_name": "Dynastical Cucks",
      "contract_hash": "bd87f15387a0487e454ae35dd98ddfb3c9b1f95a919fcf5d705e9480ae19d22f",
      "unsupported_keys": [],
      "top_players": [
        {
          "player_id": "00-0038542",
          "name": "Bijan Robinson",
          "position": "RB",
          "league_scored_points": 16.255,
          "model_fantasy_points": 23.149
        },
        {
          "player_id": "00-0036900",
          "name": "Ja'Marr Chase",
          "position": "WR",
          "league_scored_points": 15.389,
          "model_fantasy_points": 27.079
        },
        {
          "player_id": "00-0033288",
          "name": "George Kittle",
          "position": "TE",
          "league_scored_points": 13.412,
          "model_fantasy_points": 23.808
        },
        {
          "player_id": "00-0037744",
          "name": "Trey McBride",
          "position": "TE",
          "league_scored_points": 13.312,
          "model_fantasy_points": 24.536
        },
        {
          "player_id": "00-0039075",
          "name": "Puka Nacua",
          "position": "WR",
          "league_scored_points": 13.116,
          "model_fantasy_points": 22.982
        },
        {
          "player_id": "00-0036963",
          "name": "Amon-Ra St. Brown",
          "position": "WR",
          "league_scored_points": 12.572,
          "model_fantasy_points": 21.677
        },
        {
          "player_id": "00-0039040",
          "name": "De'Von Achane",
          "position": "RB",
          "league_scored_points": 12.465,
          "model_fantasy_points": 17.682
        },
        {
          "player_id": "00-0040126",
          "name": "Colston Loveland",
          "position": "TE",
          "league_scored_points": 11.915,
          "model_fantasy_points": 21.194
        },
        {
          "player_id": "00-0034351",
          "name": "Dallas Goedert",
          "position": "TE",
          "league_scored_points": 11.603,
          "model_fantasy_points": 20.5
        },
        {
          "player_id": "00-0036970",
          "name": "Kyle Pitts",
          "position": "TE",
          "league_scored_points": 10.852,
          "model_fantasy_points": 19.438
        }
      ]
    },
    {
      "league_id": "1312127020972404736",
      "display_name": "Hoe Ass Dynasty League",
      "contract_hash": "d6add0512b2bbacd98981fabe2c067f92dc42e0a1801a42d5238e992433c68b8",
      "unsupported_keys": [],
      "top_players": [
        {
          "player_id": "00-0036900",
          "name": "Ja'Marr Chase",
          "position": "WR",
          "league_scored_points": 18.655,
          "model_fantasy_points": 27.079
        },
        {
          "player_id": "00-0038542",
          "name": "Bijan Robinson",
          "position": "RB",
          "league_scored_points": 18.023,
          "model_fantasy_points": 23.149
        },
        {
          "player_id": "00-0037744",
          "name": "Trey McBride",
          "position": "TE",
          "league_scored_points": 17.005,
          "model_fantasy_points": 24.536
        },
        {
          "player_id": "00-0033288",
          "name": "George Kittle",
          "position": "TE",
          "league_scored_points": 16.526,
          "model_fantasy_points": 23.808
        },
        {
          "player_id": "00-0039075",
          "name": "Puka Nacua",
          "position": "WR",
          "league_scored_points": 15.959,
          "model_fantasy_points": 22.982
        },
        {
          "player_id": "00-0036963",
          "name": "Amon-Ra St. Brown",
          "position": "WR",
          "league_scored_points": 15.101,
          "model_fantasy_points": 21.677
        },
        {
          "player_id": "00-0040126",
          "name": "Colston Loveland",
          "position": "TE",
          "league_scored_points": 14.806,
          "model_fantasy_points": 21.194
        },
        {
          "player_id": "00-0034351",
          "name": "Dallas Goedert",
          "position": "TE",
          "league_scored_points": 14.35,
          "model_fantasy_points": 20.5
        },
        {
          "player_id": "00-0039040",
          "name": "De'Von Achane",
          "position": "RB",
          "league_scored_points": 14.119,
          "model_fantasy_points": 17.682
        },
        {
          "player_id": "00-0036970",
          "name": "Kyle Pitts",
          "position": "TE",
          "league_scored_points": 13.652,
          "model_fantasy_points": 19.438
        }
      ]
    },
    {
      "league_id": "1311470531635052544",
      "display_name": "Tainticklers",
      "contract_hash": "3b3da51a5e7ddd1ded9b4a406f748e523f22667c9d670445408d203ff071afd6",
      "unsupported_keys": [],
      "top_players": [
        {
          "player_id": "00-0036900",
          "name": "Ja'Marr Chase",
          "position": "WR",
          "league_scored_points": 18.655,
          "model_fantasy_points": 27.079
        },
        {
          "player_id": "00-0038542",
          "name": "Bijan Robinson",
          "position": "RB",
          "league_scored_points": 18.023,
          "model_fantasy_points": 23.149
        },
        {
          "player_id": "00-0037744",
          "name": "Trey McBride",
          "position": "TE",
          "league_scored_points": 17.005,
          "model_fantasy_points": 24.536
        },
        {
          "player_id": "00-0033288",
          "name": "George Kittle",
          "position": "TE",
          "league_scored_points": 16.526,
          "model_fantasy_points": 23.808
        },
        {
          "player_id": "00-0039075",
          "name": "Puka Nacua",
          "position": "WR",
          "league_scored_points": 15.959,
          "model_fantasy_points": 22.982
        },
        {
          "player_id": "00-0036963",
          "name": "Amon-Ra St. Brown",
          "position": "WR",
          "league_scored_points": 15.101,
          "model_fantasy_points": 21.677
        },
        {
          "player_id": "00-0040126",
          "name": "Colston Loveland",
          "position": "TE",
          "league_scored_points": 14.806,
          "model_fantasy_points": 21.194
        },
        {
          "player_id": "00-0034351",
          "name": "Dallas Goedert",
          "position": "TE",
          "league_scored_points": 14.35,
          "model_fantasy_points": 20.5
        },
        {
          "player_id": "00-0039040",
          "name": "De'Von Achane",
          "position": "RB",
          "league_scored_points": 14.119,
          "model_fantasy_points": 17.682
        },
        {
          "player_id": "00-0036970",
          "name": "Kyle Pitts",
          "position": "TE",
          "league_scored_points": 13.652,
          "model_fantasy_points": 19.438
        }
      ]
    }
  ],
  "cross_league_spreads": [
    {
      "player_id": "00-0037744",
      "min_score": 13.312,
      "max_score": 17.005,
      "spread": 3.694
    },
    {
      "player_id": "00-0036900",
      "min_score": 15.389,
      "max_score": 18.655,
      "spread": 3.265
    },
    {
      "player_id": "00-0033288",
      "min_score": 13.412,
      "max_score": 16.526,
      "spread": 3.114
    },
    {
      "player_id": "00-0040663",
      "min_score": 10.394,
      "max_score": 13.382,
      "spread": 2.988
    },
    {
      "player_id": "00-0040126",
      "min_score": 11.915,
      "max_score": 14.806,
      "spread": 2.892
    },
    {
      "player_id": "00-0039075",
      "min_score": 13.116,
      "max_score": 15.959,
      "spread": 2.844
    },
    {
      "player_id": "00-0036970",
      "min_score": 10.852,
      "max_score": 13.652,
      "spread": 2.799
    },
    {
      "player_id": "00-0039338",
      "min_score": 10.663,
      "max_score": 13.455,
      "spread": 2.792
    },
    {
      "player_id": "00-0034351",
      "min_score": 11.603,
      "max_score": 14.35,
      "spread": 2.747
    },
    {
      "player_id": "00-0030506",
      "min_score": 10.463,
      "max_score": 13.17,
      "spread": 2.707
    },
    {
      "player_id": "00-0036963",
      "min_score": 12.572,
      "max_score": 15.101,
      "spread": 2.529
    },
    {
      "player_id": "00-0038543",
      "min_score": 10.685,
      "max_score": 12.836,
      "spread": 2.152
    },
    {
      "player_id": "00-0036322",
      "min_score": 9.746,
      "max_score": 11.885,
      "spread": 2.139
    },
    {
      "player_id": "00-0035640",
      "min_score": 9.738,
      "max_score": 11.874,
      "spread": 2.136
    },
    {
      "player_id": "00-0039067",
      "min_score": 8.145,
      "max_score": 10.206,
      "spread": 2.061
    }
  ],
  "validation": {
    "six_distinct_contracts": true,
    "cross_league_scoring_differs": true,
    "all_contracts_publishable": true
  },
  "status": "ok",
  "source_parquet": "C:\\Users\\rdfer\\Projects\\fantasy-projections\\output\\weekly_v2\\season=2026\\week=01\\weekly_projections.parquet"
}

## Blockers
- automatic weekly publication blocked until evaluation promotion passes
- PostgreSQL runtime unverified on this machine
- Docker compose runtime unverified on this machine
- email delivery unverified
- OpenAI assistant path unverified
- internet deployment unverified
