"""League-specific draw scoring on weekly-v2 stat components."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app.scoring.compiler import compile_sleeper_scoring, score_stat_draw

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "scoring"


def _base_rb_draw() -> dict[str, float]:
    return {
        "rush_yards": 80,
        "rush_tds": 1,
        "rec_yards": 25,
        "rec_tds": 0,
        "receptions": 4,
        "rec_first_downs": 2,
        "rush_first_downs": 3,
    }


def test_full_ppr_scores_receptions_higher_than_half_ppr():
    draw = _base_rb_draw()
    full = compile_sleeper_scoring({"rec": 1.0, "rec_yd": 0.1, "rush_yd": 0.1}, ["RB", "BN"])
    half = compile_sleeper_scoring({"rec": 0.5, "rec_yd": 0.1, "rush_yd": 0.1}, ["RB", "BN"])
    assert score_stat_draw(draw, full, position="RB") > score_stat_draw(draw, half, position="RB")


def test_ppfd_rule_scores_at_draw_level():
    draw = _base_rb_draw()
    payload = json.loads((FIXTURE_DIR / "ppfd.json").read_text(encoding="utf-8"))
    with_ppfd = compile_sleeper_scoring(payload["scoring_settings"], payload["roster_positions"])
    without_ppfd = compile_sleeper_scoring(
        {k: v for k, v in payload["scoring_settings"].items() if not k.endswith("_fd")},
        payload["roster_positions"],
    )
    assert score_stat_draw(draw, with_ppfd, position="RB") > score_stat_draw(
        draw, without_ppfd, position="RB"
    )


@pytest.mark.skipif(
    not Path("output/weekly_v2/season=2026/week=01/weekly_projections.parquet").exists(),
    reason="trained weekly output missing",
)
def test_weekly_inference_output_differs_from_preseason_scaling():
    import polars as pl

    from src.app.projections.weekly_inference import hash_scaled_preseason_rows
    from src.app.projections.loader import ReleaseBundleLoader

    frame = pl.read_parquet("output/weekly_v2/season=2026/week=01/weekly_projections.parquet")
    model_fp = dict(zip(frame["gsis_id"].to_list(), frame["fantasy_points"].to_list()))
    players = ReleaseBundleLoader(season=2026).load()
    scaled = {
        row.player_id: row.mean_json["points"]
        for row in hash_scaled_preseason_rows(
            players,
            1,
            factor_fn=lambda pid, week: 1.0 + ((int(pid[-4:], 16) / 0xFFFF) - 0.5) * 0.12,
        )
    }
    overlap = [pid for pid in model_fp if pid in scaled]
    assert overlap
    diffs = [abs(model_fp[pid] - scaled[pid]) for pid in overlap[:200]]
    assert max(diffs) > 0.5


SHADOW_DB = Path("output/live_shadow/shadow_app.db")
OWNER_CONFIG = Path("config/sleeper_owner.json")


@pytest.mark.skipif(
    not Path("output/weekly_v2/season=2026/week=01/weekly_projections.parquet").exists()
    or not SHADOW_DB.exists()
    or not OWNER_CONFIG.exists(),
    reason="trained weekly output or live shadow database missing",
)
def test_six_live_leagues_score_weekly_components_differently():
    import os

    import polars as pl

    from src.app.config import get_settings
    from src.app.league.sleeper.owner_config import load_owner_config
    from src.app.league.sleeper.shadow_sync import ShadowSyncOptions, _configure_shadow_environment
    from src.app.persistence.database import get_session, reset_engine
    from src.app.persistence.models import League
    from src.app.projections.weekly_league_scoring import (
        LeagueScoringContract,
        score_weekly_frame_for_leagues,
    )

    options = ShadowSyncOptions(config_path=OWNER_CONFIG)
    _configure_shadow_environment(options)
    get_settings.cache_clear()
    reset_engine()

    config = load_owner_config(OWNER_CONFIG)
    contracts: list[LeagueScoringContract] = []
    with get_session() as session:
        for entry in config.leagues:
            league = (
                session.query(League).filter(League.league_id == entry.league_id).one_or_none()
            )
            assert league is not None and league.raw_json
            contracts.append(
                LeagueScoringContract.from_league_json(
                    league_id=entry.league_id,
                    display_name=entry.display_name,
                    raw_json=league.raw_json,
                )
            )
    assert len(contracts) == 6

    frame = pl.read_parquet("output/weekly_v2/season=2026/week=01/weekly_projections.parquet")
    artifact = score_weekly_frame_for_leagues(frame, contracts)
    assert artifact["validation"]["six_distinct_contracts"]
    assert artifact["validation"]["cross_league_scoring_differs"]
    assert artifact["validation"]["all_contracts_publishable"]

    os.environ.pop("DATABASE_URL", None)
    get_settings.cache_clear()
    reset_engine()
