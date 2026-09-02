"""Probability and availability semantics for two-stage volume + injury haircut."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.projection.weekly.features.injuries import apply_injury_haircut
from src.projection.weekly.models.base import dataframe_to_matrix
from src.projection.weekly.models.volume import TwoStageVolumeModel


def _feature_row(
  *,
  play_prob: float = 1.0,
  is_out: bool = False,
  target_share_l5: float = 0.2,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "play_prob": [play_prob],
            "is_out": [is_out],
            "is_doubtful": [False],
            "is_questionable": [False],
            "target_share_l5": [target_share_l5],
            "target_share_prior_season": [target_share_l5],
            "carry_share_l5": [0.1],
            "carry_share_prior_season": [0.1],
            "games_played_prior": [10],
            "age": [26.0],
            "is_rookie": [False],
            "depth_rank": [1.0],
        }
    )


def _train_toy_two_stage() -> TwoStageVolumeModel:
    rows = []
    for i in range(40):
        used = i % 3 != 0
        rows.append(
            {
                "play_prob": 1.0,
                "is_out": False,
                "target_share_l5": 0.15,
                "target_share_prior_season": 0.15,
                "carry_share_l5": 0.1,
                "carry_share_prior_season": 0.1,
                "games_played_prior": 10,
                "age": 26.0,
                "is_rookie": False,
                "depth_rank": 1.0,
                "target_share": 0.2 if used else 0.0,
            }
        )
    df = pl.DataFrame(rows)
    features = [c for c in df.columns if c != "target_share"]
    model = TwoStageVolumeModel(targets=["target_share"], feature_cols=features)
    X = dataframe_to_matrix(df, features)
    model.fit(X, df.select(["target_share"]), conditional_model_type="ridge", participation_model_type="ridge")
    return model


@pytest.mark.parametrize(
    "play_prob,is_out,label",
    [
        (1.0, False, "healthy_starter"),
        (0.6, False, "questionable_starter"),
        (1.0, True, "inactive"),
        (1.0, False, "healthy_reserve"),
    ],
)
def test_injury_haircut_monotone_in_play_prob(play_prob: float, is_out: bool, label: str):
    df = _feature_row(play_prob=play_prob, is_out=is_out).with_columns(
        pl.lit(0.25).alias("pred_target_share")
    )
    out = apply_injury_haircut(df, mode="shares")
    share = float(out["pred_target_share"][0])
    if is_out or play_prob <= 1e-6:
        assert share == 0.0, label
    else:
        assert share <= 0.25 + 1e-9, label
        assert share >= 0.0, label


def test_two_stage_expected_share_bounded():
    model = _train_toy_two_stage()
    healthy = _feature_row(play_prob=1.0, is_out=False)
    X = dataframe_to_matrix(healthy, model.feature_cols)
    shares, probs = model.predict(X)
    assert 0.0 <= shares["target_share"][0] <= 1.0
    assert 0.0 <= probs["target_share"][0] <= 1.0


def test_inactive_player_zeroed_after_haircut_chain():
    model = _train_toy_two_stage()
    inactive = _feature_row(play_prob=0.0, is_out=True)
    X = dataframe_to_matrix(inactive, model.feature_cols)
    shares, _ = model.predict(X)
    framed = inactive.with_columns(pl.Series("pred_target_share", shares["target_share"]))
    haircut = apply_injury_haircut(framed, mode="shares")
    assert float(haircut["pred_target_share"][0]) == 0.0


def test_play_prob_feature_does_not_double_apply_when_out():
    """Out flag should dominate play_prob; haircut must not leave residual share."""
    df = _feature_row(play_prob=0.8, is_out=True).with_columns(
        pl.lit(0.3).alias("pred_target_share")
    )
    out = apply_injury_haircut(df, mode="shares")
    assert float(out["pred_target_share"][0]) == 0.0
