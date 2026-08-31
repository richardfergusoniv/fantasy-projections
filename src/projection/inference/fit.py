"""Fit v3 probabilistic model artifacts."""
from __future__ import annotations

from src.projection.models.opportunity_shares import fit_opportunity_shares
from src.projection.models.team_environment import fit_team_environment


def fit_v3_models(feat, train_pairs, long_board=None) -> dict:
    """Train team environment and compositional share models."""
    share_frame = long_board if long_board is not None else feat
    return {
        "team_environment": fit_team_environment(feat, train_pairs),
        "opportunity_shares": fit_opportunity_shares(share_frame),
    }
