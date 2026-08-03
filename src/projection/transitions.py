"""Season N -> season N+1 transition-pair construction, shared by
train.py and backtest.py.

Train/predict framing (explicit, per the spec's ask to think carefully
about this): each training row is (player's season N feature vector) ->
(player's season N+1 per-game rate). This is a genuine next-season
projection task, not same-season leakage - season N's features (shares,
snap %, OL quality, OC tendencies) are all fully known by the end of
season N, and the label is strictly season N+1.

One real limitation, documented rather than hidden: season N's OWN
opportunity/scheme features (oc_tendency_profiles, OL quality) are used as
the proxy for season N+1 conditions, because season N+1's actual observed
tendencies obviously don't exist yet at prediction time for a genuinely
future season. This mirrors what a real preseason projection has to do
(assume the most recently observed team context persists), and is exactly
why `oc_tendency_profiles.inherited_*` exists for a genuinely new OC - but
wiring that in for a not-yet-played season is Phase 5's job (choosing which
season's row to feed the trained model here), not this module's.
"""
import numpy as np
import pandas as pd

from src.projection.features import FEATURE_COLS, TARGET_STATS

EXTRA_FEATURES = ["games_played"]
ALL_FEATURES = FEATURE_COLS + EXTRA_FEATURES


def build_transition_pairs(feat, position, stat, season_pairs):
    """Stack (X, y) rows across the given list of (season_from, season_to)
    pairs for one position/stat. Requires games_played > 0 in season_to
    (a real per-game rate to learn from) but NOT in season_from - a
    veteran coming off an injury-limited season is still a real veteran
    with real trailing features, just noisier ones (games_played is itself
    a feature, so the model can learn to discount low-sample seasons)."""
    pos_df = feat[feat["position"] == position]
    y_col = f"{stat}_pg"

    rows = []
    for season_from, season_to in season_pairs:
        a = pos_df[pos_df["season"] == season_from][["player_id", "team"] + ALL_FEATURES + [y_col]]
        a = a.rename(columns={y_col: "naive_pred"})  # season_from's own rate = naive carry-forward baseline
        b = pos_df[pos_df["season"] == season_to][["player_id", y_col, "games_played"]].rename(
            columns={"games_played": "games_played_to"}
        )
        merged = a.merge(b, on="player_id", how="inner")
        merged = merged[merged["games_played_to"] > 0]
        merged["season_from"] = season_from
        merged["season_to"] = season_to
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out
