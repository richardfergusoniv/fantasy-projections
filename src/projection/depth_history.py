"""Preseason depth charts for ANY season, harmonized across nflverse's two
schemas - the historical input the availability model (Phase 11) was
missing.

Why this module exists
----------------------
`train.fit_availability` shipped with a documented weakness: the QB games
model was no better than carrying season-N games forward (consistency
-0.25, 4/8 seasons), and every position over-predicted games for players
who were about to be nobody's starter. The stated blocker, repeated in
backtest.py, was that "no historical curated depth chart exists" - so the
only depth signal in the pipeline was src/depth_chart/starters_2026.csv,
which covers exactly one season and therefore cannot be a training feature.

That blocker was false. `depth_charts` in the project DB carries 2016-2026,
and a player's preseason position on his team's chart is a strong available
predictor of how many games he will actually play. Availability metrics must
be regenerated whenever games-played semantics or feed harmonization changes;
the canonical current values are printed by ``src.projection.backtest``.

The depth signal improves the availability model across the rolling folds,
but absolute metrics are deliberately not copied into this module: they move
when appearance semantics or feed harmonization changes. The canonical
current values are emitted by the backtest.

Two schemas, one meaning
------------------------
nflverse changed the depth-chart feed partway through this project's data
range, and the two halves live in the same table:

  2016-2024  `season`/`week`/`game_type` populated; rank is `depth_team`
             (1/2/3), and several players can share one `depth_team` value
             at the same position. Earliest available chart is week 1 REG.
  2025-2026  `season IS NULL`; daily snapshots keyed by `dt`, rank is
             `pos_rank`, a true per-(team, position) ordinal. Earliest
             August snapshot is used, matching the timing of the live
             2026 run (2026-08-01) and of a week-1 chart.

Both are reduced to a tier-preserving availability rank.  The old schema's
`depth_team` is already a tier and is never split.  The new feed is a true
ordinal; WR ordinals are paired (1-2 -> tier 1, 3-4 -> tier 2, etc.) because
the old feed carries a median of two WR slots per tier. Other positions have
a median of one player per tier and retain their ordinal.

The truncation, and why it is not optional
------------------------------------------
The two feeds are not equally deep. Old-era charts list a median of 2 QB /
3 RB / 5-6 WR / 3 TE per team; the new feed lists 4 / 6 / 11-12 / 6. So
"absent from the chart" silently means different things in the two eras -
the off-chart share of returning players runs 0.28-0.43 in 2022-2024 but
0.04-0.16 in 2025-2026. Training on the old meaning and predicting on the
new one would feed the model a feature whose definition moved underneath
it (the same denominator-shift failure this project has hit before).

PRESEASON_CHART_DEPTH truncates every season to a common per-position tier.
Those boundaries must be revalidated by the rolling availability backtest
whenever either source schema or the tier translation changes.

`load_preseason_depth_chart` returns the chart UNtruncated: a full ordinal
for the new feed and the source tier for the old feed (which has no honest
within-tier ordering). Truncation is applied by `attach_availability_depth_rank`,
which is the availability model's specific harmonization requirement.

Relationship to the curated chart
---------------------------------
This is NOT a replacement for src/depth_chart/starters_2026.csv. That file
stays authoritative for membership, team assignment, and role tier, and is
hand-verified. This module supplies a *feature*, and its one qualification
is the one the curated file cannot meet: it exists for every season, so a
model can be trained on it and held out honestly.
"""
import sys

import pandas as pd

from src.projection.data_prep import get_conn

POSITIONS = ["QB", "RB", "WR", "TE"]

# Per-position chart depth beyond which a listing is treated as off-chart.
# Chosen to equalize the off-chart rate between the 2016-2024 and 2025-2026
# feeds - see this module's docstring. Availability-model-specific; volume
# work should use the untruncated ordinal.
PRESEASON_CHART_DEPTH = {"QB": 2, "RB": 3, "WR": 3, "TE": 3}

AVAILABILITY_DEPTH_FEATURE = "target_depth_rank"

_CHART_CACHE = {}


def _load_old_schema(conn, season):
    """2016-2024: `depth_team` tiers off the week-1 regular-season chart."""
    df = pd.read_sql(
        "SELECT club_code AS team, position, depth_team, gsis_id AS player_id, full_name "
        "FROM depth_charts WHERE season = ? AND week = 1 AND game_type = 'REG' "
        "AND formation = 'Offense' AND gsis_id IS NOT NULL",
        conn, params=(season,))
    if df.empty:
        return df
    df["depth_team"] = pd.to_numeric(df["depth_team"], errors="coerce")
    df = df.dropna(subset=["depth_team", "team", "position"])
    df = df[df["position"].isin(POSITIONS)]
    # A player can be listed at more than one slot (a RB who is also the
    # FB, a TE at FB); keep his best listing per position.  depth_team is a
    # tier, not an ordinal: tied players must retain the same rank.
    df = df.sort_values("depth_team").drop_duplicates(["player_id", "position"])
    df["depth_rank"] = df["depth_team"]
    df["availability_rank"] = df["depth_team"]
    df["source"] = f"nflverse_week1_{season}"
    return df


def _load_new_schema(conn, season):
    """2025+: the earliest August daily snapshot, ranked by `pos_rank`."""
    dt = pd.read_sql(
        "SELECT MIN(dt) AS d FROM depth_charts WHERE season IS NULL AND dt >= ? AND dt < ?",
        conn, params=(f"{season}-08-01", f"{season}-09-01")).at[0, "d"]
    if dt is None:
        return pd.DataFrame()
    df = pd.read_sql(
        "SELECT team, pos_abb AS position, pos_rank AS depth_rank, gsis_id AS player_id, "
        "player_name AS full_name FROM depth_charts WHERE season IS NULL AND dt = ? "
        "AND pos_abb IN ('QB','RB','WR','TE') AND gsis_id IS NOT NULL",
        conn, params=(dt,))
    if df.empty:
        return df
    df = df.sort_values("depth_rank").drop_duplicates(["player_id", "position"])
    # Translate the true ordinal to the old feed's coarser tier semantics.
    # Old WR tiers contain a median of two formation slots; all other
    # positions contain a median of one.
    df["availability_rank"] = df["depth_rank"]
    wr = df["position"].eq("WR")
    df.loc[wr, "availability_rank"] = ((df.loc[wr, "depth_rank"] - 1) // 2) + 1
    df["source"] = f"nflverse_{dt[:10]}"
    return df


def load_preseason_depth_chart(season, conn=None):
    """The depth chart as it stood entering `season`, one row per (player,
    position), with a 1-based `depth_rank` within (team, position).

    UNtruncated - see PRESEASON_CHART_DEPTH. Returns an empty frame (not an
    error) for a season with no chart, so callers degrade to "no depth
    signal" the way load_depth_chart does for a missing curated file.

    Cached per season (a leave-one-transition-out backtest asks for the
    same handful of seasons dozens of times); a copy is handed out so a
    caller cannot mutate the cached frame out from under the next one.
    """
    if season in _CHART_CACHE:
        return _CHART_CACHE[season].copy()
    own_conn = conn is None
    conn = conn or get_conn()
    try:
        df = _load_old_schema(conn, season)
        if df.empty:
            df = _load_new_schema(conn, season)
    finally:
        if own_conn:
            conn.close()
    cols = ["player_id", "team", "position", "depth_rank", "availability_rank", "full_name", "source"]
    df = df[cols].reset_index(drop=True) if not df.empty else pd.DataFrame(columns=cols)
    _CHART_CACHE[season] = df
    return df.copy()


def attach_depth_rank(df, season, conn=None):
    """Add `nfl_depth_rank` to `df` (needs `player_id` and `position`): the
    player's untruncated source rank on the chart entering `season`, NaN when
    he is not on it at all. Old-feed ties remain tied; new-feed ranks are
    true ordinals.

    Distinct from attach_availability_depth_rank, and the difference is the
    point: availability needs the two chart eras to mean the same thing and
    therefore truncates, while the volume discount needs to tell a WR4 from
    a WR8 and therefore must not. Both are the same underlying chart.
    """
    chart = load_preseason_depth_chart(season, conn=conn)
    out = df.copy()
    if chart.empty:
        out["nfl_depth_rank"] = float("nan")
        return out
    ranks = chart.set_index(["player_id", "position"])["depth_rank"]
    idx = pd.MultiIndex.from_arrays([out["player_id"], out["position"]])
    out["nfl_depth_rank"] = ranks.reindex(idx).to_numpy(dtype=float)
    return out


def attach_availability_depth_rank(df, season, conn=None):
    """Add `target_depth_rank` to `df` (needs `player_id` and `position`):
    the player's rank on the chart entering `season`, truncated per
    PRESEASON_CHART_DEPTH, NaN when he is off the chart at that depth.

    NaN is the real signal here, not a missing value to be filled -
    LightGBM splits on it natively, and "not on his team's chart" is
    precisely the state that predicts ~1-2 games played. Leaving it NaN
    rather than encoding a sentinel is why no separate on-chart boolean is
    carried. Its incremental value must be remeasured with the current
    appearance-based target before changing this schema.
    """
    chart = load_preseason_depth_chart(season, conn=conn)
    out = df.copy()
    if chart.empty:
        # Loud, because the quiet version of this is the worst failure mode
        # in the module: NaN means "off his team's chart," which predicts
        # ~1-2 games, so a missing chart would hand the ENTIRE league a
        # replacement-level availability and nothing downstream would look
        # obviously broken - the season totals would just all be small.
        # Degrading rather than raising is still right (the rest of the
        # projection is usable), but it must never be silent.
        print(f"WARNING: no preseason depth chart for {season} - every player will be "
              f"treated as off-chart by the availability model, which will suppress "
              f"projected_games league-wide. Check the depth_charts ingest.",
              file=sys.stderr)
        out[AVAILABILITY_DEPTH_FEATURE] = float("nan")
        return out
    keep = chart[
        chart["availability_rank"] <= chart["position"].map(PRESEASON_CHART_DEPTH)
    ]
    ranks = keep.set_index(["player_id", "position"])["availability_rank"]
    idx = pd.MultiIndex.from_arrays([out["player_id"], out["position"]])
    out[AVAILABILITY_DEPTH_FEATURE] = ranks.reindex(idx).to_numpy(dtype=float)
    return out
