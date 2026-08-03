"""Compute per-team-season tendency profiles from pbp/ftn/participation.

Metrics, one row per (season, team):
- plays_off, games: raw volume, for sanity-checking small samples.
- neutral_sec_per_play: median seconds between snaps for the same offense
  within a drive, restricted to a neutral game script (win prob 0.2-0.8,
  quarters 1-3) so trailing/leading garbage-time snap-spam doesn't distort
  it. Lower = faster no-huddle-style pace. This is the standard
  "neutral pace" definition used by public sites like rbsdm.com/sumersports.
- pass_oe: mean of pbp's own `pass_oe` column (nflverse's pass-rate-over-
  expected model output) over plays with win prob in [0.05, 0.95] - the
  conventional "non-garbage-time" PROE filter. Positive = more pass-heavy
  than expected given down/distance/score/time.
- pass_oe_neutral: same but tighter wp band [0.2, 0.8], per the project
  spec's explicit ask for a neutral-situation cut. Reported alongside
  pass_oe rather than instead of it since the two only differ modestly.
- play_action_rate: share of plays with ftn.is_play_action = 1. NULL before
  2022 (ftn doesn't exist yet - real gap, not imputed).
- personnel_11_rate / _12_rate / _21_rate / _other_rate: share of
  participation-matched offensive snaps in each personnel grouping.
  n_personnel_plays is the denominator, so downstream consumers can judge
  how much a season's row should be trusted (weeks with 0 matched participation
  rows -> NaN rates, not 0).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.coordinator.data_prep import get_conn, load_offense_plays, load_personnel, load_playaction

NEUTRAL_WP_LO, NEUTRAL_WP_HI = 0.20, 0.80
GARBAGE_WP_LO, GARBAGE_WP_HI = 0.05, 0.95
PERSONNEL_GROUPS = ["11", "12", "21"]


def _neutral_pace(plays):
    neutral = plays[
        (plays["wp"].between(NEUTRAL_WP_LO, NEUTRAL_WP_HI)) & (plays["qtr"] <= 3)
    ].sort_values(["game_id", "drive", "play_id"])
    grp = neutral.groupby(["game_id", "drive"])["game_seconds_remaining"]
    diff = grp.shift(1) - neutral["game_seconds_remaining"]
    neutral = neutral.assign(secs_since_prev=diff)
    # 3-40s excludes negative/zero timestamp glitches and post-review/
    # end-of-quarter jumps that aren't real play-calling pace signal.
    valid = neutral[neutral["secs_since_prev"].between(3, 40)]
    return valid.groupby(["season", "posteam"])["secs_since_prev"].median().rename("neutral_sec_per_play")


def _pass_oe(plays):
    full = plays[plays["wp"].between(GARBAGE_WP_LO, GARBAGE_WP_HI)]
    neutral = plays[plays["wp"].between(NEUTRAL_WP_LO, NEUTRAL_WP_HI)]
    a = full.groupby(["season", "posteam"])["pass_oe"].mean().rename("pass_oe")
    b = neutral.groupby(["season", "posteam"])["pass_oe"].mean().rename("pass_oe_neutral")
    return a, b


def _volume(plays):
    games = plays.groupby(["season", "posteam"])["game_id"].nunique().rename("games")
    n = plays.groupby(["season", "posteam"]).size().rename("plays_off")
    return games, n


def _play_action(plays, ftn):
    merged = plays.merge(ftn, on=["game_id", "play_id"], how="inner")
    rate = merged.groupby(["season", "posteam"])["is_play_action"].mean().rename("play_action_rate")
    n = merged.groupby(["season", "posteam"]).size().rename("n_ftn_matched_plays")
    return rate, n


def _personnel(plays, personnel):
    merged = plays.merge(personnel, on=["game_id", "play_id"], how="inner")
    labeled = merged.dropna(subset=["personnel_group"])
    n = labeled.groupby(["season", "posteam"]).size().rename("n_personnel_plays")
    out = {"n_personnel_plays": n}
    for grp in PERSONNEL_GROUPS:
        is_grp = labeled.assign(_hit=(labeled["personnel_group"] == grp))
        rate = is_grp.groupby(["season", "posteam"])["_hit"].mean().rename(f"personnel_{grp}_rate")
        out[f"personnel_{grp}_rate"] = rate
    other_rate = 1 - pd.concat([out[f"personnel_{g}_rate"] for g in PERSONNEL_GROUPS], axis=1).sum(axis=1)
    out["personnel_other_rate"] = other_rate.rename("personnel_other_rate")
    return out


def compute_team_season_tendencies(conn=None):
    own_conn = conn is None
    conn = conn or get_conn()
    plays = load_offense_plays(conn)
    ftn = load_playaction(conn)
    personnel = load_personnel(conn)

    games, n_plays = _volume(plays)
    pace = _neutral_pace(plays)
    pass_oe, pass_oe_neutral = _pass_oe(plays)
    pa_rate, n_ftn = _play_action(plays, ftn)
    pers = _personnel(plays, personnel)

    pieces = [games, n_plays, pace, pass_oe, pass_oe_neutral, pa_rate, n_ftn] + list(pers.values())
    df = pd.concat(pieces, axis=1).reset_index()
    df = df.rename(columns={"posteam": "team"})
    df["ftn_available"] = df["season"] >= 2022
    df.loc[~df["ftn_available"], ["play_action_rate", "n_ftn_matched_plays"]] = np.nan

    if own_conn:
        conn.close()
    return df.sort_values(["team", "season"]).reset_index(drop=True)


if __name__ == "__main__":
    conn = get_conn()
    df = compute_team_season_tendencies(conn)
    df.to_sql("team_tendency_profiles", conn, if_exists="replace", index=False)
    conn.close()
    pd.set_option("display.width", 220)
    print(df.shape)
    print(df[df.team.isin(["KC", "LA", "BAL"])].to_string(index=False))
