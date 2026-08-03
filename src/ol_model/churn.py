"""Lineup-churn metric per team-season, and the resulting player-level
confidence flag - see PHASE2_STABILITY_INVESTIGATION.md section 3. A team
that runs the same 5 linemen together for most of a season makes their
indicator columns collinear, so ridge cannot statistically separate
individual credit within that block; a coefficient from a low-churn
team-season is really a shared unit effect, not an individual measurement.
"""
import numpy as np
import pandas as pd

from src.ol_model.data_prep import load_season

CHURN_THRESHOLD = 0.90  # top-lineup snap share at/above this -> 'unit_level'
RELEVANT_PLAYS = 50  # min plays for a player-team-season to count toward their flag


def team_season_churn(conn, seasons):
    """One row per (season, team): n_plays, n_distinct_lineups, top_lineup_frac,
    confidence_flag. Uses every run/pass play with a resolved 5-man line,
    both submodels share this table since it describes who was on the field."""
    frames = []
    for season in seasons:
        df, _ = load_season(conn, season)
        df["ol_tuple"] = df.ol_ids.apply(lambda x: tuple(sorted(x)))
        frames.append(df[["season", "posteam", "ol_tuple"]])
    allp = pd.concat(frames, ignore_index=True)

    rows = []
    for (season, team), grp in allp.groupby(["season", "posteam"]):
        counts = grp["ol_tuple"].value_counts()
        rows.append({
            "season": season, "team": team, "n_plays": len(grp),
            "n_distinct_lineups": len(counts),
            "top_lineup_frac": counts.iloc[0] / len(grp),
        })
    churn = pd.DataFrame(rows)
    churn["confidence_flag"] = np.where(churn.top_lineup_frac >= CHURN_THRESHOLD, "unit_level", "individual")
    return churn


def player_team_season_counts(conn, seasons):
    """One row per (gsis_id, season, team): play count. Used to attribute
    each player to the team-season(s) whose churn should inform their flag."""
    frames = []
    for season in seasons:
        df, _ = load_season(conn, season)
        exp = df[["season", "posteam", "ol_ids"]].explode("ol_ids").rename(columns={"ol_ids": "gsis_id"})
        frames.append(exp)
    allp = pd.concat(frames, ignore_index=True)
    return allp.groupby(["gsis_id", "season", "posteam"]).size().reset_index(name="n_plays")


def player_confidence_flags(conn, seasons, churn):
    """Per-player overall confidence flag: 'unit_level' if the player logged
    >=RELEVANT_PLAYS snaps for any team-season flagged low-churn, else
    'individual'. Also carries the worst (highest) top_lineup_frac seen
    across their relevant team-seasons for finer-grained interpretation."""
    counts = player_team_season_counts(conn, seasons)
    counts = counts[counts.n_plays >= RELEVANT_PLAYS]
    merged = counts.merge(churn, left_on=["season", "posteam"], right_on=["season", "team"], how="left")

    def summarize(g):
        return pd.Series({
            "worst_top_lineup_frac": g["top_lineup_frac"].max(),
            "confidence_flag": "unit_level" if (g["confidence_flag"] == "unit_level").any() else "individual",
            "n_team_seasons": len(g),
        })

    return merged.groupby("gsis_id").apply(summarize, include_groups=False).reset_index()
