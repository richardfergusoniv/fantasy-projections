"""Re-key team-season tendency profiles to the offensive coordinator who
actually called plays, and implement the first-year-OC inheritance rule.

Inheritance rule (explicit judgment call, not a silently baked-in number):
When an OC is in the first year of a new seat, we don't have a season of
their own play-by-play at that team yet. Phase 4 needs *some* starting
tendency profile for that team-season, so we blend two known profiles:

  (a) the team's own profile from the season immediately prior (offensive
      systems have inertia - personnel, scheme carryover, holdover
      players - so a new play-caller rarely fully resets it in year 1), and
  (b) the new OC's own profile from their most recent prior play-calling
      stop, if they have one (rookie OCs / first-time callers promoted
      from a non-play-calling role or hired straight from college with no
      NFL play-calling track record have no (b) to blend, and the row is
      flagged as team-inertia-only).

The blend weight is keyed off `promotion_type`:
  - internal promotion (e.g. former passing-game coordinator staying on
    the same staff): 70% team inertia / 30% incoming OC's own track record.
    Internal hires are usually kept specifically to preserve continuity.
  - outside hire (from another franchise or college): 30% team inertia /
    70% incoming OC's own track record. Outside hires are more often
    brought in specifically to install their own system.
  - 'returning' rows (not first-year-in-seat) don't get an inherited
    profile at all - the observed profile from actual play-by-play is used
    directly.

These blend weights live in ``src.coordinator.inheritance.INHERITANCE_WEIGHTS``
(Phase C3 LOSO-fit; see OC_INHERITANCE_FIT_2026-08-14.md).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.coordinator.data_prep import get_conn
from src.coordinator.inheritance import INHERITANCE_WEIGHTS
from src.coordinator.tendencies import compute_team_season_tendencies

ASSIGNMENTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oc_assignments.csv")

METRICS = [
    "neutral_sec_per_play", "pass_oe", "pass_oe_neutral", "play_action_rate",
    "screen_pass_rate", "rpo_rate", "offense_backfield_mean",
    "personnel_11_rate", "personnel_12_rate", "personnel_21_rate", "personnel_other_rate",
]


def load_oc_assignments():
    df = pd.read_csv(ASSIGNMENTS_CSV)
    df["first_year_in_seat"] = df["first_year_in_seat"].astype(bool)
    df["hc_is_playcaller"] = df["hc_is_playcaller"].astype(bool)
    return df


def _oc_prior_stop_profile(oc_name, season, assignments, team_profiles):
    """Most recent season strictly before `season` where this oc_name
    appears anywhere in the assignments table (any team) - i.e. their own
    playcalling track record, wherever it was."""
    prior_rows = assignments[(assignments["oc_name"] == oc_name) & (assignments["season"] < season)]
    if prior_rows.empty:
        return None
    last = prior_rows.sort_values("season").iloc[-1]
    match = team_profiles[(team_profiles["season"] == last["season"]) & (team_profiles["team"] == last["team"])]
    return match.iloc[0] if not match.empty else None


def build_oc_tendency_profiles(conn=None):
    own_conn = conn is None
    conn = conn or get_conn()

    team_profiles = compute_team_season_tendencies(conn)
    assignments = load_oc_assignments()

    df = assignments.merge(team_profiles, on=["season", "team"], how="left")

    for m in METRICS:
        df[f"inherited_{m}"] = pd.NA
    df["inheritance_basis"] = None  # 'blend' | 'team_only_no_oc_history' | None (not first-year)

    for idx, row in df[df["first_year_in_seat"]].iterrows():
        team_prior = team_profiles[
            (team_profiles["season"] == row["season"] - 1) & (team_profiles["team"] == row["team"])
        ]
        team_prior = team_prior.iloc[0] if not team_prior.empty else None
        oc_prior = _oc_prior_stop_profile(row["oc_name"], row["season"], assignments, team_profiles)

        weights = INHERITANCE_WEIGHTS.get(row["promotion_type"])
        if team_prior is None and oc_prior is None:
            continue  # no history at all (e.g. team's first tracked season) - leave inherited_* null
        if oc_prior is None or weights is None:
            # no OC track record to blend in (rookie/college hire, or a
            # 'returning'-type promotion_type shouldn't normally hit
            # first_year_in_seat=True, but guard anyway) -> team inertia only
            for m in METRICS:
                df.at[idx, f"inherited_{m}"] = team_prior[m] if team_prior is not None else pd.NA
            df.at[idx, "inheritance_basis"] = "team_only_no_oc_history"
        elif team_prior is None:
            for m in METRICS:
                df.at[idx, f"inherited_{m}"] = oc_prior[m]
            df.at[idx, "inheritance_basis"] = "oc_only_no_team_history"
        else:
            for m in METRICS:
                a, b = team_prior[m], oc_prior[m]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a):
                    df.at[idx, f"inherited_{m}"] = b
                elif pd.isna(b):
                    df.at[idx, f"inherited_{m}"] = a
                else:
                    df.at[idx, f"inherited_{m}"] = weights["team"] * a + weights["oc"] * b
            df.at[idx, "inheritance_basis"] = "blend"

    if own_conn:
        conn.close()
    return df.sort_values(["team", "season"]).reset_index(drop=True)


if __name__ == "__main__":
    conn = get_conn()
    df = build_oc_tendency_profiles(conn)
    df.to_sql("oc_tendency_profiles", conn, if_exists="replace", index=False)
    conn.close()
    pd.set_option("display.width", 240)
    print(df.shape)
    cols = ["season", "team", "oc_name", "first_year_in_seat", "promotion_type",
            "inheritance_basis", "pass_oe", "inherited_pass_oe", "neutral_sec_per_play", "inherited_neutral_sec_per_play"]
    print(df[df.team == "LA"][cols].to_string(index=False))
