"""Named-player spot check against the Sleeper comparison output.

Aggregate metrics (MAE tables, correlations, bias means) have repeatedly
looked clean while specific well-known players were silently wrong or
missing entirely (the Phase 5 rookie-filter bug, the Phase 6 team-change
bugs). This script is the standing countermeasure: a fixed watchlist of
players every pipeline change must be checked against BY NAME, plus
controls that must NOT move.

Usage: `python -m src.comparison.spot_check --season 2026`
"""
import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

# (name-in-output, why it's watched). Grouping matters for reading the
# report: the first block should move toward Sleeper as the WR-gap phases
# land; the controls block must stay put.
WATCHLIST = [
    # Injury-shortened 2025 elite WRs (the core consensus-gap cohort)
    ("Malik Nabers", "injury 4g"),
    ("Jayden Reed", "injury 5g"),
    ("Garrett Wilson", "injury 7g"),
    ("Mike Evans", "injury 8g + team change"),
    ("Chris Godwin Jr.", "injury 9g (2nd straight short season)"),
    ("Terry McLaurin", "injury 10g"),
    ("Christian Watson", "injury 10g"),
    # Depth-chart curation gaps
    ("Parker Washington", "curation: was uncurated deep_bench"),
    ("Wan'Dale Robinson", "curation: was uncurated deep_bench"),
    # Sophomore gap
    ("Luther Burden III", "sophomore"),
    ("Matthew Golden", "sophomore"),
    ("Jayden Higgins", "sophomore"),
]
CONTROLS = [
    ("Rashee Rice", "8g but SUSPENSION not injury - must not inflate"),
    ("Ja'Marr Chase", "healthy alpha - must stay put"),
    ("Justin Jefferson", "healthy alpha - must stay put"),
]

# "Fantasy-relevant" cutoffs for the per-position bias summary: roughly
# two starters per league slot in a 12-team league.
POSITION_TOP_N = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}


def load_comparison(season):
    path = os.path.join(OUTPUT_DIR, f"sleeper_comparison_{season}.csv")
    df = pd.read_csv(path)
    return df[df["matched_sleeper"] == True].copy()  # noqa: E712


def spot_check_table(df, entries):
    rows = []
    for name, why in entries:
        match = df[df["display_name"] == name]
        if match.empty:
            rows.append({"player": name, "why": why, "MISSING": True})
            continue
        r = match.iloc[0]
        rows.append({
            "player": name,
            "team": r["team"],
            "sleeper_team": r.get("sleeper_team"),
            "sleeper_name": r.get("sleeper_name"),
            "sleeper_id": r.get("sleeper_id"),
            "match_method": r.get("match_method"),
            "match_collision": r.get("match_collision", False),
            "role": r["role"],
            "ours_fpts_season": round(r["fantasy_pts_season"], 1),
            "sleeper_fpts_season": round(r["sleeper_fantasy_pts_season"], 1),
            "season_delta": round(r["fantasy_pts_season_delta"], 1),
            "ours_rec_yards": round(r["our_receiving_yards_season"], 0)
            if pd.notna(r.get("our_receiving_yards_season")) else None,
            "sleeper_rec_yards": round(r["sleeper_receiving_yards_season"], 0)
            if pd.notna(r.get("sleeper_receiving_yards_season")) else None,
            "why": why,
        })
    return pd.DataFrame(rows)


def position_bias(df):
    rows = []
    for pos, n in POSITION_TOP_N.items():
        sub = df[df["position"] == pos].nlargest(n, "sleeper_fantasy_pts_season")
        rows.append({
            "position": pos,
            "n": len(sub),
            "mean_season_delta": round(sub["fantasy_pts_season_delta"].mean(), 1),
            "median_season_delta": round(sub["fantasy_pts_season_delta"].median(), 1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    df = load_comparison(args.season)

    watch = spot_check_table(df, WATCHLIST)
    controls = spot_check_table(df, CONTROLS)

    print(f"=== Spot check, season {args.season}: watchlist (should converge toward Sleeper) ===")
    print(watch.to_string(index=False))
    print(f"\n=== Controls (must NOT move materially) ===")
    print(controls.to_string(index=False))
    print(f"\n=== Position bias, fantasy-relevant players (ours - Sleeper, season points) ===")
    print(position_bias(df).to_string(index=False))
    if "match_method" in df.columns:
        print("\n=== Sleeper match audit ===")
        print(df["match_method"].value_counts(dropna=False).to_string())
        collisions = int(df.get("match_collision", pd.Series(False, index=df.index)).fillna(False).sum())
        print(f"ambiguous name collisions left unmatched: {collisions}")

    missing = [r["player"] for _, r in pd.concat([watch, controls]).iterrows() if r.get("MISSING")]
    if missing:
        print(f"\nWARNING: watchlist players MISSING from comparison output: {missing}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
