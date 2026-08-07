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
            "role": r["role"],
            "ours_fpts": round(r["fantasy_pts"], 2),
            "sleeper_fpts": round(r["sleeper_fantasy_pts"], 2),
            "delta": round(r["fantasy_pts_delta"], 2),
            "ours_rec_ypg": round(r["pg_receiving_yards"], 1) if pd.notna(r["pg_receiving_yards"]) else None,
            "sleeper_rec_ypg": round(r["sleeper_receiving_yards"], 1) if pd.notna(r["sleeper_receiving_yards"]) else None,
            "why": why,
        })
    return pd.DataFrame(rows)


def position_bias(df):
    rows = []
    for pos, n in POSITION_TOP_N.items():
        sub = df[df["position"] == pos].nlargest(n, "sleeper_fantasy_pts")
        rows.append({
            "position": pos,
            "n": len(sub),
            "mean_delta": round(sub["fantasy_pts_delta"].mean(), 2),
            "median_delta": round(sub["fantasy_pts_delta"].median(), 2),
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
    print(f"\n=== Position bias, fantasy-relevant players (ours - Sleeper, fpts/g) ===")
    print(position_bias(df).to_string(index=False))

    missing = [r["player"] for _, r in pd.concat([watch, controls]).iterrows() if r.get("MISSING")]
    if missing:
        print(f"\nWARNING: watchlist players MISSING from comparison output: {missing}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
