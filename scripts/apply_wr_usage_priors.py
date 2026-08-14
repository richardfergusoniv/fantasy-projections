"""One-shot: set researched WR usage_share_prior + usage_share_reviewed on starters_2026.csv."""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.projection.data_prep import get_conn, load_weekly_usage
from src.projection.predict import load_depth_chart

SLOT = [0.1554, 0.0667, 0.0386]
# Usage order (alpha first) where formation order or injury-skewed 2025 misleads.
MANUAL = {
    "ARI": ["Marvin Harrison Jr.", "Michael Wilson", "Kendrick Bourne"],
    "ATL": ["Drake London", "Olamide Zaccheaus", "Jahan Dotson"],
    "BAL": ["Zay Flowers", "Rashod Bateman", "Devontez Walker"],
    "BUF": ["Khalil Shakir", "DJ Moore", "Keon Coleman"],
    "CLE": ["Jerry Jeudy", "Denzel Boston", "KC Concepcion"],
    "DAL": ["CeeDee Lamb", "George Pickens", "Ryan Flournoy"],
    "DEN": ["Jaylen Waddle", "Courtland Sutton", "Marvin Mims Jr."],
    "DET": ["Amon-Ra St. Brown", "Jameson Williams", "Isaac TeSlaa"],
    "GB": ["Christian Watson", "Matthew Golden", "Jayden Reed"],
    "IND": ["Josh Downs", "Alec Pierce", "Ashton Dulin"],
    "JAX": ["Jakobi Meyers", "Parker Washington", "Brian Thomas Jr."],
    "KC": ["Rashee Rice", "Xavier Worthy", "Tyquan Thornton"],
    "LA": ["Puka Nacua", "Davante Adams", "Jordan Whittington"],
    "LAC": ["Ladd McConkey", "Quentin Johnston", "Tre Harris"],
    "MIA": ["Malik Washington", "Jalen Tolbert", "Caleb Douglas"],
    "MIN": ["Justin Jefferson", "Jauan Jennings", "Jordan Addison"],
    "NO": ["Chris Olave", "Jordyn Tyson", "Devaughn Vele"],
    "NYG": ["Calvin Austin III", "Darius Slayton", "Malik Nabers"],
    "NYJ": ["Garrett Wilson", "Adonai Mitchell", "Omar Cooper Jr."],
    "PIT": ["DK Metcalf", "Michael Pittman", "Roman Wilson"],
    "SF": ["Mike Evans", "Deebo Samuel Sr.", "De'Zhaun Stribling"],
    "TB": ["Emeka Egbuka", "Chris Godwin Jr.", "Jalen McMillan"],
    "TEN": ["Calvin Ridley", "Carnell Tate", "Wan'Dale Robinson"],
    "WAS": ["Stefon Diggs", "Terry McLaurin", "Luke McCaffrey"],
}


def main():
    conn = get_conn()
    try:
        wr = load_depth_chart(2026)
        wr = wr[wr["position"] == "WR"].copy()
        u = load_weekly_usage(conn)
        u25 = u[(u["season"] == 2025) & (u["position"] == "WR")].groupby(
            "player_id", as_index=False
        )["targets"].sum()
        tgt = dict(zip(u25["player_id"], u25["targets"]))
    finally:
        conn.close()

    priors = {}
    for team, g in wr.groupby("team"):
        g = g.sort_values("depth_rank")
        if team in MANUAL:
            ordered = []
            for name in MANUAL[team]:
                match = g[g["player_name"] == name]
                if len(match) != 1:
                    raise SystemExit(f"manual name miss {team} {name!r}")
                ordered.append(match.iloc[0])
        else:
            scored = [
                (-(tgt.get(r.gsis_id, 0)), int(r.depth_rank), r)
                for _, r in g.iterrows()
            ]
            ordered = [r for _, __, r in sorted(scored)]
        for i, r in enumerate(ordered):
            priors[r.gsis_id] = SLOT[i]

    path = os.path.join(REPO, "src", "depth_chart", "starters_2026.csv")
    full = pd.read_csv(path)
    wr_mask = full["position"].eq("WR") & full["gsis_id"].isin(priors)
    full.loc[wr_mask, "usage_share_prior"] = full.loc[wr_mask, "gsis_id"].map(priors)
    full.loc[wr_mask, "usage_share_reviewed"] = True
    full.to_csv(path, index=False)
    n = int(wr_mask.sum())
    print(f"Reviewed {n} WR rows across {full.loc[wr_mask, 'team'].nunique()} teams")


if __name__ == "__main__":
    main()
