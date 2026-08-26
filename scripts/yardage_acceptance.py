"""Board-level acceptance and named-player smoke report for yardage repair."""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._history import connect, realized_weekly  # noqa: E402


NAMES = (
    "Amon-Ra St. Brown", "CeeDee Lamb", "Jonathan Taylor", "Saquon Barkley"
)


def _dominant_teams(seasons: list[int]) -> pd.DataFrame:
    conn = connect()
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
        WITH counts AS (
          SELECT season, player_id, recent_team AS team, COUNT(*) AS n
          FROM weekly
          WHERE season IN ({placeholders}) AND recent_team IS NOT NULL
          GROUP BY season, player_id, recent_team
        ), ranked AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY season, player_id ORDER BY n DESC, team
          ) AS rn FROM counts
        )
        SELECT season, player_id, team FROM ranked WHERE rn = 1
    """
    out = pd.read_sql(query, conn, params=seasons)
    conn.close()
    return out


def _room_metrics(frame: pd.DataFrame, *, season_col: str, value_col: str) -> dict:
    out = {}
    for key, position, stat in (
        ("wr1_receiving_yard_share", "WR", "receiving_yards"),
        ("rb1_rushing_yard_share", "RB", "rushing_yards"),
    ):
        rows = frame[frame["stat"].eq(stat)].copy()
        denom = rows.groupby([season_col, "team"], observed=True)[value_col].sum()
        ranked = rows[rows["position"].eq(position)].copy()
        ranked["rank"] = ranked.groupby([season_col, "team"], observed=True)[value_col].rank(
            method="first", ascending=False
        )
        top = ranked[ranked["rank"].eq(1)].copy()
        top["share"] = top[value_col] / pd.MultiIndex.from_frame(
            top[[season_col, "team"]]
        ).map(denom)
        out[key] = float(top["share"].mean())

    targets = frame[
        frame["stat"].eq("targets") & frame["position"].isin(["RB", "WR", "TE"])
    ].copy()
    targets["rank"] = targets.groupby(
        [season_col, "team", "position"], observed=True
    )[value_col].rank(method="first", ascending=False)
    tail = targets[targets["rank"].ge(6)][value_col].sum()
    out["rank6_plus_target_share"] = float(tail / targets[value_col].sum())
    return out


def build_report(projection_path: str, seasons: list[int]) -> dict:
    long = pd.read_csv(projection_path)
    wide = long.pivot_table(
        index=["player_id", "display_name", "team", "position"],
        columns="stat", values="pred_season", aggfunc="first",
    ).reset_index()
    qb_count = int(((wide["position"] == "QB") & (wide["passing_yards"] > 3000)).sum())
    rb_count = int(((wide["position"] == "RB") & (wide["rushing_yards"] > 1000)).sum())

    stats = ["passing_yards", "rushing_yards", "receiving_yards", "targets"]
    hist = realized_weekly(seasons, stats, ("QB", "RB", "WR", "TE"))
    hist = hist.merge(_dominant_teams(seasons), on=["season", "player_id"], how="inner")
    hist_long = hist.melt(
        id_vars=["season", "player_id", "position", "team"],
        value_vars=stats, var_name="stat", value_name="actual",
    )
    projected_totals = long.groupby("stat")["pred_season"].sum()
    historical_totals = hist_long.groupby(["season", "stat"])["actual"].sum().groupby("stat").mean()
    league_ratios = {
        stat: float(projected_totals[stat] / historical_totals[stat])
        for stat in ("passing_yards", "receiving_yards", "rushing_yards")
    }

    projected_room = _room_metrics(
        long.assign(board_season=int(long["season"].iloc[0])),
        season_col="board_season", value_col="pred_season",
    )
    historical_room = _room_metrics(hist_long, season_col="season", value_col="actual")
    room = {
        key: {
            "projected": projected_room[key],
            "historical": historical_room[key],
            "ratio": projected_room[key] / historical_room[key],
        }
        for key in projected_room
    }

    identities = {}
    for child, parent, ratio in (
        ("receiving_yards", "passing_yards", 1.0),
        ("receptions", "completions", 1.0),
        ("targets", "attempts", 0.952),
        ("receiving_tds", "passing_tds", 1.0),
    ):
        team = long[long["stat"].isin([child, parent])].pivot_table(
            index="team", columns="stat", values="pred_season", aggfunc="sum"
        )
        deviation = (team[child] / team[parent] - ratio).abs().max()
        identities[f"{child}_to_{parent}"] = float(deviation)

    smoke = {}
    for name in NAMES:
        row = wide[wide["display_name"].eq(name)]
        smoke[name] = (
            {
                "team": str(row.iloc[0]["team"]),
                "position": str(row.iloc[0]["position"]),
                "passing_yards": _number(row.iloc[0].get("passing_yards")),
                "rushing_yards": _number(row.iloc[0].get("rushing_yards")),
                "receiving_yards": _number(row.iloc[0].get("receiving_yards")),
            }
            if not row.empty else {"missing": True}
        )

    checks = {
        "qb_3000_count": 18 <= qb_count <= 24,
        "rb_1000_count": 12 <= rb_count <= 18,
        "league_yardage_within_5pct": all(0.95 <= value <= 1.05 for value in league_ratios.values()),
        "team_identities": all(value < 1e-9 for value in identities.values()),
        "rank6_tail_below_prior_2_2x": room["rank6_plus_target_share"]["ratio"] < 2.2,
    }
    return {
        "projection_path": os.path.abspath(projection_path),
        "history_seasons": seasons,
        "threshold_counts": {"qbs_above_3000": qb_count, "rbs_above_1000": rb_count},
        "league_yardage_ratios": league_ratios,
        "allocation": room,
        "identity_max_abs_deviation": identities,
        "named_smoke": smoke,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _number(value):
    return None if pd.isna(value) else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--history", default="2023,2024,2025")
    parser.add_argument("--projection-path", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    projection_path = args.projection_path or os.path.join(
        "output", f"projections_{args.season}.csv"
    )
    report = build_report(projection_path, [int(s) for s in args.history.split(",")])
    out = args.out or os.path.join("output", f"yardage_acceptance_{args.season}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
