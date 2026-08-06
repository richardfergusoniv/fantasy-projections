"""Compare our per-game projections against Sleeper's free, public,
no-auth projections API - the only free service found with a clean,
bulk-fetchable, per-player structured projection endpoint (FantasyPros/ESPN
don't expose one; their numbers are only available via their own web UI,
not a scriptable free API, so a broad automated comparison against them
isn't practical the same way).

Sleeper endpoints used:
  https://api.sleeper.app/v1/players/nfl                    - player master
      (includes gsis_id directly - trivial join key onto our own data)
  https://api.sleeper.app/v1/projections/nfl/regular/<year>  - full-season
      TOTALS per player (not per-week), including `gp` (games played) -
      dividing by gp gives a per-game rate directly comparable to ours.

Usage: `python -m src.comparison.sleeper_compare --season 2026`
"""
import argparse
import os

import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SEASON_PROJ_URL = "https://api.sleeper.app/v1/projections/nfl/regular/{season}"

# Sleeper's season-total field -> our stat name, so both sides can be
# divided by games-played and compared per-game, apples to apples.
STAT_MAP = {
    "pass_att": "attempts",
    "pass_cmp": "completions",
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "interceptions",
    "rush_att": "carries",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rec": "receptions",
    "rec_tgt": "targets",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
}


def _normalize_name(name):
    if not isinstance(name, str):
        return None
    name = name.lower().replace(".", "").replace("'", "")
    for suffix in (" jr", " sr", " ii", " iii", " iv", " v"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def fetch_sleeper_players():
    """Full Sleeper player master, one row per sleeper_id: gsis_id (often
    null - see below), full_name, team, position.

    Data-quality gap found while building this comparison: Sleeper's own
    gsis_id field is null for a surprising number of clearly-fantasy-
    relevant players - spot-checked Ja'Marr Chase, Bijan Robinson, De'Von
    Achane, all null, despite having sportradar_id/rotowire_id populated.
    Joining on gsis_id alone matched only ~14% of our players - too low to
    be a useful comparison. `build_join_key()` below adds a normalized-name
    fallback for exactly this gap."""
    players = requests.get(PLAYERS_URL, timeout=60).json()
    rows = []
    for sid, p in players.items():
        rows.append({
            "sleeper_id": sid,
            "player_id": p.get("gsis_id"),
            "position": p.get("position"),
            "team": p.get("team"),
            "name_key": _normalize_name(p.get("full_name")),
        })
    return pd.DataFrame(rows)


def fetch_sleeper_season_projections(season):
    """sleeper_player_id -> season-total projected stats + gp (games
    played). Returns per-game rates (total / gp), not raw totals - that's
    the comparable unit our own pred_pg is in."""
    proj = requests.get(SEASON_PROJ_URL.format(season=season), timeout=60).json()
    rows = []
    for sid, stats in proj.items():
        gp = stats.get("gp")
        if not gp:  # no games-played projection at all -> not a real season projection for this player
            continue
        row = {"sleeper_id": sid, "gp": gp, "pts_half_ppr_pg": stats.get("pts_half_ppr", 0) / gp}
        for sleeper_field, our_stat in STAT_MAP.items():
            row[our_stat] = stats.get(sleeper_field, 0) / gp
        rows.append(row)
    return pd.DataFrame(rows)


def build_sleeper_comparison_table(season):
    players = fetch_sleeper_players()
    season_proj = fetch_sleeper_season_projections(season)
    sleeper = players.merge(season_proj, on="sleeper_id", how="inner")
    return sleeper


def compare(our_fantasy_points_path, season):
    ours = pd.read_csv(our_fantasy_points_path)
    ours["name_key"] = ours["display_name"].apply(_normalize_name)
    sleeper = build_sleeper_comparison_table(season)

    stat_cols = list(STAT_MAP.values())
    rename = {c: f"sleeper_{c}" for c in stat_cols}
    rename.update({"pts_half_ppr_pg": "sleeper_fantasy_pts", "gp": "sleeper_gp"})
    sleeper_stats = sleeper.rename(columns=rename)
    sleeper_cols = ["sleeper_fantasy_pts", "sleeper_gp"] + [f"sleeper_{c}" for c in stat_cols]

    # Tier 1: join on gsis_id (player_id) - exact, unambiguous, preferred.
    by_id = sleeper_stats.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    merged = ours.merge(by_id[["player_id", "position"] + sleeper_cols], on=["player_id", "position"], how="left")
    matched_by_id = merged["sleeper_fantasy_pts"].notna()

    # Tier 2: for rows gsis_id couldn't resolve, fall back to normalized
    # name + position - the exact gap load_players() found (many real
    # players have a null gsis_id in Sleeper's own data).
    by_name = sleeper_stats.dropna(subset=["name_key"]).drop_duplicates(subset=["name_key", "position"])
    fallback = merged.loc[~matched_by_id, ["name_key", "position"]].merge(
        by_name[["name_key", "position"] + sleeper_cols], on=["name_key", "position"], how="left",
    )
    for c in sleeper_cols:
        merged.loc[~matched_by_id, c] = fallback[c].values

    merged["match_method"] = "unmatched"
    merged.loc[matched_by_id, "match_method"] = "gsis_id"
    matched_by_name = (~matched_by_id) & merged["sleeper_fantasy_pts"].notna()
    merged.loc[matched_by_name, "match_method"] = "name"

    merged["fantasy_pts_delta"] = merged["fantasy_pts"] - merged["sleeper_fantasy_pts"]
    merged["matched_sleeper"] = merged["sleeper_fantasy_pts"].notna()
    return merged.drop(columns=["name_key"]).sort_values("fantasy_pts", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in-path", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = args.in_path or os.path.join(OUTPUT_DIR, f"fantasy_points_{args.season}.csv")
    out_path = args.out or os.path.join(OUTPUT_DIR, f"sleeper_comparison_{args.season}.csv")

    merged = compare(in_path, args.season)
    merged.to_csv(out_path, index=False)

    matched = merged[merged["matched_sleeper"]]
    print(f"{len(merged)} of our rows, {len(matched)} matched to Sleeper "
          f"({len(matched) / len(merged):.0%}) - "
          f"{(merged.match_method == 'gsis_id').sum()} by gsis_id, "
          f"{(merged.match_method == 'name').sum()} by name fallback")
    print(f"\nOverall: our mean={merged['fantasy_pts'].mean():.2f}, "
          f"Sleeper mean (matched only)={matched['sleeper_fantasy_pts'].mean():.2f}")
    print(f"Correlation (matched only): {matched['fantasy_pts'].corr(matched['sleeper_fantasy_pts']):.3f}")
    print(f"Mean absolute delta (matched only): {matched['fantasy_pts_delta'].abs().mean():.2f}")
    for pos in ["QB", "RB", "WR", "TE"]:
        p = matched[matched.position == pos]
        if len(p):
            print(f"  {pos}: n={len(p)}, corr={p['fantasy_pts'].corr(p['sleeper_fantasy_pts']):.3f}, "
                  f"mean_abs_delta={p['fantasy_pts_delta'].abs().mean():.2f}, "
                  f"our_mean={p['fantasy_pts'].mean():.2f}, sleeper_mean={p['sleeper_fantasy_pts'].mean():.2f}")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
