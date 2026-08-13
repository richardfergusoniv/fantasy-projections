"""Compare our season-total projections against Sleeper's free, public,
no-auth projections API - the only free service found with a clean,
bulk-fetchable, per-player structured projection endpoint (FantasyPros/ESPN
don't expose one; their numbers are only available via their own web UI,
not a scriptable free API, so a broad automated comparison against them
isn't practical the same way).

Sleeper endpoints used:
  https://api.sleeper.app/v1/players/nfl                    - player master
      (includes gsis_id directly - trivial join key onto our own data)
  https://api.sleeper.app/v1/projections/nfl/regular/<year>  - full-season
      TOTALS per player (not per-week). Sleeper's `gp` is retained for
      diagnostics, but is not assumed to be a player-level games forecast:
      in the 2026 feed it is an almost-universal bookkeeping value of 18.

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
    response = requests.get(PLAYERS_URL, timeout=60)
    response.raise_for_status()
    players = response.json()
    if not isinstance(players, dict):
        raise ValueError("Sleeper players response was not a player-id mapping")
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
    """Return Sleeper's season totals without inventing a rate denominator.

    `gp` is carried as `reported_gp` for auditability.  Conditional-rate
    columns are populated only when the feed's denominator is credible:
    positive, no larger than the NFL schedule, and not an almost-universal
    constant across the player pool.  This deliberately leaves the 2026
    rate columns null; 18 is the number of regular-season *weeks*, not a
    player-specific projection of games played.
    """
    response = requests.get(SEASON_PROJ_URL.format(season=season), timeout=60)
    response.raise_for_status()
    proj = response.json()
    if not isinstance(proj, dict):
        raise ValueError("Sleeper projections response was not a player-id mapping")
    rows = []
    for sid, stats in proj.items():
        gp = stats.get("gp")
        # A projection can have season totals even when no usable `gp`
        # denominator exists. Do not drop that valid season-level signal.
        if not isinstance(stats, dict):
            continue
        row = {
            "sleeper_id": sid,
            "reported_gp": gp,
            "pts_half_ppr_season": stats.get("pts_half_ppr", 0),
        }
        for sleeper_field, our_stat in STAT_MAP.items():
            row[f"{our_stat}_season"] = stats.get(sleeper_field, 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    gp = pd.to_numeric(out["reported_gp"], errors="coerce")
    valid_numeric = gp.gt(0) & gp.le(17)
    non_null = gp.dropna()
    dominant_share = non_null.value_counts(normalize=True).iloc[0] if len(non_null) else 1.0
    feed_has_player_specific_gp = dominant_share < 0.95
    out["rate_denominator_valid"] = valid_numeric & feed_has_player_specific_gp

    out["pts_half_ppr_pg"] = float("nan")
    for our_stat in STAT_MAP.values():
        out[our_stat] = float("nan")
    valid = out["rate_denominator_valid"]
    if valid.any():
        out.loc[valid, "pts_half_ppr_pg"] = (
            out.loc[valid, "pts_half_ppr_season"] / gp[valid]
        )
        for our_stat in STAT_MAP.values():
            out.loc[valid, our_stat] = out.loc[valid, f"{our_stat}_season"] / gp[valid]
    return out


NO_STATS_PLAY_PROB = 0.05
HAS_STATS_PLAY_PROB = 1.0


def fetch_sleeper_play_probability(season):
    """player_id (gsis, where resolvable) + name_key/position -> play_prob.

    Data-quality finding that changed this function's design: Sleeper's
    `gp` (games-played) field is NOT a real per-player play-probability
    signal - checked the distribution directly, 9370 of 9402 players with a
    `gp` value have EXACTLY gp=18 (only 32 have gp=1), including players
    with zero other projected stats at all (e.g. rookie QBs Athan
    Kaliakmanis / Mark Gronowski have `gp=18` but no `pass_att`,
    `pts_half_ppr`, or any other field - Sleeper is not projecting them to
    play a full season, it's just a bookkeeping default for anyone Sleeper
    tracks for ADP purposes). The real signal is whether Sleeper bothers
    projecting any actual production for the player at all - that presence/
    absence is lost by fetch_sleeper_season_projections (which fills
    missing stat fields with 0), so this function re-reads the raw
    projections JSON directly rather than reusing that helper.

    play_prob is binary, not a smooth ratio: HAS_STATS_PLAY_PROB (1.0) if
    Sleeper projects real pass-attempt volume for this player, else
    NO_STATS_PLAY_PROB (0.05, same reasoning/value as rookies.py's
    NO_SLEEPER_MATCH_PLAY_PROB for a player absent from Sleeper entirely -
    "Sleeper won't even project a number" is roughly as strong a signal
    either way)."""
    players = fetch_sleeper_players()
    proj = requests.get(SEASON_PROJ_URL.format(season=season), timeout=60).json()
    rows = []
    for sid, stats in proj.items():
        has_stats = "pass_att" in stats
        rows.append({"sleeper_id": sid, "play_prob": HAS_STATS_PLAY_PROB if has_stats else NO_STATS_PLAY_PROB})
    proj_df = pd.DataFrame(rows)
    merged = players.merge(proj_df, on="sleeper_id", how="inner")
    return merged[["player_id", "name_key", "position", "play_prob"]]


def build_sleeper_comparison_table(season):
    players = fetch_sleeper_players()
    season_proj = fetch_sleeper_season_projections(season)
    sleeper = players.merge(season_proj, on="sleeper_id", how="inner")
    return sleeper


def compare(our_fantasy_points_path, season):
    ours = pd.read_csv(our_fantasy_points_path)
    if "fantasy_pts_season" not in ours.columns:
        if {"fantasy_pts", "projected_games"}.issubset(ours.columns):
            exposure = (
                ours["projected_volume_games"].fillna(ours["projected_games"])
                if "projected_volume_games" in ours.columns else ours["projected_games"]
            )
            ours["fantasy_pts_season"] = ours["fantasy_pts"] * exposure
        else:
            raise ValueError(
                "Season-total comparison requires fantasy_pts_season or both "
                "fantasy_pts and projected_games"
            )
    ours["name_key"] = ours["display_name"].apply(_normalize_name)
    sleeper = build_sleeper_comparison_table(season)

    stat_cols = list(STAT_MAP.values())
    season_stat_cols = [f"{c}_season" for c in stat_cols]
    rename = {c: f"sleeper_{c}" for c in stat_cols}
    rename.update({c: f"sleeper_{c}" for c in season_stat_cols})
    rename.update({
        "pts_half_ppr_pg": "sleeper_fantasy_pts",
        "pts_half_ppr_season": "sleeper_fantasy_pts_season",
        "reported_gp": "sleeper_gp",
        "rate_denominator_valid": "sleeper_rate_denominator_valid",
    })
    sleeper_stats = sleeper.rename(columns=rename)
    sleeper_cols = [
        "sleeper_fantasy_pts_season", "sleeper_fantasy_pts", "sleeper_gp",
        "sleeper_rate_denominator_valid",
    ] + [f"sleeper_{c}_season" for c in stat_cols] + [f"sleeper_{c}" for c in stat_cols]

    # Tier 1: join on gsis_id (player_id) - exact, unambiguous, preferred.
    by_id = sleeper_stats.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    merged = ours.merge(by_id[["player_id", "position"] + sleeper_cols], on=["player_id", "position"], how="left")
    matched_by_id = merged["sleeper_fantasy_pts_season"].notna()

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
    matched_by_name = (~matched_by_id) & merged["sleeper_fantasy_pts_season"].notna()
    merged.loc[matched_by_name, "match_method"] = "name"

    merged["fantasy_pts_season_delta"] = (
        merged["fantasy_pts_season"] - merged["sleeper_fantasy_pts_season"]
    )
    # Raw-stat season totals use the same availability decomposition as
    # fantasy_pts_season. Keep explicit `our_`/`sleeper_` names and deltas
    # so the proxy can diagnose *which* component is misweighted without
    # falling back to the invalid gp=18 rate conversion.
    exposure = (
        merged["projected_volume_games"].fillna(merged["projected_games"])
        if "projected_volume_games" in merged.columns else merged["projected_games"]
    )
    for stat in stat_cols:
        pg_col = f"pg_{stat}"
        our_total = f"our_{stat}_season"
        sleeper_total = f"sleeper_{stat}_season"
        delta = f"{stat}_season_delta"
        if pg_col in merged.columns:
            merged[our_total] = merged[pg_col] * exposure
            merged[delta] = merged[our_total] - merged[sleeper_total]
    # Backward-compatible conditional-rate delta, but only when Sleeper has
    # a valid player-specific denominator. In the current feed this is NaN,
    # preventing a season-total/18 number from masquerading as a comparable
    # conditional rate.
    merged["fantasy_pts_delta"] = merged["fantasy_pts"] - merged["sleeper_fantasy_pts"]
    invalid_rate = ~merged["sleeper_rate_denominator_valid"].fillna(False).astype(bool)
    merged.loc[invalid_rate, "fantasy_pts_delta"] = float("nan")
    merged["matched_sleeper"] = merged["sleeper_fantasy_pts_season"].notna()
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
    print(f"\nSeason totals: our mean={matched['fantasy_pts_season'].mean():.2f}, "
          f"Sleeper mean={matched['sleeper_fantasy_pts_season'].mean():.2f}")
    print(f"Correlation (matched only): "
          f"{matched['fantasy_pts_season'].corr(matched['sleeper_fantasy_pts_season']):.3f}")
    print(f"Mean absolute season-total delta (matched only): "
          f"{matched['fantasy_pts_season_delta'].abs().mean():.2f}")
    valid_rates = matched[matched["sleeper_rate_denominator_valid"].fillna(False).astype(bool)]
    if valid_rates.empty:
        print("Conditional-rate comparison: unavailable (Sleeper did not provide a credible player-level games denominator)")
    else:
        print(f"Conditional-rate correlation (n={len(valid_rates)}): "
              f"{valid_rates['fantasy_pts'].corr(valid_rates['sleeper_fantasy_pts']):.3f}")
    for pos in ["QB", "RB", "WR", "TE"]:
        p = matched[matched.position == pos]
        if len(p):
            print(f"  {pos}: n={len(p)}, "
                  f"season_corr={p['fantasy_pts_season'].corr(p['sleeper_fantasy_pts_season']):.3f}, "
                  f"season_mean_abs_delta={p['fantasy_pts_season_delta'].abs().mean():.2f}, "
                  f"our_season_mean={p['fantasy_pts_season'].mean():.2f}, "
                  f"sleeper_season_mean={p['sleeper_fantasy_pts_season'].mean():.2f}")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
