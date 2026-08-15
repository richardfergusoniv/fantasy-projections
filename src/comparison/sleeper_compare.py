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

WHAT THIS IS AND IS NOT (framing fixed 2026-08-15)
--------------------------------------------------
This module is a **read-only diagnostic**. It has earned its place: comparing
against Sleeper genuinely surfaced the share-denominator bug, the
trade-vacancy bug and the Diggs/Okonkwo/White triple-boost. Keep running it.

It is **not** an objective, and no number it prints is an acceptance
criterion. Sleeper projects a full slate -- `gp` is 18 for ~9,370 of the
9,402 players it tracks -- and allocates ~96.8% of team carries to named
players against our ~83.8%. This system projects EXPECTED VALUE, including
the probability a player does not play. Two differently-framed forecasts
disagreeing is the expected result, not a defect, so **agreement is not
accuracy** and a shrinking divergence is not evidence of an improvement.

Consequently the summary columns here are named `divergence`, never `mae`,
`error` or `bias`. To decide whether a change helps, score it on
`src/projection/fantasy_evaluation.py`, which is gated on held-out actual
fantasy outcomes. To gate the board itself, use
`src.comparison.spot_check`, which fails on incoherence and never on
disagreement.

Usage: `python -m src.comparison.sleeper_compare --season 2026`
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "sleeper_snapshots")

DIAGNOSTIC_BANNER = (
    "=== DIAGNOSTIC ONLY: divergence from Sleeper is not an error ===\n"
    "Sleeper projects full slates (gp=18 for ~9,370 of 9,402 tracked players);\n"
    "this system projects expected value. Nothing below is an acceptance\n"
    "criterion, a loss, or a gap to close. Use it to find a player whose\n"
    "difference you cannot explain -- then explain it.\n"
)

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


def _fetch_json(url, snapshot_label, snapshot_dir=SNAPSHOT_DIR):
    """Fetch validated JSON and persist a content-addressed audit snapshot."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()
    os.makedirs(snapshot_dir, exist_ok=True)
    stem = f"{snapshot_label}_{digest[:16]}"
    raw_path = os.path.join(snapshot_dir, f"{stem}.json")
    metadata_path = os.path.join(snapshot_dir, f"{stem}.metadata.json")
    if not os.path.exists(raw_path):
        with open(raw_path, "wb") as fh:
            fh.write(raw)
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump({
                "fetched_at": fetched_at,
                "endpoint": url,
                "sha256": digest,
                "raw_path": os.path.abspath(raw_path),
            }, fh, indent=2, sort_keys=True)
    else:
        with open(metadata_path, encoding="utf-8") as fh:
            fetched_at = json.load(fh)["fetched_at"]
    return payload, {
        "fetched_at": fetched_at,
        "endpoint": url,
        "sha256": digest,
        "raw_path": os.path.abspath(raw_path),
        "metadata_path": os.path.abspath(metadata_path),
    }


def fetch_sleeper_players(snapshot_dir=SNAPSHOT_DIR):
    """Full Sleeper player master, one row per sleeper_id: gsis_id (often
    null - see below), full_name, team, position.

    Data-quality gap found while building this comparison: Sleeper's own
    gsis_id field is null for a surprising number of clearly-fantasy-
    relevant players - spot-checked Ja'Marr Chase, Bijan Robinson, De'Von
    Achane, all null, despite having sportradar_id/rotowire_id populated.
    Joining on gsis_id alone matched only ~14% of our players - too low to
    be a useful comparison. `build_join_key()` below adds a normalized-name
    fallback for exactly this gap."""
    players, snapshot = _fetch_json(
        PLAYERS_URL, "players_nfl", snapshot_dir=snapshot_dir)
    if not isinstance(players, dict):
        raise ValueError("Sleeper players response was not a player-id mapping")
    rows = []
    for sid, p in players.items():
        if not isinstance(p, dict):
            continue
        rows.append({
            "sleeper_id": sid,
            "player_id": p.get("gsis_id"),
            "position": p.get("position"),
            "sleeper_team": p.get("team"),
            "sleeper_name": p.get("full_name"),
            "name_key": _normalize_name(p.get("full_name")),
            "players_snapshot_sha256": snapshot["sha256"],
            "players_snapshot_fetched_at": snapshot["fetched_at"],
            "players_snapshot_path": snapshot["raw_path"],
        })
    return pd.DataFrame(rows)


def fetch_sleeper_season_projections(season, snapshot_dir=SNAPSHOT_DIR):
    """Return Sleeper's season totals without inventing a rate denominator.

    `gp` is carried as `reported_gp` for auditability.  Conditional-rate
    columns are populated only when the feed's denominator is credible:
    positive, no larger than the NFL schedule, and not an almost-universal
    constant across the player pool.  This deliberately leaves the 2026
    rate columns null; 18 is the number of regular-season *weeks*, not a
    player-specific projection of games played.
    """
    endpoint = SEASON_PROJ_URL.format(season=season)
    proj, snapshot = _fetch_json(
        endpoint, f"projections_{season}", snapshot_dir=snapshot_dir)
    if not isinstance(proj, dict):
        raise ValueError("Sleeper projections response was not a player-id mapping")
    rows = []
    for sid, stats in proj.items():
        if not isinstance(stats, dict):
            continue
        gp = stats.get("gp")
        # A projection can have season totals even when no usable `gp`
        # denominator exists. Do not drop that valid season-level signal.
        row = {
            "sleeper_id": sid,
            "reported_gp": gp,
            "pts_half_ppr_season": stats.get("pts_half_ppr", 0),
            "projections_snapshot_sha256": snapshot["sha256"],
            "projections_snapshot_fetched_at": snapshot["fetched_at"],
            "projections_snapshot_path": snapshot["raw_path"],
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


# REMOVED 2026-08-15: `fetch_sleeper_play_probability`, `NO_STATS_PLAY_PROB`
# and `HAS_STATS_PLAY_PROB`. These turned Sleeper's willingness to publish a
# `pass_att` field into a binary multiplier on a rookie QB's entire projection
# line (`f5a5d09`). The consumer in `rookies.py` was removed in `df37452` and
# the function has had zero callers since -- verified repo-wide across `src/`,
# `tests/`, `scripts/`, notebooks and the draft assistant, including a check
# for dynamic dispatch (`getattr`/`importlib`), before deletion. It is deleted
# rather than left dormant because a Sleeper-derived multiplier on our own
# predictions is precisely what this module must never provide; keeping it
# available made re-adoption a one-line import. See SLEEPER_RETIREMENT.md.
# The surviving `sleeper_gp` / `reported_gp` diagnostics record the same
# data-quality finding (gp=18 for ~9,370 of 9,402 tracked players) without
# exposing it as a knob.


def build_sleeper_comparison_table(season, snapshot_dir=SNAPSHOT_DIR):
    players = fetch_sleeper_players(snapshot_dir=snapshot_dir)
    season_proj = fetch_sleeper_season_projections(
        season, snapshot_dir=snapshot_dir)
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
        "sleeper_id", "sleeper_team", "sleeper_name",
        "sleeper_fantasy_pts_season", "sleeper_fantasy_pts", "sleeper_gp",
        "sleeper_rate_denominator_valid",
        "players_snapshot_sha256", "players_snapshot_fetched_at",
        "players_snapshot_path", "projections_snapshot_sha256",
        "projections_snapshot_fetched_at", "projections_snapshot_path",
    ] + [f"sleeper_{c}_season" for c in stat_cols] + [f"sleeper_{c}" for c in stat_cols]
    # Direct test callers and older cached comparison frames may omit newer
    # audit metadata. Preserve the schema without inventing values.
    for col in sleeper_cols:
        if col not in sleeper_stats.columns:
            sleeper_stats[col] = pd.NA

    # Tier 1: join on gsis_id (player_id) - exact, unambiguous, preferred.
    id_candidates = sleeper_stats.dropna(subset=["player_id"])
    id_counts = id_candidates.groupby(["player_id", "position"]).size().rename("id_candidate_count")
    id_candidates = id_candidates.merge(
        id_counts.reset_index(), on=["player_id", "position"], how="left")
    by_id = id_candidates[id_candidates["id_candidate_count"] == 1]
    merged = ours.merge(
        id_counts.reset_index(), on=["player_id", "position"], how="left")
    merged["id_candidate_count"] = merged["id_candidate_count"].fillna(0).astype(int)
    merged = merged.merge(
        by_id[["player_id", "position"] + sleeper_cols],
        on=["player_id", "position"], how="left")
    matched_by_id = merged["sleeper_id"].notna()

    # Tier 2: for rows gsis_id couldn't resolve, fall back to normalized
    # name + position. A name match is accepted only when the candidate is
    # unique, or when team disambiguates multiple candidates to exactly one.
    # Ambiguity stays unmatched and is explicit in match_collision.
    name_groups = {
        key: grp for key, grp in
        sleeper_stats.dropna(subset=["name_key"]).groupby(["name_key", "position"], dropna=False)
    }
    merged["name_candidate_count"] = 0
    merged["match_collision"] = merged["id_candidate_count"] > 1
    merged["name_team_disambiguated"] = False
    name_eligible = (~matched_by_id) & (merged["id_candidate_count"] <= 1)
    for idx, row in merged.loc[name_eligible].iterrows():
        candidates = name_groups.get((row["name_key"], row["position"]))
        if candidates is None or candidates.empty:
            continue
        merged.at[idx, "name_candidate_count"] = len(candidates)
        chosen = None
        if len(candidates) == 1:
            chosen = candidates.iloc[0]
        elif pd.notna(row.get("team")) and "sleeper_team" in candidates.columns:
            team_candidates = candidates[candidates["sleeper_team"] == row["team"]]
            if len(team_candidates) == 1:
                chosen = team_candidates.iloc[0]
                merged.at[idx, "name_team_disambiguated"] = True
        if chosen is None:
            merged.at[idx, "match_collision"] = True
            continue
        for col in sleeper_cols:
            merged.at[idx, col] = chosen[col]

    merged["match_method"] = "unmatched"
    merged.loc[matched_by_id, "match_method"] = "gsis_id"
    matched_by_name = (~matched_by_id) & merged["sleeper_id"].notna()
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
    merged["matched_sleeper"] = merged["sleeper_id"].notna()
    return merged.drop(columns=["name_key"]).sort_values("fantasy_pts", ascending=False)


def comparison_summary_strata(merged):
    """Describe how the two boards differ, stratified so the zero tail cannot
    dominate the picture.

    Deliberately NOT named `mae`/`bias`/`error`. Those words name a loss, and
    a loss implies a target. Sleeper is not the target (see the module
    docstring): these columns describe the SHAPE of a difference between two
    differently-framed forecasts, and a smaller number here is not a better
    model. Read them to find a player or position whose difference you cannot
    explain, then explain it.
    """
    matched = merged[merged["matched_sleeper"]].copy()
    strata = {
        "all_matched": matched,
        "sleeper_positive": matched[matched["sleeper_fantasy_pts_season"] > 0],
        "sleeper_50_plus": matched[matched["sleeper_fantasy_pts_season"] >= 50],
    }
    rows = []
    for label, frame in strata.items():
        rows.append({
            "stratum": label,
            "n": len(frame),
            "season_corr": frame["fantasy_pts_season"].corr(
                frame["sleeper_fantasy_pts_season"]) if len(frame) > 1 else np.nan,
            "mean_abs_divergence": frame["fantasy_pts_season_delta"].abs().mean(),
            "mean_signed_divergence": frame["fantasy_pts_season_delta"].mean(),
            "our_mean": frame["fantasy_pts_season"].mean(),
            "sleeper_mean": frame["sleeper_fantasy_pts_season"].mean(),
        })
    return pd.DataFrame(rows)


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
    print(DIAGNOSTIC_BANNER)
    print(f"{len(merged)} of our rows, {len(matched)} matched to Sleeper "
          f"({len(matched) / len(merged):.0%}) - "
          f"{(merged.match_method == 'gsis_id').sum()} by gsis_id, "
          f"{(merged.match_method == 'name').sum()} by name fallback")
    print(f"\nSeason totals: our mean={matched['fantasy_pts_season'].mean():.2f}, "
          f"Sleeper mean={matched['sleeper_fantasy_pts_season'].mean():.2f}")
    print(f"Correlation with Sleeper (matched only, descriptive): "
          f"{matched['fantasy_pts_season'].corr(matched['sleeper_fantasy_pts_season']):.3f}")
    print(f"Mean absolute season-total divergence (matched only): "
          f"{matched['fantasy_pts_season_delta'].abs().mean():.2f}")
    print("\nSeason-total divergence strata (zero-only rows cannot hide the relevant players):")
    print(comparison_summary_strata(merged).to_string(index=False))
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
                  f"mean_abs_divergence={p['fantasy_pts_season_delta'].abs().mean():.2f}, "
                  f"our_season_mean={p['fantasy_pts_season'].mean():.2f}, "
                  f"sleeper_season_mean={p['sleeper_fantasy_pts_season'].mean():.2f}")
    print(f"\nWritten -> {out_path}")
    print("Reminder: nothing above is an acceptance criterion. To decide whether a "
          "change\nis an improvement, use the leakage-safe outcome harness "
          "(src/projection/fantasy_evaluation.py).\nFor a pass/fail gate on the "
          "board itself, use `python -m src.comparison.spot_check`.")


if __name__ == "__main__":
    main()
