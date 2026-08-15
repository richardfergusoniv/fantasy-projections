"""Named-player coherence check on our own board, plus a descriptive
Sleeper divergence report that cannot fail the run.

Why this file was restructured (2026-08-15)
-------------------------------------------
This script used to be, structurally, a **Sleeper-agreement regression
suite**: the watchlist was annotated with the direction each player was
expected to move *toward Sleeper*, the controls block was defined as "must
not move relative to Sleeper", and the headline framing was "should
converge toward Sleeper". That trains exactly the wrong reflex. Sleeper
projects full slates (`gp = 18` for essentially every player it tracks);
this system projects expected value including the probability a player does
not play. The two are differently framed, so **agreement is not accuracy**
and divergence is not, by itself, a defect.

The genuine value of this script was never the agreement. It was that a
fixed, named watchlist catches silent structural failures that aggregate
metrics hide -- the Phase 5 rookie-filter bug and the Phase 6 team-change
bugs both showed up here as a specific well-known player being missing or
impossible, while the MAE tables looked clean.

So the distinction this file now encodes is:

    flag INCOHERENCE, never mere DISAGREEMENT.

**Incoherence** is a statement about our board alone that cannot be true of
any real NFL season: negative volume, a player projected above his own
team's total, a team's players collectively allocated more than the team
anchor, or a watched player absent from the output entirely. These are
computed from ``output/fantasy_points_<season>.csv`` with no external
reference at all, they are the only things that can fail this script, and
they would still work if Sleeper vanished tomorrow.

**Divergence** from Sleeper is printed, sorted so the largest gaps are
impossible to miss, and explicitly labelled as information for a human to
judge. It never touches the exit code.

The reference column beside each watched player is now **actual production
in the prior completed season**, read from the project's own database, not
a third party's forecast of the coming one.

Usage: `python -m src.comparison.spot_check --season 2026`
Exit code: 0 unless the board is incoherent (or a watched player is
missing). Never non-zero for disagreeing with Sleeper.
"""
import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

SEASON_GAMES = 17.0

# (name-in-output, what this row is a canary for). These are named because a
# specific, well-known player going missing or impossible is the failure mode
# aggregate metrics cannot see. The annotations describe the STRUCTURAL risk
# each row covers; none of them assert a direction relative to any external
# projection.
WATCHLIST = [
    # Short 2025 seasons: the availability decomposition (per-game rate x
    # projected games) is easiest to get wrong here, in either direction.
    ("Malik Nabers", "4 games in 2025 - short-season availability handling"),
    ("Jayden Reed", "5 games in 2025 - short-season availability handling"),
    ("Garrett Wilson", "7 games in 2025 - short-season availability handling"),
    ("Mike Evans", "8 games in 2025 + team change - two effects at once"),
    ("Chris Godwin Jr.", "9 games, second straight short season"),
    ("Terry McLaurin", "10 games in 2025 - short-season availability handling"),
    ("Christian Watson", "10 games in 2025 - short-season availability handling"),
    # Curation boundary: these were uncurated deep_bench rows and are the
    # canaries for the depth-chart gating path dropping real players.
    ("Parker Washington", "was uncurated deep_bench - gating drop canary"),
    ("Wan'Dale Robinson", "was uncurated deep_bench - gating drop canary"),
    # Rookie/sophomore path: the Phase 5 rookie filter silently dropped this
    # whole cohort once.
    ("Luther Burden III", "sophomore - rookie-cohort carry-through canary"),
    ("Matthew Golden", "sophomore - rookie-cohort carry-through canary"),
    ("Jayden Higgins", "sophomore - rookie-cohort carry-through canary"),
]

# Rows whose situation is unambiguous, so a large swing in them is a signal
# that something upstream moved that should not have. Reported for a human to
# read; a swing here is NOT a failure condition, because there is no
# ground truth in this file to say what the right number is.
STABILITY_ANCHORS = [
    ("Rashee Rice", "2025 absence was SUSPENSION, not injury - must not be "
                    "treated as an injury-shortened season"),
    ("Ja'Marr Chase", "healthy full-season alpha - upstream-drift anchor"),
    ("Justin Jefferson", "healthy full-season alpha - upstream-drift anchor"),
]

# "Fantasy-relevant" cutoffs for the descriptive per-position divergence
# summary: roughly two starters per league slot in a 12-team league.
POSITION_TOP_N = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}

# Team-level allocation tolerance. The reconcilers pin each team's named
# supply at (or just under) the team anchor, so on a healthy board these
# ratios sit at 1.000 with float noise. Anything past this is real
# over-allocation, not rounding.
TEAM_ALLOCATION_TOLERANCE = 0.02
# A single player carrying more than his whole team is impossible at any
# tolerance; the epsilon only absorbs float error.
PLAYER_SHARE_EPSILON = 1e-6

# (per-game stat column, team per-game anchor column, human label). Only
# stats with a real team-level budget appear here; per-game rates are
# converted to season totals with each player's own exposure first, because
# a per-game rate is conditional on playing and does not sum across a roster.
TEAM_BUDGETED_STATS = [
    ("pg_carries", "team_carries_pg_pred", "carries"),
    ("pg_rushing_yards", "team_rushing_yards_pg_pred", "rushing yards"),
    ("pg_targets", "team_pass_attempts_pg_pred", "targets vs pass attempts"),
    ("pg_receiving_yards", "team_passing_yards_pg_pred", "receiving yards"),
    ("pg_attempts", "team_pass_attempts_pg_pred", "pass attempts"),
    ("pg_passing_yards", "team_passing_yards_pg_pred", "passing yards"),
]


def board_path(season):
    return os.path.join(OUTPUT_DIR, f"fantasy_points_{season}.csv")


def load_board(season):
    """Our own output. Coherence is judged against this and nothing else."""
    return pd.read_csv(board_path(season), low_memory=False)


def _exposure(board):
    """Season exposure in games, matching how the pipeline builds totals."""
    if "projected_volume_games" in board.columns:
        return pd.to_numeric(
            board["projected_volume_games"], errors="coerce"
        ).fillna(pd.to_numeric(board["projected_games"], errors="coerce"))
    return pd.to_numeric(board["projected_games"], errors="coerce")


def load_prior_season_actuals(season):
    """Real production in the last completed season, from our own database.

    This replaces the old Sleeper reference column. Returns
    ``(frame_or_None, note)`` so an unavailable database degrades to a
    printed explanation rather than a crash or, worse, a silent fallback to
    the consensus column this file just stopped trusting.
    """
    prior = season - 1
    try:
        from src.projection.data_prep import get_conn, load_weekly_usage, season_aggregate
        from src.projection.fantasy_points import SCORING
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"unavailable ({type(exc).__name__}: {exc})"
    try:
        agg = season_aggregate(load_weekly_usage(get_conn()))
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"unavailable ({type(exc).__name__}: {exc})"
    agg = agg[agg["season"] == prior].copy()
    if agg.empty:
        return None, f"no rows for season {prior}"
    points = pd.Series(0.0, index=agg.index)
    for stat, weight in SCORING.items():
        if stat in agg.columns:
            points = points + pd.to_numeric(agg[stat], errors="coerce").fillna(0.0) * weight
    agg[f"actual_{prior}_points"] = points
    keep = ["player_id", "games_played", f"actual_{prior}_points"]
    return agg[keep].rename(columns={"games_played": f"actual_{prior}_games"}), "ok"


# --------------------------------------------------------------------------
# Coherence: statements about our board that cannot be true of a real season.
# Nothing below reads any external projection.
# --------------------------------------------------------------------------

def check_watchlist_presence(board, entries):
    """A named player vanishing from the output is a defect, full stop."""
    present = set(board["display_name"].dropna())
    return [
        {"rule": "missing_from_output", "subject": name, "detail": why, "value": None}
        for name, why in entries if name not in present
    ]


def check_negative_volume(board):
    """No projected volume, rate or point total may be negative."""
    violations = []
    cols = [c for c in board.columns
            if c.startswith("pg_") or c.startswith("our_")
            or c in ("projected_games", "projected_volume_games",
                     "fantasy_pts", "fantasy_pts_season")]
    for col in cols:
        values = pd.to_numeric(board[col], errors="coerce")
        bad = board.loc[values < -PLAYER_SHARE_EPSILON]
        for idx, row in bad.iterrows():
            violations.append({
                "rule": "negative_volume",
                "subject": f"{row.get('display_name')} ({row.get('team')})",
                "detail": col,
                "value": round(float(values.loc[idx]), 4),
            })
    return violations


def check_player_exceeds_team(board):
    """No single player may out-produce his own team's projected total."""
    violations = []
    exposure = _exposure(board)
    for pg_col, anchor_col, label in TEAM_BUDGETED_STATS:
        if pg_col not in board.columns or anchor_col not in board.columns:
            continue
        player_season = pd.to_numeric(board[pg_col], errors="coerce").fillna(0.0) * exposure
        team_season = pd.to_numeric(board[anchor_col], errors="coerce") * SEASON_GAMES
        ratio = player_season / team_season
        bad = board.loc[ratio > 1.0 + PLAYER_SHARE_EPSILON]
        for idx, row in bad.iterrows():
            violations.append({
                "rule": "player_exceeds_team_total",
                "subject": f"{row.get('display_name')} ({row.get('team')})",
                "detail": label,
                "value": round(float(ratio.loc[idx]), 4),
            })
    return violations


def check_team_allocation(board):
    """A team's named players may not be allocated more than the team has.

    This is the share-summing-past-1 check, expressed against the team
    anchor rather than against a normalized share column, so it still fires
    if a share column is itself wrong.
    """
    violations = []
    exposure = _exposure(board)
    for pg_col, anchor_col, label in TEAM_BUDGETED_STATS:
        if pg_col not in board.columns or anchor_col not in board.columns:
            continue
        frame = board.assign(
            _player_season=pd.to_numeric(board[pg_col], errors="coerce").fillna(0.0) * exposure,
            _anchor=pd.to_numeric(board[anchor_col], errors="coerce"),
        )
        grouped = frame.groupby("team", dropna=True).agg(
            allocated=("_player_season", "sum"), anchor=("_anchor", "max"))
        grouped = grouped[grouped["anchor"] > 0]
        grouped["ratio"] = grouped["allocated"] / (grouped["anchor"] * SEASON_GAMES)
        for team, row in grouped[
                grouped["ratio"] > 1.0 + TEAM_ALLOCATION_TOLERANCE].iterrows():
            violations.append({
                "rule": "team_over_allocated",
                "subject": str(team),
                "detail": label,
                "value": round(float(row["ratio"]), 4),
            })
    return violations


def coherence_violations(board, watched_names):
    """Every incoherence on the board. Empty list means the board is sane."""
    return (
        check_watchlist_presence(board, watched_names)
        + check_negative_volume(board)
        + check_player_exceeds_team(board)
        + check_team_allocation(board)
    )


# --------------------------------------------------------------------------
# Reporting. Descriptive only.
# --------------------------------------------------------------------------

def watchlist_table(board, entries, actuals=None, prior_season=None):
    """Our numbers for each watched player, beside real prior-season output."""
    rows = []
    for name, why in entries:
        match = board[board["display_name"] == name]
        if match.empty:
            rows.append({"player": name, "why": why, "MISSING": True})
            continue
        r = match.iloc[0]
        row = {
            "player": name,
            "team": r.get("team"),
            "role": r.get("role"),
            "depth_rank": r.get("depth_rank"),
            "proj_games": round(float(r["projected_games"]), 1)
            if pd.notna(r.get("projected_games")) else None,
            "ours_fpts_pg": round(float(r["fantasy_pts"]), 2)
            if pd.notna(r.get("fantasy_pts")) else None,
            "ours_fpts_season": round(float(r["fantasy_pts_season"]), 1)
            if pd.notna(r.get("fantasy_pts_season")) else None,
        }
        if actuals is not None:
            act = actuals[actuals["player_id"] == r.get("player_id")]
            if not act.empty:
                a = act.iloc[0]
                row[f"actual_{prior_season}_g"] = round(
                    float(a[f"actual_{prior_season}_games"]), 1)
                row[f"actual_{prior_season}_pts"] = round(
                    float(a[f"actual_{prior_season}_points"]), 1)
        row["why"] = why
        rows.append(row)
    return pd.DataFrame(rows)


def divergence_table(comparison, entries):
    """Sleeper deltas for watched players. Information, not a verdict."""
    rows = []
    for name, _why in entries:
        match = comparison[comparison["display_name"] == name]
        if match.empty:
            rows.append({"player": name, "matched_sleeper": False})
            continue
        r = match.iloc[0]
        rows.append({
            "player": name,
            "matched_sleeper": bool(r.get("matched_sleeper", False)),
            "match_method": r.get("match_method"),
            "ours_season": round(float(r["fantasy_pts_season"]), 1)
            if pd.notna(r.get("fantasy_pts_season")) else None,
            "sleeper_season": round(float(r["sleeper_fantasy_pts_season"]), 1)
            if pd.notna(r.get("sleeper_fantasy_pts_season")) else None,
            "divergence": round(float(r["fantasy_pts_season_delta"]), 1)
            if pd.notna(r.get("fantasy_pts_season_delta")) else None,
        })
    out = pd.DataFrame(rows)
    if "divergence" in out.columns:
        out = out.reindex(
            out["divergence"].abs().sort_values(
                ascending=False, na_position="last").index)
    return out


def position_divergence(comparison):
    """Per-position central divergence. Descriptive; not an error metric.

    A non-zero number here is the expected consequence of projecting
    expected value against a source that projects full slates. It is
    reported so the SHAPE of the difference is visible, not so it can be
    driven toward zero.
    """
    rows = []
    for pos, n in POSITION_TOP_N.items():
        sub = comparison[comparison["position"] == pos].nlargest(
            n, "sleeper_fantasy_pts_season")
        rows.append({
            "position": pos,
            "n": len(sub),
            "mean_divergence": round(sub["fantasy_pts_season_delta"].mean(), 1),
            "median_divergence": round(sub["fantasy_pts_season_delta"].median(), 1),
        })
    return pd.DataFrame(rows)


SLEEPER_FRAMING_NOTE = (
    "Sleeper projects a full slate (gp=18 for essentially every player it\n"
    "  tracks) and allocates ~96.8% of team carries to named players against\n"
    "  our ~83.8%. This system projects EXPECTED VALUE, including the chance a\n"
    "  player does not play. The two are differently framed, so a divergence\n"
    "  here is not an error and closing it is not an objective. Read this\n"
    "  section to notice a player who moved for a reason you cannot explain,\n"
    "  then go and explain it -- not to make the numbers agree."
)


def main():
    parser = argparse.ArgumentParser(
        description="Coherence check on our own board; Sleeper divergence is "
                    "reported but never fails the run.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--skip-actuals", action="store_true",
                        help="Skip the prior-season actual-outcome reference "
                             "column (avoids opening the database).")
    parser.add_argument("--no-divergence", action="store_true",
                        help="Skip the descriptive Sleeper divergence section.")
    args = parser.parse_args()

    board = load_board(args.season)
    watched = WATCHLIST + STABILITY_ANCHORS

    actuals, actuals_note, prior = None, "skipped", args.season - 1
    if not args.skip_actuals:
        actuals, actuals_note = load_prior_season_actuals(args.season)

    print(f"=== Board coherence, season {args.season} "
          f"({len(board)} rows, {board['team'].nunique()} teams) ===")
    violations = coherence_violations(board, watched)
    if violations:
        print(pd.DataFrame(violations).to_string(index=False))
    else:
        print("No incoherence found: no negative volume, no player above his "
              "team's total,\nno team over-allocated beyond "
              f"{TEAM_ALLOCATION_TOLERANCE:.0%}, every watched player present.")

    print(f"\n=== Watchlist (reference column = ACTUAL {prior} production; "
          f"{actuals_note}) ===")
    print(watchlist_table(board, WATCHLIST, actuals, prior).to_string(index=False))
    print("\n=== Stability anchors (unambiguous situations; a large swing here "
          "is worth explaining) ===")
    print(watchlist_table(board, STABILITY_ANCHORS, actuals, prior).to_string(index=False))

    comparison_path = os.path.join(
        OUTPUT_DIR, f"sleeper_comparison_{args.season}.csv")
    if not args.no_divergence and os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path, low_memory=False)
        matched = comparison[comparison["matched_sleeper"] == True].copy()  # noqa: E712
        print("\n=== DIAGNOSTIC: divergence from Sleeper "
              "(information for a human; cannot fail this check) ===")
        print("  " + SLEEPER_FRAMING_NOTE)
        print("\n-- watched players, largest divergence first --")
        print(divergence_table(comparison, watched).to_string(index=False))
        print("\n-- per-position central divergence --")
        print(position_divergence(matched).to_string(index=False))
        if "match_method" in matched.columns:
            print("\n-- join audit (a join defect IS actionable) --")
            print(matched["match_method"].value_counts(dropna=False).to_string())
            collisions = int(matched.get(
                "match_collision", pd.Series(False, index=matched.index)
            ).fillna(False).sum())
            print(f"ambiguous name collisions left unmatched: {collisions}")
    elif not args.no_divergence:
        print(f"\n(no {os.path.basename(comparison_path)}; divergence section "
              f"skipped - it is optional by design)")

    if violations:
        print(f"\nFAIL: {len(violations)} incoherence(s) on the board "
              f"(see the coherence section above).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
