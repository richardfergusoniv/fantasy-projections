"""Team-season OL quality feature, built by weighting leakage-safe,
season-specific lineman coefficients by how many offensive snaps each lineman actually took
for that team-season (via `snap_counts`), not a league-wide average.

Confidence-flag handling (judgment call, per the Phase 4 spec's explicit
ask to document reasoning either way): low-churn team seasons contain
players whose *individual* credit within a fixed starting
five) team-season block is not statistically identified - ridge has no
information to separate them from their linemates for those plays. But a
team-season's *aggregate* OL quality is a sum over that team's linemen
weighted by snap share, and arbitrary mis-splits of credit within a
collinear block roughly cancel in the sum (if lineman A is over-credited
at lineman B's expense, the pair's combined contribution to the team total
is largely unaffected - the ridge fit still has to explain the same total
outcome over those plays). So the team-season aggregate is used regardless
of confidence_flag, but the team's `ol_team_season_churn.confidence_flag`
(itself, not the player-level rollup) is carried through as a separate
`ol_confidence_low_churn` feature so the downstream LightGBM model can learn
to weight it down if the aggregate turns out to be less reliable in
practice. This does NOT protect against a systematic bias in how well the
whole unit (vs. another team's whole unit) is captured by the model -
that's a real, unresolved limitation, not fixed by this weighting scheme.

Live-only trailing average (Phase C4): when ``trailing_seasons`` > 0 and
``trailing_for_seasons`` is set, those seasons' OL scores are replaced by a
snap-weighted average of exact-season scores over the trailing window ending
at that season. Historical/backtest rows stay exact-season (no pooled
leakage). Kill-switch: ``OL_TRAILING_SEASONS`` in contracts (0 = off).
"""
import numpy as np
import pandas as pd

from src.projection.data_prep import SEASONS

try:
    from src.projection.contracts import OL_TRAILING_SEASONS
except ImportError:  # pragma: no cover - contracts always present in-repo
    OL_TRAILING_SEASONS = 0

OL_POSITIONS = ["G", "T", "C", "OT", "OG", "OL"]
SUBMODELS = ["pass_protection", "run_blocking"]
_SCORE_COLS = ["ol_pass_protection_score", "ol_run_blocking_score"]


def team_season_ol_snap_shares(conn, seasons=SEASONS):
    """(season, team, gsis_id) -> share of that team-season's OL offensive
    snaps taken by that player, restricted to OL-position snap rows."""
    sc = pd.read_sql(
        f"select season, team, week, pfr_player_id, position, offense_snaps from snap_counts "
        f"where season in ({','.join(map(str, seasons))}) and game_type = 'REG' "
        f"and position in ({','.join(repr(p) for p in OL_POSITIONS)})", conn,
    )
    crosswalk = pd.read_sql("select gsis_id, pfr_id from players where pfr_id is not null", conn)
    sc = sc.merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")

    player_snaps = sc.groupby(["season", "team", "gsis_id"])["offense_snaps"].sum().reset_index()
    team_snaps = player_snaps.groupby(["season", "team"])["offense_snaps"].sum().rename("team_ol_snaps")
    player_snaps = player_snaps.merge(team_snaps, on=["season", "team"])
    player_snaps["snap_share"] = player_snaps["offense_snaps"] / player_snaps["team_ol_snaps"]
    return player_snaps


def _apply_trailing_average(out, shares, trailing_seasons, trailing_for_seasons):
    """Replace scores for live seasons with snap-weighted trailing averages."""
    if trailing_seasons <= 0 or not trailing_for_seasons:
        return out
    team_snaps = (
        shares.groupby(["season", "team"])["offense_snaps"]
        .sum()
        .rename("team_ol_snaps")
        .reset_index()
    )
    base = out.merge(team_snaps, on=["season", "team"], how="left")
    base["team_ol_snaps"] = base["team_ol_snaps"].fillna(0.0)

    for season in sorted(set(trailing_for_seasons)):
        window = list(range(season - trailing_seasons + 1, season + 1))
        hist = base[base["season"].isin(window)].copy()
        if hist.empty:
            continue
        rows = []
        for team, g in hist.groupby("team"):
            w = g["team_ol_snaps"].to_numpy(dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(g))
            row = {"season": season, "team": team}
            for c in _SCORE_COLS:
                if c not in g.columns:
                    continue
                vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(vals) & np.isfinite(w)
                if not ok.any():
                    row[c] = np.nan
                else:
                    row[c] = float(np.average(vals[ok], weights=w[ok]))
            rows.append(row)
        if not rows:
            continue
        trail = pd.DataFrame(rows)
        keep = base.loc[base["season"] != season].copy()
        patched = base.loc[base["season"] == season].drop(columns=_SCORE_COLS, errors="ignore")
        patched = patched.merge(trail, on=["season", "team"], how="left")
        base = pd.concat([keep, patched], ignore_index=True, sort=False)

    return base.drop(columns=["team_ol_snaps"], errors="ignore")


def team_season_ol_quality(
    conn,
    seasons=SEASONS,
    trailing_seasons=None,
    trailing_for_seasons=None,
):
    """One row per (season, team): weighted-average pass-pro and run-block
    OL quality score (exact-season player coefficient, snap-share
    weighted), plus n_ol_with_coef (how many of the team's OL snap-takers
    actually resolved to an exact-season coefficient) and the team-season churn
    confidence flag. Only meaningful for 2021-2025 (`ol_coefficients`'s
    window) - seasons outside that range are simply absent from the output.

    ``trailing_seasons`` / ``trailing_for_seasons``: live-only smoothing. Pass
    ``trailing_for_seasons={source_season}`` from predict so historical feature
    rows stay exact-season. Default ``trailing_seasons`` reads
    ``OL_TRAILING_SEASONS`` from contracts (0 = off).
    """
    if trailing_seasons is None:
        trailing_seasons = OL_TRAILING_SEASONS

    seasons = [s for s in seasons if s >= 2021]
    # Pull enough history for trailing windows when smoothing live seasons.
    if trailing_seasons and trailing_for_seasons:
        need = set(seasons)
        for s in trailing_for_seasons:
            need.update(range(s - trailing_seasons + 1, s + 1))
        seasons = sorted(s for s in need if s >= 2021)

    shares = team_season_ol_snap_shares(conn, seasons)

    # IMPORTANT: do not use ``ol_coefficients_pooled`` here.  That table is
    # estimated from the complete 2021-2025 play window, so attaching it to a
    # 2021-2024 feature row lets future plays influence a historical
    # transition/backtest.  ``ol_coefficients`` is fit independently within
    # each season and is therefore available as-of the end of that season.
    # Production and historical rows now use the same exact-season contract.
    coefs = pd.read_sql(
        "select season, gsis_id, coef, submodel from ol_coefficients", conn
    )
    churn = pd.read_sql(
        "select season, team, confidence_flag as team_churn_flag from ol_team_season_churn",
        conn,
    )

    frames = []
    for submodel in SUBMODELS:
        sub_coefs = coefs[coefs.submodel == submodel][["season", "gsis_id", "coef"]]
        merged = shares.merge(sub_coefs, on=["season", "gsis_id"], how="left")
        merged["fitted"] = merged["coef"]

        resolved = merged.dropna(subset=["fitted"]).copy()
        resolved["renorm_weight"] = resolved.groupby(["season", "team"])["snap_share"].transform(
            lambda s: s / s.sum()
        )
        resolved["weighted"] = resolved["renorm_weight"] * resolved["fitted"]

        agg = resolved.groupby(["season", "team"]).agg(
            score=("weighted", "sum"),
            n_ol_with_coef=("gsis_id", "nunique"),
            coef_snap_coverage=("snap_share", "sum"),
        ).reset_index()
        agg = agg.rename(columns={
            "score": f"ol_{submodel}_score",
            "n_ol_with_coef": f"ol_{submodel}_n_players",
            "coef_snap_coverage": f"ol_{submodel}_snap_coverage",
        })
        frames.append(agg)

    out = frames[0].merge(frames[1], on=["season", "team"], how="outer")
    out = out.merge(churn, on=["season", "team"], how="left")
    out["ol_confidence_low_churn"] = (out["team_churn_flag"] == "unit_level").astype(int)
    out = out.drop(columns=["team_churn_flag"])

    if trailing_seasons and trailing_for_seasons:
        out = _apply_trailing_average(out, shares, trailing_seasons, trailing_for_seasons)

    return out
