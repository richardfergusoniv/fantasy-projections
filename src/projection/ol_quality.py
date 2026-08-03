"""Team-season OL quality feature, built by weighting Phase 2's pooled
lineman coefficients by how many offensive snaps each lineman actually took
for that team-season (via `snap_counts`), not a league-wide average.

Confidence-flag handling (judgment call, per the Phase 4 spec's explicit
ask to document reasoning either way): `ol_coefficients_pooled.confidence_flag`
flags players whose *individual* credit within a low-churn (fixed starting
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
"""
import pandas as pd

from src.projection.data_prep import SEASONS

OL_POSITIONS = ["G", "T", "C", "OT", "OG", "OL"]
SUBMODELS = ["pass_protection", "run_blocking"]


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


def team_season_ol_quality(conn, seasons=SEASONS):
    """One row per (season, team): weighted-average pass-pro and run-block
    OL quality score (player coef + season fixed effect, snap-share
    weighted), plus n_ol_with_coef (how many of the team's OL snap-takers
    actually resolved to a pooled coefficient) and the team-season churn
    confidence flag. Only meaningful for 2021-2025 (ol_coefficients_pooled's
    window) - seasons outside that range are simply absent from the output.
    """
    seasons = [s for s in seasons if s >= 2021]
    shares = team_season_ol_snap_shares(conn, seasons)

    coefs = pd.read_sql("select gsis_id, coef, submodel, confidence_flag from ol_coefficients_pooled", conn)
    season_fx = pd.read_sql("select season, coef as season_coef, submodel from ol_season_effects_pooled", conn)
    churn = pd.read_sql("select season, team, confidence_flag as team_churn_flag from ol_team_season_churn", conn)

    frames = []
    for submodel in SUBMODELS:
        sub_coefs = coefs[coefs.submodel == submodel][["gsis_id", "coef"]]
        merged = shares.merge(sub_coefs, on="gsis_id", how="left")
        merged = merged.merge(season_fx[season_fx.submodel == submodel][["season", "season_coef"]], on="season", how="left")
        merged["fitted"] = merged["coef"] + merged["season_coef"]

        resolved = merged.dropna(subset=["fitted"]).copy()
        resolved["renorm_weight"] = resolved.groupby(["season", "team"])["snap_share"].transform(lambda s: s / s.sum())
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
    return out.drop(columns=["team_churn_flag"])
