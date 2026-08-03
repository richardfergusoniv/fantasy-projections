"""Rookie-season projection path - deliberately separate from the veteran
LightGBM pipeline in train.py, per the hard project rule: a rookie has no
prior-NFL-season trailing features (the entire premise of the veteran
model), and must not use same-season NFL stats or college production as
inputs (none of which exist in this DB anyway - college production is
simply not modeled here).

Allowed inputs only:
- draft capital (round/pick from `draft_picks`)
- "vacated team opportunity": the target/carry share the rookie's new team
  lost from players who were on that team in season N-1 but are NOT active
  (no games with usage) for that same team in season N. This uses only
  season N-1 data plus who's on/off the roster in season N - never the
  rookie's or anyone's season-N production - so it doesn't leak forward
  information a real preseason projection wouldn't have.

Model: rule-based, not LightGBM. Rookie sample sizes per position x
draft-round-bucket are too small (see PHASE4_REPORT.md) for a tree model
to learn anything but noise, and a rookie's feature vector is structurally
incompatible with the veteran model's inputs anyway - this is exactly the
"distinct path" the spec calls for, not a shrunken version of the same
pipeline. Historical per-game rates for rookies in the same
position/draft-round bucket are averaged, then scaled by the ratio of this
player's team's vacated opportunity to the bucket's historical average
vacated opportunity (clipped to avoid small-sample blowups).
"""
import numpy as np
import pandas as pd

from src.projection.data_prep import SEASONS, load_weekly_usage, season_aggregate
from src.projection.features import TARGET_STATS

ROUND_BUCKETS = {1: "round_1", 2: "round_2_3", 3: "round_2_3", 4: "round_4_7", 5: "round_4_7",
                  6: "round_4_7", 7: "round_4_7"}
VACATED_CLIP = (0.3, 2.5)


def _round_bucket(rnd):
    if pd.isna(rnd):
        return "undrafted"
    return ROUND_BUCKETS.get(int(rnd), "round_4_7")


def load_draft_capital(conn):
    dp = pd.read_sql(
        "select season as draft_season, gsis_id as player_id, round, pick, position "
        "from draft_picks where gsis_id is not null and position in ('QB','RB','WR','TE')", conn,
    )
    dp["round_bucket"] = dp["round"].apply(_round_bucket)
    return dp


def identify_rookie_seasons(conn, seasons=SEASONS):
    """(player_id, season) pairs where `season` is the player's first
    season with any active week in `weekly` (2016-2025 window) AND matches
    their draft season - i.e. no prior-NFL-season row exists at all."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    active = agg[agg["games_played"] > 0]
    first_season = active.groupby("player_id")["season"].min().rename("first_active_season").reset_index()

    draft = load_draft_capital(conn)
    rookies = draft.merge(first_season, on="player_id", how="inner")
    rookies = rookies[rookies["draft_season"] == rookies["first_active_season"]]
    return rookies[["player_id", "draft_season", "round", "pick", "round_bucket", "position"]].rename(
        columns={"draft_season": "season"}
    )


def team_vacated_opportunity(conn, seasons=SEASONS):
    """(season, team) -> vacated_carry_share, vacated_target_share: the
    fraction of the team's season-(N-1) carries/targets that belonged to
    players who did NOT have an active week for that same team in season N
    (retired, cut, signed elsewhere, or simply lost their role - this
    doesn't distinguish why, only that the volume is no longer theirs)."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    agg = agg[agg["games_played"] > 0]

    rows = []
    for season in seasons:
        prev = agg[agg["season"] == season - 1]
        curr = agg[agg["season"] == season]
        if prev.empty:
            continue
        curr_team_of_player = curr.set_index("player_id")["team"]
        prev = prev.copy()
        prev["returning_same_team"] = prev["player_id"].map(curr_team_of_player) == prev["team"]

        g = prev.groupby("team")
        team_totals = g[["carries", "targets"]].sum().rename(
            columns={"carries": "prev_team_carries", "targets": "prev_team_targets"}
        )
        returning = prev[prev["returning_same_team"]].groupby("team")[["carries", "targets"]].sum().rename(
            columns={"carries": "returning_carries", "targets": "returning_targets"}
        )
        merged = team_totals.join(returning, how="left").fillna(0)
        merged["vacated_carry_share"] = 1 - merged["returning_carries"] / merged["prev_team_carries"].replace(0, np.nan)
        merged["vacated_target_share"] = 1 - merged["returning_targets"] / merged["prev_team_targets"].replace(0, np.nan)
        merged["season"] = season
        rows.append(merged.reset_index()[["season", "team", "vacated_carry_share", "vacated_target_share"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "team", "vacated_carry_share", "vacated_target_share"]
    )


def build_rookie_dataset(conn, feature_table, seasons=SEASONS):
    """Rookie player-seasons with draft capital + vacated opportunity +
    actual per-game rates (for fitting the historical bucket averages /
    for backtest evaluation) - NOT the veteran feature columns."""
    rookies = identify_rookie_seasons(conn, seasons)
    vacated = team_vacated_opportunity(conn, seasons)

    stat_cols = sorted({s for stats in TARGET_STATS.values() for s in stats})
    pg_cols = [f"{s}_pg" for s in stat_cols]
    actuals = feature_table[["player_id", "season", "team", "games_played"] + pg_cols]

    df = rookies.merge(actuals, on=["player_id", "season"], how="inner")
    df = df.merge(vacated, on=["season", "team"], how="left")
    return df


def fit_rookie_baselines(rookie_df, train_seasons):
    """Historical (position, round_bucket) -> mean per-game rate + mean
    vacated_carry/target_share, fit ONLY on train_seasons (so the backtest
    holdout season's own rookies never inform their own baseline)."""
    train = rookie_df[rookie_df["season"].isin(train_seasons) & (rookie_df["games_played"] > 0)]
    pg_cols = [c for c in rookie_df.columns if c.endswith("_pg")]
    baselines = train.groupby(["position", "round_bucket"])[pg_cols + ["vacated_carry_share", "vacated_target_share"]].mean()
    counts = train.groupby(["position", "round_bucket"]).size().rename("n_train_rookies")
    return baselines.join(counts)


def predict_rookies(rookie_df, baselines, target_seasons):
    """Rule-based projection: bucket mean per-game rate, scaled by this
    player's team's vacated opportunity vs. the bucket's historical average
    vacated opportunity (RB/WR/TE: carry or target share as relevant; QB:
    target share of passing volume used as the closest available proxy).
    Falls back to the unscaled bucket mean if the bucket has no historical
    rows or the vacated feature is null (e.g. an expansion-style edge case).
    """
    pg_cols = [c for c in rookie_df.columns if c.endswith("_pg")]
    target = rookie_df[rookie_df["season"].isin(target_seasons)].copy()

    def project_row(row):
        key = (row["position"], row["round_bucket"])
        if key not in baselines.index:
            return pd.Series({c: np.nan for c in pg_cols} | {"low_confidence": True, "baseline_n": 0})
        b = baselines.loc[key]
        vac_col = "vacated_carry_share" if row["position"] == "RB" else "vacated_target_share"
        player_vac, hist_vac = row.get(vac_col), b[vac_col]
        if pd.isna(player_vac) or pd.isna(hist_vac) or hist_vac == 0:
            scale = 1.0
        else:
            scale = np.clip(player_vac / hist_vac, *VACATED_CLIP)
        preds = {c: b[c] * scale for c in pg_cols}
        preds["low_confidence"] = True
        preds["baseline_n"] = b["n_train_rookies"]
        return pd.Series(preds)

    preds = target.apply(project_row, axis=1)
    out = pd.concat([target[["player_id", "season", "team", "position", "round_bucket", "pick"]], preds], axis=1)
    return out
