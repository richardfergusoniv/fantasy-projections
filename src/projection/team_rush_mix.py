"""Team RB/QB/OTHER rush-distribution mix (hierarchical L2).

Mirrors ``team_pass_mix`` for carries: observed position-group carry shares,
scheme-conditioned predictor, first-year OC inheritance, and L3 composition
that renormalizes RB (and optionally QB/OTHER) rushing volume to
``team_carries × group_share``.

No TE/FB package splits — WR+TE (and any non-RB/QB) collapse into OTHER.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.coordinator.inheritance import INHERITANCE_WEIGHTS
from src.projection.data_prep import load_weekly_usage, season_aggregate
from src.projection.team_pass_mix import _NumpyRidge, _load_oc_assignments, _load_scheme_frame

RUSH_GROUPS = ("RB", "QB", "OTHER")
MIX_COLS = ["rb_carry_share", "qb_carry_share", "other_carry_share"]
SCHEME_FEATURES = [
    "personnel_11_rate", "personnel_12_rate", "personnel_21_rate",
    "pass_oe", "neutral_sec_per_play",
    "play_action_rate", "screen_pass_rate", "rpo_rate", "offense_backfield_mean",
]
_RIDGE_ALPHA = 1.0


def _rush_group(position: str) -> str:
    if position == "RB":
        return "RB"
    if position == "QB":
        return "QB"
    return "OTHER"


def observed_team_rush_mix(conn, seasons=None):
    """Per (season, team) RB/QB/OTHER shares of team carries."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    agg = agg[agg["carries"] > 0].copy()
    if seasons is not None:
        agg = agg[agg["season"].isin(seasons)]
    if agg.empty:
        return pd.DataFrame(columns=["season", "team"] + MIX_COLS)

    agg["rush_group"] = agg["position"].map(_rush_group)
    team = agg.groupby(["season", "team", "rush_group"], as_index=False)["carries"].sum()
    pivot = (
        team.pivot_table(
            index=["season", "team"], columns="rush_group", values="carries", fill_value=0.0
        )
        .reindex(columns=list(RUSH_GROUPS), fill_value=0.0)
        .reset_index()
    )
    total = pivot[list(RUSH_GROUPS)].sum(axis=1).replace(0, np.nan)
    out = pivot[["season", "team"]].copy()
    out["rb_carry_share"] = pivot["RB"] / total
    out["qb_carry_share"] = pivot["QB"] / total
    out["other_carry_share"] = pivot["OTHER"] / total
    return out.dropna(subset=MIX_COLS).reset_index(drop=True)


def _softmax_shares(rb_logit, qb_logit):
    """Map two free logits to three shares that sum to 1 (OTHER logit fixed at 0)."""
    m = np.maximum(0.0, np.maximum(rb_logit, qb_logit))
    e_rb = np.exp(rb_logit - m)
    e_qb = np.exp(qb_logit - m)
    e_ot = np.exp(0.0 - m)
    denom = e_rb + e_qb + e_ot
    return e_rb / denom, e_qb / denom, e_ot / denom


def _shares_to_logits(rb, qb, other):
    rb = np.clip(np.asarray(rb, dtype=float), 1e-4, None)
    qb = np.clip(np.asarray(qb, dtype=float), 1e-4, None)
    other = np.clip(np.asarray(other, dtype=float), 1e-4, None)
    return np.log(rb / other), np.log(qb / other)


def fit_team_rush_mix_model(train_mix, scheme):
    """Fit Ridge predictors for RB/QB log-odds from scheme + lagged mix."""
    df = train_mix.merge(scheme, on=["season", "team"], how="left")
    df = df.sort_values(["team", "season"]).reset_index(drop=True)
    lag = df.groupby("team")[MIX_COLS].shift(1)
    for c in MIX_COLS:
        df[f"lag_{c}"] = lag[c]
    feature_cols = [c for c in SCHEME_FEATURES if c in df.columns] + [
        f"lag_{c}" for c in MIX_COLS
    ]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    usable = df.dropna(subset=MIX_COLS + [f"lag_{c}" for c in MIX_COLS])
    if len(usable) < 32:
        return None

    rb_y, qb_y = _shares_to_logits(
        usable["rb_carry_share"], usable["qb_carry_share"], usable["other_carry_share"]
    )
    x = usable[feature_cols]
    rb_model = _NumpyRidge(alpha=_RIDGE_ALPHA).fit(x, rb_y)
    qb_model = _NumpyRidge(alpha=_RIDGE_ALPHA).fit(x, qb_y)
    return {
        "rb_model": rb_model,
        "qb_model": qb_model,
        "features": feature_cols,
        "scheme_medians": {c: float(df[c].median()) for c in feature_cols},
    }


def predict_team_rush_mix(mix_hist, scheme, model, seasons):
    """Score mix for ``seasons`` using prior-season lag + scheme features."""
    if model is None:
        return pd.DataFrame(columns=["season", "team"] + MIX_COLS + ["rush_mix_source"])

    hist = mix_hist.sort_values(["team", "season"])
    lag_lookup = {}
    for _, r in hist.iterrows():
        lag_lookup[(int(r["season"]) + 1, r["team"])] = {
            f"lag_{c}": float(r[c]) for c in MIX_COLS
        }

    scheme = scheme.copy()
    rows = []
    feature_cols = model["features"]
    for season in seasons:
        teams = sorted(set(hist.loc[hist["season"] == season - 1, "team"]).union(
            set(scheme.loc[scheme["season"] == season - 1, "team"])
            if not scheme.empty else set()
        ))
        sch_season = season - 1
        sch = scheme[scheme["season"] == sch_season].set_index("team")
        for team in teams:
            feat = {c: model["scheme_medians"].get(c, 0.0) for c in feature_cols}
            if team in sch.index:
                for c in SCHEME_FEATURES:
                    if c in feature_cols and c in sch.columns and pd.notna(sch.at[team, c]):
                        feat[c] = float(sch.at[team, c])
            lag = lag_lookup.get((season, team))
            if lag is None:
                means = hist[MIX_COLS].mean()
                rows.append({
                    "season": season, "team": team,
                    "rb_carry_share": float(means["rb_carry_share"]),
                    "qb_carry_share": float(means["qb_carry_share"]),
                    "other_carry_share": float(means["other_carry_share"]),
                    "rush_mix_source": "league_mean_fallback",
                })
                continue
            feat.update(lag)
            x = pd.DataFrame([{c: feat.get(c, model["scheme_medians"].get(c, 0.0))
                               for c in feature_cols}])
            rb_l = float(model["rb_model"].predict(x)[0])
            qb_l = float(model["qb_model"].predict(x)[0])
            rb, qb, other = _softmax_shares(rb_l, qb_l)
            rows.append({
                "season": season, "team": team,
                "rb_carry_share": rb, "qb_carry_share": qb, "other_carry_share": other,
                "rush_mix_source": "scheme_model",
            })
    return pd.DataFrame(rows)


def apply_oc_rush_mix_inheritance(mix_pred, observed_mix, assignments=None):
    """Blend year-1 OC seats toward inherited rush mix using tendency weights."""
    if assignments is None:
        assignments = _load_oc_assignments()
    out = mix_pred.copy()
    out["rush_mix_inheritance_basis"] = None
    first = assignments[assignments["first_year_in_seat"].astype(bool)]
    if first.empty:
        return out

    obs = observed_mix.set_index(["season", "team"])
    for idx, row in out.iterrows():
        seat = first[(first["season"] == row["season"]) & (first["team"] == row["team"])]
        if seat.empty:
            continue
        seat = seat.iloc[0]
        weights = INHERITANCE_WEIGHTS.get(seat["promotion_type"])
        team_prior = None
        key = (row["season"] - 1, row["team"])
        if key in obs.index:
            team_prior = obs.loc[key]
        oc_prior = None
        prior_rows = assignments[
            (assignments["oc_name"] == seat["oc_name"]) & (assignments["season"] < row["season"])
        ]
        if not prior_rows.empty:
            last = prior_rows.sort_values("season").iloc[-1]
            ok = (int(last["season"]), last["team"])
            if ok in obs.index:
                oc_prior = obs.loc[ok]
        if team_prior is None and oc_prior is None:
            continue
        if oc_prior is None or weights is None:
            src = team_prior if team_prior is not None else oc_prior
            for c in MIX_COLS:
                out.at[idx, c] = float(src[c])
            out.at[idx, "rush_mix_inheritance_basis"] = "team_only_no_oc_history"
            out.at[idx, "rush_mix_source"] = "oc_inherited"
        elif team_prior is None:
            for c in MIX_COLS:
                out.at[idx, c] = float(oc_prior[c])
            out.at[idx, "rush_mix_inheritance_basis"] = "oc_only_no_team_history"
            out.at[idx, "rush_mix_source"] = "oc_inherited"
        else:
            for c in MIX_COLS:
                out.at[idx, c] = (
                    weights["team"] * float(team_prior[c]) + weights["oc"] * float(oc_prior[c])
                )
            s = sum(out.at[idx, c] for c in MIX_COLS)
            for c in MIX_COLS:
                out.at[idx, c] = out.at[idx, c] / s
            out.at[idx, "rush_mix_inheritance_basis"] = "blend"
            out.at[idx, "rush_mix_source"] = "oc_inherited"
    return out


def build_team_rush_mix_profiles(conn, target_season=None, history_seasons=None):
    """Observed history + scheme predictions for target_season (if given).

    ``history_seasons`` bounds the observed mix used to fit the model and to
    supply lag/inheritance priors — see build_team_pass_mix_profiles. None (the
    shipped default) uses every observed season.
    """
    observed = observed_team_rush_mix(conn, seasons=history_seasons)
    scheme = _load_scheme_frame(conn)
    model = fit_team_rush_mix_model(observed, scheme)
    profiles = observed.copy()
    profiles["rush_mix_source"] = "observed"
    profiles["rush_mix_inheritance_basis"] = None
    if target_season is not None:
        pred = predict_team_rush_mix(observed, scheme, model, [target_season])
        pred = apply_oc_rush_mix_inheritance(pred, observed)
        profiles = pd.concat([profiles, pred], ignore_index=True, sort=False)
    total = profiles[MIX_COLS].sum(axis=1).replace(0, np.nan)
    for c in MIX_COLS:
        profiles[c] = profiles[c] / total
    return profiles, model


def validate_rush_mix_model(conn, seasons=None):
    """Leave-one-season-out MAE vs league-mean and prior-season baselines."""
    observed = observed_team_rush_mix(conn, seasons)
    scheme = _load_scheme_frame(conn)
    seasons = sorted(observed["season"].unique())
    if len(seasons) < 3:
        return {"ok": False, "reason": "need >=3 seasons"}

    rows = []
    for hold in seasons[1:]:
        train = observed[observed["season"] != hold]
        test = observed[observed["season"] == hold]
        if test.empty or train.empty:
            continue
        model = fit_team_rush_mix_model(train, scheme)
        pred = predict_team_rush_mix(train, scheme, model, [hold])
        merged = test.merge(pred, on=["season", "team"], suffixes=("_act", "_pred"))
        if merged.empty:
            continue
        prior = observed[observed["season"] == hold - 1][["team"] + MIX_COLS]
        prior = prior.rename(columns={c: f"{c}_prior" for c in MIX_COLS})
        merged = merged.merge(prior, on="team", how="left")
        league = {c: float(train[c].mean()) for c in MIX_COLS}

        def mae(prefix):
            errs = []
            for c in MIX_COLS:
                a = merged[f"{c}_act"]
                if prefix == "pred":
                    p = merged[f"{c}_pred"]
                elif prefix == "prior":
                    p = merged[f"{c}_prior"]
                else:
                    p = league[c]
                errs.append((a - p).abs().mean())
            return float(np.mean(errs))

        rows.append({
            "season": hold,
            "mae_scheme": mae("pred"),
            "mae_prior": mae("prior"),
            "mae_league": mae("league"),
            "n_teams": len(merged),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return {"ok": False, "reason": "no folds"}
    mean_scheme = float(summary["mae_scheme"].mean())
    mean_prior = float(summary["mae_prior"].mean())
    mean_league = float(summary["mae_league"].mean())
    return {
        "ok": True,
        "folds": summary,
        "mae_scheme": mean_scheme,
        "mae_prior": mean_prior,
        "mae_league": mean_league,
        "beats_prior": mean_scheme < mean_prior,
        "beats_league": mean_scheme < mean_league,
    }


def attach_team_rush_mix(df, mix_profiles, season):
    """Left-join L2 rush mix columns for ``season`` onto a projection frame."""
    mix = mix_profiles[mix_profiles["season"] == season][
        ["team"] + MIX_COLS + ["rush_mix_source"]
    ].drop_duplicates("team")
    out = df.merge(mix, on="team", how="left", validate="many_to_one")
    if out[MIX_COLS].isna().any(axis=None):
        means = mix[MIX_COLS].mean()
        if means.isna().any():
            means = pd.Series({c: 1.0 / 3.0 for c in MIX_COLS})
        for c in MIX_COLS:
            out[c] = out[c].fillna(float(means[c]))
        out["rush_mix_source"] = out["rush_mix_source"].fillna("league_mean_fallback")
    return out


def apply_hierarchical_rush_distribution(df, season_games=17.0):
    """L1×L2×L3 composition for rushing volume stats.

    Renormalizes RB/QB/OTHER carries (and yards/tds family) to
    ``team_carries × group_share``. No TE/FB package splits.
    """
    out = df.copy()
    out["hierarchical_rush_scale"] = np.nan
    out["within_group_carry_share"] = np.nan
    if "team_carries_pg_pred" not in out.columns:
        return out
    if not all(c in out.columns for c in MIX_COLS):
        return out

    volume_stats = ("carries", "rushing_yards", "rushing_tds")
    pos_to_group = {"RB": "RB", "QB": "QB", "WR": "OTHER", "TE": "OTHER"}
    mask = out["position"].isin(pos_to_group) & out["stat"].isin(volume_stats)
    if not mask.any():
        return out

    vol = out[mask & out["stat"].eq("carries")].copy()
    if vol.empty:
        return out

    exposure = pd.to_numeric(
        vol.get("projected_volume_games", vol.get("projected_games")),
        errors="coerce",
    ).fillna(0.0).clip(lower=0)
    vol["_exposure"] = exposure
    vol["_raw_season"] = (
        pd.to_numeric(vol["pred_pg"], errors="coerce").clip(lower=0).fillna(0) * exposure
    )
    vol["_rush_group"] = vol["position"].map(pos_to_group)

    group_col = {
        "RB": "rb_carry_share",
        "QB": "qb_carry_share",
        "OTHER": "other_carry_share",
    }
    factors = {}
    within = {}
    for (team, group), idx in vol.groupby(["team", "_rush_group"]).groups.items():
        rows = vol.loc[idx]
        raw = rows["_raw_season"].to_numpy(dtype=float)
        raw_sum = raw.sum()
        carries_pg = float(rows["team_carries_pg_pred"].iloc[0])
        gshare = float(rows[group_col[group]].iloc[0])
        budget = carries_pg * season_games * gshare
        if raw_sum <= 0 or budget <= 0:
            n = max(len(rows), 1)
            alloc = np.full(len(rows), budget / n)
            wshare = np.full(len(rows), 1.0 / n)
        else:
            wshare = raw / raw_sum
            alloc = wshare * budget
        for i, player_id in enumerate(rows["player_id"]):
            key = (player_id, team, rows["position"].iloc[i])
            old_tgt = float(rows["pred_pg"].iloc[i])
            exp = float(rows["_exposure"].iloc[i])
            new_pg = alloc[i] / exp if exp > 0 else 0.0
            factor = (new_pg / old_tgt) if old_tgt > 0 else 0.0
            factors[key] = factor
            within[key] = float(wshare[i])

    for i in out.index[mask]:
        key = (out.at[i, "player_id"], out.at[i, "team"], out.at[i, "position"])
        factor = factors.get(key)
        if factor is None:
            continue
        old = float(out.at[i, "pred_pg"]) if pd.notna(out.at[i, "pred_pg"]) else 0.0
        out.at[i, "pred_pg"] = old * factor
        for col in ("pred_pg_low", "pred_pg_high"):
            if col in out.columns and pd.notna(out.at[i, col]):
                out.at[i, col] = float(out.at[i, col]) * factor
        out.at[i, "hierarchical_rush_scale"] = factor
        out.at[i, "within_group_carry_share"] = within.get(key, np.nan)

    check = out.drop_duplicates("team")
    if not check.empty:
        s = check[MIX_COLS].sum(axis=1)
        if ((s - 1.0).abs() > 1e-3).any():
            bad = check.loc[(s - 1.0).abs() > 1e-3, "team"].tolist()
            raise ValueError(f"L2 rush mix shares do not sum to 1 for teams: {bad}")
    return out


if __name__ == "__main__":
    from src.projection.data_prep import get_conn

    conn = get_conn()
    try:
        summary = validate_rush_mix_model(conn)
        print(summary)
        if summary.get("ok"):
            print(summary["folds"].to_string(index=False))
        profiles, _ = build_team_rush_mix_profiles(conn, target_season=2026)
        print(profiles[profiles.season == 2026].head(10).to_string(index=False))
    finally:
        conn.close()
