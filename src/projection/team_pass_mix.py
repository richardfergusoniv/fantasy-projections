"""Team WR/TE/RB pass-distribution mix (hierarchical L2).

Builds observed position-group target shares from free weekly usage, fits a
scheme-conditioned predictor from nflverse play-call features (personnel,
PROE, pace, public FTN), and applies the same first-year OC inheritance
weights used for tendency profiles.

Validation gate (``validate_mix_model``): leave-one-season-out MAE must beat
both league-mean and prior-season mix baselines on average before the layer
is treated as an improvement over carrying last year's mix forward.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.coordinator.inheritance import INHERITANCE_WEIGHTS
from src.projection.contracts import (
    FORMATION_ROLE_BLEND_W,
    WR_FORMATION_ROLE_PRIORS,
    WR_FORMATION_ROLES,
)
from src.projection.data_prep import load_weekly_usage, season_aggregate

PASS_CATCH_POSITIONS = ("WR", "TE", "RB")
MIX_COLS = ["wr_target_share", "te_target_share", "rb_target_share"]
SCHEME_FEATURES = [
    "personnel_11_rate", "personnel_12_rate", "personnel_21_rate",
    "pass_oe", "neutral_sec_per_play",
    "play_action_rate", "screen_pass_rate", "rpo_rate", "offense_backfield_mean",
]
_RIDGE_ALPHA = 1.0
_ASSIGNMENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "coordinator", "oc_assignments.csv",
)


def _load_oc_assignments():
    df = pd.read_csv(_ASSIGNMENTS_CSV)
    df["first_year_in_seat"] = df["first_year_in_seat"].astype(bool)
    return df


class _NumpyRidge:
    """Minimal ridge regression so mix fitting does not require sklearn."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = x.shape
        x_mean = x.mean(axis=0)
        y_mean = y.mean()
        xc = x - x_mean
        yc = y - y_mean
        xtx = xc.T @ xc + self.alpha * np.eye(p)
        self.coef_ = np.linalg.solve(xtx, xc.T @ yc)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        self._feature_names = None
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return x @ self.coef_ + self.intercept_


def observed_team_pass_mix(conn, seasons=None):
    """Per (season, team) WR/TE/RB shares of team targets."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    agg = agg[agg["position"].isin(PASS_CATCH_POSITIONS) & (agg["targets"] > 0)]
    if seasons is not None:
        agg = agg[agg["season"].isin(seasons)]
    if agg.empty:
        return pd.DataFrame(columns=["season", "team"] + MIX_COLS)

    team = agg.groupby(["season", "team", "position"], as_index=False)["targets"].sum()
    pivot = (
        team.pivot_table(
            index=["season", "team"], columns="position", values="targets", fill_value=0.0
        )
        .reindex(columns=list(PASS_CATCH_POSITIONS), fill_value=0.0)
        .reset_index()
    )
    total = pivot[list(PASS_CATCH_POSITIONS)].sum(axis=1).replace(0, np.nan)
    out = pivot[["season", "team"]].copy()
    out["wr_target_share"] = pivot["WR"] / total
    out["te_target_share"] = pivot["TE"] / total
    out["rb_target_share"] = pivot["RB"] / total
    return out.dropna(subset=MIX_COLS).reset_index(drop=True)


def _load_scheme_frame(conn):
    """Team-season scheme features from oc_tendency_profiles when present."""
    try:
        df = pd.read_sql("select * from oc_tendency_profiles", conn)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    keep = ["season", "team"] + [c for c in SCHEME_FEATURES if c in df.columns]
    return df[keep].drop_duplicates(["season", "team"])


def _softmax_shares(wr_logit, te_logit):
    """Map two free logits to three shares that sum to 1 (RB logit fixed at 0)."""
    m = np.maximum(0.0, np.maximum(wr_logit, te_logit))
    e_wr = np.exp(wr_logit - m)
    e_te = np.exp(te_logit - m)
    e_rb = np.exp(0.0 - m)
    denom = e_wr + e_te + e_rb
    return e_wr / denom, e_te / denom, e_rb / denom


def _shares_to_logits(wr, te, rb):
    """Inverse of residual-softmax with RB logit fixed at 0."""
    wr = np.clip(np.asarray(wr, dtype=float), 1e-4, None)
    te = np.clip(np.asarray(te, dtype=float), 1e-4, None)
    rb = np.clip(np.asarray(rb, dtype=float), 1e-4, None)
    return np.log(wr / rb), np.log(te / rb)


def fit_team_pass_mix_model(train_mix, scheme):
    """Fit Ridge predictors for WR/TE log-odds from scheme + lagged mix."""
    df = train_mix.merge(scheme, on=["season", "team"], how="left")
    df = df.sort_values(["team", "season"]).reset_index(drop=True)
    lag = df.groupby("team")[MIX_COLS].shift(1)
    for c in MIX_COLS:
        df[f"lag_{c}"] = lag[c]
    feature_cols = [c for c in SCHEME_FEATURES if c in df.columns] + [
        f"lag_{c}" for c in MIX_COLS
    ]
    # Median-impute scheme gaps (FTN pre-2022).
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    usable = df.dropna(subset=MIX_COLS + [f"lag_{c}" for c in MIX_COLS])
    if len(usable) < 32:
        return None

    wr_y, te_y = _shares_to_logits(
        usable["wr_target_share"], usable["te_target_share"], usable["rb_target_share"]
    )
    x = usable[feature_cols]
    wr_model = _NumpyRidge(alpha=_RIDGE_ALPHA).fit(x, wr_y)
    te_model = _NumpyRidge(alpha=_RIDGE_ALPHA).fit(x, te_y)
    return {
        "wr_model": wr_model,
        "te_model": te_model,
        "features": feature_cols,
        "scheme_medians": {c: float(df[c].median()) for c in feature_cols},
    }


def predict_team_pass_mix(mix_hist, scheme, model, seasons):
    """Score mix for ``seasons`` using prior-season lag + scheme features."""
    if model is None:
        return pd.DataFrame(columns=["season", "team"] + MIX_COLS + ["mix_source"])

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
        # For target season N, scheme proxy is season N-1 (same as OC features).
        sch_season = season - 1
        sch = scheme[scheme["season"] == sch_season].set_index("team")
        for team in teams:
            feat = {c: model["scheme_medians"].get(c, 0.0) for c in feature_cols}
            if team in sch.index:
                for c in SCHEME_FEATURES:
                    key = c if c in feature_cols else None
                    if key and c in sch.columns and pd.notna(sch.at[team, c]):
                        feat[c] = float(sch.at[team, c])
            lag = lag_lookup.get((season, team))
            if lag is None:
                # No prior mix — league mean from hist.
                means = hist[MIX_COLS].mean()
                rows.append({
                    "season": season, "team": team,
                    "wr_target_share": float(means["wr_target_share"]),
                    "te_target_share": float(means["te_target_share"]),
                    "rb_target_share": float(means["rb_target_share"]),
                    "mix_source": "league_mean_fallback",
                })
                continue
            feat.update(lag)
            x = pd.DataFrame([{c: feat.get(c, model["scheme_medians"].get(c, 0.0))
                               for c in feature_cols}])
            wr_l = float(model["wr_model"].predict(x)[0])
            te_l = float(model["te_model"].predict(x)[0])
            wr, te, rb = _softmax_shares(wr_l, te_l)
            rows.append({
                "season": season, "team": team,
                "wr_target_share": wr, "te_target_share": te, "rb_target_share": rb,
                "mix_source": "scheme_model",
            })
    return pd.DataFrame(rows)


def apply_oc_mix_inheritance(mix_pred, observed_mix, assignments=None):
    """Blend year-1 OC seats toward inherited mix using tendency weights."""
    if assignments is None:
        assignments = _load_oc_assignments()
    out = mix_pred.copy()
    out["mix_inheritance_basis"] = None
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
            out.at[idx, "mix_inheritance_basis"] = "team_only_no_oc_history"
            out.at[idx, "mix_source"] = "oc_inherited"
        elif team_prior is None:
            for c in MIX_COLS:
                out.at[idx, c] = float(oc_prior[c])
            out.at[idx, "mix_inheritance_basis"] = "oc_only_no_team_history"
            out.at[idx, "mix_source"] = "oc_inherited"
        else:
            for c in MIX_COLS:
                out.at[idx, c] = (
                    weights["team"] * float(team_prior[c]) + weights["oc"] * float(oc_prior[c])
                )
            # Renormalize after blend.
            s = sum(out.at[idx, c] for c in MIX_COLS)
            for c in MIX_COLS:
                out.at[idx, c] = out.at[idx, c] / s
            out.at[idx, "mix_inheritance_basis"] = "blend"
            out.at[idx, "mix_source"] = "oc_inherited"
    return out


def build_team_pass_mix_profiles(conn, target_season=None, history_seasons=None):
    """Observed history + scheme predictions for target_season (if given).

    ``history_seasons`` bounds the observed mix used both to FIT the model and
    to supply the lag/inheritance priors. It exists so a held-out evaluation
    fold can build this layer from seasons <= its source season only; the
    shipped path leaves it None and uses every observed season. Scheme rows are
    read at ``season - 1`` inside ``predict_team_pass_mix`` and are joined onto
    the (already bounded) observed frame when fitting, so bounding the observed
    frame is sufficient to bound the whole layer.
    """
    observed = observed_team_pass_mix(conn, seasons=history_seasons)
    scheme = _load_scheme_frame(conn)
    model = fit_team_pass_mix_model(observed, scheme)
    profiles = observed.copy()
    profiles["mix_source"] = "observed"
    profiles["mix_inheritance_basis"] = None
    if target_season is not None:
        pred = predict_team_pass_mix(observed, scheme, model, [target_season])
        pred = apply_oc_mix_inheritance(pred, observed)
        profiles = pd.concat([profiles, pred], ignore_index=True, sort=False)
    # Ensure shares sum to 1.
    total = profiles[MIX_COLS].sum(axis=1).replace(0, np.nan)
    for c in MIX_COLS:
        profiles[c] = profiles[c] / total
    return profiles, model


def validate_mix_model(conn, seasons=None):
    """Leave-one-season-out MAE vs league-mean and prior-season baselines.

    Returns a summary dict. ``beats_prior`` is True when scheme+lag MAE is
    strictly below prior-season MAE on mean absolute error across shares.
    """
    observed = observed_team_pass_mix(conn, seasons)
    scheme = _load_scheme_frame(conn)
    seasons = sorted(observed["season"].unique())
    if len(seasons) < 3:
        return {"ok": False, "reason": "need >=3 seasons"}

    rows = []
    for hold in seasons[1:]:  # need a lag year
        train = observed[observed["season"] != hold]
        test = observed[observed["season"] == hold]
        if test.empty or train.empty:
            continue
        model = fit_team_pass_mix_model(train, scheme)
        pred = predict_team_pass_mix(train, scheme, model, [hold])
        merged = test.merge(pred, on=["season", "team"], suffixes=("_act", "_pred"))
        if merged.empty:
            continue
        # Prior-season baseline
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


def attach_team_pass_mix(df, mix_profiles, season):
    """Left-join L2 mix columns for ``season`` onto a projection frame."""
    mix = mix_profiles[mix_profiles["season"] == season][
        ["team"] + MIX_COLS + ["mix_source"]
    ].drop_duplicates("team")
    out = df.merge(mix, on="team", how="left", validate="many_to_one")
    # Fallback: league mean of that season's profiles, else equal thirds.
    if out[MIX_COLS].isna().any(axis=None):
        means = mix[MIX_COLS].mean()
        if means.isna().any():
            means = pd.Series({c: 1.0 / 3.0 for c in MIX_COLS})
        for c in MIX_COLS:
            out[c] = out[c].fillna(float(means[c]))
        out["mix_source"] = out["mix_source"].fillna("league_mean_fallback")
    return out


def _allocate_within_group(raw, budget):
    """Classic fungible split: model weights × group budget."""
    raw = np.asarray(raw, dtype=float)
    raw_sum = raw.sum()
    if raw_sum <= 0 or budget <= 0:
        n = max(len(raw), 1)
        alloc = np.full(len(raw), budget / n)
        wshare = np.full(len(raw), 1.0 / n)
    else:
        wshare = raw / raw_sum
        alloc = wshare * budget
    return alloc, wshare


def _allocate_wr_by_formation_role(raw, roles, budget, blend_w=None):
    """Two-stage WR split: formation-role budgets, then model weights within role.

    ``roles`` entries in WR_FORMATION_ROLES participate in the role mix. Other /
    missing roles share leftover budget by model mass. When no known roles are
    present, falls back to fungible ``_allocate_within_group``.
    """
    if blend_w is None:
        blend_w = FORMATION_ROLE_BLEND_W
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    role_arr = np.asarray(
        [r if isinstance(r, str) and r in WR_FORMATION_ROLE_PRIORS else "" for r in roles],
        dtype=object,
    )
    known = np.array([r in WR_FORMATION_ROLE_PRIORS for r in role_arr])
    if not known.any() or blend_w <= 0:
        return _allocate_within_group(raw, budget)

    present = [r for r in WR_FORMATION_ROLES if (role_arr == r).any()]
    prior = np.array([WR_FORMATION_ROLE_PRIORS[r] for r in present], dtype=float)
    prior = prior / prior.sum()
    model = np.array([raw[role_arr == r].sum() for r in present], dtype=float)
    model_sum = model.sum()
    model = prior.copy() if model_sum <= 0 else model / model_sum
    blended = (1.0 - blend_w) * model + blend_w * prior
    blended = blended / blended.sum()

    other_raw = raw[~known].sum()
    total_raw = raw.sum()
    if other_raw > 0 and total_raw > 0:
        assigned_budget = budget * (1.0 - other_raw / total_raw)
        other_budget = budget - assigned_budget
    else:
        assigned_budget = budget
        other_budget = 0.0

    alloc = np.zeros(n, dtype=float)
    for i, role in enumerate(present):
        mask = role_arr == role
        role_budget = assigned_budget * blended[i]
        raw_r = raw[mask]
        raw_r_sum = raw_r.sum()
        if raw_r_sum > 0:
            alloc[mask] = role_budget * (raw_r / raw_r_sum)
        else:
            alloc[mask] = role_budget / max(mask.sum(), 1)

    if other_budget > 0 and (~known).any():
        raw_o = raw[~known]
        raw_o_sum = raw_o.sum()
        if raw_o_sum > 0:
            alloc[~known] = other_budget * (raw_o / raw_o_sum)
        else:
            alloc[~known] = other_budget / max((~known).sum(), 1)

    wshare = alloc / budget if budget > 0 else np.full(n, 1.0 / max(n, 1))
    return alloc, wshare


def apply_hierarchical_pass_distribution(df, season_games=17.0, formation_role_blend_w=None):
    """L1×L2×L3 composition for receiving volume stats.

    Player models supply within-group allocation weights (raw pred_pg ×
    exposure). Group budgets come from team_pass_attempts × position mix.
    Cross-position totals are preserved by construction; within-group shares
    renormalize to the L2 budget.

    For WR rows with ``formation_role`` (LWR/RWR/SWR), within-WR budget is
    split in two stages: role mix blended toward chart priors, then model
    weights inside each role. TE/RB stay fungible within position.
    """
    out = df.copy()
    out["hierarchical_pass_scale"] = np.nan
    out["within_group_target_share"] = np.nan
    if "team_pass_attempts_pg_pred" not in out.columns:
        return out
    if not all(c in out.columns for c in MIX_COLS):
        return out

    volume_stats = ("targets", "receptions", "receiving_yards", "receiving_tds")
    mask = out["position"].isin(PASS_CATCH_POSITIONS) & out["stat"].isin(volume_stats)
    if not mask.any():
        return out

    # Allocation weights from the volume stat (targets); apply same factor
    # to the whole receiving family so yards/receptions stay coherent.
    vol = out[mask & out["stat"].eq("targets")].copy()
    if vol.empty:
        # Fall back to receiving_yards weights if targets absent.
        vol = out[mask & out["stat"].eq("receiving_yards")].copy()
        weight_stat = "receiving_yards"
    else:
        weight_stat = "targets"

    exposure = pd.to_numeric(vol.get("projected_volume_games", vol.get("projected_games")),
                             errors="coerce").fillna(0.0).clip(lower=0)
    vol["_exposure"] = exposure
    vol["_raw_season"] = pd.to_numeric(vol["pred_pg"], errors="coerce").clip(lower=0).fillna(0) * exposure
    if "formation_role" not in vol.columns:
        vol["formation_role"] = None

    group_col = {
        "WR": "wr_target_share", "TE": "te_target_share", "RB": "rb_target_share",
    }
    factors = {}
    within = {}
    for (team, position), idx in vol.groupby(["team", "position"]).groups.items():
        rows = vol.loc[idx]
        raw = rows["_raw_season"].to_numpy(dtype=float)
        attempts_pg = float(rows["team_pass_attempts_pg_pred"].iloc[0])
        gshare = float(rows[group_col[position]].iloc[0])
        budget = attempts_pg * season_games * gshare
        if position == "WR":
            alloc, wshare = _allocate_wr_by_formation_role(
                raw, rows["formation_role"].tolist(), budget,
                blend_w=formation_role_blend_w,
            )
        else:
            alloc, wshare = _allocate_within_group(raw, budget)
        for i, player_id in enumerate(rows["player_id"]):
            key = (player_id, team, position)
            exp = float(rows["_exposure"].iloc[i])
            # Factor relative to this player's targets pred_pg.
            old_tgt = float(rows["pred_pg"].iloc[i])
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
        out.at[i, "hierarchical_pass_scale"] = factor
        out.at[i, "within_group_target_share"] = within.get(key, np.nan)

    # Tripwire helper: group shares must still sum ~1 on the frame.
    check = out.drop_duplicates("team")
    if not check.empty:
        s = check[MIX_COLS].sum(axis=1)
        if ((s - 1.0).abs() > 1e-3).any():
            bad = check.loc[(s - 1.0).abs() > 1e-3, "team"].tolist()
            raise ValueError(f"L2 mix shares do not sum to 1 for teams: {bad}")
    return out


if __name__ == "__main__":
    from src.projection.data_prep import get_conn

    conn = get_conn()
    try:
        summary = validate_mix_model(conn)
        print(summary)
        if summary.get("ok"):
            print(summary["folds"].to_string(index=False))
        profiles, _ = build_team_pass_mix_profiles(conn, target_season=2026)
        profiles.to_sql("team_pass_mix_profiles", conn, if_exists="replace", index=False)
        print(profiles[profiles.season == 2026].head(10).to_string(index=False))
    finally:
        conn.close()
