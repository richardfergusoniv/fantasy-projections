"""First-year OC inheritance blend weights (single source of truth).

Used by tendency profiles (``oc_profiles``). Default weights are judgment-call
70/30 internal and 30/70 outside; Phase C3 may replace them when a LOSO grid
search beats these values and the team-only baseline.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

INHERITANCE_WEIGHTS = {
    # LOSO grid-fit 2026-08-14 (see OC_INHERITANCE_FIT_2026-08-14.md):
    # best under internal_team_w >= outside_team_w was 0.6/0.4 for both
    # promotion types; beats judgment 70/30 and team-only.
    "internal": {"team": 0.60, "oc": 0.40},
    "outside_hire": {"team": 0.60, "oc": 0.40},
}

_TEAM_WEIGHT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)

# Kept here so LOSO fitting does not import oc_profiles (nfl_data_py chain).
METRICS = [
    "neutral_sec_per_play", "pass_oe", "pass_oe_neutral", "play_action_rate",
    "screen_pass_rate", "rpo_rate", "offense_backfield_mean",
    "personnel_11_rate", "personnel_12_rate", "personnel_21_rate", "personnel_other_rate",
]

_ASSIGNMENTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "oc_assignments.csv",
)


def _blend_row(team_vals, oc_vals, team_w):
    oc_w = 1.0 - team_w
    out = []
    for a, b in zip(team_vals, oc_vals):
        if pd.isna(a) and pd.isna(b):
            out.append(np.nan)
        elif pd.isna(a):
            out.append(float(b))
        elif pd.isna(b):
            out.append(float(a))
        else:
            out.append(team_w * float(a) + oc_w * float(b))
    return np.asarray(out, dtype=float)


def _load_assignments():
    df = pd.read_csv(_ASSIGNMENTS_CSV)
    df["first_year_in_seat"] = df["first_year_in_seat"].astype(bool)
    return df


def _load_team_profiles(conn):
    """Prefer stored team_tendency_profiles; fall back to empty."""
    try:
        df = pd.read_sql("select * from team_tendency_profiles", conn)
    except Exception:
        return pd.DataFrame()
    return df


def loso_fit_inheritance_weights(conn, metrics=None, mix_cols=None):
    """Leave-one-season-out grid search over team weights.

    Constraint: ``internal_team_w >= outside_team_w``. Scores first-year seats
    on tendency ``metrics`` (MAE of blended prior vs observed season values).

    ``mix_cols`` is accepted for call-site compatibility but ignored — pass-mix
    scoring was retired with hierarchical volume composition.

    Returns a summary dict. Updates are the caller's responsibility — this
    never mutates ``INHERITANCE_WEIGHTS``.
    """
    del mix_cols

    if metrics is None:
        metrics = list(METRICS)

    assignments = _load_assignments()
    team_profiles = _load_team_profiles(conn)
    if team_profiles.empty:
        return {"ok": False, "reason": "team_tendency_profiles missing"}

    available = [m for m in metrics if m in team_profiles.columns]
    if not available:
        return {"ok": False, "reason": "no metric columns in team profiles"}
    metrics = available

    first = assignments[assignments["first_year_in_seat"].astype(bool)].copy()
    first = first[first["promotion_type"].isin(("internal", "outside_hire"))]
    seasons = sorted(first["season"].unique())
    if len(seasons) < 3:
        return {"ok": False, "reason": "need >=3 first-year seasons"}

    metric_scale = {
        m: float(team_profiles[m].std(skipna=True) or 1.0) for m in metrics
    }

    def score_weights_scaled(internal_tw, outside_tw, hold_season):
        errs = []
        fold = first[first["season"] == hold_season]
        for _, seat in fold.iterrows():
            tw = internal_tw if seat["promotion_type"] == "internal" else outside_tw
            team_prior = team_profiles[
                (team_profiles["season"] == seat["season"] - 1)
                & (team_profiles["team"] == seat["team"])
            ]
            team_prior = team_prior.iloc[0] if not team_prior.empty else None
            prior_rows = assignments[
                (assignments["oc_name"] == seat["oc_name"])
                & (assignments["season"] < seat["season"])
            ]
            oc_prior = None
            if not prior_rows.empty:
                last = prior_rows.sort_values("season").iloc[-1]
                match = team_profiles[
                    (team_profiles["season"] == last["season"])
                    & (team_profiles["team"] == last["team"])
                ]
                oc_prior = match.iloc[0] if not match.empty else None

            act = team_profiles[
                (team_profiles["season"] == seat["season"])
                & (team_profiles["team"] == seat["team"])
            ]
            if act.empty or (team_prior is None and oc_prior is None):
                continue
            act = act.iloc[0]
            if team_prior is None:
                pred_vals = np.asarray([oc_prior[m] for m in metrics], dtype=float)
            elif oc_prior is None:
                pred_vals = np.asarray([team_prior[m] for m in metrics], dtype=float)
            else:
                pred_vals = _blend_row(
                    [team_prior[m] for m in metrics],
                    [oc_prior[m] for m in metrics],
                    tw,
                )
            act_vals = np.asarray([act[m] for m in metrics], dtype=float)
            scales = np.asarray([metric_scale[m] for m in metrics], dtype=float)
            ok = np.isfinite(act_vals) & np.isfinite(pred_vals) & (scales > 0)
            if ok.any():
                errs.append(float(np.mean(np.abs(act_vals[ok] - pred_vals[ok]) / scales[ok])))
        return float(np.mean(errs)) if errs else np.nan

    grid_rows = []
    for internal_tw, outside_tw in itertools.product(_TEAM_WEIGHT_GRID, repeat=2):
        if internal_tw < outside_tw:
            continue
        fold_maes = []
        for hold in seasons:
            mae = score_weights_scaled(internal_tw, outside_tw, hold)
            if np.isfinite(mae):
                fold_maes.append(mae)
        if not fold_maes:
            continue
        grid_rows.append({
            "internal_team_w": internal_tw,
            "outside_team_w": outside_tw,
            "mae": float(np.mean(fold_maes)),
            "n_folds": len(fold_maes),
        })

    if not grid_rows:
        return {"ok": False, "reason": "no scored folds"}

    grid = pd.DataFrame(grid_rows).sort_values("mae").reset_index(drop=True)
    best = grid.iloc[0]

    judgment_mask = (grid["internal_team_w"] == 0.7) & (grid["outside_team_w"] == 0.3)
    judgment_mae = float(grid.loc[judgment_mask, "mae"].iloc[0]) if judgment_mask.any() else np.nan

    team_only_maes = []
    for hold in seasons:
        mae = score_weights_scaled(1.0, 1.0, hold)
        if np.isfinite(mae):
            team_only_maes.append(mae)
    team_only_mae = float(np.mean(team_only_maes)) if team_only_maes else np.nan

    beats_judgment = bool(best["mae"] < judgment_mae - 1e-6) if np.isfinite(judgment_mae) else False
    beats_team_only = bool(best["mae"] < team_only_mae - 1e-6) if np.isfinite(team_only_mae) else False
    same_as_judgment = (
        abs(float(best["internal_team_w"]) - 0.7) < 1e-9
        and abs(float(best["outside_team_w"]) - 0.3) < 1e-9
    )
    recommend_update = beats_judgment and beats_team_only and not same_as_judgment

    return {
        "ok": True,
        "grid": grid,
        "best": {
            "internal": {
                "team": float(best["internal_team_w"]),
                "oc": float(1.0 - best["internal_team_w"]),
            },
            "outside_hire": {
                "team": float(best["outside_team_w"]),
                "oc": float(1.0 - best["outside_team_w"]),
            },
            "mae": float(best["mae"]),
        },
        "judgment_mae": judgment_mae,
        "team_only_mae": team_only_mae,
        "beats_judgment": beats_judgment,
        "beats_team_only": beats_team_only,
        "recommend_update": recommend_update,
        "current": INHERITANCE_WEIGHTS,
    }
