"""Expanded QB rushing features and leakage-safe multi-season pooling.

Production ``FEATURE_COLS`` keep their historical definitions. New columns are
additive so sealed model artifacts (which pin their own feature lists) are
unchanged at inference. Experimental training / evaluation may consume the
expansion set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# Additive features — not yet part of sealed model contracts.
QB_RUSH_EXPANSION_FEATURES = (
    "qb_scramble_per_dropback",
    "qb_yards_per_designed_carry",
    "qb_yards_per_scramble",
    "qb_rz_designed_carry_rate",
    "qb_gl_designed_carry_rate",
    "qb_designed_run_rate_pooled",
    "qb_scramble_per_dropback_pooled",
    "qb_rush_archetype_carries_pg",
    "qb_rush_archetype_yards_pg",
)

QB_RUSH_POOL_LOOKBACK = 4
QB_RUSH_POOL_FULL_GAMES = 12.0
MOBILE_ARCHETYPE_CARRIES_PG = 5.5


@dataclass
class LineageStage:
    stage: str
    values: dict


def _season_weights(games: pd.Series) -> pd.Series:
    g = pd.to_numeric(games, errors="coerce").fillna(0.0).clip(lower=0.0)
    return (g / QB_RUSH_POOL_FULL_GAMES).clip(upper=1.0) * g


def compute_qb_rush_splits_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate designed/scramble/RZ/GL rush splits from play-level rows."""
    if pbp is None or pbp.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "season",
                "designed_carries",
                "scramble_carries",
                "designed_rushing_yards",
                "scramble_rushing_yards",
                "rz_designed_carries",
                "gl_designed_carries",
                "dropbacks",
            ]
        )
    work = pbp.copy()
    if "week" in work.columns:
        work = work[pd.to_numeric(work["week"], errors="coerce").between(1, 18)]
    rush = work[pd.to_numeric(work.get("rush_attempt"), errors="coerce").fillna(0).eq(1)].copy()
    scramble = rush[pd.to_numeric(rush.get("qb_scramble"), errors="coerce").fillna(0).eq(1)]
    designed = rush[pd.to_numeric(rush.get("qb_scramble"), errors="coerce").fillna(0).ne(1)]
    designed = designed[designed["rusher_player_id"].notna()]
    scramble = scramble[scramble["rusher_player_id"].notna()]

    def _agg(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["player_id", "season", f"{prefix}_carries", f"{prefix}_yards"])
        out = (
            frame.groupby(["rusher_player_id", "season"], as_index=False)
            .agg(
                carries=("rush_attempt", "sum"),
                yards=("rushing_yards", "sum"),
            )
            .rename(
                columns={
                    "rusher_player_id": "player_id",
                    "carries": f"{prefix}_carries",
                    "yards": f"{prefix}_yards",
                }
            )
        )
        return out

    des = _agg(designed, "designed")
    scr = _agg(scramble, "scramble")
    if "yardline_100" in designed.columns and not designed.empty:
        rz = (
            designed[pd.to_numeric(designed["yardline_100"], errors="coerce") <= 20]
            .groupby(["rusher_player_id", "season"], as_index=False)
            .size()
            .rename(columns={"rusher_player_id": "player_id", "size": "rz_designed_carries"})
        )
        gl = (
            designed[pd.to_numeric(designed["yardline_100"], errors="coerce") <= 5]
            .groupby(["rusher_player_id", "season"], as_index=False)
            .size()
            .rename(columns={"rusher_player_id": "player_id", "size": "gl_designed_carries"})
        )
    else:
        rz = pd.DataFrame(columns=["player_id", "season", "rz_designed_carries"])
        gl = pd.DataFrame(columns=["player_id", "season", "gl_designed_carries"])

    # Dropbacks ≈ pass attempts + sacks when available; else pass_attempt rows.
    if "passer_player_id" in work.columns:
        passes = work[pd.to_numeric(work.get("pass_attempt"), errors="coerce").fillna(0).eq(1)]
        if "sack" in work.columns:
            sacks = work[pd.to_numeric(work.get("sack"), errors="coerce").fillna(0).eq(1)]
            drop = pd.concat([passes, sacks], ignore_index=True)
        else:
            drop = passes
        dropbacks = (
            drop.groupby(["passer_player_id", "season"], as_index=False)
            .size()
            .rename(columns={"passer_player_id": "player_id", "size": "dropbacks"})
        )
    else:
        dropbacks = pd.DataFrame(columns=["player_id", "season", "dropbacks"])

    out = des.merge(scr, on=["player_id", "season"], how="outer")
    out = out.merge(rz, on=["player_id", "season"], how="left")
    out = out.merge(gl, on=["player_id", "season"], how="left")
    out = out.merge(dropbacks, on=["player_id", "season"], how="left")
    for col in (
        "designed_carries",
        "scramble_carries",
        "designed_yards",
        "scramble_yards",
        "rz_designed_carries",
        "gl_designed_carries",
        "dropbacks",
    ):
        # normalize names from _agg
        pass
    out = out.rename(
        columns={
            "designed_yards": "designed_rushing_yards",
            "scramble_yards": "scramble_rushing_yards",
        }
    )
    for col in (
        "designed_carries",
        "scramble_carries",
        "designed_rushing_yards",
        "scramble_rushing_yards",
        "rz_designed_carries",
        "gl_designed_carries",
        "dropbacks",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["player_id"] = out["player_id"].astype(str)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype(int)
    return out


def attach_qb_rush_expansion_features(
    base: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    team_plays_col: str = "team_plays_active",
) -> pd.DataFrame:
    """Attach expansion rush features for the current season row.

    Does not mutate legacy ``qb_designed_run_rate``. Multi-season pooled
    columns are filled by :func:`apply_qb_rush_multi_season_pooling`.
    """
    out = base.copy()
    if splits is None or splits.empty:
        for col in QB_RUSH_EXPANSION_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
        return out

    merged = out.merge(splits, on=["player_id", "season"], how="left", suffixes=("", "_split"))
    for col in (
        "designed_carries",
        "scramble_carries",
        "designed_rushing_yards",
        "scramble_rushing_yards",
        "rz_designed_carries",
        "gl_designed_carries",
        "dropbacks",
    ):
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    dropbacks = merged["dropbacks"].replace(0, np.nan)
    # Prefer attempts as dropback proxy when pbp passer coverage is thin.
    if "attempts" in merged.columns:
        dropbacks = dropbacks.fillna(pd.to_numeric(merged["attempts"], errors="coerce"))
    des = merged["designed_carries"].fillna(0.0)
    scr = merged["scramble_carries"].fillna(0.0)
    des_yds = merged["designed_rushing_yards"]
    scr_yds = merged["scramble_rushing_yards"]
    games = pd.to_numeric(merged.get("games_played", merged.get("games")), errors="coerce")

    merged["qb_scramble_per_dropback"] = scr / dropbacks
    merged["qb_yards_per_designed_carry"] = des_yds / des.replace(0, np.nan)
    merged["qb_yards_per_scramble"] = scr_yds / scr.replace(0, np.nan)
    merged["qb_rz_designed_carry_rate"] = merged["rz_designed_carries"].fillna(0.0) / games.replace(0, np.nan)
    merged["qb_gl_designed_carry_rate"] = merged["gl_designed_carries"].fillna(0.0) / games.replace(0, np.nan)

    # Keep placeholders for pooled columns; filled next.
    for col in (
        "qb_designed_run_rate_pooled",
        "qb_scramble_per_dropback_pooled",
        "qb_rush_archetype_carries_pg",
        "qb_rush_archetype_yards_pg",
    ):
        if col not in merged.columns:
            merged[col] = np.nan

    # Preserve original base columns; drop merge helpers that collide.
    keep = [c for c in out.columns if c in merged.columns] + [
        c for c in QB_RUSH_EXPANSION_FEATURES if c in merged.columns
    ]
    # also keep split raw counts for provenance
    for c in (
        "designed_carries",
        "scramble_carries",
        "designed_rushing_yards",
        "scramble_rushing_yards",
        "rz_designed_carries",
        "gl_designed_carries",
        "dropbacks",
    ):
        if c in merged.columns and c not in keep:
            keep.append(c)
    return merged.loc[:, ~merged.columns.duplicated()].copy()


def apply_qb_rush_multi_season_pooling(
    feat: pd.DataFrame,
    *,
    lookback: int = QB_RUSH_POOL_LOOKBACK,
) -> pd.DataFrame:
    """Fill pooled rush features using only seasons <= current row season.

    Pooling is games-weighted across the lookback window so one shortened or
    injured season cannot erase several seasons of stable rushing. Pooled
    columns are the weighted means themselves (not re-blended toward the
    current season with a 12-game credibility cap — that previously nullified
    the pool for any season with games >= 12).
    """
    out = feat.copy()
    if out.empty or "player_id" not in out.columns:
        return out

    # Ensure rate columns exist for pooling inputs.
    games = pd.to_numeric(out.get("games_played", out.get("games")), errors="coerce")
    if "qb_designed_run_rate" not in out.columns:
        out["qb_designed_run_rate"] = np.nan
    if "qb_scramble_per_dropback" not in out.columns:
        out["qb_scramble_per_dropback"] = np.nan
    if "carries_pg" not in out.columns and "carries" in out.columns:
        out["carries_pg"] = pd.to_numeric(out["carries"], errors="coerce") / games.replace(0, np.nan)
    if "rushing_yards_pg" not in out.columns and "rushing_yards" in out.columns:
        out["rushing_yards_pg"] = pd.to_numeric(out["rushing_yards"], errors="coerce") / games.replace(0, np.nan)

    out = out.sort_values(["player_id", "season"]).reset_index(drop=True)
    pooled_design = []
    pooled_scr = []
    arch_car = []
    arch_yds = []

    for idx, row in out.iterrows():
        pid = str(row["player_id"])
        season = int(row["season"])
        # Window includes the current season; games-weighting downweights
        # short/injured rows relative to full seasons of the same player.
        window = out[
            (out["player_id"].astype(str) == pid)
            & (out["season"] <= season)
            & (out["season"] >= season - lookback + 1)
        ]
        w = _season_weights(window.get("games_played", window.get("games")))

        def _wmean(col: str) -> float:
            if col not in window.columns or not w.gt(0).any():
                return float("nan")
            vals = pd.to_numeric(window[col], errors="coerce")
            mask = vals.notna() & w.gt(0)
            if not mask.any():
                return float("nan")
            return float(np.average(vals[mask], weights=w[mask]))

        des_p = _wmean("qb_designed_run_rate")
        scr_p = _wmean("qb_scramble_per_dropback")
        car_p = _wmean("carries_pg")
        yds_p = _wmean("rushing_yards_pg")
        pooled_design.append(des_p if pd.notna(des_p) else row.get("qb_designed_run_rate"))
        pooled_scr.append(scr_p if pd.notna(scr_p) else row.get("qb_scramble_per_dropback"))

        # Archetype prior: pool player history; thin samples shrink toward
        # mobile/pocket means estimated from other players' prior seasons only.
        peers = out[(out["season"] < season) & (out["season"] >= season - lookback)]
        mobile_mean = pocket_mean = car_p
        y_mobile = y_pocket = yds_p
        if "carries_pg" in peers.columns and not peers.empty:
            car_peer = pd.to_numeric(peers["carries_pg"], errors="coerce")
            mobile_mask = car_peer >= MOBILE_ARCHETYPE_CARRIES_PG
            if mobile_mask.any():
                mobile_mean = float(car_peer[mobile_mask].mean())
            if (~mobile_mask & car_peer.notna()).any():
                pocket_mean = float(car_peer[~mobile_mask].mean())
            if "rushing_yards_pg" in peers.columns:
                y_peer = pd.to_numeric(peers["rushing_yards_pg"], errors="coerce")
                if mobile_mask.any():
                    y_mobile = float(y_peer[mobile_mask].mean())
                if (~mobile_mask & y_peer.notna()).any():
                    y_pocket = float(y_peer[~mobile_mask].mean())

        player_level = car_p
        is_mobile = bool(pd.notna(player_level) and player_level >= MOBILE_ARCHETYPE_CARRIES_PG)
        arch = mobile_mean if is_mobile else pocket_mean
        y_arch = y_mobile if is_mobile else y_pocket
        sample_games = float(pd.to_numeric(window.get("games_played", window.get("games")), errors="coerce").fillna(0).sum())
        shrink = min(1.0, sample_games / 40.0)
        if pd.notna(player_level) and pd.notna(arch):
            arch_car.append(shrink * float(player_level) + (1 - shrink) * float(arch))
        else:
            arch_car.append(player_level if pd.notna(player_level) else arch)
        if pd.notna(yds_p) and pd.notna(y_arch):
            arch_yds.append(shrink * float(yds_p) + (1 - shrink) * float(y_arch))
        else:
            arch_yds.append(yds_p if pd.notna(yds_p) else y_arch)

    out["qb_designed_run_rate_pooled"] = pooled_design
    out["qb_scramble_per_dropback_pooled"] = pooled_scr
    out["qb_rush_archetype_carries_pg"] = arch_car
    out["qb_rush_archetype_yards_pg"] = arch_yds
    return out


def patch_inference_row_with_rush_pool(
    row: pd.Series,
    history: pd.DataFrame,
    *,
    target_season: int,
) -> tuple[pd.Series, dict]:
    """Patch sealed-model feature names using multi-season rush evidence.

    Used by the experimental 'pooled-inference' arm: existing QB_carries model
    weights stay frozen; only the inference vector changes. Never player-specific.

    Patched priors are games-weighted means over seasons in
    ``[target_season - lookback, target_season)``. A single short season is
    naturally down-weighted by ``_season_weights`` and cannot erase a multi-
    season rushing profile.
    """
    out = row.copy()
    audit = {"target_season": int(target_season), "patches": {}}
    pid = str(row.get("player_id"))
    if not pid or pid == "nan":
        return out, {**audit, "reason": "missing_player_id"}
    hist = history[
        (history["player_id"].astype(str) == pid)
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - QB_RUSH_POOL_LOOKBACK)
    ].copy()
    if hist.empty:
        return out, {**audit, "reason": "no_history"}
    if "games_played" not in hist.columns and "games" in hist.columns:
        hist["games_played"] = hist["games"]
    if "carries_pg" not in hist.columns and "carries" in hist.columns:
        hist["carries_pg"] = hist["carries"] / hist["games_played"].replace(0, np.nan)
    if "rushing_yards_pg" not in hist.columns and "rushing_yards" in hist.columns:
        hist["rushing_yards_pg"] = hist["rushing_yards"] / hist["games_played"].replace(0, np.nan)
    if "rushing_tds_pg" not in hist.columns and "rushing_tds" in hist.columns:
        hist["rushing_tds_pg"] = hist["rushing_tds"] / hist["games_played"].replace(0, np.nan)

    w = _season_weights(hist["games_played"])

    def wmean(col):
        if col not in hist.columns:
            return None
        vals = pd.to_numeric(hist[col], errors="coerce")
        mask = vals.notna() & w.gt(0)
        if not mask.any():
            return None
        return float(np.average(vals[mask], weights=w[mask]))

    mapping = {
        "qb_designed_run_rate": "qb_designed_run_rate",
        "prior_carries_pg": "carries_pg",
        "prior_rushing_yards_pg": "rushing_yards_pg",
        "prior_rushing_tds_pg": "rushing_tds_pg",
        "carry_share": "carry_share",
        "prior_role_rate": "carries_pg",
        "prior_role_rate_3y": "carries_pg",
    }
    for feat_name, hist_col in mapping.items():
        pooled = wmean(hist_col)
        if pooled is None or feat_name not in out.index:
            continue
        cur = pd.to_numeric(out.get(feat_name), errors="coerce")
        audit["patches"][feat_name] = {
            "before": float(cur) if pd.notna(cur) else None,
            "pooled": pooled,
            "after": pooled,
            "input_seasons": [int(s) for s in hist["season"].tolist()],
            "sample_games": float(pd.to_numeric(hist["games_played"], errors="coerce").fillna(0).sum()),
        }
        out[feat_name] = pooled
    return out, audit
