"""Week-varying conversion latents for Milestone 3 (Phase C).

Yards, catch/completion rates, and TDs come from season rates × opponent-only
multipliers. Completions cannot exceed attempts; receptions cannot exceed
targets. Same-week volume and ADP/Vegas are not inputs.
"""
from __future__ import annotations

import pandas as pd

from src.projection.weekly_latent.constants import (
    CONV_BLEND,
    CONV_FACTOR_MAX,
    CONV_FACTOR_MIN,
    CONVERSION_RATES,
    YARDAGE_PER_OPP,
)
from src.projection.weekly_latent.environment import _stamp_or_none, refuse_forbidden_m2_columns


def conversion_multiplier(opp_mult: object, *, invert: bool = False) -> float:
    """m = 1 ± CONV_BLEND × (opp_mult − 1), clipped. Opponent-only, not HA/env."""
    try:
        raw = float(opp_mult)
    except (TypeError, ValueError):
        raw = 1.0
    if pd.isna(raw):
        raw = 1.0
    delta = CONV_BLEND * (raw - 1.0)
    value = 1.0 - delta if invert else 1.0 + delta
    return float(max(CONV_FACTOR_MIN, min(CONV_FACTOR_MAX, value)))


def attach_conversion_multipliers(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add m_pass_conv / m_rush_conv / m_int_conv from opponent factors."""
    refuse_forbidden_m2_columns(weekly, where="M3 conversions")
    out = weekly.copy()
    pass_src = (
        out["opp_pass_mult"] if "opp_pass_mult" in out.columns else pd.Series(1.0, index=out.index)
    )
    rush_src = (
        out["opp_rush_mult"] if "opp_rush_mult" in out.columns else pd.Series(1.0, index=out.index)
    )
    out["m_pass_conv"] = [conversion_multiplier(v) for v in pass_src]
    out["m_rush_conv"] = [conversion_multiplier(v) for v in rush_src]
    out["m_int_conv"] = [conversion_multiplier(v, invert=True) for v in pass_src]
    bye = out["is_bye"].eq(1) if "is_bye" in out.columns else pd.Series(False, index=out.index)
    out.loc[bye, ["m_pass_conv", "m_rush_conv", "m_int_conv"]] = 0.0
    prior = (
        out["prior_available_at"]
        if "prior_available_at" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )
    board = (
        out["available_at_board"]
        if "available_at_board" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )
    conv_at = []
    for p, b in zip(prior, board):
        conv_at.append(_stamp_or_none(p) or _stamp_or_none(b))
    out["conv_available_at"] = conv_at
    return out


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def apply_conversion_latents(weekly: pd.DataFrame) -> pd.DataFrame:
    """Fill box stats from opportunity × week-varying rates, then clip."""
    out = attach_conversion_multipliers(weekly)
    m_pass = out["m_pass_conv"].astype(float)
    m_rush = out["m_rush_conv"].astype(float)
    m_int = out["m_int_conv"].astype(float)

    ypa = _col(out, "passing_yards_per_opp")
    ypc = _col(out, "rushing_yards_per_opp")
    ypt = _col(out, "receiving_yards_per_opp")
    attempts = _col(out, "attempts")
    carries = _col(out, "carries")
    targets = _col(out, "targets")

    out["passing_yards"] = attempts * ypa * m_pass
    out["rushing_yards"] = carries * ypc * m_rush
    out["receiving_yards"] = targets * ypt * m_pass

    out["completions"] = attempts * _col(out, "completions_rate") * m_pass
    out["passing_tds"] = attempts * _col(out, "passing_tds_rate") * m_pass
    out["interceptions"] = attempts * _col(out, "interceptions_rate") * m_int
    out["receptions"] = targets * _col(out, "receptions_rate") * m_pass
    out["receiving_tds"] = targets * _col(out, "receiving_tds_rate") * m_pass
    out["rushing_tds"] = carries * _col(out, "rushing_tds_rate") * m_rush
    return clip_internal_consistency(out)


def clip_internal_consistency(weekly: pd.DataFrame) -> pd.DataFrame:
    """Hard constraints: rec ≤ targets, cmp ≤ att, INT ≤ att − cmp, TDs ≤ counts."""
    out = weekly.copy()
    attempts = _col(out, "attempts").clip(lower=0.0)
    targets = _col(out, "targets").clip(lower=0.0)
    carries = _col(out, "carries").clip(lower=0.0)
    out["attempts"] = attempts
    out["targets"] = targets
    out["carries"] = carries
    out["completions"] = _col(out, "completions").clip(lower=0.0, upper=attempts)
    out["receptions"] = _col(out, "receptions").clip(lower=0.0, upper=targets)
    remain_pass = (attempts - out["completions"]).clip(lower=0.0)
    out["interceptions"] = _col(out, "interceptions").clip(lower=0.0, upper=remain_pass)
    out["passing_tds"] = _col(out, "passing_tds").clip(lower=0.0, upper=out["completions"])
    out["receiving_tds"] = _col(out, "receiving_tds").clip(lower=0.0, upper=out["receptions"])
    out["rushing_tds"] = _col(out, "rushing_tds").clip(lower=0.0, upper=carries)
    for col in ("passing_yards", "rushing_yards", "receiving_yards"):
        out[col] = _col(out, col).clip(lower=0.0)
    unused = [name for name in CONVERSION_RATES if name not in out.columns]
    for name in unused:
        out[name] = 0.0
    unused_yards = [name for name in YARDAGE_PER_OPP if name not in out.columns]
    for name in unused_yards:
        out[name] = 0.0
    return out
