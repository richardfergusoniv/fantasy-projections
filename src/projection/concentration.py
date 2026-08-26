"""Team-position share concentration with exact volume conservation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.artifacts import load_concentration_calibration


CONCENTRATION_FAMILIES = {
    ("WR", "receiving"): ("targets", "receptions", "receiving_yards"),
    ("TE", "receiving"): ("targets", "receptions", "receiving_yards"),
    ("RB", "receiving"): ("targets", "receptions", "receiving_yards"),
    ("RB", "rushing"): ("carries", "rushing_yards"),
}


def _exposure(frame: pd.DataFrame) -> pd.Series:
    volume = pd.to_numeric(frame.get("projected_volume_games"), errors="coerce")
    games = pd.to_numeric(frame.get("projected_games"), errors="coerce")
    return volume.fillna(games).fillna(0.0).clip(lower=0.0)


def _power_factors(values: pd.Series, groups: pd.Series, gamma: float) -> pd.Series:
    """Return monotone power-share factors, preserving every group total."""
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    result = pd.Series(1.0, index=values.index, dtype=float)
    for _, idx in groups.groupby(groups, observed=True).groups.items():
        raw = values.loc[idx]
        positive = raw.gt(0)
        if positive.sum() <= 1:
            continue
        powered = raw.loc[positive].pow(gamma)
        denominator = powered.sum()
        total = raw.sum()
        if denominator <= 0 or total <= 0:
            continue
        adjusted = powered * (total / denominator)
        result.loc[adjusted.index] = adjusted / raw.loc[positive]
    return result


def apply_concentration(df: pd.DataFrame, calibration: dict | None = None) -> pd.DataFrame:
    """Concentrate positive shares inside each team-position room.

    Each stat is transformed independently in season-volume units.  The
    pre-transform total is therefore preserved even when an explicit
    IR/PUP/suspension override gives players different exposure.  Zeros stay
    zero, player ordering is monotonic, interval endpoints receive the point
    factor, and TD rows are never selected.
    """
    out = df.copy()
    out["concentration_scale"] = 1.0
    calibration = calibration or load_concentration_calibration()
    version = str(calibration.get("version", "unfitted_identity"))
    out["concentration_calibration_version"] = "not_applicable"
    if not {"position", "stat", "pred_pg"}.issubset(out.columns):
        return out
    cells = calibration.get("cells", {})
    exposure = _exposure(out)

    for (position, family), stats in CONCENTRATION_FAMILIES.items():
        key = f"{position}:{family}"
        cell = cells.get(key, {})
        gamma = float(cell.get("exponent", 1.0)) if cell.get("promoted", True) else 1.0
        for stat in stats:
            selected = out["position"].eq(position) & out["stat"].eq(stat)
            if not selected.any():
                continue
            out.loc[selected, "concentration_calibration_version"] = version
            season_values = (
                pd.to_numeric(out.loc[selected, "pred_pg"], errors="coerce").fillna(0.0)
                * exposure.loc[selected]
            )
            groups = (
                out.loc[selected, "team"].fillna("__NO_TEAM__")
                if "team" in out.columns
                else pd.Series("__NO_TEAM__", index=out.index[selected])
            )
            factor = _power_factors(season_values, groups, gamma)
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                if col in out.columns:
                    out.loc[selected, col] = (
                        pd.to_numeric(out.loc[selected, col], errors="coerce") * factor
                    )
            out.loc[selected, "concentration_scale"] = factor
    return out
