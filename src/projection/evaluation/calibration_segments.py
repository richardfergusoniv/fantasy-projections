"""One-dimensional segment calibration for recentered distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.evaluation.calibration import coverage, pinball_loss

MINIMUM_N_FOR_GATE = 50
POSITIONS = ("QB", "RB", "WR", "TE")


def _spearman(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    return float(pd.Series(actual).corr(pd.Series(pred), method="spearman"))


def segment_metrics(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual_points",
    p10_col: str = "pred_p10",
    p25_col: str = "pred_p25",
    p50_col: str = "pred_p50",
    p75_col: str = "pred_p75",
    p90_col: str = "pred_p90",
) -> dict:
    actual = pd.to_numeric(frame[actual_col], errors="coerce")
    p10 = pd.to_numeric(frame[p10_col], errors="coerce")
    p25 = pd.to_numeric(frame[p25_col], errors="coerce") if p25_col in frame.columns else None
    p50 = pd.to_numeric(frame[p50_col], errors="coerce")
    p75 = pd.to_numeric(frame[p75_col], errors="coerce") if p75_col in frame.columns else None
    p90 = pd.to_numeric(frame[p90_col], errors="coerce")
    mask = actual.notna() & p10.notna() & p50.notna() & p90.notna()
    sub = frame.loc[mask]
    if sub.empty:
        return {
            "n": 0,
            "coverage_50": float("nan"),
            "coverage_80": float("nan"),
            "mean_interval_width_50": float("nan"),
            "mean_interval_width_80": float("nan"),
            "pinball_loss_q10": float("nan"),
            "pinball_loss_q50": float("nan"),
            "pinball_loss_q90": float("nan"),
            "median_absolute_error": float("nan"),
            "mean_error": float("nan"),
            "spearman_rank_correlation": float("nan"),
        }
    a = actual[mask].to_numpy(dtype=float)
    lo50 = p25[mask] if p25 is not None else p50[mask]
    hi50 = p75[mask] if p75 is not None else p50[mask]
    return {
        "n": int(len(sub)),
        "coverage_50": coverage(actual[mask], lo50, hi50),
        "coverage_80": coverage(actual[mask], p10[mask], p90[mask]),
        "mean_interval_width_50": float((hi50 - lo50).mean()),
        "mean_interval_width_80": float((p90[mask] - p10[mask]).mean()),
        "pinball_loss_q10": pinball_loss(a, 0.10, p10[mask].to_numpy(dtype=float)),
        "pinball_loss_q50": pinball_loss(a, 0.50, p50[mask].to_numpy(dtype=float)),
        "pinball_loss_q90": pinball_loss(a, 0.90, p90[mask].to_numpy(dtype=float)),
        "median_absolute_error": float(np.median(np.abs(a - p50[mask].to_numpy(dtype=float)))),
        "mean_error": float(np.mean(a - p50[mask].to_numpy(dtype=float))),
        "spearman_rank_correlation": _spearman(a, p50[mask].to_numpy(dtype=float)),
    }


def _bucket_games(games: pd.Series) -> pd.Series:
    g = pd.to_numeric(games, errors="coerce")
    return pd.cut(
        g,
        bins=[0, 10, 14, 17, np.inf],
        labels=["low_games", "mid_games", "high_games", "full_season"],
        include_lowest=True,
    ).astype(str)


def _bucket_percentile(values: pd.Series) -> pd.Series:
    ranked = values.rank(pct=True, method="average")
    return pd.cut(
        ranked,
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["p0_25", "p25_50", "p50_75", "p75_100"],
        include_lowest=True,
    ).astype(str)


def build_segment_assignments(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """One-dimensional segment labels; no Cartesian product."""
    out: dict[str, pd.Series] = {}
    if "position" in frame.columns:
        out["position"] = frame["position"].astype(str)
    if "depth_chart_status" in frame.columns:
        out["role_depth"] = frame["depth_chart_status"].fillna("unknown").astype(str)
    elif "role" in frame.columns:
        out["role_depth"] = frame["role"].fillna("unknown").astype(str)
    if "source" in frame.columns:
        out["rookie_status"] = np.where(
            frame["source"].astype(str).eq("rookie_rule"), "rookie", "veteran"
        )
    if "projected_games" in frame.columns:
        out["projected_games_bucket"] = _bucket_games(frame["projected_games"])
    if "injury_durability_rate" in frame.columns:
        frag = pd.to_numeric(frame["injury_durability_rate"], errors="coerce")
        out["fragility_bucket"] = pd.cut(
            frag.fillna(frag.median()),
            bins=[-np.inf, 0.15, 0.35, np.inf],
            labels=["durable", "average", "fragile"],
        ).astype(str)
    if "pred_p50" in frame.columns:
        out["point_percentile_bucket"] = _bucket_percentile(
            pd.to_numeric(frame["pred_p50"], errors="coerce")
        )
    if "target_share" in frame.columns:
        vol = pd.to_numeric(frame["target_share"], errors="coerce").fillna(
            pd.to_numeric(frame.get("carry_share"), errors="coerce")
        )
        out["volume_bucket"] = pd.cut(
            vol.fillna(0.0),
            bins=[-np.inf, 0.05, 0.15, 0.30, np.inf],
            labels=["minimal", "part_time", "starter", "workhorse"],
        ).astype(str)
    if "team_target_concentration" in frame.columns:
        conc = pd.to_numeric(frame["team_target_concentration"], errors="coerce")
        out["concentration_bucket"] = pd.cut(
            conc.fillna(conc.median()),
            bins=[-np.inf, 0.20, 0.35, np.inf],
            labels=["distributed", "moderate", "concentrated"],
        ).astype(str)
    return out


def evaluate_segments(
    frame: pd.DataFrame,
    *,
    segment_name: str,
    labels: pd.Series,
    minimum_n: int = MINIMUM_N_FOR_GATE,
) -> list[dict]:
    rows: list[dict] = []
    for label, grp in frame.groupby(labels, observed=True):
        metrics = segment_metrics(grp)
        rows.append({
            "segment_type": segment_name,
            "segment_value": str(label),
            "eligible_for_gate": metrics["n"] >= minimum_n,
            **metrics,
        })
    return rows


def evaluate_all_segments(
    frame: pd.DataFrame,
    *,
    minimum_n: int = MINIMUM_N_FOR_GATE,
) -> pd.DataFrame:
    """Evaluate each one-dimensional segment independently."""
    assignments = build_segment_assignments(frame)
    rows: list[dict] = []
    for name, labels in assignments.items():
        rows.extend(
            evaluate_segments(frame, segment_name=name, labels=labels, minimum_n=minimum_n)
        )
    overall = segment_metrics(frame)
    rows.insert(0, {
        "segment_type": "overall",
        "segment_value": "all",
        "eligible_for_gate": overall["n"] >= minimum_n,
        **overall,
    })
    for pos in POSITIONS:
        sub = frame[frame["position"] == pos] if "position" in frame.columns else frame.iloc[0:0]
        if sub.empty:
            continue
        metrics = segment_metrics(sub)
        rows.append({
            "segment_type": "position_gate",
            "segment_value": pos,
            "eligible_for_gate": metrics["n"] >= minimum_n,
            **metrics,
        })
    return pd.DataFrame(rows)
