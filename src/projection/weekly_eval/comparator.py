"""Join a shadow weekly board to timestamped Vegas snapshots and optional actuals.

Role 2 only: measure. Never blend market into the model. Never promote.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.weekly_eval.leakage import (
    JOIN_KEYS,
    assert_no_role3_blend,
    assert_prediction_frame_has_no_outcomes,
)
from src.projection.weekly_eval.match import describe_match, empty_match
from src.projection.weekly_eval.metrics import (
    brier_score,
    fallback_std,
    gaussian_crps,
    gaussian_p_over,
    implied_sigma,
    interval_coverage_80,
    mean_absolute_error,
    pinball_loss,
    root_mean_squared_error,
)
from src.projection.weekly_eval.schema import (
    PropSnapshot,
    load_outcomes,
    load_prop_snapshots,
    load_shadow_board,
)

SCHEMA_VERSION = "vegas_weekly_props_eval_v1"


def _as_board(board: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(board, pd.DataFrame):
        frame = board.copy()
        required = {"player_id", "season", "week", "market", "model_mean"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"shadow board is missing columns: {sorted(missing)}")
        frame["player_id"] = frame["player_id"].astype(str)
        frame["season"] = frame["season"].astype(int)
        frame["week"] = frame["week"].astype(int)
        frame["market"] = frame["market"].astype(str)
        frame["model_mean"] = pd.to_numeric(frame["model_mean"], errors="coerce")
        assert_prediction_frame_has_no_outcomes(tuple(frame.columns))
        return frame
    return load_shadow_board(board)


def _as_snapshots(
    snapshots: Sequence[PropSnapshot] | pd.DataFrame | Path | str,
) -> list[PropSnapshot]:
    if isinstance(snapshots, (pd.DataFrame, Path, str)):
        return load_prop_snapshots(snapshots)
    return list(snapshots)


def _as_outcomes(outcomes: pd.DataFrame | Path | str | None) -> pd.DataFrame | None:
    if outcomes is None:
        return None
    if isinstance(outcomes, pd.DataFrame):
        frame = outcomes.copy()
        required = {"player_id", "season", "week", "market", "actual"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"outcomes are missing columns: {sorted(missing)}")
        frame["player_id"] = frame["player_id"].astype(str)
        frame["season"] = frame["season"].astype(int)
        frame["week"] = frame["week"].astype(int)
        frame["market"] = frame["market"].astype(str)
        frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
        return frame
    return load_outcomes(outcomes)


def _snapshot_frame(snaps: Sequence[PropSnapshot]) -> pd.DataFrame:
    rows = []
    for snap in snaps:
        rows.append(
            {
                "player_id": snap.player_id,
                "season": snap.season,
                "week": snap.week,
                "market": snap.market,
                "line": snap.line,
                "implied_p_over": snap.implied_p_over,
                "implied_mean": snap.implied_mean,
                "market_mean": snap.implied_location(),
                "as_of": snap.as_of.isoformat(),
                "kickoff_at": snap.kickoff_at.isoformat(),
                "source": snap.source,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Duplicate (player, season, week, market) rows: do **not** keep a single
    # book by latest as_of (that is a one-book pick / MAE book-shop risk).
    # Collapse with the same weekly robust_median used for Role 1; per-book
    # quotes remain in the export CSV when mode=single.
    return _collapse_multi_book_snapshots(frame)


def _collapse_multi_book_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per JOIN_KEYS; multi-book → robust_median line (not pick-one)."""
    from src.projection.market_quotes import robust_median
    from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY

    policy = DEFAULT_WEEKLY_POLICY.quote
    out_rows: list[dict[str, Any]] = []
    for _, grp in frame.groupby(list(JOIN_KEYS), sort=False):
        if len(grp) == 1:
            out_rows.append(grp.iloc[0].to_dict())
            continue
        lines = [
            float(v)
            for v in grp["line"].tolist()
            if v is not None and pd.notna(v)
        ]
        means = [
            float(v)
            for v in grp["market_mean"].tolist()
            if v is not None and pd.notna(v)
        ]
        line = float(robust_median(lines, policy=policy)) if lines else None
        market_mean = (
            float(robust_median(means, policy=policy)) if means else line
        )
        sources = sorted(
            {
                str(s).strip().lower()
                for s in grp["source"].tolist()
                if s is not None and str(s).strip()
            }
        )
        # Matching Role 1 export: multi-book → "consensus"; else keep the book.
        source = "consensus" if len(sources) > 1 else (sources[0] if sources else "consensus")
        # Keep the freshest pre-kickoff as_of and earliest kickoff (already
        # fail-closed upstream); do not use as_of to pick a book line.
        as_of = max(str(v) for v in grp["as_of"].tolist())
        kickoff_at = min(str(v) for v in grp["kickoff_at"].tolist())
        p_overs = [
            float(v)
            for v in grp["implied_p_over"].tolist()
            if v is not None and pd.notna(v)
        ]
        implied_means = [
            float(v)
            for v in grp["implied_mean"].tolist()
            if v is not None and pd.notna(v)
        ]
        row0 = grp.iloc[0]
        out_rows.append(
            {
                "player_id": row0["player_id"],
                "season": row0["season"],
                "week": row0["week"],
                "market": row0["market"],
                "line": line,
                "implied_p_over": (
                    sum(p_overs) / len(p_overs) if p_overs else None
                ),
                "implied_mean": (
                    float(robust_median(implied_means, policy=policy))
                    if implied_means
                    else line
                ),
                "market_mean": market_mean,
                "as_of": as_of,
                "kickoff_at": kickoff_at,
                "source": source,
            }
        )
    return pd.DataFrame(out_rows)


def _finite_pair(pred: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if not np.any(mask):
        return None
    return pred[mask], actual[mask]


def _metric_block(joined: pd.DataFrame) -> dict[str, Any]:
    model = joined["model_mean"].to_numpy(dtype=float)
    market = joined["market_mean"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "n": len(joined),
        "mae_model_vs_market": None,
        "mae_model": None,
        "mae_market": None,
        "rmse_model": None,
        "rmse_market": None,
        "brier_model": None,
        "brier_market": None,
        "crps_gaussian_model": None,
        "crps_gaussian_market": None,
        "pinball_q50_model": None,
        "pinball_q50_market": None,
        "coverage_80_model": None,
    }
    vs_mkt = _finite_pair(model, market)
    if vs_mkt is not None:
        out["mae_model_vs_market"] = mean_absolute_error(*vs_mkt)

    if "actual" not in joined.columns:
        return out
    actual = joined["actual"].to_numpy(dtype=float)
    vs_act = _finite_pair(model, actual)
    vs_mkt_act = _finite_pair(market, actual)
    if vs_act is not None:
        p, a = vs_act
        out["mae_model"] = mean_absolute_error(p, a)
        out["rmse_model"] = root_mean_squared_error(p, a)
        out["pinball_q50_model"] = pinball_loss(a, 0.5, p)
        std = joined["model_std"].to_numpy(dtype=float)
        std_mask = np.isfinite(p) & np.isfinite(a) & np.isfinite(std)
        if np.any(std_mask):
            out["crps_gaussian_model"] = gaussian_crps(a[std_mask], p[std_mask], std[std_mask])
            out["coverage_80_model"] = interval_coverage_80(a[std_mask], p[std_mask], std[std_mask])
    if vs_mkt_act is not None:
        m, a = vs_mkt_act
        out["mae_market"] = mean_absolute_error(m, a)
        out["rmse_market"] = root_mean_squared_error(m, a)
        out["pinball_q50_market"] = pinball_loss(a, 0.5, m)
        mstd = joined["market_std"].to_numpy(dtype=float)
        m_mask = np.isfinite(m) & np.isfinite(a) & np.isfinite(mstd)
        if np.any(m_mask):
            out["crps_gaussian_market"] = gaussian_crps(a[m_mask], m[m_mask], mstd[m_mask])

    if "over" in joined.columns:
        over = joined["over"].to_numpy(dtype=float)
        mp = joined["model_p_over"].to_numpy(dtype=float)
        kp = joined["implied_p_over"].to_numpy(dtype=float)
        b_model = _finite_pair(mp, over)
        b_mkt = _finite_pair(kp, over)
        if b_model is not None:
            out["brier_model"] = brier_score(b_model[1], b_model[0])
        if b_mkt is not None:
            out["brier_market"] = brier_score(b_mkt[1], b_mkt[0])
    return out


def _enrich_joined(joined: pd.DataFrame) -> pd.DataFrame:
    model_std = []
    market_std = []
    model_p = []
    market_p = []
    overs = []
    for row in joined.itertuples(index=False):
        mean = float(row.model_mean)
        line = getattr(row, "line", None)
        line_f = None if line is None or (isinstance(line, float) and not np.isfinite(line)) else float(line)
        raw_std = getattr(row, "model_std", np.nan)
        if raw_std is None or (isinstance(raw_std, float) and not np.isfinite(raw_std)):
            std = fallback_std(mean)
        else:
            std = max(float(raw_std), 1e-9)
        model_std.append(std)
        m_mean = float(row.market_mean)
        m_p = getattr(row, "implied_p_over", None)
        m_p_f = None if m_p is None or (isinstance(m_p, float) and not np.isfinite(m_p)) else float(m_p)
        market_std.append(implied_sigma(line=line_f, implied_mean=m_mean, implied_p_over=m_p_f))
        raw_p = getattr(row, "model_p_over", np.nan) if hasattr(row, "model_p_over") else np.nan
        if raw_p is None or (isinstance(raw_p, float) and not np.isfinite(raw_p)):
            model_p.append(gaussian_p_over(mean, std, line_f) if line_f is not None else float("nan"))
        else:
            model_p.append(float(raw_p))
        if m_p_f is not None:
            market_p.append(m_p_f)
        elif line_f is not None:
            market_p.append(0.5)
        else:
            market_p.append(float("nan"))
        if "actual" in joined.columns:
            actual = getattr(row, "actual", np.nan)
            if line_f is not None and actual is not None and np.isfinite(actual):
                overs.append(1.0 if float(actual) > line_f else 0.0)
            else:
                overs.append(float("nan"))
    joined = joined.copy()
    joined["model_std"] = model_std
    joined["market_std"] = market_std
    joined["model_p_over"] = model_p
    joined["implied_p_over"] = market_p
    if "actual" in joined.columns:
        joined["over"] = overs
    return joined


def compare_shadow_to_vegas(
    *,
    board: pd.DataFrame | Path | str,
    snapshots: Sequence[PropSnapshot] | pd.DataFrame | Path | str,
    outcomes: pd.DataFrame | Path | str | None = None,
    blend_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score shadow player-week means against timestamped props and optional actuals."""
    assert_no_role3_blend(blend_weights)
    board_df = _as_board(board)
    snaps = _as_snapshots(snapshots)
    snap_df = _snapshot_frame(snaps)
    if snap_df.empty:
        joined = board_df.iloc[0:0].copy()
    else:
        joined = board_df.merge(snap_df, on=list(JOIN_KEYS), how="inner")
    outcomes_df = _as_outcomes(outcomes)
    n_with_actuals = 0
    if outcomes_df is not None and not outcomes_df.empty and not joined.empty:
        joined = joined.merge(
            outcomes_df[list(JOIN_KEYS) + ["actual"]],
            on=list(JOIN_KEYS),
            how="left",
        )
        n_with_actuals = int(joined["actual"].notna().sum())
        if n_with_actuals == 0:
            joined = joined.drop(columns=["actual"])
    if not joined.empty:
        joined = _enrich_joined(joined)
    match = (
        describe_match(board_df, snap_df, joined)
        if not snap_df.empty
        else empty_match(board=board_df, snapshots=snap_df)
    )
    metrics = _metric_block(joined) if not joined.empty else {
        "n": 0,
        "mae_model_vs_market": None,
        "mae_model": None,
        "mae_market": None,
        "rmse_model": None,
        "rmse_market": None,
        "brier_model": None,
        "brier_market": None,
        "crps_gaussian_model": None,
        "crps_gaussian_market": None,
        "pinball_q50_model": None,
        "pinball_q50_market": None,
        "coverage_80_model": None,
    }
    by_market: dict[str, Any] = {}
    if not joined.empty:
        for market, grp in joined.groupby("market", sort=True):
            by_market[str(market)] = _metric_block(grp)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "evaluation_comparator",
        "role3_blend": False,
        "promoting": False,
        "gate_verdict": "not_promoting",
        "n_board": len(board_df),
        "n_snapshots": len(snaps),
        "n_matched": len(joined),
        "n_with_actuals": n_with_actuals,
        "match": match,
        "metrics": metrics,
        "by_market": by_market,
        "leakage": {
            "as_of_required": True,
            "kickoff_at_required": True,
            "post_kickoff_rejected": True,
            "same_week_outcome_columns_rejected": True,
            "role3_blend_rejected": True,
        },
        "metrics_notes": {
            "mae_rmse": "Point losses vs realized actuals (and model vs market as a diagnostic, not a target).",
            "brier": "Proper score for the binary over/under relative to the snapshot line.",
            "crps_gaussian": "Proper score for a Gaussian with mean + std (std from column or CV fallback).",
            "pinball_q50": "Quantile loss at the median; calibration-friendly; equals half of MAE when the median is the mean.",
            "coverage_80": "Share of actuals inside the Gaussian p10-p90 interval.",
        },
        "caveat": (
            "Harness only. Not a promotion sample. Do not optimize toward market "
            "agreement or treat this output as weekly accuracy evidence."
        ),
    }
