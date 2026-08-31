"""Measure the QB team-volume-share change one factor at a time.

The shipped board moved two things at once in the same uncommitted batch:
``TEAM_VOLUME_SHARES[("QB", ...)]`` went 0.941/0.942 -> 1.000, and pass/rush
TDs were detached from volume scaling by emptying their entries in
``TEAM_VOLUME_SIBLINGS``. Both land on QB, so a board-level comparison cannot
say which one earned its keep -- and two changes on one position is exactly
the shape where a pair of errors can cancel and read as an improvement.

This scores each factor on held-out seasons against actual fantasy points,
varying ONE at a time from the shipped configuration.

Usage:
    python scripts/ablate_qb_volume_share.py --seasons 2024,2025
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.projection.contracts import (
    QB_STARTER_VOLUME_SHARES,
    TEAM_VOLUME_SHARES,
    TEAM_VOLUME_SIBLINGS,
)
from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features
from src.projection.fantasy_evaluation import (
    attach_actual_outcomes,
    build_leakage_safe_forecasts,
)

OUT_PATH = ROOT / "output" / "ablation_qb_volume_share.json"


def _shares_with_starter_qb() -> dict:
    """Pre-change shares: QB claims the measured STARTER fraction."""
    shares = dict(TEAM_VOLUME_SHARES)
    for stat, frac in QB_STARTER_VOLUME_SHARES.items():
        key = ("QB", stat)
        if key in shares:
            anchor_col, _ = shares[key]
            shares[key] = (anchor_col, float(frac))
    return shares


def _siblings_with_tds() -> dict:
    """Pre-change siblings: TDs ride the volume scale again."""
    siblings = dict(TEAM_VOLUME_SIBLINGS)
    qb = tuple(siblings.get(("QB", "attempts"), ()))
    if "passing_tds" not in qb:
        siblings[("QB", "attempts")] = qb + ("passing_tds",)
    rb = tuple(siblings.get(("RB", "carries"), ()))
    if "rushing_tds" not in rb:
        siblings[("RB", "carries")] = rb + ("rushing_tds",)
    return siblings


# One variant per factor. `None` means "use the shipped contract", so each
# row differs from `shipped` in exactly one place.
VARIANTS = {
    "shipped": {},
    "qb_share_starter": {"team_volume_shares": _shares_with_starter_qb},
    "tds_ride_volume": {"team_volume_siblings": _siblings_with_tds},
    "alpha_050": {"reconcile_alpha": 0.5},
    # Both pre-change factors together: the configuration before this batch.
    "pre_change_both": {
        "team_volume_shares": _shares_with_starter_qb,
        "team_volume_siblings": _siblings_with_tds,
    },
}


def _resolve(spec: dict) -> dict:
    return {k: (v() if callable(v) else v) for k, v in spec.items()}


def _score(frame: pd.DataFrame, position: str | None = None) -> dict:
    sub = frame
    if position:
        col = "preseason_position" if "preseason_position" in frame.columns else "position"
        sub = frame[frame[col] == position]
    sub = sub.dropna(subset=["actual_points", "model_points_end_to_end"])
    if sub.empty:
        return {"n": 0}
    actual = pd.to_numeric(sub["actual_points"], errors="coerce")
    pred = pd.to_numeric(sub["model_points_end_to_end"], errors="coerce")
    return {
        "n": int(len(sub)),
        "mae": float(np.mean(np.abs(actual - pred))),
        "spearman": float(actual.corr(pred, method="spearman")),
        "bias": float(np.mean(pred - actual)),
    }


def _abs_errors(frame: pd.DataFrame, position: str | None = None) -> pd.Series:
    """Per-player absolute error, indexed by player_id."""
    sub = frame
    if position:
        col = "preseason_position" if "preseason_position" in frame.columns else "position"
        sub = frame[frame[col] == position]
    sub = sub.dropna(subset=["actual_points", "model_points_end_to_end"])
    err = (
        pd.to_numeric(sub["model_points_end_to_end"], errors="coerce")
        - pd.to_numeric(sub["actual_points"], errors="coerce")
    ).abs()
    return pd.Series(err.to_numpy(), index=sub["player_id"].astype(str).to_numpy())


def _paired(base: pd.Series, variant: pd.Series) -> dict:
    """Paired comparison on the players both arms scored.

    Every arm forecasts the same frozen population, so the arms are paired
    and an unpaired summary throws away most of the power: player-to-player
    spread in fantasy points dwarfs the effect being measured.
    """
    common = base.index.intersection(variant.index)
    if len(common) < 3:
        return {"n_pairs": int(len(common))}
    diff = (variant.loc[common] - base.loc[common]).to_numpy(dtype=float)
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    out = {
        "n_pairs": int(n),
        "mean_abs_error_delta": mean,
        "se": float(se),
        "t": float(mean / se) if se else float("nan"),
        "share_improved": float(np.mean(diff < 0)),
        "ci95_low": float(mean - 1.96 * se) if se else mean,
        "ci95_high": float(mean + 1.96 * se) if se else mean,
    }
    try:
        from scipy.stats import wilcoxon
        if np.any(diff != 0):
            out["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
    except Exception:
        pass
    return out


def run_season(conn, feat, source_season: int, target_season: int) -> dict:
    results = {}
    errors = {}
    for name, spec in VARIANTS.items():
        print(f"  {target_season}: {name}...", flush=True)
        forecasts, _ = build_leakage_safe_forecasts(
            conn, feat,
            source_season=source_season,
            target_season=target_season,
            **_resolve(spec),
        )
        # Outcomes are attached only after the forecast is built, using the
        # shipped path's own helper so every arm is scored the same way.
        forecasts = attach_actual_outcomes(forecasts, feat, target_season)
        results[name] = {
            "overall": _score(forecasts),
            "QB": _score(forecasts, "QB"),
            "RB": _score(forecasts, "RB"),
        }
        errors[name] = {
            "overall": _abs_errors(forecasts),
            "QB": _abs_errors(forecasts, "QB"),
            "RB": _abs_errors(forecasts, "RB"),
        }
    for name in VARIANTS:
        if name == "shipped":
            continue
        results[name]["paired_vs_shipped"] = {
            scope: _paired(errors["shipped"][scope], errors[name][scope])
            for scope in ("overall", "QB", "RB")
        }
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2024,2025",
                        help="Comma-separated target seasons (default 2024,2025)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    targets = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]

    conn = get_conn()
    feat = build_player_season_features(conn)
    seasons = {}
    all_errors = {}
    for target in targets:
        print(f"Season {target}", flush=True)
        seasons[str(target)], all_errors[target] = run_season(
            conn, feat, target - 1, target)
    conn.close()

    # Pool paired differences across seasons. One season of ~106 QBs cannot
    # resolve a sub-1% MAE effect; pooling the per-player deltas is the only
    # way these arms get enough power to say anything.
    pooled = {}
    for name in VARIANTS:
        if name == "shipped":
            continue
        pooled[name] = {}
        for scope in ("overall", "QB", "RB"):
            base_parts, var_parts = [], []
            for target in targets:
                # Disambiguate players who appear in more than one season.
                b = all_errors[target]["shipped"][scope]
                v = all_errors[target][name][scope]
                base_parts.append(b.rename(lambda p, s=target: f"{s}:{p}"))
                var_parts.append(v.rename(lambda p, s=target: f"{s}:{p}"))
            pooled[name][scope] = _paired(
                pd.concat(base_parts), pd.concat(var_parts))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Each variant differs from `shipped` in exactly one factor, except "
            "pre_change_both which reverts the two that shipped together. "
            "share_improved is the fraction of players the variant helps: a "
            "value near 0.5 with a negative mean delta means error was "
            "redistributed, not broadly reduced."
        ),
        "seasons": seasons,
        "pooled_paired_vs_shipped": pooled,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}\n")

    for season, variants in seasons.items():
        base = variants["shipped"]
        print(f"{season}  (delta vs shipped; negative MAE delta = better)")
        for name, res in variants.items():
            if name == "shipped":
                print(f"  {name:18s} QB MAE {res['QB'].get('mae', float('nan')):7.3f}"
                      f"   overall MAE {res['overall'].get('mae', float('nan')):7.3f}")
                continue
            dq = res["QB"].get("mae", float("nan")) - base["QB"].get("mae", float("nan"))
            do = res["overall"].get("mae", float("nan")) - base["overall"].get("mae", float("nan"))
            pq = (res.get("paired_vs_shipped") or {}).get("QB", {})
            print(f"  {name:18s} QB MAE {res['QB'].get('mae', float('nan')):7.3f} "
                  f"({dq:+.3f})   overall MAE {res['overall'].get('mae', float('nan')):7.3f} ({do:+.3f})"
                  f"   QB paired t={pq.get('t', float('nan')):+.2f}"
                  f" p={pq.get('wilcoxon_p', float('nan')):.3f}"
                  f" improved={pq.get('share_improved', float('nan')):.0%}")
        print()

    print(f"POOLED across {', '.join(map(str, targets))} (paired, vs shipped)")
    for scope in ("QB", "overall"):
        print(f"  -- {scope} --")
        for name, scopes in pooled.items():
            p = scopes[scope]
            print(f"    {name:18s} n={p.get('n_pairs', 0):4d}"
                  f"  mean_delta={p.get('mean_abs_error_delta', float('nan')):+7.3f}"
                  f"  95%CI[{p.get('ci95_low', float('nan')):+.2f},"
                  f"{p.get('ci95_high', float('nan')):+.2f}]"
                  f"  p={p.get('wilcoxon_p', float('nan')):.3f}"
                  f"  improved={p.get('share_improved', float('nan')):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
