"""Measure projection accuracy for the deep board, outside the top-120 ADP population.

The accuracy-first ensemble (``src/projection/evaluation/accuracy_first.py``) is fit,
selected and applied only where ``adp <= 120``.  Its objective string is literally
"2026 top-120 ADP half-PPR point MAE".  Nothing in that work says whether the deep
board -- the players a sleeper view is drawn from -- is projected well enough to
rank at all.  This script answers that separately.

Two measurement choices carry the result and are deliberate:

* Bands are cut on **projected season-points rank**, not the board's ``overall_rank``.
  ``overall_rank`` is VORP-based and diverges hard from points order (Spearman 0.80,
  with 450 of 778 players more than 50 places apart), and the historical evaluation
  frames carry no VORP.  Banding on points here and selecting on points downstream
  keeps the measured path and the shipped path identical.

* The admission test is a **hit rate**, not MAE or Spearman.  Both of those improve
  as a band approaches the zero floor: the tail band posts the lowest MAE and the
  highest Spearman of any band while producing a startable season 2% of the time.
  That is denominator dilution plus rank noise around a floor, not skill.
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
from scipy.stats import spearmanr

from src.projection.contracts import OUTPUT_DIR

SEASONS = (2023, 2024, 2025)
POSITIONS = ("QB", "RB", "WR", "TE")

# Band edges on projected season-points rank, inclusive.
BANDS = (
    (1, 120, "top120"),
    (121, 200, "deep_primary"),
    (201, 300, "deep_speculative"),
    (301, 10_000, "tail"),
)

# A band earns a place on the sleeper board if a real share of it produces a
# startable half-PPR season.  100 points is roughly a flex-worthy floor over a
# full year; 150 is a genuine starter.
STARTABLE_POINTS = 100.0
STARTER_POINTS = 150.0
MIN_STARTABLE_RATE = 0.15

# Cells thinner than this are reported but must not be read as a group gap.
MIN_CELL_N = 30


def load_evaluation_frames(seasons=SEASONS) -> pd.DataFrame:
    """Pool the per-player evaluation frames and rank each season by projected points."""
    frames = []
    for season in seasons:
        path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{season}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing evaluation frame: {path}")
        frame = pd.read_csv(path)
        frame["season"] = season
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)

    scored = pooled[
        pooled["model_forecast_points"].notna() & pooled["actual_points"].notna()
    ].copy()
    scored["proj_points"] = pd.to_numeric(scored["model_forecast_points"], errors="coerce")
    scored["actual"] = pd.to_numeric(scored["actual_points"], errors="coerce")
    # Rank within season so a band means the same thing every year.
    scored["proj_rank"] = (
        scored.groupby("season")["proj_points"].rank(ascending=False, method="first")
    )
    return scored


def band_of(rank: float) -> str:
    for low, high, name in BANDS:
        if low <= rank <= high:
            return name
    return "unranked"


def metrics(frame: pd.DataFrame) -> dict:
    """Point accuracy, rank agreement and outcome hit rates for one cell."""
    n = int(len(frame))
    if n == 0:
        return {"n": 0}
    proj = frame["proj_points"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    if n >= 3 and np.std(proj) > 0 and np.std(actual) > 0:
        rho, p_value = spearmanr(proj, actual)
    else:
        rho, p_value = float("nan"), float("nan")
    return {
        "n": n,
        "points_mae": round(float(np.abs(proj - actual).mean()), 4),
        "spearman": None if np.isnan(rho) else round(float(rho), 4),
        "spearman_p": None if np.isnan(p_value) else round(float(p_value), 6),
        "mean_projected": round(float(proj.mean()), 3),
        "mean_actual": round(float(actual.mean()), 3),
        "p_startable_100": round(float((actual >= STARTABLE_POINTS).mean()), 4),
        "p_starter_150": round(float((actual >= STARTER_POINTS).mean()), 4),
        "thin_cell": n < MIN_CELL_N,
    }


def build_report(scored: pd.DataFrame) -> dict:
    scored = scored.copy()
    scored["band"] = scored["proj_rank"].map(band_of)

    bands = {}
    for low, high, name in BANDS:
        cell = scored[scored["band"] == name]
        block = metrics(cell)
        block["rank_range"] = [low, None if high >= 10_000 else high]
        block["per_season_n"] = {
            str(season): int((cell["season"] == season).sum()) for season in SEASONS
        }
        block["by_position"] = {
            position: metrics(cell[cell["preseason_position"] == position])
            for position in POSITIONS
        }
        block["admitted"] = bool(
            block.get("n", 0) >= MIN_CELL_N
            and block.get("p_startable_100", 0.0) >= MIN_STARTABLE_RATE
        )
        bands[name] = block

    admitted = [
        name for name, block in bands.items() if block["admitted"] and name != "top120"
    ]
    sleeper_low = min((low for low, _, n in BANDS if n in admitted), default=None)
    sleeper_high = max(
        (high for _, high, n in BANDS if n in admitted and high < 10_000), default=None
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(SEASONS),
        "population": (
            "player-seasons from output/fantasy_evaluation_<season>.csv with both a "
            "model forecast and a realised outcome"
        ),
        "n_player_seasons": int(len(scored)),
        "banding": "projected season-points rank within season (NOT board overall_rank)",
        "admission_rule": {
            "metric": "p_startable_100",
            "threshold": MIN_STARTABLE_RATE,
            "min_n": MIN_CELL_N,
            "rationale": (
                "MAE and Spearman both improve as a band approaches the zero floor and "
                "cannot separate skill from a dead tail; the hit rate can."
            ),
        },
        "bands": bands,
        "sleeper_band": {
            "admitted_bands": admitted,
            "proj_rank_min": sleeper_low,
            "proj_rank_max": sleeper_high,
        },
        "caveats": [
            "The top-120 accuracy-first ensemble does not cover any band below top120; "
            "players outside the top 120 ADP carry the untouched incumbent forecast.",
            "The tail band posts the lowest MAE and often the highest Spearman of any "
            "band while almost never producing a startable season. Its MAE is deflated "
            "by near-zero actuals and its Spearman is rank noise around a floor. Do not "
            "read either as evidence the tail is well projected.",
            "Bands are cut on projected points rank so that measurement and board "
            "selection use the same ordering; the board's own overall_rank is VORP-based "
            "and is NOT interchangeable with it.",
            f"Cells with n < {MIN_CELL_N} are marked thin_cell and must not be read as a "
            "group difference.",
        ],
    }


def format_table(report: dict) -> str:
    header = (
        f"{'band':<18} {'ranks':>10} {'n':>5} {'MAE':>8} {'rho':>7} "
        f"{'P(>=100)':>9} {'P(>=150)':>9} {'admitted':>9}"
    )
    lines = [header, "-" * len(header)]
    for _, _, name in BANDS:
        block = report["bands"][name]
        low, high = block["rank_range"]
        ranks = f"{low}-{high}" if high else f"{low}+"
        rho = "n/a" if block.get("spearman") is None else f"{block['spearman']:.3f}"
        lines.append(
            f"{name:<18} {ranks:>10} {block['n']:>5} {block['points_mae']:>8.2f} "
            f"{rho:>7} {100 * block['p_startable_100']:>8.1f}% "
            f"{100 * block['p_starter_150']:>8.1f}% {str(block['admitted']):>9}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(Path(OUTPUT_DIR) / "deep_band_accuracy"))
    parser.add_argument(
        "--board-out",
        default=str(ROOT / "draft_assistant" / "data" / "deep_band_accuracy.json"),
        help="copy the report where the static draft_assistant server can fetch it",
    )
    args = parser.parse_args()

    scored = load_evaluation_frames()
    report = build_report(scored)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # The Sleepers view states the band's measured hit rate on the page itself, and
    # the static server is rooted at draft_assistant/, so publish a copy it can fetch.
    board_path = Path(args.board_out)
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"pooled player-seasons: {report['n_player_seasons']}")
    print(format_table(report))
    print()
    band = report["sleeper_band"]
    print(
        f"admitted sleeper bands: {', '.join(band['admitted_bands']) or 'none'} "
        f"-> projected points rank {band['proj_rank_min']}-{band['proj_rank_max']}"
    )
    print(f"wrote {out_path}")
    print(f"wrote {board_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
