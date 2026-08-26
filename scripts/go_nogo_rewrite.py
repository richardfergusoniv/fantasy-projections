"""Apply go/no-go rewrite gate from Phase T1–T4 artifacts.

Does not rewrite the model. Reads freeze / market-edge / rolling-eval /
ensemble outputs and writes a decision record.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "test_before_rewrite"
DOCS_DIR = REPO_ROOT / "docs" / "decisions"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def decide(artifacts: dict) -> dict:
    market = artifacts.get("market_edge") or {}
    actionable = market.get("actionable_summary") or {}
    ensemble = artifacts.get("ensemble") or {}
    rolling = artifacts.get("rolling_eval") or {}
    seasons = market.get("seasons") or {}

    def _edge_corr(season_block: dict, label: str) -> float | None:
        edge = ((season_block.get("models") or {}).get(label) or {}).get("edge") or {}
        val = edge.get("edge_corr_residual_vs_actual_points")
        return float(val) if val is not None else None

    # Seasons where label has points-edge AND beats carry_forward's edge corr
    # (more negative = stronger actionable residual after controlling for ADP).
    beats_carry: dict[str, list[int]] = {}
    for label in ("v1", "v2", "blend"):
        hits: list[int] = []
        for season, block in seasons.items():
            if block.get("error"):
                continue
            model_c = _edge_corr(block, label)
            carry_c = _edge_corr(block, "carry_forward")
            edge = ((block.get("models") or {}).get(label) or {}).get("edge") or {}
            if not edge.get("actionable_points_edge") or model_c is None:
                continue
            if carry_c is None or model_c < carry_c - 0.02:
                hits.append(int(season))
        beats_carry[label] = hits

    multi_season_beat_carry = {
        label: seasons_ for label, seasons_ in beats_carry.items() if len(seasons_) >= 2
    }

    blend_holdout = ((ensemble.get("holdout_accuracy") or {}).get("blend") or {}).get(
        "overall"
    ) or {}
    v1_holdout = ((ensemble.get("holdout_accuracy") or {}).get("v1") or {}).get(
        "overall"
    ) or {}
    v2_holdout = ((ensemble.get("holdout_accuracy") or {}).get("v2") or {}).get(
        "overall"
    ) or {}

    blend_helps = False
    if blend_holdout and v1_holdout and v2_holdout:
        blend_mae = blend_holdout.get("points_mae")
        singles = [
            x
            for x in (v1_holdout.get("points_mae"), v2_holdout.get("points_mae"))
            if x is not None
        ]
        if blend_mae is not None and singles:
            blend_helps = blend_mae <= min(singles) + 0.25

    if not multi_season_beat_carry:
        verdict = "do_not_rewrite"
        rationale = (
            "No model’s ADP residual beats carry-forward across multiple seasons. "
            "Keep the engine; the board remains a ranking aid, not a proven "
            "mispricing system beyond a naive baseline."
        )
    elif "blend" in multi_season_beat_carry and blend_helps:
        verdict = "do_not_rewrite"
        rationale = (
            "v1/v2 post-process blend shows multi-season ADP residual edge that "
            "beats carry-forward and improves 2025 holdout MAE/Spearman. Ship the "
            "ensemble as a draft-assistant post-process; do not rewrite compose_board "
            "or LightGBM. Revisit targeted levers only if residual remains after "
            "the blend is in production use."
        )
    elif multi_season_beat_carry:
        verdict = "targeted_engine_changes_only"
        rationale = (
            f"Multi-season ADP residual (beating carry) for: "
            f"{list(multi_season_beat_carry)}. Blend did not fully absorb the lift "
            "on holdout — consider the smallest causal lever (availability / "
            "rookies / team-volume), not a generative or weekly rewrite."
        )
    else:
        verdict = "do_not_rewrite"
        rationale = "Default conservative gate: no rewrite without clear multi-season edge."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "rationale": rationale,
        "signals": {
            "seasons_edge_beats_carry": beats_carry,
            "multi_season_beat_carry": multi_season_beat_carry,
            "actionable_summary_raw": actionable,
            "blend_helps_holdout_mae": blend_helps,
            "holdout_mae": {
                "v1": v1_holdout.get("points_mae"),
                "v2": v2_holdout.get("points_mae"),
                "blend": blend_holdout.get("points_mae"),
            },
            "holdout_spearman": {
                "v1": v1_holdout.get("spearman"),
                "v2": v2_holdout.get("spearman"),
                "blend": blend_holdout.get("spearman"),
            },
            "rolling_dispersion": rolling.get("dispersion_summary"),
            "untestable_levers": (rolling.get("metadata") or {}).get(
                "untestable_on_history"
            ),
        },
        "rules_applied": [
            "Require ADP residual edge to beat carry-forward (not merely nonzero)",
            "No full rewrite if blend captures multi-season edge and helps holdout",
            "Targeted changes only if residual persists after blend",
            "Full generative/weekly/props rewrite only after explicit product decision",
        ],
    }


def write_markdown(decision: dict, path: Path) -> None:
    lines = [
        "# Test-before-rewrite go/no-go",
        "",
        f"**Generated:** {decision['generated_at']}",
        "",
        f"**Verdict:** `{decision['verdict']}`",
        "",
        decision["rationale"],
        "",
        "## Signals",
        "",
        "```json",
        json.dumps(decision["signals"], indent=2, default=str),
        "```",
        "",
        "## Rules applied",
        "",
    ]
    for rule in decision["rules_applied"]:
        lines.append(f"- {rule}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "go_nogo.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=DOCS_DIR / "TEST_BEFORE_REWRITE_2026-08-24.md",
    )
    args = parser.parse_args()

    artifacts = {
        "freeze": _load(OUT_DIR / "freeze_manifest.json"),
        "market_edge": _load(OUT_DIR / "market_edge_backtest.json"),
        "rolling_eval": _load(OUT_DIR / "rolling_eval_dispersion.json"),
        "ensemble": _load(OUT_DIR / "ensemble_report.json"),
    }
    decision = decide(artifacts)
    decision["artifact_presence"] = {k: v is not None for k, v in artifacts.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    write_markdown(decision, args.md_out)
    print(json.dumps({"verdict": decision["verdict"], "rationale": decision["rationale"]}, indent=2))
    print(f"Wrote {args.out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
