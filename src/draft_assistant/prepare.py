"""Export projection CSV to draft-assistant JSON with tiers and metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.draft_assistant.draft_value_simulation import (
    FINISH_CUTOFFS,
    compute_finish_probabilities,
)
from src.draft_assistant.positional_ranks import simulation_rank_metadata
from src.draft_assistant.tiers import (
    DEFAULT_TIER_GAPS,
    FLEX_TIER_GAP,
    TierConfig,
    add_tier_columns,
)
from src.projection.evaluation.release_report import (
    build_release_report_board,
    merge_release_reports,
    write_merged_release_report,
    write_release_report_board,
)
from src.projection.evaluation.finish_probability_gate import (
    VERDICT_READY as FINISH_PROBABILITY_READY,
    read_finish_probability_gate,
    validate_finish_probability_publication,
)
from src.projection.evaluation.simulated_vorp_gate import (
    VERDICT_READY as SIM_VORP_READY,
    gate_output_dir,
    read_simulated_vorp_gate,
    validate_simulated_vorp_publication,
)
from src.draft_assistant.replacement_contract import (
    default_selected_board_path,
    read_replacement_contract,
)
from src.projection.inference.recenter import sha256_file
from src.sentiment.markdown import RESEARCH_AS_OF
from src.sentiment.snapshot import SENTIMENT_VERSION, attach_sentiment
from src.draft_assistant.vorp import (
    DEFAULT_TEAM_COUNT,
    load_position_curves,
    FLEX_SHARE,
    OVERALL_VORP_TIER_GAP,
    ROOKIE_RANK_SCALE,
    STARTERS,
    add_vorp_columns,
    replacement_ranks,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DRAFT_DATA_DIR = os.path.join(REPO_ROOT, "draft_assistant", "data")
DEFAULT_ENSEMBLE_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ensemble_weights.json"
)
MODEL_V3_DIR = os.path.join(OUTPUT_DIR, "model_v3")

FINISH_PROBABILITY_COLS = [f"p_finish_top{cutoff}" for cutoff in FINISH_CUTOFFS]
SIM_VORP_COLS = [
    "sim_vorp_p10",
    "sim_vorp_p50",
    "sim_vorp_p90",
    "p_vorp_positive",
    "expected_pos_rank",
    "median_pos_rank",
]

EXPORT_COLS = [
    "player_id",
    "display_name",
    "position",
    "team",
    "fantasy_pts",
    "fantasy_pts_low",
    "fantasy_pts_high",
    "fantasy_pts_season",
    "projected_games",
    "source",
    "low_confidence",
    "role",
    "depth_chart_status",
    "vorp",
    "vorp_input_pts",
    "rookie_rank_scale",
    "replacement_pts",
    "vorp_curve_weight",
    "overall_rank",
    "overall_tier",
    "pos_rank",
    "pos_tier",
    "flex_rank",
    "flex_tier",
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_coverage",
    "sentiment_as_of",
    "sentiment_claim_count",
    "sentiment_source_count",
    "sentiment_model_active",
    "sentiment_version",
    "sentiment_tone",
    "sentiment_peer_label",
    "sentiment_evidence_tier",
    "fantasy_pts_p10",
    "fantasy_pts_p25",
    "fantasy_pts_p50",
    "fantasy_pts_p75",
    "fantasy_pts_p90",
    "p_finish_top6",
    "p_finish_top12",
    "p_finish_top24",
    "p_finish_top36",
    "p_finish_top48",
    "sim_vorp_p10",
    "sim_vorp_p50",
    "sim_vorp_p90",
    "p_vorp_positive",
    "expected_pos_rank",
    "median_pos_rank",
    "volatility_flag",
]


def load_projections(season: int, path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(OUTPUT_DIR, f"fantasy_points_{season}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing projection file: {path}")
    df = pd.read_csv(path)
    if "sentiment_tone" not in df.columns:
        as_of = RESEARCH_AS_OF.isoformat()
        if "sentiment_as_of" in df.columns:
            values = df["sentiment_as_of"].dropna().astype(str).unique().tolist()
            if len(values) > 1:
                raise ValueError(f"Mixed sentiment_as_of values in {path}: {values}")
            if values:
                as_of = values[0][:10]
        df = attach_sentiment(df, season=season, as_of=as_of)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df = df.sort_values("fantasy_pts_season", ascending=False).reset_index(drop=True)
    return df


def to_json_value(val, *, as_bool: bool = False):
    """Convert a pandas/scalar value to strict JSON-compatible Python types."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    if as_bool:
        return bool(val)
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 2)
    return str(val)


def build_sentiment_meta(
    season: int,
    df: pd.DataFrame,
    *,
    generated_at: str,
) -> dict:
    """Release metadata for diagnostic sentiment presentation."""
    consensus_path = os.path.join(REPO_ROOT, "data", "consensus", f"consensus_{season}.json")
    ecr_date = None
    adp_end_date = None
    if os.path.exists(consensus_path):
        with open(consensus_path, encoding="utf-8") as fh:
            consensus = json.load(fh)
        meta = consensus.get("meta") or {}
        ecr_date = (meta.get("ecr") or {}).get("scrape_date")
        adp_end_date = (meta.get("adp") or {}).get("end_date")

    if "sentiment_as_of" in df.columns:
        as_of_values = df["sentiment_as_of"].dropna().astype(str).unique().tolist()
        if len(as_of_values) > 1:
            raise ValueError(
                f"Mixed sentiment_as_of values in release frame: {as_of_values}"
            )
        research_cutoff = as_of_values[0][:10] if as_of_values else RESEARCH_AS_OF.isoformat()
    else:
        research_cutoff = RESEARCH_AS_OF.isoformat()

    return {
        "status": "diagnostic",
        "version": SENTIMENT_VERSION,
        "research_evidence_cutoff": research_cutoff,
        "ecr_date": ecr_date,
        "adp_end_date": adp_end_date,
        "generated_at": generated_at,
        "model_active": bool(df.get("sentiment_model_active", pd.Series(dtype=bool)).any()),
    }


def build_player_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        rec = {}
        for col in EXPORT_COLS:
            raw = getattr(row, col, None)
            if col == "low_confidence":
                rec[col] = to_json_value(raw, as_bool=True) or False
            elif col in (
                "overall_rank",
                "overall_tier",
                "pos_rank",
                "pos_tier",
                "flex_rank",
                "flex_tier",
            ):
                rec[col] = int(raw) if pd.notna(raw) else None
            elif col in ("vorp", "replacement_pts"):
                rec[col] = to_json_value(raw)
            else:
                rec[col] = to_json_value(raw)
        records.append(rec)
    return records


def tier_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"overall": {}, "by_position": {}}
    overall_sorted = df.sort_values("vorp", ascending=False)
    for tier, group in overall_sorted.groupby("overall_tier", sort=False):
        summary["overall"][str(int(tier))] = {
            "count": int(len(group)),
            "top": group.sort_values("vorp", ascending=False).iloc[0]["display_name"],
            "vorp_range": [
                round(float(group["vorp"].max()), 2),
                round(float(group["vorp"].min()), 2),
            ],
        }
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df["position"] == pos]
        summary["by_position"][pos] = {}
        for tier, group in pos_df.groupby("pos_tier"):
            top = group.sort_values("vorp", ascending=False).iloc[0]
            summary["by_position"][pos][str(int(tier))] = {
                "count": int(len(group)),
                "top": top["display_name"],
                "vorp_range": [
                    round(float(group["vorp"].max()), 2),
                    round(float(group["vorp"].min()), 2),
                ],
            }
    flex_df = df[df["position"].isin(["RB", "WR", "TE"])]
    summary["flex"] = {}
    for tier, group in flex_df.groupby("flex_tier"):
        summary["flex"][str(int(tier))] = {
            "count": int(len(group)),
            "top": group.sort_values("vorp", ascending=False).iloc[0]["display_name"],
            "vorp_range": [
                round(float(group["vorp"].max()), 2),
                round(float(group["vorp"].min()), 2),
            ],
        }
    return summary


def load_ensemble_weights(path: str | None) -> dict | None:
    """Load position blend weights (v1/v2). Does not alter compose_board."""
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ensemble weights: {path}")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("weights") or payload


def accuracy_ensemble_metadata(df: pd.DataFrame) -> dict | None:
    """Describe a pre-blended accuracy-first board when one is supplied."""
    if "accuracy_ensemble_applied" not in df.columns:
        return None
    applied = df["accuracy_ensemble_applied"].fillna(False).astype(bool)
    if not applied.any():
        return None
    arms = (
        df.loc[applied, "accuracy_ensemble_arm"].fillna("unknown").astype(str).value_counts().to_dict()
        if "accuracy_ensemble_arm" in df.columns else {}
    )
    return {
        "applied": True,
        "n_players": int(applied.sum()),
        "arms": {str(key): int(value) for key, value in arms.items()},
        "note": "Accuracy-first top-120 ADP point blend; canonical v1 engine unchanged",
    }


def default_v2_points_path(season: int) -> str:
    return os.path.join(OUTPUT_DIR, "model_v2", f"fantasy_points_{season}.csv")


def apply_ensemble_points(
    df: pd.DataFrame,
    weights: dict,
    *,
    v2_points_path: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Blend native board points with a v2 fantasy_points CSV.

    Returns ``(frame, applied)``. When v2 points are absent, returns the
    frame unchanged and ``applied=False``.
    """
    if not v2_points_path or not os.path.exists(v2_points_path):
        return df, False
    v2 = pd.read_csv(v2_points_path)
    v2 = v2[v2["position"].isin(["QB", "RB", "WR", "TE"])][
        ["player_id", "fantasy_pts_season"]
    ].rename(columns={"fantasy_pts_season": "v2_pts"})
    v2["player_id"] = v2["player_id"].astype(str)
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(v2, on="player_id", how="left")
    out["v2_pts"] = out["v2_pts"].fillna(out["fantasy_pts_season"])
    blended = []
    for _, row in out.iterrows():
        pos = str(row["position"])
        w = weights.get(pos) or {}
        # Weights keyed by v1_pred / v2_pred from ensemble_v1_v2.py
        w1 = float(w.get("v1_pred", 0.5))
        w2 = float(w.get("v2_pred", 0.5))
        blended.append(w1 * float(row["fantasy_pts_season"]) + w2 * float(row["v2_pts"]))
    out["fantasy_pts_season"] = blended
    # Keep per-game roughly consistent for display
    games = out["projected_games"].replace(0, pd.NA)
    out["fantasy_pts"] = out["fantasy_pts_season"] / games
    out["fantasy_pts"] = out["fantasy_pts"].fillna(out["fantasy_pts_season"] / 17.0)
    return out.drop(columns=["v2_pts"]), True


def resolve_ensemble_weights_path(
    *,
    season: int,
    ensemble_weights_path: str | None,
    use_ensemble: bool,
    ensemble_v2_points_path: str | None = None,
) -> str | None:
    """Default on when shipped weights + archived v2 points both exist."""
    if not use_ensemble:
        return None
    if ensemble_weights_path:
        return ensemble_weights_path
    v2_path = ensemble_v2_points_path or default_v2_points_path(season)
    if os.path.exists(DEFAULT_ENSEMBLE_WEIGHTS) and os.path.exists(v2_path):
        return DEFAULT_ENSEMBLE_WEIGHTS
    return None


# Verdicts that authorise the distributional overlay. promote_v3_means
# implies simulation readiness, so it qualifies too; hold_v1_default does not.
SIMULATION_READY_VERDICTS = ("simulation_ready", "promote_v3_means")


def read_promotion_gate() -> dict | None:
    """Load the v3 promotion gate report, or None when it has not been run."""
    gate_path = os.path.join(MODEL_V3_DIR, "promotion_gate.json")
    if not os.path.exists(gate_path):
        return None
    with open(gate_path, encoding="utf-8") as fh:
        return json.load(fh)


def _stale_simulation_reason(
    season: int,
    board: pd.DataFrame,
    *,
    model_v3_dir: str | None = None,
    manifest_path: str | None = None,
) -> dict | None:
    """Reason the simulation summary does not describe ``board``, or None.

    Absent provenance is treated as stale. A summary written before the
    manifest carried source_projection_run_id cannot be shown to match, and
    "cannot show it matches" is the same risk as "does not match" for a band
    that ships beside the numbers it is supposed to describe.
    """
    if "projection_run_id" not in board.columns:
        return None  # nothing to compare against; older boards predate the id
    board_ids = board["projection_run_id"].dropna().unique()
    if len(board_ids) != 1:
        return None
    resolved_manifest = manifest_path or os.path.join(
        model_v3_dir or MODEL_V3_DIR, f"simulation_manifest_{season}.json"
    )
    if not os.path.exists(resolved_manifest):
        return {"reason": "simulation_manifest_missing", "board_run_id": str(board_ids[0])}
    with open(resolved_manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    sim_run_id = manifest.get("source_projection_run_id")
    if not sim_run_id:
        return {"reason": "simulation_provenance_unknown", "board_run_id": str(board_ids[0])}
    if str(sim_run_id) != str(board_ids[0]):
        return {
            "reason": "stale_simulation",
            "board_run_id": str(board_ids[0]),
            "simulation_run_id": str(sim_run_id),
        }
    return None


def attach_v3_simulation_percentiles(
    df: pd.DataFrame,
    season: int,
    *,
    require_gate: bool = True,
    model_v3_dir: str | None = None,
    simulation_summary_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge v3 Monte Carlo percentiles when the gate authorises them.

    The percentiles are v3 output reaching the published board, so they are
    gated like the means cutover rather than attaching on file presence
    alone. Existence of simulation_summary_<season>.csv says a simulation
    ran, not that it was calibrated -- only the gate says that.
    """
    if require_gate:
        gate = read_promotion_gate()
        verdict = (gate or {}).get("verdict")
        if verdict not in SIMULATION_READY_VERDICTS:
            return df, {
                "applied": False,
                "reason": "gate_not_simulation_ready",
                "gate_verdict": verdict,
            }
    path = simulation_summary_path or os.path.join(
        model_v3_dir or MODEL_V3_DIR, f"simulation_summary_{season}.csv"
    )
    if not os.path.exists(path):
        return df, {"applied": False, "reason": "missing_simulation_summary"}
    # Percentiles describe the board they were simulated from. Merging a
    # summary built on an earlier board silently mixes two runs -- after the
    # QB anchor-share republish, the stale 2026 percentiles put p50 6.3 points
    # BELOW the point estimate for QBs while sitting above it for everyone
    # else. Refuse rather than publish a band that describes different numbers.
    stale = _stale_simulation_reason(
        season,
        df,
        model_v3_dir=model_v3_dir,
        manifest_path=(
            os.path.join(model_v3_dir, f"simulation_manifest_{season}.json")
            if model_v3_dir
            else None
        ),
    )
    if stale:
        return df, {"applied": False, **stale}
    sim = pd.read_csv(path)
    rename = {
        "p10": "fantasy_pts_p10",
        "p25": "fantasy_pts_p25",
        "p50": "fantasy_pts_p50",
        "p75": "fantasy_pts_p75",
        "p90": "fantasy_pts_p90",
    }
    sim = sim.rename(columns=rename)
    sim["player_id"] = sim["player_id"].astype(str)
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    cols = ["player_id", *rename.values()]
    out = out.merge(sim[cols], on="player_id", how="left")
    out["volatility_flag"] = (
        pd.to_numeric(out["fantasy_pts_p90"], errors="coerce")
        - pd.to_numeric(out["fantasy_pts_p10"], errors="coerce")
    ) > 40.0
    return out, {
        "applied": True,
        "source": path.replace("\\", "/"),
        "gate_verdict": (read_promotion_gate() or {}).get("verdict") if require_gate else None,
        "note": "Distributional overlay only; means/VORP/tiers unchanged",
    }


ACCURACY_FIRST_DIR = os.path.join(OUTPUT_DIR, "accuracy_first_2026")


def default_accuracy_first_board_path(season: int) -> str:
    return os.path.join(OUTPUT_DIR, "accuracy_first_2026", f"fantasy_points_{season}.csv")


def board_identity_hash(fantasy_path: str | None, season: int) -> str | None:
    selected_path = default_selected_board_path(season)
    if selected_path.exists():
        return sha256_file(selected_path)
    path = fantasy_path or os.path.join(OUTPUT_DIR, f"fantasy_points_{season}.csv")
    if not os.path.exists(path):
        return None
    return sha256_file(path)


def _load_recentered_draws(season: int, manifest: dict) -> pd.DataFrame:
    from src.projection.inference.simulate import load_partitioned_draws

    partition_dir = manifest.get("partition_dir")
    if partition_dir and os.path.exists(partition_dir):
        return load_partitioned_draws(season, "", partition_dir=partition_dir)

    run_id = manifest.get("simulation_run_id") or manifest.get("canonical_projection_run_id") or manifest.get("source_projection_run_id")
    if run_id:
        partitioned = load_partitioned_draws(season, str(run_id))
        if not partitioned.empty:
            return partitioned
    recentered_path = manifest.get("recentered_draws_path") or os.path.join(
        MODEL_V3_DIR, f"simulations_recentered_{season}.parquet"
    )
    if os.path.exists(recentered_path):
        return pd.read_parquet(recentered_path)
    legacy_path = os.path.join(MODEL_V3_DIR, f"simulations_{season}.parquet")
    if os.path.exists(legacy_path):
        return pd.read_parquet(legacy_path)
    return pd.DataFrame()


def _draft_value_gate_ok() -> tuple[bool, dict | None]:
    gate = read_finish_probability_gate()
    if not gate:
        return False, {"reason": "missing_finish_probability_gate"}
    state = gate.get("state") or gate.get("verdict")
    if state != FINISH_PROBABILITY_READY:
        return False, {
            "reason": "finish_probability_gate_hold",
            "gate_state": state,
            "gate_verdict": gate.get("verdict"),
            "reasons": gate.get("reasons"),
        }
    if gate.get("publication_verdict") != "pass":
        return False, {
            "reason": "finish_probability_publication_hold",
            "publication_verdict": gate.get("publication_verdict"),
            "reasons": gate.get("reasons"),
        }
    return True, gate


def attach_draft_value_overlay(
    df: pd.DataFrame,
    season: int,
    *,
    team_count: int = DEFAULT_TEAM_COUNT,
    fantasy_path: str | None = None,
    require_gate: bool = True,
    attach_sim_vorp: bool = False,
    model_v3_dir: str | None = None,
    simulation_manifest_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach finish probabilities from recentered draws when provenance passes.

    Simulated VORP fields remain opt-in until their replacement contract is
    validated independently.
    """
    if require_gate:
        ok, gate_meta = _draft_value_gate_ok()
        if not ok:
            return df, {"applied": False, **(gate_meta or {})}

    manifest_path = simulation_manifest_path or os.path.join(
        model_v3_dir or MODEL_V3_DIR, f"simulation_manifest_{season}.json"
    )
    if not os.path.exists(manifest_path):
        return df, {"applied": False, "reason": "missing_simulation_manifest"}
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    gate = read_finish_probability_gate() if require_gate else {}
    if require_gate:
        provenance_ok, provenance_meta = validate_finish_probability_publication(
            season=season,
            board=df,
            manifest=manifest,
            gate=gate or {},
            fantasy_path=fantasy_path,
        )
        if not provenance_ok:
            return df, {
                "applied": False,
                "reason": "finish_probability_provenance_failed",
                **provenance_meta,
            }

    stale = _stale_simulation_reason(
        season,
        df,
        model_v3_dir=model_v3_dir,
        manifest_path=manifest_path,
    )
    if stale:
        return df, {"applied": False, **stale}

    draws = _load_recentered_draws(season, manifest)
    if draws.empty:
        return df, {"applied": False, "reason": "missing_recentered_draws"}

    finish = compute_finish_probabilities(draws)
    if finish.empty:
        return df, {"applied": False, "reason": "empty_finish_probabilities"}

    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    finish["player_id"] = finish["player_id"].astype(str)
    out = out.merge(finish, on="player_id", how="left")

    attached_fields = list(FINISH_PROBABILITY_COLS)
    sim_vorp_attached = False
    sim_vorp_meta: dict = {}
    if attach_sim_vorp:
        board_hash = manifest.get("selected_board_hash") or board_identity_hash(
            fantasy_path, season
        )
        if not board_hash:
            return out, {
                "applied": True,
                "reason": "missing_selected_board_hash",
                "sim_vorp_attached": False,
                "finish_attached": True,
            }
        gate_dir = gate_output_dir(season, str(board_hash))
        vorp_gate = read_simulated_vorp_gate(gate_dir / "simulated_vorp_gate.json")
        if not vorp_gate or vorp_gate.get("state") != SIM_VORP_READY:
            return out, {
                "applied": True,
                "reason": "simulated_vorp_gate_hold",
                "sim_vorp_attached": False,
                "finish_attached": True,
            }
        contract = read_replacement_contract(gate_dir / "replacement_contract.json")
        ok, validation = validate_simulated_vorp_publication(
            manifest=manifest,
            finish_gate=gate,
            replacement_contract=contract,
            vorp_gate=vorp_gate,
        )
        if not ok:
            return out, {
                "applied": True,
                "reason": "simulated_vorp_provenance_failed",
                "sim_vorp_attached": False,
                "finish_attached": True,
                **validation,
            }
        summary_path = gate_dir / "simulated_vorp_summary.parquet"
        if not summary_path.exists():
            return out, {
                "applied": True,
                "reason": "missing_simulated_vorp_summary",
                "sim_vorp_attached": False,
                "finish_attached": True,
            }
        vorp = pd.read_parquet(summary_path)
        vorp["player_id"] = vorp["player_id"].astype(str)
        out = out.merge(vorp, on="player_id", how="left")
        attached_fields.extend(SIM_VORP_COLS)
        sim_vorp_attached = True
        sim_vorp_meta = {
            "sim_vorp_gate_state": vorp_gate.get("state"),
            "replacement_contract_hash": contract.get("contract_hash"),
        }

    return out, {
        "applied": True,
        "source": manifest_path.replace("\\", "/"),
        "transform_version": manifest.get("transform_version"),
        "selected_board_hash": manifest.get("selected_board_hash"),
        "selected_board_model_id": manifest.get("selected_board_model_id"),
        "finish_gate_state": (gate or {}).get("state") or (gate or {}).get("verdict"),
        "publication_verdict": (gate or {}).get("publication_verdict"),
        "finish_cutoffs": list(FINISH_CUTOFFS),
        "attached_fields": attached_fields,
        "sim_vorp_attached": sim_vorp_attached,
        **sim_vorp_meta,
        "rank_tie_policies": simulation_rank_metadata(),
        "note": (
            "Additive finish-probability and simulated-VORP overlays; "
            "deterministic VORP/tiers unchanged. "
            "p_finish_* and simulated rank moments use different tie policies "
            "(see rank_tie_policies)."
        ),
    }


def apply_v3_means(
    df: pd.DataFrame,
    season: int,
    *,
    enabled: bool,
    require_gate: bool = True,
) -> tuple[pd.DataFrame, dict | None]:
    """Optionally replace draft mean points with v3 simulation p50.

    Default path keeps v1 / ensemble means. When ``enabled``, uses
    ``fantasy_pts_p50`` (or summary ``p50``) as ``fantasy_pts_season`` and
    recomputes per-game. Falls back to incumbent means if artifacts or the
    promotion gate are missing.
    """
    if not enabled:
        return df, None
    gate = read_promotion_gate()
    if require_gate and (not gate or gate.get("verdict") != "promote_v3_means"):
        return df, {
            "applied": False,
            "reason": "gate_not_promote_v3_means",
            "gate_verdict": (gate or {}).get("verdict"),
            "fallback": "v1_or_ensemble_means",
        }
    summary_path = os.path.join(MODEL_V3_DIR, f"simulation_summary_{season}.csv")
    if not os.path.exists(summary_path):
        return df, {
            "applied": False,
            "reason": "missing_simulation_summary",
            "fallback": "v1_or_ensemble_means",
        }
    sim = pd.read_csv(summary_path)
    if "p50" not in sim.columns:
        return df, {
            "applied": False,
            "reason": "missing_p50",
            "fallback": "v1_or_ensemble_means",
        }
    sim = sim[["player_id", "p50"]].copy()
    sim["player_id"] = sim["player_id"].astype(str)
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(sim, on="player_id", how="left")
    missing = out["p50"].isna().mean()
    if missing > 0.5:
        return df, {
            "applied": False,
            "reason": "p50_coverage_too_low",
            "missing_frac": float(missing),
            "fallback": "v1_or_ensemble_means",
        }
    out["fantasy_pts_season"] = out["p50"].fillna(out["fantasy_pts_season"])
    games = out["projected_games"].replace(0, pd.NA)
    out["fantasy_pts"] = out["fantasy_pts_season"] / games
    out["fantasy_pts"] = out["fantasy_pts"].fillna(out["fantasy_pts_season"] / 17.0)
    out = out.drop(columns=["p50"])
    return out, {
        "applied": True,
        "source": summary_path.replace("\\", "/"),
        "gate_verdict": (gate or {}).get("verdict"),
        "note": "Draft mean points from v3 sim p50; VORP/tiers recomputed on this mean",
    }


def export_draft_data(
    season: int,
    *,
    tier_config: TierConfig | None = None,
    team_count: int = DEFAULT_TEAM_COUNT,
    rookie_rank_scale: float = ROOKIE_RANK_SCALE,
    ensemble_weights_path: str | None = None,
    ensemble_v2_points_path: str | None = None,
    use_ensemble: bool = True,
    use_v3_means: bool = False,
    require_v3_means_gate: bool = True,
    fantasy_path: str | None = None,
    out_path: str | None = None,
    attach_sim_vorp: bool = False,
    model_v3_dir: str | None = None,
    simulation_manifest_path: str | None = None,
    skip_public_release_reports: bool = False,
    require_gate: bool = True,
) -> str:
    df = load_projections(season, fantasy_path)
    accuracy_meta = accuracy_ensemble_metadata(df)
    ensemble_meta = None
    weights_path = resolve_ensemble_weights_path(
        season=season,
        ensemble_weights_path=ensemble_weights_path,
        use_ensemble=use_ensemble,
        ensemble_v2_points_path=ensemble_v2_points_path,
    )
    if weights_path:
        weights = load_ensemble_weights(weights_path)
        v2_path = ensemble_v2_points_path or default_v2_points_path(season)
        df, applied = apply_ensemble_points(df, weights, v2_points_path=v2_path)
        if applied:
            ensemble_meta = {
                "weights_path": weights_path.replace("\\", "/"),
                "v2_points_path": v2_path.replace("\\", "/"),
                "weights": weights,
                "note": "Draft post-process blend only; compose_board unchanged",
            }
    df, v3_sim_meta = attach_v3_simulation_percentiles(
        df,
        season,
        require_gate=require_gate,
        model_v3_dir=model_v3_dir,
        simulation_summary_path=(
            os.path.join(model_v3_dir, f"simulation_summary_{season}.csv")
            if model_v3_dir
            else None
        ),
    )
    df, draft_value_meta = attach_draft_value_overlay(
        df,
        season,
        team_count=team_count,
        fantasy_path=fantasy_path,
        attach_sim_vorp=attach_sim_vorp,
        model_v3_dir=model_v3_dir,
        simulation_manifest_path=simulation_manifest_path,
        require_gate=require_gate,
    )
    df, v3_means_meta = apply_v3_means(
        df,
        season,
        enabled=use_v3_means,
        require_gate=require_v3_means_gate,
    )
    df = add_vorp_columns(df, team_count=team_count, rookie_rank_scale=rookie_rank_scale)
    df = add_tier_columns(
        df,
        points_col="vorp",
        config=tier_config,
        overall_points_col="vorp",
        overall_gap=OVERALL_VORP_TIER_GAP,
    )
    players = build_player_records(df)
    sources = (
        df["source"].dropna().astype(str).value_counts().to_dict()
        if "source" in df.columns
        else {}
    )
    if any(str(s).startswith("v2_") for s in sources):
        engine = "fantasy-projections-2 (team-first) — unexpected in native output/"
    else:
        engine = "fantasy-projections (rate-forecast / LightGBM)"

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "meta": {
            "season": season,
            "generated_at": generated_at,
            "player_count": len(players),
            "scoring": "half-PPR, 4pt passing TD",
            "source_file": (
                fantasy_path.replace("\\", "/")
                if fantasy_path else f"output/fantasy_points_{season}.csv"
            ),
            "projection_engine": (
                "v3 simulation p50 means (flagged cutover); compose_board unchanged"
                if v3_means_meta and v3_means_meta.get("applied")
                else (
                    "accuracy-first v1/v2/ADP ensemble; compose_board unchanged"
                    if accuracy_meta else (
                        "v1/v2 draft ensemble (post-process); compose_board unchanged"
                        if ensemble_meta else engine
                    )
                )
            ),
            "model_id": (
                "v3_means"
                if v3_means_meta and v3_means_meta.get("applied")
                else (
                    "accuracy_first_ensemble"
                    if accuracy_meta else (
                        "v1_v2_ensemble" if ensemble_meta else "v1_rate_forecast"
                    )
                )
            ),
            "source_mix": sources,
            "roster": "1QB, 2RB, 3WR, 1TE, 1FLEX",
            "vorp_team_count": int(team_count),
            # The ranks actually used, which are deepened for availability --
            # not the nominal roster-math ranks.
            "vorp_replacement_ranks": df.attrs.get(
                "vorp_replacement_ranks", replacement_ranks(team_count)
            ),
            "vorp_replacement_ranks_nominal": replacement_ranks(team_count),
            "vorp_curve_weight": df.attrs.get("vorp_curve_weight", {}),
            # Published so the browser can reproduce the same blend when the
            # league size changes; a client-side recompute that skipped it would
            # silently re-inflate the position it corrects.
            "vorp_position_curves": load_position_curves(),
            "vorp_availability_factors": {
                k: round(v, 4)
                for k, v in (df.attrs.get("vorp_availability_factors") or {}).items()
            },
            "vorp_starters": STARTERS,
            "vorp_flex_share": FLEX_SHARE,
            "rookie_rank_scale": float(rookie_rank_scale),
            "ensemble": ensemble_meta,
            "accuracy_ensemble": accuracy_meta,
            "v3_simulation": v3_sim_meta,
            "draft_value_simulation": draft_value_meta,
            "v3_means": v3_means_meta,
            "sentiment": build_sentiment_meta(season, df, generated_at=generated_at),
        },
        "tier_gaps": {
            "overall_vorp": OVERALL_VORP_TIER_GAP,
            "flex": FLEX_TIER_GAP,
            **DEFAULT_TIER_GAPS,
        },
        "tier_summary": tier_summary(df),
        "players": players,
    }

    os.makedirs(DRAFT_DATA_DIR, exist_ok=True)
    out_path = out_path or os.path.join(DRAFT_DATA_DIR, f"players_{season}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)

    board_report = build_release_report_board(
        season=season,
        draft_value_meta=draft_value_meta,
        v3_sim_meta=v3_sim_meta,
        exported_board_path=Path(out_path),
        players_df=df,
    )
    if skip_public_release_reports:
        report_dir = Path(model_v3_dir) if model_v3_dir else Path(MODEL_V3_DIR)
        write_release_report_board(board_report, season=season, out_dir=report_dir)
        sim_report_path = report_dir / f"release_report_simulation_{season}.json"
        if sim_report_path.exists():
            sim_report = json.loads(sim_report_path.read_text(encoding="utf-8"))
            merged = merge_release_reports(sim_report, board_report)
            write_merged_release_report(merged, season=season, out_dir=report_dir)
    else:
        write_release_report_board(board_report, season=season)
        sim_report_path = Path(MODEL_V3_DIR) / f"release_report_simulation_{season}.json"
        if sim_report_path.exists():
            sim_report = json.loads(sim_report_path.read_text(encoding="utf-8"))
            merged = merge_release_reports(sim_report, board_report)
            write_merged_release_report(merged, season=season)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draft assistant data")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--ensemble-weights",
        default=None,
        help=(
            "Position blend weights JSON (default: src/draft_assistant/"
            "ensemble_weights.json when output/model_v2 fantasy points exist)"
        ),
    )
    parser.add_argument(
        "--ensemble-v2-points",
        default=None,
        help="Optional v2 fantasy_points CSV (default output/model_v2/fantasy_points_<season>.csv)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Export the native v1 board only (skip v1/v2 post-process blend)",
    )
    parser.add_argument(
        "--v3-means",
        action="store_true",
        help=(
            "Replace draft mean points with v3 simulation p50 when "
            "promotion_gate.json verdict is promote_v3_means; otherwise keep "
            "v1/ensemble means"
        ),
    )
    parser.add_argument(
        "--force-v3-means",
        action="store_true",
        help="Use v3 p50 means even if promotion gate has not cleared promote_v3_means",
    )
    parser.add_argument(
        "--attach-sim-vorp",
        action="store_true",
        help="Attach simulated VORP overlay when simulated_vorp_gate publication_verdict is pass",
    )
    args = parser.parse_args()
    path = export_draft_data(
        args.season,
        ensemble_weights_path=args.ensemble_weights,
        ensemble_v2_points_path=args.ensemble_v2_points,
        use_ensemble=not args.no_ensemble,
        use_v3_means=bool(args.v3_means or args.force_v3_means),
        require_v3_means_gate=not args.force_v3_means,
        attach_sim_vorp=bool(args.attach_sim_vorp),
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
