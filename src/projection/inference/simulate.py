"""Monte Carlo simulation for season outcome distributions."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR, MODEL_V3_DIR, V3_MODELS_DIR
from src.projection.fantasy_points import SCORING
from src.projection.inference.recenter import (
    TRANSFORM_VERSION,
    board_points_series,
    recenter_draws,
    sha256_file,
)
from src.projection.inference.wr_calibration import (
    ARTIFACT_PATH as WR_CALIBRATION_PATH,
    TRANSFORM_VERSION as WR_CALIBRATION_VERSION,
    load_wr_calibration,
    recenter_draws_wr_scaled,
)
from src.projection.inference.simulation_config import (
    deterministic_simulation_seed,
    load_simulation_config,
)

DRAW_COLUMNS = ("player_id", "draw", "position", "team", "fantasy_pts_season")


def _score_wide_totals(wide: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=wide.index)
    for stat, weight in SCORING.items():
        if stat in wide.columns:
            total = total + pd.to_numeric(wide[stat], errors="coerce").fillna(0.0) * weight
    return total


def _load_share_manifest() -> dict:
    """Fitted opportunity-share concentrations, when they exist on disk."""
    path = Path(V3_MODELS_DIR) / "opportunity_shares" / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_residuals() -> pd.DataFrame:
    path = Path(BACKTEST_DIR) / "residuals_rolling.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["position", "stat", "team", "resid"])


def _row_residual_pools(stat_rows: pd.DataFrame, residuals: pd.DataFrame) -> list[np.ndarray]:
    """Map each projection row to a residual bootstrap pool."""
    if residuals.empty:
        return [np.array([0.0]) for _ in range(len(stat_rows))]
    by_team = {
        key: grp["resid"].to_numpy()
        for key, grp in residuals.groupby(["position", "stat", "team"], observed=True)
    }
    by_stat = {
        key: grp["resid"].to_numpy()
        for key, grp in residuals.groupby(["position", "stat"], observed=True)
    }
    pools: list[np.ndarray] = []
    for row in stat_rows.itertuples(index=False):
        pool = by_team.get((row.position, row.stat, row.team))
        if pool is None or len(pool) == 0:
            pool = by_stat.get((row.position, row.stat), np.array([0.0]))
        pools.append(pool)
    return pools


SIMULATION_MODE = "full"


def simulate_season_distributions(
    projections: pd.DataFrame,
    *,
    n_draws: int = 1000,
    seed: int = 42,
    mode: str = SIMULATION_MODE,
    uncertainty_manifest: dict | None = None,
    use_projection_uncertainty: bool = True,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Draw season stat totals and fantasy points from projection board.

    ``mode=full`` (default) runs the v3 generative path: one team volume draw
    feeds a player's whole stat line, so a player's stats move together.

    ``mode=interim`` is RETIRED as a shipping mode and kept only so
    scripts/compare_simulation_modes.py can still score it. It bootstraps
    cross-fitted residuals per stat INDEPENDENTLY, which destroys the
    +0.62..+0.88 correlation between a player's own stats and understates the
    summed spread by 31%. Held out on 2025 it loses to generative on every
    metric: coverage 0.505 vs 0.537, p50 MAE 34.56 vs 33.86, rho .696 vs .741.

    Projection-uncertainty parameters are applied only after the exact
    season-outcome calibration gate selects them. A ``hold`` or missing
    verdict runs the baseline full mechanism for diagnostics but cannot
    authorize a published distributional overlay. See
    docs/decisions/SIMULATION_MODE_2026-08-26.md.
    """
    if mode == "full":
        result = _simulate_full_generative(
            projections,
            n_draws=n_draws,
            seed=seed,
            uncertainty_manifest=uncertainty_manifest,
            use_projection_uncertainty=use_projection_uncertainty,
            progress_every=progress_every,
        )
        return result  # type: ignore[return-value]
    rng = np.random.default_rng(seed)
    residuals = _load_residuals()
    stat_rows = projections.copy()
    stat_rows["pred_pg"] = pd.to_numeric(stat_rows["pred_pg"], errors="coerce").fillna(0.0)
    stat_rows["projected_games"] = pd.to_numeric(
        stat_rows["projected_games"], errors="coerce"
    ).fillna(17.0)

    base_pred = stat_rows["pred_pg"].to_numpy()
    row_pools = _row_residual_pools(stat_rows, residuals)
    noise_matrix = np.vstack([
        rng.choice(pool, size=n_draws) if len(pool) else np.zeros(n_draws)
        for pool in row_pools
    ]).T  # shape (n_draws, n_rows)

    player_games = stat_rows.groupby("player_id", observed=True)["projected_games"].first()
    players = player_games.index.to_numpy()
    probs = np.clip(player_games.to_numpy() / 17.0, 0.05, 1.0)
    games_matrix = rng.binomial(17, probs, size=(n_draws, len(players)))

    meta = stat_rows[["player_id", "position", "team", "stat"]].copy()
    draws = []
    for draw_idx in range(n_draws):
        draw_games = stat_rows["player_id"].map(
            pd.Series(games_matrix[draw_idx], index=players)
        ).fillna(17.0)
        meta["season_total"] = (
            np.clip(base_pred + noise_matrix[draw_idx, :], 0, None) * draw_games.to_numpy()
        )
        wide = meta.pivot_table(
            index=["player_id", "position", "team"],
            columns="stat",
            values="season_total",
            aggfunc="first",
        ).reset_index()
        wide["fantasy_pts_season"] = _score_wide_totals(wide)
        wide["draw"] = draw_idx
        draws.append(wide)
    return pd.concat(draws, ignore_index=True)


def _simulate_full_generative(
    projections: pd.DataFrame,
    *,
    n_draws: int = 1000,
    seed: int = 42,
    uncertainty_manifest: dict | None = None,
    use_projection_uncertainty: bool = True,
    progress_every: int | None = None,
    start_draw: int = 0,
    end_draw: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, np.random.Generator]:
    draws, final_rng = _simulate_full_generative_range(
        projections,
        start_draw=start_draw,
        end_draw=end_draw if end_draw is not None else n_draws,
        seed=seed,
        uncertainty_manifest=uncertainty_manifest,
        use_projection_uncertainty=use_projection_uncertainty,
        progress_every=progress_every,
        total_draws=n_draws,
        rng=rng,
    )
    if rng is not None or start_draw != 0 or end_draw is not None:
        return draws, final_rng
    return draws


def _simulate_full_generative_range(
    projections: pd.DataFrame,
    *,
    start_draw: int,
    end_draw: int,
    seed: int,
    uncertainty_manifest: dict | None = None,
    use_projection_uncertainty: bool = True,
    progress_every: int | None = None,
    total_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.random.Generator]:
    from src.projection.inference.reconcile import (
        reconcile_v3_generative,
        team_environment_from_board,
    )
    from src.projection.models.uncertainty import (
        draw_availability,
        draw_team_environment,
        load_uncertainty_manifest,
    )

    rng = rng or np.random.default_rng(seed)
    players = projections.copy()
    team_env = team_environment_from_board(players)
    if uncertainty_manifest is None:
        uncertainty_manifest = load_uncertainty_manifest()
    uncertainty_manifest = uncertainty_manifest or {}
    share_manifest = uncertainty_manifest or _load_share_manifest()
    total = total_draws if total_draws is not None else end_draw
    draws = []
    for draw_idx in range(start_draw, end_draw):
        if progress_every and draw_idx % progress_every == 0:
            print(f"simulation draw {draw_idx}/{total}", flush=True)
        drawn_env = (
            draw_team_environment(team_env, uncertainty_manifest, rng=rng)
            if use_projection_uncertainty and uncertainty_manifest else team_env
        )
        availability = (
            draw_availability(players, uncertainty_manifest, rng=rng)
            if use_projection_uncertainty and uncertainty_manifest else None
        )
        generative = reconcile_v3_generative(
            players,
            drawn_env,
            rng=rng,
            share_manifest=share_manifest,
            availability_games=availability,
        )
        if generative.empty:
            continue
        wide = generative.groupby(
            ["player_id", "position", "team"], observed=True
        ).sum(numeric_only=True).reset_index()
        wide["fantasy_pts_season"] = _score_wide_totals(wide)
        wide["draw"] = draw_idx
        draws.append(wide)
    frame = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    return frame, rng


def simulate_season_draw_range(
    projections: pd.DataFrame,
    *,
    start_draw: int,
    end_draw: int,
    seed: int = 42,
    uncertainty_manifest: dict | None = None,
    use_projection_uncertainty: bool = True,
    progress_every: int | None = None,
    total_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.random.Generator]:
    """Generate draws in [start_draw, end_draw) and return the advanced RNG."""
    return _simulate_full_generative_range(
        projections,
        start_draw=start_draw,
        end_draw=end_draw,
        seed=seed,
        uncertainty_manifest=uncertainty_manifest,
        use_projection_uncertainty=use_projection_uncertainty,
        progress_every=progress_every,
        total_draws=total_draws,
        rng=rng,
    )


def summarize_simulations(draws: pd.DataFrame) -> pd.DataFrame:
    """Aggregate draw-level fantasy points to player percentiles."""
    quantiles = draws.groupby(
        ["player_id", "position", "team"], observed=True
    )["fantasy_pts_season"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    quantiles.columns = ["p10", "p25", "p50", "p75", "p90"]
    return quantiles.reset_index()


def slim_draw_frame(draws: pd.DataFrame) -> pd.DataFrame:
    """Player-by-draw columns required for finish/VORP overlays."""
    cols = [col for col in DRAW_COLUMNS if col in draws.columns]
    return draws[cols].copy()


def write_partitioned_draws(
    draws: pd.DataFrame,
    *,
    season: int,
    run_id: str,
    chunk_size: int = 250,
    partition_base: Path | None = None,
) -> dict:
    """Write atomic part-*.parquet partitions for draw-level fantasy points."""
    slim = slim_draw_frame(draws)
    if slim.empty:
        return {"partition_dir": None, "partition_hashes": [], "partition_count": 0}

    base = partition_base or (Path(MODEL_V3_DIR) / "simulations")
    partition_root = base / f"season={season}" / f"run_id={run_id}"
    partition_root.mkdir(parents=True, exist_ok=True)
    for existing in partition_root.glob("part-*.parquet"):
        existing.unlink()

    hashes: list[str] = []
    draw_ids = sorted(slim["draw"].unique())
    for part_idx, start in enumerate(range(0, len(draw_ids), chunk_size)):
        chunk_draws = set(draw_ids[start : start + chunk_size])
        part = slim[slim["draw"].isin(chunk_draws)].copy()
        part_path = partition_root / f"part-{part_idx:05d}.parquet"
        part.to_parquet(part_path, index=False)
        hashes.append(hashlib.sha256(part_path.read_bytes()).hexdigest())
    return {
        "partition_dir": str(partition_root),
        "partition_hashes": hashes,
        "partition_count": len(hashes),
    }


def load_partitioned_draws(
    season: int,
    run_id: str,
    *,
    partition_base: Path | None = None,
    partition_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read all draw partitions for a season/run."""
    if partition_dir is not None:
        partition_root = Path(partition_dir)
    else:
        base = partition_base or (Path(MODEL_V3_DIR) / "simulations")
        partition_root = base / f"season={season}" / f"run_id={run_id}"
    parts = sorted(partition_root.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=list(DRAW_COLUMNS))
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)


def _calibration_hash() -> str | None:
    path = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
    if not path.exists():
        return None
    return sha256_file(str(path))


def write_simulation_outputs(
    projections: pd.DataFrame,
    season: int,
    *,
    n_draws: int = 1000,
    mode: str = SIMULATION_MODE,
    uncertainty_manifest: dict | None = None,
    selected_board: pd.DataFrame | None = None,
    selected_board_hash: str | None = None,
    selected_board_model_id: str | None = None,
    simulation_profile: str = "dev",
    out_dir: Path | None = None,
    partition_root: Path | None = None,
    simulation_run_id: str | None = None,
    rollout_label: str | None = None,
    artifact_namespace: str | None = None,
) -> dict:
    out_dir = out_dir or Path(MODEL_V3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_config = load_simulation_config()
    chunk_size = int(
        (sim_config.get("profiles") or {}).get(simulation_profile, {}).get("chunk_size")
        or 250
    )
    calibration_hash = _calibration_hash() or ""
    board_hash = selected_board_hash or ""
    configured_seed = int(sim_config.get("random_seed") or 2026)
    deterministic_seed = deterministic_simulation_seed(
        season=season,
        board_hash=board_hash,
        calibration_hash=calibration_hash,
        configured_seed=configured_seed,
    )
    started = time.perf_counter()
    if uncertainty_manifest is None:
        from src.projection.models.uncertainty import load_uncertainty_manifest

        uncertainty_manifest = load_uncertainty_manifest()
    selected_mode = uncertainty_manifest.get("selected_distribution_mode") if uncertainty_manifest else None
    # calibrate_v3_distribution.py's acceptance gate can verdict "hold" -- and
    # a manifest that predates the gate, or one whose calibration errored, has
    # no verdict at all. Either way the manifest itself was never accepted, so
    # applying it here is exactly the thing "hold" exists to prevent. Score
    # generatively but with NO option_a artifacts, matching how the gate's own
    # "baseline" arm was measured -- not "no simulation", "the last accepted
    # configuration". _simulate_full_generative applying the manifest whenever
    # it is merely non-empty, regardless of this field, is what let a
    # gate-rejected candidate (0.733 coverage, failing every acceptance gate)
    # ship to production the moment it was fitted.
    effective_manifest = uncertainty_manifest
    if selected_mode not in ("generative_projection_uncertainty", "joint_bootstrap"):
        effective_manifest = {}
    donor_path = None
    donor_hash = None
    if effective_manifest and selected_mode == "joint_bootstrap":
        from src.projection.models.uncertainty import JOINT_DONORS_PATH

        donor_meta = effective_manifest.get("joint_donors") or {}
        donor_path = Path(donor_meta.get("path") or JOINT_DONORS_PATH)
        expected_hash = donor_meta.get("sha256")
        if donor_path.exists():
            donor_hash = hashlib.sha256(donor_path.read_bytes()).hexdigest()
        # A bootstrap verdict without its exact calibrated donor artifact is
        # not permission to emit uncorrected generative draws under a
        # bootstrap label. Fall back closed and let the promotion gate reject.
        if not expected_hash or donor_hash != expected_hash:
            effective_manifest = {}
            donor_path = None
            donor_hash = None
    draws = simulate_season_distributions(
        projections,
        n_draws=n_draws,
        seed=deterministic_seed,
        mode=mode,
        uncertainty_manifest=effective_manifest,
    )
    if mode == "full" and effective_manifest and selected_mode == "joint_bootstrap":
        from src.projection.models.uncertainty import joint_bootstrap_draws

        donors = pd.read_parquet(donor_path)
        draws = joint_bootstrap_draws(
            draws,
            projections,
            donors,
            rng=np.random.default_rng(deterministic_seed + int(season)),
        )
    summary = summarize_simulations(draws)
    recentered_draws = None
    recentered_summary = None
    wr_calibration = None
    if selected_board is not None:
        selected_points = board_points_series(selected_board)
        wr_calibration = load_wr_calibration()
        wr_scale = float((wr_calibration or {}).get("selected_wr_scale", 1.0))
        recentered_draws = recenter_draws_wr_scaled(
            draws,
            selected_points,
            wr_scale=wr_scale,
        )
        recentered_summary = summarize_simulations(recentered_draws)
    draws_path = out_dir / f"simulations_{season}.parquet"
    summary_path = out_dir / f"simulation_summary_{season}.csv"
    draws.to_parquet(draws_path, index=False)
    summary.to_csv(summary_path, index=False)
    recentered_draws_path = None
    recentered_summary_path = None
    if recentered_draws is not None:
        recentered_draws_path = out_dir / f"simulations_recentered_{season}.parquet"
        recentered_summary_path = out_dir / f"simulation_summary_recentered_{season}.csv"
        slim_draw_frame(recentered_draws).to_parquet(recentered_draws_path, index=False)
        recentered_summary.to_csv(recentered_summary_path, index=False)
    # Record WHICH board this was simulated from. Without it a summary from an
    # earlier board merges into a later one unnoticed: the 2026 percentiles
    # outlived a republish that moved QB projections, leaving p50 6.3 points
    # BELOW its own point estimate for QBs and above it everywhere else.
    source_run_id = None
    if "projection_run_id" in projections.columns:
        ids = projections["projection_run_id"].dropna().unique()
        if len(ids) == 1:
            source_run_id = str(ids[0])
    partition_meta = {}
    partition_run_id = simulation_run_id or source_run_id
    if partition_run_id:
        partition_meta = write_partitioned_draws(
            recentered_draws if recentered_draws is not None else draws,
            season=season,
            run_id=partition_run_id,
            chunk_size=chunk_size,
            partition_base=partition_root,
        )
    runtime_seconds = round(time.perf_counter() - started, 3)
    finish_gate_hash = None
    finish_gate_verdict = None
    segment_report_hash = None
    finish_gate_path = Path(MODEL_V3_DIR) / "finish_probability_gate.json"
    if finish_gate_path.exists():
        finish_gate = json.loads(finish_gate_path.read_text(encoding="utf-8"))
        finish_gate_verdict = finish_gate.get("state") or finish_gate.get("verdict")
        segment_report_hash = finish_gate.get("segment_summary_hash")
        gate_body = dict(finish_gate)
        gate_body.pop("generated_at", None)
        finish_gate_hash = hashlib.sha256(
            json.dumps(gate_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    wr_calibration_artifact_hash = (
        sha256_file(str(WR_CALIBRATION_PATH))
        if recentered_draws is not None and wr_calibration and WR_CALIBRATION_PATH.exists()
        else None
    )
    calibration_hashes = {
        key: value
        for key, value in {
            "calibration_hash": calibration_hash or None,
            "wr_calibration_artifact_hash": wr_calibration_artifact_hash,
            "finish_probability_gate_hash": None,
            "segment_report_hash": segment_report_hash,
        }.items()
        if value
    }
    partition_identity = None
    from src.projection.inference.simulation_config import partition_identity_key
    from src.projection.simulation_profile_resolver import resolve_simulation_profile_identity

    profile_identity = resolve_simulation_profile_identity(profile_key=simulation_profile)
    partition_identity = partition_identity_key(
        season=season,
        board_hash=board_hash,
        calibration_hashes=calibration_hashes,
        configuration_hash=profile_identity["configuration_hash"],
        policy_hash=profile_identity["policy_hash"],
        seed=deterministic_seed,
    )
    manifest = {
        "season": season,
        "n_draws": n_draws,
        "draw_count": n_draws,
        "mode": mode,
        "distribution_mode": selected_mode if effective_manifest else mode,
        "uncertainty_gate_verdict": selected_mode,
        "uncertainty_applied": bool(effective_manifest),
        "uncertainty_version": effective_manifest.get("version") if effective_manifest else None,
        "uncertainty_artifact_hash": effective_manifest.get("artifact_hash") if effective_manifest else None,
        "uncertainty_training_cutoff": effective_manifest.get("training_cutoff") if effective_manifest else None,
        "joint_donors_hash": donor_hash if effective_manifest else None,
        "canonical_projection_run_id": source_run_id,
        "source_projection_run_id": source_run_id,
        "selected_board_hash": selected_board_hash,
        "selected_board_model_id": selected_board_model_id,
        "transform_version": TRANSFORM_VERSION if recentered_draws is not None else None,
        "wr_calibration_version": (
            WR_CALIBRATION_VERSION
            if recentered_draws is not None and wr_calibration
            else None
        ),
        "wr_residual_scale": (
            float(wr_calibration["selected_wr_scale"])
            if recentered_draws is not None and wr_calibration
            else None
        ),
        "wr_calibration_sha256": wr_calibration_artifact_hash,
        "wr_calibration_artifact_hash": wr_calibration_artifact_hash,
        "deterministic_seed": deterministic_seed,
        "simulation_profile": simulation_profile,
        "simulation_run_id": partition_run_id,
        "rollout_label": rollout_label,
        "artifact_namespace": artifact_namespace,
        "calibration_hash": calibration_hash or None,
        "segment_report_hash": segment_report_hash,
        "finish_probability_gate_hash": finish_gate_hash,
        "finish_probability_gate_verdict": finish_gate_verdict,
        "draws_path": str(draws_path),
        "summary_path": str(summary_path),
        "recentered_draws_path": str(recentered_draws_path) if recentered_draws_path else None,
        "recentered_summary_path": str(recentered_summary_path) if recentered_summary_path else None,
        "runtime_seconds": runtime_seconds,
        "partition_identity_key": partition_identity,
        **partition_meta,
    }
    if partition_meta.get("partition_hashes"):
        from src.projection.evaluation.finish_probability_gate import validate_draw_partitions

        partitions_ok, partition_validation = validate_draw_partitions(
            {**manifest, **partition_meta}
        )
        if not partitions_ok:
            raise ValueError(
                f"partition validation failed before manifest write: {partition_validation}"
            )
    (out_dir / f"simulation_manifest_{season}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
