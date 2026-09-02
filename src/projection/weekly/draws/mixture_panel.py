"""As-of mixture training panel: event labels + conditional targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.projection.weekly.draws.contracts import classify_zero_row

MIXTURE_PANEL_SCHEMA_VERSION = 1

EVENT_LABEL_COLS = (
    "has_scheduled_game",
    "is_active_label",
    "participated_label",
    "positive_usage_label",
    "zero_class",
)

CONDITIONAL_TARGET_COLS = (
    "cond_target_share",
    "cond_carry_share",
    "cond_snap_share",
    "cond_passing_first_downs",
    "cond_rushing_first_downs",
    "cond_receiving_first_downs",
)


@dataclass(frozen=True)
class MixturePanelArtifact:
    path: Path
    schema_version: int
    row_count: int
    cutoff_note: str
    panel_hash: str
    source_panel_hash: str
    event_rates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_frame(df: pl.DataFrame) -> str:
    # Stable content hash via sorted parquet bytes would be ideal; use row/col summary + sample.
    meta = {
        "height": df.height,
        "width": df.width,
        "columns": df.columns,
    }
    head = df.head(min(50, df.height)).write_csv().encode("utf-8")
    blob = json.dumps(meta, sort_keys=True).encode("utf-8") + head
    return hashlib.sha256(blob).hexdigest()


def build_mixture_panel(
    panel: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """Derive event labels and conditional targets from the leakage-safe panel.

    Labels:
    - ``has_scheduled_game``: non-null ``game_id`` (bye / missing schedule → False)
    - ``is_active_label``: not hard-out by ``is_out`` / near-zero ``play_prob`` when scheduled
    - ``participated_label``: offense snaps > 0 or dropback attempts > 0 among active
    - ``positive_usage_label``: targets+carries+attempts > 0 among participants

    Bye rows are excluded from event-rate denominators for participation/usage.
    """
    required = {"season", "week", "gsis_id", "position", "team", "game_id"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    df = panel
    if seasons is not None:
        df = df.filter(pl.col("season").is_in(list(seasons)))

    snaps = (
        pl.col("offense_snaps").fill_null(0.0)
        if "offense_snaps" in df.columns
        else pl.lit(0.0)
    )
    targets = pl.col("targets").fill_null(0.0) if "targets" in df.columns else pl.lit(0.0)
    carries = pl.col("carries").fill_null(0.0) if "carries" in df.columns else pl.lit(0.0)
    attempts = pl.col("attempts").fill_null(0.0) if "attempts" in df.columns else pl.lit(0.0)
    play_prob = pl.col("play_prob").fill_null(1.0) if "play_prob" in df.columns else pl.lit(1.0)
    is_out = pl.col("is_out").fill_null(False) if "is_out" in df.columns else pl.lit(False)

    df = df.with_columns(
        [
            pl.col("game_id").is_not_null().alias("has_scheduled_game"),
            (~is_out & (play_prob > 1e-6)).alias("_not_ruled_out"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(~pl.col("has_scheduled_game"))
            .then(None)
            .otherwise(pl.col("_not_ruled_out"))
            .alias("is_active_label"),
            (snaps > 0).alias("_has_snaps"),
            ((targets + carries + attempts) > 0).alias("_has_usage"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(~pl.col("has_scheduled_game"))
            .then(None)
            .when(~pl.col("is_active_label").fill_null(False))
            .then(False)
            .otherwise(pl.col("_has_snaps") | pl.col("_has_usage"))
            .alias("participated_label"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(~pl.col("has_scheduled_game"))
            .then(None)
            .when(~pl.col("participated_label").fill_null(False))
            .then(False)
            .otherwise(pl.col("_has_usage"))
            .alias("positive_usage_label"),
        ]
    )

    # Conditional shares among positive-usage rows only (else null).
    team_targets = (
        pl.col("team_targets").fill_null(0.0)
        if "team_targets" in df.columns
        else targets  # fallback; caller should prefer team totals
    )
    target_share = (
        pl.col("target_share")
        if "target_share" in df.columns
        else pl.when(team_targets > 0).then(targets / team_targets).otherwise(None)
    )
    carry_share = pl.col("carry_share") if "carry_share" in df.columns else pl.lit(None)
    snap_share = pl.col("snap_share") if "snap_share" in df.columns else pl.lit(None)

    df = df.with_columns(
        [
            pl.when(pl.col("positive_usage_label").fill_null(False))
            .then(target_share)
            .otherwise(None)
            .alias("cond_target_share"),
            pl.when(pl.col("positive_usage_label").fill_null(False))
            .then(carry_share)
            .otherwise(None)
            .alias("cond_carry_share"),
            pl.when(pl.col("participated_label").fill_null(False))
            .then(snap_share)
            .otherwise(None)
            .alias("cond_snap_share"),
        ]
    )

    for src, dst in (
        ("passing_first_downs", "cond_passing_first_downs"),
        ("rushing_first_downs", "cond_rushing_first_downs"),
        ("receiving_first_downs", "cond_receiving_first_downs"),
    ):
        if src in df.columns:
            df = df.with_columns(
                pl.when(pl.col("positive_usage_label").fill_null(False))
                .then(pl.col(src))
                .otherwise(None)
                .alias(dst)
            )
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(dst))

    # Zero-class via map — keep Python helper for explicit semantics.
    classes: list[str | None] = []
    for row in df.select(
        [
            "has_scheduled_game",
            "is_active_label",
            snaps.alias("offense_snaps"),
            targets.alias("targets"),
            carries.alias("carries"),
            attempts.alias("attempts"),
        ]
    ).iter_rows(named=True):
        classes.append(
            classify_zero_row(
                has_scheduled_game=bool(row["has_scheduled_game"]),
                is_active=row["is_active_label"],
                offense_snaps=float(row["offense_snaps"] or 0.0),
                targets=float(row["targets"] or 0.0),
                carries=float(row["carries"] or 0.0),
                attempts=float(row["attempts"] or 0.0),
            )
        )
    df = df.with_columns(pl.Series("zero_class", classes)).drop(
        ["_not_ruled_out", "_has_snaps", "_has_usage"]
    )
    return df


def summarize_event_rates(df: pl.DataFrame) -> dict[str, Any]:
    scheduled = df.filter(pl.col("has_scheduled_game"))
    out: dict[str, Any] = {
        "rows": df.height,
        "scheduled_rows": scheduled.height,
        "bye_or_unscheduled": int((~df["has_scheduled_game"]).sum()),
    }
    if scheduled.is_empty():
        return out
    for col in ("is_active_label", "participated_label", "positive_usage_label"):
        series = scheduled[col].drop_nulls()
        out[f"rate_{col}"] = float(series.mean()) if series.len() else None
    by_season: dict[str, Any] = {}
    for season in sorted(scheduled["season"].unique().to_list()):
        sub = scheduled.filter(pl.col("season") == season)
        by_season[str(season)] = {
            "rows": sub.height,
            "active": float(sub["is_active_label"].drop_nulls().mean()),
            "participated": float(sub["participated_label"].drop_nulls().mean()),
            "positive_usage": float(sub["positive_usage_label"].drop_nulls().mean()),
        }
    out["by_season"] = by_season
    by_pos: dict[str, Any] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        sub = scheduled.filter(pl.col("position") == pos)
        if sub.is_empty():
            continue
        by_pos[pos] = {
            "rows": sub.height,
            "active": float(sub["is_active_label"].drop_nulls().mean()),
            "participated": float(sub["participated_label"].drop_nulls().mean()),
            "positive_usage": float(sub["positive_usage_label"].drop_nulls().mean()),
        }
    out["by_position"] = by_pos
    if "zero_class" in df.columns:
        vc = df["zero_class"].value_counts()
        out["zero_class_counts"] = {
            str(r["zero_class"]): int(r["count"]) for r in vc.iter_rows(named=True)
        }
    return out


def persist_mixture_panel(
    df: pl.DataFrame,
    output_dir: Path,
    *,
    source_panel_path: Path | None = None,
    cutoff_note: str = "features as-of pre-kickoff; labels from same-week outcomes for training only",
) -> MixturePanelArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mixture_panel.parquet"
    df.write_parquet(path)
    source_hash = _sha256_file(source_panel_path) if source_panel_path and source_panel_path.exists() else ""
    rates = summarize_event_rates(df)
    meta = {
        "schema_version": MIXTURE_PANEL_SCHEMA_VERSION,
        "cutoff_note": cutoff_note,
        "row_count": df.height,
        "source_panel_hash": source_hash,
        "panel_hash": _sha256_file(path),
        "event_rates": rates,
        "event_label_cols": list(EVENT_LABEL_COLS),
        "conditional_target_cols": list(CONDITIONAL_TARGET_COLS),
    }
    (output_dir / "mixture_panel_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return MixturePanelArtifact(
        path=path,
        schema_version=MIXTURE_PANEL_SCHEMA_VERSION,
        row_count=df.height,
        cutoff_note=cutoff_note,
        panel_hash=str(meta["panel_hash"]),
        source_panel_hash=source_hash,
        event_rates=rates,
    )
