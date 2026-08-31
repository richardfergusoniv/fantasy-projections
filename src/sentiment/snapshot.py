"""Build and attach the auditable player-sentiment snapshot."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.sentiment.markdown import RESEARCH_AS_OF, norm_name, parse_research_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_DIR = REPO_ROOT / "perplexity research"
DEFAULT_CONSENSUS_DIR = REPO_ROOT / "data" / "consensus"
DEFAULT_SENTIMENT_DIR = REPO_ROOT / "data" / "sentiment"
DEFAULT_MANIFEST = REPO_ROOT / "models" / "sentiment_manifest.json"
SENTIMENT_VERSION = "markdown_market_v1"

SENTIMENT_OUTPUT_COLS = [
    "sentiment_score",
    "sentiment_feature",
    "sentiment_confidence",
    "sentiment_coverage",
    "sentiment_as_of",
    "sentiment_claim_count",
    "sentiment_source_count",
    "sentiment_model_active",
    "sentiment_version",
]


def _coerce_as_of(value: str | date | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _robust_z(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(valid.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        return pd.Series(0.0, index=values.index).where(values.notna())
    return ((values - median) / scale).clip(-3.0, 3.0)


def _position_z(frame: pd.DataFrame, column: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby("position", dropna=False).groups.items():
        out.loc[idx] = _robust_z(frame.loc[idx, column])
    return out


def _load_market(season: int, consensus_dir: str | Path) -> pd.DataFrame:
    path = Path(consensus_dir) / f"consensus_{season}.json"
    columns = ["player_id", "market_name", "position", "market_ecr", "market_adp"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload.get("rows") or [])
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows = rows.rename(
        columns={"display_name": "market_name", "ecr": "market_ecr", "adp": "market_adp"}
    )
    for col in columns:
        if col not in rows:
            rows[col] = np.nan
    return rows[columns]


def _join_market(players: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out = players[["player_id", "display_name", "team", "position"]].copy()
    out["player_id"] = out["player_id"].astype(str)
    if market.empty:
        out["market_ecr"] = np.nan
        out["market_adp"] = np.nan
        return out
    market = market.copy()
    market["player_id"] = market["player_id"].astype(str)
    by_id = market.drop_duplicates("player_id").set_index("player_id")
    by_name = (
        market.assign(_norm=market["market_name"].map(norm_name))
        .drop_duplicates(["_norm", "position"])
        .set_index(["_norm", "position"])
    )
    ecr, adp = [], []
    for row in out.itertuples(index=False):
        rec = None
        if row.player_id in by_id.index:
            rec = by_id.loc[row.player_id]
        else:
            key = (norm_name(row.display_name), row.position)
            if key in by_name.index:
                rec = by_name.loc[key]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        ecr.append(rec.get("market_ecr") if rec is not None else np.nan)
        adp.append(rec.get("market_adp") if rec is not None else np.nan)
    out["market_ecr"] = pd.to_numeric(pd.Series(ecr), errors="coerce")
    out["market_adp"] = pd.to_numeric(pd.Series(adp), errors="coerce")
    return out


def _residualize(frame: pd.DataFrame) -> pd.Series:
    """Remove observable role/availability structure without outcome labels."""
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    controls = [
        "projected_games_raw", "projected_games", "target_depth_rank",
        "nfl_depth_rank", "depth_rank", "team_changed", "low_confidence",
    ]
    for _, idx in frame.groupby("position").groups.items():
        valid_idx = [i for i in idx if pd.notna(frame.at[i, "raw_sentiment_z"])]
        if len(valid_idx) < 10:
            residual.loc[valid_idx] = frame.loc[valid_idx, "raw_sentiment_z"]
            continue
        columns = [c for c in controls if c in frame.columns]
        if not columns:
            residual.loc[valid_idx] = frame.loc[valid_idx, "raw_sentiment_z"]
            continue
        x = frame.loc[valid_idx, columns].copy()
        for col in columns:
            if x[col].dtype == bool or col in {"team_changed", "low_confidence"}:
                x[col] = x[col].fillna(False).astype(float)
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce")
                missing = x[col].isna().astype(float)
                x[f"{col}__missing"] = missing
                median = x[col].median()
                x[col] = x[col].fillna(0.0 if pd.isna(median) else median)
        std = x.std(ddof=0).replace(0, 1.0)
        x = (x - x.mean()) / std
        y = frame.loc[valid_idx, "raw_sentiment_z"].astype(float)
        model = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(x, y)
        residual.loc[valid_idx] = y - model.predict(x)
    return residual


def _manifest_active(path: str | Path) -> dict[str, bool]:
    path = Path(path)
    if not path.exists():
        return {position: False for position in ("QB", "RB", "WR", "TE")}
    payload = json.loads(path.read_text(encoding="utf-8"))
    active = payload.get("active_by_position") or {}
    return {position: bool(active.get(position, False)) for position in ("QB", "RB", "WR", "TE")}


def build_sentiment_snapshot(
    players: pd.DataFrame,
    *,
    season: int,
    as_of: str | date | None = None,
    research_dir: str | Path = DEFAULT_RESEARCH_DIR,
    consensus_dir: str | Path = DEFAULT_CONSENSUS_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> pd.DataFrame:
    """Return one row per projected player, including explicit no-signal rows."""
    as_of_date = _coerce_as_of(as_of)
    # Rebuilding from an already-enriched projection/fantasy artifact must be
    # idempotent; remove the prior public fields before merging fresh evidence.
    base = (
        players.drop(columns=SENTIMENT_OUTPUT_COLS, errors="ignore")
        .drop_duplicates("player_id")
        .copy()
        .reset_index(drop=True)
    )
    required = {"player_id", "display_name", "team", "position"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"sentiment players frame missing columns: {sorted(missing)}")
    base["player_id"] = base["player_id"].astype(str)

    text = parse_research_directory(base, research_dir, as_of=as_of_date)
    market = _join_market(base, _load_market(season, consensus_dir))
    out = base.merge(
        text.drop(columns=["display_name", "team", "position"], errors="ignore"),
        on="player_id",
        how="left",
    ).merge(
        market[["player_id", "market_ecr", "market_adp"]], on="player_id", how="left"
    )

    out["market_gap_raw"] = -(out["market_ecr"] - out["market_adp"])
    out["text_sentiment_z"] = _position_z(out, "text_sentiment_raw")
    out["market_gap_z"] = _position_z(out, "market_gap_raw")
    out["text_confidence"] = pd.to_numeric(out.get("text_confidence"), errors="coerce").fillna(0.0)
    out["market_confidence"] = out["market_gap_z"].notna().astype(float) / 3.0

    text_weight = out["text_confidence"].where(out["text_sentiment_z"].notna(), 0.0)
    market_weight = out["market_confidence"].where(out["market_gap_z"].notna(), 0.0)
    weight_sum = text_weight + market_weight
    numerator = (
        out["text_sentiment_z"].fillna(0.0) * text_weight
        + out["market_gap_z"].fillna(0.0) * market_weight
    )
    out["raw_sentiment_z"] = (numerator / weight_sum.replace(0.0, np.nan)).clip(-3, 3)
    out["sentiment_confidence"] = 1.0 - (
        (1.0 - text_weight.clip(0, 1)) * (1.0 - market_weight.clip(0, 1))
    )
    out["sentiment_residual_z"] = _residualize(out)
    out["sentiment_residual_z"] = _position_z(out, "sentiment_residual_z")
    out["sentiment_feature"] = out["sentiment_residual_z"] * out["sentiment_confidence"]

    out["sentiment_score"] = np.nan
    for _, idx in out.groupby("position").groups.items():
        valid = out.loc[idx, "sentiment_residual_z"].dropna()
        if valid.empty:
            continue
        pct = valid.rank(method="average", pct=True)
        out.loc[valid.index, "sentiment_score"] = (200.0 * (pct - 0.5)).round()

    families = out["text_sentiment_z"].notna().astype(int) + out["market_gap_z"].notna().astype(int)
    conditions = [
        families.eq(0),
        families.eq(2) & out["sentiment_confidence"].ge(0.7),
        families.eq(2) | out["sentiment_confidence"].ge(0.6),
    ]
    out["sentiment_coverage"] = np.select(
        conditions, ["none", "high", "medium"], default="low"
    )
    out["sentiment_as_of"] = as_of_date.isoformat()
    claim_count = (
        out["sentiment_claim_count"]
        if "sentiment_claim_count" in out else pd.Series(0, index=out.index)
    )
    source_count = (
        out["sentiment_source_count"]
        if "sentiment_source_count" in out else pd.Series(0, index=out.index)
    )
    out["sentiment_claim_count"] = claim_count.fillna(0).astype(int)
    out["sentiment_source_count"] = (
        source_count.fillna(0).astype(int)
        + out["market_gap_z"].notna().astype(int)
    )
    active = _manifest_active(manifest_path)
    out["sentiment_model_active"] = out["position"].map(active).fillna(False).astype(bool)
    out["sentiment_version"] = SENTIMENT_VERSION
    out["season"] = int(season)
    audit_cols = [
        "player_id", "display_name", "team", "position", "season",
        "text_sentiment_raw", "text_sentiment_z", "text_confidence",
        "sentiment_label", "sentiment_source_ref", "sentiment_parse_method",
        "market_ecr", "market_adp", "market_gap_raw", "market_gap_z",
        "market_confidence", "raw_sentiment_z", "sentiment_residual_z",
        *SENTIMENT_OUTPUT_COLS,
    ]
    return out[[c for c in audit_cols if c in out.columns]]


def attach_sentiment(
    frame: pd.DataFrame,
    *,
    season: int,
    as_of: str | date | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Attach summary sentiment fields to a player- or stat-grain frame."""
    snapshot = build_sentiment_snapshot(frame, season=season, as_of=as_of, **kwargs)
    keys = ["player_id", *SENTIMENT_OUTPUT_COLS]
    out = frame.copy()
    out["player_id"] = out["player_id"].astype(str)
    return out.merge(snapshot[keys], on="player_id", how="left", validate="many_to_one")


def coverage_report(snapshot: pd.DataFrame) -> dict:
    teams = sorted(snapshot["team"].dropna().unique().tolist())
    return {
        "version": SENTIMENT_VERSION,
        "players": int(len(snapshot)),
        "teams": len(teams),
        "team_codes": teams,
        "non_null_scores": int(snapshot["sentiment_score"].notna().sum()),
        "coverage": snapshot["sentiment_coverage"].value_counts().to_dict(),
        "active_by_position": {
            pos: bool(group["sentiment_model_active"].any())
            for pos, group in snapshot.groupby("position")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of", default=RESEARCH_AS_OF.isoformat())
    parser.add_argument("--players-path", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    players_path = Path(args.players_path) if args.players_path else (
        REPO_ROOT / "output" / f"fantasy_points_{args.season}.csv"
    )
    players = pd.read_csv(players_path)
    snapshot = build_sentiment_snapshot(players, season=args.season, as_of=args.as_of)
    DEFAULT_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        DEFAULT_SENTIMENT_DIR / f"sentiment_{args.season}_{args.as_of[:10]}.csv"
    )
    snapshot.to_csv(out_path, index=False)
    report = coverage_report(snapshot)
    report_path = Path(args.report) if args.report else out_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
