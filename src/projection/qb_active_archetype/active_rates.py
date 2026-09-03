"""H1: separate active-start rates from availability / expected games."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.qb_active_archetype.thresholds import (
    ACTIVE_MIN_ATTEMPTS,
    ACTIVE_MIN_TOUCHES,
    AVAIL_FULL_SEASON_GAMES,
    AVAIL_LOOKBACK_SEASONS,
    AVAIL_PRIOR_STRENGTH_GAMES,
    LEAGUE_PARTIAL_EXIT_RATE,
    LEAGUE_STARTER_EXPECTED_GAMES,
    PARTIAL_MAX_ATTEMPTS,
    PARTIAL_MIN_ATTEMPTS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / "data" / "raw" / "weekly_qb_repair_cache"
HIST = REPO_ROOT / "output" / "qb_repair" / "history" / "qb_season_rates.parquet"

RATE_COLS = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
)


def _is_active_start(row: pd.Series) -> bool:
    att = float(pd.to_numeric(row.get("attempts"), errors="coerce") or 0.0)
    car = float(pd.to_numeric(row.get("carries"), errors="coerce") or 0.0)
    return (att >= ACTIVE_MIN_ATTEMPTS) or ((att + car) >= ACTIVE_MIN_TOUCHES)


def _is_partial_exit(row: pd.Series) -> bool:
    """Early-exit / partial proxy using only same-week box score (post hoc label).

    For *prediction*, we never use the target season's partial flags — only
    historical rates estimated from prior seasons.
    """
    att = float(pd.to_numeric(row.get("attempts"), errors="coerce") or 0.0)
    if att < PARTIAL_MIN_ATTEMPTS or att > PARTIAL_MAX_ATTEMPTS:
        return False
    # Exclude pure kneel / garbage with zero meaningful pass volume already handled.
    return True


def load_weekly_qb() -> pd.DataFrame:
    path = CACHE / "qb_weekly.parquet"
    if not path.exists():
        return pd.DataFrame()
    w = pd.read_parquet(path)
    w = w[pd.to_numeric(w["week"], errors="coerce").between(1, 18)].copy()
    if "season_type" in w.columns:
        st = w["season_type"].astype(str).str.upper()
        w = w[st.isin(["REG", "NAN", "NONE", ""]) | w["season_type"].isna()]
    return w


def annotate_weekly_activity(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    out["active_start"] = out.apply(_is_active_start, axis=1)
    out["partial_exit"] = out.apply(_is_partial_exit, axis=1) & ~out["active_start"]
    return out


def build_active_season_rates(weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per player-season rates from active starts only (+ availability counts)."""
    if weekly is None:
        weekly = load_weekly_qb()
    if weekly.empty:
        return pd.DataFrame()
    w = annotate_weekly_activity(weekly)
    rows = []
    name_col = "player_display_name" if "player_display_name" in w.columns else "player_name"
    team_col = "recent_team" if "recent_team" in w.columns else "team"
    for (pid, season), g in w.groupby(["player_id", "season"]):
        active = g[g["active_start"]]
        partial = g[g["partial_exit"]]
        n_active = int(len(active))
        n_partial = int(len(partial))
        n_weeks = int(g["week"].nunique())
        row = {
            "player_id": str(pid),
            "season": int(season),
            "display_name": g[name_col].iloc[0] if name_col in g.columns else None,
            "team": g[team_col].iloc[0] if team_col in g.columns else None,
            "weeks_rostered_proxy": n_weeks,
            "active_starts": n_active,
            "partial_exits": n_partial,
            "partial_exit_rate": float(n_partial / max(n_active + n_partial, 1)),
        }
        for col in RATE_COLS:
            if col not in g.columns:
                continue
            total = float(pd.to_numeric(g[col], errors="coerce").fillna(0).sum())
            active_total = float(pd.to_numeric(active[col], errors="coerce").fillna(0).sum()) if n_active else 0.0
            row[f"{col}_season"] = total
            row[f"{col}_per_active"] = active_total / n_active if n_active else np.nan
            # Conflated rate: season total / weeks with any row (injury-diluted).
            row[f"{col}_per_game_conflated"] = total / n_weeks if n_weeks else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def merge_rush_splits(active: pd.DataFrame, season_hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach designed/scramble season totals; convert to per-active when possible."""
    if season_hist is None and HIST.exists():
        season_hist = pd.read_parquet(HIST)
    out = active.copy()
    empty_cols = (
        "designed_carries_per_active",
        "scramble_carries_per_active",
        "designed_rushing_yards_per_active",
        "scramble_rushing_yards_per_active",
        "scramble_per_dropback",
        "designed_ypc",
        "scramble_ypa",
    )
    if season_hist is None or season_hist.empty:
        for c in empty_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out
    cols = [
        c
        for c in (
            "player_id",
            "season",
            "designed_carries",
            "scramble_carries",
            "designed_rushing_yards",
            "scramble_rushing_yards",
            "games",
        )
        if c in season_hist.columns
    ]
    m = season_hist[cols].copy()
    m["player_id"] = m["player_id"].astype(str)
    # Drop prior split cols before re-merge to avoid _x/_y collisions.
    drop_cols = [
        c
        for c in (
            "designed_carries",
            "scramble_carries",
            "designed_rushing_yards",
            "scramble_rushing_yards",
            "games",
        )
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(m, on=["player_id", "season"], how="left")
    act = pd.to_numeric(out["active_starts"], errors="coerce").replace(0, np.nan)
    des = pd.to_numeric(out["designed_carries"], errors="coerce") if "designed_carries" in out.columns else pd.Series(np.nan, index=out.index)
    scr = pd.to_numeric(out["scramble_carries"], errors="coerce") if "scramble_carries" in out.columns else pd.Series(np.nan, index=out.index)
    des_yds = pd.to_numeric(out["designed_rushing_yards"], errors="coerce") if "designed_rushing_yards" in out.columns else pd.Series(np.nan, index=out.index)
    scr_yds = pd.to_numeric(out["scramble_rushing_yards"], errors="coerce") if "scramble_rushing_yards" in out.columns else pd.Series(np.nan, index=out.index)
    out["designed_carries_per_active"] = des / act
    out["scramble_carries_per_active"] = scr / act
    out["designed_rushing_yards_per_active"] = des_yds / act
    out["scramble_rushing_yards_per_active"] = scr_yds / act
    dropbacks = pd.to_numeric(out.get("attempts_per_active"), errors="coerce")
    out["scramble_per_dropback"] = out["scramble_carries_per_active"] / dropbacks.replace(0, np.nan)
    out["designed_ypc"] = out["designed_rushing_yards_per_active"] / out["designed_carries_per_active"].replace(0, np.nan)
    out["scramble_ypa"] = out["scramble_rushing_yards_per_active"] / out["scramble_carries_per_active"].replace(0, np.nan)
    return out


def append_eval_season_active(active: pd.DataFrame, season: int) -> pd.DataFrame:
    """When weekly rows are missing (e.g. 2025), approximate from fantasy_evaluation.

    Approximation: treat actual_games_played as active starts (no week-level
    partial split). Marked with provenance flag.
    """
    path = REPO_ROOT / "output" / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        return active
    ev = pd.read_csv(path)
    qb = ev[ev["preseason_position"].astype(str).eq("QB")].copy()
    rows = []
    for _, r in qb.iterrows():
        games = float(pd.to_numeric(r.get("actual_games_played"), errors="coerce") or 0.0)
        if games <= 0:
            continue
        pid = str(r["player_id"])
        if ((active["player_id"] == pid) & (active["season"] == season)).any():
            continue
        row = {
            "player_id": pid,
            "season": season,
            "display_name": r.get("display_name"),
            "team": r.get("preseason_team"),
            "weeks_rostered_proxy": games,
            "active_starts": games,
            "partial_exits": 0.0,
            "partial_exit_rate": 0.0,
            "provenance": "fantasy_evaluation_approx",
        }
        for col in RATE_COLS:
            total = float(pd.to_numeric(r.get(col), errors="coerce") or 0.0)
            row[f"{col}_season"] = total
            row[f"{col}_per_active"] = total / games
            row[f"{col}_per_game_conflated"] = total / games
        rows.append(row)
    if not rows:
        return active
    extra = pd.DataFrame(rows)
    out = pd.concat([active, extra], ignore_index=True)
    return out


def expected_availability(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
) -> dict:
    """Expected active starts + partial-exit rate using only seasons < target."""
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - AVAIL_LOOKBACK_SEASONS)
    ].copy()
    if hist.empty:
        return {
            "expected_active_starts": LEAGUE_STARTER_EXPECTED_GAMES,
            "partial_exit_rate": LEAGUE_PARTIAL_EXIT_RATE,
            "sample_active_starts": 0.0,
            "input_seasons": [],
            "method": "league_prior",
        }
    # Games-weight: prefer full seasons.
    w = pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # Cap each season's contribution at a full season.
    season_games = w.clip(upper=AVAIL_FULL_SEASON_GAMES)
    # Empirical mean active starts per season (not diluted by missing seasons
    # beyond the lookback window length).
    emp = float(season_games.mean()) if len(season_games) else LEAGUE_STARTER_EXPECTED_GAMES
    n = float(w.sum())
    shrink = n / (n + AVAIL_PRIOR_STRENGTH_GAMES)
    expected = shrink * emp + (1.0 - shrink) * LEAGUE_STARTER_EXPECTED_GAMES
    expected = float(np.clip(expected, 1.0, AVAIL_FULL_SEASON_GAMES))
    partial = pd.to_numeric(hist["partial_exit_rate"], errors="coerce")
    pw = w.where(partial.notna(), 0.0)
    if pw.sum() > 0:
        emp_partial = float(np.average(partial.dropna(), weights=pw[partial.notna()]))
    else:
        emp_partial = LEAGUE_PARTIAL_EXIT_RATE
    partial_rate = shrink * emp_partial + (1.0 - shrink) * LEAGUE_PARTIAL_EXIT_RATE
    return {
        "expected_active_starts": expected,
        "partial_exit_rate": float(np.clip(partial_rate, 0.0, 0.5)),
        "sample_active_starts": float(n),
        "input_seasons": [int(s) for s in hist["season"].tolist()],
        "method": "player_empirical_shrunk",
        "empirical_active_starts_mean": emp,
    }


def pooled_active_rate(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
    rate_col: str,
    lookback: int = AVAIL_LOOKBACK_SEASONS,
) -> dict:
    """Games-weighted active-start rate from seasons < target only."""
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - lookback)
    ].copy()
    if hist.empty or rate_col not in hist.columns:
        return {"value": None, "input_seasons": [], "sample_active_starts": 0.0}
    vals = pd.to_numeric(hist[rate_col], errors="coerce")
    w = pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0)
    mask = vals.notna() & w.gt(0)
    if not mask.any():
        return {"value": None, "input_seasons": [], "sample_active_starts": 0.0}
    return {
        "value": float(np.average(vals[mask], weights=w[mask])),
        "input_seasons": [int(s) for s in hist.loc[mask, "season"].tolist()],
        "sample_active_starts": float(w[mask].sum()),
    }


def player_decomposition(
    history: pd.DataFrame,
    *,
    player_id: str,
    seasons: tuple[int, ...] = (2023, 2024, 2025),
) -> list[dict]:
    """Show how shortened seasons affect per-start rate vs expected games."""
    out = []
    for season in seasons:
        row = history[(history.player_id.astype(str) == str(player_id)) & (history.season == season)]
        avail = expected_availability(history, player_id=player_id, target_season=season + 1)
        if row.empty:
            out.append({"season": season, "missing": True, "forward_availability": avail})
            continue
        r = row.iloc[0]
        out.append(
            {
                "season": season,
                "active_starts": float(r.get("active_starts") or 0.0),
                "partial_exits": float(r.get("partial_exits") or 0.0),
                "attempts_per_active": float(r["attempts_per_active"])
                if pd.notna(r.get("attempts_per_active"))
                else None,
                "attempts_per_game_conflated": float(r["attempts_per_game_conflated"])
                if pd.notna(r.get("attempts_per_game_conflated"))
                else None,
                "carries_per_active": float(r["carries_per_active"])
                if pd.notna(r.get("carries_per_active"))
                else None,
                "carries_per_game_conflated": float(r["carries_per_game_conflated"])
                if pd.notna(r.get("carries_per_game_conflated"))
                else None,
                "passing_yards_per_active": float(r["passing_yards_per_active"])
                if pd.notna(r.get("passing_yards_per_active"))
                else None,
                "designed_carries_per_active": float(r["designed_carries_per_active"])
                if pd.notna(r.get("designed_carries_per_active"))
                else None,
                "scramble_carries_per_active": float(r["scramble_carries_per_active"])
                if pd.notna(r.get("scramble_carries_per_active"))
                else None,
                "forward_availability_for_next_season": avail,
                "note": (
                    "Short seasons reduce active_starts / availability; "
                    "attempts_per_active stays near healthy starter levels when active."
                    if float(r.get("active_starts") or 0) < 14
                    else "Full-ish season"
                ),
            }
        )
    return out
