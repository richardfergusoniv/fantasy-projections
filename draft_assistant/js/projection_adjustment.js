/** Presentation-only stat normalization for blended draft forecasts. */
(function exposeProjectionAdjustment(global) {
  const STAT_KEYS = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
  ];

  function numberOrNull(value) {
    if (value == null || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function scoreStats(stats, scoring) {
    if (!stats) return null;
    const terms = [
      ["passing_yards", scoring.passYd],
      ["passing_tds", scoring.passTd],
      ["interceptions", scoring.int],
      ["rushing_yards", scoring.rushYd],
      ["rushing_tds", scoring.rushTd],
      ["receptions", scoring.rec],
      ["receiving_yards", scoring.recYd],
      ["receiving_tds", scoring.recTd],
    ];
    let found = false;
    let total = 0;
    for (const [key, weight] of terms) {
      const value = numberOrNull(stats[key]);
      if (value == null) continue;
      found = true;
      total += value * weight;
    }
    return found && Number.isFinite(total) ? total : null;
  }

  function scaleStats(stats, factor) {
    if (!stats || !Number.isFinite(factor) || factor <= 0) return stats || {};
    const scaled = { ...stats };
    for (const key of STAT_KEYS) {
      const value = numberOrNull(stats[key]);
      scaled[key] = value == null ? stats[key] ?? null : value * factor;
    }
    return scaled;
  }

  function scaleFor(target, baseline) {
    const targetValue = numberOrNull(target);
    const baselineValue = numberOrNull(baseline);
    if (targetValue == null || baselineValue == null || baselineValue <= 0) return null;
    const factor = targetValue / baselineValue;
    return Number.isFinite(factor) && factor > 0 ? factor : null;
  }

  function derive(draft, detail, scoring) {
    const canonicalPg = detail?.pg || {};
    const canonicalSeason = detail?.season || {};
    const canonicalPgPoints = scoreStats(canonicalPg, scoring);
    const canonicalSeasonPoints = scoreStats(canonicalSeason, scoring);
    const pgFactor = scaleFor(draft?.fantasy_pts, canonicalPgPoints);
    const seasonFactor = scaleFor(draft?.fantasy_pts_season, canonicalSeasonPoints);
    const adjusted = Boolean(
      (pgFactor != null && Math.abs(pgFactor - 1) > 1e-6) ||
        (seasonFactor != null && Math.abs(seasonFactor - 1) > 1e-6)
    );

    return {
      pg: pgFactor == null ? canonicalPg : scaleStats(canonicalPg, pgFactor),
      season:
        seasonFactor == null ? canonicalSeason : scaleStats(canonicalSeason, seasonFactor),
      meta: {
        adjusted,
        method: adjusted ? "proportional_stat_mix" : "canonical",
        pg_factor: pgFactor,
        season_factor: seasonFactor,
        canonical_pg_points: canonicalPgPoints,
        canonical_season_points: canonicalSeasonPoints,
        target_pg_points: numberOrNull(draft?.fantasy_pts),
        target_season_points: numberOrNull(draft?.fantasy_pts_season),
      },
    };
  }

  function mergeBoard(players, boardPlayers, scoring, modelId = null) {
    const targets = new Map(
      (boardPlayers || []).map((player) => [String(player.player_id), player])
    );
    return (players || []).map((detail) => {
      const draft = targets.get(String(detail.player_id));
      if (!draft) return detail;
      const adjusted = derive(draft, detail, scoring);
      return {
        ...detail,
        pg: adjusted.pg,
        season: adjusted.season,
        canonical_pg: detail.pg || {},
        canonical_season: detail.season || {},
        projection_adjustment: adjusted.meta,
        projection_model_id: modelId,
        fantasy_pts: draft.fantasy_pts ?? detail.fantasy_pts,
        fantasy_pts_season: draft.fantasy_pts_season ?? detail.fantasy_pts_season,
      };
    });
  }

  global.FantasyProjectionAdjustment = { derive, mergeBoard, scaleStats, scoreStats };
})(typeof window !== "undefined" ? window : globalThis);
