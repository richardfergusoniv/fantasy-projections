/** Shared historical stats table for Fantasy Tools player cards. */

const HISTORY_COLS = [
  { key: "completions", label: "CMP", digits: { season: 0, pg: 1 } },
  { key: "attempts", label: "ATT", digits: { season: 0, pg: 1 } },
  { key: "passing_yards", label: "PaYd", digits: { season: 0, pg: 1 } },
  { key: "passing_tds", label: "PaTD", digits: { season: 1, pg: 2 } },
  { key: "interceptions", label: "INT", digits: { season: 1, pg: 2 } },
  { key: "rushing_yards", label: "RuYd", digits: { season: 0, pg: 1 } },
  { key: "rushing_tds", label: "RuTD", digits: { season: 1, pg: 2 } },
  { key: "targets", label: "TGT", digits: { season: 0, pg: 1 } },
  { key: "receptions", label: "REC", digits: { season: 0, pg: 1 } },
  { key: "receiving_yards", label: "ReYd", digits: { season: 0, pg: 1 } },
  { key: "receiving_tds", label: "ReTD", digits: { season: 1, pg: 2 } },
];

function historyFmt(n, digits = 1) {
  if (n == null || Number.isNaN(n)) return "—";
  if (digits === 0) return Math.round(n).toLocaleString();
  if (Math.abs(n) >= 100 && digits <= 1) return Math.round(n).toLocaleString();
  return Number(n).toFixed(digits);
}

function historyStatValue(row, key, perGame) {
  const raw = row[key];
  if (raw == null || Number.isNaN(raw)) return null;
  if (!perGame) return raw;
  const gp = row.games;
  if (gp == null || gp === 0) return null;
  return raw / gp;
}

function projectionHistoryRow(p, season) {
  const seasonStats = p.season || {};
  const row = {
    season,
    label: String(season),
    isProj: true,
    games: p.projected_games,
    fantasy_pts: p.fantasy_pts,
    fantasy_pts_season: p.fantasy_pts_season,
  };
  for (const col of HISTORY_COLS) {
    row[col.key] = seasonStats[col.key];
  }
  return row;
}

function buildHistoryTableHtml(p, { perGame = false, season = 2026 } = {}) {
  const rows = [projectionHistoryRow(p, season)].concat(p.history || []);
  const mode = perGame ? "pg" : "season";
  const title = perGame ? "History (per game)" : "History (season totals)";

  if (!(p.history || []).length && !(p.season && Object.keys(p.season).length)) {
    return `<div class="driver-section">
      <h4>${title}</h4>
      <p class="driver-note">No historical box-score seasons available. Re-run team_stats.prepare to attach prior years.</p>
    </div>`;
  }

  const head = `<tr>
    <th class="col-year">Year</th>
    <th>GP</th>
    ${HISTORY_COLS.map((c) => `<th>${c.label}</th>`).join("")}
    <th>FPTS</th>
  </tr>`;

  const body = rows
    .map((row) => {
      const gp = row.games;
      const fpts = perGame
        ? row.fantasy_pts
        : row.fantasy_pts_season != null
          ? row.fantasy_pts_season
          : row.fantasy_pts != null && gp
            ? row.fantasy_pts * gp
            : null;
      const yearLabel = row.isProj ? `${row.label} proj` : String(row.season);
      const cells = HISTORY_COLS.map((c) => {
        const v = historyStatValue(row, c.key, perGame);
        return `<td>${historyFmt(v, c.digits[mode])}</td>`;
      }).join("");
      return `<tr class="${row.isProj ? "is-proj" : ""}">
        <td class="col-year">${escapeHtml(yearLabel)}</td>
        <td>${historyFmt(gp, 1)}</td>
        ${cells}
        <td class="num-strong">${historyFmt(fpts, 1)}</td>
      </tr>`;
    })
    .join("");

  return `<div class="driver-section">
    <h4>${title}</h4>
    <div class="history-table-wrap">
      <table class="history-table">
        <thead>${head}</thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    <p class="driver-note">Half-PPR · 4-pt pass TD. Projection row uses ${season} model output; prior years are REG season actuals.</p>
  </div>`;
}
