const SEASON = 2026;

// Half-PPR, 4pt passing TD — matches fantasy_points.py
const SCORING = {
  passYd: 0.04,
  passTd: 4,
  int: -2,
  rushYd: 0.1,
  rushTd: 6,
  rec: 0.5,
  recYd: 0.1,
  recTd: 6,
};

const state = {
  data: null,
  byId: new Map(),
  team: "PHI",
  view: "season", // "season" | "pg"
  section: "passing",
  sort: {
    passing: { key: "yds", dir: "desc" },
    rushing: { key: "yds", dir: "desc" },
    receiving: { key: "yds", dir: "desc" },
  },
  hoverId: null,
  hoverTimer: null,
};

const els = {};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheElements();
  bindEvents();

  try {
    const [res, boardRes] = await Promise.all([
      fetch(`../data/team_stats_${SEASON}.json?v=${Date.now()}`, { cache: "no-store" }),
      fetch(`../data/players_${SEASON}.json?v=${Date.now()}`, { cache: "no-store" }),
    ]);
    if (!res.ok) throw new Error(`Failed to load projections (${res.status})`);
    state.data = await res.json();
    if (boardRes.ok) {
      const board = await boardRes.json();
      state.data.players = FantasyProjectionAdjustment.mergeBoard(
        state.data.players,
        board.players,
        SCORING,
        board.meta?.model_id || null
      );
      state.data.meta.display_board = board.meta || null;
    }
    state.byId = new Map(state.data.players.map((p) => [p.player_id, p]));
    els.seasonBadge.textContent = state.data.meta.season;
    populateTeamSelect();
    const params = new URLSearchParams(window.location.search);
    const teamParam = (params.get("team") || "").toUpperCase();
    if (teamParam && state.data.teams.some((t) => t.abbr === teamParam)) {
      state.team = teamParam;
      els.teamSelect.value = teamParam;
    }
    renderAll();
  } catch (err) {
    const msg = `${err.message}. Run: python -m src.team_stats.prepare --season ${SEASON}`;
    els.passingBody.innerHTML = emptyRow(10, msg);
    els.depthBody.innerHTML = `<tr><td colspan="5" class="empty-row">${escapeHtml(msg)}</td></tr>`;
  }
}

function cacheElements() {
  els.seasonBadge = document.getElementById("seasonBadge");
  els.teamSelect = document.getElementById("teamSelect");
  els.teamKicker = document.getElementById("teamKicker");
  els.teamName = document.getElementById("teamName");
  els.teamSub = document.getElementById("teamSub");
  els.teamSummary = document.getElementById("teamSummary");
  els.statTabs = document.getElementById("statTabs");
  els.passingHead = document.getElementById("passingHead");
  els.passingBody = document.getElementById("passingBody");
  els.rushingHead = document.getElementById("rushingHead");
  els.rushingBody = document.getElementById("rushingBody");
  els.receivingHead = document.getElementById("receivingHead");
  els.receivingBody = document.getElementById("receivingBody");
  els.passingCaption = document.getElementById("passingCaption");
  els.rushingCaption = document.getElementById("rushingCaption");
  els.receivingCaption = document.getElementById("receivingCaption");
  els.depthCaption = document.getElementById("depthCaption");
  els.depthBody = document.getElementById("depthBody");
  els.footerNote = document.getElementById("footerNote");
  els.viewSeason = document.getElementById("viewSeason");
  els.viewPg = document.getElementById("viewPg");
  els.hoverCard = document.getElementById("playerHoverCard");
  els.modal = document.getElementById("playerModal");
  els.modalBody = document.getElementById("playerModalBody");
  els.mainContent = document.getElementById("mainContent");
}

function bindEvents() {
  els.teamSelect.addEventListener("change", () => {
    state.team = els.teamSelect.value;
    const url = new URL(window.location.href);
    url.searchParams.set("team", state.team);
    window.history.replaceState({}, "", url);
    hideHoverCard();
    renderAll();
  });

  els.viewSeason.addEventListener("click", () => setView("season"));
  els.viewPg.addEventListener("click", () => setView("pg"));

  els.statTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".stat-tab");
    if (!btn) return;
    state.section = btn.dataset.section;
    document.querySelectorAll(".stat-tab").forEach((el) => {
      el.classList.toggle("active", el.dataset.section === state.section);
    });
    document.querySelectorAll(".stat-section").forEach((el) => {
      el.hidden = el.dataset.section !== state.section;
    });
    hideHoverCard();
  });

  els.mainContent.addEventListener("mouseover", onPlayerMouseOver);
  els.mainContent.addEventListener("mouseout", onPlayerMouseOut);
  els.mainContent.addEventListener("focusin", onPlayerFocusIn);
  els.mainContent.addEventListener("focusout", onPlayerFocusOut);
  els.mainContent.addEventListener("click", onPlayerClick);

  els.modal.addEventListener("click", (e) => {
    if (e.target.closest("[data-close-modal]")) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!els.modal.hidden) closeModal();
      else hideHoverCard();
    }
  });
  window.addEventListener("scroll", () => hideHoverCard(), { passive: true });
  window.addEventListener("resize", () => hideHoverCard());

  FantasySort.bindHeader(els.passingHead, (key, defaultDir) => {
    FantasySort.toggleSort(state.sort.passing, key, { defaultDir });
    renderPassing(teamPlayers());
  });
  FantasySort.bindHeader(els.rushingHead, (key, defaultDir) => {
    FantasySort.toggleSort(state.sort.rushing, key, { defaultDir });
    renderRushing(teamPlayers());
  });
  FantasySort.bindHeader(els.receivingHead, (key, defaultDir) => {
    FantasySort.toggleSort(state.sort.receiving, key, { defaultDir });
    renderReceiving(teamPlayers());
  });
}

function setView(view) {
  state.view = view;
  els.viewSeason.classList.toggle("active", view === "season");
  els.viewPg.classList.toggle("active", view === "pg");
  hideHoverCard();
  renderAll();
}

function populateTeamSelect() {
  const groups = { AFC: {}, NFC: {} };
  for (const team of state.data.teams) {
    if (!groups[team.conference][team.division]) {
      groups[team.conference][team.division] = [];
    }
    groups[team.conference][team.division].push(team);
  }

  const frag = document.createDocumentFragment();
  for (const conf of ["AFC", "NFC"]) {
    for (const div of ["East", "North", "South", "West"]) {
      const teams = groups[conf][div] || [];
      if (!teams.length) continue;
      const og = document.createElement("optgroup");
      og.label = `${conf} ${div}`;
      for (const t of teams) {
        const opt = document.createElement("option");
        opt.value = t.abbr;
        opt.textContent = t.name;
        if (t.abbr === state.team) opt.selected = true;
        og.appendChild(opt);
      }
      frag.appendChild(og);
    }
  }
  els.teamSelect.innerHTML = "";
  els.teamSelect.appendChild(frag);
}

function teamPlayers() {
  return state.data.players.filter((p) => p.team === state.team);
}

function statsOf(player) {
  return player[state.view] || {};
}

function fmt(n, digits = 1) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 100) return Math.round(n).toLocaleString();
  return Number(n).toFixed(digits);
}

function fmtPct(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return `${n.toFixed(1)}%`;
}

function avg(num, den) {
  if (num == null || den == null || den === 0) return null;
  return num / den;
}

function emptyRow(cols, msg) {
  return `<tr class="empty-row"><td colspan="${cols}">${escapeHtml(msg)}</td></tr>`;
}

function playerCell(p) {
  return `<td class="col-player">
    <div class="player-cell">
      <span class="pos-badge ${p.position}">${p.position}</span>
      <span class="player-name">
        <button type="button" class="name player-link" data-player-id="${escapeHtml(p.player_id)}" aria-haspopup="dialog">
          ${escapeHtml(p.display_name)}
        </button>
      </span>
    </div>
  </td>`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fantasyBreakdown(p) {
  const s = p.pg || {};
  const pass =
    SCORING.passYd * (s.passing_yards || 0) +
    SCORING.passTd * (s.passing_tds || 0) +
    SCORING.int * (s.interceptions || 0);
  const rush =
    SCORING.rushYd * (s.rushing_yards || 0) +
    SCORING.rushTd * (s.rushing_tds || 0);
  const rec =
    SCORING.rec * (s.receptions || 0) +
    SCORING.recYd * (s.receiving_yards || 0) +
    SCORING.recTd * (s.receiving_tds || 0);
  const total = pass + rush + rec;
  return { pass, rush, rec, total };
}


function buildCardHtml(p, { full = false } = {}) {
  const d = p.drivers || {};
  const br = fantasyBreakdown(p);
  const absTotal = Math.max(Math.abs(br.total), 0.01);
  const fpts = state.view === "pg" ? p.fantasy_pts : p.fantasy_pts_season;
  const fptsLow = state.view === "pg" ? d.fantasy_pts_low : d.fantasy_pts_season_low;
  const fptsHigh = state.view === "pg" ? d.fantasy_pts_high : d.fantasy_pts_season_high;
  const titleTag = full ? "h2" : "h3";
  const titleId = full ? ' id="playerModalTitle"' : "";

  const pills = [];
  pills.push(`<span class="pill">${escapeHtml(p.team || "FA")}</span>`);
  if (p.low_confidence) pills.push(`<span class="pill warn">Low confidence</span>`);
  else pills.push(`<span class="pill ok">Modeled</span>`);
  if (d.any_stat_low_n_flag) pills.push(`<span class="pill warn">Low-N interval</span>`);
  if (d.any_receiving_share_capped) pills.push(`<span class="pill warn">Share capped</span>`);
  if (d.role_discount_applied) pills.push(`<span class="pill warn">Role discounted</span>`);
  if (p.projection_adjustment?.adjusted) {
    pills.push(`<span class="pill ok">Blend-adjusted stats</span>`);
  }
  if (d.sentiment_model_active) pills.push(`<span class="pill ok">Sentiment active</span>`);
  else pills.push(`<span class="pill">Sentiment diagnostic</span>`);

  const contrib = [
    { label: "Passing", pts: br.pass, cls: "pass" },
    { label: "Rushing", pts: br.rush, cls: "rush" },
    { label: "Receiving", pts: br.rec, cls: "rec" },
  ].filter((c) => Math.abs(c.pts) >= 0.05);

  const gpHint =
    p.projected_games != null
      ? `${fmt(p.projected_games, 1)} projected games`
      : "Projected games unavailable";

  return `
    <div class="player-card-header">
      <span class="pos-badge ${p.position}">${p.position}</span>
      <div class="title-block">
        <${titleTag}${titleId}>${escapeHtml(p.display_name)}</${titleTag}>
        <p class="player-card-sub">${escapeHtml(p.team || "")} · ${escapeHtml(
    (p.role || "unlisted").replace(/_/g, " ")
  )} · what drives ${p.projection_adjustment?.adjusted ? "the blended forecast" : "this projection"}</p>
        <div class="player-card-pills">${pills.join("")}</div>
      </div>
    </div>

    <div class="card-stats">
      <div class="card-stat">
        <span class="label">${state.view === "pg" ? "Pts / G" : "Season Pts"}</span>
        <span class="value">${fmt(fpts, 1)}</span>
        <span class="hint">${
          fptsLow != null && fptsHigh != null
            ? `Range ${fmt(fptsLow, 1)} – ${fmt(fptsHigh, 1)}`
            : gpHint
        }</span>
      </div>
      <div class="card-stat">
        <span class="label">Availability</span>
        <span class="value">${fmt(p.projected_games, 1)}</span>
        <span class="hint">games</span>
      </div>
      <div class="card-stat">
        <span class="label">Depth</span>
        <span class="value">${p.depth_rank != null ? Math.round(p.depth_rank) : "—"}</span>
        <span class="hint">${escapeHtml((p.depth_chart_status || "chart").replace(/_/g, " "))}</span>
      </div>
      <div class="card-stat">
        <span class="label">Sentiment</span>
        <span class="value">${d.sentiment_score == null ? "—" : `${d.sentiment_score > 0 ? "+" : ""}${Math.round(d.sentiment_score)}`}</span>
        <span class="hint">${escapeHtml(d.sentiment_coverage || "no signal")} · ${d.sentiment_confidence == null ? 0 : Math.round(100 * Number(d.sentiment_confidence))}% confidence</span>
      </div>
    </div>

    <div class="driver-section">
      <h4>${p.projection_adjustment?.adjusted ? "Blend-adjusted" : "Fantasy points"} drivers (per game)</h4>
      ${
        contrib.length
          ? contrib
              .map((c) => {
                const pct = Math.min(100, Math.round((100 * Math.abs(c.pts)) / absTotal));
                return `<div class="driver-row">
                  <span class="driver-label">${c.label}</span>
                  <div class="driver-bar ${c.cls}"><span style="width:${pct}%"></span></div>
                  <span class="driver-value">${c.pts >= 0 ? "+" : ""}${fmt(c.pts, 1)} (${pct}%)</span>
                </div>`;
              })
              .join("")
          : `<p class="driver-note">No meaningful scoring volume projected.</p>`
      }
      <p class="driver-note">Half-PPR · 4-pt pass TD. Bars show share of this player's projected fantasy points.${
        p.projection_adjustment?.adjusted
          ? " Displayed stats preserve the canonical stat mix and reconcile to the blended point forecast."
          : ""
      }</p>
    </div>

    ${buildHistoryTableHtml(p, { perGame: state.view === "pg", season: SEASON })}
  `;
}

function onPlayerMouseOver(e) {
  const link = e.target.closest(".player-link");
  if (!link || els.modal.hidden === false) return;
  const id = link.dataset.playerId;
  if (!id || id === state.hoverId) return;
  clearTimeout(state.hoverTimer);
  state.hoverTimer = setTimeout(() => showHoverCard(id, link), 120);
}

function onPlayerMouseOut(e) {
  const link = e.target.closest(".player-link");
  if (!link) return;
  const next = e.relatedTarget && e.relatedTarget.closest
    ? e.relatedTarget.closest(".player-link")
    : null;
  if (next && next.dataset.playerId === link.dataset.playerId) return;
  clearTimeout(state.hoverTimer);
  state.hoverTimer = setTimeout(hideHoverCard, 80);
}

function onPlayerFocusIn(e) {
  const link = e.target.closest(".player-link");
  if (!link || !els.modal.hidden) return;
  showHoverCard(link.dataset.playerId, link);
}

function onPlayerFocusOut(e) {
  const link = e.target.closest(".player-link");
  if (!link) return;
  clearTimeout(state.hoverTimer);
  state.hoverTimer = setTimeout(hideHoverCard, 80);
}

function onPlayerClick(e) {
  const link = e.target.closest(".player-link");
  if (!link) return;
  e.preventDefault();
  hideHoverCard();
  openModal(link.dataset.playerId);
}

function showHoverCard(playerId, anchor) {
  const p = state.byId.get(playerId);
  if (!p) return;
  state.hoverId = playerId;
  els.hoverCard.innerHTML = buildCardHtml(p, { full: false });
  els.hoverCard.hidden = false;
  positionHoverCard(anchor);
}

function positionHoverCard(anchor) {
  const rect = anchor.getBoundingClientRect();
  const card = els.hoverCard;
  const pad = 12;
  const cw = card.offsetWidth;
  const ch = card.offsetHeight;
  let left = rect.right + 10;
  let top = rect.top;
  if (left + cw > window.innerWidth - pad) left = rect.left - cw - 10;
  if (left < pad) left = pad;
  if (top + ch > window.innerHeight - pad) top = window.innerHeight - ch - pad;
  if (top < pad) top = pad;
  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
}

function hideHoverCard() {
  clearTimeout(state.hoverTimer);
  state.hoverId = null;
  els.hoverCard.hidden = true;
  els.hoverCard.innerHTML = "";
}

function openModal(playerId) {
  const p = state.byId.get(playerId);
  if (!p) return;
  els.modalBody.innerHTML = buildCardHtml(p, { full: true });
  els.modal.hidden = false;
  document.body.classList.add("modal-open");
}

function closeModal() {
  els.modal.hidden = true;
  els.modalBody.innerHTML = "";
  document.body.classList.remove("modal-open");
}

function renderAll() {
  const team = state.data.teams.find((t) => t.abbr === state.team);
  if (!team) return;

  const players = teamPlayers();
  const isPg = state.view === "pg";
  const blendApplied = players.some((p) => p.projection_adjustment?.adjusted);
  const caption = `${blendApplied ? "Blend-adjusted " : ""}${
    isPg ? "projected per game" : "projected season totals"
  }`;

  els.teamKicker.textContent = `${team.conference} ${team.division}`;
  els.teamName.textContent = team.name;
  els.teamSub.textContent = `${state.data.meta.season} ${
    blendApplied ? "blend-adjusted " : ""
  }projected offensive stats · ${players.length} players`;
  els.passingCaption.textContent = caption;
  els.rushingCaption.textContent = caption;
  els.receivingCaption.textContent = caption;
  els.depthCaption.textContent = "Offense";

  renderSummary(players);
  renderPassing(players);
  renderRushing(players);
  renderReceiving(players);
  renderDepthChart(players);

  const boardId = state.data.meta.display_board?.model_id;
  els.footerNote.textContent = `Source: ${state.data.meta.source_file}${
    boardId ? ` · displayed against ${boardId}` : ""
  } · generated ${new Date(state.data.meta.generated_at).toLocaleString()}`;
}

function renderSummary(players) {
  const isPg = state.view === "pg";
  let passYds = 0;
  let rushYds = 0;
  let recYds = 0;
  let passTd = 0;
  let rushTd = 0;
  let recTd = 0;

  for (const p of players) {
    const s = p.season || {};
    passYds += s.passing_yards || 0;
    rushYds += s.rushing_yards || 0;
    recYds += s.receiving_yards || 0;
    passTd += s.passing_tds || 0;
    rushTd += s.rushing_tds || 0;
    recTd += s.receiving_tds || 0;
  }

  const TEAM_GAMES = 17;
  const scale = isPg ? 1 / TEAM_GAMES : 1;
  const tdDigits = isPg ? 2 : 1;
  const sentiment = players
    .map((p) => ({
      score: p.drivers?.sentiment_score,
      confidence: Number(p.drivers?.sentiment_confidence || 0),
    }))
    .filter((x) => x.score != null && x.confidence > 0);
  const sentimentWeight = sentiment.reduce((sum, x) => sum + x.confidence, 0);
  const sentimentScore = sentimentWeight
    ? sentiment.reduce((sum, x) => sum + Number(x.score) * x.confidence, 0) / sentimentWeight
    : null;
  const items = [
    { label: isPg ? "Pass Yds/G" : "Pass Yds", value: fmt(passYds * scale, isPg ? 1 : 0) },
    { label: isPg ? "Rush Yds/G" : "Rush Yds", value: fmt(rushYds * scale, isPg ? 1 : 0) },
    { label: isPg ? "Rec Yds/G" : "Rec Yds", value: fmt(recYds * scale, isPg ? 1 : 0) },
    { label: isPg ? "Pass TD/G" : "Pass TD", value: fmt(passTd * scale, tdDigits) },
    { label: isPg ? "Rush TD/G" : "Rush TD", value: fmt(rushTd * scale, tdDigits) },
    { label: isPg ? "Rec TD/G" : "Rec TD", value: fmt(recTd * scale, tdDigits) },
    {
      label: "Sentiment",
      value: sentimentScore == null ? "—" : `${sentimentScore > 0 ? "+" : ""}${Math.round(sentimentScore)}`,
    },
  ];

  els.teamSummary.innerHTML = items
    .map(
      (i) =>
        `<div class="summary-stat"><span class="label">${i.label}</span><span class="value">${i.value}</span></div>`
    )
    .join("");
}

function depthSortKey(p) {
  const chartRank = p.depth_rank != null ? p.depth_rank : 50;
  const nflRank =
    p.drivers && p.drivers.nfl_depth_rank != null
      ? p.drivers.nfl_depth_rank
      : 50;
  const pts = -(p.fantasy_pts || 0);
  return [chartRank, nflRank, pts, p.display_name || ""];
}

function compareDepth(a, b) {
  const ka = depthSortKey(a);
  const kb = depthSortKey(b);
  for (let i = 0; i < ka.length; i++) {
    if (ka[i] < kb[i]) return -1;
    if (ka[i] > kb[i]) return 1;
  }
  return 0;
}

function depthPlayerCell(p, { starter = false } = {}) {
  if (!p) return `<td class="depth-cell"><span class="empty">-</span></td>`;
  return `<td class="depth-cell${starter ? " is-starter" : ""}">
    <button type="button" class="player-link" data-player-id="${escapeHtml(
      p.player_id
    )}" aria-haspopup="dialog">${escapeHtml(p.display_name)}</button>
  </td>`;
}

/** Build ESPN-style rows: Pos | Starter | 2nd | 3rd | 4th */
function buildDepthRows(players) {
  const DEPTH_COLS = 4;
  const groups = [
    { pos: "QB", slots: 1 },
    { pos: "RB", slots: 1 },
    { pos: "WR", slots: 3 },
    { pos: "TE", slots: 1 },
  ];
  const rows = [];

  for (const group of groups) {
    const sorted = players.filter((p) => p.position === group.pos).sort(compareDepth);
    const slotCount = Math.min(group.slots, Math.max(1, sorted.length));
    for (let slot = 0; slot < slotCount; slot++) {
      const cells = [];
      for (let depth = 0; depth < DEPTH_COLS; depth++) {
        // Column-major fill across slots (ESPN WR lanes)
        cells.push(sorted[slot + depth * slotCount] || null);
      }
      // Skip empty trailing WR lanes if no players at that slot
      if (group.slots > 1 && !cells.some(Boolean)) continue;
      const label = group.slots > 1 ? `${group.pos}${slot + 1}` : group.pos;
      rows.push({ label, players: cells });
    }
  }
  return rows;
}

function renderDepthChart(players) {
  const rows = buildDepthRows(players);
  if (!rows.length) {
    els.depthBody.innerHTML = `<tr><td colspan="5" class="empty-row">No projected players</td></tr>`;
    return;
  }
  els.depthBody.innerHTML = rows
    .map((row) => {
      const cells = row.players
        .map((p, i) => depthPlayerCell(p, { starter: i === 0 }))
        .join("");
      return `<tr>
        <td class="col-pos">${row.label}</td>
        ${cells}
      </tr>`;
    })
    .join("");
}

function renderPassing(players) {
  const isPg = state.view === "pg";
  const sort = state.sort.passing;
  const cols = [
    { label: "Player", key: "name", className: "col-player", defaultDir: "asc" },
    { label: "GP", key: "gp", defaultDir: "desc" },
    { label: "CMP", key: "cmp", defaultDir: "desc" },
    { label: "ATT", key: "att", defaultDir: "desc" },
    { label: "CMP%", key: "cmp_pct", defaultDir: "desc" },
    { label: "YDS", key: "yds", defaultDir: "desc" },
  ];
  if (!isPg) cols.push({ label: "YDS/G", key: "yds_g", defaultDir: "desc" });
  cols.push(
    { label: "AVG", key: "avg", defaultDir: "desc" },
    { label: "TD", key: "td", defaultDir: "desc" },
    { label: "INT", key: "int", defaultDir: "desc" },
    { label: "FPTS", key: "fpts", defaultDir: "desc" }
  );

  els.passingHead.innerHTML = `<tr>${cols
    .map((c) =>
      FantasySort.thAttrs({
        ...c,
        sortKey: sort.key,
        sortDir: sort.dir,
      })
    )
    .join("")}</tr>`;

  let rows = players
    .filter((p) => (statsOf(p).attempts || 0) >= (isPg ? 1 : 5))
    .map((p) => {
      const s = statsOf(p);
      const cmp = s.completions || 0;
      const att = s.attempts || 0;
      const yds = s.passing_yards || 0;
      const gp = p.projected_games || 0;
      return {
        p,
        name: p.display_name || "",
        gp,
        cmp,
        att,
        cmp_pct: att ? (100 * cmp) / att : null,
        yds,
        yds_g: isPg ? null : avg(yds, gp),
        avg: avg(yds, att),
        td: s.passing_tds || 0,
        int: s.interceptions || 0,
        fpts: isPg ? p.fantasy_pts : p.fantasy_pts_season,
      };
    });

  rows = FantasySort.sortRows(rows, { key: sort.key, dir: sort.dir });

  if (!rows.length) {
    els.passingBody.innerHTML = emptyRow(cols.length, "No projected passers");
    return;
  }

  els.passingBody.innerHTML = rows
    .map((r) => {
      const cells = [
        playerCell(r.p),
        `<td>${fmt(r.gp, 1)}</td>`,
        `<td>${fmt(r.cmp, isPg ? 1 : 0)}</td>`,
        `<td>${fmt(r.att, isPg ? 1 : 0)}</td>`,
        `<td>${fmtPct(r.cmp_pct)}</td>`,
        `<td class="num-strong">${fmt(r.yds, isPg ? 1 : 0)}</td>`,
      ];
      if (!isPg) cells.push(`<td>${fmt(r.yds_g, 1)}</td>`);
      cells.push(
        `<td>${fmt(r.avg, 1)}</td>`,
        `<td>${fmt(r.td, isPg ? 2 : 1)}</td>`,
        `<td>${fmt(r.int, isPg ? 2 : 1)}</td>`,
        `<td>${fmt(r.fpts, 1)}</td>`
      );
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
}

function renderRushing(players) {
  const isPg = state.view === "pg";
  const sort = state.sort.rushing;
  const cols = [
    { label: "Player", key: "name", className: "col-player", defaultDir: "asc" },
    { label: "GP", key: "gp", defaultDir: "desc" },
    { label: "CAR", key: "car", defaultDir: "desc" },
    { label: "YDS", key: "yds", defaultDir: "desc" },
  ];
  if (!isPg) cols.push({ label: "YDS/G", key: "yds_g", defaultDir: "desc" });
  cols.push(
    { label: "AVG", key: "avg", defaultDir: "desc" },
    { label: "TD", key: "td", defaultDir: "desc" },
    { label: "FPTS", key: "fpts", defaultDir: "desc" }
  );

  els.rushingHead.innerHTML = `<tr>${cols
    .map((c) =>
      FantasySort.thAttrs({
        ...c,
        sortKey: sort.key,
        sortDir: sort.dir,
      })
    )
    .join("")}</tr>`;

  let rows = players
    .filter((p) => (statsOf(p).carries || 0) >= (isPg ? 0.5 : 5))
    .map((p) => {
      const s = statsOf(p);
      const car = s.carries || 0;
      const yds = s.rushing_yards || 0;
      const gp = p.projected_games || 0;
      return {
        p,
        name: p.display_name || "",
        gp,
        car,
        yds,
        yds_g: isPg ? null : avg(yds, gp),
        avg: avg(yds, car),
        td: s.rushing_tds || 0,
        fpts: isPg ? p.fantasy_pts : p.fantasy_pts_season,
      };
    });

  rows = FantasySort.sortRows(rows, { key: sort.key, dir: sort.dir });

  if (!rows.length) {
    els.rushingBody.innerHTML = emptyRow(cols.length, "No projected rushers");
    return;
  }

  els.rushingBody.innerHTML = rows
    .map((r) => {
      const cells = [
        playerCell(r.p),
        `<td>${fmt(r.gp, 1)}</td>`,
        `<td>${fmt(r.car, isPg ? 1 : 0)}</td>`,
        `<td class="num-strong">${fmt(r.yds, isPg ? 1 : 0)}</td>`,
      ];
      if (!isPg) cells.push(`<td>${fmt(r.yds_g, 1)}</td>`);
      cells.push(
        `<td>${fmt(r.avg, 1)}</td>`,
        `<td>${fmt(r.td, isPg ? 2 : 1)}</td>`,
        `<td>${fmt(r.fpts, 1)}</td>`
      );
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
}

function renderReceiving(players) {
  const isPg = state.view === "pg";
  const sort = state.sort.receiving;
  const cols = [
    { label: "Player", key: "name", className: "col-player", defaultDir: "asc" },
    { label: "GP", key: "gp", defaultDir: "desc" },
    { label: "REC", key: "rec", defaultDir: "desc" },
    { label: "TGTS", key: "tgt", defaultDir: "desc" },
    { label: "YDS", key: "yds", defaultDir: "desc" },
  ];
  if (!isPg) cols.push({ label: "YDS/G", key: "yds_g", defaultDir: "desc" });
  cols.push(
    { label: "AVG", key: "avg", defaultDir: "desc" },
    { label: "TD", key: "td", defaultDir: "desc" },
    { label: "FPTS", key: "fpts", defaultDir: "desc" }
  );

  els.receivingHead.innerHTML = `<tr>${cols
    .map((c) =>
      FantasySort.thAttrs({
        ...c,
        sortKey: sort.key,
        sortDir: sort.dir,
      })
    )
    .join("")}</tr>`;

  let rows = players
    .filter((p) => {
      const s = statsOf(p);
      return (s.targets || 0) >= (isPg ? 0.5 : 5) || (s.receptions || 0) >= (isPg ? 0.3 : 3);
    })
    .map((p) => {
      const s = statsOf(p);
      const rec = s.receptions || 0;
      const tgt = s.targets || 0;
      const yds = s.receiving_yards || 0;
      const gp = p.projected_games || 0;
      return {
        p,
        name: p.display_name || "",
        gp,
        rec,
        tgt,
        yds,
        yds_g: isPg ? null : avg(yds, gp),
        avg: avg(yds, rec),
        td: s.receiving_tds || 0,
        fpts: isPg ? p.fantasy_pts : p.fantasy_pts_season,
      };
    });

  rows = FantasySort.sortRows(rows, { key: sort.key, dir: sort.dir });

  if (!rows.length) {
    els.receivingBody.innerHTML = emptyRow(cols.length, "No projected receivers");
    return;
  }

  els.receivingBody.innerHTML = rows
    .map((r) => {
      const cells = [
        playerCell(r.p),
        `<td>${fmt(r.gp, 1)}</td>`,
        `<td>${fmt(r.rec, isPg ? 1 : 0)}</td>`,
        `<td>${fmt(r.tgt, isPg ? 1 : 0)}</td>`,
        `<td class="num-strong">${fmt(r.yds, isPg ? 1 : 0)}</td>`,
      ];
      if (!isPg) cells.push(`<td>${fmt(r.yds_g, 1)}</td>`);
      cells.push(
        `<td>${fmt(r.avg, 1)}</td>`,
        `<td>${fmt(r.td, isPg ? 2 : 1)}</td>`,
        `<td>${fmt(r.fpts, 1)}</td>`
      );
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
}
