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

const POS_DEFAULT_SECTION = {
  ALL: "passing",
  QB: "passing",
  RB: "rushing",
  WR: "receiving",
  TE: "receiving",
};

const state = {
  data: null,
  byId: new Map(),
  position: "ALL",
  search: "",
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
    const res = await fetch(`../data/team_stats_${SEASON}.json?v=${Date.now()}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`Failed to load projections (${res.status})`);
    state.data = await res.json();
    state.byId = new Map(state.data.players.map((p) => [p.player_id, p]));
    els.seasonBadge.textContent = state.data.meta.season;

    const params = new URLSearchParams(window.location.search);
    const posParam = (params.get("pos") || "").toUpperCase();
    if (["ALL", "QB", "RB", "WR", "TE"].includes(posParam)) {
      state.position = posParam;
      els.posSelect.value = posParam;
    }
    const sectionParam = (params.get("section") || "").toLowerCase();
    if (["passing", "rushing", "receiving"].includes(sectionParam)) {
      state.section = sectionParam;
    } else {
      state.section = POS_DEFAULT_SECTION[state.position] || "passing";
    }
    setSection(state.section, { skipUrl: true });
    renderAll();
  } catch (err) {
    const msg = `${err.message}. Run: python -m src.team_stats.prepare --season ${SEASON}`;
    els.passingBody.innerHTML = emptyRow(12, msg);
  }
}

function cacheElements() {
  els.seasonBadge = document.getElementById("seasonBadge");
  els.posSelect = document.getElementById("posSelect");
  els.searchInput = document.getElementById("searchInput");
  els.heroKicker = document.getElementById("heroKicker");
  els.heroTitle = document.getElementById("heroTitle");
  els.heroSub = document.getElementById("heroSub");
  els.totalsSummary = document.getElementById("totalsSummary");
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
  els.footerNote = document.getElementById("footerNote");
  els.viewSeason = document.getElementById("viewSeason");
  els.viewPg = document.getElementById("viewPg");
  els.hoverCard = document.getElementById("playerHoverCard");
  els.modal = document.getElementById("playerModal");
  els.modalBody = document.getElementById("playerModalBody");
  els.mainContent = document.getElementById("mainContent");
}

function bindEvents() {
  els.posSelect.addEventListener("change", () => {
    state.position = els.posSelect.value;
    state.section = POS_DEFAULT_SECTION[state.position] || "passing";
    setSection(state.section);
    hideHoverCard();
    renderAll();
  });

  els.searchInput.addEventListener("input", () => {
    state.search = els.searchInput.value.trim().toLowerCase();
    hideHoverCard();
    renderAll();
  });

  els.viewSeason.addEventListener("click", () => setView("season"));
  els.viewPg.addEventListener("click", () => setView("pg"));

  els.statTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".stat-tab");
    if (!btn) return;
    setSection(btn.dataset.section);
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
    renderPassing(filteredPlayers());
  });
  FantasySort.bindHeader(els.rushingHead, (key, defaultDir) => {
    FantasySort.toggleSort(state.sort.rushing, key, { defaultDir });
    renderRushing(filteredPlayers());
  });
  FantasySort.bindHeader(els.receivingHead, (key, defaultDir) => {
    FantasySort.toggleSort(state.sort.receiving, key, { defaultDir });
    renderReceiving(filteredPlayers());
  });
}

function setView(view) {
  state.view = view;
  els.viewSeason.classList.toggle("active", view === "season");
  els.viewPg.classList.toggle("active", view === "pg");
  hideHoverCard();
  renderAll();
}

function setSection(section, { skipUrl = false } = {}) {
  state.section = section;
  document.querySelectorAll(".stat-tab").forEach((el) => {
    el.classList.toggle("active", el.dataset.section === state.section);
  });
  document.querySelectorAll(".stat-section").forEach((el) => {
    el.hidden = el.dataset.section !== state.section;
  });
  if (!skipUrl) syncUrl();
}

function syncUrl() {
  const url = new URL(window.location.href);
  if (state.position && state.position !== "ALL") url.searchParams.set("pos", state.position);
  else url.searchParams.delete("pos");
  url.searchParams.set("section", state.section);
  window.history.replaceState({}, "", url);
}

function filteredPlayers() {
  let list = state.data.players;
  if (state.position !== "ALL") {
    list = list.filter((p) => p.position === state.position);
  }
  if (state.search) {
    list = list.filter((p) => {
      const name = (p.display_name || "").toLowerCase();
      const team = (p.team || "").toLowerCase();
      return name.includes(state.search) || team.includes(state.search);
    });
  }
  return list;
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

function teamCell(p) {
  const team = p.team || "FA";
  if (!p.team) return `<td class="col-team">${escapeHtml(team)}</td>`;
  return `<td class="col-team"><a href="/teams/?team=${encodeURIComponent(team)}">${escapeHtml(
    team
  )}</a></td>`;
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
  )} · what drives this projection</p>
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
    </div>

    <div class="driver-section">
      <h4>Fantasy points drivers (per game)</h4>
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
      <p class="driver-note">Half-PPR · 4-pt pass TD. Bars show share of this player's projected fantasy points.</p>
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
  const next =
    e.relatedTarget && e.relatedTarget.closest
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
  const players = filteredPlayers();
  const isPg = state.view === "pg";
  const caption = isPg ? "Projected per game" : "Projected season totals";
  const posLabel = state.position === "ALL" ? "All positions" : state.position;

  els.heroKicker.textContent = "League leaders";
  els.heroTitle.textContent =
    state.position === "ALL" ? "Total Projections" : `${state.position} Projections`;
  els.heroSub.textContent = `${state.data.meta.season} projected offensive stats · ${players.length} players · ${posLabel}`;
  els.passingCaption.textContent = caption;
  els.rushingCaption.textContent = caption;
  els.receivingCaption.textContent = caption;

  renderSummary(players);
  renderPassing(players);
  renderRushing(players);
  renderReceiving(players);
  syncUrl();

  els.footerNote.textContent = `Source: ${state.data.meta.source_file} · generated ${new Date(
    state.data.meta.generated_at
  ).toLocaleString()}`;
}

function renderSummary(players) {
  const isPg = state.view === "pg";
  const byFpts = [...players].sort(
    (a, b) =>
      (isPg ? b.fantasy_pts || 0 : b.fantasy_pts_season || 0) -
      (isPg ? a.fantasy_pts || 0 : a.fantasy_pts_season || 0)
  );
  const top = byFpts[0];
  const topPass = [...players]
    .filter((p) => (statsOf(p).passing_yards || 0) > 0)
    .sort((a, b) => (statsOf(b).passing_yards || 0) - (statsOf(a).passing_yards || 0))[0];
  const topRush = [...players]
    .filter((p) => (statsOf(p).rushing_yards || 0) > 0)
    .sort((a, b) => (statsOf(b).rushing_yards || 0) - (statsOf(a).rushing_yards || 0))[0];
  const topRec = [...players]
    .filter((p) => (statsOf(p).receiving_yards || 0) > 0)
    .sort((a, b) => (statsOf(b).receiving_yards || 0) - (statsOf(a).receiving_yards || 0))[0];

  const items = [
    {
      label: isPg ? "Top PPG" : "Top FPTS",
      value: top ? `${top.display_name.split(" ").slice(-1)[0]} ${fmt(isPg ? top.fantasy_pts : top.fantasy_pts_season, 1)}` : "—",
    },
    {
      label: isPg ? "Pass Yds/G" : "Pass Yds",
      value: topPass
        ? `${fmt(statsOf(topPass).passing_yards, isPg ? 1 : 0)}`
        : "—",
    },
    {
      label: isPg ? "Rush Yds/G" : "Rush Yds",
      value: topRush
        ? `${fmt(statsOf(topRush).rushing_yards, isPg ? 1 : 0)}`
        : "—",
    },
    {
      label: isPg ? "Rec Yds/G" : "Rec Yds",
      value: topRec
        ? `${fmt(statsOf(topRec).receiving_yards, isPg ? 1 : 0)}`
        : "—",
    },
  ];

  els.totalsSummary.innerHTML = items
    .map(
      (i) =>
        `<div class="summary-stat"><span class="label">${i.label}</span><span class="value">${escapeHtml(
          String(i.value)
        )}</span></div>`
    )
    .join("");
}

function renderPassing(players) {
  const isPg = state.view === "pg";
  const sort = state.sort.passing;
  const cols = [
    { label: "Player", key: "name", className: "col-player", defaultDir: "asc" },
    { label: "Team", key: "team", className: "col-team", defaultDir: "asc" },
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

  els.passingHead.innerHTML = `<tr><th class="col-rank">#</th>${cols
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
        team: p.team || "",
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
    els.passingBody.innerHTML = emptyRow(cols.length + 1, "No projected passers");
    return;
  }

  els.passingBody.innerHTML = rows
    .map((r, i) => {
      const cells = [
        `<td class="col-rank">${i + 1}</td>`,
        playerCell(r.p),
        teamCell(r.p),
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
    { label: "Team", key: "team", className: "col-team", defaultDir: "asc" },
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

  els.rushingHead.innerHTML = `<tr><th class="col-rank">#</th>${cols
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
        team: p.team || "",
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
    els.rushingBody.innerHTML = emptyRow(cols.length + 1, "No projected rushers");
    return;
  }

  els.rushingBody.innerHTML = rows
    .map((r, i) => {
      const cells = [
        `<td class="col-rank">${i + 1}</td>`,
        playerCell(r.p),
        teamCell(r.p),
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
    { label: "Team", key: "team", className: "col-team", defaultDir: "asc" },
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

  els.receivingHead.innerHTML = `<tr><th class="col-rank">#</th>${cols
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
        team: p.team || "",
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
    els.receivingBody.innerHTML = emptyRow(cols.length + 1, "No projected receivers");
    return;
  }

  els.receivingBody.innerHTML = rows
    .map((r, i) => {
      const cells = [
        `<td class="col-rank">${i + 1}</td>`,
        playerCell(r.p),
        teamCell(r.p),
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
