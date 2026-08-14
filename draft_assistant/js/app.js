const FLEX_POSITIONS = new Set(["RB", "WR", "TE"]);

const ROSTER_TEMPLATE = [
  { slot: "QB", count: 1 },
  { slot: "RB", count: 2 },
  { slot: "WR", count: 3 },
  { slot: "TE", count: 1 },
  { slot: "FLEX", count: 1 },
  { slot: "BN", count: 6 },
];

const STARTERS = { QB: 1, RB: 2, WR: 3, TE: 1 };
const FLEX_SHARE = { QB: 0, RB: 0.4, WR: 0.5, TE: 0.1 };
const OVERALL_VORP_TIER_GAP = 12;
const SKILL_STARTER_SLOTS = 7; // 2 RB + 3 WR + 1 TE + 1 FLEX
const SEASON = 2026;

// Half-PPR, 4pt passing TD — matches fantasy_points.py / team cards
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

const STORAGE_KEY = "draft_assistant_state_v1";

const state = {
  data: null,
  cardById: new Map(),
  drafted: new Map(), // player_id -> { pick, teamSlot, mine }
  draftHistory: [],
  currentPick: 1,
  teamCount: 12,
  draftSlot: 1,
  positionFilter: "ALL",
  search: "",
  hideDrafted: true,
  usePosTiers: true,
  vorpTeamCount: null,
  hoverId: null,
  hoverTimer: null,
};

const els = {};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheElements();
  bindEvents();
  populateDraftSlotOptions();

  try {
    const res = await fetch(`data/players_${SEASON}.json`);
    if (!res.ok) throw new Error(`Failed to load projections (${res.status})`);
    state.data = await res.json();
    document.getElementById("seasonBadge").textContent = state.data.meta.season;
  } catch (err) {
    document.getElementById("rankingsBody").innerHTML =
      `<tr><td colspan="9" class="empty-state">${err.message}. Run: python -m src.draft_assistant.prepare --season ${SEASON}</td></tr>`;
    return;
  }

  try {
    const cardRes = await fetch(`data/team_stats_${SEASON}.json`);
    if (cardRes.ok) {
      const cardData = await cardRes.json();
      state.cardById = new Map((cardData.players || []).map((p) => [p.player_id, p]));
    }
  } catch {
    /* cards degrade gracefully without team-stats detail */
  }

  loadPersistedState();
  renderAll();
}

function cacheElements() {
  els.rankingsBody = document.getElementById("rankingsBody");
  els.suggestionsList = document.getElementById("suggestionsList");
  els.rosterSlots = document.getElementById("rosterSlots");
  els.rosterCount = document.getElementById("rosterCount");
  els.draftLog = document.getElementById("draftLog");
  els.draftStatus = document.getElementById("draftStatus");
  els.teamCount = document.getElementById("teamCount");
  els.draftSlot = document.getElementById("draftSlot");
  els.currentPick = document.getElementById("currentPick");
  els.searchInput = document.getElementById("searchInput");
  els.hideDrafted = document.getElementById("hideDrafted");
  els.posTiers = document.getElementById("posTiers");
  els.positionTabs = document.getElementById("positionTabs");
  els.hoverCard = document.getElementById("playerHoverCard");
  els.modal = document.getElementById("playerModal");
  els.modalBody = document.getElementById("playerModalBody");
  els.appRoot = document.querySelector(".app");
}

function bindEvents() {
  els.teamCount.addEventListener("change", () => {
    state.teamCount = Number(els.teamCount.value);
    populateDraftSlotOptions();
    persistState();
    renderAll();
  });

  els.draftSlot.addEventListener("change", () => {
    state.draftSlot = Number(els.draftSlot.value);
    persistState();
    renderAll();
  });

  els.currentPick.addEventListener("change", () => {
    state.currentPick = Math.max(1, Number(els.currentPick.value) || 1);
    els.currentPick.value = state.currentPick;
    persistState();
    renderAll();
  });

  els.searchInput.addEventListener("input", () => {
    state.search = els.searchInput.value.trim().toLowerCase();
    renderRankings();
  });

  els.hideDrafted.addEventListener("change", () => {
    state.hideDrafted = els.hideDrafted.checked;
    renderRankings();
  });

  els.posTiers.addEventListener("change", () => {
    state.usePosTiers = els.posTiers.checked;
    renderRankings();
  });

  els.positionTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    els.positionTabs.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    state.positionFilter = btn.dataset.pos;
    renderRankings();
  });

  document.getElementById("undoBtn").addEventListener("click", undoLastPick);
  document.getElementById("resetBtn").addEventListener("click", resetDraft);

  els.appRoot.addEventListener("mouseover", onPlayerMouseOver);
  els.appRoot.addEventListener("mouseout", onPlayerMouseOut);
  els.appRoot.addEventListener("focusin", onPlayerFocusIn);
  els.appRoot.addEventListener("focusout", onPlayerFocusOut);
  els.appRoot.addEventListener("click", onPlayerClick);

  els.modal.addEventListener("click", (e) => {
    if (e.target.closest("[data-close-modal]")) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.modal.hidden) closeModal();
  });
}

function populateDraftSlotOptions() {
  const prev = state.draftSlot;
  els.draftSlot.innerHTML = "";
  for (let i = 1; i <= state.teamCount; i += 1) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = i;
    els.draftSlot.appendChild(opt);
  }
  state.draftSlot = Math.min(prev, state.teamCount);
  els.draftSlot.value = state.draftSlot;
  els.teamCount.value = state.teamCount;
}

function getPlayer(id) {
  return state.data.players.find((p) => p.player_id === id);
}

function pickTeamForRound(pick) {
  const round = Math.ceil(pick / state.teamCount);
  const pickInRound = ((pick - 1) % state.teamCount) + 1;
  if (round % 2 === 1) return pickInRound;
  return state.teamCount - pickInRound + 1;
}

function isMyPick(pick = state.currentPick) {
  return pickTeamForRound(pick) === state.draftSlot;
}

function availablePlayers() {
  return state.data.players.filter((p) => !state.drafted.has(p.player_id));
}

function replacementRank(position, teamCount) {
  const starters = STARTERS[position];
  if (starters == null) return 1;
  const share = FLEX_SHARE[position] ?? 0;
  return Math.floor(teamCount * starters + teamCount * share) + 1;
}

function kthScore(values, rank) {
  const ordered = values
    .filter((v) => v != null && !Number.isNaN(v))
    .sort((a, b) => b - a);
  if (!ordered.length) return 0;
  const idx = Math.min(Math.max(rank, 1), ordered.length) - 1;
  return ordered[idx];
}

function assignTiers(sortedValues, { gap, pctGap }) {
  if (!sortedValues.length) return [];
  const tiers = [1];
  let current = 1;
  for (let i = 1; i < sortedValues.length; i += 1) {
    const prev = sortedValues[i - 1];
    const drop = prev - sortedValues[i];
    const rel = prev > 0 ? drop / prev : 0;
    if (drop > gap || (pctGap != null && rel > pctGap)) current += 1;
    tiers.push(current);
  }
  return tiers;
}

function applyLiveVorp(teamCount = state.teamCount) {
  if (!state.data?.players) return;
  if (state.vorpTeamCount === teamCount) return;

  const players = state.data.players;
  const byPos = { QB: [], RB: [], WR: [], TE: [] };
  for (const p of players) {
    if (byPos[p.position]) byPos[p.position].push(p);
  }

  const baselines = {};
  for (const pos of Object.keys(byPos)) {
    const pts = byPos[pos].map((p) => Number(p.fantasy_pts_season) || 0);
    baselines[pos] = kthScore(pts, replacementRank(pos, teamCount));
  }

  for (const p of players) {
    const baseline = baselines[p.position] ?? 0;
    const season = Number(p.fantasy_pts_season) || 0;
    p.live_replacement_pts = baseline;
    p.live_vorp = Math.max(0, season - baseline);
  }

  const ordered = players.slice().sort((a, b) => {
    if (b.live_vorp !== a.live_vorp) return b.live_vorp - a.live_vorp;
    return (b.fantasy_pts_season || 0) - (a.fantasy_pts_season || 0);
  });
  const tiers = assignTiers(
    ordered.map((p) => p.live_vorp),
    { gap: OVERALL_VORP_TIER_GAP, pctGap: 0.04 }
  );
  ordered.forEach((p, i) => {
    p.live_overall_rank = i + 1;
    p.live_overall_tier = tiers[i];
  });

  state.vorpTeamCount = teamCount;
}

function draftPlayer(playerId, { mine = false, advancePick = true } = {}) {
  if (state.drafted.has(playerId)) return;
  const player = getPlayer(playerId);
  if (!player) return;

  const pick = state.currentPick;
  const teamSlot = pickTeamForRound(pick);
  state.drafted.set(playerId, { pick, teamSlot, mine });
  state.draftHistory.push({ playerId, pick, teamSlot, mine });

  if (advancePick) {
    state.currentPick += 1;
    els.currentPick.value = state.currentPick;
  }

  persistState();
  renderAll();
}

function undoLastPick() {
  if (state.draftHistory.length === 0) return;
  const last = state.draftHistory.pop();
  state.drafted.delete(last.playerId);
  state.currentPick = last.pick;
  els.currentPick.value = state.currentPick;
  persistState();
  renderAll();
}

function resetDraft() {
  if (!confirm("Reset the entire draft board?")) return;
  state.drafted.clear();
  state.draftHistory = [];
  state.currentPick = 1;
  els.currentPick.value = 1;
  persistState();
  renderAll();
}

function rosterNeeds() {
  const mine = [...state.drafted.entries()]
    .filter(([, d]) => d.mine)
    .map(([id]) => getPlayer(id))
    .filter(Boolean);

  const counts = { QB: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, BN: 0 };
  for (const p of mine) counts[p.position] += 1;

  const needs = [];
  if (counts.QB < 1) needs.push("QB");
  if (counts.RB < 2) needs.push("RB");
  if (counts.WR < 3) needs.push("WR");
  if (counts.TE < 1) needs.push("TE");
  if (counts.RB + counts.WR + counts.TE < SKILL_STARTER_SLOTS) needs.push("FLEX");
  return needs;
}

function rankingView(player) {
  const { positionFilter, usePosTiers } = state;

  if (positionFilter === "FLEX") {
    return {
      rank: player.flex_rank,
      tier: player.flex_tier,
      vorp: player.live_vorp ?? player.vorp ?? 0,
    };
  }
  if (usePosTiers && positionFilter !== "ALL") {
    return {
      rank: player.pos_rank,
      tier: player.pos_tier,
      vorp: player.live_vorp ?? player.vorp ?? 0,
    };
  }
  return {
    rank: player.live_overall_rank ?? player.overall_rank,
    tier: player.live_overall_tier ?? player.overall_tier,
    vorp: player.live_vorp ?? player.vorp ?? 0,
  };
}

function scoreSuggestion(player, needs) {
  const { tier, rank, vorp } = rankingView(player);
  let score = (vorp ?? 0) - (rank ?? 999) * 0.01;

  if (needs.includes(player.position)) score += 8;
  if (needs.includes("FLEX") && FLEX_POSITIONS.has(player.position)) score += 3;
  if (tier === 1) score += 5;
  if (player.low_confidence) score -= 2;

  return { score, tier, rank, vorp };
}

function buildSuggestions() {
  const avail = availablePlayers();
  const needs = rosterNeeds();
  const onClock = isMyPick();

  const scored = avail.map((p) => {
    const { score, tier, rank, vorp } = scoreSuggestion(p, needs);
    let reason = `VORP ${Math.round(vorp ?? 0)} · Tier ${tier}, #${rank}`;
    if (needs.includes(p.position)) reason = `Need ${p.position} · ${reason}`;
    else if (onClock && tier === 1) reason = `Top tier value · ${reason}`;
    return { player: p, score, reason, vorp };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 8);
}

function filteredRankings() {
  let rows = state.data.players.slice();

  if (state.positionFilter === "FLEX") {
    rows = rows.filter((p) => FLEX_POSITIONS.has(p.position));
  } else if (state.positionFilter !== "ALL") {
    rows = rows.filter((p) => p.position === state.positionFilter);
  }

  if (state.search) {
    rows = rows.filter(
      (p) =>
        p.display_name.toLowerCase().includes(state.search) ||
        p.team.toLowerCase().includes(state.search)
    );
  }

  if (state.hideDrafted) {
    rows = rows.filter((p) => !state.drafted.has(p.player_id));
  }

  rows.sort((a, b) => {
    const rankA = rankingView(a).rank ?? 9999;
    const rankB = rankingView(b).rank ?? 9999;
    return rankA - rankB;
  });

  return rows;
}

function renderRankings() {
  const rows = filteredRankings();
  let lastTier = null;

  els.rankingsBody.innerHTML = rows
    .map((p) => {
      const { rank, tier, vorp } = rankingView(p);

      const drafted = state.drafted.get(p.player_id);
      const classes = [];
      if (drafted) classes.push("drafted");
      if (drafted?.mine) classes.push("mine");

      let tierHeader = "";
      if (tier !== lastTier) {
        lastTier = tier;
        classes.push("tier-break");
        tierHeader = `<tr class="tier-header"><td colspan="9">Tier ${tier}</td></tr>`;
      }

      const conf = p.low_confidence
        ? '<span class="low-confidence" title="Low confidence projection">⚠</span>'
        : "";

      return `${tierHeader}
        <tr class="${classes.join(" ")}" data-id="${p.player_id}">
          <td class="col-check">
            <input type="checkbox" ${drafted ? "checked" : ""} aria-label="Draft ${p.display_name}" />
          </td>
          <td class="col-rank">${rank}</td>
          <td class="col-tier"><span class="tier-pill">${tier}</span></td>
          <td class="col-player">
            <span class="player-name">
              <button type="button" class="player-link" data-player-id="${escapeHtml(p.player_id)}" aria-haspopup="dialog">${escapeHtml(p.display_name)}</button>
            </span>${conf}
          </td>
          <td class="col-pos"><span class="pos-badge pos-${p.position}">${p.position}</span></td>
          <td class="col-team">${p.team}</td>
          <td class="col-pts">${p.fantasy_pts.toFixed(1)}</td>
          <td class="col-season">${Math.round(p.fantasy_pts_season)}</td>
          <td class="col-vorp">${Math.round(vorp ?? 0)}</td>
        </tr>`;
    })
    .join("");

  els.rankingsBody.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener("change", (e) => {
      const row = e.target.closest("tr[data-id]");
      const id = row.dataset.id;
      if (e.target.checked) {
        const mine = isMyPick();
        draftPlayer(id, { mine });
      } else {
        const entry = state.draftHistory.find((h) => h.playerId === id);
        if (entry) {
          state.draftHistory = state.draftHistory.filter((h) => h.playerId !== id);
          state.drafted.delete(id);
          persistState();
          renderAll();
        }
      }
    });
  });
}

function renderSuggestions() {
  const suggestions = buildSuggestions();
  if (suggestions.length === 0) {
    els.suggestionsList.innerHTML = '<li class="empty-state">No players left</li>';
    return;
  }

  els.suggestionsList.innerHTML = suggestions
    .map(
      ({ player, reason, vorp }) => `
      <li class="suggestion-item" data-id="${player.player_id}">
        <div>
          <div>
            <button type="button" class="player-link" data-player-id="${escapeHtml(player.player_id)}" aria-haspopup="dialog">${escapeHtml(player.display_name)}</button>
            <span class="pos-badge pos-${player.position}">${player.position}</span>
          </div>
          <div class="reason">${reason}</div>
        </div>
        <span class="pts">${Math.round(vorp ?? 0)}</span>
      </li>`
    )
    .join("");

  els.suggestionsList.querySelectorAll(".suggestion-item").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest(".player-link")) return;
      draftPlayer(el.dataset.id, { mine: isMyPick() });
    });
  });
}

function buildRosterDisplay() {
  const mine = [...state.drafted.entries()]
    .filter(([, d]) => d.mine)
    .map(([id, d]) => ({ player: getPlayer(id), pick: d.pick }))
    .filter((x) => x.player)
    .sort((a, b) => a.pick - b.pick);

  const filled = { QB: [], RB: [], WR: [], TE: [], FLEX: [], BN: [] };
  for (const { player } of mine) {
    if (filled[player.position].length < ROSTER_TEMPLATE.find((r) => r.slot === player.position)?.count) {
      filled[player.position].push(player);
    } else if (filled.FLEX.length < 1 && FLEX_POSITIONS.has(player.position)) {
      filled.FLEX.push(player);
    } else {
      filled.BN.push(player);
    }
  }

  const slots = [];
  for (const { slot, count } of ROSTER_TEMPLATE) {
    for (let i = 0; i < count; i += 1) {
      const p = filled[slot][i];
      slots.push({ label: slot, player: p });
    }
  }
  return { slots, total: mine.length };
}

function renderRoster() {
  const { slots, total } = buildRosterDisplay();
  els.rosterCount.textContent = `${total} players`;

  els.rosterSlots.innerHTML = slots
    .map(({ label, player }) => {
      if (!player) {
        return `<div class="roster-slot empty"><span class="label">${label}</span><span>Empty</span></div>`;
      }
      return `<div class="roster-slot">
        <span class="label">${label}</span>
        <span>
          <button type="button" class="player-link" data-player-id="${escapeHtml(player.player_id)}" aria-haspopup="dialog">${escapeHtml(player.display_name)}</button>
          <span class="pos-badge pos-${player.position}">${player.position}</span>
        </span>
      </div>`;
    })
    .join("");
}

function renderDraftLog() {
  if (state.draftHistory.length === 0) {
    els.draftLog.innerHTML = '<li class="empty-state">No picks yet</li>';
    return;
  }

  els.draftLog.innerHTML = state.draftHistory
    .slice()
    .reverse()
    .map(({ playerId, pick, teamSlot, mine }) => {
      const p = getPlayer(playerId);
      const cls = mine ? "mine" : "";
      return `<li class="${cls}"><span class="pick-num">#${pick}</span> Team ${teamSlot}: ${p.display_name} (${p.position})</li>`;
    })
    .join("");
}

function renderDraftStatus() {
  const round = Math.ceil(state.currentPick / state.teamCount);
  const pickInRound = ((state.currentPick - 1) % state.teamCount) + 1;
  const onClock = pickTeamForRound(state.currentPick);
  const mine = isMyPick();
  const avail = availablePlayers().length;
  const ranks = Object.entries(STARTERS)
    .map(([pos]) => `${pos}${replacementRank(pos, state.teamCount)}`)
    .join(" · ");

  els.draftStatus.innerHTML = `
    <span class="pick-info">Round ${round}, Pick ${pickInRound} · Overall #${state.currentPick}</span>
    <span class="${mine ? "on-clock" : ""}">${mine ? "You're on the clock" : `Team ${onClock} on the clock`}</span>
    <span class="pick-info">${avail} players available</span>
    <span class="pick-info">${state.data.meta.scoring} · All board by VORP (${ranks})</span>
  `;
}

function renderAll() {
  applyLiveVorp(state.teamCount);
  renderDraftStatus();
  renderRankings();
  renderSuggestions();
  renderRoster();
  renderDraftLog();
}

function persistState() {
  const payload = {
    drafted: [...state.drafted.entries()],
    draftHistory: state.draftHistory,
    currentPick: state.currentPick,
    teamCount: state.teamCount,
    draftSlot: state.draftSlot,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function loadPersistedState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    state.drafted = new Map(saved.drafted || []);
    state.draftHistory = saved.draftHistory || [];
    state.currentPick = saved.currentPick || 1;
    state.teamCount = saved.teamCount || 12;
    state.draftSlot = saved.draftSlot || 1;
    els.currentPick.value = state.currentPick;
  } catch {
    /* ignore corrupt storage */
  }
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(n, digits = 1) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 100) return Math.round(n).toLocaleString();
  return Number(n).toFixed(digits);
}

function cardPlayer(playerId) {
  const draft = getPlayer(playerId);
  const detail = state.cardById.get(playerId);
  if (!draft && !detail) return null;

  const merged = {
    ...(detail || {}),
    ...(draft || {}),
    drivers: detail?.drivers || {},
    pg: detail?.pg || {},
    season: detail?.season || {},
    depth_rank: detail?.depth_rank ?? draft?.depth_rank ?? null,
    fantasy_pts: draft?.fantasy_pts ?? detail?.fantasy_pts,
    fantasy_pts_season: draft?.fantasy_pts_season ?? detail?.fantasy_pts_season,
    fantasy_pts_low: draft?.fantasy_pts_low ?? detail?.drivers?.fantasy_pts_low,
    fantasy_pts_high: draft?.fantasy_pts_high ?? detail?.drivers?.fantasy_pts_high,
  };
  return merged;
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
  return { pass, rush, rec, total: pass + rush + rec };
}

function scaleDrivers(p) {
  const d = p.drivers || {};
  const items = [];
  const map = [
    ["Pass attempts volume scale", "normalization_scale_attempts"],
    ["Pass yards volume scale", "normalization_scale_passing_yards"],
    ["Carry volume scale", "normalization_scale_carries"],
    ["Rush yards volume scale", "normalization_scale_rushing_yards"],
    ["Receptions volume scale", "normalization_scale_receptions"],
    ["Rec yards volume scale", "normalization_scale_receiving_yards"],
    ["Rec TD volume scale", "normalization_scale_receiving_tds"],
  ];
  for (const [label, key] of map) {
    const v = d[key];
    if (v == null || Math.abs(v - 1) < 0.005) continue;
    items.push({ label, value: v });
  }
  if (d.role_discount_applied && d.role_discount_factor != null && d.role_discount_factor < 0.999) {
    items.push({ label: "Role / depth discount", value: d.role_discount_factor });
  }
  if (d.qb_volume_games_scale != null && Math.abs(d.qb_volume_games_scale - 1) >= 0.005) {
    items.push({ label: "QB volume-games scale", value: d.qb_volume_games_scale });
  }
  if (d.rookie_vacancy_scale != null && Math.abs(d.rookie_vacancy_scale - 1) >= 0.005) {
    items.push({ label: "Rookie vacancy scale", value: d.rookie_vacancy_scale });
  }
  return items;
}

function contextFacts(p) {
  const d = p.drivers || {};
  const facts = [];
  const sourceLabel =
    p.source === "rookie_rule"
      ? "Rookie rule path"
      : p.source === "veteran_model"
        ? "Veteran model"
        : p.source || "Model";
  facts.push({ k: "Projection path", v: sourceLabel });

  const overall = p.live_overall_rank ?? p.overall_rank;
  if (overall != null) facts.push({ k: "Overall rank (VORP)", v: String(overall) });
  if (p.pos_rank != null) facts.push({ k: "Position rank", v: String(p.pos_rank) });
  if (p.role) facts.push({ k: "Role", v: String(p.role).replace(/_/g, " ") });
  if (p.depth_rank != null) facts.push({ k: "Depth rank", v: String(Math.round(p.depth_rank)) });
  if (d.nfl_depth_rank != null) {
    facts.push({ k: "NFL depth rank", v: String(Math.round(d.nfl_depth_rank)) });
  }
  if (p.depth_chart_status) {
    facts.push({ k: "Depth status", v: String(p.depth_chart_status).replace(/_/g, " ") });
  }
  if (p.projected_games != null) facts.push({ k: "Projected games", v: fmt(p.projected_games, 1) });
  if (d.team_changed) facts.push({ k: "Team change", v: "Yes (new team)" });
  if (d.rookie_tier) facts.push({ k: "Rookie tier", v: String(d.rookie_tier) });
  if (d.team_pass_attempts_pg_pred != null && ["QB", "WR", "TE", "RB"].includes(p.position)) {
    facts.push({ k: "Team pass att/G", v: fmt(d.team_pass_attempts_pg_pred, 1) });
  }
  if (d.team_passing_yards_pg_pred != null && ["QB", "WR", "TE"].includes(p.position)) {
    facts.push({ k: "Team pass yds/G", v: fmt(d.team_passing_yards_pg_pred, 1) });
  }
  if (d.team_carries_pg_pred != null && ["RB", "QB"].includes(p.position)) {
    facts.push({ k: "Team carries/G", v: fmt(d.team_carries_pg_pred, 1) });
  }
  return facts;
}

function buildCardHtml(p, { full = false } = {}) {
  const d = p.drivers || {};
  const br = fantasyBreakdown(p);
  const absTotal = Math.max(Math.abs(br.total), 0.01);
  const vorp = p.live_vorp ?? p.vorp ?? 0;
  const titleTag = full ? "h2" : "h3";
  const titleId = full ? ' id="playerModalTitle"' : "";

  const pills = [];
  pills.push(`<span class="pill">${escapeHtml(p.team || "FA")}</span>`);
  if (p.low_confidence) pills.push(`<span class="pill warn">Low confidence</span>`);
  else pills.push(`<span class="pill ok">Modeled</span>`);
  if (d.any_stat_low_n_flag) pills.push(`<span class="pill warn">Low-N interval</span>`);
  if (d.role_discount_applied) pills.push(`<span class="pill warn">Role discounted</span>`);

  const contrib = [
    { label: "Passing", pts: br.pass, cls: "pass" },
    { label: "Rushing", pts: br.rush, cls: "rush" },
    { label: "Receiving", pts: br.rec, cls: "rec" },
  ].filter((c) => Math.abs(c.pts) >= 0.05);

  const scales = scaleDrivers(p);
  const facts = contextFacts(p);

  return `
    <div class="player-card-header">
      <span class="pos-badge pos-${escapeHtml(p.position)}">${escapeHtml(p.position)}</span>
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
        <span class="label">Pts / G</span>
        <span class="value">${fmt(p.fantasy_pts, 1)}</span>
        <span class="hint">${
          p.fantasy_pts_low != null && p.fantasy_pts_high != null
            ? `Range ${fmt(p.fantasy_pts_low, 1)} – ${fmt(p.fantasy_pts_high, 1)}`
            : "Half-PPR"
        }</span>
      </div>
      <div class="card-stat">
        <span class="label">Season</span>
        <span class="value">${fmt(p.fantasy_pts_season, 0)}</span>
        <span class="hint">${fmt(p.projected_games, 1)} games</span>
      </div>
      <div class="card-stat">
        <span class="label">VORP</span>
        <span class="value">${Math.round(vorp)}</span>
        <span class="hint">vs replacement</span>
      </div>
      ${
        full
          ? `<div class="card-stat">
              <span class="label">Depth</span>
              <span class="value">${p.depth_rank != null ? Math.round(p.depth_rank) : "—"}</span>
              <span class="hint">${escapeHtml((p.depth_chart_status || "chart").replace(/_/g, " "))}</span>
            </div>`
          : ""
      }
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
          : `<p class="driver-note">${
              state.cardById.size
                ? "No meaningful scoring volume projected."
                : "Run team_stats.prepare for volume drivers on this card."
            }</p>`
      }
      <p class="driver-note">Half-PPR · 4-pt pass TD. Bars show share of this player's projected fantasy points.</p>
    </div>

    ${
      scales.length
        ? `<div class="driver-section">
            <h4>Volume / role adjustments</h4>
            ${scales
              .map((s) => {
                const pct = Math.min(140, Math.round(100 * s.value));
                const delta = s.value - 1;
                const deltaTxt =
                  delta >= 0 ? `+${fmt(100 * delta, 0)}%` : `${fmt(100 * delta, 0)}%`;
                return `<div class="driver-row">
                  <span class="driver-label">${escapeHtml(s.label)}</span>
                  <div class="driver-bar scale"><span style="width:${Math.min(100, pct)}%"></span></div>
                  <span class="driver-value">${fmt(s.value, 2)}× (${deltaTxt})</span>
                </div>`;
              })
              .join("")}
          </div>`
        : full
          ? `<div class="driver-section">
              <h4>Volume / role adjustments</h4>
              <p class="driver-note">No material volume or role scales applied — rates sit near the model baseline.</p>
            </div>`
          : ""
    }

    <div class="driver-section">
      <h4>Context</h4>
      <ul class="driver-list">
        ${facts
          .map(
            (f) =>
              `<li><span class="k">${escapeHtml(f.k)}</span><span class="v">${escapeHtml(
                f.v
              )}</span></li>`
          )
          .join("")}
      </ul>
    </div>
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
  e.stopPropagation();
  hideHoverCard();
  openModal(link.dataset.playerId);
}

function showHoverCard(playerId, anchor) {
  const p = cardPlayer(playerId);
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
  const p = cardPlayer(playerId);
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
