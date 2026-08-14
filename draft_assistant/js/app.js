const FLEX_POSITIONS = new Set(["RB", "WR", "TE"]);

const ROSTER_TEMPLATE = [
  { slot: "QB", count: 1 },
  { slot: "RB", count: 2 },
  { slot: "WR", count: 2 },
  { slot: "TE", count: 1 },
  { slot: "FLEX", count: 1 },
  { slot: "BN", count: 6 },
];

const STORAGE_KEY = "draft_assistant_state_v1";

const state = {
  data: null,
  drafted: new Map(), // player_id -> { pick, teamSlot, mine }
  draftHistory: [],
  currentPick: 1,
  teamCount: 12,
  draftSlot: 1,
  positionFilter: "ALL",
  search: "",
  hideDrafted: true,
  usePosTiers: true,
};

const els = {};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheElements();
  bindEvents();
  populateDraftSlotOptions();

  try {
    const res = await fetch("data/players_2026.json");
    if (!res.ok) throw new Error(`Failed to load projections (${res.status})`);
    state.data = await res.json();
    document.getElementById("seasonBadge").textContent = state.data.meta.season;
  } catch (err) {
    document.getElementById("rankingsBody").innerHTML =
      `<tr><td colspan="8" class="empty-state">${err.message}. Run: python -m src.draft_assistant.prepare --season 2026</td></tr>`;
    return;
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
  if (counts.WR < 2) needs.push("WR");
  if (counts.TE < 1) needs.push("TE");
  if (counts.RB + counts.WR + counts.TE < 5) needs.push("FLEX");
  return needs;
}

function rankingView(player) {
  const { positionFilter, usePosTiers } = state;

  if (positionFilter === "FLEX") {
    return { rank: player.flex_rank, tier: player.flex_tier };
  }
  if (usePosTiers && positionFilter !== "ALL") {
    return { rank: player.pos_rank, tier: player.pos_tier };
  }
  return { rank: player.overall_rank, tier: player.overall_tier };
}

function scoreSuggestion(player, needs) {
  const { tier, rank } = rankingView(player);
  let score = player.fantasy_pts * 10 - (rank ?? 999);

  if (needs.includes(player.position)) score += 8;
  if (needs.includes("FLEX") && FLEX_POSITIONS.has(player.position)) score += 3;
  if (tier === 1) score += 5;
  if (player.low_confidence) score -= 2;

  return { score, tier, rank };
}

function buildSuggestions() {
  const avail = availablePlayers();
  const needs = rosterNeeds();
  const onClock = isMyPick();

  const scored = avail.map((p) => {
    const { score, tier, rank } = scoreSuggestion(p, needs);
    let reason = `Tier ${tier}, #${rank}`;
    if (needs.includes(p.position)) reason = `Need ${p.position} · ${reason}`;
    else if (onClock && tier === 1) reason = `Top tier value · ${reason}`;
    return { player: p, score, reason };
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
      const { rank, tier } = rankingView(p);

      const drafted = state.drafted.get(p.player_id);
      const classes = [];
      if (drafted) classes.push("drafted");
      if (drafted?.mine) classes.push("mine");

      let tierHeader = "";
      if (tier !== lastTier) {
        lastTier = tier;
        classes.push("tier-break");
        tierHeader = `<tr class="tier-header"><td colspan="8">Tier ${tier}</td></tr>`;
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
          <td class="col-player"><span class="player-name">${p.display_name}</span>${conf}</td>
          <td class="col-pos"><span class="pos-badge pos-${p.position}">${p.position}</span></td>
          <td class="col-team">${p.team}</td>
          <td class="col-pts">${p.fantasy_pts.toFixed(1)}</td>
          <td class="col-season">${Math.round(p.fantasy_pts_season)}</td>
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
      ({ player, reason }) => `
      <li class="suggestion-item" data-id="${player.player_id}">
        <div>
          <div>${player.display_name} <span class="pos-badge pos-${player.position}">${player.position}</span></div>
          <div class="reason">${reason}</div>
        </div>
        <span class="pts">${player.fantasy_pts.toFixed(1)}</span>
      </li>`
    )
    .join("");

  els.suggestionsList.querySelectorAll(".suggestion-item").forEach((el) => {
    el.addEventListener("click", () => {
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
        <span>${player.display_name} <span class="pos-badge pos-${player.position}">${player.position}</span></span>
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

  els.draftStatus.innerHTML = `
    <span class="pick-info">Round ${round}, Pick ${pickInRound} · Overall #${state.currentPick}</span>
    <span class="${mine ? "on-clock" : ""}">${mine ? "You're on the clock" : `Team ${onClock} on the clock`}</span>
    <span class="pick-info">${avail} players available</span>
    <span class="pick-info">${state.data.meta.scoring}</span>
  `;
}

function renderAll() {
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
