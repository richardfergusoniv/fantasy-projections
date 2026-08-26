const SEASON = 2026;

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
  cardById: new Map(),
  byId: new Map(),
  pos: "ALL",
  search: "",
  sortKey: "our_rank",
  sortDir: "asc",
  hoverId: null,
  hoverTimer: null,
};

const els = {};

function fmt(n, digits = 1) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}

function formatVorp(v) {
  return (Number(v) || 0).toFixed(1);
}

function deltaClass(d) {
  if (d == null) return "";
  if (d < -0.5) return "delta-up";
  if (d > 0.5) return "delta-down";
  return "delta-flat";
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadData() {
  const res = await fetch(`../data/comparison_${SEASON}.json`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(
      `Missing comparison_${SEASON}.json — run: python -m src.draft_assistant.compare_prepare --season ${SEASON}`
    );
  }
  state.data = await res.json();
  state.byId = new Map((state.data.players || []).map((p) => [p.player_id, p]));

  document.getElementById("seasonBadge").textContent = String(
    state.data.meta?.season || SEASON
  );
  const m = state.data.meta || {};
  document.getElementById("metaLine").textContent =
    `${m.player_count || 0} players · ECR matched ${m.matched_ecr || 0} · ADP matched ${m.matched_adp || 0}` +
    (m.delta_note ? ` · ${m.delta_note}` : "");
  const adp = m.adp || {};
  const ecr = m.ecr || {};
  document.getElementById("attribution").innerHTML =
    `Our board: ${m.scoring_our || "half-PPR"} · ` +
    `ECR: <a href="https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php" target="_blank" rel="noopener">FantasyPros</a>` +
    ` via nflverse/DynastyProcess` +
    (ecr.scrape_date ? ` (${ecr.scrape_date})` : "") +
    ` · ADP: <a href="https://fantasyfootballcalculator.com/adp" target="_blank" rel="noopener">Fantasy Football Calculator</a>` +
    (adp.scoring ? ` ${adp.scoring}` : "") +
    (adp.teams ? ` / ${adp.teams} teams` : "");

  try {
    const cardRes = await fetch(`../data/team_stats_${SEASON}.json`, { cache: "no-store" });
    if (cardRes.ok) {
      const cardData = await cardRes.json();
      state.cardById = new Map((cardData.players || []).map((p) => [p.player_id, p]));
    }
  } catch {
    /* cards degrade without team-stats detail */
  }

  render();
}

function sentimentHtml(player) {
  const score = player.sentiment_score;
  if (score == null) return '<span class="sentiment-score none" title="No reviewed sentiment signal">—</span>';
  const cls = score > 15 ? "positive" : score < -15 ? "negative" : "neutral";
  const sign = score > 0 ? "+" : "";
  const confidence = player.sentiment_confidence == null
    ? "unknown"
    : `${Math.round(100 * Number(player.sentiment_confidence))}%`;
  const mode = player.sentiment_model_active ? "active model feature" : "diagnostic only";
  const title = `Player sentiment ${sign}${Math.round(score)} · ${confidence} confidence · ${mode} · as of ${player.sentiment_as_of || "unknown"}`;
  return `<span class="sentiment-score ${cls}" title="${escapeHtml(title)}">${sign}${Math.round(score)}</span>`;
}

function filteredRows() {
  let rows = state.data?.players || [];
  if (state.pos !== "ALL") {
    rows = rows.filter((p) => p.position === state.pos);
  }
  const q = state.search.trim().toLowerCase();
  if (q) {
    rows = rows.filter(
      (p) =>
        (p.display_name || "").toLowerCase().includes(q) ||
        (p.team || "").toLowerCase().includes(q)
    );
  }
  return FantasySort.sortRows(rows, {
    key: state.sortKey,
    dir: state.sortDir,
    getValue: (p) => p[state.sortKey],
  });
}

function render() {
  const body = document.getElementById("tableBody");
  const rows = filteredRows();
  body.innerHTML = rows
    .map((p) => {
      const name = p.player_id
        ? `<button type="button" class="player-link" data-player-id="${escapeHtml(
            p.player_id
          )}" aria-haspopup="dialog">${escapeHtml(p.display_name || "")}</button>`
        : escapeHtml(p.display_name || "");
      return `<tr>
        <td class="num">${p.our_rank ?? "—"}</td>
        <td class="col-player">${name}</td>
        <td><span class="pos-pill pos-${(p.position || "").toLowerCase()}">${p.position || ""}</span></td>
        <td>${escapeHtml(p.team || "")}</td>
        <td class="num">${fmt(p.fantasy_pts, 1)}</td>
        <td class="num">${fmt(p.vorp, 2)}</td>
        <td class="col-sentiment">${sentimentHtml(p)}</td>
        <td class="num">${p.ecr != null ? fmt(p.ecr, 1) : "—"}</td>
        <td class="num">${p.adp != null ? fmt(p.adp, 1) : "—"}</td>
        <td class="num ${deltaClass(p.delta_ecr)}">${p.delta_ecr != null ? fmt(p.delta_ecr, 1) : "—"}</td>
        <td class="num ${deltaClass(p.delta_adp)}">${p.delta_adp != null ? fmt(p.delta_adp, 1) : "—"}</td>
      </tr>`;
    })
    .join("");

  FantasySort.markStaticHeaders(document.getElementById("compareTable"), state.sortKey, state.sortDir);
}

function cardPlayer(playerId) {
  const compare = state.byId.get(playerId);
  const detail = state.cardById.get(playerId);
  if (!compare && !detail) return null;

  return {
    ...(detail || {}),
    ...(compare || {}),
    drivers: detail?.drivers || {},
    pg: detail?.pg || {},
    season: detail?.season || {},
    history: detail?.history || [],
    depth_rank: detail?.depth_rank ?? null,
    fantasy_pts: compare?.fantasy_pts ?? detail?.fantasy_pts,
    fantasy_pts_season: compare?.fantasy_pts_season ?? detail?.fantasy_pts_season,
    fantasy_pts_low: detail?.drivers?.fantasy_pts_low,
    fantasy_pts_high: detail?.drivers?.fantasy_pts_high,
    vorp: compare?.vorp ?? detail?.vorp,
    projected_games: detail?.projected_games,
    role: detail?.role,
    depth_chart_status: detail?.depth_chart_status,
    low_confidence: detail?.low_confidence,
  };
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

function buildCardHtml(p, { full = false } = {}) {
  const d = p.drivers || {};
  const br = fantasyBreakdown(p);
  const absTotal = Math.max(Math.abs(br.total), 0.01);
  const vorp = p.vorp ?? 0;
  const titleTag = full ? "h2" : "h3";
  const titleId = full ? ' id="playerModalTitle"' : "";

  const pills = [];
  pills.push(`<span class="pill">${escapeHtml(p.team || "FA")}</span>`);
  if (p.low_confidence) pills.push(`<span class="pill warn">Low confidence</span>`);
  else if (state.cardById.size) pills.push(`<span class="pill ok">Modeled</span>`);
  if (d.any_stat_low_n_flag) pills.push(`<span class="pill warn">Low-N interval</span>`);
  if (d.role_discount_applied) pills.push(`<span class="pill warn">Role discounted</span>`);

  const contrib = [
    { label: "Passing", pts: br.pass, cls: "pass" },
    { label: "Rushing", pts: br.rush, cls: "rush" },
    { label: "Receiving", pts: br.rec, cls: "rec" },
  ].filter((c) => Math.abs(c.pts) >= 0.05);

  return `
    <div class="player-card-header">
      <span class="pos-badge ${escapeHtml(p.position)}">${escapeHtml(p.position)}</span>
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
        <span class="value">${formatVorp(vorp)}</span>
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

    ${buildHistoryTableHtml(p, { perGame: false, season: SEASON })}
  `;
}

function onPlayerMouseOver(e) {
  const link = e.target.closest(".player-link");
  if (!link || !els.modal.hidden) return;
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

function bind() {
  els.hoverCard = document.getElementById("playerHoverCard");
  els.modal = document.getElementById("playerModal");
  els.modalBody = document.getElementById("playerModalBody");
  els.main = document.querySelector(".main");

  document.getElementById("posFilter").addEventListener("change", (e) => {
    state.pos = e.target.value;
    hideHoverCard();
    render();
  });
  document.getElementById("searchInput").addEventListener("input", (e) => {
    state.search = e.target.value;
    hideHoverCard();
    render();
  });
  document.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      FantasySort.toggleSort(state, th.dataset.sort, {
        keyProp: "sortKey",
        dirProp: "sortDir",
        defaultDir: th.dataset.defaultDir || "asc",
      });
      hideHoverCard();
      render();
    });
  });

  els.main.addEventListener("mouseover", onPlayerMouseOver);
  els.main.addEventListener("mouseout", onPlayerMouseOut);
  els.main.addEventListener("focusin", onPlayerFocusIn);
  els.main.addEventListener("focusout", onPlayerFocusOut);
  els.main.addEventListener("click", onPlayerClick);

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
}

bind();
loadData().catch((err) => {
  document.getElementById("metaLine").textContent = String(err.message || err);
  document.getElementById("tableBody").innerHTML = "";
});
