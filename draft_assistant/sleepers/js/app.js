// Sleepers: the deep board, ranked by upside against cost.
//
// Two constraints shape this page and neither is cosmetic.
//
// 1. The accuracy-first ensemble does not cover anyone here. It was fit, selected
//    and applied only where ADP <= 120 (TOP_ADP in accuracy_first.py), so every
//    player on this page carries the untouched incumbent forecast. Inclusion is
//    instead gated on scripts/evaluate_deep_band_accuracy.py, which measures how
//    often a projected-points band actually produced a startable season.
//
// 2. Sentiment is a secondary column and never the sort key. The parsed research
//    corpus is 91.5% positive / 1.2% negative, is missing for most of this
//    population, and has never been validated against outcomes.

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

// Players inside this ADP cutoff belong to the draft board, not here. It is the
// same constant the accuracy-first ensemble uses to bound its own population.
const TOP_ADP = 120;

// Fallback band edges, used only if deep_band_accuracy.json has not been built.
// The real edges come from the measurement.
const FALLBACK_BANDS = [
  { name: "deep_core", low: 61, high: 120 },
  { name: "deep_primary", low: 121, high: 200 },
  { name: "deep_speculative", low: 201, high: 300 },
];

const BAND_LABEL = {
  deep_core: { text: "61-120", cls: "core" },
  deep_primary: { text: "121-200", cls: "primary" },
  deep_speculative: { text: "201-300", cls: "speculative" },
};

// Quarterbacks are not sleeper output. They post large raw season totals while
// being replacement-rich, so a points-rank band ranks them alongside flex
// players it cannot actually compare them to -- deep QBs led this board on raw
// upside while carrying roughly -210 VORP. The band measurement reports a
// matching RB/WR/TE-only population for the rates quoted below.
const EXCLUDED_POSITIONS = new Set(["QB"]);

const state = {
  rows: [],
  byId: new Map(),
  cardById: new Map(),
  accuracy: null,
  bands: FALLBACK_BANDS,
  pos: "ALL",
  band: "ALL",
  search: "",
  sortKey: "fantasy_pts_p90",
  sortDir: "desc",
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

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function pct(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Math.round(100 * Number(v))}%`;
}

// The board is mid-migration from p_top* to p_finish_top*. Read whichever the
// published JSON actually carries so this page survives the switch either way.
function finishOdds(p) {
  const candidates = [
    ["p_finish_top24", "top 24"],
    ["p_top24", "top 24"],
    ["p_finish_top36", "top 36"],
    ["p_top36", "top 36"],
    ["p_finish_top48", "top 48"],
  ];
  for (const [key, label] of candidates) {
    if (p[key] != null) return { value: Number(p[key]), label };
  }
  if (p.p_vorp_positive != null) {
    return { value: Number(p.p_vorp_positive), label: "positive VORP" };
  }
  return { value: null, label: null };
}

// Finish probabilities sit behind a promotion gate. While it holds, the board
// publishes the columns empty rather than shipping uncalibrated probabilities,
// so say that instead of rendering a bare dash that reads as a broken cell.
function finishGateNote(board) {
  const sim = board?.meta?.draft_value_simulation;
  if (!sim || sim.applied) return null;
  return sim.reason === "finish_probability_gate_hold" ||
    sim.reason === "missing_finish_probability_gate"
    ? "Held by the finish-probability promotion gate — see output/model_v3/finish_probability_gate.json"
    : `Not published (${sim.reason || "unknown reason"})`;
}

// sentiment_confidence is not a measurement — it is a constant chosen by how the
// mention was parsed (0.75 table row, 0.62 bullet) blended with a hardcoded 1/3
// for the market family. Report the provenance it actually encodes instead of
// dressing it up as a confidence percentage.
function sentimentProvenance(p) {
  const c = p.sentiment_confidence;
  if (c == null) return "no signal";
  const near = (x) => Math.abs(Number(c) - x) < 0.02;
  if (near(1 / 3)) return "market gap only (ECR vs ADP)";
  if (near(0.62)) return "one bullet mention";
  if (near(0.75)) return "one table row";
  if (near(0.7467)) return "bullet mention + market gap";
  if (near(0.8333)) return "table row + market gap";
  return p.sentiment_coverage === "high" ? "text + market gap" : "text only";
}

function sentimentHtml(p) {
  const score = p.sentiment_score;
  if (score == null) {
    return '<span class="sentiment-score none" title="No mention found in the research corpus. Absence is not a negative signal.">—</span>';
  }
  const cls = score > 15 ? "positive" : score < -15 ? "negative" : "neutral";
  const sign = score > 0 ? "+" : "";
  const title =
    `Buzz ${sign}${Math.round(score)} — percentile of reviewed mentions within ${p.position}, ` +
    `across the whole board (not just this page). ` +
    `Evidence: ${sentimentProvenance(p)}. ` +
    `Research frozen 2026-08-24. Unvalidated: the corpus is 91.5% positive, 1.2% negative.`;
  return `<span class="sentiment-score ${cls}" title="${escapeHtml(title)}">${sign}${Math.round(score)}</span>`;
}

function seasonRange(seasons) {
  const list = (seasons || []).slice().sort();
  if (!list.length) return "historical";
  return list.length === 1 ? String(list[0]) : `${list[0]}–${list[list.length - 1]}`;
}

function bandOf(rank) {
  for (const b of state.bands) {
    if (rank >= b.low && rank <= b.high) return b.name;
  }
  return null;
}

async function loadJson(path, { required = true, hint = "" } = {}) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    if (required) throw new Error(`Missing ${path}${hint ? ` — ${hint}` : ""}`);
    return null;
  }
  return res.json();
}

async function loadData() {
  const ctx = await FantasyRelease.loadContext({ season: SEASON, dataRoot: "../data" });
  state.release = ctx;
  const [board, compare] = await Promise.all([
    FantasyRelease.loadJson(ctx, "players"),
    FantasyRelease.loadJson(ctx, "comparison"),
  ]);

  try {
    state.accuracy = await FantasyRelease.loadJson(ctx, "deep_band_accuracy");
  } catch (err) {
    if (ctx.mode === "namespaced" && ctx.urls.deep_band_accuracy) throw err;
    state.accuracy = null;
  }
  if (state.accuracy?.bands) {
    const admitted = state.accuracy.sleeper_band?.admitted_bands || [];
    const bands = admitted
      .map((name) => {
        const range = state.accuracy.bands[name]?.rank_range || [];
        return { name, low: range[0], high: range[1] ?? Infinity };
      })
      .filter((b) => b.low != null);
    if (bands.length) state.bands = bands;
  }

  document.getElementById("seasonBadge").textContent = String(
    board.meta?.season || SEASON
  );

  const market = new Map((compare?.players || []).map((p) => [p.player_id, p]));
  const players = board.players || [];

  // Rank the whole board by projected season points. This is deliberately not
  // overall_rank: overall_rank is VORP-based and diverges from points order by
  // more than 50 places for most of the board, and the band measurement that
  // gates this page is cut on points. Measuring one ordering and selecting on
  // another would put a different population on the page than the one tested.
  const ranked = players
    .filter((p) => p.fantasy_pts_season != null)
    .slice()
    .sort((a, b) => Number(b.fantasy_pts_season) - Number(a.fantasy_pts_season));
  ranked.forEach((p, i) => {
    p.points_rank = i + 1;
  });

  const rows = [];
  for (const p of ranked) {
    const m = market.get(p.player_id) || {};
    const adp = m.adp ?? null;
    if (adp != null && adp <= TOP_ADP) continue; // belongs to the draft board
    if (p.source === "replacement_level") continue; // synthetic filler, not a player
    if (EXCLUDED_POSITIONS.has(p.position)) continue;
    const band = bandOf(p.points_rank);
    if (!band) continue;

    const p50 = p.fantasy_pts_p50 ?? null;
    const p90 = p.fantasy_pts_p90 ?? null;
    const odds = finishOdds(p);
    rows.push({
      ...p,
      band,
      ecr: m.ecr ?? null,
      adp,
      // Raw point gap, not a ratio: (p90 - p50) / p50 explodes toward a
      // meaningless #1 whenever the median sits near zero, which is common in
      // this population because availability drags the median down.
      upside_gap: p50 != null && p90 != null ? p90 - p50 : null,
      market_gap: m.ecr != null ? p.points_rank - Number(m.ecr) : null,
      finish_odds: odds.value,
      finish_label: odds.label,
    });
  }

  state.rows = rows;
  state.byId = new Map(rows.map((p) => [p.player_id, p]));
  state.finishGateNote = finishGateNote(board);

  try {
    const cards = await loadJson(`../data/team_stats_${SEASON}.json`, { required: false });
    if (cards) state.cardById = new Map((cards.players || []).map((p) => [p.player_id, p]));
  } catch {
    /* cards degrade without team-stats detail */
  }

  renderAccuracyNote();
  renderAttribution(board, compare);
  render();
}

function renderAccuracyNote() {
  const note = document.getElementById("accuracyNote");
  const acc = state.accuracy;
  if (!acc) {
    note.innerHTML =
      '<p class="accuracy-headline">Band measurement not built.</p>' +
      "<p>Run <code>python scripts/evaluate_deep_band_accuracy.py</code> to measure how often " +
      "each projected-points band actually produced a startable season. Until then this page " +
      "is showing its fallback band edges with no accuracy evidence behind them.</p>";
    return;
  }

  const parts = state.bands.map((b) => {
    // sleeper_population is the RB/WR/TE-only cut, matching what this page
    // renders; the pooled band rate would describe a different population.
    const blk = acc.bands[b.name]?.sleeper_population || acc.bands[b.name] || {};
    return (
      `<strong>${BAND_LABEL[b.name]?.text || b.name}</strong>: ` +
      `${Math.round(100 * (blk.p_startable_100 ?? 0))}% produced a 100+ point season ` +
      `(${Math.round(100 * (blk.p_starter_150 ?? 0))}% reached 150+, n=${blk.n ?? 0})`
    );
  });

  note.innerHTML =
    '<p class="accuracy-headline">These players are outside the top-120 ADP, which the accuracy build never covered.</p>' +
    `<p>The accuracy-first ensemble was fit and applied only where ADP &le; ${TOP_ADP}, so everyone here ` +
    "carries the untouched incumbent forecast. Inclusion is gated on a separate measurement of " +
    `${seasonRange(acc.seasons)} outcomes, banded by projected season-points rank: ${parts.join("; ")}.</p>` +
    "<p>Ranks beyond 300 are excluded on evidence: that band produced a startable season " +
    `${Math.round(100 * (acc.bands?.tail?.sleeper_population?.p_startable_100 ?? 0))}% of the time. ` +
    "Its MAE and rank correlation look better than any band shown here, but both are artifacts " +
    "of a near-zero floor. Ranks inside " +
    `${(acc.bands?.top60?.rank_range || [1, 60]).join("-")} are excluded for the opposite reason: ` +
    `at ${Math.round(100 * (acc.bands?.top60?.sleeper_population?.p_startable_100 ?? 0))}% they are ` +
    "the most accurate band measured, which makes them draft-board players rather than sleepers.</p>" +
    `<p>Quarterbacks are excluded. Every rate above is the ${(acc.sleeper_band?.positions || []).join("/")} ` +
    "cut, matching what this table shows. QBs post large raw season totals while being " +
    "replacement-rich, so a points-rank band ranks them against flex players it cannot " +
    "meaningfully compare them to.</p>" +
    "<p><strong>Proj</strong> is the blended board projection; <strong>Median</strong> and " +
    "<strong>Upside</strong> come from the simulated season distribution, and the two are not " +
    "centered on each other — board-wide the simulated median runs about 0.85&times; the blended " +
    "projection, and higher than it for roughly a quarter of players. Compare Median against " +
    "Upside, and Proj against Proj; the <strong>Gap</strong> column is the safe upside measure " +
    "because both of its terms come from the same distribution.</p>" +
    (state.finishGateNote
      ? `<p><strong>Finish</strong> is empty: ${escapeHtml(state.finishGateNote)}. The board withholds ` +
        "these probabilities rather than publishing uncalibrated ones, so the column will fill in " +
        "once that gate passes.</p>"
      : "");
}

function renderAttribution(board, compare) {
  const ecr = compare?.meta?.ecr || {};
  const adp = compare?.meta?.adp || {};
  document.getElementById("attribution").innerHTML =
    `Board: ${escapeHtml(board.meta?.model_id || "unknown model")} · ` +
    `${escapeHtml(board.meta?.scoring || "half-PPR")} · ` +
    `ECR: FantasyPros via nflverse/DynastyProcess${ecr.scrape_date ? ` (${escapeHtml(ecr.scrape_date)})` : ""} · ` +
    `ADP: Fantasy Football Calculator${adp.end_date ? ` (through ${escapeHtml(adp.end_date)})` : ""}. ` +
    "Cost is shown as ECR because it covers roughly four times as much of this population as ADP does. " +
    "Band accuracy from <code>scripts/evaluate_deep_band_accuracy.py</code>.";
}

function filteredRows() {
  let rows = state.rows;
  if (state.pos !== "ALL") rows = rows.filter((p) => p.position === state.pos);
  if (state.band !== "ALL") rows = rows.filter((p) => p.band === state.band);
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

function marketGapHtml(p) {
  if (p.market_gap == null) return '<td class="num muted">—</td>';
  const cls = p.market_gap < -10 ? "market-cheap" : p.market_gap > 10 ? "market-rich" : "muted";
  const sign = p.market_gap > 0 ? "+" : "";
  const title =
    p.market_gap < 0
      ? "We rank this player higher than the expert consensus does."
      : "The expert consensus ranks this player higher than we do.";
  return `<td class="num ${cls}" title="${escapeHtml(title)}">${sign}${Math.round(p.market_gap)}</td>`;
}

function render() {
  const body = document.getElementById("tableBody");
  const rows = filteredRows();

  document.getElementById("metaLine").textContent =
    `${rows.length} of ${state.rows.length} eligible players · ` +
    `outside top-${TOP_ADP} ADP · projected points rank ` +
    `${state.bands[0]?.low ?? "?"}–${state.bands[state.bands.length - 1]?.high ?? "?"} · ` +
    "sorted by upside unless you change it";

  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="15" class="empty-state">No players match these filters.</td></tr>';
    return;
  }

  body.innerHTML = rows
    .map((p) => {
      const band = BAND_LABEL[p.band] || { text: p.band, cls: "" };
      const ecrTitle = p.adp != null ? `ADP ${fmt(p.adp, 1)}` : "No ADP — undrafted in the sample";
      return `<tr>
        <td class="num">${p.points_rank}</td>
        <td class="col-player">
          <button type="button" class="player-link" data-player-id="${escapeHtml(p.player_id)}" aria-haspopup="dialog">${escapeHtml(p.display_name || "")}</button>
        </td>
        <td><span class="pos-badge pos-${escapeHtml(p.position || "")}">${escapeHtml(p.position || "")}</span></td>
        <td>${escapeHtml(p.team || "")}</td>
        <td class="num">${fmt(p.fantasy_pts_season, 0)}</td>
        <td class="num muted">${fmt(p.fantasy_pts_p50, 0)}</td>
        <td class="num upside">${fmt(p.fantasy_pts_p90, 0)}</td>
        <td class="num">${p.upside_gap != null ? `+${fmt(p.upside_gap, 0)}` : "—"}</td>
        <td class="num" title="${escapeHtml(ecrTitle)}">${p.ecr != null ? fmt(p.ecr, 0) : "—"}</td>
        ${marketGapHtml(p)}
        <td class="num">${formatVorp(p.vorp)}</td>
        <td class="num ${p.finish_odds == null ? "muted" : ""}" title="${escapeHtml(p.finish_label ? `Probability of a ${p.finish_label} finish` : state.finishGateNote || "No simulated finish probability published for this player")}">${p.finish_odds == null ? "gated" : pct(p.finish_odds)}</td>
        <td><span class="band-pill ${band.cls}">${band.text}</span></td>
        <td class="col-sentiment">${sentimentHtml(p)}</td>
        <td class="role-cell">${escapeHtml((p.role || "—").replace(/_/g, " "))}</td>
      </tr>`;
    })
    .join("");

  FantasySort.markStaticHeaders(
    document.getElementById("sleepersTable"),
    state.sortKey,
    state.sortDir
  );
}

function cardPlayer(playerId) {
  const row = state.byId.get(playerId);
  const detail = state.cardById.get(playerId);
  if (!row && !detail) return null;
  return {
    ...(detail || {}),
    ...(row || {}),
    drivers: detail?.drivers || {},
    pg: detail?.pg || {},
    season: detail?.season || {},
    history: detail?.history || [],
    depth_rank: detail?.depth_rank ?? null,
    fantasy_pts_low: detail?.drivers?.fantasy_pts_low,
    fantasy_pts_high: detail?.drivers?.fantasy_pts_high,
    projected_games: row?.projected_games ?? detail?.projected_games,
    depth_chart_status: row?.depth_chart_status ?? detail?.depth_chart_status,
  };
}

function fantasyBreakdown(p) {
  const s = p.pg || {};
  const pass =
    SCORING.passYd * (s.passing_yards || 0) +
    SCORING.passTd * (s.passing_tds || 0) +
    SCORING.int * (s.interceptions || 0);
  const rush = SCORING.rushYd * (s.rushing_yards || 0) + SCORING.rushTd * (s.rushing_tds || 0);
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
  const titleTag = full ? "h2" : "h3";
  const titleId = full ? ' id="playerModalTitle"' : "";

  const pills = [];
  pills.push(`<span class="pill">${escapeHtml(p.team || "FA")}</span>`);
  pills.push(`<span class="pill">${escapeHtml(BAND_LABEL[p.band]?.text || p.band || "")}</span>`);
  if (p.low_confidence) pills.push(`<span class="pill warn">Low confidence</span>`);
  if (d.any_stat_low_n_flag) pills.push(`<span class="pill warn">Low-N interval</span>`);
  if (d.role_discount_applied) pills.push(`<span class="pill warn">Role discounted</span>`);
  pills.push(
    '<span class="pill warn" title="The accuracy-first ensemble was fit and applied only inside the top-120 ADP. This forecast is the untouched incumbent.">Outside accuracy build</span>'
  );

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
  )} · upside case for a deep pick</p>
        <div class="player-card-pills">${pills.join("")}</div>
      </div>
    </div>

    <div class="card-stats">
      <div class="card-stat">
        <span class="label">Median</span>
        <span class="value">${fmt(p.fantasy_pts_p50, 0)}</span>
        <span class="hint">50th percentile season</span>
      </div>
      <div class="card-stat">
        <span class="label">Upside</span>
        <span class="value">${fmt(p.fantasy_pts_p90, 0)}</span>
        <span class="hint">90th percentile season</span>
      </div>
      <div class="card-stat">
        <span class="label">Cost</span>
        <span class="value">${p.ecr != null ? fmt(p.ecr, 0) : "—"}</span>
        <span class="hint">${p.adp != null ? `ECR · ADP ${fmt(p.adp, 1)}` : "ECR · no ADP"}</span>
      </div>
      <div class="card-stat">
        <span class="label">VORP</span>
        <span class="value">${formatVorp(p.vorp)}</span>
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
                const share = Math.min(100, Math.round((100 * Math.abs(c.pts)) / absTotal));
                return `<div class="driver-row">
                  <span class="driver-label">${c.label}</span>
                  <div class="driver-bar ${c.cls}"><span style="width:${share}%"></span></div>
                  <span class="driver-value">${c.pts >= 0 ? "+" : ""}${fmt(c.pts, 1)} (${share}%)</span>
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
    e.relatedTarget && e.relatedTarget.closest ? e.relatedTarget.closest(".player-link") : null;
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
  document.getElementById("bandFilter").addEventListener("change", (e) => {
    state.band = e.target.value;
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
  document.getElementById("tableBody").innerHTML =
    `<tr><td colspan="15" class="empty-state">${escapeHtml(String(err.message || err))}</td></tr>`;
});
