const assert = require("node:assert/strict");

require("../../draft_assistant/js/projection_adjustment.js");

const scoring = {
  passYd: 0.04,
  passTd: 4,
  int: -2,
  rushYd: 0.1,
  rushTd: 6,
  rec: 0.5,
  recYd: 0.1,
  recTd: 6,
};

const detail = {
  pg: { targets: 8, receptions: 5, receiving_yards: 70, receiving_tds: 0.5 },
  season: {
    targets: 136,
    receptions: 85,
    receiving_yards: 1190,
    receiving_tds: 8.5,
  },
};
const original = JSON.parse(JSON.stringify(detail));
const draft = { fantasy_pts: 15, fantasy_pts_season: 272 };

const result = globalThis.FantasyProjectionAdjustment.derive(draft, detail, scoring);

assert.equal(result.meta.adjusted, true);
assert.equal(result.meta.method, "proportional_stat_mix");
assert.ok(
  Math.abs(globalThis.FantasyProjectionAdjustment.scoreStats(result.pg, scoring) - 15) < 1e-9
);
assert.ok(
  Math.abs(
    globalThis.FantasyProjectionAdjustment.scoreStats(result.season, scoring) - 272
  ) < 1e-9
);
assert.equal(result.pg.receptions / result.pg.targets, 5 / 8);
assert.deepEqual(detail, original, "canonical detail stats must not be mutated");

const fallback = globalThis.FantasyProjectionAdjustment.derive(
  { fantasy_pts: 10, fantasy_pts_season: 170 },
  { pg: {}, season: {} },
  scoring
);
assert.equal(fallback.meta.adjusted, false);
assert.deepEqual(fallback.pg, {});
assert.deepEqual(fallback.season, {});

const merged = globalThis.FantasyProjectionAdjustment.mergeBoard(
  [{ player_id: "wr1", ...detail, fantasy_pts: 12.5, fantasy_pts_season: 212.5 }],
  [{ player_id: "wr1", ...draft }],
  scoring,
  "accuracy_first_ensemble"
);
assert.equal(merged[0].fantasy_pts_season, 272);
assert.equal(merged[0].projection_model_id, "accuracy_first_ensemble");
assert.equal(merged[0].projection_adjustment.adjusted, true);
assert.deepEqual(merged[0].canonical_season, detail.season);
assert.ok(
  Math.abs(
    globalThis.FantasyProjectionAdjustment.scoreStats(merged[0].season, scoring) - 272
  ) < 1e-9
);

console.log("projection adjustment tests passed");
