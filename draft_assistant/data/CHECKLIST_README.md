# Draft checklist prepare notes
#
# Market flavor is half-PPR, 12-team ADP only (Fantasy Football Calculator) plus
# FantasyPros PPR ECR. Context checks still use that market snapshot.
#
# All-tab order is league-scoring VORP (6-pt pass TD, -3 INT, 0 PPR) with
# replacement baselines QB12 / RB30 / WR42 / TE12 — matching the Fantasy
# Projections /draft "League" board. O-line team ranks are not rewritten by
# the board-order step.
#
# Pre-draft refresh (host with network):
#   python -m src.draft_assistant.compare_prepare --season 2026 --teams 12 --ffc-scoring half-ppr
#   python -m src.draft_assistant.checklist_prepare --season 2026
#   python -m src.draft_assistant.checklist_apply_board_order \
#     --season 2026 --projections /path/to/legacy_draft_projections.json
#
# Capture projections from the legacy /draft RSC payload (player_gsis_id +
# proj_* components). The apply step sets overall_rank / pos_market_rank /
# league_pts and meta.rank_source=league_vorp; it leaves teams[] (O-line) alone.
#
# OL unit ranks require projections.db (ol_quality). Offense + SOS can use the
# DB or nflverse fallbacks; SOS is omitted when 2026 REG schedules are missing.
#
# The checklist is NOT part of a sealed release bundle. It lives at
# draft_assistant/data/draft_checklist_{season}.json and is served from there.
# Do not copy it into releases/<namespace>/: every browser-consumed artifact in
# a frozen namespace needs a release_bundle_manifest.json entry with a sha256,
# or scripts/verify_browser_surfaces.py cannot integrity-check it. Sealing the
# checklist means publishing a new namespace, not editing a frozen one.
#
# compare_prepare rewrites comparison_{season}.json in place and drops the
# board_generated_at / board_model_id / board_source_file /
# market_snapshot_preserved provenance keys. Only ever point it at
# draft_assistant/data/, never at a releases/<namespace>/ copy.
