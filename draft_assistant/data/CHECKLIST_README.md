# Draft checklist prepare notes
#
# Market flavor is half-PPR, 12-team ADP only (Fantasy Football Calculator) plus
# FantasyPros PPR ECR. There is no per-league scoring branch on the checklist
# endpoint.
#
# Pre-draft refresh (host with network):
#   python -m src.draft_assistant.compare_prepare --season 2026 --teams 12 --ffc-scoring half-ppr
#   python -m src.draft_assistant.checklist_prepare --season 2026
#
# OL unit ranks prefer sealed draft_assistant/data/ol_unit_ranks_{season}.json
# (manual composite board from the O-line rankings screenshot). That file drives
# TOP 16 O-LINE checks for QB/RB and the O-line pane unit ranks. When the sealed
# board is absent, prepare falls back to projections.db ol_quality; if both are
# missing, OL checks are omitted. Offense + SOS can use the DB or nflverse
# fallbacks; SOS is omitted when 2026 REG schedules are missing.
#
# Offline OL-only refresh (no nflverse / no projections.db):
#   python -m src.draft_assistant.checklist_prepare --season 2026 --patch-ol-only
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
