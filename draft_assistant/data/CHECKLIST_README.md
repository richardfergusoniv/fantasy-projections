# Draft checklist prepare notes
#
# Market flavor is half-PPR, 12-team ADP only (Fantasy Football Calculator) plus
# FantasyPros PPR ECR. There is no per-league scoring branch on the checklist
# endpoint.
#
# Current context checks (2026 board): transcribed from
# @SUNDAYSPORTSSOCIETY positional checklist graphics via
#   python scripts/apply_sss_checklist_override.py
# That overlay replaces projection-derived offense/QB/SOS/volume flags for the
# published SSS board (WR60 / RB48 / QB24 / TE24) and clears checks for
# everyone else so sources are not mixed. TOP 16 O-LINE for QB/RB is taken from
# the 32-team OL unit rating chart (tackles weighted) applied in the same
# script — not from projections.db. Re-run the overlay after checklist_prepare
# if you refresh market ranks.
#
# Pre-draft market refresh (host with network):
#   python -m src.draft_assistant.compare_prepare --season 2026 --teams 12 --ffc-scoring half-ppr
#   python -m src.draft_assistant.checklist_prepare --season 2026
#   python scripts/apply_sss_checklist_override.py
#
# OL unit ranks for the checklist's TOP 16 O-LINE column come from the
# transcribed 32-team OL rating chart in apply_sss_checklist_override.py.
# projections.db ol_quality is no longer required for that player check.
# Offense + SOS context still come from the SSS graphics (or, if regenerating
# without the overlay, from the DB / nflverse fallbacks).
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
