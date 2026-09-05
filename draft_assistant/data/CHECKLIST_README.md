# Draft checklist prepare notes
#
# The checklist board is transcribed from Sunday Sports Society screenshots
# (PPR positional ranks + context checks). It does NOT use our ADP/ECR or
# model offense/SOS ranks for checks.
#
# Sealed inputs:
#   draft_assistant/data/screenshot_checklist_{season}.json
#   draft_assistant/data/ol_unit_ranks_{season}.json
#
# QB/RB TOP 16 O-LINE is taken from ol_unit_ranks (O-line unit rankings
# screenshot), replacing the SSS OL column. WR/TE have no OL column.
#
# Identity (player_id / team) is joined from team_stats_{season}.json by name.
#
# Refresh:
#   python -m src.draft_assistant.checklist_prepare --season 2026
#
# OL-only rewrite of an existing checklist JSON:
#   python -m src.draft_assistant.checklist_prepare --season 2026 --patch-ol-only
#
# The checklist is NOT part of a sealed release bundle. It lives at
# draft_assistant/data/draft_checklist_{season}.json and is served from there.
# Do not copy it into releases/<namespace>/.
