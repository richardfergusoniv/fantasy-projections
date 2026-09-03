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
# OL unit ranks require projections.db (ol_quality). Offense + SOS can use the
# DB or nflverse fallbacks; SOS is omitted when 2026 REG schedules are missing.
