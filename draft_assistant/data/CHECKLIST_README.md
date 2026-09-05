# Draft checklist prepare notes
#
# Context ranks (1 = best) from sealed Vegas consensus + Sharp SOS + O-line board:
#   draft_assistant/data/vegas_consensus_{season}.json
#   draft_assistant/data/sharp_fantasy_sos_{season}.json
#   draft_assistant/data/ol_unit_ranks_{season}.json
# Raw multi-book scrapes live under:
#   draft_assistant/data/vegas_raw/
#
# Checklist columns (all positions):
#   FP, offense pts, offense yards, O-line, Sharp fantasy SOS
# FP = half-PPR / 4-pt pass TD from the median of scraped Vegas
#   yards/receptions/TD O/Us (component volume ranks folded into FP).
#   Attempts/targets are not publicly posted. INTs/fumbles omitted.
# Offense pts/yards: median Vegas-implied team season totals.
# Board: every rostered QB/RB/WR/TE in team_stats_{season}.json
#
# Board order: market average of available sources
#   - ESPN PPR ADP
#   - Fantasy Football Calculator PPR ADP
#   - MyFantasyLeague ADP
#   - FantasyPros ECR (from comparison_{season}.json)
#
# Rebuild consensus then checklist (needs network for live ADP pulls):
#   python -m src.draft_assistant.vegas_consensus
#   python -m src.draft_assistant.checklist_prepare --season 2026
#
# OL-only rewrite:
#   python -m src.draft_assistant.checklist_prepare --season 2026 --patch-ol-only
#
# Output:
#   draft_assistant/data/draft_checklist_{season}.json
# Not part of a sealed release bundle — do not copy into releases/<namespace>/.
