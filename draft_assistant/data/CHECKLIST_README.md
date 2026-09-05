# Draft checklist prepare notes
#
# Checks: transcribed from Sunday Sports Society screenshots
#   draft_assistant/data/screenshot_checklist_{season}.json
# QB/RB TOP 16 O-LINE: overwritten from
#   draft_assistant/data/ol_unit_ranks_{season}.json
#
# Board order: market average of available sources
#   - ESPN PPR ADP
#   - Fantasy Football Calculator PPR ADP
#   - MyFantasyLeague ADP
#   - FantasyPros ECR (from comparison_{season}.json)
# Missing sources are skipped per player; mean of whatever is present.
#
# Refresh (needs network for live ADP pulls):
#   python -m src.draft_assistant.checklist_prepare --season 2026
#
# OL-only rewrite:
#   python -m src.draft_assistant.checklist_prepare --season 2026 --patch-ol-only
#
# Output:
#   draft_assistant/data/draft_checklist_{season}.json
# Not part of a sealed release bundle — do not copy into releases/<namespace>/.
