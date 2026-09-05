# Draft checklist prepare notes
#
# Context ranks (1 = best) from sealed Vegas consensus + Sharp SOS + O-line board:
#   draft_assistant/data/vegas_consensus_{season}.json
#   draft_assistant/data/sharp_fantasy_sos_{season}.json
#   draft_assistant/data/ol_unit_ranks_{season}.json
# Raw multi-book scrapes live under:
#   draft_assistant/data/vegas_raw/
#
# Volume ranks (public boards lack season attempt/target O/Us — checked
# DraftKings/FanDuel/BettingPros/VegasInsider/Unabated/Kalshi/Polymarket):
#   QB: total yards (pass + rush), rush yards, passing TDs
#   RB: receptions, total yards (rush + receiving), total TDs (rush + receiving)
#   WR/TE: receptions, receiving yards, receiving TDs
# Yardage / team points: median sportsbook season O/Us + Vegas-implied PPG.
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
