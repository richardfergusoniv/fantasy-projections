# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.
from src.projection.weekly.features.depth import attach_depth_features
from src.projection.weekly.features.injuries import attach_injury_features
from src.projection.weekly.features.leakage import filter_as_of
from src.projection.weekly.features.panel import build_player_week_panel, load_panel, save_panel
from src.projection.weekly.features.team_context import add_opponent_defense_features

__all__ = [
    "add_opponent_defense_features",
    "attach_depth_features",
    "attach_injury_features",
    "build_player_week_panel",
    "filter_as_of",
    "load_panel",
    "save_panel",
]
