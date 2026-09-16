"""Run the Role 2 Vegas weekly props evaluation comparator (research/shadow only).

Does not promote, reseal, train, blend Role 3, or change APP_PROJECTION_SOURCE.

Examples:

  uv run python scripts/compare_shadow_vegas_props.py --dry-run
  uv run python scripts/compare_shadow_vegas_props.py --props PATH --board PATH --outcomes PATH
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.weekly_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
