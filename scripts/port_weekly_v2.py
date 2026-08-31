"""One-shot port helper: copy team-first weekly modules from fantasy-projections-2."""

from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1].parent / "fantasy-projections-2" / "src" / "projections"
DST_ROOT = Path(__file__).resolve().parents[1] / "src" / "projection" / "weekly"

FILES = [
    "pipeline/accounting.py",
    "pipeline/availability.py",
    "pipeline/veteran_projector.py",
    "pipeline/rookie_projector.py",
    "pipeline/season_projector.py",
    "pipeline/__init__.py",
    "models/base.py",
    "models/registry.py",
    "models/volume.py",
    "models/efficiency.py",
    "models/team_totals.py",
    "models/calibration.py",
    "models/rookie.py",
    "models/__init__.py",
    "features/injuries.py",
    "features/depth.py",
    "features/team_context.py",
    "features/leakage.py",
    "features/rolling.py",
    "features/effective_depth.py",
    "features/contracts.py",
    "features/sleeper.py",
    "features/xfp.py",
    "features/advanced_public.py",
    "features/panel.py",
    "features/rookie_college.py",
    "features/__init__.py",
    "config/scoring.py",
    "config/paths.py",
    "config/__init__.py",
    "data/teams.py",
    "data/ids.py",
    "data/nflverse_loader.py",
    "data/espn_injuries.py",
    "data/sleeper.py",
    "data/cfbd_loader.py",
    "data/__init__.py",
    "scoring/fantasy_points.py",
    "scoring/__init__.py",
]

PROVENANCE = (
    "# Ported from fantasy-projections-2 (team-first weekly v2). "
    "See docs/WEEKLY_V2_PORT_PROVENANCE.md.\n"
)
IMPORT_RE = re.compile(r"\b(from|import)\s+projections\.")


def add_provenance(text: str) -> str:
    if text.startswith("# Ported from fantasy-projections-2"):
        return text
    if text.startswith('"""'):
        end = text.find('"""', 3)
        if end != -1:
            return text[: end + 3] + "\n" + PROVENANCE.rstrip() + text[end + 3 :]
    return PROVENANCE + text


def main() -> None:
    for rel in FILES:
        src = SRC_ROOT / rel
        dst = DST_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        text = IMPORT_RE.sub(r"\1 src.projection.weekly.", text)
        dst.write_text(add_provenance(text), encoding="utf-8")
        print(f"ported {rel}")

    init = '''"""Team-first weekly projection pipeline (ported from fantasy-projections-2).

See docs/WEEKLY_V2_PORT_PROVENANCE.md for source mapping.
"""
from src.projection.weekly.pipeline import (
    apply_accounting,
    assert_shares_sum,
    normalize_shares,
    project_season,
    project_veterans_week,
    project_week_with_rookies,
    write_projections,
    write_season_outputs,
)

__all__ = [
    "apply_accounting",
    "assert_shares_sum",
    "normalize_shares",
    "project_season",
    "project_veterans_week",
    "project_week_with_rookies",
    "write_projections",
    "write_season_outputs",
]
'''
    (DST_ROOT / "__init__.py").write_text(init, encoding="utf-8")
    print("created __init__.py")


if __name__ == "__main__":
    main()
