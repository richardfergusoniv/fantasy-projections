"""Port weekly-v2 CLI scripts from fantasy-projections-2."""

from __future__ import annotations

import re
from pathlib import Path

SIBLING = Path(__file__).resolve().parents[1].parent / "fantasy-projections-2" / "scripts"
DST = Path(__file__).resolve().parents[1] / "scripts"

MAPPING = {
    "ingest_data.py": "weekly_v2_ingest.py",
    "build_features.py": "weekly_v2_build_features.py",
    "tune_preseason.py": "weekly_v2_tune_preseason.py",
    "train.py": "weekly_v2_train.py",
    "preseason_eval.py": "weekly_v2_evaluate.py",
    "walkforward_eval.py": "weekly_v2_walkforward_eval.py",
    "fit_calibration.py": "weekly_v2_fit_calibration.py",
    "evaluate.py": "weekly_v2_evaluate_season.py",
    "project.py": "weekly_v2_project.py",
    "project_season.py": "weekly_v2_project_season.py",
}

IMPORT_RE = re.compile(r"\b(from|import)\s+projections\.")


def main() -> None:
    for src_name, dst_name in MAPPING.items():
        src = SIBLING / src_name
        if not src.exists():
            print(f"skip missing {src_name}")
            continue
        text = src.read_text(encoding="utf-8")
        text = IMPORT_RE.sub(r"\1 src.projection.weekly.", text)
        text = text.replace("scripts/train.py", "scripts/weekly_v2_train.py")
        text = text.replace("models/", "output/weekly_v2/models/")
        (DST / dst_name).write_text(text, encoding="utf-8")
        print(f"ported {dst_name}")


if __name__ == "__main__":
    main()
