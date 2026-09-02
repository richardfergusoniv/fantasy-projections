"""One-shot port of evaluate modules from sibling repo."""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].parent / "fantasy-projections-2" / "src" / "projections" / "evaluate"
DST = Path(__file__).resolve().parents[1] / "src" / "projection" / "weekly" / "evaluate"
PROV = (
    "# Ported from fantasy-projections-2 (team-first weekly v2). "
    "See docs/WEEKLY_V2_PORT_PROVENANCE.md.\n\n"
)
IMPORT_RE = re.compile(r"\b(from|import)\s+projections\.")


def add_provenance(text: str) -> str:
    if text.startswith("# Ported from fantasy-projections-2"):
        return text
    if text.startswith('"""'):
        end = text.find('"""', 3)
        if end != -1:
            return text[: end + 3] + "\n" + PROV.rstrip() + text[end + 3 :]
    return PROV + text


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.py", "preseason.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        text = IMPORT_RE.sub(r"\1 src.projection.weekly.", text)
        (DST / name).write_text(add_provenance(text), encoding="utf-8")
        print(f"ported {name}")


if __name__ == "__main__":
    main()
