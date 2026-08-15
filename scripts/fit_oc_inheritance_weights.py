"""CLI: LOSO grid-search OC inheritance weights."""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.coordinator.inheritance import loso_fit_inheritance_weights
from src.projection.data_prep import get_conn


def main():
    conn = get_conn()
    try:
        summary = loso_fit_inheritance_weights(conn)
    finally:
        conn.close()
    if not summary.get("ok"):
        print(summary)
        return 1
    printable = {k: v for k, v in summary.items() if k != "grid"}
    print(json.dumps(printable, indent=2, default=str))
    print(summary["grid"].to_string(index=False))
    if summary["recommend_update"]:
        print("RECOMMEND: update INHERITANCE_WEIGHTS to best")
    else:
        print("KEEP: judgment INHERITANCE_WEIGHTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
