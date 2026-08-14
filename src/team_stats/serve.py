"""Serve the combined Fantasy Tools app (draft + team projections).

Thin wrapper around draft_assistant.serve so older commands still work.
"""

from __future__ import annotations

from src.draft_assistant.serve import main

if __name__ == "__main__":
    main()
