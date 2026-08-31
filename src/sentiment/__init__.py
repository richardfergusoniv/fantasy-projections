"""Point-in-time player sentiment features.

The production projection models do not consume these features until a
historical ablation explicitly enables a position in the sentiment manifest.
The current implementation publishes an auditable diagnostic built from the
local Perplexity research markdowns and the frozen market-consensus snapshot.
"""

__all__: list[str] = []
