from __future__ import annotations

import numpy as np

from ..compiler import Encoding
from ..models import ActionScore, SearchResult


class GreedyBeliefSearch:
    """Simple non-search baseline: choose the largest immediate action utility."""

    name = "greedy_baseline"

    def select(self, encoding: Encoding, action_scores: list[ActionScore], costs: np.ndarray, valid: np.ndarray) -> SearchResult:
        # Legacy actions are Enums; ARC actions are plain candidate IDs.
        row = max(action_scores, key=lambda s: (s.utility, str(getattr(s.action, "value", s.action))))
        candidates = []
        for hypothesis in encoding.hypotheses:
            idx = encoding.encode(hypothesis.id, row.action)
            if valid[idx]:
                candidates.append(idx)
        idx = min(candidates, key=lambda candidate: costs[candidate])
        hypothesis_id, _ = encoding.decode(idx)
        return SearchResult(self.name, row.action, hypothesis_id, row.utility, idx, {"policy": "maximum immediate utility"})

