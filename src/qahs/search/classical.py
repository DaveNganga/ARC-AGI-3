from __future__ import annotations

import numpy as np

from ..compiler import Encoding
from ..models import ActionScore, SearchResult


class ClassicalSearch:
    name = "classical_exact"

    def select(self, encoding: Encoding, action_scores: list[ActionScore], costs: np.ndarray, valid: np.ndarray) -> SearchResult:
        candidates = np.flatnonzero(valid)
        idx = int(candidates[np.argmin(costs[candidates])])
        hypothesis_id, action = encoding.decode(idx)  # type: ignore[misc]
        utility = next(s.utility for s in action_scores if s.action == action)
        return SearchResult(
            backend=self.name,
            action=action,
            hypothesis_id=hypothesis_id,
            utility=float(utility),
            encoded_index=idx,
            diagnostics={"states_evaluated": int(len(candidates)), "minimum_cost": float(costs[idx])},
        )

