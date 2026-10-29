from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arc_models import ArcActionCandidate, ArcHypothesis


@dataclass(frozen=True)
class ArcEncoding:
    hypotheses: tuple[ArcHypothesis, ...]
    candidates: tuple[ArcActionCandidate, ...]

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def hypothesis_bits(self) -> int:
        return max(1, math.ceil(math.log2(len(self.hypotheses))))

    @property
    def action_bits(self) -> int:
        return max(1, math.ceil(math.log2(len(self.candidates))))

    @property
    def total_bits(self) -> int:
        return self.hypothesis_bits + self.action_bits

    @property
    def state_count(self) -> int:
        return 1 << self.total_bits

    def encode(self, hypothesis_id: int, candidate_id: str) -> int:
        action_index = self.actions.index(candidate_id)
        return (hypothesis_id << self.action_bits) | action_index

    def decode(self, index: int) -> tuple[int, str] | None:
        action_mask = (1 << self.action_bits) - 1
        action_index = index & action_mask
        hypothesis_id = index >> self.action_bits
        if hypothesis_id >= len(self.hypotheses) or action_index >= len(self.candidates):
            return None
        return hypothesis_id, self.actions[action_index]

    def valid_mask(self) -> np.ndarray:
        mask = np.zeros(self.state_count, dtype=bool)
        for hypothesis in self.hypotheses:
            for candidate in self.candidates:
                mask[self.encode(hypothesis.id, candidate.candidate_id)] = True
        return mask
