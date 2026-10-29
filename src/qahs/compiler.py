from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Action, Hypothesis


@dataclass(frozen=True)
class Encoding:
    hypothesis_bits: int
    action_bits: int
    hypotheses: tuple[Hypothesis, ...]
    actions: tuple[Action, ...]

    @property
    def total_bits(self) -> int:
        return self.hypothesis_bits + self.action_bits

    @property
    def state_count(self) -> int:
        return 1 << self.total_bits

    def encode(self, hypothesis_id: int, action: Action) -> int:
        action_id = self.actions.index(action)
        return (hypothesis_id << self.action_bits) | action_id

    def decode(self, index: int) -> tuple[int, Action] | None:
        action_mask = (1 << self.action_bits) - 1
        action_id = index & action_mask
        hypothesis_id = index >> self.action_bits
        if hypothesis_id >= len(self.hypotheses) or action_id >= len(self.actions):
            return None
        return hypothesis_id, self.actions[action_id]


def build_encoding(hypotheses: list[Hypothesis], actions: tuple[Action, ...]) -> Encoding:
    if not hypotheses or not actions:
        raise ValueError("At least one hypothesis and action are required")
    h_bits = max(1, math.ceil(math.log2(len(hypotheses))))
    a_bits = max(1, math.ceil(math.log2(len(actions))))
    return Encoding(h_bits, a_bits, tuple(hypotheses), tuple(actions))

