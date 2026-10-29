from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .factorized import FactorizedHypothesis, ProbeAction
from .models import ActionScore


@dataclass
class QUBOModel:
    variable_names: list[str]
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    constant: float = 0.0

    @property
    def variable_count(self) -> int:
        return len(self.variable_names)

    def cost_diagonal(self) -> np.ndarray:
        count = 1 << self.variable_count
        bits = ((np.arange(count)[:, None] >> np.arange(self.variable_count)) & 1).astype(float)
        costs = np.full(count, self.constant, dtype=float)
        for index, coefficient in self.linear.items():
            costs += coefficient * bits[:, index]
        for (left, right), coefficient in self.quadratic.items():
            costs += coefficient * bits[:, left] * bits[:, right]
        return costs

    def to_ising(self) -> tuple[float, dict[int, float], dict[tuple[int, int], float]]:
        """Return offset, Z coefficients, and ZZ coefficients for x=(1-Z)/2."""
        offset = self.constant
        z: dict[int, float] = {}
        zz: dict[tuple[int, int], float] = {}
        for index, coefficient in self.linear.items():
            offset += coefficient / 2
            z[index] = z.get(index, 0.0) - coefficient / 2
        for (left, right), coefficient in self.quadratic.items():
            offset += coefficient / 4
            z[left] = z.get(left, 0.0) - coefficient / 4
            z[right] = z.get(right, 0.0) - coefficient / 4
            zz[(left, right)] = coefficient / 4
        return offset, z, zz

    def resources(self, depth: int = 1) -> dict[str, int | float]:
        return {
            "decision_qubits": self.variable_count,
            "linear_terms": len(self.linear),
            "quadratic_terms": len(self.quadratic),
            "estimated_cost_layer_rz": len(self.linear) * depth,
            "estimated_cost_layer_rzz": len(self.quadratic) * depth,
            "statevector_complex_amplitudes": 1 << self.variable_count,
            "statevector_bytes_complex128": (1 << self.variable_count) * 16,
        }


@dataclass(frozen=True)
class OneHotEncoding:
    hypotheses: tuple[FactorizedHypothesis, ...]
    actions: tuple[ProbeAction, ...]

    @property
    def hypothesis_bits(self) -> int:
        return 6

    @property
    def total_bits(self) -> int:
        return self.hypothesis_bits + len(self.actions)

    @property
    def state_count(self) -> int:
        return 1 << self.total_bits

    def encode(self, hypothesis_id: int, action: ProbeAction) -> int:
        action_position = self.hypothesis_bits + self.actions.index(action)
        return hypothesis_id | (1 << action_position)

    def decode(self, index: int) -> tuple[int, ProbeAction] | None:
        hypothesis_id = index & ((1 << self.hypothesis_bits) - 1)
        action_bits = [(index >> (self.hypothesis_bits + i)) & 1 for i in range(len(self.actions))]
        if hypothesis_id >= len(self.hypotheses) or sum(action_bits) != 1:
            return None
        return hypothesis_id, self.actions[action_bits.index(1)]

    def valid_mask(self) -> np.ndarray:
        mask = np.zeros(self.state_count, dtype=bool)
        for hypothesis in self.hypotheses:
            for action in self.actions:
                mask[self.encode(hypothesis.id, action)] = True
        return mask


def build_factorized_qubo(
    hypotheses: list[FactorizedHypothesis],
    action_scores: list[ActionScore],
    marginal_probabilities: list[tuple[float, float]],
    one_hot_penalty: float = 4.0,
    belief_tiebreak: float = 0.50,
    consistency_penalty: float = 4.0,
) -> tuple[QUBOModel, OneHotEncoding, np.ndarray]:
    """Build a compact QUBO over six rule bits and seven one-hot actions."""
    actions = tuple(ProbeAction)
    encoding = OneHotEncoding(tuple(hypotheses), actions)
    names = [f"hypothesis_{i}" for i in range(6)] + [f"action_{a.value}" for a in actions]
    model = QUBOModel(names, constant=one_hot_penalty)

    # Prefer the more probable rule value without enumerating 64 hypotheses.
    for index, (p0, p1) in enumerate(marginal_probabilities):
        if p1 >= 1.0 - 1e-12:
            # Observed bit is 1: P(1-x) contributes -P*x plus a constant.
            model.constant += consistency_penalty
            model.linear[index] = -consistency_penalty
        elif p0 >= 1.0 - 1e-12:
            # Observed bit is 0: P*x.
            model.linear[index] = consistency_penalty
        else:
            model.linear[index] = belief_tiebreak * (p0 - p1)

    utilities = {row.action: row.utility for row in action_scores}
    action_indices = []
    for position, action in enumerate(actions):
        index = 6 + position
        action_indices.append(index)
        # P(sum y - 1)^2 contributes -P*y for binary y.
        model.linear[index] = model.linear.get(index, 0.0) - one_hot_penalty - utilities[action]
    for i, left in enumerate(action_indices):
        for right in action_indices[i + 1 :]:
            model.quadratic[(left, right)] = 2.0 * one_hot_penalty

    return model, encoding, encoding.valid_mask()

