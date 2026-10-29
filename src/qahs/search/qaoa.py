"""Small exact-statevector QAOA backend.

The circuit pattern adapts the instructor-provided malware clustering notebook:
uniform state, diagonal cost phase, X mixer, COBYLA parameter search, and sampled
bitstring extraction. The Max-Cut graph objective is replaced by QAHS costs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..compiler import Encoding
from ..models import ActionScore, SearchResult


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    axes = [qubit] + [i for i in range(n_qubits) if i != qubit]
    inverse = np.argsort(axes)
    tensor = np.transpose(tensor, axes).reshape(2, -1)
    tensor = gate @ tensor
    tensor = tensor.reshape([2] + [2] * (n_qubits - 1))
    return np.transpose(tensor, inverse).reshape(-1)


def qaoa_state(costs: np.ndarray, theta: np.ndarray, depth: int) -> np.ndarray:
    n_qubits = int(np.log2(len(costs)))
    if 1 << n_qubits != len(costs):
        raise ValueError("Cost diagonal length must be a power of two")
    theta = np.asarray(theta, dtype=float)
    if len(theta) != 2 * depth:
        raise ValueError("QAOA requires depth beta and depth gamma values")
    betas, gammas = theta[:depth], theta[depth:]
    state = np.ones(len(costs), dtype=np.complex128) / np.sqrt(len(costs))
    for beta, gamma in zip(betas, gammas):
        state *= np.exp(-1j * gamma * costs)
        mixer = np.array(
            [[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]],
            dtype=np.complex128,
        )
        for qubit in range(n_qubits):
            state = _apply_single_qubit_gate(state, mixer, qubit, n_qubits)
    return state


class QAOASearch:
    name = "qaoa_statevector"

    def __init__(self, depth: int = 2, restarts: int = 5, maxiter: int = 120, shots: int = 1024, seed: int = 7):
        self.depth = depth
        self.restarts = restarts
        self.maxiter = maxiter
        self.shots = shots
        self.seed = seed

    def select(self, encoding: Encoding, action_scores: list[ActionScore], costs: np.ndarray, valid: np.ndarray) -> SearchResult:
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.restarts):
            x0 = rng.uniform(-np.pi, np.pi, 2 * self.depth)
            result = minimize(
                lambda theta: float(np.dot(np.abs(qaoa_state(costs, theta, self.depth)) ** 2, costs)),
                x0,
                method="COBYLA",
                options={"maxiter": self.maxiter, "rhobeg": 0.5},
            )
            if best is None or result.fun < best.fun:
                best = result

        assert best is not None
        probabilities = np.abs(qaoa_state(costs, best.x, self.depth)) ** 2
        sampled = rng.choice(len(costs), size=self.shots, p=probabilities)
        valid_samples = sampled[valid[sampled]]
        if not len(valid_samples):
            raise RuntimeError("QAOA produced no valid samples; increase the constraint penalty or shots")
        idx = int(valid_samples[np.argmin(costs[valid_samples])])
        decoded = encoding.decode(idx)
        if decoded is None:
            raise RuntimeError("QAOA returned an invalid encoded state")
        hypothesis_id, action = decoded
        utility = next(s.utility for s in action_scores if s.action == action)
        return SearchResult(
            self.name,
            action,
            hypothesis_id,
            float(utility),
            idx,
            {
                "qubits": encoding.total_bits,
                "depth_p": self.depth,
                "shots": self.shots,
                "expected_cost": float(best.fun),
                "selected_probability": float(probabilities[idx]),
                "valid_sample_rate": float(np.mean(valid[sampled])),
                "optimizer_evaluations": int(getattr(best, "nfev", 0)),
            },
        )

