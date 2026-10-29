from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .compiler import Encoding
from .models import Action, ActionScore, GameState, Hypothesis, ObservationKind
from .predictor import predict, progress_value


def entropy(probabilities: list[float] | np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def normalise_beliefs(beliefs: dict[int, float]) -> dict[int, float]:
    total = float(sum(beliefs.values()))
    if total <= 0:
        raise ValueError("Belief mass must be positive")
    return {k: float(v / total) for k, v in beliefs.items()}


def update_beliefs(
    hypotheses: list[Hypothesis],
    beliefs: dict[int, float],
    state_before: GameState,
    action: Action,
    observed: ObservationKind,
    mismatch_likelihood: float = 1e-6,
) -> dict[int, float]:
    posterior = {}
    for hypothesis in hypotheses:
        likelihood = 1.0 if predict(hypothesis, state_before, action) == observed else mismatch_likelihood
        posterior[hypothesis.id] = beliefs[hypothesis.id] * likelihood
    return normalise_beliefs(posterior)


def expected_information_gain(
    hypotheses: list[Hypothesis], beliefs: dict[int, float], state: GameState, action: Action
) -> float:
    prior_h = entropy(list(beliefs.values()))
    groups: dict[ObservationKind, list[int]] = {}
    for hypothesis in hypotheses:
        groups.setdefault(predict(hypothesis, state, action), []).append(hypothesis.id)

    expected_posterior_h = 0.0
    for ids in groups.values():
        outcome_probability = sum(beliefs[i] for i in ids)
        conditional = [beliefs[i] / outcome_probability for i in ids]
        expected_posterior_h += outcome_probability * entropy(conditional)
    return prior_h - expected_posterior_h


def score_actions(
    hypotheses: list[Hypothesis],
    beliefs: dict[int, float],
    state: GameState,
    actions: tuple[Action, ...],
    information_weight: float = 1.0,
    progress_weight: float = 0.35,
) -> list[ActionScore]:
    scores = []
    for action in actions:
        ig = expected_information_gain(hypotheses, beliefs, state, action)
        progress = sum(
            beliefs[h.id] * progress_value(h, state, action) for h in hypotheses
        )
        scores.append(ActionScore(action, ig, progress, information_weight * ig + progress_weight * progress))
    return scores


def build_cost_diagonal(
    encoding: Encoding,
    action_scores: list[ActionScore],
    beliefs: dict[int, float],
    invalid_penalty: float = 4.0,
    belief_tiebreak: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode QAHS utility as a diagonal minimization Hamiltonian."""
    utilities = {row.action: row.utility for row in action_scores}
    costs = np.full(encoding.state_count, invalid_penalty, dtype=float)
    valid = np.zeros(encoding.state_count, dtype=bool)
    for hypothesis in encoding.hypotheses:
        for action in encoding.actions:
            idx = encoding.encode(hypothesis.id, action)
            costs[idx] = -utilities[action] - belief_tiebreak * beliefs[hypothesis.id]
            valid[idx] = True
    return costs, valid

