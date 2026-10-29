from __future__ import annotations

from collections import defaultdict

from .factorized import ACTION_DIMENSION, DIMENSIONS, FactorizedHypothesis, FactorizedObservation, FactorizedState, ProbeAction
from .models import ActionScore
from .objective import entropy, normalise_beliefs


def predicted_token(hypothesis: FactorizedHypothesis, action: ProbeAction) -> str:
    if action == ProbeAction.ATTEMPT_SOLUTION:
        return "solution:unknown"
    dimension = ACTION_DIMENSION[action]
    return f"{dimension.value}:{hypothesis.bits[DIMENSIONS.index(dimension)]}"


def information_gain(
    hypotheses: list[FactorizedHypothesis], beliefs: dict[int, float], action: ProbeAction
) -> float:
    if action == ProbeAction.ATTEMPT_SOLUTION:
        return 0.0
    groups: dict[str, list[int]] = defaultdict(list)
    for hypothesis in hypotheses:
        groups[predicted_token(hypothesis, action)].append(hypothesis.id)
    prior = entropy(list(beliefs.values()))
    posterior = 0.0
    for ids in groups.values():
        mass = sum(beliefs[i] for i in ids)
        if mass <= 0:
            continue
        posterior += mass * entropy([beliefs[i] / mass for i in ids])
    return prior - posterior


def score_factorized_actions(
    hypotheses: list[FactorizedHypothesis], beliefs: dict[int, float]
) -> list[ActionScore]:
    return score_from_marginals(marginals(hypotheses, beliefs))


def score_from_marginals(
    marginal_probabilities: list[tuple[float, float]],
) -> list[ActionScore]:
    """Calculate probe utility from six marginals without enumerating 64 worlds."""
    best_belief = 1.0
    for p0, p1 in marginal_probabilities:
        best_belief *= max(p0, p1)
    rows = []
    for action in ProbeAction:
        if action == ProbeAction.ATTEMPT_SOLUTION:
            progress = best_belief
            utility = 0.90 * best_belief - 0.05 * (1.0 - best_belief)
            rows.append(ActionScore(action, 0.0, progress, utility))
        else:
            dimension = ACTION_DIMENSION[action]
            gain = entropy(marginal_probabilities[DIMENSIONS.index(dimension)])
            rows.append(ActionScore(action, gain, 0.0, gain))
    return rows


def update_factorized_beliefs(
    hypotheses: list[FactorizedHypothesis],
    beliefs: dict[int, float],
    action: ProbeAction,
    observation: FactorizedObservation,
    proposed_hypothesis_id: int | None = None,
) -> dict[int, float]:
    if action == ProbeAction.ATTEMPT_SOLUTION:
        if observation.token == "solution:correct":
            return {h.id: float(h.id == proposed_hypothesis_id) for h in hypotheses}
        posterior = {h.id: (0.0 if h.id == proposed_hypothesis_id else beliefs[h.id]) for h in hypotheses}
        return normalise_beliefs(posterior)
    posterior = {
        h.id: beliefs[h.id] if predicted_token(h, action) == observation.token else 0.0
        for h in hypotheses
    }
    return normalise_beliefs(posterior)


def marginals(hypotheses: list[FactorizedHypothesis], beliefs: dict[int, float]) -> list[tuple[float, float]]:
    output = []
    for i in range(len(DIMENSIONS)):
        p1 = sum(beliefs[h.id] for h in hypotheses if h.bits[i] == 1)
        output.append((1.0 - p1, p1))
    return output

