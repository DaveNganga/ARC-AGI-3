from __future__ import annotations

from collections import defaultdict

import numpy as np

from .arc_encoding import ArcEncoding
from .arc_models import ArcActionCandidate, ArcProposal
from .models import ActionScore
from .objective import entropy


def normalized_confidences(proposal: ArcProposal) -> dict[int, float]:
    total = sum(max(0.0, hypothesis.confidence) for hypothesis in proposal.hypotheses)
    if total <= 0:
        return {hypothesis.id: 1.0 / len(proposal.hypotheses) for hypothesis in proposal.hypotheses}
    return {hypothesis.id: max(0.0, hypothesis.confidence) / total for hypothesis in proposal.hypotheses}


def action_information_gain(proposal: ArcProposal, candidate_id: str) -> float:
    beliefs = normalized_confidences(proposal)
    prior = entropy(list(beliefs.values()))
    groups: dict[str, list[int]] = defaultdict(list)
    for hypothesis in proposal.hypotheses:
        groups[hypothesis.predictions.get(candidate_id, "unknown")].append(hypothesis.id)
    posterior = 0.0
    for ids in groups.values():
        mass = sum(beliefs[i] for i in ids)
        if mass > 0:
            posterior += mass * entropy([beliefs[i] / mass for i in ids])
    return prior - posterior


def action_signature(candidate: ArcActionCandidate) -> str:
    """Identify equivalent physical actions even when the LLM renames a candidate."""
    return f"{candidate.action_type}:{candidate.x}:{candidate.y}"


def score_arc_actions(
    proposal: ArcProposal,
    history: list[dict] | None = None,
    frame_signature: str | None = None,
    information_weight: float = 1.0,
    progress_weight: float = 0.65,
    risk_weight: float = 0.65,
) -> list[ActionScore]:
    history = history or []
    scores = []
    for candidate in proposal.action_candidates:
        gain = action_information_gain(proposal, candidate.candidate_id)
        signature = action_signature(candidate)
        matching = [item for item in history if item.get("action_signature") == signature]
        same_context = [
            item for item in matching
            if frame_signature and item.get("before_frame_signature") == frame_signature
        ]

        novelty_bonus = 0.20 if not matching else 0.0
        if frame_signature and not same_context:
            novelty_bonus += 0.10

        # Actual level progress is strong evidence that an action is worth revisiting.
        outcome_bonus = min(1.5, 1.5 * sum(int(item.get("levels_delta", 0)) for item in matching))
        repetition_penalty = min(1.5, 0.12 * len(matching))
        repetition_penalty += min(1.5, 0.75 * len(same_context))
        repetition_penalty += min(0.75, 0.25 * sum(int(item.get("changed_cells", 0)) == 0 for item in matching))
        if history and history[-1].get("action_signature") == signature:
            repetition_penalty += 0.35

        utility = (
            information_weight * gain
            + progress_weight * candidate.expected_progress
            - risk_weight * candidate.risk
            + novelty_bonus
            + outcome_bonus
            - repetition_penalty
        )
        scores.append(ActionScore(
            candidate.candidate_id,
            gain,
            candidate.expected_progress,
            utility,
            novelty_bonus,
            outcome_bonus,
            repetition_penalty,
        ))
    return scores


def build_arc_costs(
    encoding: ArcEncoding,
    proposal: ArcProposal,
    scores: list[ActionScore],
    invalid_penalty: float = 8.0,
    belief_tiebreak: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    beliefs = normalized_confidences(proposal)
    utilities = {row.action: row.utility for row in scores}
    costs = np.full(encoding.state_count, invalid_penalty, dtype=float)
    valid = encoding.valid_mask()
    for hypothesis in encoding.hypotheses:
        for candidate in encoding.candidates:
            index = encoding.encode(hypothesis.id, candidate.candidate_id)
            costs[index] = -utilities[candidate.candidate_id] - belief_tiebreak * beliefs[hypothesis.id]
    return costs, valid
