from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil, log2
import hashlib
import re

import numpy as np

from .arc_encoding import ArcEncoding
from .arc_models import ArcActionCandidate, ArcHypothesis, ArcProposal
from .models import ActionScore
from .objective import entropy


def normalized_confidences(proposal: ArcProposal) -> dict[int, float]:
    total = sum(max(0.0, hypothesis.confidence) for hypothesis in proposal.hypotheses)
    if total <= 0:
        return {hypothesis.id: 1.0 / len(proposal.hypotheses) for hypothesis in proposal.hypotheses}
    return {hypothesis.id: max(0.0, hypothesis.confidence) / total for hypothesis in proposal.hypotheses}


def evidence_adjusted_confidences(proposal: ArcProposal, history: list[dict] | None = None) -> dict[int, float]:
    """Combine the current proposal with the most recent persistent belief ledger."""
    history = history or []
    ledger = history[-1].get("belief_ledger", []) if history else []
    by_statement = {
        " ".join(str(item.get("statement", "")).lower().split()): item
        for item in ledger
        if item.get("statement")
    }
    status_weight = {"supported": 1.25, "weakened": 0.55, "rejected": 0.02, "uncertain": 0.85, "new": 1.0}
    raw: dict[int, float] = {}
    for hypothesis in proposal.hypotheses:
        key = " ".join(hypothesis.statement.lower().split())
        evidence = by_statement.get(key)
        # Once a belief is established, additional confidence should not make it dominate forever.
        value = max(0.001, min(0.75, hypothesis.confidence))
        if evidence:
            value *= max(0.001, float(evidence.get("confidence", hypothesis.confidence)))
            value *= status_weight.get(str(evidence.get("status", "uncertain")), 0.85)
        raw[hypothesis.id] = value
    total = sum(raw.values())
    normalized = {key: value / total for key, value in raw.items()}
    floor = min(0.02, 0.5 / len(normalized))
    floored = {key: max(floor, value) for key, value in normalized.items()}
    floored_total = sum(floored.values())
    return {key: value / floored_total for key, value in floored.items()}


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "may", "be", "this", "that"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stop}


def grounded_goal_tokens(hypothesis: ArcHypothesis) -> set[str]:
    goal = hypothesis.grounded_goal
    values = (
        hypothesis.object_role, hypothesis.goal_condition, goal.target_component,
        goal.target_attributes, goal.current_bounds, goal.destination, goal.desired_state,
        goal.proposed_action, goal.prerequisites, goal.partial_success,
    )
    meaningful = [value for value in values if str(value).strip().lower() not in {"", "none", "unknown", "unspecified", "n/a"}]
    text = " ".join(map(str, meaningful)).lower()
    for source, target in (("grey", "gray"), ("gold", "yellow"), ("footprint", "target"), ("onto", "overlap")):
        text = text.replace(source, target)
    return _tokens(text)


def grounded_goal_score(hypothesis: ArcHypothesis) -> float:
    values = tuple(vars(hypothesis.grounded_goal).values())
    specified = sum(str(value).strip().lower() not in {"", "none", "unknown", "unspecified", "n/a"} for value in values)
    return specified / len(values)


def canonical_families(proposal: ArcProposal, history: list[dict] | None = None) -> dict[int, str]:
    """Group semantic near-duplicates and preserve their identity across turns."""
    representatives: list[tuple[str, set[str]]] = []
    prior = []
    for item in reversed(history or []):
        family = str(item.get("goal_family", ""))
        tokens = set(item.get("goal_signature_tokens", []))
        if family and tokens and all(existing[0] != family for existing in prior):
            prior.append((family, tokens))
    result: dict[int, str] = {}
    for hypothesis in proposal.hypotheses:
        tokens = grounded_goal_tokens(hypothesis)
        if len(tokens) < 3:
            tokens |= _tokens(f"{hypothesis.role} {hypothesis.statement} {hypothesis.rule} {hypothesis.goal_condition} {hypothesis.object_role}")
        family = None
        best = 0.0
        for name, reference in prior:
            similarity = len(tokens & reference) / max(1, len(tokens | reference))
            if similarity > best:
                family, best = name, similarity
        if best < 0.28:
            family, best = None, 0.0
        if family is None:
            for name, reference in representatives:
                similarity = len(tokens & reference) / max(1, len(tokens | reference))
                if similarity > best:
                    family, best = name, similarity
        if family is None or (not any(name == family for name, _ in prior) and best < 0.62):
            digest = hashlib.sha1(" ".join(sorted(tokens)).encode("utf-8")).hexdigest()[:10]
            family = f"{hypothesis.role}-{digest}"
            representatives.append((family, tokens))
        result[hypothesis.id] = family
    return result


def goal_relevance(hypothesis) -> float:
    text = f"{hypothesis.statement} {hypothesis.goal_condition}".lower()
    role_base = {"mechanism": 0.05, "obstacle": 0.25, "interaction": 0.45, "goal": 0.8, "terminal": 1.0}
    value = role_base.get(hypothesis.role, 0.05)
    if any(word in text for word in ("goal", "target", "objective", "win", "finish", "complete", "score", "reach")):
        value += 0.2
    value += 0.25 * grounded_goal_score(hypothesis)
    return min(1.0, value)


def goal_action_alignment(hypothesis, candidate: ArcActionCandidate) -> float:
    prediction = hypothesis.predictions.get(candidate.candidate_id, "UNKNOWN").upper()
    value = 0.7 if prediction in {"PROGRESS", "TERMINAL"} else 0.0
    goal_words = _tokens(hypothesis.goal_condition)
    action_words = _tokens(candidate.description)
    value += min(0.3, 0.1 * len(goal_words & action_words))
    if hypothesis.role in {"goal", "interaction", "terminal"} and candidate.action_type in {"ACTION5", "ACTION6"}:
        value += 0.15
    proposed = hypothesis.grounded_goal.proposed_action.upper()
    if candidate.action_type in proposed:
        value += 0.25
    return min(1.0, value)


def action_compatibility(hypothesis: ArcHypothesis, candidate: ArcActionCandidate) -> float:
    proposed = hypothesis.grounded_goal.proposed_action.upper()
    mentioned = re.findall(r"ACTION[1-7]", proposed)
    if mentioned and candidate.action_type not in mentioned:
        return 0.0
    if candidate.action_type == "ACTION6" and (candidate.x is None or candidate.y is None):
        return 0.0
    return 1.0


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
    return (
        f"{candidate.action_type}:{candidate.x}:{candidate.y}"
        f"->{candidate.next_action_type}:{candidate.next_x}:{candidate.next_y}"
    )


def score_arc_actions(
    proposal: ArcProposal,
    history: list[dict] | None = None,
    frame_signature: str | None = None,
    information_weight: float = 1.0,
    progress_weight: float = 1.2,
    risk_weight: float = 1.0,
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
        outcome_bonus = min(3.0, 2.5 * sum(int(item.get("levels_delta", 0)) for item in matching))
        repetition_penalty = min(1.5, 0.12 * len(matching))
        repetition_penalty += min(1.5, 0.75 * len(same_context))
        repetition_penalty += min(0.75, 0.25 * sum(int(item.get("changed_cells", 0)) == 0 for item in matching))
        if history and history[-1].get("action_signature") == signature:
            repetition_penalty += 0.35

        utility = (
            information_weight * gain
            + progress_weight * candidate.expected_progress
            - risk_weight * candidate.risk
            + 0.35 * (1.0 - candidate.risk)
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
    history: list[dict] | None = None,
    selected_hypotheses: list[int] | None = None,
    invalid_penalty: float = 8.0,
    interaction_weight: float = 1.25,
    max_steps: int = 40,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a history-aware hypothesis-action cost diagonal.

    The action utility remains shared across hypotheses, while the interaction
    term rewards a supported hypothesis whose predicted outcome would be
    informative. This makes changing the belief ledger change the QAOA energy
    landscape rather than merely breaking otherwise equal action ties.
    """
    beliefs = evidence_adjusted_confidences(proposal, history)
    utilities = {row.action: row.utility for row in scores}
    utility_values = list(utilities.values())
    utility_min, utility_max = min(utility_values), max(utility_values)
    utility_span = utility_max - utility_min
    normalized_utilities = {
        key: (value - utility_min) / utility_span if utility_span > 1e-9 else 0.5
        for key, value in utilities.items()
    }
    costs = np.full(encoding.state_count, invalid_penalty, dtype=float)
    valid = encoding.valid_mask()
    outcome_mass: dict[str, dict[str, float]] = {}
    for candidate in encoding.candidates:
        masses: dict[str, float] = defaultdict(float)
        for hypothesis in encoding.hypotheses:
            outcome = hypothesis.predictions.get(candidate.candidate_id, "unknown")
            masses[outcome] += beliefs[hypothesis.id]
        outcome_mass[candidate.candidate_id] = dict(masses)

    pair_rows: list[tuple[int, str, int, float, bool]] = []
    interactions: list[float] = []
    unknown_pairs = 0
    selection_counts = Counter(selected_hypotheses or [])
    families = canonical_families(proposal, history)
    family_sizes = Counter(families.values())
    family_representatives: dict[str, int] = {}
    for hypothesis_id, family in families.items():
        current = family_representatives.get(family)
        if current is None or beliefs[hypothesis_id] > beliefs[current]:
            family_representatives[family] = hypothesis_id
    for hypothesis in encoding.hypotheses:
        if family_representatives[families[hypothesis.id]] != hypothesis.id:
            for candidate in encoding.candidates:
                valid[encoding.encode(hypothesis.id, candidate.candidate_id)] = False
    role_counts = Counter(str(item.get("selected_role", "")) for item in (history or [])[-6:])
    turns_since_progress = 0
    for item in reversed(history or []):
        if int(item.get("levels_delta", 0)) > 0:
            break
        turns_since_progress += 1
    stagnation_threshold = max(1, ceil(0.15 * max_steps))
    active_goal_family = ""
    active_goal_stagnation = 0
    if history:
        last = history[-1]
        if str(last.get("selected_role", "")) in {"goal", "terminal", "interaction"}:
            active_goal_family = str(last.get("goal_family", ""))
            for item in reversed(history):
                if (int(item.get("levels_delta", 0)) > 0
                        or str(item.get("goal_family", "")) != active_goal_family):
                    break
                active_goal_stagnation += 1
    stagnation_pressure = min(
        1.0,
        max(0, active_goal_stagnation - stagnation_threshold + 1) / stagnation_threshold,
    )
    credited_families: Counter[str] = Counter()
    for item in history or []:
        credited_families[str(item.get("goal_family"))] += float(item.get("progress_credit", 0.0))
    term_rows: dict[str, dict[str, float | str]] = {}
    ranked_beliefs = sorted(beliefs, key=beliefs.get, reverse=True)
    second_hypothesis = ranked_beliefs[1] if len(ranked_beliefs) > 1 else None
    for hypothesis in encoding.hypotheses:
        for candidate in encoding.candidates:
            index = encoding.encode(hypothesis.id, candidate.candidate_id)
            prediction = hypothesis.predictions.get(candidate.candidate_id, "unknown")
            mass = max(1e-9, outcome_mass[candidate.candidate_id].get(prediction, 0.0))
            information_contribution = beliefs[hypothesis.id] * -log2(mass)
            belief_support = beliefs[hypothesis.id]
            diversity_bonus = 0.12 / (1 + selection_counts[hypothesis.id])
            second_best_bonus = 0.0
            family = families[hypothesis.id]
            goal_score = goal_relevance(hypothesis)
            alignment = goal_action_alignment(hypothesis, candidate)
            compatibility = action_compatibility(hypothesis, candidate)
            falsification = 0.25 if hypothesis.falsification.lower() not in {"", "unspecified", "unknown"} else 0.0
            duplicate_penalty = 0.12 * max(0, family_sizes[family] - 1)
            role_penalty = min(0.4, 0.08 * role_counts[hypothesis.role]) if turns_since_progress else 0.0
            is_stagnant_goal = bool(active_goal_family and family == active_goal_family)
            stagnant_family_penalty = min(0.7, 0.12 * active_goal_stagnation * stagnation_pressure) if is_stagnant_goal else 0.0
            alternative_goal_pressure = stagnation_pressure if not is_stagnant_goal else 0.0
            goal_bonus = goal_score * (0.35 + 0.85 * alternative_goal_pressure)
            alignment_bonus = 0.8 * alignment
            progress_credit = 0.35 * credited_families[family]
            compatibility_penalty = 0.65 * (1.0 - compatibility)
            interaction = (information_contribution + 0.18 * belief_support + diversity_bonus + second_best_bonus
                           + goal_bonus + alignment_bonus + falsification + progress_credit
                           - duplicate_penalty - role_penalty - stagnant_family_penalty - compatibility_penalty)
            unknown = prediction == "unknown"
            unknown_pairs += int(prediction == "unknown")
            interactions.append(interaction)
            pair_rows.append((index, candidate.candidate_id, hypothesis.id, interaction, unknown))
            term_rows[f"{hypothesis.id}:{candidate.candidate_id}"] = {
                "family": family, "role": hypothesis.role, "mechanism_support": round(belief_support, 6),
                "goal_relevance": round(goal_score, 6), "goal_action_alignment": round(alignment, 6),
                "grounding_score": round(grounded_goal_score(hypothesis), 6),
                "action_compatibility": compatibility,
                "falsification_value": falsification, "duplicate_penalty": round(duplicate_penalty, 6),
                "role_repetition_penalty": round(role_penalty, 6), "stagnation_goal_bonus": round(goal_bonus, 6),
                "stagnation_pressure": round(stagnation_pressure, 6),
                "stagnant_family_penalty": round(stagnant_family_penalty, 6),
                "progress_credit": round(progress_credit, 6), "effective_confidence": min(0.75, hypothesis.confidence),
            }
    interaction_min, interaction_max = min(interactions, default=0.0), max(interactions, default=0.0)
    interaction_span = interaction_max - interaction_min
    for index, candidate_id, _hypothesis_id, interaction, unknown in pair_rows:
        normalized_interaction = (interaction - interaction_min) / interaction_span if interaction_span > 1e-9 else 0.5
        costs[index] = -normalized_utilities[candidate_id] - interaction_weight * normalized_interaction + (0.35 if unknown else 0.0)
    diagnostics = {
        "cost_model": "goal-aligned-hypothesis-action-v2",
        "evidence_beliefs": {str(key): round(value, 6) for key, value in beliefs.items()},
        "interaction_weight": interaction_weight,
        "raw_utility_min": float(utility_min),
        "raw_utility_max": float(utility_max),
        "interaction_min": float(interaction_min),
        "interaction_max": float(interaction_max),
        "selected_hypothesis_counts": {str(key): value for key, value in selection_counts.items()},
        "second_hypothesis_bonus_id": second_hypothesis,
        "unknown_prediction_pairs": unknown_pairs,
        "canonical_families": {str(key): value for key, value in families.items()},
        "family_sizes": dict(family_sizes),
        "family_representatives": family_representatives,
        "stagnation_turns": turns_since_progress,
        "stagnation_threshold": stagnation_threshold,
        "stagnation_pressure": round(stagnation_pressure, 6),
        "active_goal_family": active_goal_family,
        "active_goal_stagnation": active_goal_stagnation,
        "term_contributions": term_rows,
        "catalog_coverage": {
            str(hypothesis.id): {
                "grounding_score": round(grounded_goal_score(hypothesis), 6),
                "compatible_candidates": sum(action_compatibility(hypothesis, candidate) > 0 for candidate in encoding.candidates),
                "target_grounded": hypothesis.grounded_goal.target_component.lower() not in {"", "unknown", "unspecified"},
                "destination_grounded": hypothesis.grounded_goal.destination.lower() not in {"", "unknown", "unspecified"},
            }
            for hypothesis in encoding.hypotheses
        },
    }
    return costs, valid, diagnostics
