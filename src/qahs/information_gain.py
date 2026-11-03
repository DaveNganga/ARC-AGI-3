"""V11: falsifiable hypothesis search driven by expected information gain.

V10 remains frozen in :mod:`qahs.compositional`.  This module deliberately owns
the V11 control loop so a saved trace always identifies the behavior that made it.
"""
from __future__ import annotations

import json
import math
from itertools import permutations
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .arc_environment import ArcEnvironment
from .arc_models import (
    ArcActionCandidate, ArcBelief, ArcComparisonTrace, ArcEpisodeTrace,
    ArcGroundedGoal, ArcHypothesis, ArcProposal, ArcReflection, ArcTurn,
)
from .arc_reasoner import frame_context, frame_image_data_url
from .compositional import EntityFact, assess_transition, extract_entities
from .models import ActionScore
from .search import ClassicalSearch, QAOASearch


HARNESS_VERSION = "v11-information-gain-search"
INPUT_REPRESENTATION = "stable-interactive-entities+lossless-grid+validated-persistence-v4"
OUTCOMES = ("NO_CHANGE", "LOCAL_CHANGE", "BROAD_CHANGE", "MOVE", "GROUP_TRANSFORM",
            "PROGRESS", "TERMINAL", "FAIL", "UNKNOWN")
RELATION_CHANGES = ("ANY_CHANGE", "NO_CHANGE", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
                    "DISTANCE_DECREASES", "DISTANCE_INCREASES", "GROUP_TRANSFORM")
MOVES = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}


@dataclass(frozen=True)
class Prediction:
    candidate_id: str
    outcome: str
    confidence: float


@dataclass(frozen=True)
class TestableHypothesis:
    id: int
    statement: str
    focus_entity_ids: tuple[str, ...]
    mechanism: str
    success_condition: str
    falsification: str
    prior: float
    predictions: tuple[Prediction, ...]
    test_sequence: tuple[str, ...] = tuple()
    sequence_target_state: str = ""
    stop_condition: str = ""
    persistence_candidate_id: str = ""
    tracked_entity_id: str = ""
    target_entity_id: str = ""
    expected_relation_change: str = "ANY_CHANGE"
    confirmation_outcomes: tuple[str, ...] = ("PROGRESS", "TERMINAL")
    refutation_outcomes: tuple[str, ...] = ("FAIL",)
    actuator_entity_id: str = ""
    affected_entity_ids: tuple[str, ...] = tuple()
    goal_component_id: str = ""
    goal_lineage_action: str = "new"
    parent_goal_component_id: str = ""
    topology_assumption: str = "unknown"

    def prediction_for(self, candidate_id: str) -> Prediction:
        return next((row for row in self.predictions if row.candidate_id == candidate_id),
                    Prediction(candidate_id, "UNKNOWN", .2))


@dataclass(frozen=True)
class HypothesisCatalog:
    summary: str
    hypotheses: tuple[TestableHypothesis, ...]
    candidates: tuple[ArcActionCandidate, ...]
    recommended_candidate_id: str
    recommendation_reason: str


HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "hypotheses": {"type": "array", "minItems": 6, "maxItems": 10, "items": {
            "type": "object", "properties": {
                "id": {"type": "integer"}, "statement": {"type": "string"},
                "focus_entity_ids": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
                "mechanism": {"type": "string"}, "success_condition": {"type": "string"},
                "falsification": {"type": "string"}, "prior": {"type": "number"},
                "test_sequence": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
                "sequence_target_state": {"type": "string"}, "stop_condition": {"type": "string"},
                "persistence_candidate_id": {"type": "string"}, "tracked_entity_id": {"type": "string"},
                "target_entity_id": {"type": "string"},
                "actuator_entity_id": {"type": "string"},
                "affected_entity_ids": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                "expected_relation_change": {"type": "string", "enum": list(RELATION_CHANGES)},
                "confirmation_outcomes": {"type": "array", "minItems": 1,
                                          "items": {"type": "string", "enum": list(OUTCOMES)}},
                "refutation_outcomes": {"type": "array", "minItems": 1,
                                        "items": {"type": "string", "enum": list(OUTCOMES)}},
                "predictions": {"type": "array", "items": {"type": "object", "properties": {
                    "candidate_id": {"type": "string"}, "outcome": {"type": "string", "enum": list(OUTCOMES)},
                    "confidence": {"type": "number"}},
                    "required": ["candidate_id", "outcome", "confidence"], "additionalProperties": False}},
            }, "required": ["id", "statement", "focus_entity_ids", "mechanism", "success_condition",
                             "falsification", "prior", "test_sequence", "sequence_target_state",
                             "stop_condition", "persistence_candidate_id", "tracked_entity_id", "target_entity_id",
                             "actuator_entity_id", "affected_entity_ids",
                             "expected_relation_change", "confirmation_outcomes", "refutation_outcomes",
                             "predictions"], "additionalProperties": False}},
        "recommended_candidate_id": {"type": "string"},
        "recommendation_reason": {"type": "string"},
    },
    "required": ["summary", "hypotheses", "recommended_candidate_id", "recommendation_reason"],
    "additionalProperties": False,
}


def _interactive_entities(observation: Any, entities: tuple[EntityFact, ...]) -> tuple[EntityFact, ...]:
    """Keep every discrete component except an unambiguous full-scene background."""
    if not observation.frame:
        return tuple()
    height, width = len(observation.frame), len(observation.frame[0])
    area = max(1, height * width)
    result = []
    for entity in entities:
        x0, y0, x1, y1 = entity.bounds
        if entity.size <= max(64, int(.10 * area)):
            result.append(entity)
    return tuple(result)


def _tracked_entities(frame: tuple[tuple[int, ...], ...], previous: tuple[EntityFact, ...]) -> tuple[EntityFact, ...]:
    """Globally match components so IDs survive reordering, motion, and modest color changes."""
    current = extract_entities(frame, 64)
    if not previous or not current:
        return current
    height, width = len(frame), len(frame[0]) if frame else 1
    diagonal = max(1.0, math.hypot(width, height))
    costs = np.zeros((len(previous), len(current)), dtype=float)
    for old_index, old in enumerate(previous):
        old_width, old_height = old.bounds[2] - old.bounds[0] + 1, old.bounds[3] - old.bounds[1] + 1
        for new_index, new in enumerate(current):
            new_width, new_height = new.bounds[2] - new.bounds[0] + 1, new.bounds[3] - new.bounds[1] + 1
            distance = math.dist(old.center, new.center) / diagonal
            size_delta = abs(old.size - new.size) / max(1, old.size, new.size)
            shape_delta = abs(old_width - new_width) / max(1, old_width, new_width) + \
                          abs(old_height - new_height) / max(1, old_height, new_height)
            # Color is the strongest identity cue when equal-size tiles cross; a
            # nearby shape may still retain its ID through a genuine color change.
            color_penalty = 0.0 if old.color == new.color else 1.25
            costs[old_index, new_index] = distance + .9 * size_delta + .25 * shape_delta + color_penalty
    old_rows, new_rows = linear_sum_assignment(costs)
    assigned = {new_index: previous[old_index].id for old_index, new_index in zip(old_rows, new_rows)
                if costs[old_index, new_index] <= 1.8}
    used = {row.id for row in previous} | set(assigned.values())
    next_id = 1 + max((int(row.id[1:]) for row in previous if row.id[1:].isdigit()), default=-1)
    tracked = []
    for index, row in enumerate(current):
        entity_id = assigned.get(index)
        if entity_id is None:
            while f"e{next_id}" in used:
                next_id += 1
            entity_id = f"e{next_id}"; used.add(entity_id); next_id += 1
        tracked.append(EntityFact(entity_id, row.color, row.size, row.bounds, row.center))
    return tuple(tracked)


def action_catalog(observation: Any, entities: tuple[EntityFact, ...]) -> tuple[ArcActionCandidate, ...]:
    """Create a compact catalog of physical tests; Luna explains rather than invents them."""
    rows: list[ArcActionCandidate] = []
    for action in observation.available_actions:
        if action in MOVES:
            rows.append(ArcActionCandidate(f"move-{action[-1]}", action, action.lower(), expected_progress=.25, risk=.1))
        elif action == "ACTION5":
            rows.append(ArcActionCandidate("interact", action, "interact at the current position", expected_progress=.35, risk=.15))
        elif action == "ACTION6":
            for entity in _interactive_entities(observation, entities):
                rows.append(ArcActionCandidate(f"click-{entity.id}", action, f"click {entity.id} at {entity.center}",
                                               x=entity.center[0], y=entity.center[1], expected_progress=.3, risk=.15))
        elif action == "ACTION7":
            interactive = _interactive_entities(observation, entities)
            for source in interactive:
                targets = sorted((target for target in interactive if target.id != source.id),
                                 key=lambda target: math.dist(source.center, target.center))[:3]
                for target in targets:
                    rows.append(ArcActionCandidate(f"drag-{source.id}-{target.id}", action,
                        f"drag {source.id} from {source.center} toward {target.id} at {target.center}",
                        x=source.center[0], y=source.center[1], next_action_type=action,
                        next_x=target.center[0], next_y=target.center[1], expected_progress=.35, risk=.2))
    return tuple(rows)


def _entity_rows(entities: tuple[EntityFact, ...]) -> list[dict[str, Any]]:
    return [asdict(entity) for entity in entities]


class InformationGainReasoner:
    """Luna supplies falsifiable world models; Python owns evidence and selection."""

    def __init__(self, caller: Any, refresh_turns: int = 4):
        self.caller = caller
        self.refresh_turns = refresh_turns
        self.cache: dict[str, HypothesisCatalog] = {}
        self.entity_cache: dict[str, tuple[EntityFact, ...]] = {}
        self.previous_entities: tuple[EntityFact, ...] = tuple()
        self.catalog: HypothesisCatalog | None = None
        self.posterior: dict[int, float] = {}
        self.evidence_count = 0
        self.pending_success: dict[str, Any] | None = None
        self.contradictions: dict[int, int] = {}
        self.retired_statements: list[str] = []
        self.refuted_hypothesis_ids: set[int] = set()
        self.validated_hypotheses: list[dict[str, Any]] = []
        self.active_sequence: list[str] = []
        self.active_hypothesis_id: int | None = None
        self.active_hypothesis: TestableHypothesis | None = None
        self.active_confidence = 0.0
        self.active_last_candidate = ""
        self.active_last_outcome = ""
        self.active_target = ""
        self.last_call_count = 0
        self.last_transfer_hypothesis_id: int | None = None
        self.architecture_prompt = ""

    @property
    def harness_version(self) -> str:
        return HARNESS_VERSION

    @property
    def input_representation(self) -> str:
        return INPUT_REPRESENTATION

    @property
    def harness_flags(self) -> dict[str, bool]:
        return {"falsifiable_hypotheses": True, "objective_outcome_classifier": True,
                "bayesian_evidence_update": True, "ordered_information_gain_plans": True,
                "shared_classical_quantum_objective": True, "stable_entity_matching": True,
                "conditional_plan_information": True, "two_no_change_reconsideration": True,
                "evidence_bound_persistence": True, "ambiguous_components_testable": True,
                "event_triggered_reasoning": True, "persistence_objective_bypass": True,
                "goal_directed_persistence": True, "cross_hypothesis_evidence_transfer": True,
                "level_scoped_refutation_ledger": True}

    @property
    def hypothesis_schema(self) -> dict[str, Any]:
        return HYPOTHESIS_SCHEMA

    def build_plan_costs(self, catalog: HypothesisCatalog, posterior: dict[int, float],
                         history: list[dict[str, Any]]):
        return build_plan_costs(catalog, posterior, history)

    def architecture_diagnostics(self) -> dict[str, Any]:
        return {}

    def reset_episode(self) -> None:
        self.entity_cache = {}; self.previous_entities = tuple(); self.catalog = None
        self.posterior = {}; self.evidence_count = 0; self.pending_success = None
        self.contradictions = {}; self.retired_statements = []; self.refuted_hypothesis_ids = set()
        self.validated_hypotheses = []
        self.last_transfer_hypothesis_id = None
        self._clear_sequence()

    def _clear_sequence(self) -> None:
        self.active_sequence = []; self.active_hypothesis_id = None; self.active_hypothesis = None
        self.active_confidence = 0.0; self.active_last_candidate = ""; self.active_last_outcome = ""
        self.active_target = ""

    @staticmethod
    def _goal_directed_persistence(hypothesis: TestableHypothesis) -> bool:
        """Require a measurable direction, not merely a repeatable board effect."""
        return hypothesis.expected_relation_change in {
            "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
            "DISTANCE_DECREASES", "DISTANCE_INCREASES",
        }

    def entities_for(self, observation: Any) -> tuple[EntityFact, ...]:
        key = json.dumps(observation.frame)
        if key not in self.entity_cache:
            self.entity_cache[key] = _tracked_entities(observation.frame, self.previous_entities)
        entities = self.entity_cache[key]
        self.previous_entities = entities
        return entities

    def _parse(self, raw: dict[str, Any], candidates: tuple[ArcActionCandidate, ...]) -> HypothesisCatalog:
        candidate_ids = {row.candidate_id for row in candidates}
        hypotheses = []
        seen_ids: set[int] = set()
        for position, row in enumerate(raw.get("hypotheses", [])):
            hypothesis_id = int(row.get("id", position + 1))
            while hypothesis_id in seen_ids:
                hypothesis_id += 100
            seen_ids.add(hypothesis_id)
            supplied = {str(item.get("candidate_id")): item for item in row.get("predictions", [])
                        if str(item.get("candidate_id")) in candidate_ids}
            predictions = tuple(Prediction(candidate.candidate_id,
                str(supplied.get(candidate.candidate_id, {}).get("outcome", "UNKNOWN")),
                min(1.0, max(0.0, float(supplied.get(candidate.candidate_id, {}).get("confidence", .2)))))
                for candidate in candidates)
            hypotheses.append(TestableHypothesis(hypothesis_id, str(row.get("statement", "Unspecified rule")),
                tuple(str(value) for value in row.get("focus_entity_ids", [])), str(row.get("mechanism", "unknown")),
                str(row.get("success_condition", "unknown")), str(row.get("falsification", "unknown")),
                max(.001, float(row.get("prior", .1))), predictions,
                tuple(candidate_id for candidate_id in row.get("test_sequence", []) if candidate_id in candidate_ids),
                str(row.get("sequence_target_state", "")), str(row.get("stop_condition", "")),
                str(row.get("persistence_candidate_id", "")) if str(row.get("persistence_candidate_id", "")) in candidate_ids else "",
                str(row.get("tracked_entity_id", "")), str(row.get("target_entity_id", "")),
                str(row.get("expected_relation_change", "ANY_CHANGE"))
                    if str(row.get("expected_relation_change", "ANY_CHANGE")) in RELATION_CHANGES else "ANY_CHANGE",
                tuple(value for value in row.get("confirmation_outcomes", []) if value in OUTCOMES),
                tuple(value for value in row.get("refutation_outcomes", []) if value in OUTCOMES),
                str(row.get("actuator_entity_id", "")),
                tuple(str(value) for value in row.get("affected_entity_ids", [])),
                str(row.get("goal_component_id", "")),
                str(row.get("goal_lineage_action", "new")),
                str(row.get("parent_goal_component_id", "")),
                str(row.get("topology_assumption", "unknown"))))
        if not hypotheses:
            hypotheses = [TestableHypothesis(1, "The next action reveals a useful rule.", tuple(), "unknown",
                "measurable progress", "no measurable change", 1.0,
                tuple(Prediction(candidate.candidate_id, "UNKNOWN", .2) for candidate in candidates),
                (candidates[0].candidate_id,), "measurable change", "no change")]
        recommended = str(raw.get("recommended_candidate_id", ""))
        if recommended not in candidate_ids:
            recommended = candidates[0].candidate_id
        return HypothesisCatalog(str(raw.get("summary", "Competing falsifiable explanations.")), tuple(hypotheses),
            candidates, recommended, str(raw.get("recommendation_reason", "Highest expected evidence.")))

    def propose(self, observation: Any, history: list[dict[str, Any]], force: bool = False) -> HypothesisCatalog:
        entities = self.entities_for(observation)
        candidates = action_catalog(observation, entities)
        if not candidates:
            raise RuntimeError("No physical actions are available for this state")
        repeated_no_change = len(history) >= 2 and history[-1].get("observed_outcome") == "NO_CHANGE" and \
            history[-2].get("observed_outcome") == "NO_CHANGE" and \
            history[-1].get("candidate_id") == history[-2].get("candidate_id")
        force = force or repeated_no_change
        same_actions = self.catalog and tuple(row.candidate_id for row in self.catalog.candidates) == tuple(row.candidate_id for row in candidates)
        if self.catalog and same_actions and not force and self.evidence_count < self.refresh_turns and not self.pending_success:
            self.last_call_count = 0
            return self.catalog
        recent = [{key: row.get(key) for key in ("turn", "candidate_id", "observed_outcome",
                                                  "physical_outcome", "levels_delta", "validated_hypothesis_id")}
                  for row in history[-8:]]
        success_text = "None"
        images = [url for url in [frame_image_data_url(observation.frame)] if url]
        if self.pending_success:
            success_text = json.dumps({key: value for key, value in self.pending_success.items()
                                       if key not in {"before_frame", "after_frame"}}, sort_keys=True)
            images = [url for url in [frame_image_data_url(self.pending_success["before_frame"]),
                                      frame_image_data_url(self.pending_success["after_frame"]),
                                      frame_image_data_url(observation.frame)] if url]
            # A new level may change the interface and make an old explanation viable again.
            self.retired_statements = []
        prompt = f"""You are proposing competing, falsifiable explanations for an unfamiliar visual game.
{self.architecture_prompt}
Do not choose actions by prose intuition alone. Return 6-10 genuinely distinct complete hypotheses. For EVERY
hypothesis, predict the discrete result of EVERY candidate action. Prediction outcomes are {', '.join(OUTCOMES)}.
Use stable entity IDs. Do not dismiss a small or edge-attached component as decoration: generate a test when that
assumption matters. A success transition does not prove the active hypothesis: compare its prediction with the
before/after state and revise the rule. Prefer hypotheses that disagree on available tests. Give each hypothesis a
short initial test_sequence. Distinguish the actuator_entity_id that is clicked from affected_entity_ids that change;
do not claim the control itself moved merely because clicking it transformed the board. Use GROUP_TRANSFORM when
several entities change position together. When an action should persist, set persistence_candidate_id,
tracked_entity_id, target_entity_id, and the expected_relation_change. Define confirmation_outcomes and
refutation_outcomes for that specific hypothesis. Confirmation supports a continuing procedure; it does not finish
it unless the game is terminal. There is no arbitrary repetition limit: persistence is valid only while its predicted
outcome and entity relationship continue to occur. A NO_CHANGE, reversal, or progress event is neither confirmation
nor refutation unless the hypothesis says it is. State the target state and evidence-based stop condition.

STATE: {observation.state}; LEVELS: {observation.levels_completed}/{observation.win_levels}
LOSSLESS GRID:\n{frame_context(observation.frame)}
ENTITIES: {json.dumps(_entity_rows(entities), sort_keys=True)}
CANDIDATES: {json.dumps([asdict(row) for row in candidates], sort_keys=True)}
RECENT OBJECTIVE RESULTS: {json.dumps(recent, sort_keys=True)}
SUCCESS TRANSITION TO AUDIT: {success_text}
REJECTED IN THIS LEVEL (do not restate without new evidence): {json.dumps(self.retired_statements[-8:])}

Recommend the single action you would execute as the LLM-only baseline. Python will independently maintain
probabilities and classical/quantum QAHS will optimize an ordered three-test plan, execute its first action, observe,
then replan. Return predictions for every listed candidate_id exactly once per hypothesis."""
        cache_key = json.dumps({"frame": observation.frame, "candidates": [row.candidate_id for row in candidates],
                                "recent": recent, "success": success_text,
                                "architecture": self.architecture_prompt}, sort_keys=True)
        if cache_key in self.cache:
            catalog = self.cache[cache_key]; self.last_call_count = 0
        else:
            raw = self.caller.call_json(prompt, self.hypothesis_schema, "hypothesis_catalog", images)
            catalog = self._parse(raw, candidates); self.cache[cache_key] = catalog; self.last_call_count = 1
        self.catalog = catalog
        total = sum(row.prior for row in catalog.hypotheses)
        self.posterior = {row.id: row.prior / total for row in catalog.hypotheses}
        self.contradictions = {row.id: 0 for row in catalog.hypotheses}
        self.refuted_hypothesis_ids = set()
        self.evidence_count = 0; self.pending_success = None
        return catalog

    def update(self, candidate_id: str, observed: str,
               supporting_outcomes: tuple[str, ...] = tuple()) -> tuple[dict[int, float], dict[int, float]]:
        if not self.catalog:
            return {}, {}
        before = dict(self.posterior)
        weights: dict[int, float] = {}
        for hypothesis in self.catalog.hypotheses:
            prediction = hypothesis.prediction_for(candidate_id)
            if prediction.outcome == "UNKNOWN":
                likelihood = .35
            elif prediction.outcome in {observed, *supporting_outcomes}:
                likelihood = .55 + .4 * prediction.confidence
            else:
                likelihood = max(.03, .2 - .15 * prediction.confidence)
                if prediction.confidence >= .6:
                    self.contradictions[hypothesis.id] = self.contradictions.get(hypothesis.id, 0) + 1
            weights[hypothesis.id] = before.get(hypothesis.id, 0) * likelihood
        total = sum(weights.values())
        self.posterior = ({key: value / total for key, value in weights.items()} if total else
                          {key: 1 / len(weights) for key in weights})
        self.evidence_count += 1
        retired = [row for row in self.catalog.hypotheses if self.posterior.get(row.id, 0) < .025 and
                   self.contradictions.get(row.id, 0) >= 2]
        if retired and len(retired) < len(self.catalog.hypotheses):
            for row in retired:
                self.posterior[row.id] = 0.0
                if row.statement not in self.retired_statements:
                    self.retired_statements.append(row.statement)
            surviving = sum(self.posterior.values())
            self.posterior = {key: value / surviving for key, value in self.posterior.items()}
        return before, dict(self.posterior)

    def queued_action(self, candidates: tuple[ArcActionCandidate, ...]) -> str | None:
        valid = {row.candidate_id for row in candidates}
        while self.active_sequence and self.active_sequence[0] not in valid:
            self.active_sequence.pop(0)
        return self.active_sequence.pop(0) if self.active_sequence else None

    def start_sequence(self, catalog: HypothesisCatalog, posterior: dict[int, float], candidate_id: str) -> None:
        matches = [row for row in catalog.hypotheses if row.test_sequence and row.test_sequence[0] == candidate_id]
        if not matches:
            return
        chosen = max(matches, key=lambda row: posterior.get(row.id, 0))
        self.active_hypothesis_id = chosen.id
        self.active_hypothesis = chosen
        self.active_confidence = posterior.get(chosen.id, 0.0)
        self.active_sequence = list(chosen.test_sequence[1:])
        self.active_target = chosen.sequence_target_state

    @staticmethod
    def _relation_holds(hypothesis: TestableHypothesis, before: tuple[EntityFact, ...],
                        after: tuple[EntityFact, ...], observed: str) -> bool:
        relation = hypothesis.expected_relation_change
        if relation == "GROUP_TRANSFORM":
            return observed == "GROUP_TRANSFORM"
        if relation == "ANY_CHANGE":
            return observed != "NO_CHANGE"
        if relation == "NO_CHANGE":
            return observed == "NO_CHANGE"
        old = {row.id: row for row in before}; new = {row.id: row for row in after}
        tracked_before = old.get(hypothesis.tracked_entity_id)
        tracked_after = new.get(hypothesis.tracked_entity_id)
        if not tracked_before or not tracked_after:
            return False
        dx = tracked_after.center[0] - tracked_before.center[0]
        dy = tracked_after.center[1] - tracked_before.center[1]
        if relation == "MOVE_LEFT": return dx < 0
        if relation == "MOVE_RIGHT": return dx > 0
        if relation == "MOVE_UP": return dy < 0
        if relation == "MOVE_DOWN": return dy > 0
        target_before = old.get(hypothesis.target_entity_id)
        target_after = new.get(hypothesis.target_entity_id)
        if not target_before or not target_after:
            return False
        prior_distance = math.dist(tracked_before.center, target_before.center)
        next_distance = math.dist(tracked_after.center, target_after.center)
        return next_distance < prior_distance if relation == "DISTANCE_DECREASES" else next_distance > prior_distance

    def reconsider_sequence(self, catalog: HypothesisCatalog, posterior: dict[int, float]) -> bool:
        """Allow a better-supported explanation to inherit an active physical procedure."""
        active = self.active_hypothesis
        if not active or not active.persistence_candidate_id:
            return False
        alternatives = [row for row in catalog.hypotheses
            if row != active and row.persistence_candidate_id == active.persistence_candidate_id and
            (not self.active_last_outcome or row.prediction_for(active.persistence_candidate_id).outcome
             in {self.active_last_outcome, "UNKNOWN"})]
        if not alternatives:
            return False
        replacement = max(alternatives, key=lambda row: posterior.get(row.id, 0.0))
        replacement_confidence = posterior.get(replacement.id, 0.0)
        if replacement_confidence < self.active_confidence + .10:
            return False
        self.active_hypothesis = replacement; self.active_hypothesis_id = replacement.id
        self.active_confidence = replacement_confidence; self.active_target = replacement.sequence_target_state
        if self.active_last_outcome in replacement.confirmation_outcomes:
            self.active_sequence = [replacement.persistence_candidate_id]
        else:
            remaining = list(replacement.test_sequence)
            if remaining and remaining[0] == self.active_last_candidate:
                remaining.pop(0)
            self.active_sequence = remaining or [replacement.persistence_candidate_id]
        return True

    def assess_sequence(self, candidate_id: str, observed: str, physical_outcome: str,
                        before: tuple[EntityFact, ...], after: tuple[EntityFact, ...],
                        completed_level: int) -> tuple[str, dict[str, Any] | None]:
        hypothesis = self.active_hypothesis
        if not hypothesis:
            return "inactive", None
        self.last_transfer_hypothesis_id = None
        self.active_last_candidate = candidate_id; self.active_last_outcome = observed
        evidence = {observed, physical_outcome}
        prediction = hypothesis.prediction_for(candidate_id)
        predicted = prediction.outcome == "UNKNOWN" or prediction.outcome in evidence
        explicitly_predicted = prediction.outcome != "UNKNOWN" and prediction.outcome in evidence
        relation_holds = self._relation_holds(hypothesis, before, after, physical_outcome)
        if observed in {"PROGRESS", "TERMINAL"} and explicitly_predicted and relation_holds:
            validation = {"hypothesis_id": hypothesis.id, "statement": hypothesis.statement,
                          "validated_for_level": completed_level, "candidate_id": candidate_id,
                          "predicted_outcome": prediction.outcome, "physical_outcome": physical_outcome,
                          "game_outcome": observed}
            self.validated_hypotheses.append(validation)
            self._clear_sequence()
            return ("validated_terminal" if observed == "TERMINAL" else "validated_level"), validation
        if observed in {"PROGRESS", "TERMINAL"}:
            self._clear_sequence()
            return "progress_unattributed", None
        # Explicit game-level failure evidence must not be masked by a matching
        # physical effect. An action can transform the board exactly as predicted
        # and still disprove the procedure by causing FAIL.
        contradicted = bool(evidence.intersection(hypothesis.refutation_outcomes))
        if contradicted or not predicted or not relation_holds:
            # A direct contradiction retires this explanation for the current
            # catalog. It cannot be resurrected by a later transfer unless Luna
            # supplies a genuinely revised catalog after new evidence.
            self.refuted_hypothesis_ids.add(hypothesis.id)
            self.posterior[hypothesis.id] = 0.0
            alternatives = [row for row in (self.catalog.hypotheses if self.catalog else tuple())
                if row.id not in self.refuted_hypothesis_ids and
                row.prediction_for(candidate_id).outcome != "UNKNOWN" and
                row.prediction_for(candidate_id).outcome in evidence and
                not evidence.intersection(row.refutation_outcomes) and
                self._relation_holds(row, before, after, physical_outcome)]
            if alternatives:
                replacement = max(alternatives, key=lambda row: self.posterior.get(row.id, 0.0))
                self.active_hypothesis = replacement; self.active_hypothesis_id = replacement.id
                self.active_confidence = self.posterior.get(replacement.id, 0.0)
                self.active_target = replacement.sequence_target_state
                remaining = list(replacement.test_sequence)
                if candidate_id in remaining:
                    remaining = remaining[remaining.index(candidate_id) + 1:]
                if (self._goal_directed_persistence(replacement) and
                        replacement.persistence_candidate_id == candidate_id):
                    remaining = [candidate_id]
                self.active_sequence = remaining
                self.last_transfer_hypothesis_id = replacement.id
                if remaining:
                    return "reframed_continuing", None
                self.evidence_count = max(self.evidence_count, self.refresh_turns)
                self._clear_sequence()
                return "reframed_supported", None
            self.evidence_count = max(self.evidence_count, self.refresh_turns)
            self._clear_sequence(); return "refuted", None
        confirmed = bool(evidence.intersection(hypothesis.confirmation_outcomes))
        if observed == "TERMINAL":
            self._clear_sequence(); return "confirmed", None
        self.active_confidence = self.posterior.get(hypothesis.id, self.active_confidence)
        goal_directed = self._goal_directed_persistence(hypothesis)
        if confirmed and hypothesis.persistence_candidate_id and goal_directed:
            # Only a measurable directional relationship can justify indefinite
            # persistence. A generic transformation validates an actuator but not
            # whether repeatedly using it approaches the game's objective.
            self.active_sequence = [hypothesis.persistence_candidate_id]
        elif not self.active_sequence and hypothesis.persistence_candidate_id and goal_directed:
            self.active_sequence.append(hypothesis.persistence_candidate_id)
        if self.active_sequence:
            return ("confirmed_continuing" if confirmed else "continuing"), None
        if confirmed:
            if hypothesis.persistence_candidate_id and not goal_directed:
                self.evidence_count = max(self.evidence_count, self.refresh_turns)
            self._clear_sequence(); return "confirmed", None
        return "supported", None


def classify_outcome(before: Any, after: Any, before_entities: tuple[EntityFact, ...],
                     after_entities: tuple[EntityFact, ...], history: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    transition = assess_transition(before, after, history)
    old = {row.id: row for row in before_entities}; new = {row.id: row for row in after_entities}
    moved = [key for key in old if key in new and old[key].center != new[key].center and
             old[key].size == new[key].size]
    size = max(1, len(before.frame) * (len(before.frame[0]) if before.frame else 1))
    if int(transition["changed_cells"]) == 0:
        physical_outcome = "NO_CHANGE"
    elif len(moved) >= 3:
        physical_outcome = "GROUP_TRANSFORM"
    elif moved:
        physical_outcome = "MOVE"
    elif int(transition["changed_cells"]) / size >= .15:
        physical_outcome = "BROAD_CHANGE"
    else:
        physical_outcome = "LOCAL_CHANGE"
    if after.success:
        outcome = "TERMINAL"
    elif after.state.upper() == "GAME_OVER":
        outcome = "FAIL"
    elif int(transition["levels_delta"]) > 0:
        outcome = "PROGRESS"
    else:
        outcome = physical_outcome
    transition["moved_entity_ids"] = moved
    transition["physical_outcome"] = physical_outcome
    transition["game_outcome"] = outcome if outcome in {"PROGRESS", "TERMINAL", "FAIL"} else None
    transition["classified_outcome"] = outcome
    return outcome, transition


class PlanEncoding:
    """Encode up to three distinct tests; decode stays compatible with both search backends."""
    def __init__(self, candidates: tuple[ArcActionCandidate, ...], depth: int = 3):
        self.actions = tuple(row.candidate_id for row in candidates)
        self.depth = min(depth, len(self.actions))
        self.plans = tuple(permutations(self.actions, self.depth))
        self.total_bits = max(1, math.ceil(math.log2(max(2, len(self.plans)))))
        self.state_count = 1 << self.total_bits

    def plan(self, index: int) -> tuple[str, ...] | None:
        return self.plans[index] if 0 <= index < len(self.plans) else None

    def decode(self, index: int) -> tuple[int, str] | None:
        plan = self.plan(index)
        return None if plan is None else (index, plan[0])


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


OUTCOME_VALUE = {"NO_CHANGE": .12, "LOCAL_CHANGE": .55, "BROAD_CHANGE": .65, "MOVE": .75,
                 "GROUP_TRANSFORM": .85,
                 "PROGRESS": 1.35, "TERMINAL": 1.6, "FAIL": .05, "UNKNOWN": .25}


def expected_information_gain(catalog: HypothesisCatalog, posterior: dict[int, float], candidate_id: str) -> float:
    """Reward disagreement most when one plausible branch represents real progress."""
    score = 0.0
    for left_index, left in enumerate(catalog.hypotheses):
        left_outcome = left.prediction_for(candidate_id).outcome
        for right in catalog.hypotheses[left_index + 1:]:
            right_outcome = right.prediction_for(candidate_id).outcome
            if left_outcome == right_outcome:
                continue
            relevance = max(OUTCOME_VALUE.get(left_outcome, .25), OUTCOME_VALUE.get(right_outcome, .25))
            score += 2 * posterior.get(left.id, 0) * posterior.get(right.id, 0) * relevance
    return score


def _prediction_partition_entropy(catalog: HypothesisCatalog, posterior: dict[int, float],
                                  plan: tuple[str, ...]) -> float:
    groups: dict[tuple[str, ...], float] = {}
    for hypothesis in catalog.hypotheses:
        signature = tuple(hypothesis.prediction_for(candidate_id).outcome for candidate_id in plan)
        groups[signature] = groups.get(signature, 0) + posterior.get(hypothesis.id, 0)
    return _entropy(list(groups.values()))


def _conditional_plan_information(catalog: HypothesisCatalog, posterior: dict[int, float],
                                  plan: tuple[str, ...]) -> tuple[float, list[float]]:
    """Value each later test only for distinctions not already made by earlier tests."""
    increments = []
    previous_entropy = 0.0
    for depth in range(1, len(plan) + 1):
        entropy = _prediction_partition_entropy(catalog, posterior, plan[:depth])
        increments.append(max(0.0, entropy - previous_entropy))
        previous_entropy = entropy
    return sum(discount * value for discount, value in zip((1.0, .65, .4), increments)), increments


def build_plan_costs(catalog: HypothesisCatalog, posterior: dict[int, float], history: list[dict[str, Any]]):
    encoding = PlanEncoding(catalog.candidates)
    costs = np.full(encoding.state_count, 1000.0)
    valid = np.zeros(encoding.state_count, dtype=bool)
    recent = [str(row.get("candidate_id")) for row in history[-4:]]
    blocked = recent[-1] if len(catalog.candidates) > 1 and len(history) >= 2 and history[-1].get("observed_outcome") == "NO_CHANGE" and \
        history[-2].get("observed_outcome") == "NO_CHANGE" and recent[-1] == recent[-2] else None
    info = {row.candidate_id: expected_information_gain(catalog, posterior, row.candidate_id) for row in catalog.candidates}
    progress = {}
    for candidate in catalog.candidates:
        progress[candidate.candidate_id] = sum(posterior.get(hypothesis.id, 0) *
            ({"PROGRESS": .75, "TERMINAL": 1.0}.get(hypothesis.prediction_for(candidate.candidate_id).outcome, 0.0))
            for hypothesis in catalog.hypotheses)
    candidate_map = {row.candidate_id: row for row in catalog.candidates}
    for index in range(encoding.state_count):
        plan = encoding.plan(index)
        if plan is None:
            continue
        if blocked and plan[0] == blocked:
            continue
        valid[index] = True
        _conditional_information, increments = _conditional_plan_information(catalog, posterior, plan)
        # Information distinguishes hypotheses; the relevance term favors distinctions
        # involving progress over equally predictable inactivity.
        conditional_relevance = sum(discount * increment * (1 + .75 * info[action])
                                    for discount, increment, action in zip((1.0, .65, .4), increments, plan))
        utility = conditional_relevance + sum(
            discount * .7 * progress[action] for discount, action in zip((1.0, .65, .4), plan))
        utility -= .18 * sum(recent.count(action) for action in plan)
        utility -= .12 * sum(candidate_map[action].risk for action in plan)
        costs[index] = -utility
    diagnostics = {"cost_model": "conditional-progress-information-plan-v2", "plan_depth": encoding.depth,
                   "valid_plans": int(valid.sum()), "information_gain": info,
                   "expected_progress": progress, "posterior_entropy": _entropy(list(posterior.values()))}
    diagnostics["blocked_after_repeated_no_change"] = blocked
    return encoding, costs, valid, diagnostics


def _matrix(catalog: HypothesisCatalog, posterior: dict[int, float]) -> list[dict[str, Any]]:
    return [{"hypothesis_id": row.id, "statement": row.statement,
             "focus_entity_ids": list(row.focus_entity_ids), "probability": posterior.get(row.id, 0),
             "test_sequence": list(row.test_sequence), "sequence_target_state": row.sequence_target_state,
             "stop_condition": row.stop_condition,
             "persistence_candidate_id": row.persistence_candidate_id,
             "tracked_entity_id": row.tracked_entity_id, "target_entity_id": row.target_entity_id,
             "actuator_entity_id": row.actuator_entity_id,
             "affected_entity_ids": list(row.affected_entity_ids),
             "expected_relation_change": row.expected_relation_change,
             "confirmation_outcomes": list(row.confirmation_outcomes),
             "refutation_outcomes": list(row.refutation_outcomes),
             "goal_component_id": row.goal_component_id,
             "goal_lineage_action": row.goal_lineage_action,
             "parent_goal_component_id": row.parent_goal_component_id,
             "topology_assumption": row.topology_assumption,
             "predictions": [{"candidate_id": item.candidate_id, "outcome": item.outcome,
                              "confidence": item.confidence} for item in row.predictions]}
            for row in catalog.hypotheses]


def _arc_hypotheses(catalog: HypothesisCatalog, posterior: dict[int, float]) -> tuple[ArcHypothesis, ...]:
    return tuple(ArcHypothesis(row.id, row.statement, posterior.get(row.id, 0),
        {item.candidate_id: item.outcome for item in row.predictions}, object_role=", ".join(row.focus_entity_ids) or "scene",
        rule=row.mechanism, goal_condition=row.success_condition, falsification=row.falsification,
        role="testable_world_model", grounded_goal=ArcGroundedGoal(target_component=", ".join(row.focus_entity_ids) or "scene",
            desired_state=row.success_condition, partial_success="predicted observable outcome")) for row in catalog.hypotheses)


def run_information_gain_episode(environment: ArcEnvironment, reasoner: InformationGainReasoner, condition: str,
                                 source: str, seed: int, max_steps: int, batch_id: str | None,
                                 progress_callback: Callable[[dict[str, Any]], None] | None, condition_index: int) -> ArcEpisodeTrace:
    started = perf_counter(); emit = progress_callback or (lambda _event: None)
    trace = ArcEpisodeTrace(condition, source, environment.game_id, seed, action_budget=max_steps,
        model=reasoner.caller.model if source == "arc" else "Deterministic local reasoner",
        reasoning_effort=getattr(reasoner.caller, "reasoning_effort", None), batch_id=batch_id,
        harness_version=reasoner.harness_version, input_representation=reasoner.input_representation,
        harness_flags=reasoner.harness_flags)
    history: list[dict[str, Any]] = []
    try:
        reasoner.reset_episode(); observation = environment.reset()
        emit({"phase": "condition_started", "condition": condition, "condition_index": condition_index,
              "condition_total": 3, "turn": 0, "max_steps": max_steps})
        for turn_number in range(1, max_steps + 1):
            turn_started = perf_counter(); before = observation
            emit({"phase": "reasoning", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps})
            stage = perf_counter()
            before_entities = reasoner.entities_for(observation)
            current_candidates = action_catalog(observation, before_entities)
            queued_candidate = reasoner.queued_action(current_candidates) if reasoner.catalog else None
            # A procedure is already an evidence-bound decision. Rebuilding the
            # hypothesis catalog and every three-action plan cannot change this
            # turn's action, so execute it directly. assess_sequence below will
            # clear persistence on failure, unexpected effects, or a broken
            # entity relationship; progress also forces a fresh catalog.
            if queued_candidate and reasoner.catalog:
                catalog = reasoner.catalog
                posterior_before = dict(reasoner.posterior)
                proposal_seconds = perf_counter() - stage; model_calls = 0
                objective_seconds = 0.0
                information = {row.candidate_id: expected_information_gain(
                    catalog, posterior_before, row.candidate_id) for row in current_candidates}
                progress = {row.candidate_id: sum(posterior_before.get(hypothesis.id, 0) *
                    ({"PROGRESS": .75, "TERMINAL": 1.0}.get(
                        hypothesis.prediction_for(row.candidate_id).outcome, 0.0))
                    for hypothesis in catalog.hypotheses) for row in current_candidates}
                objective = {"information_gain": information, "expected_progress": progress,
                    "blocked_after_repeated_no_change": None, "cost_model": "persistence-bypass",
                    "plan_depth": 1, "valid_plans": 1}
                scores = [ActionScore(row.candidate_id, information[row.candidate_id],
                                      progress[row.candidate_id],
                                      information[row.candidate_id] + progress[row.candidate_id])
                          for row in current_candidates]
                candidate_id = queued_candidate; selected_plan = [candidate_id]
                diagnostics = {"backend": "hypothesis_procedure", "quantum_used": False,
                    "selected_cost": 0.0, "regret": 0.0,
                    "active_hypothesis_id": reasoner.active_hypothesis_id,
                    "procedure_target_state": reasoner.active_target, "procedure_continued": True,
                    "procedure_reframed": False, "reasoning_reused": True}
                selection_seconds = 0.0
                selected = next(row for row in current_candidates if row.candidate_id == candidate_id)
                proposal_candidates = current_candidates
                depth = min(3, len(current_candidates))
                plan_count = math.perm(len(current_candidates), depth) if depth else 1
                decision_qubits = max(1, math.ceil(math.log2(max(2, plan_count))))
            else:
                stage = perf_counter(); catalog = reasoner.propose(observation, history)
                proposal_seconds = perf_counter() - stage; model_calls = reasoner.last_call_count
                before_entities = reasoner.entities_for(observation)
                posterior_before = dict(reasoner.posterior)
                procedure_reframed = reasoner.reconsider_sequence(catalog, posterior_before)
                stage = perf_counter(); encoding, costs, valid, objective = reasoner.build_plan_costs(
                    catalog, posterior_before, history)
                objective_seconds = perf_counter() - stage
                scores = [ActionScore(row.candidate_id, objective["information_gain"][row.candidate_id],
                                      objective["expected_progress"][row.candidate_id],
                                      objective["information_gain"][row.candidate_id] + objective["expected_progress"][row.candidate_id])
                          for row in catalog.candidates]
                stage = perf_counter()
                queued_candidate = reasoner.queued_action(catalog.candidates)
                if queued_candidate:
                    candidate_id = queued_candidate; selected_plan = [candidate_id]
                    plan_indices = [index for index in np.flatnonzero(valid) if encoding.plan(int(index))[0] == candidate_id]
                    selected_index = int(plan_indices[0]) if plan_indices else int(np.flatnonzero(valid)[0])
                    diagnostics = {"backend": "hypothesis_procedure", "quantum_used": False,
                        "selected_cost": float(costs[selected_index]), "regret": float(costs[selected_index] - min(costs[valid])),
                        "active_hypothesis_id": reasoner.active_hypothesis_id,
                        "procedure_target_state": reasoner.active_target, "procedure_continued": True,
                        "procedure_reframed": procedure_reframed, "reasoning_reused": False}
                elif condition == "llm_only":
                    blocked = objective.get("blocked_after_repeated_no_change")
                    candidate_id = catalog.recommended_candidate_id
                    reconsidered = False
                    if blocked == candidate_id:
                        alternatives = [row for row in catalog.candidates if row.candidate_id != blocked]
                        if alternatives:
                            candidate_id = max(alternatives, key=lambda row:
                                objective["information_gain"][row.candidate_id] +
                                objective["expected_progress"][row.candidate_id]).candidate_id
                            reconsidered = True
                    selected_plan = [candidate_id]
                    plan_indices = [index for index in np.flatnonzero(valid) if encoding.plan(int(index))[0] == candidate_id]
                    selected_index = int(plan_indices[0]) if plan_indices else int(np.flatnonzero(valid)[0])
                    diagnostics = {"backend": "llm_recommendation", "quantum_used": False,
                        "selected_cost": float(costs[selected_index]), "regret": float(costs[selected_index] - min(costs[valid]))}
                    diagnostics["forced_reconsideration"] = reconsidered
                else:
                    backend = ClassicalSearch() if condition == "qahs_classical" else QAOASearch(
                        depth=2, restarts=3, maxiter=80, shots=512, seed=seed + turn_number)
                    result = backend.select(encoding, scores, costs, valid)
                    candidate_id = str(result.action); selected_index = result.encoded_index
                    selected_plan = list(encoding.plan(selected_index) or (candidate_id,))
                    diagnostics = {**result.diagnostics, "backend": result.backend, "quantum_used": condition == "qahs_qaoa"}
                if not queued_candidate:
                    reasoner.start_sequence(catalog, posterior_before, candidate_id)
                selection_seconds = perf_counter() - stage
                selected = next(row for row in catalog.candidates if row.candidate_id == candidate_id)
                proposal_candidates = catalog.candidates
                decision_qubits = encoding.total_bits
            proposal = ArcProposal(catalog.summary, _arc_hypotheses(catalog, posterior_before), proposal_candidates,
                                   catalog.recommended_candidate_id, catalog.recommendation_reason)
            diagnostics.update({**objective, "selected_plan": selected_plan,
                "prediction_matrix": _matrix(catalog, posterior_before),
                "hypothesis_probabilities_before": {str(k): v for k, v in posterior_before.items()},
                "entity_catalog": _entity_rows(before_entities),
                "model_recommended_candidate_id": catalog.recommended_candidate_id,
                "selector_candidate_id": candidate_id,
                "override_opportunity": candidate_id != catalog.recommended_candidate_id,
                "override_permitted": condition != "llm_only" and candidate_id != catalog.recommended_candidate_id,
                "decision_qubits": decision_qubits})
            diagnostics.update(reasoner.architecture_diagnostics())
            emit({"phase": "environment", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps})
            stage = perf_counter(); observation = environment.step(selected); environment_seconds = perf_counter() - stage
            after_entities = reasoner.entities_for(observation)
            observed, transition = classify_outcome(before, observation, before_entities, after_entities, history)
            physical_outcome = str(transition["physical_outcome"])
            procedure_hypothesis = reasoner.active_hypothesis
            beliefs_before, beliefs_after = reasoner.update(candidate_id, observed, (physical_outcome,))
            procedure_status, validation = reasoner.assess_sequence(
                candidate_id, observed, physical_outcome, before_entities, after_entities,
                int(observation.levels_completed))
            diagnostics.update({"observed_outcome": observed,
                "physical_outcome": physical_outcome,
                "procedure_evidence_status": procedure_status,
                "evidence_transferred_to_hypothesis_id": reasoner.last_transfer_hypothesis_id,
                "validated_for_level": validation,
                "hypothesis_probabilities_after": {str(k): v for k, v in beliefs_after.items()},
                "retired_hypotheses": list(reasoner.retired_statements),
                "refuted_hypothesis_ids": sorted(reasoner.refuted_hypothesis_ids),
                "after_entity_catalog": _entity_rows(after_entities), "changed_cells": transition["changed_cells"],
                "levels_delta": transition["levels_delta"],
                "moved_entity_ids": transition.get("moved_entity_ids", [])})
            best = max(catalog.hypotheses, key=lambda row: beliefs_after.get(row.id, 0))
            active = max(catalog.hypotheses, key=lambda row: beliefs_before.get(row.id, 0))
            predicted = active.prediction_for(candidate_id).outcome
            reflection = ArcReflection(f"Observed {observed}; updated competing explanations objectively.", observed,
                tuple(ArcBelief(row.id, row.statement, beliefs_after.get(row.id, 0),
                    "refuted" if {observed, physical_outcome}.intersection(row.refutation_outcomes) else
                    "supported" if row.prediction_for(candidate_id).outcome in {observed, physical_outcome} else "weakened",
                    f"Predicted {row.prediction_for(candidate_id).outcome}; physical effect {physical_outcome}; game effect {observed}.", role="testable_world_model")
                    for row in catalog.hypotheses), f"Test an action that best separates the remaining explanations; current leader: {best.id}.")
            if int(transition["levels_delta"]) > 0 or observation.success:
                credited = procedure_hypothesis or active
                credited_prediction = credited.prediction_for(candidate_id).outcome
                reasoner.pending_success = {"selected_hypothesis": credited.statement,
                    "selected_hypothesis_id": credited.id, "predicted_outcome": credited_prediction,
                    "physical_outcome": physical_outcome, "observed_outcome": observed,
                    "validated_for_level": validation, "selected_action": candidate_id,
                    "before_frame": before.frame, "after_frame": observation.frame}
                reasoner.evidence_count = reasoner.refresh_turns
            history.append({"turn": turn_number, "candidate_id": candidate_id, "observed_outcome": observed,
                            "physical_outcome": physical_outcome,
                            "predicted_outcome": predicted, "levels_delta": transition["levels_delta"],
                            "validated_hypothesis_id": validation.get("hypothesis_id") if validation else None,
                            "selected_plan": selected_plan, "state": observation.state})
            trace.turns.append(ArcTurn(turn_number, before, proposal, condition, selected,
                [asdict(row) for row in scores], diagnostics, observation, reflection, model_calls,
                {"proposal_seconds": proposal_seconds, "objective_seconds": objective_seconds,
                 "selection_seconds": selection_seconds, "environment_seconds": environment_seconds,
                 "reflection_seconds": 0.0, "turn_seconds": perf_counter() - turn_started}))
            emit({"phase": "turn_completed", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps,
                  "state": observation.state, "levels": observation.levels_completed})
            if observation.terminal:
                break
        trace.success = observation.success
    except Exception as error:
        trace.error = f"{type(error).__name__}: run failed; internal details omitted"
    trace.wall_seconds = perf_counter() - started
    trace.intervention_summary = {"opportunities": sum(bool(t.search_diagnostics.get("override_opportunity")) for t in trace.turns),
        "permitted": sum(bool(t.search_diagnostics.get("override_permitted")) for t in trace.turns),
        "blocked": 0, "helpful": 0, "neutral": 0, "harmful": 0}
    emit({"phase": "condition_completed", "condition": condition, "condition_index": condition_index,
          "condition_total": 3, "turn": len(trace.turns), "max_steps": max_steps,
          "success": trace.success, "error": bool(trace.error)})
    return trace


def run_information_gain_comparison(environment_factory: Callable[[], ArcEnvironment], reasoner: InformationGainReasoner,
                                    source: str, game_id: str, seed: int, max_steps: int,
                                    batch_id: str | None, progress_callback=None) -> ArcComparisonTrace:
    episodes = [run_information_gain_episode(environment_factory(), reasoner, condition, source, seed, max_steps,
                                               batch_id, progress_callback, index)
                for index, condition in enumerate(("llm_only", "qahs_classical", "qahs_qaoa"))]
    trace = ArcComparisonTrace(source, game_id, seed, *episodes, action_budget=max_steps,
        model=reasoner.caller.model if source == "arc" else "Deterministic local reasoner",
        reasoning_effort=getattr(reasoner.caller, "reasoning_effort", None), batch_id=batch_id,
        harness_version=reasoner.harness_version, input_representation=reasoner.input_representation,
        harness_flags=reasoner.harness_flags)
    return trace
