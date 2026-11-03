"""V12: compositional, uncertain world models above the measured V11 state.

The professor's factorized-world philosophy is used without importing a fixed
game ontology or a privileged simulator.  Luna proposes falsifiable complete
models; Python decomposes them into independently updateable dimensions, learns
only observed local effects, and gives every selector the same objective.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .information_gain import (
    HYPOTHESIS_SCHEMA, HypothesisCatalog, InformationGainReasoner,
    build_plan_costs as build_v11_plan_costs,
    run_information_gain_comparison,
)


HARNESS_VERSION = "v12-structured-world-models"
INPUT_REPRESENTATION = "factorized-evidence-channels+explicit-goal-lineage+observable-discrimination-v3"
FAMILIES = ("controller", "goal", "dynamics", "topology", "interaction", "sequence")


def _canonical(value: str, fallback: str = "unknown") -> str:
    value = re.sub(r"\be\d+\b", "<entity>", str(value).strip().lower())
    value = re.sub(r"\d+", "<n>", value)
    value = re.sub(r"\s+", " ", value)
    return value[:160] or fallback


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in distribution.values() if value > 0)


@dataclass(frozen=True)
class WorldComponent:
    family: str
    key: str
    description: str
    hypothesis_ids: tuple[int, ...]


class FactorizedWorldPosterior:
    """Marginalize complete Luna hypotheses into six revisable dimensions."""

    def __init__(self) -> None:
        self.components: dict[str, tuple[WorldComponent, ...]] = {}
        self.hypothesis_components: dict[int, dict[str, str]] = {}
        self.marginals: dict[str, dict[str, float]] = {}
        self.role_bindings: dict[str, dict[str, float]] = {}

    @staticmethod
    def _parts(hypothesis: Any) -> dict[str, str]:
        focus = hypothesis.focus_entity_ids[0] if hypothesis.focus_entity_ids else "unknown"
        return {
            "controller": _canonical(hypothesis.actuator_entity_id or focus),
            # Goal identity is declared by Luna. Python never merges goals by
            # interpreting their prose; missing IDs remain explicitly unknown.
            "goal": hypothesis.goal_component_id or "unknown",
            "dynamics": _canonical(f"{hypothesis.expected_relation_change}: {hypothesis.mechanism}"),
            "topology": _canonical(hypothesis.topology_assumption),
            "interaction": _canonical(hypothesis.mechanism),
            "sequence": _canonical(hypothesis.sequence_target_state or hypothesis.stop_condition),
        }

    def rebuild(self, catalog: HypothesisCatalog, posterior: dict[int, float], preserve: bool = False) -> None:
        previous = self.marginals if preserve else {}
        grouped: dict[str, dict[str, list[int]]] = {family: defaultdict(list) for family in FAMILIES}
        self.hypothesis_components = {}
        roles: dict[str, Counter[str]] = defaultdict(Counter)
        for hypothesis in catalog.hypotheses:
            parts = self._parts(hypothesis)
            self.hypothesis_components[hypothesis.id] = parts
            for family, key in parts.items():
                grouped[family][key].append(hypothesis.id)
            for entity_id in hypothesis.focus_entity_ids:
                roles[entity_id]["focus"] += posterior.get(hypothesis.id, 0.0)
            if hypothesis.actuator_entity_id:
                roles[hypothesis.actuator_entity_id]["controller"] += posterior.get(hypothesis.id, 0.0)
            if hypothesis.target_entity_id:
                roles[hypothesis.target_entity_id]["target"] += posterior.get(hypothesis.id, 0.0)
            for entity_id in hypothesis.affected_entity_ids:
                roles[entity_id]["affected"] += posterior.get(hypothesis.id, 0.0)

        self.components = {}
        self.marginals = {}
        for family in FAMILIES:
            rows = [WorldComponent(family, key, key, tuple(ids)) for key, ids in grouped[family].items()]
            if "unknown" not in grouped[family]:
                rows.append(WorldComponent(family, "unknown", "unknown / extendable", tuple()))
            weights = {row.key: (previous.get(family, {}).get(row.key, 0.0) if preserve else
                                 sum(posterior.get(item, 0.0) for item in row.hypothesis_ids))
                       for row in rows}
            if preserve:
                for row in rows:
                    if weights[row.key] <= 0:
                        weights[row.key] = .03
            # Unknown is always retained as a weak open-world alternative.
            weights["unknown"] = max(weights.get("unknown", 0.0), .03)
            total = sum(weights.values()) or 1.0
            self.components[family] = tuple(rows)
            self.marginals[family] = {key: value / total for key, value in weights.items()}

        self.role_bindings = {}
        for entity_id, counts in roles.items():
            total = sum(counts.values()) or 1.0
            self.role_bindings[entity_id] = {key: value / total for key, value in counts.items()}

    def update_family(self, family: str, hypothesis_likelihoods: dict[int, float],
                      hypothesis_weights: dict[int, float]) -> None:
        distribution = self.marginals.get(family)
        if not distribution:
            return
        component_map = {row.key: row for row in self.components.get(family, ())}
        updated: dict[str, float] = {}
        for key, prior in distribution.items():
            ids = component_map.get(key, WorldComponent(family, key, key, tuple())).hypothesis_ids
            denominator = sum(hypothesis_weights.get(item, 0.0) for item in ids)
            if ids and denominator > 0:
                likelihood = sum(hypothesis_weights.get(item, 0.0) *
                                 hypothesis_likelihoods.get(item, .5) for item in ids) / denominator
            else:
                likelihood = .5
            updated[key] = prior * max(.03, likelihood)
        total = sum(updated.values()) or 1.0
        self.marginals[family] = {key: value / total for key, value in updated.items()}

    def hypothesis_posterior(self, catalog: HypothesisCatalog,
                             excluded: set[int] | None = None) -> dict[int, float]:
        excluded = excluded or set()
        weights: dict[int, float] = {}
        for hypothesis in catalog.hypotheses:
            if hypothesis.id in excluded:
                weights[hypothesis.id] = 0.0
                continue
            parts = self.hypothesis_components.get(hypothesis.id, {})
            weight = 1.0
            for family in FAMILIES:
                weight *= max(1e-8, self.marginals.get(family, {}).get(parts.get(family, "unknown"), 1e-8))
            weights[hypothesis.id] = weight
        total = sum(weights.values())
        return ({key: value / total for key, value in weights.items()} if total else
                {row.id: 1 / len(catalog.hypotheses) for row in catalog.hypotheses})

    def family_entropy(self, family: str) -> float:
        values = self.marginals.get(family, {})
        maximum = math.log2(max(2, len(values)))
        return _entropy(values) / maximum if maximum else 0.0

    @property
    def mode(self) -> str:
        mean_entropy = sum(self.family_entropy(family) for family in FAMILIES) / len(FAMILIES)
        if mean_entropy >= .62:
            return "explore"
        if self.family_entropy("goal") >= .32:
            return "goal_discrimination"
        return "exploit"


class LocalEffectMemory:
    """Empirical action effects only; this is not a hidden game simulator."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)

    def observe(self, candidate_id: str, outcome: str) -> None:
        self.counts[candidate_id][outcome] += 1

    def distribution(self, candidate_id: str) -> dict[str, float]:
        counts = self.counts.get(candidate_id, Counter())
        total = sum(counts.values())
        return {key: value / total for key, value in counts.items()} if total else {}


class StructuredWorldReasoner(InformationGainReasoner):
    """V12 reasoner with a factorized posterior and open-world components."""

    def __init__(self, caller: Any, refresh_turns: int = 4):
        super().__init__(caller, refresh_turns)
        self.world = FactorizedWorldPosterior()
        self.effects = LocalEffectMemory()
        self.goal_registry: dict[str, dict[str, Any]] = {}
        self.goal_weights: dict[str, float] = {}
        self.architecture_prompt = """Build each complete hypothesis compositionally. Explicitly distinguish:
(1) which entity is the controller, (2) the desired relation or goal, (3) movement/effect dynamics,
(4) topology or boundary assumptions, (5) interaction mechanism, and (6) temporal/sequence rules.
Keep uncertain bindings uncertain. You may propose a novel relation rather than forcing the scene into a known
category. These are candidate latent world models, not facts, and Python will update each dimension separately.
For every hypothesis assign a goal_component_id and a goal_lineage_action of reuse, revise, or new. Reuse an
existing goal ID only when you intend the same goal despite changed wording. Revise creates a new goal ID and names
the old ID in parent_goal_component_id. New creates a new ID with no parent. Python will not infer paraphrase
equivalence for you."""

    @property
    def harness_version(self) -> str:
        return HARNESS_VERSION

    @property
    def input_representation(self) -> str:
        return INPUT_REPRESENTATION

    @property
    def harness_flags(self) -> dict[str, bool]:
        return {**super().harness_flags,
                "factorized_world_posterior": True,
                "uncertain_role_bindings": True,
                "open_world_components": True,
                "learned_local_effects": True,
                "explore_goal_discriminate_exploit": True,
                "luna_declared_goal_lineage": True,
                "persistent_goal_evidence": True,
                "observable_goal_discrimination": True,
                "no_privileged_simulator": True}

    @property
    def hypothesis_schema(self) -> dict[str, Any]:
        schema = deepcopy(HYPOTHESIS_SCHEMA)
        item = schema["properties"]["hypotheses"]["items"]
        item["properties"].update({
            "goal_component_id": {"type": "string"},
            "goal_lineage_action": {"type": "string", "enum": ["reuse", "revise", "new"]},
            "parent_goal_component_id": {"type": "string"},
            "topology_assumption": {"type": "string"},
        })
        item["required"].extend(("goal_component_id", "goal_lineage_action", "parent_goal_component_id",
                                 "topology_assumption"))
        return schema

    def reset_episode(self) -> None:
        super().reset_episode()
        self.world = FactorizedWorldPosterior()
        self.effects = LocalEffectMemory()
        self.goal_registry = {}
        self.goal_weights = {}

    def _goal_prompt(self) -> str:
        rows = [{"goal_component_id": key,
                 "description": value.get("description", ""),
                 "confirmation": value.get("confirmation", ""),
                 "falsification": value.get("falsification", ""),
                 "current_probability": self.goal_weights.get(key, 0.0)}
                for key, value in sorted(self.goal_registry.items())]
        return f"\nEXISTING GOAL COMPONENTS AND EVIDENCE: {rows or 'None; create new IDs such as g1.'}"

    def _validate_goal_lineage(self, catalog: HypothesisCatalog) -> HypothesisCatalog:
        """Validate explicit IDs mechanically without interpreting goal prose."""
        known = set(self.goal_registry)
        allocated = set(known)
        rows = []
        for position, hypothesis in enumerate(catalog.hypotheses, 1):
            action = hypothesis.goal_lineage_action
            goal_id = re.sub(r"[^a-zA-Z0-9_-]", "", hypothesis.goal_component_id)[:40]
            parent = re.sub(r"[^a-zA-Z0-9_-]", "", hypothesis.parent_goal_component_id)[:40]
            if action == "reuse" and goal_id not in allocated:
                action = "new"
            if action == "revise" and parent not in known:
                action = "new"; parent = ""
            if not goal_id or (action in {"new", "revise"} and goal_id in allocated):
                suffix = 1
                while f"g{position}-{suffix}" in allocated:
                    suffix += 1
                goal_id = f"g{position}-{suffix}"
            allocated.add(goal_id)
            revised = replace(hypothesis, goal_component_id=goal_id,
                              goal_lineage_action=action,
                              parent_goal_component_id=parent)
            rows.append(revised)
            self.goal_registry[goal_id] = {
                "description": revised.success_condition,
                "confirmation": list(revised.confirmation_outcomes),
                "falsification": revised.falsification,
                "parent": parent,
            }
        return replace(catalog, hypotheses=tuple(rows))

    def propose(self, observation: Any, history: list[dict[str, Any]], force: bool = False) -> HypothesisCatalog:
        base_prompt = self.architecture_prompt.split("\nEXISTING GOAL COMPONENTS", 1)[0]
        self.architecture_prompt = base_prompt + self._goal_prompt()
        previous_catalog = self.catalog
        catalog = super().propose(observation, history, force)
        if catalog is previous_catalog:
            self.world.rebuild(catalog, self.posterior)
            self.goal_weights = dict(self.world.marginals.get("goal", {}))
            return catalog
        catalog = self._validate_goal_lineage(catalog)
        self.catalog = catalog
        self.world.rebuild(catalog, self.posterior, preserve=bool(self.world.marginals))
        self.posterior = self.world.hypothesis_posterior(catalog, self.refuted_hypothesis_ids)
        self.goal_weights = dict(self.world.marginals.get("goal", {}))
        return catalog

    def update(self, candidate_id: str, observed: str,
               supporting_outcomes: tuple[str, ...] = tuple()):
        before = dict(self.posterior)
        _ledger_before, ledger_after = super().update(candidate_id, observed, supporting_outcomes)
        physical = supporting_outcomes[0] if supporting_outcomes else observed
        self.effects.observe(candidate_id, physical)
        if not self.catalog:
            return before, ledger_after
        physical_outcomes = {"NO_CHANGE", "LOCAL_CHANGE", "BROAD_CHANGE", "MOVE", "GROUP_TRANSFORM"}
        likelihoods: dict[int, float] = {}
        for hypothesis in self.catalog.hypotheses:
            predicted = hypothesis.prediction_for(candidate_id).outcome
            if predicted == physical:
                likelihoods[hypothesis.id] = .9
            elif predicted == "UNKNOWN" or predicted not in physical_outcomes:
                likelihoods[hypothesis.id] = .5
            else:
                likelihoods[hypothesis.id] = .15
        for family in ("controller", "interaction", "dynamics"):
            self.world.update_family(family, likelihoods, before)
        # Game outcomes are the only direct goal evidence available here.
        # Generic board transformations intentionally do not touch the goal family.
        if observed in {"PROGRESS", "TERMINAL", "FAIL"}:
            goal_likelihoods = {}
            for hypothesis in self.catalog.hypotheses:
                prediction = hypothesis.prediction_for(candidate_id).outcome
                goal_likelihoods[hypothesis.id] = .9 if prediction == observed else \
                    .5 if prediction == "UNKNOWN" else .15
            self.world.update_family("goal", goal_likelihoods, before)
        self.posterior = self.world.hypothesis_posterior(self.catalog, self.refuted_hypothesis_ids)
        self.goal_weights = dict(self.world.marginals.get("goal", {}))
        return before, dict(self.posterior)

    @staticmethod
    def _boundary_observed(before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
        old = {row.id: row for row in before}; new = {row.id: row for row in after}
        all_rows = (*before, *after)
        if not all_rows:
            return False
        max_x = max(row.bounds[2] for row in all_rows)
        max_y = max(row.bounds[3] for row in all_rows)
        changed = [key for key in old.keys() | new.keys()
                   if key not in old or key not in new or old[key].bounds != new[key].bounds]
        return any(any(value == edge for value, edge in zip(row.bounds, (0, 0, max_x, max_y)))
                   for key in changed for row in (old.get(key), new.get(key)) if row is not None)

    def assess_sequence(self, candidate_id: str, observed: str, physical_outcome: str,
                        before: tuple[Any, ...], after: tuple[Any, ...], completed_level: int):
        catalog = self.catalog
        weights = dict(self.posterior)
        relation_likelihoods: dict[int, float] = {}
        measurable = False
        if catalog:
            old_ids = {row.id for row in before}; new_ids = {row.id for row in after}
            directional = {"MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
                           "DISTANCE_DECREASES", "DISTANCE_INCREASES"}
            for hypothesis in catalog.hypotheses:
                eligible = hypothesis.expected_relation_change in directional and \
                    hypothesis.tracked_entity_id in old_ids & new_ids
                if hypothesis.expected_relation_change.startswith("DISTANCE_"):
                    eligible = eligible and hypothesis.target_entity_id in old_ids & new_ids
                if eligible:
                    measurable = True
                    relation_likelihoods[hypothesis.id] = .88 if self._relation_holds(
                        hypothesis, before, after, physical_outcome) else .16
                else:
                    relation_likelihoods[hypothesis.id] = .5
        status, validation = super().assess_sequence(
            candidate_id, observed, physical_outcome, before, after, completed_level)
        if catalog:
            # A declared directional/distance condition is measurable goal
            # evidence. ANY_CHANGE and GROUP_TRANSFORM are mechanism evidence only.
            if measurable and observed not in {"PROGRESS", "TERMINAL", "FAIL"}:
                self.world.update_family("goal", relation_likelihoods, weights)
            if status != "inactive":
                positive = status in {"continuing", "confirmed", "confirmed_continuing",
                                      "validated_level", "validated_terminal", "reframed_continuing"}
                sequence_likelihoods = {
                    hypothesis.id: (.82 if positive else .18)
                    if candidate_id in {*hypothesis.test_sequence, hypothesis.persistence_candidate_id}
                    else .5 for hypothesis in catalog.hypotheses
                }
                self.world.update_family("sequence", sequence_likelihoods, weights)
            if self._boundary_observed(before, after):
                boundary_likelihoods = {
                    hypothesis.id: .72 if hypothesis.prediction_for(candidate_id).outcome
                    in {physical_outcome, "UNKNOWN"} else .28 for hypothesis in catalog.hypotheses
                }
                self.world.update_family("topology", boundary_likelihoods, weights)
            self.posterior = self.world.hypothesis_posterior(catalog, self.refuted_hypothesis_ids)
            self.goal_weights = dict(self.world.marginals.get("goal", {}))
        return status, validation

    def build_plan_costs(self, catalog: HypothesisCatalog, posterior: dict[int, float],
                         history: list[dict[str, Any]]):
        encoding, costs, valid, diagnostics = build_v11_plan_costs(catalog, posterior, history)
        mode = self.world.mode
        goal_disagreement: dict[str, float] = {}
        novelty: dict[str, float] = {}
        for candidate in catalog.candidates:
            joint: dict[tuple[str, str], float] = defaultdict(float)
            goals: dict[str, float] = defaultdict(float)
            outcomes: dict[str, float] = defaultdict(float)
            meaningful_mass = 0.0
            for hypothesis in catalog.hypotheses:
                goal = self.world.hypothesis_components.get(hypothesis.id, {}).get("goal", "unknown")
                outcome = hypothesis.prediction_for(candidate.candidate_id).outcome
                probability = posterior.get(hypothesis.id, 0.0)
                joint[(goal, outcome)] += probability
                goals[goal] += probability
                outcomes[outcome] += probability
                if outcome not in {"NO_CHANGE", "UNKNOWN"}:
                    meaningful_mass += probability
            mutual_information = sum(
                probability * math.log2(probability / max(1e-12, goals[goal] * outcomes[outcome]))
                for (goal, outcome), probability in joint.items() if probability > 0)
            # Distinguishing prose goals is useless when every branch predicts
            # inactivity. Require observable, non-null evidence.
            goal_disagreement[candidate.candidate_id] = max(0.0, mutual_information) * meaningful_mass
            novelty[candidate.candidate_id] = 1.0 / (1.0 + sum(self.effects.counts[candidate.candidate_id].values()))

        progress = diagnostics["expected_progress"]
        information = diagnostics["information_gain"]
        if mode == "explore":
            bonus = {key: .55 * novelty[key] + .35 * information[key] for key in novelty}
        elif mode == "goal_discrimination":
            bonus = {key: .6 * goal_disagreement[key] + .25 * information[key] for key in novelty}
        else:
            bonus = {key: .7 * progress[key] + .15 * (1 - novelty[key]) for key in novelty}
        for index in np.flatnonzero(valid):
            plan = encoding.plan(int(index))
            if plan:
                costs[int(index)] -= sum(weight * bonus[action]
                                         for weight, action in zip((1.0, .55, .3), plan))
        diagnostics.update({"cost_model": "factorized-world-mode-objective-v1",
                            "world_model_mode": mode,
                            "goal_disagreement": goal_disagreement,
                            "local_effect_novelty": novelty})
        return encoding, costs, valid, diagnostics

    def architecture_diagnostics(self) -> dict[str, Any]:
        return {
            "world_model_mode": self.world.mode,
            "family_posteriors": self.world.marginals,
            "family_entropy": {family: self.world.family_entropy(family) for family in FAMILIES},
            "uncertain_role_bindings": self.world.role_bindings,
            "goal_component_registry": self.goal_registry,
            "goal_component_probabilities": self.goal_weights,
            "learned_local_effects": {key: self.effects.distribution(key) for key in self.effects.counts},
        }


def run_structured_world_comparison(*args: Any, **kwargs: Any):
    return run_information_gain_comparison(*args, **kwargs)

