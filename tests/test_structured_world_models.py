import unittest
from dataclasses import replace
from types import SimpleNamespace

from qahs.information_gain import HypothesisCatalog, Prediction, TestableHypothesis
from qahs.arc_models import ArcActionCandidate, ArcObservation
from qahs.structured_world_models import (
    FactorizedWorldPosterior, FAMILIES, LocalEffectMemory, ObservableGoalPredicate,
    StructuredWorldReasoner, evaluate_goal_predicate, relational_effect_signature,
)
from qahs.compositional import EntityFact


def _hypothesis(identifier: int, goal: str, actuator: str) -> TestableHypothesis:
    return TestableHypothesis(
        identifier, f"Use {actuator} to reach {goal}.", (actuator,), "toggle", goal,
        "no visible effect", 0.5, (Prediction("click-e1", "LOCAL_CHANGE", .8),),
        ("click-e1",), goal, "goal reached", "", actuator, "", "ANY_CHANGE",
        ("LOCAL_CHANGE",), ("FAIL",), actuator, ("e2",),
    )


class StructuredWorldModelTests(unittest.TestCase):
    def test_cached_catalog_restores_observable_goal_predicates(self):
        class Caller:
            model = "test"
            reasoning_effort = "none"

            def __init__(self):
                self.calls = 0

            def call_json(self, _prompt, _schema, _name, _images):
                self.calls += 1
                return {"summary": "test", "hypotheses": [{
                    "id": 1,
                    "statement": "move the tile beside the marker",
                    "goal_component_id": "g1",
                    "goal_lineage_action": "new",
                    "parent_goal_component_id": "",
                    "observable_goal_predicates": [{
                        "kind": "adjacent", "subject_id": "e0", "target_id": "e1",
                        "reference_id": "", "desired": True,
                    }],
                }], "recommended_candidate_id": "", "recommendation_reason": "test"}

        caller = Caller()
        reasoner = StructuredWorldReasoner(caller)
        frame = tuple(tuple(1 if (x, y) == (1, 1) else 2 if (x, y) == (3, 1) else 0
                            for x in range(5)) for y in range(4))
        observation = ArcObservation("test", "NOT_FINISHED", 0, 1, ("ACTION6",), frame)

        reasoner.propose(observation, [], force=True)
        first = reasoner.goal_predicates
        first_logical_seconds = reasoner.last_logical_model_seconds
        reasoner.reset_episode()
        reasoner.propose(observation, [], force=True)

        self.assertEqual(caller.calls, 1)
        self.assertTrue(reasoner.last_catalog_cache_hit)
        self.assertEqual(reasoner.last_logical_model_seconds, first_logical_seconds)
        self.assertEqual(reasoner.goal_predicates, first)
        self.assertEqual(reasoner.goal_predicates[1][0].kind, "adjacent")

    def test_catalog_reuse_preserves_factorized_evidence_channels(self):
        candidate = ArcActionCandidate("click-e0", "ACTION6", "click e0", x=1, y=1)
        first = replace(_hypothesis(1, "goal one", "e1"), goal_component_id="g1")
        second = replace(_hypothesis(2, "goal two", "e2"), goal_component_id="g2")
        catalog = HypothesisCatalog("test", (first, second), (candidate,), "click-e0", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        reasoner.catalog = catalog
        reasoner.posterior = {1: .5, 2: .5}
        reasoner.world.rebuild(catalog, reasoner.posterior)
        reasoner.world.marginals["goal"] = {"g1": .8, "g2": .15, "unknown": .05}
        expected = dict(reasoner.world.marginals["goal"])
        # No new catalog is needed while the candidate set is unchanged and the
        # refresh threshold has not been reached.
        reasoner.evidence_count = 0
        frame = tuple(tuple(1 if (x, y) == (1, 1) else 0 for x in range(4)) for y in range(4))
        observation = ArcObservation("test", "NOT_FINISHED", 0, 1, ("ACTION6",), frame)

        reasoner.propose(observation, [])

        self.assertEqual(reasoner.world.marginals["goal"], expected)
        self.assertEqual(reasoner.goal_weights, expected)

    def test_relational_fingerprint_distinguishes_opposite_transformations(self):
        before = (EntityFact("e1", 1, 1, (2, 2, 2, 2), (2, 2)),
                  EntityFact("e2", 2, 1, (6, 2, 6, 2), (6, 2)))
        left = (EntityFact("e1", 1, 1, (1, 2, 1, 2), (1, 2)), before[1])
        right = (EntityFact("e1", 1, 1, (3, 2, 3, 2), (3, 2)), before[1])

        left_signature = relational_effect_signature(before, left)
        right_signature = relational_effect_signature(before, right)

        self.assertNotEqual(left_signature["fingerprint"], right_signature["fingerprint"])
        self.assertEqual(left_signature["motions"]["e1"], [-1, 0])
        self.assertEqual(right_signature["motions"]["e1"], [1, 0])

    def test_python_evaluates_categorical_goal_relation(self):
        before = (EntityFact("piece", 2, 1, (0, 0, 0, 0), (0, 0)),
                  EntityFact("left", 1, 1, (2, 0, 2, 0), (2, 0)),
                  EntityFact("right", 3, 1, (6, 0, 6, 0), (6, 0)))
        after = (EntityFact("piece", 2, 1, (4, 0, 4, 0), (4, 0)), before[1], before[2])
        predicate = ObservableGoalPredicate("between_x", "piece", "left", "right", True)

        self.assertEqual(evaluate_goal_predicate(predicate, before, after), "improved")

    def test_measured_goal_predicate_updates_goal_family(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        first = replace(_hypothesis(1, "piece is between anchors", "e1"),
                        goal_component_id="g1", goal_lineage_action="new")
        second = replace(_hypothesis(2, "piece remains outside anchors", "e2"),
                         goal_component_id="g2", goal_lineage_action="new")
        catalog = HypothesisCatalog("test", (first, second), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        catalog = reasoner._validate_goal_lineage(catalog)
        reasoner.catalog = catalog; reasoner.posterior = {1: .5, 2: .5}
        reasoner.world.rebuild(catalog, reasoner.posterior)
        reasoner.goal_predicates = {
            1: (ObservableGoalPredicate("between_x", "piece", "left", "right"),),
            2: (ObservableGoalPredicate("between_x", "piece", "left", "right", False),),
        }
        before = (EntityFact("piece", 2, 1, (0, 0, 0, 0), (0, 0)),
                  EntityFact("left", 1, 1, (2, 0, 2, 0), (2, 0)),
                  EntityFact("right", 3, 1, (6, 0, 6, 0), (6, 0)))
        after = (EntityFact("piece", 2, 1, (4, 0, 4, 0), (4, 0)), before[1], before[2])

        reasoner.assess_sequence("click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", before, after, 0)

        self.assertGreater(reasoner.world.marginals["goal"]["g1"],
                           reasoner.world.marginals["goal"]["g2"])

    def test_reaching_observable_goal_relation_reopens_reasoning(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(_hypothesis(1, "piece is between anchors", "e1"),
                      goal_component_id="g1", goal_lineage_action="new",
                      predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .9),),
                      persistence_candidate_id="click-e1",
                      confirmation_outcomes=("GROUP_TRANSFORM",))
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        catalog = reasoner._validate_goal_lineage(catalog)
        reasoner.catalog = catalog; reasoner.posterior = {1: 1.0}; reasoner.world.rebuild(catalog, reasoner.posterior)
        reasoner.goal_predicates = {1: (ObservableGoalPredicate("between_x", "piece", "left", "right"),)}
        reasoner.start_sequence(catalog, reasoner.posterior, "click-e1")
        before = (EntityFact("piece", 2, 1, (0, 0, 0, 0), (0, 0)),
                  EntityFact("left", 1, 1, (2, 0, 2, 0), (2, 0)),
                  EntityFact("right", 3, 1, (6, 0, 6, 0), (6, 0)))
        after = (EntityFact("piece", 2, 1, (4, 0, 4, 0), (4, 0)), before[1], before[2])

        status, _ = reasoner.assess_sequence(
            "click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", before, after, 0)

        self.assertEqual(status, "goal_predicate_reached")
        self.assertIsNone(reasoner.queued_action((candidate,)))

    def test_nonproductive_state_cycle_ends_persistence(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(_hypothesis(1, "cycle until a relation changes", "e1"),
                      predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .9),),
                      persistence_candidate_id="click-e1",
                      confirmation_outcomes=("GROUP_TRANSFORM",))
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        reasoner.catalog = catalog; reasoner.posterior = {1: 1.0}; reasoner.world.rebuild(catalog, reasoner.posterior)
        reasoner.start_sequence(catalog, reasoner.posterior, "click-e1")
        first = (EntityFact("e1", 1, 1, (0, 0, 0, 0), (0, 0)),)
        second = (EntityFact("e1", 1, 1, (1, 0, 1, 0), (1, 0)),)
        reasoner.assess_sequence("click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", first, second, 0)

        status, _ = reasoner.assess_sequence(
            "click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", second, first, 0)

        self.assertEqual(status, "refuted_cycle")
        self.assertIsNone(reasoner.queued_action((candidate,)))

    def test_goal_relation_failure_preserves_validated_controller_evidence(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(_hypothesis(1, "controller moves piece toward target", "e1"),
                      predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .9),),
                      persistence_candidate_id="click-e1", tracked_entity_id="piece",
                      target_entity_id="target", expected_relation_change="DISTANCE_DECREASES",
                      confirmation_outcomes=("GROUP_TRANSFORM",))
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        reasoner.catalog = catalog; reasoner.posterior = {1: 1.0}; reasoner.world.rebuild(catalog, reasoner.posterior)
        reasoner.start_sequence(catalog, reasoner.posterior, "click-e1")
        before = (EntityFact("piece", 1, 1, (2, 0, 2, 0), (2, 0)),
                  EntityFact("target", 2, 1, (6, 0, 6, 0), (6, 0)))
        farther = (EntityFact("piece", 1, 1, (1, 0, 1, 0), (1, 0)), before[1])

        status, _ = reasoner.assess_sequence(
            "click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", before, farther, 0)

        self.assertEqual(status, "confirmed_continuing")
        self.assertEqual(reasoner.last_goal_component_status, "weakened")
        self.assertNotIn(1, reasoner.refuted_hypothesis_ids)
        self.assertEqual(reasoner.queued_action((candidate,)), "click-e1")
        self.assertGreater(reasoner.world.marginals["controller"].get("<entity>", 0), 0)

    def test_repeated_goal_weakening_forces_review_without_erasing_mechanism(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(_hypothesis(1, "controller moves piece toward target", "e1"),
                      predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .9),),
                      persistence_candidate_id="click-e1", tracked_entity_id="piece",
                      target_entity_id="target", expected_relation_change="DISTANCE_DECREASES",
                      confirmation_outcomes=("GROUP_TRANSFORM",))
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        reasoner.catalog = catalog; reasoner.posterior = {1: 1.0}; reasoner.world.rebuild(catalog, reasoner.posterior)
        reasoner.start_sequence(catalog, reasoner.posterior, "click-e1")
        states = [
            (EntityFact("piece", 1, 1, (x, 0, x, 0), (x, 0)),
             EntityFact("target", 2, 1, (6, 0, 6, 0), (6, 0)))
            for x in (3, 2, 1, 0)
        ]

        for before, after in zip(states, states[1:]):
            status, _ = reasoner.assess_sequence(
                "click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", before, after, 0)
            self.assertEqual(status, "confirmed_continuing")

        self.assertTrue(reasoner.requires_hypothesis_review)
        self.assertIn("3 consecutive observations", reasoner.prepare_hypothesis_review())
        self.assertFalse(reasoner.requires_hypothesis_review)
        self.assertNotIn(1, reasoner.refuted_hypothesis_ids)

    def test_factorized_world_keeps_open_unknown_and_normalizes_each_family(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        hypotheses = (_hypothesis(1, "colors match", "e1"),
                      _hypothesis(2, "objects align", "e2"))
        catalog = HypothesisCatalog("test", hypotheses, (candidate,), "click-e1", "test")
        world = FactorizedWorldPosterior()
        world.rebuild(catalog, {1: .7, 2: .3})

        self.assertEqual(set(world.marginals), set(FAMILIES))
        for distribution in world.marginals.values():
            self.assertIn("unknown", distribution)
            self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(world.role_bindings["e1"]["controller"], 0)

    def test_local_effect_memory_is_empirical_and_normalized(self):
        memory = LocalEffectMemory()
        memory.observe("click-e1", "MOVE")
        memory.observe("click-e1", "MOVE")
        memory.observe("click-e1", "NO_CHANGE")

        self.assertEqual(memory.distribution("unseen"), {})
        self.assertEqual(memory.distribution("click-e1"), {"MOVE": 2 / 3, "NO_CHANGE": 1 / 3})

    def test_luna_declared_goal_id_is_reused_without_python_semantic_matching(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        first = replace(_hypothesis(1, "colors match", "e1"),
                        goal_component_id="g1", goal_lineage_action="new")
        second = replace(_hypothesis(2, "make the colors identical", "e2"),
                         goal_component_id="g1", goal_lineage_action="reuse")
        catalog = HypothesisCatalog("test", (first, second), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))

        validated = reasoner._validate_goal_lineage(catalog)

        self.assertEqual([row.goal_component_id for row in validated.hypotheses], ["g1", "g1"])
        self.assertEqual(validated.hypotheses[1].goal_lineage_action, "reuse")

    def test_invalid_reuse_becomes_new_without_comparing_prose(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(_hypothesis(1, "colors match", "e1"),
                      goal_component_id="missing", goal_lineage_action="reuse")
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))

        validated = reasoner._validate_goal_lineage(catalog)

        self.assertEqual(validated.hypotheses[0].goal_lineage_action, "new")

    def test_generic_transform_does_not_update_goal_family(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        first = replace(_hypothesis(1, "colors match", "e1"), goal_component_id="g1",
                        goal_lineage_action="new", expected_relation_change="ANY_CHANGE",
                        predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .8),))
        second = replace(_hypothesis(2, "objects align", "e2"), goal_component_id="g2",
                         goal_lineage_action="new", expected_relation_change="ANY_CHANGE",
                         predictions=(Prediction("click-e1", "NO_CHANGE", .8),))
        catalog = HypothesisCatalog("test", (first, second), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        catalog = reasoner._validate_goal_lineage(catalog)
        reasoner.catalog = catalog
        reasoner.posterior = {1: .5, 2: .5}
        reasoner.world.rebuild(catalog, reasoner.posterior)
        goals_before = dict(reasoner.world.marginals["goal"])
        dynamics_before = dict(reasoner.world.marginals["dynamics"])

        reasoner.update("click-e1", "GROUP_TRANSFORM", ("GROUP_TRANSFORM",))

        self.assertEqual(reasoner.world.marginals["goal"], goals_before)
        self.assertNotEqual(reasoner.world.marginals["dynamics"], dynamics_before)

    def test_matching_physical_effect_sustains_procedure_without_validating_goal(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(
            _hypothesis(1, "unknown arrangement is reached", "e1"),
            goal_component_id="g1", goal_lineage_action="new",
            expected_relation_change="ANY_CHANGE",
            predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .9),),
            persistence_candidate_id="click-e1",
            confirmation_outcomes=("GROUP_TRANSFORM",),
            refutation_outcomes=("NO_CHANGE", "FAIL"),
        )
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        catalog = reasoner._validate_goal_lineage(catalog)
        reasoner.catalog = catalog
        reasoner.posterior = {1: 1.0}
        reasoner.world.rebuild(catalog, reasoner.posterior)
        goals_before = dict(reasoner.world.marginals["goal"])
        reasoner.start_sequence(catalog, reasoner.posterior, "click-e1")

        reasoner.update("click-e1", "GROUP_TRANSFORM", ("GROUP_TRANSFORM",))
        status, validation = reasoner.assess_sequence(
            "click-e1", "GROUP_TRANSFORM", "GROUP_TRANSFORM", tuple(), tuple(), 0)

        self.assertEqual(status, "confirmed_continuing")
        self.assertIsNone(validation)
        self.assertEqual(reasoner.queued_action((candidate,)), "click-e1")
        self.assertEqual(reasoner.world.marginals["goal"], goals_before)

    def test_procedure_stops_when_declared_physical_prediction_fails(self):
        candidate = ArcActionCandidate("click-e1", "ACTION6", "click e1", x=1, y=1)
        row = replace(
            _hypothesis(1, "unknown arrangement is reached", "e1"),
            expected_relation_change="ANY_CHANGE",
            predictions=(Prediction("click-e1", "GROUP_TRANSFORM", .9),),
            persistence_candidate_id="click-e1",
            confirmation_outcomes=("GROUP_TRANSFORM",),
            refutation_outcomes=("NO_CHANGE", "FAIL"),
        )
        catalog = HypothesisCatalog("test", (row,), (candidate,), "click-e1", "test")
        reasoner = StructuredWorldReasoner(SimpleNamespace(model="test", reasoning_effort="none"))
        reasoner.catalog = catalog
        reasoner.posterior = {1: 1.0}
        reasoner.start_sequence(catalog, reasoner.posterior, "click-e1")

        status, _ = reasoner.assess_sequence(
            "click-e1", "NO_CHANGE", "NO_CHANGE", tuple(), tuple(), 0)

        self.assertEqual(status, "refuted")
        self.assertIsNone(reasoner.queued_action((candidate,)))


if __name__ == "__main__":
    unittest.main()

import unittest

