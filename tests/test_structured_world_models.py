import unittest
from dataclasses import replace
from types import SimpleNamespace

from qahs.information_gain import HypothesisCatalog, Prediction, TestableHypothesis
from qahs.arc_models import ArcActionCandidate
from qahs.structured_world_models import (
    FactorizedWorldPosterior, FAMILIES, LocalEffectMemory, StructuredWorldReasoner,
)


def _hypothesis(identifier: int, goal: str, actuator: str) -> TestableHypothesis:
    return TestableHypothesis(
        identifier, f"Use {actuator} to reach {goal}.", (actuator,), "toggle", goal,
        "no visible effect", 0.5, (Prediction("click-e1", "LOCAL_CHANGE", .8),),
        ("click-e1",), goal, "goal reached", "", actuator, "", "ANY_CHANGE",
        ("LOCAL_CHANGE",), ("FAIL",), actuator, ("e2",),
    )


class StructuredWorldModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

import unittest

