import unittest
from types import SimpleNamespace

from qahs.arc_models import ArcActionCandidate
from qahs.information_gain import (
    HypothesisCatalog, InformationGainReasoner, PlanEncoding, Prediction,
    TestableHypothesis, build_plan_costs, expected_information_gain,
    _interactive_entities, _tracked_entities, action_catalog,
    run_information_gain_episode,
)
from qahs.compositional import EntityFact
from qahs.arc_models import ArcObservation


def hypothesis(hid, prior, outcomes):
    return TestableHypothesis(hid, f"rule {hid}", tuple(), "mechanism", "progress", "mismatch", prior,
                              tuple(Prediction(candidate, outcome, .9) for candidate, outcome in outcomes.items()))


class InformationGainTests(unittest.TestCase):
    def setUp(self):
        self.candidates = tuple(ArcActionCandidate(name, "ACTION6", name) for name in ("a", "b", "c"))
        self.catalog = HypothesisCatalog("test", (
            hypothesis(1, .5, {"a": "MOVE", "b": "NO_CHANGE", "c": "LOCAL_CHANGE"}),
            hypothesis(2, .5, {"a": "NO_CHANGE", "b": "NO_CHANGE", "c": "LOCAL_CHANGE"}),
        ), self.candidates, "a", "diagnostic")

    def test_information_gain_rewards_disagreement(self):
        posterior = {1: .5, 2: .5}
        self.assertGreater(expected_information_gain(self.catalog, posterior, "a"),
                           expected_information_gain(self.catalog, posterior, "b"))

    def test_matching_prediction_gains_probability(self):
        reasoner = InformationGainReasoner(object())
        reasoner.catalog = self.catalog
        reasoner.posterior = {1: .5, 2: .5}
        _, after = reasoner.update("a", "MOVE")
        self.assertGreater(after[1], after[2])

    def test_repeated_contradiction_retires_but_records_hypothesis(self):
        reasoner = InformationGainReasoner(object())
        reasoner.catalog = self.catalog
        reasoner.posterior = {1: .5, 2: .5}
        reasoner.contradictions = {1: 0, 2: 0}
        reasoner.update("a", "MOVE")
        reasoner.update("a", "MOVE")
        self.assertEqual(reasoner.posterior[2], 0)
        self.assertIn("rule 2", reasoner.retired_statements)

    def test_productive_hypothesis_persists_without_an_arbitrary_horizon(self):
        sequenced = TestableHypothesis(3, "move to target", tuple(), "movement", "aligned", "blocked", .8,
            tuple(Prediction(candidate.candidate_id, "MOVE", .9) for candidate in self.candidates),
            ("a",), "aligned", "movement contradicts the hypothesis", "a", "e1", "e2",
            "DISTANCE_DECREASES", ("PROGRESS", "TERMINAL"), ("FAIL",))
        catalog = HypothesisCatalog("test", (sequenced,), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object())
        reasoner.catalog = catalog; reasoner.posterior = {3: 1.0}
        reasoner.start_sequence(catalog, {3: 1.0}, "a")
        before = (EntityFact("e1", 1, 1, (1, 1, 1, 1), (1, 1)),
                  EntityFact("e2", 2, 1, (5, 1, 5, 1), (5, 1)))
        after = (EntityFact("e1", 1, 1, (2, 1, 2, 1), (2, 1)), before[1])
        self.assertEqual(reasoner.assess_sequence("a", "MOVE", "MOVE", before, after, 0)[0], "continuing")
        self.assertEqual(reasoner.queued_action(self.candidates), "a")
        later = (EntityFact("e1", 1, 1, (3, 1, 3, 1), (3, 1)), before[1])
        self.assertEqual(reasoner.assess_sequence("a", "MOVE", "MOVE", after, later, 0)[0], "continuing")
        self.assertEqual(reasoner.queued_action(self.candidates), "a")
        self.assertEqual(reasoner.assess_sequence("a", "NO_CHANGE", "NO_CHANGE", later, later, 0)[0], "refuted")
        self.assertIsNone(reasoner.queued_action(self.candidates))

    def test_better_explanation_can_inherit_an_active_procedure(self):
        first = TestableHypothesis(1, "move toward marker", tuple(), "movement", "aligned", "wrong way", .4,
            (Prediction("a", "MOVE", .7),), ("a",), "aligned", "contradiction", "a", "e1", "e2",
            "DISTANCE_DECREASES", ("PROGRESS",), ("FAIL",))
        revised = TestableHypothesis(2, "move toward matching marker", tuple(), "matching", "aligned", "wrong color", .6,
            (Prediction("a", "MOVE", .9),), ("a",), "matched and aligned", "contradiction", "a", "e1", "e3",
            "DISTANCE_DECREASES", ("PROGRESS",), ("FAIL",))
        catalog = HypothesisCatalog("test", (first, revised), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object())
        reasoner.start_sequence(HypothesisCatalog("test", (first,), self.candidates, "a", "test"), {1: .2}, "a")
        reasoner.active_last_candidate = "a"; reasoner.active_last_outcome = "MOVE"
        self.assertTrue(reasoner.reconsider_sequence(catalog, {1: .2, 2: .8}))
        self.assertEqual(reasoner.active_hypothesis_id, 2)
        self.assertEqual(reasoner.queued_action(self.candidates), "a")

    def test_confirmation_supports_persistence_only_after_relation_validation(self):
        persistent = TestableHypothesis(4, "control moves token toward target", tuple(), "movement",
            "aligned", "distance does not decrease", .9, (Prediction("a", "MOVE", .9),),
            ("a", "b"), "aligned", "relationship contradiction", "a", "e1", "e2",
            "DISTANCE_DECREASES", ("MOVE",), ("FAIL",), "control", ("e1",))
        catalog = HypothesisCatalog("test", (persistent,), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object()); reasoner.catalog = catalog; reasoner.posterior = {4: 1.0}
        reasoner.start_sequence(catalog, {4: 1.0}, "a")
        before = (EntityFact("e1", 1, 1, (1, 1, 1, 1), (1, 1)),
                  EntityFact("e2", 2, 1, (5, 1, 5, 1), (5, 1)))
        closer = (EntityFact("e1", 1, 1, (2, 1, 2, 1), (2, 1)), before[1])
        self.assertEqual(reasoner.assess_sequence("a", "MOVE", "MOVE", before, closer, 0)[0], "confirmed_continuing")
        self.assertEqual(reasoner.queued_action(self.candidates), "a")
        reasoner.start_sequence(catalog, {4: 1.0}, "a")
        farther = (EntityFact("e1", 1, 1, (0, 1, 0, 1), (0, 1)), before[1])
        self.assertEqual(reasoner.assess_sequence("a", "MOVE", "MOVE", before, farther, 0)[0], "refuted")
        self.assertIsNone(reasoner.queued_action(self.candidates))

    def test_game_failure_refutes_even_when_physical_prediction_matches(self):
        persistent = TestableHypothesis(6, "transform the group until progress", tuple(), "cycle",
            "level advances", "game failure", .9, (Prediction("a", "GROUP_TRANSFORM", .9),),
            ("a",), "next level", "failure", "a", "", "", "GROUP_TRANSFORM",
            ("GROUP_TRANSFORM",), ("FAIL",), "control", ("e1",))
        catalog = HypothesisCatalog("test", (persistent,), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object()); reasoner.catalog = catalog; reasoner.posterior = {6: 1.0}
        reasoner.start_sequence(catalog, {6: 1.0}, "a")
        status, validation = reasoner.assess_sequence(
            "a", "FAIL", "GROUP_TRANSFORM", tuple(), tuple(), 0)
        self.assertEqual(status, "refuted")
        self.assertIsNone(validation)
        self.assertIsNone(reasoner.queued_action(self.candidates))

    def test_generic_transform_completes_test_sequence_without_blind_persistence(self):
        actuator = TestableHypothesis(7, "paired controls transform the board", tuple(), "transform",
            "discover which control is useful", "transform stops", 1.0,
            tuple(Prediction(candidate.candidate_id, "GROUP_TRANSFORM", .9) for candidate in self.candidates),
            ("a", "b"), "compare both controls", "unexpected effect", "a", "", "",
            "GROUP_TRANSFORM", ("GROUP_TRANSFORM",), ("NO_CHANGE", "FAIL"))
        catalog = HypothesisCatalog("test", (actuator,), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object()); reasoner.catalog = catalog; reasoner.posterior = {7: 1.0}
        reasoner.start_sequence(catalog, {7: 1.0}, "a")
        self.assertEqual(reasoner.assess_sequence(
            "a", "GROUP_TRANSFORM", "GROUP_TRANSFORM", tuple(), tuple(), 0)[0], "confirmed_continuing")
        self.assertEqual(reasoner.queued_action(self.candidates), "b")
        self.assertEqual(reasoner.assess_sequence(
            "b", "GROUP_TRANSFORM", "GROUP_TRANSFORM", tuple(), tuple(), 0)[0], "confirmed")
        self.assertIsNone(reasoner.queued_action(self.candidates))
        self.assertEqual(reasoner.evidence_count, reasoner.refresh_turns)

    def test_unexpected_effect_transfers_to_hypothesis_that_predicted_it(self):
        wrong = TestableHypothesis(8, "control is inert", tuple(), "inert", "no change", "movement",
            .6, (Prediction("a", "NO_CHANGE", .9),), ("a",), "inert", "movement",
            "", "e1", "", "NO_CHANGE", ("NO_CHANGE",), ("MOVE",))
        better = TestableHypothesis(9, "control moves the token right", tuple(), "movement", "reach target",
            "wrong direction", .4, (Prediction("a", "MOVE", .9), Prediction("b", "MOVE", .8)),
            ("a", "b"), "reach the right side", "movement stops", "a", "e1", "",
            "MOVE_RIGHT", ("MOVE",), ("NO_CHANGE", "FAIL"))
        catalog = HypothesisCatalog("test", (wrong, better), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object()); reasoner.catalog = catalog
        reasoner.posterior = {8: .8, 9: .2}; reasoner.start_sequence(catalog, reasoner.posterior, "a")
        reasoner.posterior = {8: .1, 9: .9}
        before = (EntityFact("e1", 1, 1, (1, 1, 1, 1), (1, 1)),)
        after = (EntityFact("e1", 1, 1, (2, 1, 2, 1), (2, 1)),)
        self.assertEqual(reasoner.assess_sequence("a", "MOVE", "MOVE", before, after, 0)[0],
                         "reframed_continuing")
        self.assertEqual(reasoner.last_transfer_hypothesis_id, 9)
        self.assertEqual(reasoner.active_hypothesis_id, 9)
        self.assertEqual(reasoner.queued_action(self.candidates), "a")

    def test_refuted_hypotheses_cannot_resurrect_each_other(self):
        left = TestableHypothesis(10, "only a transforms", tuple(), "transform", "compare controls",
            "b also transforms", .5, (Prediction("a", "GROUP_TRANSFORM", .9),
                                      Prediction("b", "NO_CHANGE", .9)),
            ("a", "b"), "compare both controls", "unexpected effect", "a", "", "",
            "GROUP_TRANSFORM", ("GROUP_TRANSFORM",), ("NO_CHANGE", "FAIL"))
        right = TestableHypothesis(11, "only b transforms", tuple(), "transform", "compare controls",
            "a also transforms", .5, (Prediction("a", "NO_CHANGE", .9),
                                      Prediction("b", "GROUP_TRANSFORM", .9)),
            ("b", "a"), "compare both controls", "unexpected effect", "b", "", "",
            "GROUP_TRANSFORM", ("GROUP_TRANSFORM",), ("NO_CHANGE", "FAIL"))
        catalog = HypothesisCatalog("test", (left, right), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object()); reasoner.catalog = catalog
        reasoner.posterior = {10: .6, 11: .4}; reasoner.start_sequence(catalog, reasoner.posterior, "a")
        self.assertEqual(reasoner.assess_sequence(
            "a", "GROUP_TRANSFORM", "GROUP_TRANSFORM", tuple(), tuple(), 0)[0], "confirmed_continuing")
        self.assertEqual(reasoner.queued_action(self.candidates), "b")
        self.assertEqual(reasoner.assess_sequence(
            "b", "GROUP_TRANSFORM", "GROUP_TRANSFORM", tuple(), tuple(), 0)[0], "reframed_continuing")
        self.assertEqual(reasoner.refuted_hypothesis_ids, {10})
        self.assertEqual(reasoner.queued_action(self.candidates), "a")
        self.assertEqual(reasoner.assess_sequence(
            "a", "GROUP_TRANSFORM", "GROUP_TRANSFORM", tuple(), tuple(), 0)[0], "refuted")
        self.assertEqual(reasoner.refuted_hypothesis_ids, {10, 11})
        self.assertIsNone(reasoner.queued_action(self.candidates))
        self.assertEqual(reasoner.evidence_count, reasoner.refresh_turns)

    def test_progress_validates_matching_physical_hypothesis_for_level(self):
        persistent = TestableHypothesis(5, "control transforms the board until progress", tuple(), "cycle",
            "level advances", "transform stops", .9, (Prediction("a", "GROUP_TRANSFORM", .9),),
            ("a",), "next level", "contradiction", "a", "", "", "GROUP_TRANSFORM",
            ("GROUP_TRANSFORM",), ("NO_CHANGE", "PROGRESS"), "control", ("e1", "e2"))
        catalog = HypothesisCatalog("test", (persistent,), self.candidates, "a", "test")
        reasoner = InformationGainReasoner(object()); reasoner.catalog = catalog; reasoner.posterior = {5: 1.0}
        reasoner.start_sequence(catalog, {5: 1.0}, "a")
        status, validation = reasoner.assess_sequence("a", "PROGRESS", "GROUP_TRANSFORM", tuple(), tuple(), 1)
        self.assertEqual(status, "validated_level")
        self.assertEqual(validation["validated_for_level"], 1)
        self.assertEqual(validation["hypothesis_id"], 5)
        self.assertIsNone(reasoner.queued_action(self.candidates))

    def test_plan_encoding_rejects_duplicate_slots(self):
        encoding = PlanEncoding(self.candidates)
        valid = [encoding.plan(index) for index in range(encoding.state_count)]
        plans = [plan for plan in valid if plan is not None]
        self.assertEqual(len(plans), 6)
        self.assertTrue(all(len(set(plan)) == 3 for plan in plans))

    def test_exact_objective_starts_with_informative_action(self):
        encoding, costs, valid, diagnostics = build_plan_costs(self.catalog, {1: .5, 2: .5}, [])
        best = min((index for index, is_valid in enumerate(valid) if is_valid), key=lambda index: costs[index])
        self.assertEqual(encoding.plan(best)[0], "a")
        self.assertEqual(diagnostics["plan_depth"], 3)

    def test_progress_disagreement_outweighs_inactivity_disagreement(self):
        catalog = HypothesisCatalog("test", (
            hypothesis(1, .5, {"a": "NO_CHANGE", "b": "NO_CHANGE", "c": "NO_CHANGE"}),
            hypothesis(2, .5, {"a": "LOCAL_CHANGE", "b": "PROGRESS", "c": "NO_CHANGE"}),
        ), self.candidates, "a", "test")
        self.assertGreater(expected_information_gain(catalog, {1: .5, 2: .5}, "b"),
                           expected_information_gain(catalog, {1: .5, 2: .5}, "a"))

    def test_two_identical_no_changes_block_same_first_action(self):
        history = [{"candidate_id": "a", "observed_outcome": "NO_CHANGE"},
                   {"candidate_id": "a", "observed_outcome": "NO_CHANGE"}]
        encoding, costs, valid, diagnostics = build_plan_costs(self.catalog, {1: .5, 2: .5}, history)
        best = min((index for index, is_valid in enumerate(valid) if is_valid), key=lambda index: costs[index])
        self.assertEqual(diagnostics["blocked_after_repeated_no_change"], "a")
        self.assertNotEqual(encoding.plan(best)[0], "a")

    def test_backgrounds_are_excluded_but_ambiguous_edge_components_remain_testable(self):
        frame = tuple(tuple(0 for _ in range(10)) for _ in range(10))
        observation = SimpleNamespace(frame=frame)
        entities = (
            SimpleNamespace(id="bg", size=80, bounds=(0, 0, 9, 9)),
            SimpleNamespace(id="edge", size=10, bounds=(0, 0, 0, 9)),
            SimpleNamespace(id="tile", size=4, bounds=(3, 3, 4, 4)),
        )
        self.assertEqual([row.id for row in _interactive_entities(observation, entities)], ["edge", "tile"])

    def test_global_tracking_preserves_ids_when_component_order_changes(self):
        frame1 = tuple(tuple(1 if (x, y) == (1, 1) else 2 if (x, y) == (6, 6) else 0
                             for x in range(8)) for y in range(8))
        first = _tracked_entities(frame1, tuple())
        frame2 = tuple(tuple(1 if (x, y) == (6, 5) else 2 if (x, y) == (1, 2) else 0
                             for x in range(8)) for y in range(8))
        second = _tracked_entities(frame2, first)
        first_by_color = {row.color: row.id for row in first if row.color in {1, 2}}
        second_by_color = {row.color: row.id for row in second if row.color in {1, 2}}
        self.assertEqual(first_by_color, second_by_color)

    def test_persistent_procedure_reuses_reasoning_and_skips_objective(self):
        class Caller:
            model = "test-model"
            reasoning_effort = "test"

        class Reasoner(InformationGainReasoner):
            def __init__(self):
                super().__init__(Caller())
                self.calls = 0

            def propose(self, observation, history, force=False):
                self.calls += 1
                candidates = action_catalog(observation, self.entities_for(observation))
                candidate_id = candidates[0].candidate_id
                entity_id = next(row.id for row in self.entities_for(observation) if row.color != 0)
                row = TestableHypothesis(1, "the control moves the tile right", tuple(), "movement",
                    "reach the right edge", "movement stops", 1.0,
                    (Prediction(candidate_id, "MOVE", .95),), (candidate_id,),
                    "reach the right edge", "movement stops", candidate_id, entity_id, "", "MOVE_RIGHT",
                    ("MOVE",), ("NO_CHANGE", "FAIL"))
                self.catalog = HypothesisCatalog("test", (row,), candidates, candidate_id, "test")
                self.posterior = {1: 1.0}; self.contradictions = {1: 0}; self.last_call_count = 1
                return self.catalog

        class Environment:
            game_id = "test"

            def __init__(self):
                self.turn = 0

            def observation(self):
                frame = tuple(tuple(1 if (x, y) == (self.turn, 1) else 0 for x in range(4)) for y in range(4))
                return ArcObservation("test", "NOT_FINISHED", 0, 1, ("ACTION6",), frame)

            def reset(self):
                self.turn = 0
                return self.observation()

            def step(self, _candidate):
                self.turn += 1
                return self.observation()

        reasoner = Reasoner()
        trace = run_information_gain_episode(Environment(), reasoner, "llm_only", "arc", 0, 3,
                                             "test", None, 1)
        self.assertIsNone(trace.error)
        self.assertEqual(reasoner.calls, 1)
        self.assertEqual([turn.model_calls for turn in trace.turns], [1, 0, 0])
        self.assertGreater(trace.turns[0].timings["objective_seconds"], 0)
        self.assertEqual(trace.turns[1].timings["objective_seconds"], 0)
        self.assertTrue(trace.turns[1].search_diagnostics["reasoning_reused"])


if __name__ == "__main__":
    unittest.main()
