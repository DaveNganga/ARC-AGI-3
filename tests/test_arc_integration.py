import unittest

from qahs.arc_agent import run_arc_comparison
from qahs.arc_environment import DemoArcEnvironment, validate_arc_game_id
from qahs.arc_models import ArcObservation
from qahs.arc_objective import action_signature, score_arc_actions
from qahs.arc_reasoner import DemoArcReasoner, frame_summary, parse_proposal


class ArcIntegrationTests(unittest.TestCase):
    def test_game_id_validation(self):
        self.assertEqual(validate_arc_game_id("ft09"), "ft09")
        self.assertEqual(validate_arc_game_id("ls20-1234abcd"), "ls20-1234abcd")
        with self.assertRaises(ValueError):
            validate_arc_game_id("../../secret")

    def test_demo_comparison_runs_with_and_without_qahs(self):
        trace = run_arc_comparison(
            lambda: DemoArcEnvironment(seed=0),
            DemoArcReasoner,
            source="demo",
            game_id=DemoArcEnvironment.game_id,
            seed=0,
            max_steps=6,
        )
        self.assertTrue(trace.without_qahs.success)
        self.assertEqual(trace.action_budget, 6)
        self.assertEqual(trace.model, "Deterministic local reasoner")
        self.assertTrue(trace.with_classical.success)
        self.assertTrue(trace.with_qahs.success)
        self.assertEqual(len(trace.without_qahs.turns), 3)
        self.assertEqual(len(trace.with_classical.turns), 2)
        self.assertEqual(len(trace.with_qahs.turns), 2)
        self.assertFalse(trace.without_qahs.turns[0].search_diagnostics["quantum_used"])
        self.assertFalse(trace.with_classical.turns[0].search_diagnostics["quantum_used"])
        self.assertEqual(trace.with_classical.turns[0].search_diagnostics["backend"], "classical_exact")
        self.assertTrue(trace.with_qahs.turns[0].search_diagnostics["quantum_used"])

    def test_information_seeking_candidate_scores_above_wander(self):
        environment = DemoArcEnvironment()
        proposal = DemoArcReasoner().propose(environment.reset(), [])
        scores = {row.action: row.utility for row in score_arc_actions(proposal)}
        self.assertGreater(scores["probe"], scores["wander"])

    def test_parser_drops_unavailable_action(self):
        payload = {
            "summary": "test",
            "hypotheses": [{
                "statement": "rule",
                "confidence": 1.0,
                "predictions": [
                    {"candidate_id": "valid", "outcome": "moves"},
                    {"candidate_id": "invalid", "outcome": "unknown"},
                ],
            }],
            "action_candidates": [
                {"candidate_id": "valid", "action_type": "ACTION1", "description": "up", "x": None, "y": None, "expected_progress": 0.2, "risk": 0.0},
                {"candidate_id": "invalid", "action_type": "ACTION7", "description": "undo", "x": None, "y": None, "expected_progress": 0.0, "risk": 0.0},
            ],
            "recommended_candidate_id": "invalid",
            "recommendation_reason": "test",
        }
        proposal = parse_proposal(payload, ("ACTION1",))
        self.assertEqual([candidate.candidate_id for candidate in proposal.action_candidates], ["valid"])
        self.assertEqual(proposal.recommended_candidate_id, "valid")

    def test_frame_summary_keeps_shape_and_values(self):
        observation = ArcObservation("x", "NOT_FINISHED", 0, 0, ("ACTION1",), ((0, 0, 2), (3, 3, 3)))
        summary = frame_summary(observation.frame)
        self.assertIn("shape=2x3", summary)
        self.assertIn("coordinates use x=column,y=row", summary)
        self.assertIn("x2=2", summary)
        self.assertIn("color 3: count=3", summary)

    def test_repeated_no_change_action_is_penalized(self):
        environment = DemoArcEnvironment()
        observation = environment.reset()
        proposal = DemoArcReasoner().propose(observation, [])
        probe = next(candidate for candidate in proposal.action_candidates if candidate.candidate_id == "probe")
        history = [{
            "action_signature": action_signature(probe),
            "before_frame_signature": "same-frame",
            "changed_cells": 0,
            "levels_delta": 0,
        }]
        scores = {row.action: row for row in score_arc_actions(proposal, history, "same-frame")}
        self.assertGreater(scores["probe"].repetition_penalty, 0.0)
        self.assertGreater(scores["wander"].utility, scores["probe"].utility)


if __name__ == "__main__":
    unittest.main()
