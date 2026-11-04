import unittest
from tempfile import TemporaryDirectory

from qahs.arc_agent import movement_persistence_is_safe, run_arc_comparison, safe_override_choice, save_arc_trace
from qahs.arc_encoding import ArcEncoding
from qahs.arc_environment import DemoArcEnvironment, validate_arc_game_id
from qahs.arc_models import ArcActionCandidate, ArcGroundedGoal, ArcHypothesis, ArcObservation, ArcProposal, INPUT_REPRESENTATION
from qahs.arc_objective import action_signature, build_arc_costs, canonical_families, grounded_goal_score, grounded_goal_tokens, score_arc_actions
from qahs.arc_reasoner import (
    DemoArcReasoner,
    compact_plan_catalog,
    frame_image_data_url,
    frame_summary,
    generate_plan_catalog,
    lossless_frame_data,
    merge_reflection,
    parse_proposal,
    parse_reflection,
)


class ArcIntegrationTests(unittest.TestCase):
    def test_movement_persistence_requires_productive_predictable_movement(self):
        self.assertTrue(movement_persistence_is_safe("ACTION3", 109, 0, False, True))
        self.assertFalse(movement_persistence_is_safe("ACTION6", 109, 0, False, True))
        self.assertFalse(movement_persistence_is_safe("ACTION3", 0, 0, False, True))
        self.assertFalse(movement_persistence_is_safe("ACTION3", 109, 1, False, True))
        self.assertFalse(movement_persistence_is_safe("ACTION3", 109, 0, True, True))
        self.assertFalse(movement_persistence_is_safe("ACTION3", 109, 0, False, False))
        self.assertFalse(movement_persistence_is_safe("ACTION3", 109, 0, False, True, strategy_reset=True))

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
        self.assertEqual(trace.harness_version, "v7-grounded-goals")
        self.assertEqual(trace.input_representation, INPUT_REPRESENTATION)
        self.assertIsNone(trace.batch_id)
        self.assertTrue(trace.with_classical.success)
        self.assertTrue(trace.with_qahs.success)
        self.assertEqual(len(trace.without_qahs.turns), 3)
        self.assertEqual(len(trace.with_classical.turns), 2)
        self.assertEqual(len(trace.with_qahs.turns), 2)
        self.assertFalse(trace.without_qahs.turns[0].search_diagnostics["quantum_used"])
        self.assertFalse(trace.with_classical.turns[0].search_diagnostics["quantum_used"])
        self.assertEqual(trace.with_classical.turns[0].search_diagnostics["backend"], "classical_exact")
        self.assertTrue(trace.with_qahs.turns[0].search_diagnostics["quantum_used"])
        self.assertIn("regret", trace.with_qahs.turns[0].search_diagnostics)
        self.assertIn("exact_optimum_probability", trace.with_qahs.turns[0].search_diagnostics)
        self.assertIn("selection_seconds", trace.with_qahs.turns[0].timings)

    def test_versioned_batch_filename_matches_trace_metadata(self):
        trace = run_arc_comparison(
            lambda: DemoArcEnvironment(seed=1),
            DemoArcReasoner,
            source="demo",
            game_id=DemoArcEnvironment.game_id,
            seed=1,
            max_steps=4,
            batch_id="batch-01",
        )
        with TemporaryDirectory() as directory:
            path = save_arc_trace(trace, directory)
            self.assertIn("__v7-grounded-goals__batch-01__demo-hidden-door__seed-01__actions-4__", path.name)

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

    def test_adaptive_grid_encoding_is_lossless_and_uses_shorter_form(self):
        sparse_frame = [[0 for _ in range(64)] for _ in range(64)]
        sparse_frame[0][2] = 2
        sparse_mode, sparse = lossless_frame_data(tuple(tuple(row) for row in sparse_frame))
        self.assertEqual(sparse_mode, "adaptive-sparse")
        self.assertIn("default=0", sparse)
        self.assertIn("2,0:2", sparse)
        dense_frame = tuple(tuple((x + y) % 2 for x in range(8)) for y in range(8))
        dense_mode, dense = lossless_frame_data(dense_frame)
        self.assertEqual(dense_mode, "adaptive-matrix")
        self.assertIn("rows=", dense)

    def test_visual_input_and_two_step_catalog_are_generated(self):
        observation = DemoArcEnvironment().reset()
        image = frame_image_data_url(observation.frame)
        self.assertIsNotNone(image)
        self.assertTrue(image.startswith("data:image/png;base64,"))
        plans = generate_plan_catalog(observation)
        self.assertGreaterEqual(len(plans), len(observation.available_actions) ** 2)
        self.assertLessEqual(len(plans), 32)
        self.assertTrue(all(plan.next_action_type for plan in plans))
        compact = compact_plan_catalog(plans)
        self.assertLessEqual(len(compact["actions"]), 8)
        self.assertEqual(len(compact["plans"]), len(plans))
        self.assertEqual(set(compact["plans"][0]), {"candidate_id", "actions"})

    def test_reflection_delta_preserves_rule_fields(self):
        proposal = DemoArcReasoner().propose(DemoArcEnvironment().reset(), [])
        reflection = merge_reflection([], proposal, {
            "summary": "Probe changed confidence.", "observed_effect": "No change.",
            "beliefs": [{"id": 0, "status": "weakened", "confidence_delta": -0.2, "evidence": "No effect."}],
            "next_focus": "Try another action.",
        })
        self.assertAlmostEqual(reflection.beliefs[0].confidence, 0.3)
        self.assertEqual(reflection.beliefs[0].statement, proposal.hypotheses[0].statement)
        self.assertEqual(len(reflection.beliefs), len(proposal.hypotheses))

    def test_history_changes_hypothesis_action_costs(self):
        environment = DemoArcEnvironment()
        proposal = DemoArcReasoner().propose(environment.reset(), [])
        encoding = ArcEncoding(proposal.hypotheses, proposal.action_candidates)
        scores = score_arc_actions(proposal)
        history = [{"belief_ledger": [
            {"statement": "The object is a switch", "confidence": 0.1, "status": "rejected"},
            {"statement": "The object is a hazard", "confidence": 0.9, "status": "supported"},
        ]}]
        costs, valid, diagnostics = build_arc_costs(encoding, proposal, scores, history)
        switch_probe = encoding.encode(0, "probe")
        hazard_probe = encoding.encode(1, "probe")
        self.assertTrue(valid[switch_probe])
        self.assertLess(costs[hazard_probe], costs[switch_probe])
        self.assertEqual(diagnostics["cost_model"], "goal-aligned-hypothesis-action-v2")
        self.assertGreater(diagnostics["interaction_max"], diagnostics["interaction_min"])

    def test_v6_stagnation_prefers_goal_aligned_hypothesis(self):
        candidate = ArcActionCandidate("touch", "ACTION5", "interact with target", expected_progress=0.4)
        proposal = ArcProposal("test", (
            ArcHypothesis(0, "Movement controls the actor", 1.0, {"touch": "CHANGE"},
                          rule="directions move actor", role="mechanism"),
            ArcHypothesis(1, "The yellow target completes the level", 0.2, {"touch": "PROGRESS"},
                          goal_condition="interact with yellow target to complete level", role="goal"),
        ), (candidate,), "touch", "test")
        goal_family = canonical_families(proposal)[1]
        history = [{"levels_delta": 0, "selected_role": "goal", "goal_family": goal_family} for _ in range(7)]
        encoding = ArcEncoding(proposal.hypotheses, proposal.action_candidates)
        costs, _, diagnostics = build_arc_costs(encoding, proposal, score_arc_actions(proposal), history, max_steps=40)
        self.assertLess(costs[encoding.encode(1, "touch")], costs[encoding.encode(0, "touch")])
        self.assertGreater(diagnostics["stagnation_pressure"], 0)
        self.assertEqual(diagnostics["active_goal_family"], goal_family)

    def test_v6_groups_semantic_duplicate_rules(self):
        candidate = ArcActionCandidate("move", "ACTION1", "move")
        proposal = ArcProposal("test", (
            ArcHypothesis(0, "Up moves the actor", .8, {"move": "MOVE"}, rule="ACTION1 moves actor upward", role="mechanism"),
            ArcHypothesis(1, "The actor moves up", .7, {"move": "MOVE"}, rule="ACTION1 moves the actor upward", role="mechanism"),
        ), (candidate,), "move", "test")
        families = canonical_families(proposal)
        self.assertEqual(families[0], families[1])

    def test_v7_grounded_goal_identity_survives_paraphrase(self):
        candidate = ArcActionCandidate("down", "ACTION2", "move down", expected_progress=.8)
        first = ArcProposal("first", (
            ArcHypothesis(0, "Align gray with yellow", .8, {"down": "PROGRESS"}, role="goal",
                          grounded_goal=ArcGroundedGoal("gray L", "gray", "x51-59 y24-32", "yellow x51-59 y45-53",
                                                       "exact overlap", "ACTION2", "horizontal alignment", "distance decreases")),
        ), (candidate,), "down", "align")
        family = canonical_families(first)[0]
        history = [{"goal_family": family, "goal_signature_tokens": sorted(grounded_goal_tokens(first.hypotheses[0]))}]
        second = ArcProposal("second", (
            ArcHypothesis(0, "Put the grey piece onto its gold footprint", .8, {"down": "PROGRESS"}, role="goal",
                          grounded_goal=ArcGroundedGoal("grey L", "grey", "x51-59 y27-35", "yellow x51-59 y45-53",
                                                       "overlap target", "ACTION2", "same column", "gap shrinks")),
        ), (candidate,), "down", "continue")
        self.assertEqual(canonical_families(second, history)[0], family)
        self.assertGreater(grounded_goal_score(second.hypotheses[0]), .75)

    def test_v7_blocks_weak_override_of_grounded_model_plan(self):
        model = ArcActionCandidate("model", "ACTION2", "continue alignment", expected_progress=.8)
        alternate = ArcActionCandidate("alternate", "ACTION3", "explore left", expected_progress=.5)
        goal = ArcHypothesis(0, "Align gray with yellow", .8, {"model": "PROGRESS", "alternate": "MOVE"}, role="goal",
                             grounded_goal=ArcGroundedGoal("gray L", "gray", "x51 y24", "yellow x51 y45",
                                                          "exact overlap", "ACTION2", "same column", "distance decreases"))
        proposal = ArcProposal("test", (goal,), (model, alternate), "model", "direct alignment")
        encoding = ArcEncoding(proposal.hypotheses, proposal.action_candidates)
        valid = encoding.valid_mask()
        costs = __import__("numpy").array([-1.0, -1.2])
        flags = {"grounded_goals": True, "override_safety": True, "commitment_window": True, "shadow_mode": False}
        choice, diagnostics, _ = safe_override_choice(proposal, encoding, costs, valid, "alternate", {}, [], flags, None)
        self.assertEqual(choice, "model")
        self.assertEqual(diagnostics["override_category"], "blocked")
        costs = __import__("numpy").array([-1.0, -1.8])
        choice, diagnostics, _ = safe_override_choice(proposal, encoding, costs, valid, "alternate", {}, [], flags, None)
        self.assertEqual(choice, "alternate")
        self.assertTrue(diagnostics["override_permitted"])

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

    def test_reflection_builds_a_persistent_belief_ledger(self):
        reflection = parse_reflection({
            "summary": "The action supported the switch rule.",
            "observed_effect": "A door opened.",
            "beliefs": [{
                "id": 4,
                "statement": "The object is a switch.",
                "confidence": 1.4,
                "status": "supported",
                "evidence": "The door changed after contact.",
            }],
            "next_focus": "Try the opened path.",
        })
        self.assertEqual(reflection.beliefs[0].id, 0)
        self.assertEqual(reflection.beliefs[0].confidence, 1.0)
        self.assertEqual(reflection.beliefs[0].status, "supported")


if __name__ == "__main__":
    unittest.main()
