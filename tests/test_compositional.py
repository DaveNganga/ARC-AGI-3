from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from qahs.arc_environment import DemoArcEnvironment
from qahs.arc_models import ArcActionCandidate
from qahs.compositional import (
    ATOM_KINDS, CompositionalEncoding, CompositionalReasoner, EntityFact, SemanticAtom, Vocabulary,
    action_is_grounded, assess_trajectory, assess_transition, blocked_no_change_candidates,
    build_compositional_costs, canonical_atom_value, expand_atoms, experiment_catalog, extract_entities,
    historical_evidence_score,
)
from qahs.search import ClassicalSearch, QAOASearch
from qahs.models import ActionScore


class CompositionalHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = DemoArcEnvironment().reset()
        self.entities = extract_entities(self.observation.frame)

    def test_entity_extraction_is_deterministic_and_grounded(self) -> None:
        self.assertEqual(self.entities, extract_entities(self.observation.frame))
        self.assertGreaterEqual(len(self.entities), 2)
        self.assertEqual(self.entities[0].id, "e0")

    def test_entity_identity_survives_size_order_changes(self) -> None:
        before = (
            (0, 0, 0, 0, 0),
            (0, 9, 9, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 8, 0),
            (0, 0, 0, 0, 0),
        )
        after = (
            (0, 0, 0, 0, 0),
            (0, 9, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 8, 8, 0),
            (0, 0, 0, 0, 0),
        )
        first = extract_entities(before)
        tracked = extract_entities(after, previous=first)
        self.assertEqual(next(item.id for item in first if item.color == 9),
                         next(item.id for item in tracked if item.color == 9))
        self.assertEqual(next(item.id for item in first if item.color == 8),
                         next(item.id for item in tracked if item.color == 8))

    def test_python_mutation_survives_full_luna_vocabulary(self) -> None:
        ids = [item.id for item in self.entities]
        raw = {
            "roles": [{"entity_id": ids[0], "role": role, "prior": .9 - index / 10}
                      for index, role in enumerate(("switch", "player", "marker", "target"))],
            "relations": [{"subject_id": ids[0], "relation": relation, "target_id": ids[1], "prior": .8}
                          for relation in ("near", "behind", "controls", "tracks")],
            "effects": [{"target_id": ids[1], "effect": effect, "prior": .8}
                        for effect in ("flashes", "rotates", "shrinks", "grows")],
        }
        atoms = expand_atoms(raw, self.entities)
        for kind in ATOM_KINDS:
            group = [item for item in atoms if item.kind == kind]
            self.assertEqual(len(group), 3)
            self.assertTrue(any(item.source == "python_mutation" for item in group))

    def test_semantic_paraphrases_share_stable_identity(self) -> None:
        self.assertEqual(canonical_atom_value("role", "central divider or barrier"), "barrier")
        self.assertEqual(canonical_atom_value("role", "vertical movement wall"), "barrier")
        raw = {
            "roles": [
                {"entity_id": "e0", "role": "central divider", "prior": .5},
                {"entity_id": "e0", "role": "boundary wall", "prior": .8},
            ],
            "relations": [{"subject_id": "e0", "relation": "touching", "target_id": "e1", "prior": .5}],
            "effects": [{"target_id": "e0", "effect": "blocks movement", "prior": .5}],
        }
        atoms = expand_atoms(raw, self.entities)
        barriers = [atom for atom in atoms if atom.key == "role:e0:barrier"]
        self.assertEqual(len(barriers), 1)
        self.assertEqual(barriers[0].prior, .8)

    def test_transition_evidence_distinguishes_motion_color_change_and_reversal(self) -> None:
        before = self.observation
        moved_frame = tuple(tuple(9 if (x, y) == (2, 5) else 0 for x in range(8)) for y in range(8))
        simple_before = replace(before, frame=tuple(tuple(9 if (x, y) == (1, 5) else 0 for x in range(8)) for y in range(8)))
        moved = replace(before, frame=moved_frame)
        motion = assess_transition(simple_before, moved, [])
        self.assertEqual(motion["meaningful_change"], "novel_motion")
        self.assertEqual(motion["evidence_score"], .05)

        history = [{"before_state_signature": motion["before_state_signature"],
                    "after_state_signature": motion["after_state_signature"]}]
        reversal = assess_transition(moved, simple_before, history)
        self.assertEqual(reversal["meaningful_change"], "revisited_state")
        self.assertLess(reversal["evidence_score"], 0)
        trajectory = [motion, reversal]
        self.assertLess(historical_evidence_score(trajectory, 0), 0)

        larger = replace(before, frame=tuple(tuple(9 if (x, y) in {(1, 5), (2, 5)} else 0
                                                   for x in range(8)) for y in range(8)))
        color_change = assess_transition(simple_before, larger, [])
        self.assertEqual(color_change["meaningful_change"], "minor_color_change")
        self.assertEqual(color_change["color_changed_cells"], 1)
        self.assertGreater(color_change["evidence_score"], motion["evidence_score"])

    def test_repeated_directional_progress_is_rewarded_not_penalized(self) -> None:
        before = (EntityFact("e0", 2, 1, (1, 1, 1, 1), (1, 1)),
                  EntityFact("e1", 3, 1, (5, 1, 5, 1), (5, 1)))
        after = (EntityFact("e0", 2, 1, (2, 1, 2, 1), (2, 1)), before[1])
        atoms = (SemanticAtom("relation:e0:touches:e1", "relation", "e0", "touches", "e1", .5, "luna"),)
        action = ArcActionCandidate("move-4", "ACTION4", "right")
        first = assess_trajectory(before, after, action, atoms, [])
        self.assertEqual(first["trajectory_status"], "productive_motion")
        history = [{"candidate_id": "move-4", **first}]
        next_after = (EntityFact("e0", 2, 1, (3, 1, 3, 1), (3, 1)), before[1])
        repeated = assess_trajectory(after, next_after, action, atoms, history)
        self.assertEqual(repeated["trajectory_status"], "productive_repetition")
        self.assertGreater(repeated["trajectory_score"], 0)

    def test_blocked_repetition_is_penalized(self) -> None:
        entities = (EntityFact("e0", 2, 1, (1, 1, 1, 1), (1, 1)),)
        action = ArcActionCandidate("move-4", "ACTION4", "right")
        history = [{"candidate_id": "move-4", "tracked_entity_id": "e0", "after_center": (1, 1)}]
        result = assess_trajectory(entities, entities, action, tuple(), history)
        self.assertEqual(result["trajectory_status"], "blocked")
        self.assertLess(result["trajectory_score"], 0)

    def test_objective_uses_exactly_one_atom_per_category(self) -> None:
        atoms = (
            SemanticAtom("role:e0:key", "role", "e0", "key", None, .4, "luna"),
            SemanticAtom("role:e0:goal", "role", "e0", "goal", None, .2, "python_mutation"),
            SemanticAtom("relation:e0:touches:e1", "relation", "e0", "touches", "e1", .3, "luna"),
            SemanticAtom("effect:e1:opens", "effect", "e1", "opens", None, .3, "python_mutation"),
        )
        candidates = experiment_catalog(self.observation, self.entities)
        vocabulary = Vocabulary("test", atoms, candidates, candidates[0].candidate_id, "test")
        encoding, costs, valid, diagnostics = build_compositional_costs(vocabulary, self.entities, [])
        self.assertGreater(int(valid.sum()), 0)
        self.assertEqual(diagnostics["model_prior_weight"], .10)
        self.assertTrue(diagnostics["action_grounding_required"])
        for index in np.flatnonzero(valid):
            mask, _ = encoding.decode(int(index))
            chosen = encoding.selected_atoms(mask)
            self.assertTrue(all(sum(atom.kind == kind for atom in chosen) == 1 for kind in ATOM_KINDS))

    def test_click_must_be_grounded_in_predicted_effect_or_direct_relation(self) -> None:
        atoms = (
            SemanticAtom("role:e0:goal", "role", "e0", "goal", None, .4, "luna"),
            SemanticAtom("relation:e3:matches:e4", "relation", "e3", "matches", "e4", .4, "luna"),
            SemanticAtom("effect:e3:changes", "effect", "e3", "changes", None, .4, "luna"),
        )
        self.assertTrue(action_is_grounded(atoms, ArcActionCandidate("click-e3", "ACTION6", "click e3")))
        self.assertTrue(action_is_grounded(atoms, ArcActionCandidate("click-e4", "ACTION6", "click e4")))
        self.assertFalse(action_is_grounded(atoms, ArcActionCandidate("click-e0", "ACTION6", "click e0")))

    def test_two_no_change_clicks_block_same_candidate_in_unchanged_state(self) -> None:
        history = [{"candidate_id": "click-e0", "meaningful_change": "no_change", "levels_delta": 0,
                    "before_state_signature": "same", "after_state_signature": "same"} for _ in range(2)]
        self.assertEqual(blocked_no_change_candidates(history), {"click-e0"})

    def test_both_search_backends_select_valid_compositions(self) -> None:
        atoms = (
            SemanticAtom("role:e0:key", "role", "e0", "key", None, .3, "luna"),
            SemanticAtom("role:e0:obstacle", "role", "e0", "obstacle", None, .1, "python_mutation"),
            SemanticAtom("relation:e0:touches:e1", "relation", "e0", "touches", "e1", .3, "luna"),
            SemanticAtom("effect:e1:opens", "effect", "e1", "opens", None, .3, "python_mutation"),
        )
        candidates = experiment_catalog(self.observation, self.entities)
        vocabulary = Vocabulary("test", atoms, candidates, candidates[0].candidate_id, "test")
        encoding, costs, valid, _ = build_compositional_costs(vocabulary, self.entities, [])
        scores = [ActionScore(item.candidate_id, 0, item.expected_progress, item.expected_progress) for item in candidates]
        classical = ClassicalSearch().select(encoding, scores, costs, valid)
        quantum = QAOASearch(depth=1, restarts=1, maxiter=8, shots=128, seed=3).select(encoding, scores, costs, valid)
        self.assertTrue(valid[classical.encoded_index])
        self.assertTrue(valid[quantum.encoded_index])
        self.assertEqual(classical.diagnostics["regret"], 0)

    def test_success_audit_revises_hypothesis_and_preserves_transition_images(self) -> None:
        class AuditCaller:
            model = "test-model"
            reasoning_effort = "low"

            def __init__(self) -> None:
                self.images = []

            def call_json(self, _prompt, _schema, name, images=None):
                self.images = images or []
                self.assert_name = name
                return {
                    "verdict": "partially_correct",
                    "original_hypothesis": "e0 is a player",
                    "predicted_effect": "e0 moves",
                    "observed_effect": "e0 changed color and the level advanced",
                    "supported_atom_keys": ["role:e0:player"],
                    "contradicted_atom_keys": ["effect:e0:moves", "not-a-selected-key"],
                    "corrected_hypothesis": "transform the controllable object to match the target",
                    "transferable_relation": "controllable object matches target",
                    "interface_change": "the edge status strip increased",
                    "next_level_test": "identify a new controllable object and target",
                    "alternatives": ["the repeated count may matter"],
                    "confidence": .8,
                }

        caller = AuditCaller()
        reasoner = CompositionalReasoner(caller)
        before = self.observation
        after = replace(before, levels_completed=1, frame=tuple(tuple(14 if (x, y) == (0, 0) else value
                        for x, value in enumerate(row)) for y, row in enumerate(before.frame)))
        reasoner.begin_level(before)
        atoms = (
            SemanticAtom("role:e0:player", "role", "e0", "player", None, .6, "luna"),
            SemanticAtom("effect:e0:moves", "effect", "e0", "moves", None, .6, "luna"),
        )
        selected = ArcActionCandidate("click-e0", "ACTION6", "click e0", x=1, y=5)
        audit = reasoner.audit_success(before, after, "e0 moves", atoms, selected, [])
        self.assertEqual(audit.verdict, "partially_correct")
        self.assertEqual(audit.supported_atom_keys, ("role:e0:player",))
        self.assertEqual(audit.contradicted_atom_keys, ("effect:e0:moves",))
        self.assertIn("status strip", audit.interface_change)
        self.assertEqual(caller.assert_name, "arc_success_hypothesis_audit")
        self.assertGreaterEqual(len(caller.images), 2)

    def test_success_audit_component_credit_overrides_blanket_progress_credit(self) -> None:
        atoms = (
            SemanticAtom("role:e0:player", "role", "e0", "player", None, .3, "luna"),
            SemanticAtom("relation:e0:touches:e1", "relation", "e0", "touches", "e1", .3, "luna"),
            SemanticAtom("effect:e1:opens", "effect", "e1", "opens", None, .3, "luna"),
        )
        candidates = experiment_catalog(self.observation, self.entities)
        vocabulary = Vocabulary("test", atoms, candidates, candidates[0].candidate_id, "test")
        history = [{
            "selected_atom_keys": [atom.key for atom in atoms],
            "candidate_id": candidates[0].candidate_id,
            "levels_delta": 1,
            "evidence_score": 2.0,
            "hypothesis_credit": .75,
            "atom_evidence": {atoms[0].key: 2.0, atoms[1].key: -1.0, atoms[2].key: .15},
        }]
        _, _, _, diagnostics = build_compositional_costs(vocabulary, self.entities, history)
        self.assertEqual(diagnostics["cost_model"], "success-audited-action-conditioned-v4")


if __name__ == "__main__":
    unittest.main()
