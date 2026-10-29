import unittest

import numpy as np

from qahs.agent import run_episode
from qahs.compiler import build_encoding
from qahs.environment import HiddenRule, TinyRuleGame
from qahs.interpreter import RuleBasedInterpreter
from qahs.models import Action, ObservationKind
from qahs.objective import expected_information_gain
from qahs.search import ClassicalSearch, QAOASearch
from qahs.factorized import generate_hypotheses
from qahs.factorized_agent import run_factorized_episode
from qahs.factorized_objective import marginals, score_factorized_actions
from qahs.qubo import build_factorized_qubo
from qahs.rule_environment import RuleDiscoveryEnvironment
from qahs.scaling import resource_estimate


class QAHSTests(unittest.TestCase):
    def setUp(self):
        self.environment = TinyRuleGame(HiddenRule.RED_IS_SWITCH)
        state, observation = self.environment.reset()
        self.state = state
        self.hypotheses = RuleBasedInterpreter().propose(state, [observation])
        self.beliefs = {h.id: 0.25 for h in self.hypotheses}

    def test_encoding_is_three_qubits(self):
        encoding = build_encoding(self.hypotheses, self.environment.actions)
        self.assertEqual(encoding.total_bits, 3)
        index = encoding.encode(2, Action.APPROACH_GREEN)
        self.assertEqual(encoding.decode(index), (2, Action.APPROACH_GREEN))

    def test_red_action_has_more_information(self):
        red = expected_information_gain(self.hypotheses, self.beliefs, self.state, Action.APPROACH_RED)
        green = expected_information_gain(self.hypotheses, self.beliefs, self.state, Action.APPROACH_GREEN)
        self.assertAlmostEqual(red, 2.0)
        self.assertGreater(red, green)

    def test_classical_loop_solves_switch_world(self):
        trace = run_episode(TinyRuleGame(HiddenRule.RED_IS_SWITCH), ClassicalSearch())
        self.assertTrue(trace.success)
        self.assertEqual([t.search.action for t in trace.turns], [Action.APPROACH_RED, Action.APPROACH_GREEN])
        self.assertEqual(trace.turns[0].observation.kind, ObservationKind.DOOR_OPENED)
        self.assertGreater(trace.turns[0].beliefs_after[1], 0.999)

    def test_qaoa_loop_solves_switch_world(self):
        backend = QAOASearch(depth=2, restarts=3, maxiter=80, shots=512, seed=11)
        trace = run_episode(TinyRuleGame(HiddenRule.RED_IS_SWITCH), backend)
        self.assertTrue(trace.success)
        self.assertEqual(trace.turns[0].search.action, Action.APPROACH_RED)
        self.assertEqual(trace.turns[-1].search.action, Action.APPROACH_GREEN)

    def test_factorized_space_has_64_unique_world_models(self):
        hypotheses = generate_hypotheses()
        self.assertEqual(len(hypotheses), 64)
        self.assertEqual(len({h.bits for h in hypotheses}), 64)
        self.assertEqual({h.id for h in hypotheses}, set(range(64)))

    def test_factorized_qubo_is_compact_and_one_hot(self):
        hypotheses = generate_hypotheses()
        beliefs = {h.id: 1 / 64 for h in hypotheses}
        scores = score_factorized_actions(hypotheses, beliefs)
        model, encoding, valid = build_factorized_qubo(hypotheses, scores, marginals(hypotheses, beliefs))
        self.assertEqual(model.variable_count, 13)
        self.assertEqual(int(valid.sum()), 64 * 7)
        self.assertEqual(len(model.quadratic), 21)

    def test_qubo_and_ising_energies_match(self):
        hypotheses = generate_hypotheses()
        beliefs = {h.id: 1 / 64 for h in hypotheses}
        model, _, _ = build_factorized_qubo(
            hypotheses,
            score_factorized_actions(hypotheses, beliefs),
            marginals(hypotheses, beliefs),
        )
        costs = model.cost_diagonal()
        offset, z, zz = model.to_ising()
        for index in (0, 17, 413, len(costs) - 1):
            bits = np.array([(index >> i) & 1 for i in range(model.variable_count)])
            spins = 1 - 2 * bits
            energy = offset + sum(value * spins[i] for i, value in z.items())
            energy += sum(value * spins[i] * spins[j] for (i, j), value in zz.items())
            self.assertAlmostEqual(costs[index], energy)

    def test_factorized_classical_episode_solves_in_seven_actions(self):
        hypotheses = generate_hypotheses()
        trace = run_factorized_episode(RuleDiscoveryEnvironment(hypotheses[37]), ClassicalSearch())
        self.assertTrue(trace.success)
        self.assertEqual(len(trace.turns), 7)
        self.assertEqual(trace.turns[-1].search.hypothesis_id, 37)

    def test_scaling_estimate_exposes_statevector_cost(self):
        row = resource_estimate(64)
        self.assertEqual(row["hypothesis_bits"], 6)
        self.assertEqual(row["one_hot_qaoa_qubits"], 13)
        self.assertEqual(row["statevector_amplitudes"], 8192)


if __name__ == "__main__":
    unittest.main()

