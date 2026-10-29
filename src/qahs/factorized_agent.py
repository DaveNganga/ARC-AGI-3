from __future__ import annotations

from time import perf_counter

from .factorized import FactorizedTrace, FactorizedTurn, ProbeAction, generate_hypotheses
from .factorized_objective import marginals, score_factorized_actions, update_factorized_beliefs
from .qubo import build_factorized_qubo
from .rule_environment import RuleDiscoveryEnvironment
from .search.base import SearchBackend
from .search.classical import ClassicalSearch


def run_factorized_episode(
    environment: RuleDiscoveryEnvironment,
    backend: SearchBackend | None = None,
    max_steps: int = 10,
) -> FactorizedTrace:
    """Run a matched 64-hypothesis experiment with a QUBO-compatible objective."""
    started = perf_counter()
    backend = backend or ClassicalSearch()
    hypotheses = generate_hypotheses()
    beliefs = {hypothesis.id: 1.0 / len(hypotheses) for hypothesis in hypotheses}
    state, _ = environment.reset()
    turns = []

    for _ in range(max_steps):
        state_before = state
        beliefs_before = beliefs.copy()
        scores = score_factorized_actions(hypotheses, beliefs)
        model, encoding, valid = build_factorized_qubo(
            hypotheses, scores, marginals(hypotheses, beliefs)
        )
        costs = model.cost_diagonal()
        result = backend.select(encoding, scores, costs, valid)
        result.diagnostics.update(model.resources(getattr(backend, "depth", 1)))
        state, observation = environment.step(
            result.action, proposed_hypothesis_id=result.hypothesis_id
        )
        beliefs = update_factorized_beliefs(
            hypotheses,
            beliefs,
            result.action,
            observation,
            proposed_hypothesis_id=result.hypothesis_id,
        )
        turns.append(FactorizedTurn(
            turn=state.turn,
            state_before=state_before,
            beliefs_before=beliefs_before,
            action_scores=scores,
            search=result,
            observation=observation,
            state_after=state,
            beliefs_after=beliefs.copy(),
        ))
        if state.finished:
            break

    return FactorizedTrace(
        backend=backend.name,
        hidden_hypothesis_id=environment.hidden_hypothesis.id,
        hypotheses=hypotheses,
        turns=turns,
        success=state.success,
        wall_seconds=perf_counter() - started,
    )

