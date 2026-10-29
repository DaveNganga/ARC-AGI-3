from __future__ import annotations

from .compiler import build_encoding
from .environment import TinyRuleGame
from .interpreter import Interpreter, RuleBasedInterpreter
from .models import EpisodeTrace, Observation, TurnRecord
from .objective import build_cost_diagonal, score_actions, update_beliefs
from .search.base import SearchBackend
from .search.classical import ClassicalSearch


def run_episode(
    environment: TinyRuleGame,
    backend: SearchBackend | None = None,
    interpreter: Interpreter | None = None,
    max_steps: int = 6,
) -> EpisodeTrace:
    """Run observe -> interpret -> search -> act -> update until termination."""
    backend = backend or ClassicalSearch()
    interpreter = interpreter or RuleBasedInterpreter()
    state, initial_observation = environment.reset()
    history: list[Observation] = [initial_observation]
    hypotheses = interpreter.propose(state, history)
    beliefs = {h.id: 1.0 / len(hypotheses) for h in hypotheses}
    encoding = build_encoding(hypotheses, environment.actions)
    turns: list[TurnRecord] = []

    for _ in range(max_steps):
        state_before = state
        beliefs_before = beliefs.copy()
        action_scores = score_actions(hypotheses, beliefs, state, environment.actions)
        costs, valid = build_cost_diagonal(encoding, action_scores, beliefs)
        search_result = backend.select(encoding, action_scores, costs, valid)
        state, observation = environment.step(search_result.action)
        beliefs = update_beliefs(hypotheses, beliefs, state_before, search_result.action, observation.kind)
        history.append(observation)
        turns.append(TurnRecord(
            turn=state.turn,
            state_before=state_before,
            beliefs_before=beliefs_before,
            action_scores=action_scores,
            search=search_result,
            observation=observation,
            state_after=state,
            beliefs_after=beliefs.copy(),
        ))
        if state.finished:
            break

    return EpisodeTrace(
        backend=backend.name,
        hidden_rule=environment.hidden_rule.value,
        hypotheses=hypotheses,
        turns=turns,
        success=state.success,
    )

