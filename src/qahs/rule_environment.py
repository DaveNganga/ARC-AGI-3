from __future__ import annotations

from .factorized import (
    ACTION_DIMENSION,
    DIMENSIONS,
    FactorizedHypothesis,
    FactorizedObservation,
    FactorizedState,
    ProbeAction,
)


class RuleDiscoveryEnvironment:
    """ARC-like controlled environment with a hidden six-rule world model."""

    actions = tuple(ProbeAction)

    def __init__(self, hidden_hypothesis: FactorizedHypothesis):
        self.hidden_hypothesis = hidden_hypothesis
        self._state = FactorizedState(0, {}, False, False)

    @property
    def state(self) -> FactorizedState:
        return self._state

    def reset(self) -> tuple[FactorizedState, FactorizedObservation]:
        self._state = FactorizedState(0, {}, False, False)
        return self._state, FactorizedObservation(
            "start", "A colored grid is visible, but its six rule dimensions are unknown."
        )

    def step(
        self, action: ProbeAction, proposed_hypothesis_id: int | None = None
    ) -> tuple[FactorizedState, FactorizedObservation]:
        if self._state.finished:
            raise RuntimeError("The episode has already finished")
        observations = dict(self._state.observations)
        success = False
        finished = False
        if action == ProbeAction.ATTEMPT_SOLUTION:
            success = proposed_hypothesis_id == self.hidden_hypothesis.id
            finished = success
            token = "solution:correct" if success else "solution:incorrect"
            message = "The proposed world model solved the environment." if success else "The proposed world model was rejected."
        else:
            dimension = ACTION_DIMENSION[action]
            position = DIMENSIONS.index(dimension)
            value = self.hidden_hypothesis.bits[position]
            observations[dimension.value] = value
            token = f"{dimension.value}:{value}"
            message = f"The experiment revealed {dimension.value} option {value}."
        self._state = FactorizedState(self._state.turn + 1, observations, finished, success)
        return self._state, FactorizedObservation(token, message)

