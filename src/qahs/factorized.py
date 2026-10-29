from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import product
from typing import Any


class Dimension(str, Enum):
    PLAYER = "player"
    GOAL = "goal"
    MOVEMENT = "movement"
    TOPOLOGY = "topology"
    INTERACTION = "interaction"
    STATE = "state"


DIMENSIONS = tuple(Dimension)
RULE_OPTIONS: dict[Dimension, tuple[str, str]] = {
    Dimension.PLAYER: ("blue_is_controlled", "nearest_object_is_controlled"),
    Dimension.GOAL: ("reach_green", "reach_red"),
    Dimension.MOVEMENT: ("move_one_cell", "slide_until_obstacle"),
    Dimension.TOPOLOGY: ("gray_is_wall", "gray_has_passable_gap"),
    Dimension.INTERACTION: ("red_is_hazard", "red_is_switch"),
    Dimension.STATE: ("stateless", "sequence_dependent"),
}


class ProbeAction(str, Enum):
    TEST_PLAYER = "test_player"
    TEST_GOAL = "test_goal"
    TEST_MOVEMENT = "test_movement"
    TEST_TOPOLOGY = "test_topology"
    TEST_INTERACTION = "test_interaction"
    TEST_STATE = "test_state"
    ATTEMPT_SOLUTION = "attempt_solution"


ACTION_DIMENSION: dict[ProbeAction, Dimension] = {
    ProbeAction.TEST_PLAYER: Dimension.PLAYER,
    ProbeAction.TEST_GOAL: Dimension.GOAL,
    ProbeAction.TEST_MOVEMENT: Dimension.MOVEMENT,
    ProbeAction.TEST_TOPOLOGY: Dimension.TOPOLOGY,
    ProbeAction.TEST_INTERACTION: Dimension.INTERACTION,
    ProbeAction.TEST_STATE: Dimension.STATE,
}


@dataclass(frozen=True)
class FactorizedHypothesis:
    id: int
    bits: tuple[int, ...]
    rules: dict[str, str]

    @property
    def description(self) -> str:
        return "; ".join(self.rules.values())


@dataclass(frozen=True)
class FactorizedState:
    turn: int
    observations: dict[str, int]
    finished: bool
    success: bool
    grid: tuple[str, ...] = (".....", ".B.R.", "#####", "..G..", ".....")


@dataclass(frozen=True)
class FactorizedObservation:
    token: str
    message: str


@dataclass
class FactorizedTurn:
    turn: int
    state_before: FactorizedState
    beliefs_before: dict[int, float]
    action_scores: list[Any]
    search: Any
    observation: FactorizedObservation
    state_after: FactorizedState
    beliefs_after: dict[int, float]


@dataclass
class FactorizedTrace:
    backend: str
    hidden_hypothesis_id: int
    hypotheses: list[FactorizedHypothesis]
    turns: list[FactorizedTurn]
    success: bool
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(self)


def generate_hypotheses() -> list[FactorizedHypothesis]:
    hypotheses = []
    for bits in product((0, 1), repeat=len(DIMENSIONS)):
        identifier = sum(bit << i for i, bit in enumerate(bits))
        rules = {dimension.value: RULE_OPTIONS[dimension][bit] for dimension, bit in zip(DIMENSIONS, bits)}
        hypotheses.append(FactorizedHypothesis(identifier, tuple(bits), rules))
    return sorted(hypotheses, key=lambda h: h.id)

