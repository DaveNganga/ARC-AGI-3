from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Action(str, Enum):
    APPROACH_RED = "approach_red"
    APPROACH_GREEN = "approach_green"


class RedEffect(str, Enum):
    HAZARD = "hazard"
    SWITCH = "switch"
    GOAL = "goal"
    NOTHING = "nothing"


class ObservationKind(str, Enum):
    START = "start"
    PLAYER_DESTROYED = "player_destroyed"
    DOOR_OPENED = "door_opened"
    RED_GOAL_REACHED = "red_goal_reached"
    NOTHING_HAPPENED = "nothing_happened"
    GREEN_BLOCKED = "green_blocked"
    GREEN_GOAL_REACHED = "green_goal_reached"


@dataclass(frozen=True)
class Hypothesis:
    id: int
    name: str
    red_effect: RedEffect
    description: str


@dataclass(frozen=True)
class GameState:
    turn: int
    door_open: bool
    finished: bool
    success: bool


@dataclass(frozen=True)
class Observation:
    kind: ObservationKind
    message: str


@dataclass(frozen=True)
class ActionScore:
    action: Action
    information_gain: float
    expected_progress: float
    utility: float
    novelty_bonus: float = 0.0
    outcome_bonus: float = 0.0
    repetition_penalty: float = 0.0


@dataclass
class SearchResult:
    backend: str
    action: Action
    hypothesis_id: int
    utility: float
    encoded_index: int
    diagnostics: dict[str, Any]


@dataclass
class TurnRecord:
    turn: int
    state_before: GameState
    beliefs_before: dict[int, float]
    action_scores: list[ActionScore]
    search: SearchResult
    observation: Observation
    state_after: GameState
    beliefs_after: dict[int, float]


@dataclass
class EpisodeTrace:
    backend: str
    hidden_rule: str
    hypotheses: list[Hypothesis]
    turns: list[TurnRecord]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(self)

