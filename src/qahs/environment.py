from __future__ import annotations

from enum import Enum

from .models import Action, GameState, Observation, ObservationKind


class HiddenRule(str, Enum):
    RED_IS_HAZARD = "red_is_hazard"
    RED_IS_SWITCH = "red_is_switch"
    RED_IS_GOAL = "red_is_goal"
    RED_DOES_NOTHING = "red_does_nothing"


class TinyRuleGame:
    """A two-action hidden-rule environment for validating the QAHS loop."""

    actions = (Action.APPROACH_RED, Action.APPROACH_GREEN)

    def __init__(self, hidden_rule: HiddenRule = HiddenRule.RED_IS_SWITCH):
        self.hidden_rule = hidden_rule
        self._state = GameState(turn=0, door_open=False, finished=False, success=False)

    @property
    def state(self) -> GameState:
        return self._state

    def reset(self) -> tuple[GameState, Observation]:
        self._state = GameState(turn=0, door_open=False, finished=False, success=False)
        return self._state, Observation(
            ObservationKind.START,
            "A blue player sees a red tile and a closed green door.",
        )

    def step(self, action: Action) -> tuple[GameState, Observation]:
        if self._state.finished:
            raise RuntimeError("The episode has already finished")

        door_open = self._state.door_open
        finished = False
        success = False

        if action == Action.APPROACH_RED:
            if self.hidden_rule == HiddenRule.RED_IS_HAZARD:
                obs = Observation(ObservationKind.PLAYER_DESTROYED, "The red tile destroyed the player.")
                finished = True
            elif self.hidden_rule == HiddenRule.RED_IS_SWITCH:
                obs = Observation(ObservationKind.DOOR_OPENED, "Touching red opened the green door.")
                door_open = True
            elif self.hidden_rule == HiddenRule.RED_IS_GOAL:
                obs = Observation(ObservationKind.RED_GOAL_REACHED, "Touching red completed the game.")
                finished = success = True
            else:
                obs = Observation(ObservationKind.NOTHING_HAPPENED, "Touching red caused no visible change.")
        else:
            if door_open or self.hidden_rule == HiddenRule.RED_IS_HAZARD:
                obs = Observation(ObservationKind.GREEN_GOAL_REACHED, "The player reached green and completed the game.")
                finished = success = True
            else:
                obs = Observation(ObservationKind.GREEN_BLOCKED, "The closed door blocked the player.")

        self._state = GameState(
            turn=self._state.turn + 1,
            door_open=door_open,
            finished=finished,
            success=success,
        )
        return self._state, obs

