from __future__ import annotations

from .models import Action, GameState, Hypothesis, ObservationKind, RedEffect


def predict(hypothesis: Hypothesis, state: GameState, action: Action) -> ObservationKind:
    """Return the deterministic outcome predicted by one candidate world model."""
    if action == Action.APPROACH_RED:
        return {
            RedEffect.HAZARD: ObservationKind.PLAYER_DESTROYED,
            RedEffect.SWITCH: ObservationKind.DOOR_OPENED,
            RedEffect.GOAL: ObservationKind.RED_GOAL_REACHED,
            RedEffect.NOTHING: ObservationKind.NOTHING_HAPPENED,
        }[hypothesis.red_effect]

    if state.door_open:
        return ObservationKind.GREEN_GOAL_REACHED
    if hypothesis.red_effect == RedEffect.HAZARD:
        return ObservationKind.GREEN_GOAL_REACHED
    return ObservationKind.GREEN_BLOCKED


def progress_value(hypothesis: Hypothesis, state: GameState, action: Action) -> float:
    """Small goal-progress term used after uncertainty has been resolved."""
    outcome = predict(hypothesis, state, action)
    if outcome in (ObservationKind.GREEN_GOAL_REACHED, ObservationKind.RED_GOAL_REACHED):
        return 1.0
    if outcome == ObservationKind.DOOR_OPENED:
        return 0.55
    if outcome == ObservationKind.PLAYER_DESTROYED:
        return -1.0
    return 0.0

