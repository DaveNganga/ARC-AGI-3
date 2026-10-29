from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Protocol

from .models import GameState, Hypothesis, Observation, RedEffect


class Interpreter(Protocol):
    def propose(self, state: GameState, history: Sequence[Observation]) -> list[Hypothesis]: ...


class RuleBasedInterpreter:
    """Deterministic stand-in so the search experiment does not require an API."""

    def propose(self, state: GameState, history: Sequence[Observation]) -> list[Hypothesis]:
        del state, history
        return [
            Hypothesis(0, "Red is a hazard", RedEffect.HAZARD, "Contact with red destroys the player."),
            Hypothesis(1, "Red is a switch", RedEffect.SWITCH, "Contact with red opens the green door."),
            Hypothesis(2, "Red is the goal", RedEffect.GOAL, "Contact with red completes the game."),
            Hypothesis(3, "Red is irrelevant", RedEffect.NOTHING, "Contact with red causes no change."),
        ]


class StructuredLLMInterpreter:
    """API-ready adapter. ``call_model`` receives a prompt and returns JSON text."""

    def __init__(self, call_model: Callable[[str], str]):
        self.call_model = call_model

    def propose(self, state: GameState, history: Sequence[Observation]) -> list[Hypothesis]:
        prompt = (
            "Return JSON with a 'hypotheses' list. Each item must contain name, "
            "red_effect (hazard, switch, goal, or nothing), and description. "
            f"State={state}. History={[o.message for o in history]}"
        )
        payload = json.loads(self.call_model(prompt))
        rows = payload.get("hypotheses", [])
        if not 1 <= len(rows) <= 16:
            raise ValueError("The interpreter must return between 1 and 16 hypotheses")
        return [
            Hypothesis(i, str(row["name"]), RedEffect(row["red_effect"]), str(row["description"]))
            for i, row in enumerate(rows)
        ]

