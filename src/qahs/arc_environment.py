from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Protocol

from .arc_models import ArcActionCandidate, ArcObservation


GAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{4}(?:-[A-Za-z0-9]{1,32})?$")


def validate_arc_game_id(value: str) -> str:
    game_id = value.strip()
    if not GAME_ID_PATTERN.fullmatch(game_id):
        raise ValueError("Game ID must be four letters or numbers, optionally followed by a version suffix.")
    return game_id


class ArcEnvironment(Protocol):
    game_id: str

    def reset(self) -> ArcObservation: ...
    def step(self, candidate: ArcActionCandidate) -> ArcObservation: ...


def _state_name(value: Any) -> str:
    return str(getattr(value, "name", getattr(value, "value", value)))


def _last_grid(raw_frame: Any) -> tuple[tuple[int, ...], ...]:
    if raw_frame is None or len(raw_frame) == 0:
        return tuple()
    layer = raw_frame[-1]
    if hasattr(layer, "tolist"):
        layer = layer.tolist()
    return tuple(tuple(int(value) for value in row) for row in layer)


def _available_names(raw: Any, env: Any) -> tuple[str, ...]:
    actions = list(getattr(env, "action_space", []) or [])
    if not actions:
        try:
            from arcengine import GameAction

            actions = [GameAction.from_id(int(value)) for value in getattr(raw, "available_actions", [])]
        except Exception:
            actions = []
    names = []
    for action in actions:
        name = str(getattr(action, "name", action)).upper()
        if name != "RESET" and name not in names:
            names.append(name)
    return tuple(names)


class ArcSDKEnvironment:
    """Thin adapter around the official ARC-AGI-3 Python toolkit."""

    def __init__(self, game_id: str, seed: int = 0, recordings_dir: str | Path = "output/arc_recordings"):
        game_id = validate_arc_game_id(game_id)
        try:
            import arc_agi
        except ImportError as error:
            raise RuntimeError("Install ARC support with: pip install -e '.[arc]'") from error

        self.game_id = game_id
        self.seed = seed
        self._last: ArcObservation | None = None
        arc_api_key = os.environ.get("ARC_API_KEY", "")
        if not arc_api_key:
            raise RuntimeError(
                "Set ARC_API_KEY before using the live ARC-AGI-3 service. "
                "The service returned HTTP 401 when anonymous access was verified."
            )
        self.arcade = arc_agi.Arcade(
            arc_api_key=arc_api_key,
            recordings_dir=str(recordings_dir),
        )
        self.env = self.arcade.make(
            game_id,
            seed=seed,
            save_recording=True,
            include_frame_data=True,
            render_mode=None,
        )
        if self.env is None:
            raise RuntimeError(f"ARC-AGI-3 game '{game_id}' is unavailable. Check the game ID and ARC_API_KEY.")

    def _convert(self, raw: Any, event: str) -> ArcObservation:
        if raw is None:
            raise RuntimeError("ARC-AGI-3 returned no observation")
        frame = _last_grid(getattr(raw, "frame", []))
        return ArcObservation(
            game_id=str(getattr(raw, "game_id", self.game_id)),
            state=_state_name(getattr(raw, "state", "UNKNOWN")),
            levels_completed=int(getattr(raw, "levels_completed", 0)),
            win_levels=int(getattr(raw, "win_levels", 0)),
            available_actions=_available_names(raw, self.env),
            frame=frame,
            raw_frame_count=len(getattr(raw, "frame", []) or []),
            event=event,
        )

    def reset(self) -> ArcObservation:
        raw = self.env.reset()
        self._last = self._convert(raw, "reset")
        return self._last

    def step(self, candidate: ArcActionCandidate) -> ArcObservation:
        from arcengine import GameAction

        action = GameAction[candidate.action_type]
        data = {"x": candidate.x, "y": candidate.y} if candidate.action_type == "ACTION6" else None
        raw = self.env.step(
            action,
            data=data,
            reasoning={"candidate_id": candidate.candidate_id, "description": candidate.description},
        )
        current = self._convert(raw, "action_completed")
        if self._last and current.frame and self._last.frame:
            changed = sum(
                before != after
                for before_row, after_row in zip(self._last.frame, current.frame)
                for before, after in zip(before_row, after_row)
            )
            event = f"state={current.state}; changed_cells={changed}; levels_completed={current.levels_completed}"
            current = ArcObservation(**{**current.__dict__, "event": event})
        self._last = current
        return current


class DemoArcEnvironment:
    """Local key-free environment for verifying the comparison harness."""

    game_id = "demo-hidden-door"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.turn = 0
        self.door_open = False
        self.finished = False

    def _frame(self) -> tuple[tuple[int, ...], ...]:
        grid = [[0 for _ in range(8)] for _ in range(8)]
        grid[5][1] = 9
        grid[4][3] = 8
        grid[2][5] = 14 if self.door_open else 5
        return tuple(tuple(row) for row in grid)

    def _observation(self, event: str, state: str = "NOT_FINISHED") -> ArcObservation:
        return ArcObservation(
            game_id=self.game_id,
            state=state,
            levels_completed=1 if state == "WIN" else 0,
            win_levels=1,
            available_actions=("ACTION1", "ACTION2", "ACTION4"),
            frame=self._frame(),
            event=event,
        )

    def reset(self) -> ArcObservation:
        self.turn = 0
        self.door_open = False
        self.finished = False
        return self._observation("reset")

    def step(self, candidate: ArcActionCandidate) -> ArcObservation:
        self.turn += 1
        if candidate.action_type == "ACTION1":
            self.door_open = True
            return self._observation("door_opened")
        if candidate.action_type == "ACTION2" and self.door_open:
            self.finished = True
            return self._observation("goal_reached", "WIN")
        return self._observation("no_visible_change")
