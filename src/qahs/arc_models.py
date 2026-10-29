from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArcActionCandidate:
    candidate_id: str
    action_type: str
    description: str
    x: int | None = None
    y: int | None = None
    expected_progress: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class ArcHypothesis:
    id: int
    statement: str
    confidence: float
    predictions: dict[str, str]


@dataclass(frozen=True)
class ArcProposal:
    summary: str
    hypotheses: tuple[ArcHypothesis, ...]
    action_candidates: tuple[ArcActionCandidate, ...]
    recommended_candidate_id: str
    recommendation_reason: str


@dataclass(frozen=True)
class ArcObservation:
    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    available_actions: tuple[str, ...]
    frame: tuple[tuple[int, ...], ...]
    raw_frame_count: int = 1
    event: str = "observation"

    @property
    def success(self) -> bool:
        return self.state.upper() == "WIN"

    @property
    def terminal(self) -> bool:
        return self.state.upper() in {"WIN", "GAME_OVER"}


@dataclass
class ArcTurn:
    turn: int
    observation_before: ArcObservation
    proposal: ArcProposal
    condition: str
    selected_candidate: ArcActionCandidate
    action_scores: list[dict[str, Any]]
    search_diagnostics: dict[str, Any]
    observation_after: ArcObservation
    model_calls: int = 1


@dataclass
class ArcEpisodeTrace:
    condition: str
    source: str
    game_id: str
    seed: int
    turns: list[ArcTurn] = field(default_factory=list)
    success: bool = False
    wall_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArcComparisonTrace:
    source: str
    game_id: str
    seed: int
    without_qahs: ArcEpisodeTrace
    with_classical: ArcEpisodeTrace
    with_qahs: ArcEpisodeTrace
    action_budget: int = 0
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
