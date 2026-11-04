from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

HARNESS_VERSION = "v7-grounded-goals"
INPUT_REPRESENTATION = "animation+summary+adaptive-lossless-delta-v3"


@dataclass(frozen=True)
class ArcGroundedGoal:
    target_component: str = "unspecified"
    target_attributes: str = "unspecified"
    current_bounds: str = "unspecified"
    destination: str = "unspecified"
    desired_state: str = "unspecified"
    proposed_action: str = "unspecified"
    prerequisites: str = "none"
    partial_success: str = "unspecified"


@dataclass(frozen=True)
class ArcActionCandidate:
    candidate_id: str
    action_type: str
    description: str
    x: int | None = None
    y: int | None = None
    expected_progress: float = 0.0
    risk: float = 0.0
    next_action_type: str | None = None
    next_x: int | None = None
    next_y: int | None = None


@dataclass(frozen=True)
class ArcHypothesis:
    id: int
    statement: str
    confidence: float
    predictions: dict[str, str]
    object_role: str = "unspecified"
    rule: str = "unspecified"
    goal_condition: str = "unspecified"
    falsification: str = "unspecified"
    role: str = "mechanism"
    grounded_goal: ArcGroundedGoal = field(default_factory=ArcGroundedGoal)


@dataclass(frozen=True)
class ArcProposal:
    summary: str
    hypotheses: tuple[ArcHypothesis, ...]
    action_candidates: tuple[ArcActionCandidate, ...]
    recommended_candidate_id: str
    recommendation_reason: str


@dataclass(frozen=True)
class ArcBelief:
    id: int
    statement: str
    confidence: float
    status: str
    evidence: str
    object_role: str = "unspecified"
    rule: str = "unspecified"
    goal_condition: str = "unspecified"
    falsification: str = "unspecified"
    role: str = "mechanism"


@dataclass(frozen=True)
class ArcReflection:
    summary: str
    observed_effect: str
    beliefs: tuple[ArcBelief, ...]
    next_focus: str


@dataclass(frozen=True)
class ArcObservation:
    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    available_actions: tuple[str, ...]
    frame: tuple[tuple[int, ...], ...]
    raw_frame_count: int = 1
    animation_frames: tuple[tuple[tuple[int, ...], ...], ...] = field(default_factory=tuple)
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
    reflection: ArcReflection | None = None
    model_calls: int = 1
    timings: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ArcEpisodeTrace:
    condition: str
    source: str
    game_id: str
    seed: int
    turns: list[ArcTurn] = field(default_factory=list)
    success: bool = False
    wall_seconds: float = 0.0
    comparable_seconds: float = 0.0
    error: str | None = None
    action_budget: int = 0
    model: str | None = None
    reasoning_effort: str | None = None
    harness_version: str = HARNESS_VERSION
    batch_id: str | None = None
    input_representation: str = INPUT_REPRESENTATION
    harness_flags: dict[str, bool] = field(default_factory=dict)
    intervention_summary: dict[str, int] = field(default_factory=dict)
    error_traceback: str | None = None
    stop_reason: str = ""
    resumed: bool = False
    replayed_actions: int = 0
    connection_events: list[dict[str, Any]] = field(default_factory=list)

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
    reasoning_effort: str | None = None
    harness_version: str = HARNESS_VERSION
    batch_id: str | None = None
    input_representation: str = INPUT_REPRESENTATION
    harness_flags: dict[str, bool] = field(default_factory=dict)
    model_settings: dict[str, Any] = field(default_factory=dict)
    condition_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_from_dict(data: dict[str, Any]) -> ArcActionCandidate:
    return ArcActionCandidate(**data)


def observation_from_dict(data: dict[str, Any]) -> ArcObservation:
    animation = tuple(tuple(tuple(row) for row in frame) for frame in data.get("animation_frames", ()))
    return ArcObservation(**{
        **data,
        "available_actions": tuple(data["available_actions"]),
        "frame": tuple(tuple(row) for row in data["frame"]),
        "animation_frames": animation,
    })


def proposal_from_dict(data: dict[str, Any]) -> ArcProposal:
    hypotheses = []
    for item in data["hypotheses"]:
        grounded = item.get("grounded_goal", {})
        hypotheses.append(ArcHypothesis(**{**item, "grounded_goal": ArcGroundedGoal(**grounded)}))
    return ArcProposal(
        summary=data["summary"], hypotheses=tuple(hypotheses),
        action_candidates=tuple(candidate_from_dict(item) for item in data["action_candidates"]),
        recommended_candidate_id=data["recommended_candidate_id"],
        recommendation_reason=data["recommendation_reason"],
    )


def reflection_from_dict(data: dict[str, Any]) -> ArcReflection:
    return ArcReflection(
        summary=data["summary"], observed_effect=data["observed_effect"],
        beliefs=tuple(ArcBelief(**item) for item in data["beliefs"]),
        next_focus=data["next_focus"],
    )
