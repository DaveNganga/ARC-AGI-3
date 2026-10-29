from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any, Protocol

from .arc_models import (
    ArcActionCandidate,
    ArcBelief,
    ArcHypothesis,
    ArcObservation,
    ArcProposal,
    ArcReflection,
)
from .retry import call_with_retries


class ArcReasoner(Protocol):
    def propose(self, observation: ArcObservation, history: list[dict[str, Any]]) -> ArcProposal: ...

    def reflect(
        self,
        before: ArcObservation,
        after: ArcObservation,
        proposal: ArcProposal,
        selected: ArcActionCandidate,
        history: list[dict[str, Any]],
    ) -> ArcReflection: ...


ARC_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "statement": {"type": "string"},
                    "confidence": {"type": "number"},
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "outcome": {"type": "string"},
                            },
                            "required": ["candidate_id", "outcome"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "statement", "confidence", "predictions"],
                "additionalProperties": False,
            },
        },
        "action_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "action_type": {"type": "string"},
                    "description": {"type": "string"},
                    "x": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "y": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "expected_progress": {"type": "number"},
                    "risk": {"type": "number"},
                },
                "required": ["candidate_id", "action_type", "description", "x", "y", "expected_progress", "risk"],
                "additionalProperties": False,
            },
        },
        "recommended_candidate_id": {"type": "string"},
        "recommendation_reason": {"type": "string"},
    },
    "required": ["summary", "hypotheses", "action_candidates", "recommended_candidate_id", "recommendation_reason"],
    "additionalProperties": False,
}

ARC_REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "observed_effect": {"type": "string"},
        "beliefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "statement": {"type": "string"},
                    "confidence": {"type": "number"},
                    "status": {"type": "string", "enum": ["supported", "weakened", "rejected", "uncertain", "new"]},
                    "evidence": {"type": "string"},
                },
                "required": ["id", "statement", "confidence", "status", "evidence"],
                "additionalProperties": False,
            },
        },
        "next_focus": {"type": "string"},
    },
    "required": ["summary", "observed_effect", "beliefs", "next_focus"],
    "additionalProperties": False,
}


def frame_summary(frame: tuple[tuple[int, ...], ...]) -> str:
    """Represent a grid compactly while preserving coordinates and spatial runs."""
    if not frame:
        return "empty frame"
    height, width = len(frame), len(frame[0])
    counts = Counter(value for row in frame for value in row)
    background = counts.most_common(1)[0][0]
    color_summary = []
    for value, count in sorted(counts.items()):
        locations = [(x, y) for y, row in enumerate(frame) for x, cell in enumerate(row) if cell == value]
        xs = [point[0] for point in locations]
        ys = [point[1] for point in locations]
        color_summary.append(
            f"color {value}: count={count}, bounds=x[{min(xs)}..{max(xs)}],y[{min(ys)}..{max(ys)}]"
        )

    rows = []
    for y, row in enumerate(frame):
        runs = []
        start = 0
        while start < width:
            value = row[start]
            end = start
            while end + 1 < width and row[end + 1] == value:
                end += 1
            if value != background:
                runs.append(f"x{start}" if start == end else f"x{start}-{end}" + f"={value}")
                if start == end:
                    runs[-1] += f"={value}"
            start = end + 1
        if runs:
            rows.append(f"y{y}: " + ", ".join(runs))
    return (
        f"shape={height}x{width}; coordinates use x=column,y=row; inferred background={background}\n"
        + "Color statistics:\n" + "\n".join(color_summary)
        + "\nNon-background horizontal runs (exact coordinates):\n"
        + ("\n".join(rows) if rows else "none")
    )


class OpenAIArcReasoner:
    def __init__(self, caller: Any, attempts: int = 3, retry_delay: float = 2.0):
        self.caller = caller
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.last_call_count = 0
        self.last_retry_count = 0

    def _request(self, build: Any) -> Any:
        """Request and parse together so malformed but schema-valid output is also retried."""
        result, retries = call_with_retries(build, self.attempts, self.retry_delay)
        self.last_retry_count = retries
        self.last_call_count = 1 + retries
        return result

    def propose(self, observation: ArcObservation, history: list[dict[str, Any]]) -> ArcProposal:
        ledger = history[-1].get("belief_ledger", []) if history else []
        prompt = f"""You are controlling an unfamiliar ARC-AGI-3 game. No instructions reveal its goal.
Infer possible world rules from the current grid and accumulated evidence. Return 2 to 8 competing hypotheses and 2 to 8 concrete action candidates. Each hypothesis must predict a short observable outcome for every candidate_id. Also recommend one candidate yourself; this recommendation is the LLM-only baseline.

The grid description preserves exact x/y coordinates and non-background shapes. Reason spatially from those runs. Prefer actions that can distinguish hypotheses or produce level progress. Treat changed pixels as evidence, not as progress by itself. Do not repeat a click from the same frame state when it previously caused no change or only toggled the same pattern; propose a genuinely different test instead.

Maintain the belief ledger across turns. Preserve supported explanations, lower confidence in contradicted explanations, and do not silently restore rejected ideas. Prefer a focused experiment that separates the leading hypotheses over random exploration. Every response must use contiguous IDs starting at 0.

Only use these currently available action types: {list(observation.available_actions)}.
ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right, ACTION5=interact/space, ACTION6=click and requires x,y from 0 through 63, ACTION7=undo. Meanings may still be game-specific.
Keep confidence, expected_progress, and risk between 0 and 1. Candidate IDs must be unique.

Game state: {observation.state}; levels completed: {observation.levels_completed}; win levels: {observation.win_levels}
Current grid:
{frame_summary(observation.frame)}

Current belief ledger:
{ledger}

Recent evidence:
{history[-8:]}
"""
        return self._request(lambda: parse_proposal(
            self.caller.call_json(prompt, ARC_PROPOSAL_SCHEMA, "arc_qahs_proposal"),
            observation.available_actions,
        ))

    def reflect(
        self,
        before: ArcObservation,
        after: ArcObservation,
        proposal: ArcProposal,
        selected: ArcActionCandidate,
        history: list[dict[str, Any]],
    ) -> ArcReflection:
        previous_ledger = history[-1].get("belief_ledger", []) if history else []
        prompt = f"""Act as an evidence reviewer for an unfamiliar ARC-AGI-3 game. An action was just taken. Update the persistent belief ledger; do not choose the next action yet.

Compare each hypothesis's prediction for the selected candidate with the actual observation. Mark it supported, weakened, rejected, or uncertain. Preserve stable IDs and statements when the underlying explanation is unchanged. You may introduce a genuinely new explanation only when the observation demands it, and the complete returned ledger must contain 2 to 8 beliefs with contiguous IDs starting at 0. Confidence must be between 0 and 1. Pixel change alone is not proof of progress; level completion and terminal state are stronger evidence.

Previous ledger: {previous_ledger}
Hypotheses tested: {[asdict(item) for item in proposal.hypotheses]}
Selected action: {asdict(selected)}
Before: state={before.state}, levels={before.levels_completed}\n{frame_summary(before.frame)}
After: state={after.state}, levels={after.levels_completed}, event={after.event}\n{frame_summary(after.frame)}
Recent evidence: {history[-6:]}
"""
        return self._request(lambda: parse_reflection(
            self.caller.call_json(prompt, ARC_REFLECTION_SCHEMA, "arc_qahs_reflection")
        ))


def parse_proposal(payload: dict[str, Any], available_actions: tuple[str, ...]) -> ArcProposal:
    candidates = []
    seen: set[str] = set()
    for row in payload.get("action_candidates", [])[:8]:
        candidate_id = str(row["candidate_id"])
        action_type = str(row["action_type"]).upper()
        if candidate_id in seen or action_type not in available_actions:
            continue
        x = row.get("x")
        y = row.get("y")
        if action_type == "ACTION6":
            if x is None or y is None or not (0 <= int(x) <= 63 and 0 <= int(y) <= 63):
                continue
            x, y = int(x), int(y)
        else:
            x = y = None
        candidates.append(ArcActionCandidate(
            candidate_id=candidate_id,
            action_type=action_type,
            description=str(row["description"]),
            x=x,
            y=y,
            expected_progress=min(1.0, max(0.0, float(row["expected_progress"]))),
            risk=min(1.0, max(0.0, float(row["risk"]))),
        ))
        seen.add(candidate_id)
    if not candidates:
        raise ValueError("The model returned no valid action candidates")

    candidate_ids = {candidate.candidate_id for candidate in candidates}
    hypotheses = []
    hypothesis_rows = sorted(payload.get("hypotheses", [])[:8], key=lambda row: int(row.get("id", 0)))
    for index, row in enumerate(hypothesis_rows):
        predictions = {
            str(item["candidate_id"]): str(item["outcome"])
            for item in row.get("predictions", [])
            if str(item["candidate_id"]) in candidate_ids
        }
        for candidate_id in candidate_ids:
            predictions.setdefault(candidate_id, "unknown")
        hypotheses.append(ArcHypothesis(
            id=index,
            statement=str(row["statement"]),
            confidence=min(1.0, max(0.0, float(row["confidence"]))),
            predictions=predictions,
        ))
    if not hypotheses:
        raise ValueError("The model returned no valid hypotheses")

    recommended = str(payload.get("recommended_candidate_id", ""))
    if recommended not in candidate_ids:
        recommended = candidates[0].candidate_id
    return ArcProposal(
        summary=str(payload.get("summary", "")),
        hypotheses=tuple(hypotheses),
        action_candidates=tuple(candidates),
        recommended_candidate_id=recommended,
        recommendation_reason=str(payload.get("recommendation_reason", "")),
    )


class DemoArcReasoner:
    """Deterministic reasoner used to test the dashboard without external APIs."""

    def __init__(self):
        self.last_call_count = 0

    def propose(self, observation: ArcObservation, history: list[dict[str, Any]]) -> ArcProposal:
        self.last_call_count = 0
        door_open = any(item.get("result") == "door_opened" for item in history)
        if door_open:
            candidates = (
                ArcActionCandidate("finish", "ACTION2", "Move through the open door", expected_progress=1.0),
                ArcActionCandidate("wait", "ACTION4", "Move away from the door"),
            )
            hypotheses = (
                ArcHypothesis(0, "The opened path is the goal", 0.95, {"finish": "win", "wait": "no progress"}),
                ArcHypothesis(1, "The opened path is a decoy", 0.05, {"finish": "blocked", "wait": "no progress"}),
            )
            recommended = "finish"
        else:
            candidates = (
                ArcActionCandidate("probe", "ACTION1", "Touch the uncertain object", expected_progress=0.2, risk=0.1),
                ArcActionCandidate("wander", "ACTION4", "Move toward the visible edge", expected_progress=0.55),
            )
            hypotheses = (
                ArcHypothesis(0, "The object is a switch", 0.5, {"probe": "door opens", "wander": "no change"}),
                ArcHypothesis(1, "The object is a hazard", 0.5, {"probe": "player damaged", "wander": "no change"}),
            )
            already_wandered = any(item.get("candidate_id") == "wander" for item in history)
            recommended = "probe" if already_wandered else "wander"
        return ArcProposal(
            summary="A small hidden-rule demonstration with two competing explanations.",
            hypotheses=hypotheses,
            action_candidates=candidates,
            recommended_candidate_id=recommended,
            recommendation_reason="The baseline favors apparent immediate progress.",
        )

    def reflect(
        self,
        before: ArcObservation,
        after: ArcObservation,
        proposal: ArcProposal,
        selected: ArcActionCandidate,
        history: list[dict[str, Any]],
    ) -> ArcReflection:
        del before, history
        self.last_call_count = 0
        progressed = after.levels_completed > 0 or after.state.upper() == "WIN" or after.event == "door_opened"
        beliefs = tuple(
            ArcBelief(
                item.id,
                item.statement,
                min(1.0, item.confidence + 0.25) if progressed else item.confidence,
                "supported" if progressed else "uncertain",
                f"{selected.candidate_id} produced {after.event}",
            )
            for item in proposal.hypotheses
        )
        return ArcReflection("Updated beliefs from the observed result.", after.event, beliefs, "Test the leading explanation.")


class CachingArcReasoner:
    """Reuse identical proposals across matched conditions for a fairer comparison."""

    def __init__(self, reasoner: ArcReasoner):
        self.reasoner = reasoner
        self.cache: dict[str, ArcProposal] = {}
        self.reflection_cache: dict[str, ArcReflection] = {}
        self.last_call_count = 0

    def propose(self, observation: ArcObservation, history: list[dict[str, Any]]) -> ArcProposal:
        key = json.dumps({"observation": asdict(observation), "history": history}, sort_keys=True)
        if key in self.cache:
            self.last_call_count = 0
            return self.cache[key]
        proposal = self.reasoner.propose(observation, history)
        self.last_call_count = int(getattr(self.reasoner, "last_call_count", 1))
        self.cache[key] = proposal
        return proposal

    def reflect(
        self,
        before: ArcObservation,
        after: ArcObservation,
        proposal: ArcProposal,
        selected: ArcActionCandidate,
        history: list[dict[str, Any]],
    ) -> ArcReflection:
        key = json.dumps({
            "before": asdict(before),
            "after": asdict(after),
            "proposal": asdict(proposal),
            "selected": asdict(selected),
            "history": history,
        }, sort_keys=True)
        if key in self.reflection_cache:
            self.last_call_count = 0
            return self.reflection_cache[key]
        reflection = self.reasoner.reflect(before, after, proposal, selected, history)
        self.last_call_count = int(getattr(self.reasoner, "last_call_count", 1))
        self.reflection_cache[key] = reflection
        return reflection


def parse_reflection(payload: dict[str, Any]) -> ArcReflection:
    rows = sorted(payload.get("beliefs", [])[:8], key=lambda row: int(row.get("id", 0)))
    if not rows:
        raise ValueError("The model returned no belief updates")
    beliefs = tuple(
        ArcBelief(
            id=index,
            statement=str(row["statement"]),
            confidence=min(1.0, max(0.0, float(row["confidence"]))),
            status=str(row["status"]),
            evidence=str(row["evidence"]),
        )
        for index, row in enumerate(rows)
    )
    return ArcReflection(
        summary=str(payload.get("summary", "")),
        observed_effect=str(payload.get("observed_effect", "")),
        beliefs=beliefs,
        next_focus=str(payload.get("next_focus", "")),
    )
