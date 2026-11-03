from __future__ import annotations

import json
import base64
import io
import re
from collections import Counter
from dataclasses import asdict
from typing import Any, Protocol

from .arc_models import (
    ArcActionCandidate,
    ArcBelief,
    ArcGroundedGoal,
    ArcHypothesis,
    ArcObservation,
    ArcProposal,
    ArcReflection,
)


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

    def review_and_propose(
        self, before: ArcObservation, after: ArcObservation, proposal: ArcProposal,
        selected: ArcActionCandidate, history: list[dict[str, Any]],
    ) -> tuple[ArcReflection, ArcProposal]: ...


OUTCOME_CODES = ("MOVE", "CHANGE", "PROGRESS", "FAIL", "NO_CHANGE", "TERMINAL", "UNKNOWN")
OUTCOME_LEGEND = "MOVE=position only, CHANGE=grid changed, PROGRESS=level progress, FAIL=harm/reset, NO_CHANGE=unchanged, TERMINAL=win/end, UNKNOWN=uncertain"


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
                    "object_role": {"type": "string"},
                    "rule": {"type": "string"},
                    "goal_condition": {"type": "string"},
                    "falsification": {"type": "string"},
                    "role": {"type": "string", "enum": ["mechanism", "goal", "obstacle", "interaction", "terminal"]},
                    "grounded_goal": {
                        "type": "object",
                        "properties": {
                            "target_component": {"type": "string"},
                            "target_attributes": {"type": "string"},
                            "current_bounds": {"type": "string"},
                            "destination": {"type": "string"},
                            "desired_state": {"type": "string"},
                            "proposed_action": {"type": "string"},
                            "prerequisites": {"type": "string"},
                            "partial_success": {"type": "string"},
                        },
                        "required": ["target_component", "target_attributes", "current_bounds", "destination", "desired_state", "proposed_action", "prerequisites", "partial_success"],
                        "additionalProperties": False,
                    },
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "outcome": {"type": "string", "enum": list(OUTCOME_CODES)},
                            },
                            "required": ["candidate_id", "outcome"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "statement", "confidence", "object_role", "rule", "goal_condition", "falsification", "role", "grounded_goal", "predictions"],
                "additionalProperties": False,
            },
        },
        "action_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "expected_progress": {"type": "number"},
                    "risk": {"type": "number"},
                },
                "required": ["candidate_id", "expected_progress", "risk"],
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
        "beliefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "status": {"type": "string", "enum": ["supported", "weakened", "rejected", "uncertain", "new"]},
                    "confidence_delta": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["id", "status", "confidence_delta", "evidence"],
                "additionalProperties": False,
            },
        },
        "next_focus": {"type": "string"},
    },
    "required": ["beliefs", "next_focus"],
    "additionalProperties": False,
}

ARC_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"reflection": ARC_REFLECTION_SCHEMA, "proposal": ARC_PROPOSAL_SCHEMA},
    "required": ["reflection", "proposal"],
    "additionalProperties": False,
}


def compact_history(history: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Keep decision evidence but omit old hypothesis and prediction matrices."""
    fields = ("turn", "candidate_id", "action_type", "x", "y", "next_action_type", "next_x", "next_y",
              "changed_cells", "changed_coordinates", "levels_delta", "reflection", "next_focus", "result", "state",
              "selected_role", "goal_family", "grounded_goal", "meaningful_change", "strategy_reset", "transfer_attempted")
    return [{key: row[key] for key in fields if key in row} for row in history[-limit:]]


def normalize_role(role: str, row: dict[str, Any]) -> str:
    """Validate the model role and repair obvious mismatches from structured fields."""
    allowed = {"mechanism", "goal", "obstacle", "interaction", "terminal"}
    role = role.lower().strip()
    if role in allowed:
        return role
    text = " ".join(str(row.get(key, "")) for key in ("statement", "object_role", "goal_condition", "rule")).lower()
    if any(word in text for word in ("win", "terminal", "finish", "complete the game")):
        return "terminal"
    if any(word in text for word in ("goal", "target", "objective", "reach", "score")) and role == "mechanism":
        return "goal"
    if any(word in text for word in ("wall", "barrier", "blocked", "obstacle", "gate")) and role not in {"goal", "terminal"}:
        return "obstacle"
    return "mechanism"


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


def lossless_frame_data(frame: tuple[tuple[int, ...], ...]) -> tuple[str, str]:
    """Return the shorter of two exact grid encodings for model input."""
    if not frame:
        return "adaptive-empty", "empty frame"
    height, width = len(frame), len(frame[0])
    counts = Counter(value for row in frame for value in row)
    background = counts.most_common(1)[0][0]
    cells = [
        f"{x},{y}:{value}"
        for y, row in enumerate(frame)
        for x, value in enumerate(row)
        if value != background
    ]
    sparse = f"shape={height}x{width}; default={background}; exceptions=" + (" ".join(cells) if cells else "none")
    matrix = f"shape={height}x{width}; rows=" + "/".join(",".join(str(value) for value in row) for row in frame)
    if len(sparse) <= len(matrix):
        return "adaptive-sparse", sparse
    return "adaptive-matrix", matrix


def frame_context(frame: tuple[tuple[int, ...], ...]) -> str:
    mode, exact = lossless_frame_data(frame)
    return f"{frame_summary(frame)}\nLossless grid ({mode}; unspecified cells use default):\n{exact}"


def transition_context(before: ArcObservation, after: ArcObservation, history: list[dict[str, Any]]) -> str:
    """Use compact textual deltas between periodic lossless refreshes; images remain attached."""
    changes = [(x, y, old, new) for y, (a, b) in enumerate(zip(before.frame, after.frame))
               for x, (old, new) in enumerate(zip(a, b)) if old != new]
    refresh = (not history or len(history) % 5 == 0 or after.levels_completed != before.levels_completed
               or len(changes) >= 0.15 * max(1, sum(len(row) for row in before.frame)))
    if refresh:
        return f"Full refresh after action:\n{frame_context(after.frame)}"
    return (f"shape={len(after.frame)}x{len(after.frame[0]) if after.frame else 0}; "
            f"changed_cells={len(changes)}; changes(old>new)={changes[:128]}")


ARC_COLORS = (
    "#ffffff", "#f0f0f0", "#c8c8c8", "#969696", "#4b4b4b", "#000000",
    "#e53aa3", "#ff7fcc", "#e53b2c", "#2e5cb8", "#62a5ff", "#f5d442",
    "#f28c28", "#7f1d1d", "#35a853", "#7e57c2",
)


def frame_image_data_url(frame: tuple[tuple[int, ...], ...]) -> str | None:
    """Render an exact nearest-neighbor grid image for multimodal reasoning."""
    if not frame:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    height, width = len(frame), len(frame[0])
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            color = ARC_COLORS[value] if 0 <= value < len(ARC_COLORS) else "#000000"
            pixels[x, y] = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    image = image.resize((max(512, width * 8), max(512, height * 8)), Image.Resampling.NEAREST)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def observation_images(observation: ArcObservation, limit: int = 4) -> list[str]:
    """Keep the causal animation sequence while removing duplicate frames."""
    frames = list(observation.animation_frames) or [observation.frame]
    if observation.frame and (not frames or frames[-1] != observation.frame):
        frames.append(observation.frame)
    images = [frame_image_data_url(frame) for frame in frames[-limit:]]
    return [image for image in images if image]


def _component_points(frame: tuple[tuple[int, ...], ...], limit: int = 12) -> list[tuple[int, int]]:
    if not frame:
        return []
    height, width = len(frame), len(frame[0])
    background = Counter(value for row in frame for value in row).most_common(1)[0][0]
    remaining = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if value != background}
    points: list[tuple[int, int]] = []
    while remaining and len(points) < limit:
        seed = min(remaining, key=lambda point: (point[1], point[0]))
        color = frame[seed[1]][seed[0]]
        stack, component = [seed], []
        remaining.remove(seed)
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                nx, ny = neighbor
                if 0 <= nx < width and 0 <= ny < height and neighbor in remaining and frame[ny][nx] == color:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        xs, ys = [x for x, _ in component], [y for _, y in component]
        candidates = [
            (round(sum(xs) / len(xs)), round(sum(ys) / len(ys))),
            (min(xs), min(ys)), (max(xs), min(ys)),
            (min(xs), max(ys)), (max(xs), max(ys)),
        ]
        for point in candidates:
            if point not in points:
                points.append(point)
                if len(points) >= limit:
                    break
    return points


def generate_plan_catalog(
    observation: ArcObservation,
    history: list[dict[str, Any]] | None = None,
    limit: int = 16,
) -> tuple[ArcActionCandidate, ...]:
    """Create reproducible two-step experiments from actions and visible objects."""
    base: list[tuple[str, int | None, int | None, str]] = []
    history = history or []
    recent_changes = [
        tuple(point)
        for item in reversed(history[-4:])
        for point in item.get("changed_coordinates", [])[:4]
    ]
    grounded_points: list[tuple[int, int]] = []
    for item in reversed(history[-8:]):
        grounded = item.get("grounded_goal") or {}
        text = " ".join(str(grounded.get(key, "")) for key in ("current_bounds", "destination", "target_attributes"))
        for x, y in re.findall(r"(?:x\s*=?\s*)?(\d{1,2})\s*[,;:]\s*(?:y\s*=?\s*)?(\d{1,2})", text, flags=re.IGNORECASE):
            point = (min(63, int(x)), min(63, int(y)))
            if point not in grounded_points:
                grounded_points.append(point)
    clicked = {(item.get("x"), item.get("y")) for item in history if item.get("action_type") == "ACTION6"}
    for action in observation.available_actions:
        if action == "ACTION6":
            height = len(observation.frame)
            width = len(observation.frame[0]) if height else 0
            boundary = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)] if width else []
            exploration = [(width // 4, height // 4), (3 * width // 4, height // 4), (width // 4, 3 * height // 4), (3 * width // 4, 3 * height // 4)] if width else []
            points = grounded_points + recent_changes + _component_points(observation.frame) + exploration + boundary
            unique = []
            for point in points:
                if point not in unique:
                    unique.append(point)
            unique.sort(key=lambda point: (point in clicked, point not in grounded_points, point not in recent_changes))
            for x, y in unique:
                base.append((action, int(x), int(y), f"click evidence/object/boundary at ({x},{y})"))
        else:
            base.append((action, None, None, action.lower()))
    base = base[:8]
    if not base:
        return tuple()
    plans: list[ArcActionCandidate] = []
    # Round-robin pairings keep every physical action represented as a first step.
    for offset in range(len(base)):
        for first_index, first in enumerate(base):
            second = base[(first_index + offset) % len(base)]
            plans.append(ArcActionCandidate(
                candidate_id=f"plan-{len(plans):02d}",
                action_type=first[0], x=first[1], y=first[2],
                description=f"{first[3]}, then {second[3]}",
                expected_progress=0.5, risk=0.5,
                next_action_type=second[0], next_x=second[1], next_y=second[2],
            ))
            if len(plans) >= limit:
                return tuple(plans)
    return tuple(plans)


def compact_plan_catalog(catalog: tuple[ArcActionCandidate, ...]) -> dict[str, list[dict[str, Any]]]:
    """Describe physical actions once and encode each two-step plan as two IDs."""
    physical: list[tuple[str, int | None, int | None]] = []
    for plan in catalog:
        for signature in ((plan.action_type, plan.x, plan.y),
                          (plan.next_action_type or plan.action_type, plan.next_x, plan.next_y)):
            if signature not in physical:
                physical.append(signature)
    action_ids = {signature: f"a{index}" for index, signature in enumerate(physical)}
    return {
        "actions": [{"id": action_ids[item], "type": item[0], "x": item[1], "y": item[2]} for item in physical],
        "plans": [{"candidate_id": plan.candidate_id,
                   "actions": [action_ids[(plan.action_type, plan.x, plan.y)],
                               action_ids[(plan.next_action_type or plan.action_type, plan.next_x, plan.next_y)]]}
                  for plan in catalog],
    }


class OpenAIArcReasoner:
    def __init__(self, caller: Any):
        self.caller = caller
        self.last_call_count = 0

    def propose(self, observation: ArcObservation, history: list[dict[str, Any]]) -> ArcProposal:
        self.last_call_count = 1
        ledger = history[-1].get("belief_ledger", []) if history else []
        catalog = generate_plan_catalog(observation, history)
        prompt = f"""You are controlling an unfamiliar ARC-AGI-3 game. No instructions reveal its goal.
Infer possible world rules from the current grid and accumulated evidence. Return 6 to 10 distinct structured hypotheses. Classify each as mechanism, goal, obstacle, interaction, or terminal. Identify its object role, governing rule, goal condition, and falsification. For every hypothesis, fill grounded_goal. Goal hypotheses must name the visible target, its attributes and current bounds, destination, desired state, proposed available action, prerequisites, and a measurable partial-success signal. Use "unspecified" only when the evidence truly cannot ground a field. Do not restate semantic duplicates under new IDs. Each hypothesis must predict one compact outcome code for every candidate_id: {OUTCOME_LEGEND}.

The controller generated the compact two-step plan catalog below. Physical actions are defined once; each plan references two action IDs. Return every candidate_id exactly once with only expected_progress and risk. Recommend one candidate yourself; this recommendation is the LLM-only baseline. The harness reconstructs physical fields locally, executes the first step, observes the result, then replans.

Plan catalog:
{compact_plan_catalog(catalog)}

The grid description preserves exact x/y coordinates and non-background shapes. Reason spatially from those runs. Prefer actions that can distinguish hypotheses or produce level progress. Treat changed pixels as evidence, not as progress by itself. Do not repeat a click from the same frame state when it previously caused no change or only toggled the same pattern; propose a genuinely different test instead.

Maintain the belief ledger across turns. Preserve supported explanations, lower confidence in contradicted explanations, and do not silently restore rejected ideas. Established mechanics are background knowledge: prefer a focused, goal-directed or decisive falsification test over repeatedly confirming movement. Every response must use contiguous IDs starting at 0.

Only use these currently available action types: {list(observation.available_actions)}.
ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right, ACTION5=interact/space, ACTION6=click and requires x,y from 0 through 63, ACTION7=undo. Meanings may still be game-specific.
Keep confidence, expected_progress, and risk between 0 and 1. Candidate IDs must be unique.

Game state: {observation.state}; levels completed: {observation.levels_completed}; win levels: {observation.win_levels}
Current grid:
{frame_context(observation.frame)}

Current belief ledger:
{ledger}

Recent evidence (old prediction matrices intentionally omitted):
{compact_history(history)}
"""
        images = observation_images(observation)
        payload = self.caller.call_json(prompt, ARC_PROPOSAL_SCHEMA, "arc_qahs_proposal", images=images or None)
        return parse_proposal(payload, observation.available_actions, catalog)

    def reflect(
        self,
        before: ArcObservation,
        after: ArcObservation,
        proposal: ArcProposal,
        selected: ArcActionCandidate,
        history: list[dict[str, Any]],
    ) -> ArcReflection:
        self.last_call_count = 1
        previous_ledger = history[-1].get("belief_ledger", []) if history else []
        prompt = f"""Act as an evidence reviewer for an unfamiliar ARC-AGI-3 game. Return only beliefs that changed. For each, provide its stable ID, status, signed confidence_delta from -1 to 1, and brief evidence. Unchanged beliefs are merged by the harness; an empty update list is valid. Do not restate rule fields.

Compare each hypothesis's prediction for the selected candidate with the actual observation. Mark it supported, weakened, rejected, or uncertain. Pixel change alone is not proof of progress; level completion and terminal state are stronger evidence.

Previous ledger: {previous_ledger}
Hypotheses tested: {[asdict(item) for item in proposal.hypotheses]}
Selected action: {asdict(selected)}
Before: state={before.state}, levels={before.levels_completed}
After: state={after.state}, levels={after.levels_completed}, event={after.event}\n{transition_context(before, after, history)}
Recent evidence: {compact_history(history, 6)}
"""
        images = observation_images(before, 2) + observation_images(after, 4)
        payload = self.caller.call_json(prompt, ARC_REFLECTION_SCHEMA, "arc_qahs_reflection", images=images)
        payload = {**payload, "summary": "Beliefs updated from the observed action.", "observed_effect": after.event}
        return merge_reflection(previous_ledger, proposal, payload)

    def review_and_propose(
        self, before: ArcObservation, after: ArcObservation, proposal: ArcProposal,
        selected: ArcActionCandidate, history: list[dict[str, Any]],
    ) -> tuple[ArcReflection, ArcProposal]:
        self.last_call_count = 1
        previous_ledger = history[-1].get("belief_ledger", []) if history else []
        changed = [[x, y] for y, (a, b) in enumerate(zip(before.frame, after.frame)) for x, (old, new) in enumerate(zip(a, b)) if old != new][:64]
        evidence = compact_history(history) + [{"turn": len(history) + 1, "candidate_id": selected.candidate_id,
            "action_type": selected.action_type, "changed_cells": len(changed), "changed_coordinates": changed,
            "levels_delta": after.levels_completed - before.levels_completed, "result": after.event, "state": after.state}]
        catalog = generate_plan_catalog(after, evidence)
        prompt = f"""Review the action just taken, then plan the next turn in one response.

REFLECTION: return only beliefs that changed, using stable ID, status, signed confidence_delta from -1 to 1, and brief evidence. Do not restate rule fields. The harness preserves and merges every unchanged belief. Compare the tested prediction with the observed effect; pixel change alone is not progress.

NEXT PROPOSAL: return 6 to 10 semantically distinct structured hypotheses, each classified as mechanism, goal, obstacle, interaction, or terminal, and every generated candidate_id exactly once with only expected_progress and risk. Fill every grounded_goal field; make goal hypotheses spatially concrete and executable with an available action. Preserve stable beliefs and rejected ideas. Established mechanics are background knowledge; prefer goal-directed or decisive falsification tests. Each prediction must be one code: {OUTCOME_LEGEND}. Recommend the most informative plan. The harness reconstructs actions locally, executes only the first action, and replans after observing it.

Previous ledger: {previous_ledger}
Tested hypotheses: {[asdict(item) for item in proposal.hypotheses]}
Selected plan: {asdict(selected)}
Before: state={before.state}, levels={before.levels_completed}
After: state={after.state}, levels={after.levels_completed}, event={after.event}\n{transition_context(before, after, history)}
Compact evidence: {evidence}
Available actions: {list(after.available_actions)}
Next compact plan catalog: {compact_plan_catalog(catalog)}
"""
        images = observation_images(before, 2) + observation_images(after, 4)
        payload = self.caller.call_json(prompt, ARC_REVIEW_SCHEMA, "arc_qahs_review_plan", images=images)
        reflection_payload = {**payload["reflection"], "summary": "Beliefs updated from the observed action.",
                              "observed_effect": after.event}
        reflection = merge_reflection(previous_ledger, proposal, reflection_payload)
        return reflection, parse_proposal(payload["proposal"], after.available_actions, catalog)


def parse_proposal(
    payload: dict[str, Any],
    available_actions: tuple[str, ...],
    catalog: tuple[ArcActionCandidate, ...] | None = None,
) -> ArcProposal:
    catalog_by_id = {item.candidate_id: item for item in catalog or ()}
    candidates = []
    seen: set[str] = set()
    for row in payload.get("action_candidates", [])[:32]:
        candidate_id = str(row["candidate_id"])
        generated = catalog_by_id.get(candidate_id)
        if catalog_by_id and generated is None:
            continue
        action_type = generated.action_type if generated else str(row["action_type"]).upper()
        if candidate_id in seen or action_type not in available_actions:
            continue
        x = generated.x if generated else row.get("x")
        y = generated.y if generated else row.get("y")
        if action_type == "ACTION6":
            if x is None or y is None or not (0 <= int(x) <= 63 and 0 <= int(y) <= 63):
                continue
            x, y = int(x), int(y)
        else:
            x = y = None
        candidates.append(ArcActionCandidate(
            candidate_id=candidate_id,
            action_type=action_type,
            description=str(row.get("description") or (generated.description if generated else "")),
            x=x,
            y=y,
            expected_progress=min(1.0, max(0.0, float(row["expected_progress"]))),
            risk=min(1.0, max(0.0, float(row["risk"]))),
            next_action_type=generated.next_action_type if generated else row.get("next_action_type"),
            next_x=generated.next_x if generated else row.get("next_x"),
            next_y=generated.next_y if generated else row.get("next_y"),
        ))
        seen.add(candidate_id)
    for generated in catalog or ():
        if generated.candidate_id not in seen:
            candidates.append(generated)
            seen.add(generated.candidate_id)
    if not candidates:
        raise ValueError("The model returned no valid action candidates")

    candidate_ids = {candidate.candidate_id for candidate in candidates}
    hypotheses = []
    hypothesis_rows = sorted(payload.get("hypotheses", [])[:16], key=lambda row: int(row.get("id", 0)))
    for index, row in enumerate(hypothesis_rows):
        predictions = {
            str(item["candidate_id"]): str(item["outcome"]).upper() if str(item["outcome"]).upper() in OUTCOME_CODES else "UNKNOWN"
            for item in row.get("predictions", [])
            if str(item["candidate_id"]) in candidate_ids
        }
        for candidate_id in candidate_ids:
            predictions.setdefault(candidate_id, "UNKNOWN")
        hypotheses.append(ArcHypothesis(
            id=index,
            statement=str(row["statement"]),
            confidence=min(1.0, max(0.0, float(row["confidence"]))),
            predictions=predictions,
            object_role=str(row.get("object_role", "unspecified")),
            rule=str(row.get("rule", row.get("statement", "unspecified"))),
            goal_condition=str(row.get("goal_condition", "unspecified")),
            falsification=str(row.get("falsification", "unspecified")),
            role=normalize_role(str(row.get("role", "mechanism")), row),
            grounded_goal=ArcGroundedGoal(**{
                key: str((row.get("grounded_goal") or {}).get(key, default))
                for key, default in {
                    "target_component": "unspecified", "target_attributes": "unspecified",
                    "current_bounds": "unspecified", "destination": "unspecified",
                    "desired_state": "unspecified", "proposed_action": "unspecified",
                    "prerequisites": "none", "partial_success": "unspecified",
                }.items()
            }),
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

    def review_and_propose(
        self, before: ArcObservation, after: ArcObservation, proposal: ArcProposal,
        selected: ArcActionCandidate, history: list[dict[str, Any]],
    ) -> tuple[ArcReflection, ArcProposal]:
        reflection = self.reflect(before, after, proposal, selected, history)
        evidence = history + [{"candidate_id": selected.candidate_id, "result": after.event,
                               "belief_ledger": [asdict(item) for item in reflection.beliefs]}]
        return reflection, self.propose(after, evidence)


class CachingArcReasoner:
    """Reuse identical proposals across matched conditions for a fairer comparison."""

    def __init__(self, reasoner: ArcReasoner):
        self.reasoner = reasoner
        self.cache: dict[str, ArcProposal] = {}
        self.reflection_cache: dict[str, ArcReflection] = {}
        self.review_cache: dict[str, tuple[ArcReflection, ArcProposal]] = {}
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

    def review_and_propose(
        self, before: ArcObservation, after: ArcObservation, proposal: ArcProposal,
        selected: ArcActionCandidate, history: list[dict[str, Any]],
    ) -> tuple[ArcReflection, ArcProposal]:
        key = json.dumps({"before": asdict(before), "after": asdict(after), "proposal": asdict(proposal),
                          "selected": asdict(selected), "history": history}, sort_keys=True)
        if key in self.review_cache:
            self.last_call_count = 0
            return self.review_cache[key]
        result = self.reasoner.review_and_propose(before, after, proposal, selected, history)
        self.last_call_count = int(getattr(self.reasoner, "last_call_count", 1))
        self.review_cache[key] = result
        return result


def parse_reflection(payload: dict[str, Any]) -> ArcReflection:
    rows = sorted(payload.get("beliefs", [])[:16], key=lambda row: int(row.get("id", 0)))
    if not rows:
        raise ValueError("The model returned no belief updates")
    beliefs = tuple(
        ArcBelief(
            id=index,
            statement=str(row["statement"]),
            confidence=min(1.0, max(0.0, float(row["confidence"]))),
            status=str(row["status"]),
            evidence=str(row["evidence"]),
            object_role=str(row.get("object_role", "unspecified")),
            rule=str(row.get("rule", row.get("statement", "unspecified"))),
            goal_condition=str(row.get("goal_condition", "unspecified")),
            falsification=str(row.get("falsification", "unspecified")),
            role=str(row.get("role", "mechanism")),
        )
        for index, row in enumerate(rows)
    )
    return ArcReflection(
        summary=str(payload.get("summary", "")),
        observed_effect=str(payload.get("observed_effect", "")),
        beliefs=beliefs,
        next_focus=str(payload.get("next_focus", "")),
    )


def merge_reflection(
    previous_ledger: list[dict[str, Any]], proposal: ArcProposal, payload: dict[str, Any]
) -> ArcReflection:
    """Merge model-supplied belief deltas into the complete persistent ledger."""
    base = {int(row["id"]): dict(row) for row in previous_ledger if "id" in row}
    if not base:
        base = {item.id: {"id": item.id, "statement": item.statement, "confidence": item.confidence,
                "status": "uncertain", "evidence": "Proposed but not yet tested", "object_role": item.object_role,
                "rule": item.rule, "goal_condition": item.goal_condition, "falsification": item.falsification,
                "role": item.role}
                for item in proposal.hypotheses}
    for row in payload.get("beliefs", []):
        belief_id = int(row.get("id", len(base)))
        if belief_id not in base:
            hypothesis = next((item for item in proposal.hypotheses if item.id == belief_id), None)
            if hypothesis is None:
                continue
            base[belief_id] = {"id": belief_id, "statement": hypothesis.statement,
                               "confidence": hypothesis.confidence, "status": "new", "evidence": "New proposal",
                               "object_role": hypothesis.object_role, "rule": hypothesis.rule,
                               "goal_condition": hypothesis.goal_condition, "falsification": hypothesis.falsification,
                               "role": hypothesis.role}
        current = base[belief_id]
        current["status"] = str(row["status"])
        current["confidence"] = min(1.0, max(0.0, float(current.get("confidence", 0.5)) + float(row["confidence_delta"])))
        current["evidence"] = str(row["evidence"])
    normalized = {**payload, "beliefs": [base[key] for key in sorted(base)[:16]]}
    return parse_reflection(normalized)
