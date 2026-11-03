from __future__ import annotations

import json
import math
import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .arc_environment import ArcEnvironment
from .arc_models import (
    ArcActionCandidate, ArcBelief, ArcComparisonTrace, ArcEpisodeTrace, ArcGroundedGoal,
    ArcHypothesis, ArcProposal, ArcReflection, ArcTurn,
)
from .arc_reasoner import frame_context, frame_image_data_url
from .models import ActionScore
from .search import ClassicalSearch, QAOASearch


HARNESS_VERSION = "v10-success-conditioned-revision"
INPUT_REPRESENTATION = "tracked-objects+canonical-atoms+success-transition-audit-v4"
ATOM_KINDS = ("role", "relation", "effect")
MOVES = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}


@dataclass(frozen=True)
class EntityFact:
    id: str
    color: int
    size: int
    bounds: tuple[int, int, int, int]
    center: tuple[int, int]


@dataclass(frozen=True)
class SemanticAtom:
    key: str
    kind: str
    subject: str
    value: str
    target: str | None
    prior: float
    source: str


@dataclass(frozen=True)
class WorkingHypothesis:
    statement: str
    focus_entity_ids: tuple[str, ...]
    target_candidate_id: str
    expected_effect: str
    success_signal: str
    falsification: str


@dataclass(frozen=True)
class Vocabulary:
    summary: str
    atoms: tuple[SemanticAtom, ...]
    candidates: tuple[ArcActionCandidate, ...]
    recommended_candidate_id: str
    recommendation_reason: str
    working_hypothesis: WorkingHypothesis | None = None


@dataclass(frozen=True)
class SuccessAudit:
    verdict: str
    original_hypothesis: str
    predicted_effect: str
    observed_effect: str
    supported_atom_keys: tuple[str, ...]
    contradicted_atom_keys: tuple[str, ...]
    corrected_hypothesis: str
    transferable_relation: str
    interface_change: str
    next_level_test: str
    alternatives: tuple[str, ...]
    confidence: float


VOCABULARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "roles": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "entity_id": {"type": "string"}, "role": {"type": "string"}, "prior": {"type": "number"}},
            "required": ["entity_id", "role", "prior"], "additionalProperties": False}},
        "relations": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "subject_id": {"type": "string"}, "relation": {"type": "string"}, "target_id": {"type": "string"}, "prior": {"type": "number"}},
            "required": ["subject_id", "relation", "target_id", "prior"], "additionalProperties": False}},
        "effects": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "target_id": {"type": "string"}, "effect": {"type": "string"}, "prior": {"type": "number"}},
            "required": ["target_id", "effect", "prior"], "additionalProperties": False}},
        "working_hypothesis": {"type": "object", "properties": {
            "statement": {"type": "string"},
            "focus_entity_ids": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
            "target_candidate_id": {"type": "string"},
            "expected_effect": {"type": "string"},
            "success_signal": {"type": "string"},
            "falsification": {"type": "string"}},
            "required": ["statement", "focus_entity_ids", "target_candidate_id", "expected_effect", "success_signal", "falsification"],
            "additionalProperties": False},
        "recommended_candidate_id": {"type": "string"}, "recommendation_reason": {"type": "string"},
    },
    "required": ["summary", "roles", "relations", "effects", "working_hypothesis", "recommended_candidate_id", "recommendation_reason"],
    "additionalProperties": False,
}


SUCCESS_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "partially_correct", "incorrect", "coincidental"]},
        "original_hypothesis": {"type": "string"},
        "predicted_effect": {"type": "string"},
        "observed_effect": {"type": "string"},
        "supported_atom_keys": {"type": "array", "items": {"type": "string"}},
        "contradicted_atom_keys": {"type": "array", "items": {"type": "string"}},
        "corrected_hypothesis": {"type": "string"},
        "transferable_relation": {"type": "string"},
        "interface_change": {"type": "string"},
        "next_level_test": {"type": "string"},
        "alternatives": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "original_hypothesis", "predicted_effect", "observed_effect",
                 "supported_atom_keys", "contradicted_atom_keys", "corrected_hypothesis",
                 "transferable_relation", "interface_change", "next_level_test", "alternatives", "confidence"],
    "additionalProperties": False,
}


def extract_entities(frame: tuple[tuple[int, ...], ...], limit: int = 5,
                     previous: tuple[EntityFact, ...] = tuple()) -> tuple[EntityFact, ...]:
    if not frame:
        return tuple()
    height, width = len(frame), len(frame[0])
    background = Counter(value for row in frame for value in row).most_common(1)[0][0]
    remaining = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if value != background}
    components: list[tuple[int, list[tuple[int, int]]]] = []
    while remaining:
        seed = min(remaining, key=lambda point: (point[1], point[0]))
        color = frame[seed[1]][seed[0]]
        remaining.remove(seed)
        stack, points = [seed], []
        while stack:
            x, y = stack.pop(); points.append((x, y))
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                nx, ny = neighbor
                if 0 <= nx < width and 0 <= ny < height and neighbor in remaining and frame[ny][nx] == color:
                    remaining.remove(neighbor); stack.append(neighbor)
        components.append((color, points))
    components.sort(key=lambda item: (-len(item[1]), item[1][0][1], item[1][0][0]))
    current = []
    for color, points in components[:limit]:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        current.append((color, len(points), (min(xs), min(ys), max(xs), max(ys)),
                        (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))))
    if not previous:
        return tuple(EntityFact(f"e{index}", *row) for index, row in enumerate(current))

    # Greedy minimum-cost matching is sufficient for the small (<=5 entity) catalog.
    pairs = []
    diagonal = max(1.0, math.hypot(width, height))
    for current_index, row in enumerate(current):
        color, size, _, center = row
        for old in previous:
            if old.color != color:
                continue
            size_delta = abs(size - old.size) / max(1, size, old.size)
            distance = math.dist(center, old.center) / diagonal
            pairs.append((1.6 * size_delta + distance, current_index, old.id))
    assigned_current: dict[int, str] = {}
    assigned_old: set[str] = set()
    for cost, current_index, old_id in sorted(pairs):
        if cost > 1.05 or current_index in assigned_current or old_id in assigned_old:
            continue
        assigned_current[current_index] = old_id; assigned_old.add(old_id)
    used_ids = {item.id for item in previous} | set(assigned_current.values())
    next_id = 1 + max((int(item.id[1:]) for item in previous if item.id[1:].isdigit()), default=-1)
    facts = []
    for index, row in enumerate(current):
        entity_id = assigned_current.get(index)
        if entity_id is None:
            while f"e{next_id}" in used_ids:
                next_id += 1
            entity_id = f"e{next_id}"; used_ids.add(entity_id); next_id += 1
        facts.append(EntityFact(entity_id, *row))
    return tuple(facts)


def observed_relations(entities: tuple[EntityFact, ...]) -> list[str]:
    rows = []
    for left in entities:
        for right in entities:
            if left.id == right.id:
                continue
            lx, ly = left.center; rx, ry = right.center
            horizontal = "left_of" if lx < rx else "right_of" if lx > rx else "x_aligned"
            vertical = "above" if ly < ry else "below" if ly > ry else "y_aligned"
            rows.append(f"{left.id} {horizontal} {right.id}; {left.id} {vertical} {right.id}")
    return rows


def experiment_catalog(observation, entities: tuple[EntityFact, ...], limit: int = 8) -> tuple[ArcActionCandidate, ...]:
    candidates = []
    for action in observation.available_actions:
        if action in MOVES:
            candidates.append(ArcActionCandidate(f"move-{action[-1]}", action, action.lower(), expected_progress=.35, risk=.15))
        elif action == "ACTION5":
            candidates.append(ArcActionCandidate("interact", action, "interact with the current object", expected_progress=.45, risk=.2))
        elif action == "ACTION6":
            for entity in entities:
                candidates.append(ArcActionCandidate(f"click-{entity.id}", action, f"click {entity.id} at {entity.center}",
                                                     x=entity.center[0], y=entity.center[1], expected_progress=.4, risk=.2))
    return tuple(candidates[:limit])


def _clean_word(value: Any, fallback: str) -> str:
    text = "_".join(str(value).lower().strip().split())
    return "".join(char for char in text if char.isalnum() or char == "_")[:40] or fallback


_CONCEPT_TERMS = {
    "role": {
        "barrier": {"barrier", "boundary", "divider", "wall"},
        "goal": {"destination", "goal", "objective", "target"},
        "player": {"agent", "avatar", "player"},
        "key": {"key", "switch", "trigger"},
        "obstacle": {"blocker", "hazard", "obstacle"},
    },
    "relation": {
        "touches": {"adjacent", "contact", "contacts", "touch", "touches"},
        "matches": {"align", "aligned", "aligns", "match", "matches", "same"},
        "enters": {"enter", "enters", "inside", "overlap", "overlaps"},
    },
    "effect": {
        "blocks_movement": {"barrier", "block", "blocks", "movement", "redirect", "redirects"},
        "completes": {"complete", "completes", "finish", "finishes", "solve", "solves"},
        "opens": {"open", "opens", "unlock", "unlocks"},
        "moves": {"move", "moves", "shift", "shifts", "travel", "travels"},
        "changes": {"change", "changes", "transform", "transforms"},
    },
}
_SEMANTIC_STOPWORDS = {"a", "an", "and", "as", "at", "be", "central", "for", "it", "may",
                       "of", "or", "the", "to", "when", "with"}


def canonical_atom_value(kind: str, value: Any) -> str:
    """Map paraphrases to a stable, deliberately small semantic identity."""
    cleaned = _clean_word(value, "unknown")
    tokens = set(re.findall(r"[a-z0-9]+", cleaned.replace("_", " ")))
    matches = [name for name, terms in _CONCEPT_TERMS.get(kind, {}).items() if tokens & terms]
    if matches:
        # Effects may mention movement incidentally; prefer the more specific causal concept.
        return sorted(matches, key=lambda name: (-len(_CONCEPT_TERMS[kind][name] & tokens), name))[0]
    compact = sorted(token for token in tokens if token not in _SEMANTIC_STOPWORDS)
    return "_".join(compact)[:40] or "unknown"


def _atom_key(kind: str, subject: str, value: Any, target: str | None = None) -> str:
    canonical = canonical_atom_value(kind, value)
    return f"{kind}:{subject}:{canonical}" + (f":{target}" if target is not None else "")


def expand_atoms(raw: dict[str, Any], entities: tuple[EntityFact, ...]) -> tuple[SemanticAtom, ...]:
    valid_ids = {entity.id for entity in entities}
    groups: dict[str, list[SemanticAtom]] = {kind: [] for kind in ATOM_KINDS}
    def add(atom: SemanticAtom) -> None:
        if atom.subject not in valid_ids or (atom.target is not None and atom.target not in valid_ids):
            return
        for index, existing in enumerate(groups[atom.kind]):
            if existing.key == atom.key:
                if atom.prior > existing.prior:
                    groups[atom.kind][index] = atom
                return
        groups[atom.kind].append(atom)
    for row in raw.get("roles", []):
        subject, value = str(row.get("entity_id", "")), _clean_word(row.get("role"), "unknown")
        add(SemanticAtom(_atom_key("role", subject, value), "role", subject, value, None, float(row.get("prior", .5)), "luna"))
    for row in raw.get("relations", []):
        subject, target = str(row.get("subject_id", "")), str(row.get("target_id", ""))
        value = _clean_word(row.get("relation"), "related")
        add(SemanticAtom(_atom_key("relation", subject, value, target), "relation", subject, value, target, float(row.get("prior", .5)), "luna"))
    for row in raw.get("effects", []):
        subject, value = str(row.get("target_id", "")), _clean_word(row.get("effect"), "changes")
        add(SemanticAtom(_atom_key("effect", subject, value), "effect", subject, value, None, float(row.get("prior", .5)), "luna"))
    # Generic mutations ensure the search space is not identical to Luna's complete interpretation.
    if entities:
        primary = entities[0].id
        for value in ("goal", "key", "obstacle"):
            add(SemanticAtom(f"role:{primary}:{value}", "role", primary, value, None, .15, "python_mutation"))
    if len(entities) > 1:
        first, second = entities[0].id, entities[1].id
        for value in ("touches", "matches", "enters"):
            add(SemanticAtom(f"relation:{first}:{value}:{second}", "relation", first, value, second, .12, "python_mutation"))
        for target in (first, second):
            for value in ("moves", "opens", "completes"):
                add(SemanticAtom(f"effect:{target}:{value}", "effect", target, value, None, .12, "python_mutation"))
    fallbacks = {
        "role": SemanticAtom("role:e0:unknown", "role", "e0", "unknown", None, 0, "fallback"),
        "relation": SemanticAtom("relation:e0:changes:e0", "relation", "e0", "changes", "e0", 0, "fallback"),
        "effect": SemanticAtom("effect:e0:changes", "effect", "e0", "changes", None, 0, "fallback"),
    }
    result = []
    for kind in ATOM_KINDS:
        rows = groups[kind]
        luna = sorted((item for item in rows if item.source == "luna"), key=lambda item: -max(0, min(1, item.prior)))
        expanded = sorted((item for item in rows if item.source != "luna"), key=lambda item: -max(0, min(1, item.prior)))
        chosen = luna[:2]
        if expanded:
            chosen.append(expanded[0])
        chosen.extend(item for item in luna[2:] + expanded[1:] if item not in chosen)
        result.extend((chosen or [fallbacks[kind]])[:3])
    return tuple(result)


class CompositionalReasoner:
    def __init__(self, caller: Any):
        self.caller = caller
        self.cache: dict[str, Vocabulary] = {}
        self.success_audit_cache: dict[str, SuccessAudit] = {}
        self.last_call_count = 0
        self.last_audit_call_count = 0
        self.previous_entities: tuple[EntityFact, ...] = tuple()
        self.entity_cache: dict[str, tuple[EntityFact, ...]] = {}
        self.success_audits: list[SuccessAudit] = []
        self.level_start_observation: Any | None = None

    def reset_episode(self) -> None:
        self.previous_entities = tuple()
        self.entity_cache = {}
        self.success_audits = []
        self.level_start_observation = None

    def begin_level(self, observation: Any) -> None:
        self.level_start_observation = observation

    def audit_success(self, before: Any, after: Any, statement: str,
                      selected_atoms: tuple[SemanticAtom, ...], selected: ArcActionCandidate,
                      history: list[dict[str, Any]]) -> SuccessAudit:
        """Audit the active explanation instead of crediting it merely because its action succeeded."""
        atom_rows = [asdict(atom) for atom in selected_atoms]
        episode = [row for row in history if int(row.get("level_before", before.levels_completed)) == before.levels_completed]
        actions = [{key: row.get(key) for key in ("turn", "candidate_id", "action_type", "x", "y",
                   "composed_statement", "meaningful_change", "changed_cells")} for row in episode[-12:]]
        valid_keys = {atom.key for atom in selected_atoms}
        cache_key = json.dumps({"level_start": (self.level_start_observation.frame if self.level_start_observation else before.frame),
                                "before": before.frame, "after": after.frame, "statement": statement,
                                "atoms": atom_rows, "action": asdict(selected), "history": actions}, sort_keys=True)
        if cache_key in self.success_audit_cache:
            audit = self.success_audit_cache[cache_key]
            self.last_audit_call_count = 0
            self.success_audits.append(audit)
            self.level_start_observation = after
            return audit
        prompt = f"""A level of an unfamiliar ARC-AGI-3 game has just been completed. Audit the hypothesis that led to the final action; do not assume that a successful action makes its explanation true.

Selected hypothesis: {statement}
Selected semantic components: {atom_rows}
Final action: {asdict(selected)}
Recent actions in this level: {actions}
Before: state={before.state}; levels={before.levels_completed}/{before.win_levels}
After: state={after.state}; levels={after.levels_completed}/{after.win_levels}; event={after.event}

Compare the images and exact grids. State which components of the selected hypothesis were supported or contradicted, whether success was accidental relative to that explanation, and a corrected relational hypothesis. Separate the causal game rule from persistent interface changes such as a score or progress indicator. Generalize by visible attributes and relationships, not stale entity IDs or coordinates. Propose one test for the new level and retain plausible alternatives when causality is ambiguous.

Attached images are ordered: level start, immediately before success, transition frames in chronological order, final post-success state.

Episode-start grid:
{frame_context(self.level_start_observation.frame if self.level_start_observation else before.frame)}
Immediately-before grid:
{frame_context(before.frame)}
Post-success grid:
{frame_context(after.frame)}

Only return atom keys from the selected semantic components in supported_atom_keys and contradicted_atom_keys.
"""
        frames = []
        for frame in ((self.level_start_observation.frame if self.level_start_observation else before.frame),
                      before.frame, *after.animation_frames[-4:], after.frame):
            if frame and frame not in frames:
                frames.append(frame)
        images = [image for frame in frames if (image := frame_image_data_url(frame))]
        raw = self.caller.call_json(prompt, SUCCESS_AUDIT_SCHEMA, "arc_success_hypothesis_audit", images)
        self.last_audit_call_count = 1
        supported = tuple(key for key in raw.get("supported_atom_keys", []) if key in valid_keys)
        contradicted = tuple(key for key in raw.get("contradicted_atom_keys", []) if key in valid_keys and key not in supported)
        audit = SuccessAudit(
            verdict=str(raw.get("verdict", "partially_correct")),
            original_hypothesis=str(raw.get("original_hypothesis", statement)),
            predicted_effect=str(raw.get("predicted_effect", "unknown")),
            observed_effect=str(raw.get("observed_effect", after.event)),
            supported_atom_keys=supported,
            contradicted_atom_keys=contradicted,
            corrected_hypothesis=str(raw.get("corrected_hypothesis", "uncertain")),
            transferable_relation=str(raw.get("transferable_relation", "uncertain")),
            interface_change=str(raw.get("interface_change", "none identified")),
            next_level_test=str(raw.get("next_level_test", "re-observe current entities")),
            alternatives=tuple(str(item) for item in raw.get("alternatives", [])[:3]),
            confidence=max(0.0, min(1.0, float(raw.get("confidence", .5)))),
        )
        self.success_audit_cache[cache_key] = audit
        self.success_audits.append(audit)
        self.level_start_observation = after
        return audit

    def entities_for(self, observation) -> tuple[EntityFact, ...]:
        key = frame_signature(observation.frame)
        if key not in self.entity_cache:
            self.entity_cache[key] = extract_entities(observation.frame, previous=self.previous_entities)
        self.previous_entities = self.entity_cache[key]
        return self.previous_entities

    def propose(self, observation, history: list[dict[str, Any]]) -> Vocabulary:
        entities = self.entities_for(observation)
        candidates = experiment_catalog(observation, entities)
        key = json.dumps({"frame": observation.frame, "history": history[-6:],
                          "success_audits": [asdict(item) for item in self.success_audits]}, sort_keys=True)
        if key in self.cache:
            self.last_call_count = 0
            return self.cache[key]
        facts = [asdict(entity) for entity in entities]
        catalog = [{"candidate_id": item.candidate_id, "action": item.action_type, "x": item.x, "y": item.y,
                    "description": item.description} for item in candidates]
        evidence = [{"atoms": row.get("selected_atom_keys"), "result": row.get("meaningful_change"),
                     "evidence_score": row.get("evidence_score"), "levels_delta": row.get("levels_delta", 0)}
                    for row in history[-6:]]
        prompt = f"""You are supplying a compact semantic vocabulary for an unfamiliar ARC-AGI-3 state.
Do not enumerate or grade a catalog of complete explanations. Label visible entity IDs with a few possible roles, relations and effects that could participate in causal explanations, then state one explicit working hypothesis for your recommended experiment. Distinct alternatives are valuable. Priors are weak suggestions in [0,1], not evidence.

Python-observed entities: {facts}
Python-observed spatial relations: {observed_relations(entities)}
Available experiments: {catalog}
Measured prior evidence: {evidence}
Persistent success-conditioned hypothesis audits: {[asdict(item) for item in self.success_audits]}
State={observation.state}; levels={observation.levels_completed}/{observation.win_levels}
Grid data:
{frame_context(observation.frame)}

Use success audits to propose current-level analogues of corrected relational hypotheses, while retaining alternatives when an earlier explanation was uncertain. Return at most 8 entries per vocabulary group. Also state one explicit working hypothesis: what visible entities you are examining, the action you are targeting, the effect you predict, the measurable success signal, and what result would falsify it. Reference only listed entity and candidate IDs. The working hypothesis target_candidate_id and recommended_candidate_id must identify the same physical experiment.
"""
        image = frame_image_data_url(observation.frame)
        raw = self.caller.call_json(prompt, VOCABULARY_SCHEMA, "arc_compositional_vocabulary", [image] if image else None)
        self.last_call_count = 1
        atoms = expand_atoms(raw, entities)
        recommended = str(raw.get("recommended_candidate_id", ""))
        if recommended not in {item.candidate_id for item in candidates}:
            recommended = candidates[0].candidate_id
        hypothesis_raw = raw.get("working_hypothesis", {})
        valid_ids = {entity.id for entity in entities}
        working = WorkingHypothesis(
            statement=str(hypothesis_raw.get("statement", "Unspecified working hypothesis.")),
            focus_entity_ids=tuple(str(item) for item in hypothesis_raw.get("focus_entity_ids", []) if str(item) in valid_ids),
            target_candidate_id=recommended,
            expected_effect=str(hypothesis_raw.get("expected_effect", "Unspecified effect.")),
            success_signal=str(hypothesis_raw.get("success_signal", "Unspecified signal.")),
            falsification=str(hypothesis_raw.get("falsification", "No falsification supplied.")),
        )
        result = Vocabulary(str(raw.get("summary", "")), atoms, candidates, recommended,
                            str(raw.get("recommendation_reason", "")), working)
        self.cache[key] = result
        return result


@dataclass(frozen=True)
class CompositionalEncoding:
    atoms: tuple[SemanticAtom, ...]
    candidates: tuple[ArcActionCandidate, ...]

    @property
    def action_bits(self) -> int:
        return max(1, math.ceil(math.log2(len(self.candidates))))
    @property
    def atom_bits(self) -> int:
        return len(self.atoms)
    @property
    def total_bits(self) -> int:
        return self.atom_bits + self.action_bits
    @property
    def state_count(self) -> int:
        return 1 << self.total_bits
    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)
    def decode(self, index: int) -> tuple[int, str] | None:
        action_mask = (1 << self.action_bits) - 1
        action_index = index & action_mask
        if action_index >= len(self.candidates):
            return None
        atom_mask = index >> self.action_bits
        return atom_mask, self.actions[action_index]
    def selected_atoms(self, atom_mask: int) -> tuple[SemanticAtom, ...]:
        return tuple(atom for bit, atom in enumerate(self.atoms) if atom_mask & (1 << bit))
    def encode(self, atom_mask: int, candidate_id: str) -> int:
        return (atom_mask << self.action_bits) | self.actions.index(candidate_id)


def _direction_compatibility(atoms: tuple[SemanticAtom, ...], candidate: ArcActionCandidate,
                             entities: dict[str, EntityFact]) -> float:
    if candidate.action_type not in MOVES:
        return .15 if candidate.action_type in {"ACTION5", "ACTION6"} else 0
    dx, dy = MOVES[candidate.action_type]
    score = 0.0
    for atom in atoms:
        if atom.kind != "relation" or not atom.target or atom.subject not in entities or atom.target not in entities:
            continue
        sx, sy = entities[atom.subject].center; tx, ty = entities[atom.target].center
        toward = dx * (tx - sx) + dy * (ty - sy)
        if atom.value in {"touches", "enters", "matches", "aligns_with"}:
            score += .25 if toward > 0 else -.15
    return score


def frame_signature(frame: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps(frame, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def color_signature(frame: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, int], ...]:
    """Count colors so translation and changing connectivity do not masquerade as progress."""
    return tuple(sorted(Counter(value for row in frame for value in row).items()))


def color_change_magnitude(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> int:
    """Return the minimum number of cells whose color population changed."""
    before_counts = Counter(value for row in before for value in row)
    after_counts = Counter(value for row in after for value in row)
    return sum(abs(before_counts[color] - after_counts[color]) for color in before_counts | after_counts) // 2


def assess_transition(before, after, history: list[dict[str, Any]]) -> dict[str, Any]:
    changed = sum(a != b for row_a, row_b in zip(before.frame, after.frame) for a, b in zip(row_a, row_b))
    levels_delta = after.levels_completed - before.levels_completed
    before_key, after_key = frame_signature(before.frame), frame_signature(after.frame)
    seen_states = {str(row.get(key, "")) for row in history for key in ("before_state_signature", "after_state_signature")}
    revisited = after_key in seen_states
    color_delta = color_change_magnitude(before.frame, after.frame)
    if levels_delta > 0 or after.success:
        label, score = "progress", 2.0
    elif changed == 0:
        label, score = "no_change", -.55
    elif revisited:
        label, score = "revisited_state", -.35
    elif color_delta:
        # A one-cell trail/toggle is weak; substantial recoloring approaches the full evidence award.
        score = min(.45, .05 + .04 * color_delta)
        label = "novel_color_change" if color_delta >= 5 else "minor_color_change"
    else:
        # Movement is useful exploratory evidence, but not proof that a causal account is correct.
        label, score = "novel_motion", .05
    return {"changed_cells": changed, "levels_delta": levels_delta, "meaningful_change": label,
            "evidence_score": score, "novel_state": not revisited, "color_change": bool(color_delta),
            "color_changed_cells": color_delta,
            "before_state_signature": before_key, "after_state_signature": after_key}


def historical_evidence_score(history: list[dict[str, Any]], index: int) -> float:
    """Downgrade provisional positive evidence when a later action restores the old state."""
    row = history[index]
    score = float(row.get("evidence_score", 2.0 if row.get("levels_delta", 0) > 0 else
                        .05 if row.get("changed_cells", 0) > 0 else -.55))
    if score <= 0 or row.get("levels_delta", 0) > 0:
        return score
    old_state = row.get("before_state_signature")
    if old_state and any(later.get("after_state_signature") == old_state for later in history[index + 1:]):
        return -.20
    return score


def assess_trajectory(before_entities: tuple[EntityFact, ...], after_entities: tuple[EntityFact, ...],
                      candidate: ArcActionCandidate, atoms: tuple[SemanticAtom, ...],
                      history: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"trajectory_status": "not_applicable", "trajectory_score": 0.0,
              "tracked_entity_id": None, "target_entity_id": None, "distance_delta": None,
              "before_center": None, "after_center": None, "frontier_position": False}
    if candidate.action_type not in MOVES:
        return result
    before_map, after_map = ({item.id: item for item in rows} for rows in (before_entities, after_entities))
    dx, dy = MOVES[candidate.action_type]
    movable = []
    for entity_id in before_map.keys() & after_map.keys():
        bx, by = before_map[entity_id].center; ax, ay = after_map[entity_id].center
        displacement = (ax - bx, ay - by)
        projection = displacement[0] * dx + displacement[1] * dy
        movable.append((projection, abs(displacement[0]) + abs(displacement[1]), entity_id, displacement))
    if not movable:
        return {**result, "trajectory_status": "blocked", "trajectory_score": -.40}
    projection, movement, tracked_id, displacement = max(movable)
    before_center, after_center = before_map[tracked_id].center, after_map[tracked_id].center
    relation = next((atom for atom in atoms if atom.kind == "relation"), None)
    target_id = relation.target if relation and relation.target in before_map and relation.target in after_map else None
    distance_delta = None
    if target_id and target_id != tracked_id:
        target_before, target_after = before_map[target_id].center, after_map[target_id].center
        old_distance = math.dist(before_center, target_before)
        new_distance = math.dist(after_center, target_after)
        distance_delta = old_distance - new_distance
    seen_positions = {tuple(row.get("after_center")) for row in history
                      if row.get("tracked_entity_id") == tracked_id and row.get("after_center") is not None}
    frontier = tuple(after_center) not in seen_positions
    repeated = bool(history and history[-1].get("candidate_id") == candidate.candidate_id and
                    history[-1].get("tracked_entity_id") == tracked_id)
    if movement == 0 or projection <= 0:
        status, score = ("blocked", -.40) if repeated else ("unproductive", -.10)
    elif not frontier:
        status, score = "oscillating", -.35
    elif distance_delta is not None and distance_delta < -.25:
        status, score = "moving_away", -.15
    elif repeated:
        status, score = "productive_repetition", .18
    else:
        status, score = "productive_motion", .08
    return {"trajectory_status": status, "trajectory_score": score, "tracked_entity_id": tracked_id,
            "target_entity_id": target_id, "distance_delta": distance_delta,
            "before_center": before_center, "after_center": after_center,
            "displacement": displacement, "frontier_position": frontier}


def action_is_grounded(atoms: tuple[SemanticAtom, ...], candidate: ArcActionCandidate) -> bool:
    """Require a click to target the predicted effect or an entity directly related to it."""
    if candidate.action_type != "ACTION6" or not candidate.candidate_id.startswith("click-"):
        return True
    clicked = candidate.candidate_id.removeprefix("click-")
    effect = next((atom for atom in atoms if atom.kind == "effect"), None)
    relation = next((atom for atom in atoms if atom.kind == "relation"), None)
    role = next((atom for atom in atoms if atom.kind == "role"), None)
    if effect is None:
        return False
    if effect.subject == clicked:
        return True
    if relation and {relation.subject, relation.target} == {clicked, effect.subject}:
        return True
    role_value = canonical_atom_value("role", role.value) if role else "unknown"
    return bool(role and role.subject == clicked and role_value in {"key", "player"} and relation and
                clicked in {relation.subject, relation.target} and effect.subject in {relation.subject, relation.target})


def blocked_no_change_candidates(history: list[dict[str, Any]], threshold: int = 2) -> set[str]:
    """Block clicks that repeatedly produced no effect from the exact current state."""
    if not history:
        return set()
    current = history[-1].get("after_state_signature")
    if not current:
        return set()
    counts: Counter[str] = Counter()
    for row in reversed(history):
        if row.get("levels_delta", 0) > 0:
            break
        if row.get("before_state_signature") != current or row.get("after_state_signature") != current:
            continue
        if row.get("meaningful_change") == "no_change":
            counts[str(row.get("candidate_id", ""))] += 1
    return {candidate_id for candidate_id, count in counts.items() if candidate_id and count >= threshold}


def build_compositional_costs(vocabulary: Vocabulary, entities: tuple[EntityFact, ...], history: list[dict[str, Any]]):
    encoding = CompositionalEncoding(vocabulary.atoms, vocabulary.candidates)
    costs = np.full(encoding.state_count, 20.0)
    valid = np.zeros(encoding.state_count, dtype=bool)
    evidence: Counter[str] = Counter()
    pair_evidence: Counter[tuple[tuple[str, ...], str]] = Counter()
    trajectory_evidence: Counter[str] = Counter()
    stagnant_actions: Counter[str] = Counter()
    reversal_adjustments = 0
    blocked_candidates = blocked_no_change_candidates(history)
    for row_index, row in enumerate(history):
        outcome = historical_evidence_score(history, row_index)
        reversal_adjustments += int(outcome != row.get("evidence_score", outcome))
        for key in row.get("selected_atom_keys", []):
            evidence[key] += float(row.get("atom_evidence", {}).get(key, outcome))
        pair = (tuple(sorted(row.get("selected_atom_keys", []))), str(row.get("candidate_id", "")))
        pair_evidence[pair] += float(row.get("hypothesis_credit", outcome))
        candidate_id = str(row.get("candidate_id", ""))
        trajectory_evidence[candidate_id] += float(row.get("trajectory_score", 0.0))
        if row.get("trajectory_status") in {"blocked", "oscillating", "moving_away", "unproductive"}:
            stagnant_actions[candidate_id] += 1
    entity_map = {item.id: item for item in entities}
    compositions = []
    for mask in range(1 << encoding.atom_bits):
        atoms = encoding.selected_atoms(mask)
        if any(sum(atom.kind == kind for atom in atoms) != 1 for kind in ATOM_KINDS):
            continue
        compositions.append((mask, atoms))
        for candidate in vocabulary.candidates:
            index = encoding.encode(mask, candidate.candidate_id)
            if candidate.candidate_id in blocked_candidates or not action_is_grounded(atoms, candidate):
                continue
            valid[index] = True
            prior = sum(max(0, min(1, atom.prior)) for atom in atoms) / len(atoms)
            atom_keys = tuple(sorted(atom.key for atom in atoms))
            measured = sum(evidence[atom.key] for atom in atoms) / len(atoms)
            paired = pair_evidence[(atom_keys, candidate.candidate_id)]
            values = {atom.kind: canonical_atom_value(atom.kind, atom.value) for atom in atoms}
            coherence = 0.0
            if values["role"] == "key" and values["effect"] in {"opens", "removes", "changes"}: coherence += .35
            if values["role"] == "goal" and values["effect"] in {"completes", "matches", "changes"}: coherence += .4
            if values["role"] == "obstacle" and values["effect"] in {"opens", "removes", "moves"}: coherence += .3
            if values["relation"] in {"touches", "enters", "matches", "aligns_with"}: coherence += .15
            experiment_fit = _direction_compatibility(atoms, candidate, entity_map)
            trajectory = trajectory_evidence[candidate.candidate_id]
            stagnation = .22 * stagnant_actions[candidate.candidate_id]
            follow_up = .55 if history and float(history[-1].get("evidence_score", 0)) >= .40 and \
                history[-1].get("candidate_id") == candidate.candidate_id else 0.0
            # Most evidence remains bound to the exact composition/action that produced it.
            costs[index] = -(0.10 * prior + 0.12 * measured + 0.52 * paired + 0.30 * trajectory +
                             follow_up + coherence + experiment_fit + .15 * candidate.expected_progress) + stagnation
    diagnostics = {"cost_model": "success-audited-action-conditioned-v4", "atom_count": len(vocabulary.atoms),
                   "composition_count": len(compositions), "valid_states": int(valid.sum()),
                   "model_prior_weight": .10, "transferable_atom_evidence_weight": .12,
                   "composition_action_evidence_weight": .52, "trajectory_weight": .30,
                   "reversal_adjustments": reversal_adjustments,
                   "blocked_no_change_candidates": sorted(blocked_candidates),
                   "action_grounding_required": True}
    return encoding, costs, valid, diagnostics


def decode_statement(atoms: tuple[SemanticAtom, ...], candidate: ArcActionCandidate) -> str:
    role = next(atom for atom in atoms if atom.kind == "role")
    relation = next(atom for atom in atoms if atom.kind == "relation")
    effect = next(atom for atom in atoms if atom.kind == "effect")
    return (f"Treat {role.subject} as {role.value}; test whether {relation.subject} {relation.value} "
            f"{relation.target} causes {effect.subject} to {effect.value}, using {candidate.description}.")


def run_compositional_episode(environment: ArcEnvironment, reasoner: CompositionalReasoner, condition: str,
                              source: str, seed: int, max_steps: int, batch_id: str | None,
                              progress_callback: Callable[[dict[str, Any]], None] | None, condition_index: int) -> ArcEpisodeTrace:
    started = perf_counter(); emit = progress_callback or (lambda _event: None)
    trace = ArcEpisodeTrace(condition, source, environment.game_id, seed, action_budget=max_steps,
        model=reasoner.caller.model if source == "arc" else "Deterministic local reasoner",
        reasoning_effort=getattr(reasoner.caller, "reasoning_effort", None), batch_id=batch_id,
        harness_version=HARNESS_VERSION, input_representation=INPUT_REPRESENTATION,
        harness_flags={"compositional_atoms": True, "action_conditioned_trajectories": True,
                       "success_conditioned_hypothesis_revision": True, "model_prior_weight": .10})
    history: list[dict[str, Any]] = []
    try:
        reasoner.reset_episode()
        observation = environment.reset()
        reasoner.begin_level(observation)
        emit({"phase": "condition_started", "condition": condition, "condition_index": condition_index,
              "condition_total": 3, "turn": 0, "max_steps": max_steps})
        for turn_number in range(1, max_steps + 1):
            turn_started = perf_counter(); before = observation
            emit({"phase": "reasoning", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps})
            stage = perf_counter(); vocabulary = reasoner.propose(observation, history)
            proposal_seconds = perf_counter() - stage; model_calls = reasoner.last_call_count
            entities = reasoner.entities_for(observation)
            stage = perf_counter(); encoding, costs, valid, cost_diagnostics = build_compositional_costs(vocabulary, entities, history)
            objective_seconds = perf_counter() - stage
            scores = [ActionScore(candidate.candidate_id, 0, candidate.expected_progress, candidate.expected_progress)
                      for candidate in vocabulary.candidates]
            emit({"phase": "selection", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps})
            stage = perf_counter()
            if condition == "llm_only":
                candidate_id = vocabulary.recommended_candidate_id
                indices = [index for index in np.flatnonzero(valid) if encoding.decode(int(index))[1] == candidate_id]
                recommendation_grounded = bool(indices)
                if indices:
                    selected_index = min(indices, key=lambda index: costs[index])
                else:
                    selected_index = min(np.flatnonzero(valid), key=lambda index: costs[index])
                    decoded = encoding.decode(int(selected_index))
                    if decoded is None:
                        raise RuntimeError("No grounded hypothesis/action pair is available.")
                    _, candidate_id = decoded
                atom_mask, _ = encoding.decode(int(selected_index))
                diagnostics = {"backend": "llm_recommendation", "quantum_used": False,
                    "selected_cost": float(costs[selected_index]), "regret": float(costs[selected_index] - min(costs[valid])),
                    "selected_rank": int(1 + sum(costs[index] < costs[selected_index] for index in np.flatnonzero(valid))),
                    "recommendation_grounded": recommendation_grounded}
            else:
                backend = ClassicalSearch() if condition == "qahs_classical" else QAOASearch(depth=2, restarts=3, maxiter=80, shots=512, seed=seed + turn_number)
                result = backend.select(encoding, scores, costs, valid)
                candidate_id, atom_mask = str(result.action), int(result.hypothesis_id)
                diagnostics = {**result.diagnostics, "backend": result.backend, "quantum_used": condition == "qahs_qaoa"}
            selection_seconds = perf_counter() - stage
            selected = next(item for item in vocabulary.candidates if item.candidate_id == candidate_id)
            selected_atoms = encoding.selected_atoms(atom_mask)
            statement = decode_statement(selected_atoms, selected)
            hypothesis = ArcHypothesis(0, statement, 0.0, {candidate_id: "UNKNOWN"}, role="interaction",
                                       grounded_goal=ArcGroundedGoal(proposed_action=selected.action_type))
            proposal = ArcProposal(vocabulary.summary, (hypothesis,), vocabulary.candidates,
                                   vocabulary.recommended_candidate_id, vocabulary.recommendation_reason)
            diagnostics.update({**cost_diagnostics, "atom_mask": atom_mask,
                "selected_atoms": [asdict(atom) for atom in selected_atoms], "composed_statement": statement,
                "atom_catalog": [asdict(atom) for atom in vocabulary.atoms],
                "entity_catalog": [asdict(entity) for entity in entities],
                "luna_working_hypothesis": asdict(vocabulary.working_hypothesis) if vocabulary.working_hypothesis else None,
                "model_recommended_candidate_id": vocabulary.recommended_candidate_id,
                "selector_candidate_id": candidate_id,
                "override_opportunity": candidate_id != vocabulary.recommended_candidate_id,
                "override_permitted": condition != "llm_only" and candidate_id != vocabulary.recommended_candidate_id,
                "decision_qubits": encoding.total_bits})
            emit({"phase": "environment", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps})
            stage = perf_counter(); observation = environment.step(selected); environment_seconds = perf_counter() - stage
            after_entities = reasoner.entities_for(observation)
            transition = assess_transition(before, observation, history)
            trajectory = assess_trajectory(entities, after_entities, selected, selected_atoms, history)
            diagnostics.update({"trajectory_status": trajectory["trajectory_status"],
                "trajectory_score": trajectory["trajectory_score"],
                "tracked_entity_id": trajectory["tracked_entity_id"],
                "target_entity_id": trajectory["target_entity_id"],
                "distance_delta": trajectory["distance_delta"],
                "frontier_position": trajectory["frontier_position"],
                "color_changed_cells": transition["color_changed_cells"],
                "after_entity_catalog": [asdict(entity) for entity in after_entities]})
            cells_changed = int(transition["changed_cells"])
            levels_delta = int(transition["levels_delta"])
            meaningful = str(transition["meaningful_change"])
            atom_evidence = float(transition["evidence_score"])
            success_audit = None
            reflection_seconds = 0.0
            if levels_delta > 0 or observation.success:
                emit({"phase": "reviewing", "condition": condition, "condition_index": condition_index,
                      "condition_total": 3, "turn": turn_number, "max_steps": max_steps})
                stage = perf_counter()
                success_audit = reasoner.audit_success(before, observation, statement, selected_atoms, selected, history)
                reflection_seconds = perf_counter() - stage
                model_calls += reasoner.last_audit_call_count
                diagnostics["success_hypothesis_audit"] = asdict(success_audit)
            verdict = success_audit.verdict if success_audit else None
            if success_audit:
                component_evidence = {
                    atom.key: (2.0 if atom.key in success_audit.supported_atom_keys else
                               -1.0 if atom.key in success_audit.contradicted_atom_keys else .15)
                    for atom in selected_atoms
                }
                hypothesis_credit = {"supported": 2.0, "partially_correct": .75,
                                     "incorrect": 0.0, "coincidental": 0.0}[verdict]
            else:
                component_evidence = {atom.key: float(transition["evidence_score"]) for atom in selected_atoms}
                hypothesis_credit = float(transition["evidence_score"])
            belief_score = hypothesis_credit if success_audit else atom_evidence
            belief_status = ("supported" if belief_score > 0 else
                             "rejected" if success_audit and verdict in {"incorrect", "coincidental"} else "weakened")
            reflection = ArcReflection(
                summary=(f"Measured {meaningful} ({trajectory['trajectory_status']}); "
                         + (f"success audit marked the hypothesis {verdict}." if success_audit else
                            f"updated the selected composition by {hypothesis_credit:+.2f}.")),
                observed_effect=observation.event,
                beliefs=(ArcBelief(0, statement, min(1, max(0, .5 + belief_score / 4)),
                    belief_status, observation.event, role="interaction"),),
                next_focus="Select a composition that remains coherent with measured evidence.")
            history.append({"turn": turn_number, "candidate_id": candidate_id, "action_type": selected.action_type,
                "x": selected.x, "y": selected.y, "selected_atom_keys": [atom.key for atom in selected_atoms],
                "composed_statement": statement, "level_before": before.levels_completed,
                "atom_evidence": component_evidence, "hypothesis_credit": hypothesis_credit,
                "success_hypothesis_audit": asdict(success_audit) if success_audit else None,
                **transition, **trajectory, "state": observation.state})
            trace.turns.append(ArcTurn(turn_number, before, proposal, condition, selected,
                [asdict(row) for row in scores], diagnostics, observation, reflection, model_calls,
                {"proposal_seconds": proposal_seconds, "objective_seconds": objective_seconds,
                 "selection_seconds": selection_seconds, "environment_seconds": environment_seconds,
                 "reflection_seconds": reflection_seconds, "turn_seconds": perf_counter() - turn_started}))
            emit({"phase": "turn_completed", "condition": condition, "condition_index": condition_index,
                  "condition_total": 3, "turn": turn_number, "max_steps": max_steps,
                  "state": observation.state, "levels": observation.levels_completed})
            if observation.terminal:
                break
        trace.success = observation.success
    except Exception as error:
        trace.error = f"{type(error).__name__}: run failed; internal details omitted"
    trace.wall_seconds = perf_counter() - started
    trace.intervention_summary = {"opportunities": sum(bool(t.search_diagnostics.get("override_opportunity")) for t in trace.turns),
        "permitted": sum(bool(t.search_diagnostics.get("override_permitted")) for t in trace.turns), "blocked": 0,
        "helpful": 0, "neutral": 0, "harmful": 0}
    emit({"phase": "condition_completed", "condition": condition, "condition_index": condition_index,
          "condition_total": 3, "turn": len(trace.turns), "max_steps": max_steps,
          "success": trace.success, "error": bool(trace.error)})
    return trace


def run_compositional_comparison(environment_factory: Callable[[], ArcEnvironment], reasoner: CompositionalReasoner,
                                 source: str, game_id: str, seed: int, max_steps: int,
                                 batch_id: str | None, progress_callback=None) -> ArcComparisonTrace:
    episodes = [run_compositional_episode(environment_factory(), reasoner, condition, source, seed, max_steps,
                                           batch_id, progress_callback, index)
                for index, condition in enumerate(("llm_only", "qahs_classical", "qahs_qaoa"))]
    trace = ArcComparisonTrace(source, game_id, seed, *episodes, action_budget=max_steps,
        model=reasoner.caller.model if source == "arc" else "Deterministic local reasoner",
        reasoning_effort=getattr(reasoner.caller, "reasoning_effort", None), batch_id=batch_id,
        harness_flags={"compositional_atoms": True, "action_conditioned_trajectories": True,
                       "success_conditioned_hypothesis_revision": True, "model_prior_weight": .10})
    trace.harness_version = HARNESS_VERSION
    trace.input_representation = INPUT_REPRESENTATION
    return trace
