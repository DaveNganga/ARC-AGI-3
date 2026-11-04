from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .arc_encoding import ArcEncoding
from .arc_models import ArcActionCandidate, ArcGroundedGoal, ArcHypothesis
from .models import ActionScore
from .providers.openai_provider import OpenAIJSONCaller
from .search import ClassicalSearch, QAOASearch

HARNESS_VERSION = "arc2-v1-shared-catalog"
LUNA_MEDIUM_TASKS = (
    "136b0064", "1818057f", "291dc1e1", "2ba387bc", "31f7f899", "3dc255db",
    "45a5af55", "58f5dbd5", "8f3a5a89", "981571dc", "b5ca7ac4", "b99e7126",
    "bf45cf4b", "c4d067a0", "d59b0160", "db0c5428", "e8686506",
)

ARC2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "hypotheses": {"type": "array", "minItems": 4, "maxItems": 8, "items": {
            "type": "object", "properties": {
                "id": {"type": "integer"}, "statement": {"type": "string"},
                "confidence": {"type": "number"}, "training_fit": {"type": "number"},
                "candidate_support": {"type": "array", "items": {"type": "object", "properties": {
                    "candidate_id": {"type": "string"}, "support": {"type": "number"}},
                    "required": ["candidate_id", "support"], "additionalProperties": False}},
            }, "required": ["id", "statement", "confidence", "training_fit", "candidate_support"],
            "additionalProperties": False}},
        "candidates": {"type": "array", "minItems": 3, "maxItems": 6, "items": {
            "type": "object", "properties": {
                "candidate_id": {"type": "string"}, "explanation": {"type": "string"},
                "outputs": {"type": "array", "items": {"type": "array", "items": {
                    "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 9}}}},
            }, "required": ["candidate_id", "explanation", "outputs"], "additionalProperties": False}},
        "recommended_candidate_id": {"type": "string"}, "recommendation_reason": {"type": "string"},
    },
    "required": ["summary", "hypotheses", "candidates", "recommended_candidate_id", "recommendation_reason"],
    "additionalProperties": False,
}

def _grid_text(grid: list[list[int]]) -> str:
    return "/".join(",".join(str(cell) for cell in row) for row in grid)

def _prompt(task_id: str, task: dict[str, Any]) -> str:
    demonstrations = "\n\n".join(
        f"TRAIN {index}\ninput={_grid_text(pair['input'])}\noutput={_grid_text(pair['output'])}"
        for index, pair in enumerate(task["train"], 1))
    tests = "\n".join(f"TEST {index} input={_grid_text(pair['input'])}" for index, pair in enumerate(task["test"], 1))
    return f"""Solve ARC-AGI-2 task {task_id}. Infer the transformation from the training pairs.
Generate 4-8 genuinely distinct rule hypotheses and 3-6 complete candidate output sets. A candidate output set must contain one rectangular integer grid for every test input, in test order. Do not make cosmetic duplicates.
For each hypothesis, report confidence, estimated fit to every training example, and its support for every candidate. Values are in [0,1]. Recommend the candidate you believe is most likely correct. The QAHS selectors independently compare the same catalog.

{demonstrations}

{tests}
"""

def load_task(root: Path, task_id: str) -> dict[str, Any]:
    if task_id not in LUNA_MEDIUM_TASKS:
        raise ValueError("Choose one of the curated Luna Medium ARC-AGI-2 tasks.")
    return json.loads((root / "datasets" / "arc_agi_2" / "evaluation" / f"{task_id}.json").read_text())

def _normalise(raw: dict[str, Any], task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_count = len(task["test"])
    raw_candidates = list(raw.get("candidates", []))[:6]
    candidates = []
    id_map: dict[str, str] = {}
    for item in raw_candidates:
        outputs = item.get("outputs", [])
        valid = len(outputs) == expected_count and all(isinstance(grid, list) and grid and all(isinstance(row, list) and row for row in grid) for grid in outputs)
        if valid:
            new_id = f"candidate-{len(candidates)}"
            id_map[str(item.get("candidate_id", ""))] = new_id
            candidates.append({"candidate_id": new_id, "explanation": str(item.get("explanation", "")), "outputs": outputs})
    if not candidates:
        raise RuntimeError("Luna returned no usable candidate grids.")
    hypotheses = []
    for index, item in enumerate(list(raw.get("hypotheses", []))[:8]):
        support = {id_map.get(str(row.get("candidate_id")), ""): min(1.0, max(0.0, float(row.get("support", 0)))) for row in item.get("candidate_support", [])}
        hypotheses.append({"id": index, "statement": str(item.get("statement", "")),
            "confidence": min(1.0, max(0.0, float(item.get("confidence", 0)))),
            "training_fit": min(1.0, max(0.0, float(item.get("training_fit", 0)))), "support": support})
    if not hypotheses:
        raise RuntimeError("Luna returned no usable hypotheses.")
    return hypotheses, candidates

def _catalog(hypotheses: list[dict[str, Any]], candidates: list[dict[str, Any]]):
    encoded_hypotheses = tuple(ArcHypothesis(id=item["id"], statement=item["statement"], confidence=item["confidence"], predictions={}, role="mechanism", grounded_goal=ArcGroundedGoal()) for item in hypotheses)
    encoded_candidates = tuple(ArcActionCandidate(candidate_id=item["candidate_id"], action_type="OUTPUT_GRID", description=item["explanation"], expected_progress=.5) for item in candidates)
    encoding = ArcEncoding(encoded_hypotheses, encoded_candidates)
    valid = encoding.valid_mask()
    costs = np.full(encoding.state_count, 8.0)
    for hypothesis in hypotheses:
        for candidate in candidates:
            support = hypothesis["support"].get(candidate["candidate_id"], 0.0)
            costs[encoding.encode(hypothesis["id"], candidate["candidate_id"])] = -(.45 * hypothesis["confidence"] + .35 * hypothesis["training_fit"] + .20 * support)
    scores = [ActionScore(candidate["candidate_id"], 0.0, 0.0, float(-min(costs[encoding.encode(h["id"], candidate["candidate_id"])] for h in hypotheses))) for candidate in candidates]
    return encoding, costs, valid, scores

def _correct(outputs: list[list[list[int]]], task: dict[str, Any]) -> bool:
    return len(outputs) == len(task["test"]) and all(output == pair.get("output") for output, pair in zip(outputs, task["test"]))

def _episode(label: str, candidate: dict[str, Any], task: dict[str, Any], hypotheses: list[dict[str, Any]], diagnostics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    success = _correct(candidate["outputs"], task)
    return {"condition": label, "success": success, "wall_seconds": elapsed, "error": None, "turns": [{
        "turn": 1, "observation_before": {"frame": task["test"][0]["input"], "levels_completed": 0, "event": "test_input"},
        "observation_after": {"frame": candidate["outputs"][0], "levels_completed": int(success), "event": "correct" if success else "incorrect"},
        "selected_candidate": {"candidate_id": candidate["candidate_id"], "action_type": "OUTPUT_GRID", "description": candidate["explanation"]},
        "proposal": {"recommended_candidate_id": diagnostics.get("model_recommended_candidate_id", candidate["candidate_id"]), "summary": "Static transformation catalog", "hypotheses": hypotheses},
        "search_diagnostics": diagnostics, "reflection": None, "model_calls": 1 if label == "llm_only" else 0,
        "timings": {"proposal_seconds": elapsed if label == "llm_only" else 0, "selection_seconds": diagnostics.get("selection_seconds", 0)},
        "candidate_outputs": candidate["outputs"],
    }]}

def run_arc2_comparison(root: Path, task_id: str, model: str, effort: str, batch_id: str | None = None) -> Path:
    task = load_task(root, task_id)
    started = perf_counter()
    raw = OpenAIJSONCaller(model, effort).call_json(_prompt(task_id, task), ARC2_SCHEMA, "arc_agi_2_catalog")
    proposal_seconds = perf_counter() - started
    hypotheses, candidates = _normalise(raw, task)
    encoding, costs, valid, scores = _catalog(hypotheses, candidates)
    by_id = {item["candidate_id"]: item for item in candidates}
    raw_ids = [str(item.get("candidate_id", "")) for item in raw.get("candidates", [])]
    raw_recommended = str(raw.get("recommended_candidate_id", ""))
    recommended = f"candidate-{raw_ids.index(raw_recommended)}" if raw_recommended in raw_ids and raw_ids.index(raw_recommended) < len(candidates) else candidates[0]["candidate_id"]
    base_diag = {"backend": "llm_recommendation", "model_recommended_candidate_id": recommended, "quantum_used": False, "regret": 0.0, "selected_rank": 1}
    episodes = {"without_qahs": _episode("llm_only", by_id[recommended], task, hypotheses, base_diag, proposal_seconds)}
    for key, condition, backend in (("with_classical", "qahs_classical", ClassicalSearch()), ("with_qahs", "qahs_qaoa", QAOASearch(depth=2, restarts=3, maxiter=80, shots=512, seed=7))):
        selection_started = perf_counter()
        result = backend.select(encoding, scores, costs, valid)
        selection_seconds = perf_counter() - selection_started
        diagnostics = {**result.diagnostics, "backend": result.backend, "quantum_used": condition == "qahs_qaoa", "model_recommended_candidate_id": recommended, "selection_seconds": selection_seconds}
        episodes[key] = _episode(condition, by_id[str(result.action)], task, hypotheses, diagnostics, selection_seconds)
    trace = {"benchmark": "arc_agi_2", "source": "arc_agi_2", "game_id": task_id, "seed": 0, "action_budget": 1,
        "model": model, "reasoning_effort": effort, "batch_id": batch_id, "harness_version": HARNESS_VERSION,
        "input_representation": "lossless-grid-json", "catalog": {"summary": raw.get("summary"), "recommendation_reason": raw.get("recommendation_reason"), "hypotheses": hypotheses, "candidates": candidates}, **episodes}
    output = root / "output" / "arc_runs"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output / f"{stamp}__{HARNESS_VERSION}__{batch_id or 'unbatched'}__{task_id}__seed-00__actions-1__{model}__comparison.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path
