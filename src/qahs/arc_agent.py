from __future__ import annotations

import json
import hashlib
import os
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

from .arc_encoding import ArcEncoding
from .arc_environment import ArcEnvironment
from .arc_models import ArcComparisonTrace, ArcEpisodeTrace, ArcTurn
from .arc_objective import action_signature, build_arc_costs, score_arc_actions
from .arc_reasoner import ArcReasoner, CachingArcReasoner
from .search import ClassicalSearch, QAOASearch


CONDITIONS = ("llm_only", "qahs_classical", "qahs_qaoa")


def frame_signature(frame: tuple[tuple[int, ...], ...]) -> str:
    payload = ";".join(",".join(str(cell) for cell in row) for row in frame)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def changed_cell_count(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> int:
    return sum(
        old != new
        for old_row, new_row in zip(before, after)
        for old, new in zip(old_row, new_row)
    )


def run_arc_episode(
    environment: ArcEnvironment,
    reasoner: ArcReasoner,
    condition: str,
    source: str,
    seed: int = 0,
    max_steps: int = 12,
) -> ArcEpisodeTrace:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    started = perf_counter()
    trace = ArcEpisodeTrace(condition, source, environment.game_id, seed)
    history: list[dict] = []
    try:
        observation = environment.reset()
        for turn_number in range(1, max_steps + 1):
            proposal = reasoner.propose(observation, history)
            model_calls = int(getattr(reasoner, "last_call_count", 1))
            before_signature = frame_signature(observation.frame)
            scores = score_arc_actions(proposal, history, before_signature)
            score_rows = [asdict(row) for row in scores]

            if condition == "llm_only":
                candidate_id = proposal.recommended_candidate_id
                diagnostics = {
                    "backend": "llm_recommendation",
                    "reason": proposal.recommendation_reason,
                    "quantum_used": False,
                }
            else:
                encoding = ArcEncoding(proposal.hypotheses, proposal.action_candidates)
                costs, valid = build_arc_costs(encoding, proposal, scores)
                backend = ClassicalSearch() if condition == "qahs_classical" else QAOASearch(
                    depth=2, restarts=3, maxiter=80, shots=512, seed=seed + turn_number
                )
                result = backend.select(encoding, scores, costs, valid)
                candidate_id = str(result.action)
                diagnostics = {
                    **result.diagnostics,
                    "backend": result.backend,
                    "quantum_used": condition == "qahs_qaoa",
                    "hypothesis_id": result.hypothesis_id,
                    "decision_qubits": encoding.total_bits,
                }

            selected = next(candidate for candidate in proposal.action_candidates if candidate.candidate_id == candidate_id)
            before = observation
            observation = environment.step(selected)
            reflection = reasoner.reflect(before, observation, proposal, selected, history)
            model_calls += int(getattr(reasoner, "last_call_count", 1))
            cells_changed = changed_cell_count(before.frame, observation.frame)
            belief_ledger = [asdict(item) for item in reflection.beliefs]
            history.append({
                "turn": turn_number,
                "candidate_id": selected.candidate_id,
                "action_type": selected.action_type,
                "x": selected.x,
                "y": selected.y,
                "action_signature": action_signature(selected),
                "before_frame_signature": before_signature,
                "after_frame_signature": frame_signature(observation.frame),
                "changed_cells": cells_changed,
                "levels_delta": observation.levels_completed - before.levels_completed,
                "hypotheses": [
                    {"id": item.id, "statement": item.statement, "confidence": item.confidence}
                    for item in proposal.hypotheses
                ],
                "reflection": reflection.summary,
                "belief_ledger": belief_ledger,
                "next_focus": reflection.next_focus,
                "result": observation.event,
                "state": observation.state,
            })
            trace.turns.append(ArcTurn(
                turn=turn_number,
                observation_before=before,
                proposal=proposal,
                condition=condition,
                selected_candidate=selected,
                action_scores=score_rows,
                search_diagnostics=diagnostics,
                observation_after=observation,
                reflection=reflection,
                model_calls=model_calls,
            ))
            if observation.terminal:
                break
        trace.success = observation.success
        if observation.success:
            trace.stop_reason = "success"
        elif observation.terminal:
            trace.stop_reason = "game_over"
        else:
            trace.stop_reason = "budget_exhausted"
    except Exception as error:
        # Private traces keep full details; export_dashboard_runs.py redacts them before publishing.
        trace.error = f"{type(error).__name__}: {error}"
        trace.error_traceback = traceback.format_exc()
        trace.stop_reason = "error"
    trace.wall_seconds = perf_counter() - started
    return trace


def run_arc_comparison(
    environment_factory: Callable[[], ArcEnvironment],
    reasoner_factory: Callable[[], ArcReasoner],
    source: str,
    game_id: str,
    seed: int = 0,
    max_steps: int = 12,
) -> ArcComparisonTrace:
    reasoner = reasoner_factory()
    shared_reasoner = CachingArcReasoner(reasoner)
    # Rotate which condition fills the shared cache first so no condition is always first.
    shift = seed % len(CONDITIONS)
    order = list(CONDITIONS[shift:] + CONDITIONS[:shift])
    episodes = {
        condition: run_arc_episode(environment_factory(), shared_reasoner, condition, source, seed, max_steps)
        for condition in order
    }
    return ArcComparisonTrace(
        source, game_id, seed,
        episodes["llm_only"], episodes["qahs_classical"], episodes["qahs_qaoa"],
        action_budget=max_steps,
        model=os.environ.get("OPENAI_MODEL") if source == "arc" else "Deterministic local reasoner",
        model_settings=dict(getattr(getattr(reasoner, "caller", None), "settings", {}) or {}),
        condition_order=order,
    )


def save_arc_trace(trace: ArcComparisonTrace | ArcEpisodeTrace, directory: str | Path = "output/arc_runs") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "comparison" if isinstance(trace, ArcComparisonTrace) else trace.condition
    path = directory / f"{timestamp}_{trace.game_id}_{kind}.json"
    path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
    return path
