from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .arc_encoding import ArcEncoding
from .arc_environment import ArcEnvironment
from .arc_models import ArcBelief, ArcComparisonTrace, ArcEpisodeTrace, ArcReflection, ArcTurn
from .arc_objective import (
    action_signature,
    build_arc_costs,
    grounded_goal_score,
    grounded_goal_tokens,
    score_arc_actions,
)
from .arc_reasoner import ArcReasoner, CachingArcReasoner
from .search import ClassicalSearch, QAOASearch


CONDITIONS = ("llm_only", "qahs_classical", "qahs_qaoa")
MOVEMENT_ACTIONS = frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4"})
MOVEMENT_PERSISTENCE_STEPS = 3
OVERRIDE_MARGIN = 0.30
GROUNDED_OVERRIDE_MARGIN = 0.65
MODEL_COMMITMENT_STEPS = 3


def harness_flags() -> dict[str, bool]:
    return {
        "grounded_goals": os.environ.get("QAHS_GROUNDED_GOALS", "1") != "0",
        "override_safety": os.environ.get("QAHS_OVERRIDE_SAFETY", "1") != "0",
        "commitment_window": os.environ.get("QAHS_COMMITMENT_WINDOW", "1") != "0",
        "shadow_mode": os.environ.get("QAHS_SHADOW_MODE", "0") == "1",
    }


def meaningful_change_kind(before, after, cells_changed: int, levels_delta: int) -> str:
    if after.terminal:
        return "terminal"
    if levels_delta > 0:
        return "progress"
    if cells_changed > 0:
        return "state_change"
    if after.raw_frame_count > 1 and any(frame != before.frame for frame in after.animation_frames):
        return "cosmetic_animation"
    return "no_change"


def safe_override_choice(
    proposal, encoding, costs, valid, selector_candidate_id: str, diagnostics: dict,
    history: list[dict], flags: dict[str, bool], commitment: tuple[str, int | None, int | None, int] | None,
) -> tuple[str, dict, tuple[str, int | None, int | None, int] | None]:
    """Require material evidence before replacing a concrete model policy."""
    model_id = proposal.recommended_candidate_id
    selector_id = selector_candidate_id
    model_candidate = next(item for item in proposal.action_candidates if item.candidate_id == model_id)
    selector_candidate = next(item for item in proposal.action_candidates if item.candidate_id == selector_id)
    model_states = [encoding.encode(h.id, model_id) for h in encoding.hypotheses if valid[encoding.encode(h.id, model_id)]]
    model_index = min(model_states, key=lambda index: costs[index])
    model_hypothesis_id = encoding.decode(model_index)[0]
    paired_model_hypothesis = next(item for item in proposal.hypotheses if item.id == model_hypothesis_id)
    model_hypothesis = max(
        proposal.hypotheses,
        key=lambda item: (
            item.role in {"goal", "interaction", "terminal"},
            item.predictions.get(model_id, "UNKNOWN").upper() in {"PROGRESS", "TERMINAL"},
            grounded_goal_score(item),
            item.confidence,
        ),
    )
    selected_states = [encoding.encode(h.id, selector_id) for h in encoding.hypotheses if valid[encoding.encode(h.id, selector_id)]]
    selector_index = min(selected_states, key=lambda index: costs[index])
    advantage = float(costs[model_index] - costs[selector_index])
    grounded = (
        model_hypothesis.role in {"goal", "interaction", "terminal"}
        and grounded_goal_score(model_hypothesis) >= 0.5
        and model_candidate.expected_progress >= 0.4
    )
    diagnostics.update({
        "model_recommended_candidate_id": model_id,
        "selector_candidate_id": selector_id,
        "model_recommendation_cost": float(costs[model_index]),
        "model_paired_hypothesis_id": paired_model_hypothesis.id,
        "model_grounded_hypothesis_id": model_hypothesis.id,
        "selector_advantage": advantage,
        "model_goal_grounding": round(grounded_goal_score(model_hypothesis), 6),
        "model_recommendation_grounded": grounded,
        "override_opportunity": (selector_candidate.action_type, selector_candidate.x, selector_candidate.y)
        != (model_candidate.action_type, model_candidate.x, model_candidate.y),
    })
    if flags["shadow_mode"]:
        diagnostics.update({"override_permitted": False, "override_category": "shadow", "override_reason": "shadow_mode"})
        return model_id, diagnostics, commitment
    if commitment and flags["commitment_window"]:
        action_type, x, y, remaining = commitment
        if grounded and (model_candidate.action_type, model_candidate.x, model_candidate.y) != (action_type, x, y):
            diagnostics["model_commitment"] = "released_on_grounded_replan"
            commitment = None
        match = next((item for item in proposal.action_candidates if (item.action_type, item.x, item.y) == (action_type, x, y)), None)
        if commitment and match:
            diagnostics.update({"override_permitted": False, "override_category": "commitment", "override_reason": "validated_model_policy"})
            remaining -= 1
            return match.candidate_id, diagnostics, (action_type, x, y, remaining) if remaining > 0 else None
        commitment = None
    if selector_id == model_id:
        diagnostics.update({"override_permitted": False, "override_category": "none", "override_reason": "selector_agreed"})
        return selector_id, diagnostics, commitment
    if not diagnostics["override_opportunity"]:
        diagnostics.update({"override_permitted": False, "override_category": "none", "override_reason": "same_physical_action"})
        return selector_id, diagnostics, commitment
    recent_override = any(item.get("override_permitted") for item in history[-2:])
    threshold = GROUNDED_OVERRIDE_MARGIN if grounded else OVERRIDE_MARGIN
    if recent_override:
        threshold += 0.20
    permitted = not flags["override_safety"] or advantage >= threshold
    selected_hypothesis_id = encoding.decode(selector_index)[0]
    selected_hypothesis = next(item for item in proposal.hypotheses if item.id == selected_hypothesis_id)
    exact_index = diagnostics.get("exact_minimum_index")
    exact_action = encoding.decode(int(exact_index))[1] if exact_index is not None and encoding.decode(int(exact_index)) else selector_id
    category = ("approximation" if selector_id != exact_action else
                "goal" if selected_hypothesis.role in {"goal", "terminal", "interaction"} else "evidence")
    diagnostics.update({
        "override_permitted": permitted,
        "override_category": category if permitted else "blocked",
        "override_reason": "evidence_margin_met" if permitted else "insufficient_evidence_margin",
        "override_required_margin": threshold,
    })
    return (selector_id if permitted else model_id), diagnostics, commitment


def frame_signature(frame: tuple[tuple[int, ...], ...]) -> str:
    payload = ";".join(",".join(str(cell) for cell in row) for row in frame)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def changed_cell_count(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> int:
    return sum(
        old != new
        for old_row, new_row in zip(before, after)
        for old, new in zip(old_row, new_row)
    )


def movement_persistence_is_safe(
    action_type: str,
    cells_changed: int,
    levels_delta: int,
    terminal: bool,
    prediction_matches: bool,
    strategy_reset: bool = False,
) -> bool:
    """Continue only a bounded, visibly productive, predictable movement burst."""
    return (
        action_type in MOVEMENT_ACTIONS
        and cells_changed > 0
        and levels_delta == 0
        and not terminal
        and prediction_matches
        and not strategy_reset
    )


def run_arc_episode(
    environment: ArcEnvironment,
    reasoner: ArcReasoner,
    condition: str,
    source: str,
    seed: int = 0,
    max_steps: int = 12,
    batch_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    condition_index: int = 0,
    condition_total: int = 1,
) -> ArcEpisodeTrace:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    started = perf_counter()
    trace = ArcEpisodeTrace(
        condition,
        source,
        environment.game_id,
        seed,
        action_budget=max_steps,
        model=os.environ.get("OPENAI_MODEL") if source == "arc" else "Deterministic local reasoner",
        reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT") if source == "arc" else None,
        batch_id=batch_id,
        harness_flags=harness_flags(),
    )
    flags = trace.harness_flags
    history: list[dict] = []
    next_proposal = None
    queued_followup: tuple[str, int | None, int | None] | None = None
    movement_persistence: tuple[str, int | None, int | None, int] | None = None
    model_commitment: tuple[str, int | None, int | None, int] | None = None
    emit = progress_callback or (lambda _event: None)
    try:
        observation = environment.reset()
        emit({"phase": "condition_started", "condition": condition, "condition_index": condition_index,
              "condition_total": condition_total, "turn": 0, "max_steps": max_steps})
        for turn_number in range(1, max_steps + 1):
            turn_started = perf_counter()
            stage_started = perf_counter()
            if next_proposal is None:
                emit({"phase": "reasoning", "condition": condition, "condition_index": condition_index,
                      "condition_total": condition_total, "turn": turn_number, "max_steps": max_steps})
                proposal = reasoner.propose(observation, history)
                proposal_seconds = perf_counter() - stage_started
                model_calls = int(getattr(reasoner, "last_call_count", 1))
            else:
                proposal, next_proposal = next_proposal, None
                proposal_seconds, model_calls = 0.0, 0
            before_signature = frame_signature(observation.frame)
            stage_started = perf_counter()
            emit({"phase": "objective", "condition": condition, "condition_index": condition_index,
                  "condition_total": condition_total, "turn": turn_number, "max_steps": max_steps})
            scores = score_arc_actions(proposal, history, before_signature)
            score_rows = [asdict(row) for row in scores]
            encoding = ArcEncoding(proposal.hypotheses, proposal.action_candidates)
            selected_hypotheses = [
                int(turn.search_diagnostics["hypothesis_id"])
                for turn in trace.turns
                if turn.search_diagnostics.get("hypothesis_id") is not None
            ]
            costs, valid, cost_diagnostics = build_arc_costs(
                encoding, proposal, scores, history, selected_hypotheses, max_steps=max_steps
            )
            objective_seconds = perf_counter() - stage_started

            stage_started = perf_counter()
            emit({"phase": "selection", "condition": condition, "condition_index": condition_index,
                  "condition_total": condition_total, "turn": turn_number, "max_steps": max_steps})
            if condition == "llm_only":
                candidate_id = proposal.recommended_candidate_id
                candidate_states = [
                    encoding.encode(hypothesis.id, candidate_id)
                    for hypothesis in encoding.hypotheses
                    if valid[encoding.encode(hypothesis.id, candidate_id)]
                ]
                selected_index = min(candidate_states, key=lambda index: costs[index])
                valid_states = [index for index, is_valid in enumerate(valid) if is_valid]
                exact_index = min(valid_states, key=lambda index: costs[index])
                selected_cost, exact_cost = float(costs[selected_index]), float(costs[exact_index])
                diagnostics = {
                    **cost_diagnostics,
                    "backend": "llm_recommendation",
                    "reason": proposal.recommendation_reason,
                    "quantum_used": False,
                    "evaluated_hypothesis_id": encoding.decode(selected_index)[0],
                    "exact_minimum_cost": exact_cost,
                    "selected_cost": selected_cost,
                    "regret": selected_cost - exact_cost,
                    "selected_rank": int(1 + sum(bool(costs[index] < selected_cost - 1e-12) for index in valid_states)),
                    "exact_minimum_index": exact_index,
                }
            else:
                backend = ClassicalSearch() if condition == "qahs_classical" else QAOASearch(
                    depth=2, restarts=3, maxiter=80, shots=512, seed=seed + turn_number
                )
                result = backend.select(encoding, scores, costs, valid)
                candidate_id = str(result.action)
                diagnostics = {
                    **result.diagnostics,
                    **cost_diagnostics,
                    "backend": result.backend,
                    "quantum_used": condition == "qahs_qaoa",
                    "hypothesis_id": result.hypothesis_id,
                    "decision_qubits": encoding.total_bits,
                }
            valid_states = [index for index, is_valid in enumerate(valid) if is_valid]
            exact_index = min(valid_states, key=lambda index: costs[index])
            exact_decoded = encoding.decode(exact_index)
            diagnostics["exact_minimum_index"] = exact_index
            diagnostics["exact_hypothesis_id"] = exact_decoded[0]
            diagnostics["exact_candidate_id"] = exact_decoded[1]
            diagnostics["candidate_table"] = [
                {
                    "candidate_id": candidate.candidate_id,
                    "action_type": candidate.action_type,
                    "x": candidate.x, "y": candidate.y,
                    "expected_progress": candidate.expected_progress, "risk": candidate.risk,
                    "best_cost": float(min(
                        costs[encoding.encode(hypothesis.id, candidate.candidate_id)]
                        for hypothesis in encoding.hypotheses
                        if valid[encoding.encode(hypothesis.id, candidate.candidate_id)]
                    )),
                }
                for candidate in proposal.action_candidates
            ]
            decision_flags = flags if condition != "llm_only" else {
                **flags, "override_safety": False, "commitment_window": False, "shadow_mode": False,
            }
            candidate_id, diagnostics, model_commitment = safe_override_choice(
                proposal, encoding, costs, valid, candidate_id, diagnostics, history, decision_flags, model_commitment
            )
            model_deference = diagnostics.get("override_category") in {"shadow", "commitment", "blocked"}
            if model_deference:
                queued_followup = None
                movement_persistence = None
            selected_hypothesis_id = int(diagnostics.get("hypothesis_id", diagnostics.get("evaluated_hypothesis_id", 0)))
            selected_hypothesis = next(item for item in proposal.hypotheses if item.id == selected_hypothesis_id)
            family = str(cost_diagnostics.get("canonical_families", {}).get(str(selected_hypothesis_id), f"{selected_hypothesis.role}-{selected_hypothesis_id}"))
            transfer_attempt = False
            if history and int(history[-1].get("levels_delta", 0)) > 0 and not history[-1].get("transfer_attempted"):
                prior = history[-1]
                matching = next((item for item in proposal.action_candidates
                                 if item.action_type == prior.get("action_type") and item.x == prior.get("x") and item.y == prior.get("y")), None)
                if matching is not None:
                    candidate_id = matching.candidate_id
                    transfer_attempt = True
                    family = str(prior.get("goal_family", family))
                    diagnostics["policy_transfer"] = "one_attempt"
            was_followup = False
            if queued_followup is not None and not transfer_attempt and not model_deference:
                matching = next((item for item in proposal.action_candidates
                                 if (item.action_type, item.x, item.y) == queued_followup), None)
                if matching is not None:
                    candidate_id = matching.candidate_id
                    was_followup = True
                    diagnostics["conditional_second_action"] = "executed"
                else:
                    diagnostics["conditional_second_action"] = "invalidated"
                queued_followup = None
            was_persistent = False
            if movement_persistence is not None and not transfer_attempt and not was_followup and not model_deference:
                action_type, x, y, remaining = movement_persistence
                model_candidate = next(item for item in proposal.action_candidates
                                       if item.candidate_id == proposal.recommended_candidate_id)
                if (diagnostics.get("model_recommendation_grounded")
                        and (model_candidate.action_type, model_candidate.x, model_candidate.y) != (action_type, x, y)):
                    movement_persistence = None
                    diagnostics["movement_persistence"] = "released_on_grounded_replan"
                matching = next((item for item in proposal.action_candidates
                                 if (item.action_type, item.x, item.y) == (action_type, x, y)), None)
                if movement_persistence is not None and matching is not None:
                    candidate_id = matching.candidate_id
                    was_persistent = True
                    remaining -= 1
                    movement_persistence = (action_type, x, y, remaining) if remaining > 0 else None
                    diagnostics["movement_persistence"] = "executed"
                    diagnostics["movement_persistence_remaining"] = max(0, remaining)
                else:
                    movement_persistence = None
                    if diagnostics.get("movement_persistence") != "released_on_grounded_replan":
                        diagnostics["movement_persistence"] = "invalidated"
            executed_states = [encoding.encode(item.id, candidate_id) for item in encoding.hypotheses
                               if valid[encoding.encode(item.id, candidate_id)]]
            if executed_states:
                executed_index = min(executed_states, key=lambda index: costs[index])
                selected_hypothesis_id = encoding.decode(executed_index)[0]
                selected_hypothesis = next(item for item in proposal.hypotheses if item.id == selected_hypothesis_id)
                family = str(cost_diagnostics.get("canonical_families", {}).get(
                    str(selected_hypothesis_id), f"{selected_hypothesis.role}-{selected_hypothesis_id}"
                ))
                diagnostics["executed_hypothesis_id"] = selected_hypothesis_id
            if transfer_attempt:
                family = str(history[-1].get("goal_family", family))
            selection_seconds = perf_counter() - stage_started

            selected = next(candidate for candidate in proposal.action_candidates if candidate.candidate_id == candidate_id)
            before = observation
            stage_started = perf_counter()
            emit({"phase": "environment", "condition": condition, "condition_index": condition_index,
                  "condition_total": condition_total, "turn": turn_number, "max_steps": max_steps})
            observation = environment.step(selected)
            environment_seconds = perf_counter() - stage_started
            cells_changed = changed_cell_count(before.frame, observation.frame)
            levels_delta = observation.levels_completed - before.levels_completed
            change_kind = meaningful_change_kind(before, observation, cells_changed, levels_delta)
            frame_cells = max(1, sum(len(row) for row in before.frame))
            expected_outcome = selected_hypothesis.predictions.get(selected.candidate_id, "UNKNOWN").upper()
            actual_outcome = ("TERMINAL" if observation.terminal else "PROGRESS" if levels_delta > 0
                              else "NO_CHANGE" if cells_changed == 0 else "CHANGE")
            prediction_matches = expected_outcome == actual_outcome or (
                actual_outcome == "CHANGE" and expected_outcome in {"MOVE", "CHANGE"}
            )
            diagnostics["meaningful_change"] = change_kind
            diagnostics["intervention_outcome"] = (
                "helpful" if diagnostics.get("override_permitted") and levels_delta > 0 else
                "harmful" if diagnostics.get("override_permitted") and change_kind in {"no_change", "cosmetic_animation"} else
                "neutral" if diagnostics.get("override_permitted") else "not_applicable"
            )
            review_reasons = []
            if turn_number == 1 or (turn_number % 3 == 0 and not was_persistent):
                review_reasons.append("periodic")
            if observation.terminal or levels_delta > 0:
                review_reasons.append("progress")
            if not prediction_matches:
                review_reasons.append("prediction_miss")
            if selected.action_type == "ACTION6":
                review_reasons.append("interaction")
            if cells_changed / frame_cells >= 0.15:
                review_reasons.append("major_change")
            if cost_diagnostics.get("stagnation_pressure", 0) >= 1.0:
                review_reasons.append("strategy_reset")
            stage_started = perf_counter()
            emit({"phase": "review", "condition": condition, "condition_index": condition_index,
                  "condition_total": condition_total, "turn": turn_number, "max_steps": max_steps})
            if not review_reasons and turn_number < max_steps:
                reflection = ArcReflection(
                    "Routine evidence updated locally.", observation.event,
                    tuple(ArcBelief(item.id, item.statement, item.confidence, "supported" if prediction_matches else "uncertain",
                                    f"{selected.candidate_id} produced {actual_outcome}", item.object_role, item.rule,
                                    item.goal_condition, item.falsification, item.role)
                          for item in proposal.hypotheses),
                    "Continue the validated plan until a meaningful event or scheduled review.",
                )
                next_proposal = proposal
                reasoner.last_call_count = 0
                diagnostics["reasoning_trigger"] = "local_routine_update"
                if (not was_followup and not was_persistent and selected.action_type not in MOVEMENT_ACTIONS
                        and selected.risk <= 0.25 and selected.next_action_type in observation.available_actions):
                    queued_followup = (selected.next_action_type, selected.next_x, selected.next_y)
                    diagnostics["conditional_second_action"] = "queued"
            elif observation.terminal or turn_number == max_steps:
                reflection = reasoner.reflect(before, observation, proposal, selected, history)
            else:
                reflection, next_proposal = reasoner.review_and_propose(before, observation, proposal, selected, history)
            if review_reasons:
                diagnostics["reasoning_trigger"] = ",".join(review_reasons)
            reflection_seconds = perf_counter() - stage_started
            model_calls += int(getattr(reasoner, "last_call_count", 1))
            persistence_safe = movement_persistence_is_safe(
                selected.action_type,
                cells_changed,
                levels_delta,
                observation.terminal,
                prediction_matches,
                strategy_reset="strategy_reset" in review_reasons,
            )
            model_candidate = next(item for item in proposal.action_candidates
                                   if item.candidate_id == proposal.recommended_candidate_id)
            if (condition != "llm_only" and flags["commitment_window"] and selected.action_type == model_candidate.action_type
                    and selected.x == model_candidate.x and selected.y == model_candidate.y
                    and diagnostics.get("model_recommendation_grounded") and persistence_safe):
                model_commitment = (selected.action_type, selected.x, selected.y, MODEL_COMMITMENT_STEPS)
                diagnostics["model_commitment"] = "queued"
            elif levels_delta > 0 or observation.terminal or not prediction_matches or change_kind in {"no_change", "cosmetic_animation"}:
                model_commitment = None
            if was_persistent:
                if not persistence_safe:
                    movement_persistence = None
                    diagnostics["movement_persistence"] = "stopped"
            elif persistence_safe:
                movement_persistence = (
                    selected.action_type,
                    selected.x,
                    selected.y,
                    MOVEMENT_PERSISTENCE_STEPS,
                )
                diagnostics["movement_persistence"] = "queued"
                diagnostics["movement_persistence_remaining"] = MOVEMENT_PERSISTENCE_STEPS
            elif selected.action_type not in MOVEMENT_ACTIONS:
                movement_persistence = None
            changed_coordinates = [
                [x, y]
                for y, (before_row, after_row) in enumerate(zip(before.frame, observation.frame))
                for x, (old, new) in enumerate(zip(before_row, after_row))
                if old != new
            ][:64]
            belief_ledger = [asdict(item) for item in reflection.beliefs]
            if levels_delta > 0:
                credit = 0.8
                for prior in reversed(history):
                    if prior.get("goal_family") != family:
                        break
                    prior["progress_credit"] = round(credit, 6)
                    credit *= 0.8
            history.append({
                "turn": turn_number,
                "candidate_id": selected.candidate_id,
                "action_type": selected.action_type,
                "x": selected.x,
                "y": selected.y,
                "next_action_type": selected.next_action_type,
                "next_x": selected.next_x,
                "next_y": selected.next_y,
                "action_signature": action_signature(selected),
                "before_frame_signature": before_signature,
                "after_frame_signature": frame_signature(observation.frame),
                "changed_cells": cells_changed,
                "changed_coordinates": changed_coordinates,
                "levels_delta": levels_delta,
                "hypotheses": [
                    {
                        "id": item.id,
                        "statement": item.statement,
                        "confidence": item.confidence,
                        "predictions": item.predictions,
                        "object_role": item.object_role,
                        "rule": item.rule,
                        "goal_condition": item.goal_condition,
                        "falsification": item.falsification,
                        "role": item.role,
                    }
                    for item in proposal.hypotheses
                ],
                "reflection": reflection.summary,
                "belief_ledger": belief_ledger,
                "next_focus": reflection.next_focus,
                "result": observation.event,
                "state": observation.state,
                "selected_role": selected_hypothesis.role,
                "goal_family": family,
                "goal_signature_tokens": sorted(grounded_goal_tokens(selected_hypothesis)),
                "grounded_goal": asdict(selected_hypothesis.grounded_goal),
                "meaningful_change": change_kind,
                "prediction_matches": prediction_matches,
                "executed_model_recommendation": selected.candidate_id == proposal.recommended_candidate_id,
                "override_permitted": bool(diagnostics.get("override_permitted")),
                "override_category": diagnostics.get("override_category"),
                "intervention_outcome": diagnostics.get("intervention_outcome"),
                "transfer_attempted": transfer_attempt,
                "strategy_reset": bool(cost_diagnostics.get("stagnation_pressure", 0) >= 1.0),
                "movement_persistence": diagnostics.get("movement_persistence"),
                "progress_credit": 1.0 if levels_delta > 0 else 0.0,
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
                timings={
                    "proposal_seconds": proposal_seconds,
                    "objective_seconds": objective_seconds,
                    "selection_seconds": selection_seconds,
                    "environment_seconds": environment_seconds,
                    "reflection_seconds": reflection_seconds,
                    "turn_seconds": perf_counter() - turn_started,
                },
            ))
            emit({"phase": "turn_completed", "condition": condition, "condition_index": condition_index,
                  "condition_total": condition_total, "turn": turn_number, "max_steps": max_steps,
                  "state": observation.state, "levels": observation.levels_completed,
                  "turn_seconds": perf_counter() - turn_started})
            if observation.terminal:
                break
        trace.success = observation.success
    except Exception as error:
        trace.error = f"{type(error).__name__}: run failed; internal details omitted"
    outcomes = [turn.search_diagnostics.get("intervention_outcome") for turn in trace.turns]
    trace.intervention_summary = {
        "opportunities": sum(
            bool(turn.search_diagnostics.get("override_opportunity"))
            for turn in trace.turns
        ),
        "permitted": sum(bool(turn.search_diagnostics.get("override_permitted")) for turn in trace.turns),
        "blocked": sum(turn.search_diagnostics.get("override_category") == "blocked" for turn in trace.turns),
        "helpful": outcomes.count("helpful"),
        "neutral": outcomes.count("neutral"),
        "harmful": outcomes.count("harmful"),
    }
    trace.wall_seconds = perf_counter() - started
    emit({"phase": "condition_completed", "condition": condition, "condition_index": condition_index,
          "condition_total": condition_total, "turn": len(trace.turns), "max_steps": max_steps,
          "success": trace.success, "error": bool(trace.error)})
    return trace


def run_arc_comparison(
    environment_factory: Callable[[], ArcEnvironment],
    reasoner_factory: Callable[[], ArcReasoner],
    source: str,
    game_id: str,
    seed: int = 0,
    max_steps: int = 12,
    batch_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ArcComparisonTrace:
    shared_reasoner = CachingArcReasoner(reasoner_factory())
    without = run_arc_episode(environment_factory(), shared_reasoner, "llm_only", source, seed, max_steps, batch_id, progress_callback, 0, 3)
    with_classical = run_arc_episode(environment_factory(), shared_reasoner, "qahs_classical", source, seed, max_steps, batch_id, progress_callback, 1, 3)
    with_qahs = run_arc_episode(environment_factory(), shared_reasoner, "qahs_qaoa", source, seed, max_steps, batch_id, progress_callback, 2, 3)
    return ArcComparisonTrace(
        source, game_id, seed, without, with_classical, with_qahs,
        action_budget=max_steps,
        model=os.environ.get("OPENAI_MODEL") if source == "arc" else "Deterministic local reasoner",
        reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT") if source == "arc" else None,
        batch_id=batch_id,
        harness_flags=harness_flags(),
    )


def save_arc_trace(trace: ArcComparisonTrace | ArcEpisodeTrace, directory: str | Path = "output/arc_runs") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "comparison" if isinstance(trace, ArcComparisonTrace) else trace.condition
    safe = lambda value: "".join(char.lower() if char.isalnum() else "-" for char in str(value)).strip("-") or "unknown"
    batch = safe(trace.batch_id or "unbatched")
    model = safe(trace.model or "unrecorded")
    path = directory / (
        f"{timestamp}__{safe(trace.harness_version)}__{batch}__{safe(trace.game_id)}"
        f"__seed-{trace.seed:02d}__actions-{trace.action_budget}__{model}__{safe(kind)}.json"
    )
    path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
    return path
