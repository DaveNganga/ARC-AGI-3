from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .arc_encoding import ArcEncoding
from .arc_environment import ArcEnvironment
from .arc_models import (
    ArcActionCandidate,
    ArcComparisonTrace,
    ArcEpisodeTrace,
    ArcProposal,
    ArcTurn,
    proposal_from_dict,
    reflection_from_dict,
)
from .arc_objective import action_signature, build_arc_costs, score_arc_actions
from .arc_reasoner import ArcReasoner, CachingArcReasoner
from .search import ClassicalSearch, QAOASearch
from .session import (
    ConnectivityTimeout,
    ReplayDivergence,
    RunJournal,
    always_connected,
    classify_failure,
    restore_environment,
    wait_for_connectivity,
)


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


def condition_order(seed: int) -> list[str]:
    """Rotate which condition fills the shared cache first so no condition is always first."""
    shift = seed % len(CONDITIONS)
    return list(CONDITIONS[shift:] + CONDITIONS[:shift])


@dataclass
class _Decision:
    proposal: ArcProposal
    selected: ArcActionCandidate
    score_rows: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    model_calls: int
    usage: dict[str, int]


def _add_usage(total: dict[str, int], usage: dict[str, int] | None) -> dict[str, int]:
    for key, value in (usage or {}).items():
        total[key] = total.get(key, 0) + int(value)
    return total


def _decide(reasoner: ArcReasoner, condition: str, observation, history: list[dict], seed: int, turn_number: int) -> _Decision:
    proposal = reasoner.propose(observation, history)
    model_calls = int(getattr(reasoner, "last_call_count", 1))
    usage = _add_usage({}, getattr(reasoner, "last_usage", None))
    scores = score_arc_actions(proposal, history, frame_signature(observation.frame))

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
    return _Decision(proposal, selected, [asdict(row) for row in scores], diagnostics, model_calls, usage)


def run_arc_episode(
    environment: ArcEnvironment,
    reasoner: ArcReasoner,
    condition: str,
    source: str,
    seed: int = 0,
    max_steps: int = 12,
    *,
    journal: RunJournal | None = None,
    environment_factory: Callable[[], ArcEnvironment] | None = None,
    connectivity: Callable[[], bool] = always_connected,
    prior: dict[str, Any] | None = None,
    max_reconnects: int = 5,
    max_wait: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ArcEpisodeTrace:
    """Run one condition, committing every turn to ``journal`` and surviving dropped connections.

    ``prior`` holds the journaled state of an interrupted episode (committed turns and the
    reset-frame signature); the episode then continues from its last committed turn.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    started = perf_counter()
    trace = ArcEpisodeTrace(condition, source, environment.game_id, seed)
    prior = prior or {}
    committed: list[dict[str, Any]] = list(prior.get("committed", []))
    reset_signature: str | None = prior.get("reset_signature")
    history: list[dict] = [record["history_entry"] for record in committed]
    trace.turns = [record["turn_data"] for record in committed]  # type: ignore[misc]
    previous_wall = float(committed[-1].get("wall_seconds", 0.0)) if committed else 0.0
    trace.resumed = bool(committed)

    current_env = environment
    observation = None
    restores = 0

    def restore() -> None:
        """(Re)build game state from committed turns; after the first attempt, on a fresh environment."""
        nonlocal current_env, observation, reset_signature, restores
        if restores:
            if environment_factory is None:
                raise RuntimeError("Connection lost and no environment factory is available to restore state")
            current_env = environment_factory()
        restores += 1
        if committed:
            observation = restore_environment(current_env, committed, reset_signature, frame_signature)
            trace.replayed_actions += len(committed)
            return
        observation = current_env.reset()
        signature = frame_signature(observation.frame)
        if reset_signature is None:
            reset_signature = signature
            if journal:
                journal.append("episode_started", condition=condition, reset_signature=signature)
        elif signature != reset_signature:
            raise ReplayDivergence("Initial frame differs from the journaled run")

    def recover(error: Exception, stage: str, turn_number: int) -> None:
        if classify_failure(error) != "transient":
            raise error
        event = {"turn": turn_number, "stage": stage, "error": f"{type(error).__name__}: {error}"}
        if journal:
            journal.append("connection_lost", condition=condition, **event)
        event["waited_seconds"] = round(wait_for_connectivity(connectivity, max_wait, sleep=sleep), 2)
        trace.connection_events.append(event)
        if journal:
            journal.append("resumed", condition=condition, turn=turn_number, waited_seconds=event["waited_seconds"])

    try:
        needs_restore = True
        reconnects = 0
        turn_number = len(committed) + 1
        terminal = bool(committed) and committed[-1]["history_entry"]["state"].upper() in {"WIN", "GAME_OVER"}
        while needs_restore and not terminal and turn_number <= max_steps:
            try:
                restore()
                needs_restore = False
            except ConnectivityTimeout:
                raise
            except Exception as error:
                reconnects += 1
                if reconnects > max_reconnects:
                    raise
                recover(error, "restore", turn_number)

        while not terminal and turn_number <= max_steps:
            decision: _Decision | None = None
            after = None
            reflection = None
            reflection_calls = 0
            reflection_usage: dict[str, int] = {}
            reconnects = 0
            while reflection is None:
                stage = "restore" if needs_restore else "propose"
                try:
                    if needs_restore:
                        restore()
                        needs_restore = False
                    before = observation
                    if decision is None:
                        stage = "propose"
                        decision = _decide(reasoner, condition, before, history, seed, turn_number)
                    if after is None:
                        stage = "step"
                        after = current_env.step(decision.selected)
                    stage = "reflect"
                    reflection = reasoner.reflect(before, after, decision.proposal, decision.selected, history)
                    reflection_calls = int(getattr(reasoner, "last_call_count", 1))
                    reflection_usage = dict(getattr(reasoner, "last_usage", None) or {})
                except ConnectivityTimeout:
                    raise
                except Exception as error:
                    reconnects += 1
                    if reconnects > max_reconnects:
                        raise
                    recover(error, stage, turn_number)
                    if stage in ("step", "restore"):
                        # The action may or may not have been applied: rebuild from committed turns only.
                        after = None
                        needs_restore = True

            before = observation
            observation = after
            cells_changed = changed_cell_count(before.frame, observation.frame)
            belief_ledger = [asdict(item) for item in reflection.beliefs]
            selected = decision.selected
            history_entry = {
                "turn": turn_number,
                "candidate_id": selected.candidate_id,
                "action_type": selected.action_type,
                "x": selected.x,
                "y": selected.y,
                "action_signature": action_signature(selected),
                "before_frame_signature": frame_signature(before.frame),
                "after_frame_signature": frame_signature(observation.frame),
                "changed_cells": cells_changed,
                "levels_delta": observation.levels_completed - before.levels_completed,
                "hypotheses": [
                    {"id": item.id, "statement": item.statement, "confidence": item.confidence}
                    for item in decision.proposal.hypotheses
                ],
                "reflection": reflection.summary,
                "belief_ledger": belief_ledger,
                "next_focus": reflection.next_focus,
                "result": observation.event,
                "state": observation.state,
            }
            turn = ArcTurn(
                turn=turn_number,
                observation_before=before,
                proposal=decision.proposal,
                condition=condition,
                selected_candidate=selected,
                action_scores=decision.score_rows,
                search_diagnostics=decision.diagnostics,
                observation_after=observation,
                reflection=reflection,
                model_calls=decision.model_calls + reflection_calls,
                token_usage=_add_usage(dict(decision.usage), reflection_usage),
            )
            history.append(history_entry)
            trace.turns.append(turn)
            record = {
                "turn": turn_number,
                "candidate": asdict(selected),
                "after_signature": history_entry["after_frame_signature"],
                "history_entry": history_entry,
                "turn_data": asdict(turn),
                "wall_seconds": previous_wall + perf_counter() - started,
            }
            committed.append(record)
            if journal:
                journal.append("turn_committed", condition=condition, **record)
            terminal = observation.terminal
            turn_number += 1

        final_state = committed[-1]["history_entry"]["state"].upper() if committed else str(observation.state).upper()
        trace.success = final_state == "WIN"
        if trace.success:
            trace.stop_reason = "success"
        elif final_state == "GAME_OVER":
            trace.stop_reason = "game_over"
        else:
            trace.stop_reason = "budget_exhausted"
    except Exception as error:
        # Private traces keep full details; export_dashboard_runs.py redacts them before publishing.
        trace.error = f"{type(error).__name__}: {error}"
        trace.error_traceback = traceback.format_exc()
        trace.stop_reason = "error"
    trace.wall_seconds = previous_wall + perf_counter() - started
    if journal:
        journal.append(
            "episode_finished",
            condition=condition,
            success=trace.success,
            stop_reason=trace.stop_reason,
            error=trace.error,
            error_traceback=trace.error_traceback,
            wall_seconds=trace.wall_seconds,
            resumed=trace.resumed,
            replayed_actions=trace.replayed_actions,
            connection_events=trace.connection_events,
        )
    return trace


def _episode_from_journal(events: list[dict], condition: str, source: str, game_id: str, seed: int) -> ArcEpisodeTrace:
    finished = next(item for item in events if item["event"] == "episode_finished" and item["condition"] == condition)
    turns = [item["turn_data"] for item in events if item["event"] == "turn_committed" and item["condition"] == condition]
    return ArcEpisodeTrace(
        condition, source, game_id, seed,
        turns=turns,  # type: ignore[arg-type]
        success=finished["success"],
        wall_seconds=finished["wall_seconds"],
        error=finished["error"],
        error_traceback=finished["error_traceback"],
        stop_reason=finished["stop_reason"],
        resumed=finished["resumed"],
        replayed_actions=finished["replayed_actions"],
        connection_events=finished["connection_events"],
    )


def _prior_from_journal(events: list[dict], condition: str) -> dict[str, Any]:
    started = next((item for item in events if item["event"] == "episode_started" and item["condition"] == condition), None)
    return {
        "reset_signature": started["reset_signature"] if started else None,
        "committed": [item for item in events if item["event"] == "turn_committed" and item["condition"] == condition],
    }


def shared_reasoner(reasoner: ArcReasoner, journal: RunJournal | None) -> CachingArcReasoner:
    """Wrap a reasoner in the matched-input cache, journaling fresh responses and restoring old ones."""
    def record(kind: str, key: str, value: Any) -> None:
        if journal:
            journal.append("model_response", kind=kind, key=key, value=value)

    cache = CachingArcReasoner(reasoner, on_new_entry=record)
    for item in journal.load() if journal else []:
        if item["event"] == "model_response":
            decode = proposal_from_dict if item["kind"] == "proposal" else reflection_from_dict
            cache.prime(item["kind"], item["key"], decode(item["value"]))
    return cache


def run_arc_conditions(
    conditions: list[str],
    environment_factory: Callable[[], ArcEnvironment],
    reasoner: ArcReasoner,
    source: str,
    game_id: str,
    seed: int = 0,
    max_steps: int = 12,
    *,
    journal: RunJournal | None = None,
    **episode_options: Any,
) -> dict[str, ArcEpisodeTrace]:
    """Run (or resume) conditions in order; finished episodes are rebuilt from the journal."""
    events = journal.load() if journal else []
    traces = {}
    for condition in conditions:
        if any(item["event"] == "episode_finished" and item["condition"] == condition for item in events):
            traces[condition] = _episode_from_journal(events, condition, source, game_id, seed)
            continue
        traces[condition] = run_arc_episode(
            environment_factory(), reasoner, condition, source, seed, max_steps,
            journal=journal,
            environment_factory=environment_factory,
            prior=_prior_from_journal(events, condition),
            **episode_options,
        )
    return traces


def run_arc_comparison(
    environment_factory: Callable[[], ArcEnvironment],
    reasoner_factory: Callable[[], ArcReasoner],
    source: str,
    game_id: str,
    seed: int = 0,
    max_steps: int = 12,
    *,
    journal: RunJournal | None = None,
    extra_settings: dict[str, Any] | None = None,
    **episode_options: Any,
) -> ArcComparisonTrace:
    reasoner = reasoner_factory()
    order = condition_order(seed)
    episodes = run_arc_conditions(
        order, environment_factory, shared_reasoner(reasoner, journal), source, game_id, seed, max_steps,
        journal=journal, **episode_options,
    )
    settings = dict(getattr(getattr(reasoner, "caller", None), "settings", {}) or {})
    settings.update(extra_settings or {})
    return ArcComparisonTrace(
        source, game_id, seed,
        episodes["llm_only"], episodes["qahs_classical"], episodes["qahs_qaoa"],
        action_budget=max_steps,
        model=os.environ.get("OPENAI_MODEL") if source == "arc" else "Deterministic local reasoner",
        model_settings=settings,
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
