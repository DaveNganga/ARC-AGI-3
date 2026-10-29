from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from .factorized import generate_hypotheses
from .factorized_agent import run_factorized_episode
from .rule_environment import RuleDiscoveryEnvironment
from .search import ClassicalSearch, GreedyBeliefSearch, QAOASearch


def backend_factory(name: str, seed: int = 7):
    if name == "policy_proxy":
        return GreedyBeliefSearch()
    if name == "classical":
        return ClassicalSearch()
    if name == "qaoa":
        return QAOASearch(depth=2, restarts=4, maxiter=100, shots=1024, seed=seed)
    raise ValueError(f"Unknown backend: {name}")


def run_benchmark(
    backend_names: tuple[str, ...] = ("policy_proxy", "classical"),
    world_ids: list[int] | None = None,
    seed: int = 7,
) -> dict:
    hypotheses = generate_hypotheses()
    world_ids = world_ids if world_ids is not None else list(range(64))
    rows = []
    for backend_name in backend_names:
        for world_id in world_ids:
            trace = run_factorized_episode(
                RuleDiscoveryEnvironment(hypotheses[world_id]),
                backend_factory(backend_name, seed + world_id),
            )
            last_diag = trace.turns[-1].search.diagnostics if trace.turns else {}
            rows.append({
                "condition": backend_name,
                "world_id": world_id,
                "success": trace.success,
                "actions_to_solution": len(trace.turns),
                "wall_seconds": trace.wall_seconds,
                "qubits": last_diag.get("decision_qubits"),
                "shots": sum(t.search.diagnostics.get("shots", 0) for t in trace.turns),
                "optimizer_evaluations": sum(t.search.diagnostics.get("optimizer_evaluations", 0) for t in trace.turns),
            })
    summary = {}
    for name in backend_names:
        subset = [row for row in rows if row["condition"] == name]
        summary[name] = {
            "runs": len(subset),
            "solve_rate": mean(row["success"] for row in subset),
            "mean_actions_to_solution": mean(row["actions_to_solution"] for row in subset),
            "mean_wall_seconds": mean(row["wall_seconds"] for row in subset),
            "mean_shots": mean(row["shots"] for row in subset),
            "mean_optimizer_evaluations": mean(row["optimizer_evaluations"] for row in subset),
        }
    return {"hypothesis_count": 64, "conditions": list(backend_names), "rows": rows, "summary": summary}


def save_benchmark(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path

