"""Pair comparison traces from different models on (game, seed, budget) and report descriptive metrics.

Usage: python scripts/mirror_compare.py [--json] [DIR ...]
Defaults to frontend/public/runs and output/arc_runs. Demo runs are skipped.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = (ROOT / "frontend" / "public" / "runs", ROOT / "output" / "arc_runs")
CONDITIONS = (("without_qahs", "llm_only"), ("with_classical", "classical"), ("with_qahs", "qaoa"))


def signature(candidate: dict) -> str:
    return f"{candidate.get('action_type')}:{candidate.get('x')}:{candidate.get('y')}"


def episode_metrics(episode: dict | None) -> dict:
    turns = (episode or {}).get("turns", [])
    top_utility = 0
    for turn in turns:
        scores = turn.get("action_scores") or []
        if scores:
            best = max(scores, key=lambda row: row.get("utility", float("-inf")))
            top_utility += best.get("action") == (turn.get("selected_candidate") or {}).get("candidate_id")
    tokens = defaultdict(int)
    for turn in turns:
        for key, value in (turn.get("token_usage") or {}).items():
            tokens[key] += int(value)
    return {
        "actions": len(turns),
        "levels": max((t["observation_after"].get("levels_completed", 0) for t in turns), default=0),
        "success": bool((episode or {}).get("success")),
        "stop_reason": (episode or {}).get("stop_reason") or ("error" if (episode or {}).get("error") else "unrecorded"),
        "distinct_actions": len({signature(t.get("selected_candidate") or {}) for t in turns}),
        "top_utility_picks": top_utility,
        "model_calls": sum(int(t.get("model_calls", 0) or 0) for t in turns),
        "tokens": dict(tokens),
    }


def load_runs(directories) -> list[dict]:
    runs, seen = [], set()
    for directory in directories:
        for path in sorted(Path(directory).glob("*_comparison.json")):
            if path.name in seen:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("source") != "arc":
                continue
            seen.add(path.name)
            turns = [t for key, _ in CONDITIONS for t in (data.get(key) or {}).get("turns", [])]
            runs.append({
                "file": path.name,
                "model": data.get("model") or "unknown",
                "key": (data["game_id"], data.get("seed"), data.get("action_budget")),
                # Traces without reflection come from the pre-ledger harness (one model call per turn).
                "harness": "reflection" if any(t.get("reflection") for t in turns) else "no_reflection",
                "conditions": {label: episode_metrics(data.get(key)) for key, label in CONDITIONS},
            })
    return runs


def markdown(runs: list[dict]) -> str:
    by_key = defaultdict(list)
    for run in runs:
        by_key[run["key"]].append(run)
    lines = [
        "| game | seed | budget | model | harness | condition | levels | actions | stop | distinct | top-utility | calls |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (game, seed, budget), group in sorted(by_key.items(), key=lambda item: tuple(map(str, item[0]))):
        if len({run["model"] for run in group}) < 2:
            continue  # only pairs with at least two models are mirrors
        for run in sorted(group, key=lambda r: (r["model"], r["file"])):
            for label, m in run["conditions"].items():
                lines.append(
                    f"| {game} | {seed} | {budget} | {run['model']} | {run['harness']} | {label} | {m['levels']} | "
                    f"{m['actions']} | {m['stop_reason']} | {m['distinct_actions']} | {m['top_utility_picks']}/{m['actions']} | {m['model_calls']} |"
                )
    lines += ["", "Aggregate over paired keys:", "", "| model | condition | runs | total levels | episodes with ≥1 level | top-utility | stop reasons |", "|---|---|---|---|---|---|---|"]
    paired = {key for key, group in by_key.items() if len({run["model"] for run in group}) >= 2}
    totals = defaultdict(lambda: {"runs": 0, "levels": 0, "any": 0, "top": 0, "actions": 0, "stops": defaultdict(int)})
    for run in runs:
        if run["key"] not in paired:
            continue
        for label, m in run["conditions"].items():
            row = totals[(run["model"], label)]
            row["runs"] += 1
            row["levels"] += m["levels"]
            row["any"] += m["levels"] > 0
            row["top"] += m["top_utility_picks"]
            row["actions"] += m["actions"]
            row["stops"][m["stop_reason"]] += 1
    for (model, label), row in sorted(totals.items()):
        stops = ", ".join(f"{k}={v}" for k, v in sorted(row["stops"].items()))
        lines.append(f"| {model} | {label} | {row['runs']} | {row['levels']} | {row['any']} | {row['top']}/{row['actions']} | {stops} |")
    harnesses = {(run["model"], run["harness"]) for run in runs if run["key"] in paired}
    if len({h for _, h in harnesses}) > 1:
        lines += ["", "WARNING: paired runs mix harness versions (with and without the reflection ledger): "
                  + ", ".join(f"{m}={h}" for m, h in sorted(harnesses)) + ". Differences are not attributable to the model alone."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="*", default=[str(d) for d in DEFAULT_DIRS])
    parser.add_argument("--json", action="store_true", help="Print per-run metrics as JSON")
    args = parser.parse_args()
    runs = load_runs(d for d in args.directories if Path(d).is_dir())
    if args.json:
        print(json.dumps([{**run, "key": list(run["key"])} for run in runs], indent=2))
    else:
        print(markdown(runs))


if __name__ == "__main__":
    main()
