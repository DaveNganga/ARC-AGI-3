from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .arc_agent import (
    run_arc_comparison,
    run_arc_conditions,
    save_arc_trace,
    shared_reasoner,
)
from .arc_environment import ARC_MODES, ArcSDKEnvironment, DemoArcEnvironment
from .arc_reasoner import DemoArcReasoner, OpenAIArcReasoner
from .frame_encoding import FRAME_FORMATS
from .providers.openai_provider import OpenAIJSONCaller
from .session import SESSIONS_DIR, always_connected, new_journal, open_journal, tcp_probe, unfinished_runs

RUN_PARAMETERS = ("source", "game", "condition", "seed", "max_steps", "frame_format", "arc_mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 with and without QAHS")
    parser.add_argument("--source", choices=("demo", "arc"), default="demo")
    parser.add_argument("--game", default="ls20", help="ARC-AGI-3 game ID")
    parser.add_argument("--condition", choices=("compare", "llm_only", "qahs_classical", "qahs_qaoa"), default="compare")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--frame-format", choices=FRAME_FORMATS, default="runs", help="How the grid is written for the model")
    parser.add_argument("--arc-mode", choices=ARC_MODES, default="online", help="offline plays games from environment_files/")
    parser.add_argument("--output-dir", default="output/arc_runs")
    parser.add_argument("--sessions-dir", default=str(SESSIONS_DIR))
    parser.add_argument("--resume", metavar="RUN_ID", help="Continue an interrupted run ('latest' for the newest)")
    parser.add_argument("--max-wait", type=float, default=900.0, help="Seconds to wait for a dropped connection")
    return parser


def setup(params: dict):
    """Return environment factory, reasoner factory, connectivity check, and game ID for a run."""
    if params["source"] == "demo":
        fault = os.environ.get("QAHS_FAULT_AT_STEP")
        DemoArcEnvironment.fault_at_step = int(fault) if fault else None
        DemoArcEnvironment._steps_taken = 0
        return (
            lambda: DemoArcEnvironment(params["seed"]),
            DemoArcReasoner,
            always_connected,
            DemoArcEnvironment.game_id,
        )
    recordings = Path("output/arc_recordings")
    mode = params["arc_mode"]
    hosts = ["api.openai.com"] + (["three.arcprize.org"] if mode == "online" else [])
    return (
        lambda: ArcSDKEnvironment(params["game"], params["seed"], recordings, mode=mode),
        lambda: OpenAIArcReasoner(OpenAIJSONCaller(), frame_format=params["frame_format"]),
        tcp_probe(hosts),
        params["game"],
    )


def episode_summary(trace) -> dict:
    return {
        "success": trace.success,
        "actions": len(trace.turns),
        "stop_reason": trace.stop_reason,
        "resumed": trace.resumed,
        "connection_events": len(trace.connection_events),
        "error": trace.error,
    }


def main() -> None:
    args = build_parser().parse_args()

    if args.resume:
        journal = open_journal(args.resume, args.sessions_dir)
        started = journal.first("run_started")
        if started is None:
            sys.exit(f"Run {journal.run_id} has no run_started record and cannot be resumed.")
        if journal.finished:
            sys.exit(f"Run {journal.run_id} already finished.")
        params = {key: started[key] for key in RUN_PARAMETERS}
        print(f"Resuming {journal.run_id} ({params['condition']}, {params['game'] if params['source'] == 'arc' else 'demo'})")
    else:
        params = {key: getattr(args, key) for key in RUN_PARAMETERS}
        pending = unfinished_runs(args.sessions_dir)
        if pending:
            print(f"Unfinished run found: {pending[0]}. Resume it with: qahs-arc --resume {pending[0]}", file=sys.stderr)
        game_id = DemoArcEnvironment.game_id if params["source"] == "demo" else params["game"]
        journal = new_journal(game_id, args.sessions_dir)
        journal.append("run_started", **params)

    environment_factory, reasoner_factory, connectivity, game_id = setup(params)
    options = {"journal": journal, "connectivity": connectivity, "max_wait": args.max_wait}
    extra = {"frame_format": params["frame_format"], "arc_mode": params["arc_mode"]} if params["source"] == "arc" else {}

    try:
        if params["condition"] == "compare":
            trace = run_arc_comparison(
                environment_factory, reasoner_factory, params["source"], game_id,
                params["seed"], params["max_steps"], extra_settings=extra, **options,
            )
            summary = {
                "without_qahs": episode_summary(trace.without_qahs),
                "with_classical": episode_summary(trace.with_classical),
                "with_qahs": episode_summary(trace.with_qahs),
            }
        else:
            reasoner = shared_reasoner(reasoner_factory(), journal)
            trace = run_arc_conditions(
                [params["condition"]], environment_factory, reasoner, params["source"], game_id,
                params["seed"], params["max_steps"], **options,
            )[params["condition"]]
            summary = episode_summary(trace)
    except KeyboardInterrupt:
        sys.exit(f"\nInterrupted. Nothing committed is lost. Resume with: qahs-arc --resume {journal.run_id}")

    path = save_arc_trace(trace, args.output_dir)
    journal.append("run_finished", trace=str(path))
    print(json.dumps(summary, indent=2))
    print(f"Trace: {path}")
    print(f"Journal: {journal.path}")


if __name__ == "__main__":
    main()
