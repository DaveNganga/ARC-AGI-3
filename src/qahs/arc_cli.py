from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arc_agent import run_arc_comparison, run_arc_episode, save_arc_trace
from .arc_environment import ArcSDKEnvironment, DemoArcEnvironment
from .arc_reasoner import DemoArcReasoner, OpenAIArcReasoner
from .providers.openai_provider import OpenAIJSONCaller


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 with and without QAHS")
    parser.add_argument("--source", choices=("demo", "arc"), default="demo")
    parser.add_argument("--game", default="ls20", help="ARC-AGI-3 game ID")
    parser.add_argument("--condition", choices=("compare", "llm_only", "qahs_classical", "qahs_qaoa"), default="compare")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output-dir", default="output/arc_runs")
    args = parser.parse_args()

    if args.source == "demo":
        game_id = DemoArcEnvironment.game_id
        environment_factory = lambda: DemoArcEnvironment(args.seed)
        reasoner_factory = DemoArcReasoner
    else:
        game_id = args.game
        recordings = Path("output/arc_recordings")
        environment_factory = lambda: ArcSDKEnvironment(args.game, args.seed, recordings)
        reasoner_factory = lambda: OpenAIArcReasoner(OpenAIJSONCaller())

    if args.condition == "compare":
        trace = run_arc_comparison(environment_factory, reasoner_factory, args.source, game_id, args.seed, args.max_steps)
        summary = {
            "without_qahs": {"success": trace.without_qahs.success, "actions": len(trace.without_qahs.turns), "stop_reason": trace.without_qahs.stop_reason, "error": trace.without_qahs.error},
            "with_classical": {"success": trace.with_classical.success, "actions": len(trace.with_classical.turns), "stop_reason": trace.with_classical.stop_reason, "error": trace.with_classical.error},
            "with_qahs": {"success": trace.with_qahs.success, "actions": len(trace.with_qahs.turns), "stop_reason": trace.with_qahs.stop_reason, "error": trace.with_qahs.error},
        }
    else:
        trace = run_arc_episode(environment_factory(), reasoner_factory(), args.condition, args.source, args.seed, args.max_steps)
        summary = {"success": trace.success, "actions": len(trace.turns), "stop_reason": trace.stop_reason, "error": trace.error}

    path = save_arc_trace(trace, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Trace: {path}")


if __name__ == "__main__":
    main()
