from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .arc_agent import run_arc_comparison, run_arc_episode, save_arc_trace
from .arc_environment import ArcSDKEnvironment, DemoArcEnvironment
from .arc_reasoner import DemoArcReasoner, OpenAIArcReasoner
from .providers.anthropic_provider import AnthropicJSONCaller
from .providers.openai_provider import OpenAIJSONCaller
from .compositional import CompositionalReasoner, run_compositional_comparison
from .information_gain import InformationGainReasoner, run_information_gain_comparison
from .structured_world_models import StructuredWorldReasoner, run_structured_world_comparison

DEFAULT_MODELS = {"openai": ("OPENAI_MODEL", "gpt-6-luna"), "anthropic": ("ANTHROPIC_MODEL", "claude-opus-5-5")}
DEFAULT_EFFORTS = {"openai": "OPENAI_REASONING_EFFORT", "anthropic": "ANTHROPIC_EFFORT"}


def make_caller(args):
    """Build the JSON caller for the chosen provider; prompts, schemas, and selectors are unchanged."""
    if args.provider == "anthropic":
        return AnthropicJSONCaller(args.model, args.reasoning_effort)
    return OpenAIJSONCaller(args.model, args.reasoning_effort)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 with and without QAHS")
    parser.add_argument("--source", choices=("demo", "arc"), default="demo")
    parser.add_argument("--game", default="ls20", help="ARC-AGI-3 game ID")
    parser.add_argument("--condition", choices=("compare", "llm_only", "qahs_classical", "qahs_qaoa"), default="compare")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--batch", default="", help="Batch ID for grouping matched runs, such as batch-01")
    parser.add_argument("--provider", choices=tuple(DEFAULT_MODELS), default="openai", help="LLM provider for arc runs")
    parser.add_argument("--model", default=None, help="Defaults to OPENAI_MODEL/gpt-6-luna or ANTHROPIC_MODEL/claude-opus-5-5")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default=None)
    parser.add_argument("--output-dir", default="output/arc_runs")
    parser.add_argument("--harness", choices=("v7", "v10", "v11", "v12"), default=os.environ.get("QAHS_HARNESS", "v12"))
    args = parser.parse_args()
    model_env, default_model = DEFAULT_MODELS[args.provider]
    args.model = args.model or os.environ.get(model_env, default_model)
    args.reasoning_effort = args.reasoning_effort or os.environ.get(DEFAULT_EFFORTS[args.provider], "medium")
    if args.provider == "anthropic" and args.reasoning_effort == "none":
        parser.error("--reasoning-effort none is not available for the anthropic provider")
    if args.provider == "openai":
        os.environ["OPENAI_MODEL"] = args.model
        os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    if args.source == "arc" and args.condition != "compare" and args.harness in ("v10", "v11", "v12"):
        # Single-condition runs only exist for the v7 path; running one under --harness v10-v12
        # used to fall through to v7 silently.
        parser.error(f"--harness {args.harness} supports --condition compare only; use --harness v7 for single conditions")
    batch_id = args.batch.strip().lower() or None
    if batch_id and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", batch_id):
        parser.error("--batch must use lowercase letters, numbers, and hyphens (32 characters maximum)")

    if args.source == "demo":
        game_id = DemoArcEnvironment.game_id
        environment_factory = lambda: DemoArcEnvironment(args.seed)
        reasoner_factory = DemoArcReasoner
    else:
        game_id = args.game
        recordings = Path("output/arc_recordings")
        environment_factory = lambda: ArcSDKEnvironment(args.game, args.seed, recordings)
        reasoner_factory = lambda: OpenAIArcReasoner(make_caller(args))

    if args.condition == "compare" and args.harness == "v12" and args.source == "arc":
        progress = lambda event: print("QAHS_PROGRESS " + json.dumps(event), flush=True)
        caller = make_caller(args)
        trace = run_structured_world_comparison(
            environment_factory, StructuredWorldReasoner(caller), args.source, game_id, args.seed,
            args.max_steps, batch_id, progress,
        )
        summary = {
            "without_qahs": {"success": trace.without_qahs.success, "actions": len(trace.without_qahs.turns), "error": trace.without_qahs.error},
            "with_classical": {"success": trace.with_classical.success, "actions": len(trace.with_classical.turns), "error": trace.with_classical.error},
            "with_qahs": {"success": trace.with_qahs.success, "actions": len(trace.with_qahs.turns), "error": trace.with_qahs.error},
        }
    elif args.condition == "compare" and args.harness == "v11" and args.source == "arc":
        progress = lambda event: print("QAHS_PROGRESS " + json.dumps(event), flush=True)
        caller = make_caller(args)
        trace = run_information_gain_comparison(
            environment_factory, InformationGainReasoner(caller), args.source, game_id, args.seed,
            args.max_steps, batch_id, progress,
        )
        summary = {
            "without_qahs": {"success": trace.without_qahs.success, "actions": len(trace.without_qahs.turns), "error": trace.without_qahs.error},
            "with_classical": {"success": trace.with_classical.success, "actions": len(trace.with_classical.turns), "error": trace.with_classical.error},
            "with_qahs": {"success": trace.with_qahs.success, "actions": len(trace.with_qahs.turns), "error": trace.with_qahs.error},
        }
    elif args.condition == "compare" and args.harness == "v10" and args.source == "arc":
        progress = lambda event: print("QAHS_PROGRESS " + json.dumps(event), flush=True)
        caller = make_caller(args)
        trace = run_compositional_comparison(
            environment_factory, CompositionalReasoner(caller), args.source, game_id, args.seed,
            args.max_steps, batch_id, progress,
        )
        summary = {
            "without_qahs": {"success": trace.without_qahs.success, "actions": len(trace.without_qahs.turns), "error": trace.without_qahs.error},
            "with_classical": {"success": trace.with_classical.success, "actions": len(trace.with_classical.turns), "error": trace.with_classical.error},
            "with_qahs": {"success": trace.with_qahs.success, "actions": len(trace.with_qahs.turns), "error": trace.with_qahs.error},
        }
    elif args.condition == "compare":
        progress = lambda event: print("QAHS_PROGRESS " + json.dumps(event), flush=True)
        trace = run_arc_comparison(environment_factory, reasoner_factory, args.source, game_id, args.seed, args.max_steps,
                                   batch_id=batch_id, progress_callback=progress)
        summary = {
            "without_qahs": {"success": trace.without_qahs.success, "actions": len(trace.without_qahs.turns), "error": trace.without_qahs.error},
            "with_classical": {"success": trace.with_classical.success, "actions": len(trace.with_classical.turns), "error": trace.with_classical.error},
            "with_qahs": {"success": trace.with_qahs.success, "actions": len(trace.with_qahs.turns), "error": trace.with_qahs.error},
        }
    else:
        trace = run_arc_episode(environment_factory(), reasoner_factory(), args.condition, args.source, args.seed, args.max_steps)
        trace.batch_id = batch_id
        summary = {"success": trace.success, "actions": len(trace.turns), "error": trace.error}

    if args.condition == "compare" and args.source == "arc" and args.harness in ("v10", "v11", "v12"):
        # These runners do not record caller settings; v7 does so in run_arc_comparison.
        trace.model_settings = {**(getattr(caller, "settings", None) or {}), **trace.model_settings}
    path = save_arc_trace(trace, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Trace: {path}")


if __name__ == "__main__":
    main()
