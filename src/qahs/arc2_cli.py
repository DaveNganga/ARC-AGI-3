from __future__ import annotations
import argparse
import os
from pathlib import Path
from .arc2_harness import DEFAULT_MODELS, LUNA_MEDIUM_TASKS, run_arc2_comparison

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI-2 with a shared Luna candidate catalog")
    parser.add_argument("--task", choices=LUNA_MEDIUM_TASKS, required=True)
    parser.add_argument("--provider", choices=tuple(DEFAULT_MODELS), default="openai")
    parser.add_argument("--model", default=None, help="Defaults to the chosen provider's model")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--batch", default="arc2-luna-medium")
    args = parser.parse_args()
    model_env, default_model = DEFAULT_MODELS[args.provider]
    args.model = args.model or os.environ.get(model_env, default_model)
    print(f"Trace: {run_arc2_comparison(Path.cwd(), args.task, args.model, args.reasoning_effort, args.batch, args.provider)}")

if __name__ == "__main__":
    main()
