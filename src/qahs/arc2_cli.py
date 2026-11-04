from __future__ import annotations
import argparse
from pathlib import Path
from .arc2_harness import LUNA_MEDIUM_TASKS, run_arc2_comparison

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI-2 with a shared Luna candidate catalog")
    parser.add_argument("--task", choices=LUNA_MEDIUM_TASKS, required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--batch", default="arc2-luna-medium")
    args = parser.parse_args()
    print(f"Trace: {run_arc2_comparison(Path.cwd(), args.task, args.model, args.reasoning_effort, args.batch)}")

if __name__ == "__main__":
    main()
