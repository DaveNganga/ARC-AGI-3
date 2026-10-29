from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark, save_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched 64-hypothesis QAHS conditions")
    parser.add_argument("--conditions", nargs="+", choices=("policy_proxy", "classical", "qaoa"), default=["policy_proxy", "classical"])
    parser.add_argument("--worlds", type=int, default=64, help="Number of hidden worlds, from 1 to 64")
    parser.add_argument("--output", default="output/experiments/factorized_benchmark.json")
    args = parser.parse_args()
    if not 1 <= args.worlds <= 64:
        parser.error("--worlds must be between 1 and 64")
    result = run_benchmark(tuple(args.conditions), list(range(args.worlds)))
    save_benchmark(result, args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

