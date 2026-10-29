from __future__ import annotations

import argparse

from .agent import run_episode
from .environment import HiddenRule, TinyRuleGame
from .run_store import save_trace
from .search import ClassicalSearch, GreedyBeliefSearch, QAOASearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QAHS hidden-rule demonstration")
    parser.add_argument("--backend", choices=("greedy", "classical", "qaoa"), default="classical")
    parser.add_argument("--hidden-rule", choices=[r.value for r in HiddenRule], default=HiddenRule.RED_IS_SWITCH.value)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    backend = {"greedy": GreedyBeliefSearch(), "classical": ClassicalSearch(), "qaoa": QAOASearch()}[args.backend]
    trace = run_episode(TinyRuleGame(HiddenRule(args.hidden_rule)), backend=backend)
    print(f"Backend: {trace.backend} | success: {trace.success} | turns: {len(trace.turns)}")
    for turn in trace.turns:
        scores = ", ".join(f"{s.action.value}={s.utility:.3f}" for s in turn.action_scores)
        print(f"Turn {turn.turn}: {scores} -> {turn.search.action.value} -> {turn.observation.kind.value}")
    if not args.no_save:
        print("Trace:", save_trace(trace))


if __name__ == "__main__":
    main()

