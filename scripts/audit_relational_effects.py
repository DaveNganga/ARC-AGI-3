#!/usr/bin/env python3
"""Audit saved ARC transitions without making ARC or model API calls."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from qahs.information_gain import _tracked_entities
from qahs.structured_world_models import relational_effect_signature


def audit(path: Path) -> dict[str, list[dict]]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, list[dict]] = defaultdict(list)
    for condition in ("without_qahs", "with_classical", "with_qahs"):
        previous = tuple()
        for turn in trace.get(condition, {}).get("turns", []):
            before_frame = tuple(tuple(row) for row in turn["observation_before"]["frame"])
            after_frame = tuple(tuple(row) for row in turn["observation_after"]["frame"])
            before = _tracked_entities(before_frame, previous)
            after = _tracked_entities(after_frame, before)
            previous = after
            candidate = turn["selected_candidate"]
            signature = relational_effect_signature(before, after)
            action = f"{candidate.get('action_type')}@{candidate.get('x')},{candidate.get('y')}"
            results[action].append({
                "condition": condition,
                "turn": turn["turn"],
                "fingerprint": signature["fingerprint"],
                "moved_entities": len(signature["motions"]),
                "changed_relations": len(signature["relations"]),
                "levels_completed": turn["observation_after"].get("levels_completed", 0),
            })
    return dict(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.trace), indent=2))


if __name__ == "__main__":
    main()
