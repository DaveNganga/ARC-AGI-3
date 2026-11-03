from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "arc_runs"
TARGET = ROOT / "frontend" / "public" / "runs"
SENSITIVE = ("api_key", "authorization", "password", "secret", "token")


def clean(value, key: str = ""):
    lowered = key.lower()
    if any(term in lowered for term in SENSITIVE):
        return "[redacted]"
    if lowered == "error" and value:
        return "Run failed; internal details omitted."
    if isinstance(value, dict):
        return {item_key: clean(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [clean(item, key) for item in value]
    return value


def label(path: Path, data: dict) -> str:
    if data.get("harness_version"):
        batch = data.get("batch_id") or "unbatched"
        return f"{data.get('game_id', 'unknown')} · seed {data.get('seed', '?')} · {batch} · {data['harness_version']}"
    match = re.match(r"(\d{8})T(\d{6})Z_([^_]+)", path.name)
    if not match:
        return path.stem
    date, time, game = match.groups()
    return f"{game} · {date[4:6]}/{date[6:8]} {time[:2]}:{time[2:4]}"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    full = "--full" in sys.argv[1:]
    previous: dict[str, dict] = {}
    index_path = TARGET / "index.json"
    try:
        previous = {row["filename"]: row for row in json.loads(index_path.read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    manifest = []
    sources = sorted(SOURCE.glob("*.json"), reverse=True)
    source_names = {source.name for source in sources}
    exported = 0
    for source in sources:
        destination = TARGET / source.name
        cached = previous.get(source.name)
        if (not full and cached and destination.exists() and
                destination.stat().st_mtime_ns >= source.stat().st_mtime_ns):
            manifest.append(cached)
            continue
        data = clean(json.loads(source.read_text(encoding="utf-8")))
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        temporary.replace(destination)
        manifest.append({
            "filename": source.name,
            "label": label(source, data),
            "benchmark": data.get("benchmark", "arc_agi_3"),
            "game_id": data.get("game_id"),
            "seed": data.get("seed"),
            "action_budget": data.get("action_budget"),
            "model": data.get("model"),
            "reasoning_effort": data.get("reasoning_effort"),
            "harness_version": data.get("harness_version"),
            "input_representation": data.get("input_representation"),
            "batch_id": data.get("batch_id"),
        })
        exported += 1
    for old in TARGET.glob("*.json"):
        if old.name != "index.json" and old.name not in source_names:
            old.unlink()
    index_temporary = index_path.with_suffix(".json.tmp")
    index_temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index_temporary.replace(index_path)
    print(f"Published {len(manifest)} sanitized run(s) ({exported} updated) to {TARGET}")


if __name__ == "__main__":
    main()
