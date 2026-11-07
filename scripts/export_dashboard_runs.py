from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "arc_runs"
TARGET = ROOT / "frontend" / "public" / "runs"
SENSITIVE = ("api_key", "authorization", "password", "secret", "token")
PUBLISHED_HARNESS_PATTERN = re.compile(r"^v(?:10|11|12)(?:-|$)", re.IGNORECASE)


def is_published_harness(value: object) -> bool:
    return bool(PUBLISHED_HARNESS_PATTERN.match(str(value or "")))


def clean(value, key: str = ""):
    lowered = key.lower()
    if any(term in lowered for term in SENSITIVE):
        return "[redacted]"
    if lowered in {"error", "error_traceback"} and value:
        return "Run failed; internal details omitted."
    if isinstance(value, dict):
        return {item_key: clean(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [clean(item, key) for item in value]
    return value


def label(path: Path, data: dict) -> str:
    if data.get("harness_version"):
        model_names = {
            "gpt-6-luna": "Luna",
            "gpt-6-sol": "GPT-6 Sol",
            "claude-haiku-5-5": "Haiku 5.5",
            "claude-opus-5-5": "Opus 5.5",
        }
        parts = [str(data.get("game_id", "unknown")), f"seed {data.get('seed', '?')}"]
        model = data.get("model")
        if model:
            parts.append(model_names.get(model, str(model)))
        if data.get("action_budget") is not None:
            parts.append(f"{data['action_budget']} actions")
        batch = re.sub(r"^claude-(?:haiku-)?", "", str(data.get("batch_id") or ""))
        if batch and batch != "unbatched":
            parts.append(batch)
        return " · ".join(parts)
    match = re.match(r"(\d{8})T(\d{6})Z_([^_]+)", path.name)
    if not match:
        return path.stem
    date, time, game = match.groups()
    return f"{game} · {date[4:6]}/{date[6:8]} {time[:2]}:{time[2:4]}"


def export_runs(source_dir: Path = SOURCE, target_dir: Path = TARGET, full: bool = False) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    indexed: dict[str, dict] = {}
    previous: dict[str, dict] = {}
    index_path = target_dir / "index.json"
    try:
        indexed = {
            row["filename"]: row
            for row in json.loads(index_path.read_text(encoding="utf-8"))
        }
        previous = {
            filename: row
            for filename, row in indexed.items()
            if is_published_harness(row.get("harness_version"))
        }
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Recover already-published files when an older bundle predates index.json.
    for published in target_dir.glob("*.json"):
        if published.name != "index.json" and published.name not in previous:
            try:
                published_data = json.loads(published.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if is_published_harness(published_data.get("harness_version")):
                previous[published.name] = {
                    "filename": published.name,
                    "label": label(published, published_data),
                    "benchmark": published_data.get("benchmark", "arc_agi_3"),
                    "game_id": published_data.get("game_id"),
                    "seed": published_data.get("seed"),
                    "action_budget": published_data.get("action_budget"),
                    "model": published_data.get("model"),
                    "reasoning_effort": published_data.get("reasoning_effort"),
                    "harness_version": published_data.get("harness_version"),
                    "input_representation": published_data.get("input_representation"),
                    "batch_id": published_data.get("batch_id"),
                }
    manifest = []
    sources = sorted(source_dir.glob("*.json"), reverse=True) if source_dir.exists() else []
    exported = 0
    for source in sources:
        indexed_row = indexed.get(source.name)
        if indexed_row and not is_published_harness(indexed_row.get("harness_version")):
            continue
        destination = target_dir / source.name
        cached = previous.get(source.name)
        cache_is_complete = cached and {
            "label", "game_id", "seed", "model", "batch_id"
        }.issubset(cached)
        if (not full and cache_is_complete and destination.exists() and
                destination.stat().st_mtime_ns >= source.stat().st_mtime_ns):
            manifest.append(cached)
            continue
        data = clean(json.loads(source.read_text(encoding="utf-8")))
        if not is_published_harness(data.get("harness_version")):
            continue
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
    # Keep allowed public-only history so a fresh clone does not need raw traces.
    present = {row["filename"] for row in manifest}
    for filename, row in previous.items():
        if filename not in present and (target_dir / filename).exists():
            manifest.append(row)
    published_names = {row["filename"] for row in manifest}
    for published in target_dir.glob("*.json"):
        if published.name != "index.json" and published.name not in published_names:
            published.unlink()
    index_temporary = index_path.with_suffix(".json.tmp")
    index_temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index_temporary.replace(index_path)
    return exported


def main() -> None:
    full = "--full" in sys.argv[1:]
    exported = export_runs(SOURCE, TARGET, full)
    count = len(json.loads((TARGET / "index.json").read_text(encoding="utf-8")))
    print(f"Published {count} sanitized run(s) ({exported} updated) to {TARGET}")


if __name__ == "__main__":
    main()
