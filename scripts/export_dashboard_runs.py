from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "arc_runs"
TARGET = ROOT / "frontend" / "public" / "runs"
SENSITIVE = ("api_key", "authorization", "password", "secret", "token")


def clean(value, key: str = ""):
    lowered = key.lower()
    if any(term in lowered for term in SENSITIVE):
        return "[redacted]"
    if lowered.startswith("error") and value:
        return "Run failed; internal details omitted."
    if isinstance(value, dict):
        return {item_key: clean(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [clean(item, key) for item in value]
    return value


def label(path: Path) -> str:
    match = re.match(r"(\d{8})T(\d{6})Z_([^_]+)", path.name)
    if not match:
        return path.stem
    date, time, game = match.groups()
    return f"{game} · {date[4:6]}/{date[6:8]} {time[:2]}:{time[2:4]}"


def export_runs(source: Path = SOURCE, target: Path = TARGET) -> int:
    """Merge sanitized local traces into the dashboard folder without deleting published runs."""
    runs = sorted(source.glob("*.json")) if source.is_dir() else []
    if not runs:
        print(f"No local runs found in {source}; keeping existing dashboard data.")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    for run in runs:
        data = clean(json.loads(run.read_text(encoding="utf-8")))
        (target / run.name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    published = sorted((path for path in target.glob("*.json") if path.name != "index.json"), reverse=True)
    manifest = [{"filename": path.name, "label": label(path)} for path in published]
    (target / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(runs)} sanitized run(s); {len(manifest)} run(s) published in {target}")
    return len(runs)


def main() -> None:
    export_runs()


if __name__ == "__main__":
    main()
