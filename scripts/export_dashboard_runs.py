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
    if lowered == "error" and value:
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


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for old in TARGET.glob("*.json"):
        old.unlink()
    manifest = []
    for source in sorted(SOURCE.glob("*.json"), reverse=True):
        data = clean(json.loads(source.read_text(encoding="utf-8")))
        (TARGET / source.name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        manifest.append({"filename": source.name, "label": label(source)})
    (TARGET / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest)} sanitized run(s) to {TARGET}")


if __name__ == "__main__":
    main()
