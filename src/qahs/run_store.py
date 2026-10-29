from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import EpisodeTrace


def save_trace(trace: EpisodeTrace, directory: str | Path = "output/runs") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{timestamp}_{trace.backend}_{trace.hidden_rule}.json"
    path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
    return path


def list_traces(directory: str | Path = "output/runs") -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob("*.json"), reverse=True) if directory.exists() else []

