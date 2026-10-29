"""Crash-safe run journals, connection-drop detection, and state restoration by replay.

The journal is an append-only JSONL file and is the single source of truth for a run. A turn
is committed only after the environment confirmed the action and the reflection exists, so a
crash at any point loses at most the uncommitted turn, whose model responses are still reused.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .arc_environment import TransientEnvironmentError
from .arc_models import ArcObservation, candidate_from_dict

SESSIONS_DIR = Path("output/arc_sessions")
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
TRANSIENT_NAMES = ("Connection", "Timeout", "RateLimit", "ServiceUnavailable", "InternalServerError", "APIStatusError")
FATAL_STATUS = {400, 401, 403, 404, 422}


class ReplayDivergence(RuntimeError):
    """Replaying committed actions produced a different frame than the one journaled."""


class ConnectivityTimeout(ConnectionError):
    """Connectivity did not return within the allowed waiting time."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


class RunJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self.path.parent.name

    def append(self, event: str, **data: Any) -> None:
        record = {"event": event, "at": utc_now(), **{key: _jsonable(value) for key, value in data.items()}}
        line = json.dumps(record, separators=(",", ":"), default=_jsonable) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash mid-write leaves at most one partial final line; it was never committed.
                continue
        return events

    def first(self, event: str) -> dict[str, Any] | None:
        return next((item for item in self.load() if item["event"] == event), None)

    @property
    def finished(self) -> bool:
        return self.first("run_finished") is not None


def new_journal(game_id: str, directory: str | Path = SESSIONS_DIR) -> RunJournal:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{game_id}"
    return RunJournal(Path(directory) / run_id / "journal.jsonl")


def open_journal(run_id: str, directory: str | Path = SESSIONS_DIR) -> RunJournal:
    directory = Path(directory)
    if run_id == "latest":
        unfinished = unfinished_runs(directory)
        if not unfinished:
            raise FileNotFoundError("No unfinished run to resume.")
        run_id = unfinished[0]
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID.")
    journal = RunJournal(directory / run_id / "journal.jsonl")
    if not journal.path.exists():
        raise FileNotFoundError(f"No journal for run {run_id}.")
    return journal


def unfinished_runs(directory: str | Path = SESSIONS_DIR) -> list[str]:
    """Run IDs whose journal has no run_finished event, newest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    runs = [path.parent.name for path in directory.glob("*/journal.jsonl") if not RunJournal(path).finished]
    return sorted(runs, reverse=True)


def classify_failure(error: BaseException) -> str:
    """Return "transient" for network-type failures that are worth waiting out, else "fatal"."""
    status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    if status in FATAL_STATUS:
        return "fatal"
    if status in TRANSIENT_STATUS:
        return "transient"
    if isinstance(error, (TransientEnvironmentError, ConnectionError, TimeoutError, socket.timeout)):
        return "transient"
    if any(name in type(error).__name__ for name in TRANSIENT_NAMES):
        return "transient"
    return "fatal"


def tcp_probe(hosts: Iterable[str], timeout: float = 5.0) -> Callable[[], bool]:
    """Connectivity check that needs no credentials: can we open TCP 443 to every host?"""
    hosts = tuple(hosts)

    def check() -> bool:
        for host in hosts:
            try:
                with socket.create_connection((host, 443), timeout=timeout):
                    pass
            except OSError:
                return False
        return True

    return check


def always_connected() -> bool:
    return True


def wait_for_connectivity(
    check: Callable[[], bool],
    max_wait: float = 900.0,
    first_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Block until ``check`` succeeds, backing off exponentially; return seconds waited."""
    started = clock()
    delay = first_delay
    while True:
        if check():
            return clock() - started
        if clock() - started >= max_wait:
            raise ConnectivityTimeout(f"Connection did not return within {max_wait:.0f} s")
        sleep(delay)
        delay = min(max_delay, delay * 2)


def restore_environment(
    environment: Any,
    committed: list[dict[str, Any]],
    reset_signature: str | None,
    signature: Callable[[tuple[tuple[int, ...], ...]], str],
) -> ArcObservation:
    """Rebuild game state on a fresh environment by replaying committed actions.

    Only actions the environment confirmed are replayed, so a step whose response was lost is
    never assumed to have happened. Every frame is checked against the journal.
    """
    observation = environment.reset()
    if reset_signature and signature(observation.frame) != reset_signature:
        raise ReplayDivergence("Initial frame differs from the journaled run")
    for record in committed:
        observation = environment.step(candidate_from_dict(record["candidate"]))
        if signature(observation.frame) != record["after_signature"]:
            raise ReplayDivergence(f"Frame after replayed turn {record['turn']} differs from the journal")
    return observation
