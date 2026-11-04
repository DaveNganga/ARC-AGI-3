from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qahs.arc_environment import ArcSDKEnvironment
from qahs.arc_models import ArcActionCandidate


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = int(os.environ.get("QAHS_RUNNER_PORT", "8765"))
GAME_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
BATCH_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
MODELS = {"gpt-6-luna", "gpt-6-sol"}
ARC2_TASKS = {
    "136b0064", "1818057f", "291dc1e1", "2ba387bc", "31f7f899", "3dc255db",
    "45a5af55", "58f5dbd5", "8f3a5a89", "981571dc", "b5ca7ac4", "b99e7126",
    "bf45cf4b", "c4d067a0", "d59b0160", "db0c5428", "e8686506",
}
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
ALLOWED_ORIGINS = re.compile(
    r"^https?://(?:localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}):\d+$"
)
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
SESSION_DIR = ROOT / "output" / "job_sessions"
ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
PROCESS_LOCK = threading.Lock()
RUN_FILE = re.compile(r"^[A-Za-z0-9_.-]+\.json$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_path(job_id: str) -> Path:
    return SESSION_DIR / f"{job_id}.json"


def save_job(job: dict) -> None:
    """Atomically checkpoint a batch without storing API credentials."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    target = session_path(job["job_id"])
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
    temporary.replace(target)


def update_job(job_id: str, **changes: object) -> dict:
    with JOB_LOCK:
        job = JOBS[job_id]
        job.update(changes)
        job["updated_at"] = utc_now()
        snapshot = dict(job)
        save_job(snapshot)
        return snapshot


def load_jobs() -> None:
    """Restore prior jobs; an in-flight process becomes explicitly resumable."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    for path in SESSION_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") in {"running", "starting"}:
                job["status"] = "paused"
                job["message"] = "Runner restarted during this game. Resume to retry it from the same seed."
                job["updated_at"] = utc_now()
                save_job(job)
            JOBS[str(job["job_id"])] = job
        except (OSError, ValueError, KeyError, TypeError):
            print(f"Ignoring unreadable session: {path.name}", file=sys.stderr)


def stop_active_processes() -> None:
    with PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def handle_shutdown(_signum: int, _frame: object) -> None:
    stop_active_processes()
    raise SystemExit(0)


class PublicError(ValueError):
    pass


def validate(payload: dict) -> tuple[str, str, int, int, str | None, str, str]:
    source = str(payload.get("source", "demo"))
    if source not in {"demo", "arc", "arc2"}:
        raise PublicError("Choose Demo, ARC-AGI-2, or ARC-AGI-3.")
    game = str(payload.get("game", "ls20")).strip()
    if not GAME_ID.fullmatch(game):
        raise PublicError("Enter a valid game ID.")
    batch = str(payload.get("batch", "")).strip().lower() or None
    if batch and not BATCH_ID.fullmatch(batch):
        raise PublicError("Batch must use lowercase letters, numbers, and hyphens.")
    try:
        seed = int(payload.get("seed", 0))
        steps = int(payload.get("max_steps", 8))
    except (TypeError, ValueError):
        raise PublicError("Seed and actions must be whole numbers.") from None
    if not 0 <= seed <= 1_000_000:
        raise PublicError("Seed must be between 0 and 1,000,000.")
    if not (1 if source == "arc2" else 2) <= steps <= 100:
        raise PublicError("Choose a valid action budget.")
    model = str(payload.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-6-luna").strip().lower()
    reasoning_effort = str(payload.get("reasoning_effort") or os.environ.get("OPENAI_REASONING_EFFORT") or "medium").strip().lower()
    if model not in MODELS:
        raise PublicError("Choose GPT-6 Luna or GPT-6 Sol.")
    if reasoning_effort not in REASONING_EFFORTS:
        raise PublicError("Choose a valid reasoning effort.")
    if source == "arc2" and game not in ARC2_TASKS:
        raise PublicError("Choose a curated Luna Medium ARC-AGI-2 task.")
    if source == "arc":
        if not os.environ.get("ARC_API_KEY"):
            raise PublicError("ARC_API_KEY is not available in this terminal.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise PublicError("OpenAI credentials are not available in this terminal.")
    if source == "arc2" and not os.environ.get("OPENAI_API_KEY"):
        raise PublicError("OpenAI credentials are not available in this terminal.")
    return source, game, seed, steps, batch, model, reasoning_effort


def run_experiment(payload: dict, progress_callback=None) -> dict:
    source, game, seed, steps, batch, model, reasoning_effort = validate(payload)
    harness = str(payload.get("harness") or "v12").strip().lower()
    if harness not in {"v7", "v10", "v11", "v12"}:
        raise PublicError("Choose a valid harness version.")
    before = set((ROOT / "output" / "arc_runs").glob("*.json"))
    if source == "arc2":
        command = [str(ROOT / ".venv" / "bin" / "qahs-arc2"), "--task", game,
                   "--model", model, "--reasoning-effort", reasoning_effort]
    else:
        command = [str(ROOT / ".venv" / "bin" / "qahs-arc"), "--source", source,
                   "--game", game, "--condition", "compare", "--seed", str(seed),
                   "--max-steps", str(steps), "--model", model, "--reasoning-effort", reasoning_effort,
                   "--harness", harness]
    if batch:
        command.extend(["--batch", batch])
    timeout_value = int(os.environ.get("QAHS_EXPERIMENT_TIMEOUT", "0"))
    timeout = timeout_value if timeout_value > 0 else None
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with PROCESS_LOCK:
        ACTIVE_PROCESSES.add(process)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("QAHS_PROGRESS "):
                try:
                    if progress_callback:
                        progress_callback(json.loads(line.removeprefix("QAHS_PROGRESS ")))
                except (ValueError, TypeError):
                    pass
            else:
                stdout_lines.append(line)

    def read_stderr() -> None:
        assert process.stderr is not None
        stderr_lines.extend(process.stderr)

    readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise PublicError(f"The experiment exceeded its {timeout_value // 60}-minute limit.") from error
    finally:
        for reader in readers:
            reader.join(timeout=5)
        with PROCESS_LOCK:
            ACTIVE_PROCESSES.discard(process)
    completed = subprocess.CompletedProcess(command, process.returncode, "".join(stdout_lines), "".join(stderr_lines))
    if completed.returncode:
        raise PublicError("The experiment failed. Check the local terminal for details.")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_dashboard_runs.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    after = set((ROOT / "output" / "arc_runs").glob("*.json"))
    created = sorted(after - before, reverse=True)
    if not created:
        raise PublicError("The experiment finished without creating a result.")
    return {"ok": True, "filename": created[0].name}


def validate_batch(payload: dict) -> tuple[list[str], dict]:
    raw_games = payload.get("games", [])
    if isinstance(raw_games, str):
        raw_games = re.split(r"[\s,]+", raw_games.strip())
    if not isinstance(raw_games, list):
        raise PublicError("Enter a valid game list.")
    games = []
    for value in raw_games:
        game = str(value).strip().lower()
        if game and game not in games:
            if not GAME_ID.fullmatch(game):
                raise PublicError(f"Invalid game ID: {game[:40]}")
            games.append(game)
    if not 1 <= len(games) <= 25:
        raise PublicError("Choose between 1 and 25 games.")
    batch = str(payload.get("batch", "")).strip().lower()
    if not batch:
        raise PublicError("Give this batch an ID.")
    normalized = dict(payload)
    normalized["batch"] = batch
    normalized["game"] = games[0]
    _, _, seed, steps, _, model, reasoning_effort = validate(normalized)
    normalized.update({
        "seed": seed,
        "max_steps": steps,
        "model": model,
        "reasoning_effort": reasoning_effort,
    })
    return games, normalized


def run_batch_job(job_id: str, games: list[str], payload: dict) -> None:
    with JOB_LOCK:
        initial = dict(JOBS[job_id])
    errors = list(initial.get("errors", []))
    files = list(initial.get("files", []))
    start = int(initial.get("completed", 0))
    update_job(job_id, status="running", message="Running", last_error=None)
    for index, game in enumerate(games[start:], start=start + 1):
        attempts = dict(JOBS[job_id].get("attempts", {}))
        attempts[game] = int(attempts.get(game, 0)) + 1
        run_started = time.monotonic()
        started_at = utc_now()
        update_job(job_id, current=game, completed=index - 1, attempts=attempts, run_started_at=started_at,
                   progress={"phase": "starting", "percent": 0, "elapsed_seconds": 0, "estimated_remaining_seconds": None})

        def report(event: dict) -> None:
            elapsed = time.monotonic() - run_started
            condition_index = int(event.get("condition_index", 0))
            condition_total = max(1, int(event.get("condition_total", 3)))
            turn = int(event.get("turn", 0))
            max_steps = max(1, int(event.get("max_steps", payload.get("max_steps", 8))))
            completed_turns = condition_index * max_steps + max(0, turn - (event.get("phase") != "turn_completed"))
            total_turns = condition_total * max_steps
            fraction = min(0.99, completed_turns / total_turns)
            if event.get("phase") == "condition_completed" and condition_index + 1 == condition_total:
                fraction = 1.0
            eta = (elapsed / fraction - elapsed) if fraction > 0 else None
            progress = {**event, "percent": round(fraction * 100, 1), "elapsed_seconds": round(elapsed, 1),
                        "estimated_remaining_seconds": round(eta, 1) if eta is not None else None}
            update_job(job_id, progress=progress, message="Running")
        try:
            result = run_experiment({**payload, "game": game}, report)
            files.append(result["filename"])
        except Exception as error:
            print(f"Batch game {game} failed: {type(error).__name__}", file=sys.stderr)
            errors = [item for item in errors if item.get("game") != game]
            errors.append({"game": game, "error": "Run interrupted; resume retries this game."})
            update_job(
                job_id,
                status="paused",
                current=game,
                files=files,
                errors=errors,
                last_error="Run interrupted.",
                message="Paused. Resume retries the current game from the same seed.",
            )
            return
        errors = [item for item in errors if item.get("game") != game]
        update_job(job_id, completed=index, current=None, files=files.copy(), errors=errors.copy())
    update_job(
        job_id,
        status="completed_with_errors" if errors else "completed",
        current=None,
        completed=len(games),
        files=files,
        errors=errors,
        message="Completed", progress={"phase": "done", "percent": 100, "elapsed_seconds": 0,
                                       "estimated_remaining_seconds": 0},
    )


def start_batch(payload: dict) -> dict:
    games, normalized = validate_batch(payload)
    with JOB_LOCK:
        if any(job.get("status") in {"running", "starting"} for job in JOBS.values()):
            raise PublicError("A batch is already running.")
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "batch_id": normalized["batch"],
            "seed": int(normalized.get("seed", 0)),
            "games": games,
            "total": len(games),
            "completed": 0,
            "current": None,
            "files": [],
            "errors": [],
            "attempts": {},
            "payload": normalized,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "message": "Starting",
            "progress": {"phase": "starting", "percent": 0, "elapsed_seconds": 0,
                         "estimated_remaining_seconds": None},
        }
        save_job(JOBS[job_id])
    threading.Thread(target=run_batch_job, args=(job_id, games, normalized), daemon=True).start()
    return JOBS[job_id]


def resume_batch(job_id: str) -> dict:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise PublicError("Batch not found.")
        if job.get("status") not in {"paused", "interrupted"}:
            raise PublicError("This batch is not paused.")
        if any(item.get("status") in {"running", "starting"} for key, item in JOBS.items() if key != job_id):
            raise PublicError("Another batch is already running.")
        games = list(job["games"])
        payload = dict(job["payload"])
        job["status"] = "starting"
        job["message"] = "Resuming"
        job["updated_at"] = utc_now()
        save_job(dict(job))
        snapshot = dict(job)
    threading.Thread(target=run_batch_job, args=(job_id, games, payload), daemon=True).start()
    return snapshot


def _candidate(row: dict) -> ArcActionCandidate:
    return ArcActionCandidate(
        candidate_id=str(row.get("candidate_id", "replay")), action_type=str(row["action_type"]),
        description=str(row.get("description", "recorded action")), x=row.get("x"), y=row.get("y"),
        expected_progress=float(row.get("expected_progress", 0)), risk=float(row.get("risk", 0)),
    )


def run_counterfactual(payload: dict) -> dict:
    """Replay a saved prefix and replace one blocked QAHS decision without new LLM calls."""
    filename = str(payload.get("filename", ""))
    condition = str(payload.get("condition", "with_qahs"))
    if not RUN_FILE.fullmatch(filename) or condition not in {"with_classical", "with_qahs"}:
        raise PublicError("Choose a valid saved QAHS run.")
    try:
        turn_number = int(payload.get("turn", 0))
        horizon = min(5, max(1, int(payload.get("horizon", 1))))
    except (TypeError, ValueError):
        raise PublicError("Choose a valid turn and replay length.") from None
    path = ROOT / "output" / "arc_runs" / filename
    if not path.is_file():
        raise PublicError("The selected run is not available locally.")
    trace = json.loads(path.read_text(encoding="utf-8"))
    if trace.get("source") != "arc":
        raise PublicError("Counterfactual replay currently supports ARC-AGI-3 runs.")
    episode = trace.get(condition) or {}
    turns = list(episode.get("turns") or [])
    if not 1 <= turn_number <= len(turns):
        raise PublicError("Choose a recorded turn.")
    target = turns[turn_number - 1]
    diagnostics = target.get("search_diagnostics") or {}
    alternative_id = diagnostics.get("selector_candidate_id")
    if not alternative_id or alternative_id == target.get("selected_candidate", {}).get("candidate_id"):
        raise PublicError("That turn has no rejected selector alternative.")
    catalog = list((target.get("proposal") or {}).get("action_candidates") or [])
    alternative_row = next((item for item in catalog if item.get("candidate_id") == alternative_id), None)
    if not alternative_row:
        raise PublicError("The recorded alternative is missing its physical action.")
    environment = ArcSDKEnvironment(str(trace["game_id"]), int(trace.get("seed", 0)), ROOT / "output" / "arc_recordings")
    observation = environment.reset()
    for row in turns[:turn_number - 1]:
        observation = environment.step(_candidate(row["selected_candidate"]))
    expected = target.get("observation_before", {}).get("frame")
    if expected and [list(row) for row in observation.frame] != expected:
        raise PublicError("Replay diverged before the selected branch; no alternative was executed.")
    before_levels = observation.levels_completed
    branch_actions = [_candidate(alternative_row)]
    branch_actions.extend(_candidate(row["selected_candidate"]) for row in turns[turn_number:turn_number - 1 + horizon])
    observations = []
    for action in branch_actions:
        observation = environment.step(action)
        observations.append({"action": action.candidate_id, "frame": observation.frame, "event": observation.event,
                             "levels_completed": observation.levels_completed, "state": observation.state})
        if observation.terminal:
            break
    original_after = target.get("observation_after", {})
    original_frame = original_after.get("frame") or []
    alternative_frame = observations[0]["frame"] if observations else []
    changed_from_original = sum(a != b for row_a, row_b in zip(original_frame, alternative_frame) for a, b in zip(row_a, row_b))
    report = {
        "source_run": filename, "condition": condition, "turn": turn_number, "horizon": horizon,
        "original_candidate": target.get("selected_candidate"), "alternative_candidate": alternative_row,
        "selector_advantage": diagnostics.get("selector_advantage"),
        "original_immediate": {"event": original_after.get("event"), "levels_completed": original_after.get("levels_completed"), "frame": original_frame},
        "alternative_path": observations, "immediate_frame_difference": changed_from_original,
        "alternative_levels_delta": observation.levels_completed - before_levels,
        "created_at": utc_now(),
    }
    target_dir = ROOT / "output" / "counterfactuals"
    target_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"{Path(filename).stem}__{condition}__turn-{turn_number:03d}__counterfactual.json"
    (target_dir / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"ok": True, "filename": report_name, **report}


class Handler(BaseHTTPRequestHandler):
    server_version = "QAHSLocalRunner/1.0"

    def _origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if origin and ALLOWED_ORIGINS.fullmatch(origin) else None

    def _respond(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        if not self._origin():
            self._respond(403, {"error": "Local access only."})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._origin():
            self._respond(404, {"error": "Not found."})
            return
        match = re.fullmatch(r"/jobs/([a-f0-9]{12})", self.path)
        if not match:
            self._respond(404, {"error": "Not found."})
            return
        with JOB_LOCK:
            job = dict(JOBS.get(match.group(1), {}))
        if not job:
            self._respond(404, {"error": "Batch not found."})
            return
        self._respond(200, job)

    def do_POST(self) -> None:
        resume_match = re.fullmatch(r"/jobs/([a-f0-9]{12})/resume", self.path)
        if self.path not in {"/run", "/batch", "/counterfactual"} and not resume_match:
            self._respond(404, {"error": "Not found."})
            return
        if not self._origin():
            self._respond(404, {"error": "Not found."})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if resume_match:
                self._respond(202, resume_batch(resume_match.group(1)))
            elif self.path == "/counterfactual":
                self._respond(200, run_counterfactual(payload))
            elif self.path == "/batch":
                self._respond(202, start_batch(payload))
            else:
                self._respond(200, run_experiment(payload))
        except PublicError as error:
            self._respond(400, {"error": str(error)})
        except (json.JSONDecodeError, TypeError):
            self._respond(400, {"error": "Invalid request."})
        except Exception as error:
            print(f"Local run failed: {type(error).__name__}", file=sys.stderr)
            self._respond(500, {"error": "The local experiment could not be completed."})

    def log_message(self, message: str, *args) -> None:
        print(f"[local runner] {message % args}")


def main() -> None:
    load_jobs()
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    print(f"Local experiment service: http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
