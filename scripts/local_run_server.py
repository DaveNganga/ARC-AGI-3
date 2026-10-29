from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8765
GAME_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ALLOWED_ORIGINS = re.compile(
    r"^https?://(?:localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}):\d+$"
)


# One run at a time: overlapping runs share API quotas and output folders.
RUN_LOCK = threading.Lock()


class PublicError(ValueError):
    pass


class BusyError(PublicError):
    pass


def validate(payload: dict) -> tuple[str, str, int, int]:
    source = str(payload.get("source", "demo"))
    if source not in {"demo", "arc"}:
        raise PublicError("Choose Demo or ARC.")
    game = str(payload.get("game", "ls20")).strip()
    if not GAME_ID.fullmatch(game):
        raise PublicError("Enter a valid game ID.")
    try:
        seed = int(payload.get("seed", 0))
        steps = int(payload.get("max_steps", 8))
    except (TypeError, ValueError):
        raise PublicError("Seed and actions must be whole numbers.") from None
    if not 0 <= seed <= 1_000_000:
        raise PublicError("Seed must be between 0 and 1,000,000.")
    if not 2 <= steps <= 30:
        raise PublicError("Actions must be between 2 and 30.")
    if source == "arc":
        if not os.environ.get("ARC_API_KEY"):
            raise PublicError("ARC_API_KEY is not available in this terminal.")
        if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"):
            raise PublicError("OpenAI credentials are not available in this terminal.")
    return source, game, seed, steps


def run_experiment(payload: dict) -> dict:
    source, game, seed, steps = validate(payload)
    if not RUN_LOCK.acquire(blocking=False):
        raise BusyError("A run is already in progress. Wait for it to finish.")
    try:
        return _run_locked(source, game, seed, steps)
    finally:
        RUN_LOCK.release()


def _run_locked(source: str, game: str, seed: int, steps: int) -> dict:
    before = set((ROOT / "output" / "arc_runs").glob("*.json"))
    command = [
        str(ROOT / ".venv" / "bin" / "qahs-arc"),
        "--source", source,
        "--game", game,
        "--condition", "compare",
        "--seed", str(seed),
        "--max-steps", str(steps),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    if completed.returncode:
        # Details stay in this local terminal; the browser only receives a generic message.
        tail = "\n".join(completed.stderr.splitlines()[-40:])
        print(f"qahs-arc exited with code {completed.returncode}:\n{tail}", file=sys.stderr, flush=True)
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
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/run" or not self._origin():
            self._respond(404, {"error": "Not found."})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._respond(200, run_experiment(payload))
        except BusyError as error:
            self._respond(409, {"error": str(error)})
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
    print(f"Local experiment service: http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
