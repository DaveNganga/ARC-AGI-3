from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import local_run_server as runner


class LocalRunSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_directory = runner.SESSION_DIR
        runner.SESSION_DIR = Path(self.temporary.name)
        runner.JOBS.clear()

    def tearDown(self) -> None:
        runner.JOBS.clear()
        runner.SESSION_DIR = self.previous_directory
        self.temporary.cleanup()

    def job(self, status: str = "running") -> dict:
        return {
            "job_id": "abc123def456",
            "status": status,
            "batch_id": "test-batch",
            "seed": 3,
            "games": ["ar25", "vc33"],
            "total": 2,
            "completed": 0,
            "current": "ar25",
            "files": [],
            "errors": [],
            "attempts": {},
            "payload": {"source": "arc", "batch": "test-batch", "seed": 3, "max_steps": 8},
        }

    def test_running_job_becomes_resumable_after_server_restart(self) -> None:
        runner.save_job(self.job())
        runner.load_jobs()
        restored = runner.JOBS["abc123def456"]
        self.assertEqual(restored["status"], "paused")
        self.assertEqual(restored["completed"], 0)
        self.assertEqual(restored["current"], "ar25")

    def test_batch_checkpoints_each_completed_game(self) -> None:
        runner.JOBS["abc123def456"] = self.job()
        with patch.object(runner, "run_experiment", side_effect=[
            {"filename": "ar25.json"},
            {"filename": "vc33.json"},
        ]):
            runner.run_batch_job("abc123def456", ["ar25", "vc33"], self.job()["payload"])
        saved = json.loads(runner.session_path("abc123def456").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["completed"], 2)
        self.assertEqual(saved["files"], ["ar25.json", "vc33.json"])

    def test_interrupted_game_is_paused_without_losing_completed_games(self) -> None:
        job = self.job()
        job.update({"completed": 1, "current": "vc33", "files": ["ar25.json"]})
        runner.JOBS["abc123def456"] = job
        with patch.object(runner, "run_experiment", side_effect=TimeoutError("offline")):
            runner.run_batch_job("abc123def456", ["ar25", "vc33"], job["payload"])
        saved = runner.JOBS["abc123def456"]
        self.assertEqual(saved["status"], "paused")
        self.assertEqual(saved["completed"], 1)
        self.assertEqual(saved["current"], "vc33")
        self.assertEqual(saved["files"], ["ar25.json"])

    @patch.dict("os.environ", {"ARC_API_KEY": "arc", "OPENAI_API_KEY": "openai"}, clear=True)
    def test_batch_normalizes_and_preserves_model_configuration(self) -> None:
        games, payload = runner.validate_batch({
            "source": "arc",
            "games": "r11l, vc33",
            "batch": "sol-baseline",
            "seed": "1",
            "max_steps": "40",
            "model": "gpt-6-sol",
            "reasoning_effort": "low",
        })
        self.assertEqual(games, ["r11l", "vc33"])
        self.assertEqual(payload["model"], "gpt-6-sol")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["seed"], 1)
        self.assertEqual(payload["max_steps"], 40)

    @patch.dict("os.environ", {"ARC_API_KEY": "arc", "OPENAI_API_KEY": "openai"}, clear=True)
    def test_rejects_unapproved_model_and_effort(self) -> None:
        base = {"source": "arc", "game": "r11l", "batch": "test", "seed": 0, "max_steps": 8}
        with self.assertRaisesRegex(runner.PublicError, "Luna or GPT-6 Sol"):
            runner.validate({**base, "model": "unknown"})
        with self.assertRaisesRegex(runner.PublicError, "reasoning effort"):
            runner.validate({**base, "model": "gpt-6-luna", "reasoning_effort": "extreme"})

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai"}, clear=True)
    def test_arc2_batch_uses_curated_tasks_without_arc_credentials(self) -> None:
        games, payload = runner.validate_batch({
            "source": "arc2", "games": "136b0064, 1818057f", "batch": "arc2-night",
            "seed": 0, "max_steps": 1, "model": "gpt-6-luna", "reasoning_effort": "medium",
        })
        self.assertEqual(games, ["136b0064", "1818057f"])
        self.assertEqual(payload["max_steps"], 1)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai"}, clear=True)
    def test_arc2_rejects_tasks_outside_curated_set(self) -> None:
        with self.assertRaisesRegex(runner.PublicError, "curated"):
            runner.validate({"source": "arc2", "game": "deadbeef", "seed": 0, "max_steps": 1,
                             "model": "gpt-6-luna", "reasoning_effort": "medium"})


if __name__ == "__main__":
    unittest.main()
