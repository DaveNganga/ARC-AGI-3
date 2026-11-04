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
        with self.assertRaisesRegex(runner.PublicError, "GPT-6 Luna, GPT-6 Sol"):
            runner.validate({**base, "model": "unknown"})
        with self.assertRaisesRegex(runner.PublicError, "reasoning effort"):
            runner.validate({**base, "model": "gpt-6-luna", "reasoning_effort": "extreme"})

    @patch.dict("os.environ", {"ARC_API_KEY": "arc", "ANTHROPIC_API_KEY": "anthropic"}, clear=True)
    def test_claude_model_needs_anthropic_not_openai_credentials(self) -> None:
        base = {"source": "arc", "game": "lp85", "batch": "claude-haiku-v12-final-lp85", "seed": 6, "max_steps": 40,
                "reasoning_effort": "medium"}
        self.assertEqual(runner.validate({**base, "model": "claude-haiku-5-5"})[5], "claude-haiku-5-5")
        with self.assertRaisesRegex(runner.PublicError, "OpenAI credentials"):
            runner.validate({**base, "model": "gpt-6-luna"})
        with self.assertRaisesRegex(runner.PublicError, "reasoning effort none"):
            runner.validate({**base, "model": "claude-haiku-5-5", "reasoning_effort": "none"})
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(runner.PublicError, "ANTHROPIC_API_KEY"):
                runner.validate({**base, "model": "claude-haiku-5-5"})

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai", "ANTHROPIC_API_KEY": "anthropic"}, clear=True)
    def test_claude_models_are_refused_for_arc2(self) -> None:
        with self.assertRaisesRegex(runner.PublicError, "ARC-AGI-3 runs only"):
            runner.validate({"source": "arc2", "game": "136b0064", "seed": 0, "max_steps": 1,
                             "model": "claude-haiku-5-5", "reasoning_effort": "medium"})

    @patch.dict("os.environ", {"ARC_API_KEY": "arc", "OPENAI_API_KEY": "openai", "ANTHROPIC_API_KEY": "anthropic"}, clear=True)
    def test_command_adds_anthropic_provider_only_for_claude(self) -> None:
        class Captured(Exception):
            pass

        commands = []

        def fake_popen(command, **kwargs):
            commands.append(command)
            raise Captured

        base = {"source": "arc", "game": "lp85", "batch": "b", "seed": 8, "max_steps": 40,
                "reasoning_effort": "medium", "harness": "v12"}
        for model in ("claude-haiku-5-5", "gpt-6-luna"):
            with patch.object(runner.subprocess, "Popen", fake_popen), self.assertRaises(Captured):
                runner.run_experiment({**base, "model": model})
        claude, gpt = commands
        self.assertEqual(claude[claude.index("--provider") + 1], "anthropic")
        self.assertEqual(claude[claude.index("--model") + 1], "claude-haiku-5-5")
        self.assertNotIn("--provider", gpt)
        self.assertEqual(gpt[gpt.index("--harness") + 1], "v12")

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
