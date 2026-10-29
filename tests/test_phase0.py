import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qahs.arc_agent import run_arc_comparison, run_arc_episode
from qahs.arc_environment import DemoArcEnvironment
from qahs.arc_models import ArcObservation
from qahs.arc_reasoner import DemoArcReasoner, OpenAIArcReasoner
from qahs.models import ActionScore
from qahs.retry import call_with_retries
from qahs.search import GreedyBeliefSearch


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("export_dashboard_runs", ROOT / "scripts" / "export_dashboard_runs.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

VALID_PROPOSAL = {
    "summary": "test",
    "hypotheses": [{"id": 0, "statement": "h", "confidence": 0.5, "predictions": [{"candidate_id": "a", "outcome": "moves"}]}],
    "action_candidates": [{
        "candidate_id": "a", "action_type": "ACTION1", "description": "up",
        "x": None, "y": None, "expected_progress": 0.1, "risk": 0.0,
    }],
    "recommended_candidate_id": "a",
    "recommendation_reason": "only option",
}


class FlakyCaller:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def call_json(self, prompt, schema, name):
        return self.payloads.pop(0)


class FailingEnvironment(DemoArcEnvironment):
    def step(self, candidate):
        raise RuntimeError("boom")


class StuckEnvironment(DemoArcEnvironment):
    def step(self, candidate):
        return self._observation("no_visible_change")


class ExportTests(unittest.TestCase):
    def test_missing_source_keeps_published_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "runs"
            target.mkdir()
            (target / "old.json").write_text("{}")
            (target / "index.json").write_text('[{"filename": "old.json"}]')
            self.assertEqual(export.export_runs(Path(tmp) / "missing", target), 0)
            self.assertTrue((target / "old.json").exists())
            self.assertEqual(json.loads((target / "index.json").read_text()), [{"filename": "old.json"}])

    def test_export_merges_with_existing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "src", Path(tmp) / "runs"
            source.mkdir()
            target.mkdir()
            (target / "20260101T000000Z_ls20_comparison.json").write_text("{}")
            (source / "20260102T000000Z_ar25_comparison.json").write_text('{"error": "secret detail"}')
            self.assertEqual(export.export_runs(source, target), 1)
            names = [item["filename"] for item in json.loads((target / "index.json").read_text())]
            self.assertEqual(names, ["20260102T000000Z_ar25_comparison.json", "20260101T000000Z_ls20_comparison.json"])
            published = json.loads((target / "20260102T000000Z_ar25_comparison.json").read_text())
            self.assertNotIn("secret", published["error"])

    def test_clean_redacts_error_details(self):
        cleaned = export.clean({"error": "KeyError: x", "error_traceback": "Traceback ...", "stop_reason": "error"})
        self.assertNotIn("KeyError", cleaned["error"])
        self.assertNotIn("Traceback", cleaned["error_traceback"])
        self.assertEqual(cleaned["stop_reason"], "error")


class RetryTests(unittest.TestCase):
    def test_retries_until_success(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("transient")
            return "ok"

        self.assertEqual(call_with_retries(flaky, sleep=lambda _: None), ("ok", 2))

    def test_raises_after_last_attempt(self):
        def broken():
            raise ConnectionError("down")

        with self.assertRaises(ConnectionError):
            call_with_retries(broken, attempts=3, sleep=lambda _: None)

    def test_reasoner_retries_unparseable_proposal(self):
        reasoner = OpenAIArcReasoner(FlakyCaller([{"action_candidates": []}, VALID_PROPOSAL]), retry_delay=0)
        observation = ArcObservation("ls20", "NOT_FINISHED", 0, 1, ("ACTION1",), ((0,),))
        proposal = reasoner.propose(observation, [])
        self.assertEqual(proposal.recommended_candidate_id, "a")
        self.assertEqual(reasoner.last_call_count, 2)
        self.assertEqual(reasoner.last_retry_count, 1)


class EpisodeRecordTests(unittest.TestCase):
    def test_error_details_are_kept_in_private_trace(self):
        trace = run_arc_episode(FailingEnvironment(), DemoArcReasoner(), "qahs_classical", "demo")
        self.assertEqual(trace.stop_reason, "error")
        self.assertIn("boom", trace.error)
        self.assertIn("RuntimeError", trace.error_traceback)

    def test_budget_exhaustion_is_recorded(self):
        trace = run_arc_episode(StuckEnvironment(), DemoArcReasoner(), "llm_only", "demo", max_steps=3)
        self.assertEqual(len(trace.turns), 3)
        self.assertEqual(trace.stop_reason, "budget_exhausted")

    def test_condition_order_rotates_without_changing_demo_results(self):
        traces = [
            run_arc_comparison(lambda: DemoArcEnvironment(seed), DemoArcReasoner, "demo", DemoArcEnvironment.game_id, seed, 6)
            for seed in (0, 1)
        ]
        self.assertNotEqual(traces[0].condition_order, traces[1].condition_order)
        for trace in traces:
            self.assertEqual(sorted(trace.condition_order), ["llm_only", "qahs_classical", "qahs_qaoa"])
            self.assertEqual(trace.harness_version, "2")
            self.assertEqual(trace.without_qahs.stop_reason, "success")
            self.assertEqual(
                [len(trace.without_qahs.turns), len(trace.with_classical.turns), len(trace.with_qahs.turns)],
                [3, 2, 2],
            )


class GreedyTests(unittest.TestCase):
    def test_greedy_accepts_string_actions(self):
        class Encoding:
            hypotheses = (type("H", (), {"id": 0})(),)

            def encode(self, hypothesis_id, action):
                return {"a": 0, "b": 1}[action]

            def decode(self, index):
                return 0, ("a", "b")[index]

        scores = [ActionScore("a", 0.1, 0.0, 0.2), ActionScore("b", 0.3, 0.0, 0.5)]
        result = GreedyBeliefSearch().select(Encoding(), scores, np.array([0.0, -1.0]), np.array([True, True]))
        self.assertEqual(result.action, "b")


if __name__ == "__main__":
    unittest.main()
