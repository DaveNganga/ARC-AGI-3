import tempfile
import unittest
from pathlib import Path

from qahs.arc_agent import run_arc_comparison, run_arc_episode
from qahs.arc_environment import DemoArcEnvironment, TransientEnvironmentError
from qahs.arc_reasoner import DemoArcReasoner
from qahs.session import (
    ConnectivityTimeout,
    RunJournal,
    classify_failure,
    new_journal,
    open_journal,
    unfinished_runs,
    wait_for_connectivity,
)


def no_sleep(_seconds):
    pass


class CountingReasoner(DemoArcReasoner):
    """Counts real model-equivalent calls; can simulate a process kill on a given proposal call."""

    calls = 0

    def __init__(self, interrupt_on_call=None):
        super().__init__()
        self.interrupt_on_call = interrupt_on_call

    def propose(self, observation, history):
        CountingReasoner.calls += 1
        if CountingReasoner.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        return super().propose(observation, history)

    def reflect(self, *args):
        CountingReasoner.calls += 1
        return super().reflect(*args)


class ShiftedDemo(DemoArcEnvironment):
    """A replacement environment whose frames differ, so replay cannot match the journal."""

    def _frame(self):
        frame = [list(row) for row in super()._frame()]
        frame[0][0] = 3
        return tuple(tuple(row) for row in frame)


def actions(trace):
    return [
        [turn.selected_candidate.candidate_id if hasattr(turn, "selected_candidate") else turn["selected_candidate"]["candidate_id"]
         for turn in episode.turns]
        for episode in (trace.without_qahs, trace.with_classical, trace.with_qahs)
    ]


def compare(journal=None, reasoner=CountingReasoner, **options):
    return run_arc_comparison(
        lambda: DemoArcEnvironment(0), reasoner, "demo", DemoArcEnvironment.game_id, 0, 6,
        journal=journal, sleep=no_sleep, **options,
    )


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        CountingReasoner.calls = 0
        DemoArcEnvironment.fault_at_step = None
        DemoArcEnvironment._steps_taken = 0
        self.baseline = compare()
        self.baseline_calls = CountingReasoner.calls
        CountingReasoner.calls = 0
        DemoArcEnvironment._steps_taken = 0

    def tearDown(self):
        DemoArcEnvironment.fault_at_step = None
        self.tmp.cleanup()

    def test_dropped_connection_is_recovered_by_replay(self):
        DemoArcEnvironment.fault_at_step = 2
        journal = new_journal("demo", self.dir)
        trace = compare(journal)
        self.assertEqual(actions(trace), actions(self.baseline))
        first = trace.without_qahs
        self.assertEqual(first.stop_reason, "success")
        self.assertEqual(len(first.connection_events), 1)
        self.assertEqual(first.connection_events[0]["stage"], "step")
        self.assertEqual(first.replayed_actions, 1)
        # The proposal for the interrupted turn was reused, not requested again.
        self.assertEqual(CountingReasoner.calls, self.baseline_calls)
        events = [item["event"] for item in journal.load()]
        self.assertIn("connection_lost", events)
        self.assertIn("resumed", events)

    def test_resume_after_process_kill_matches_uninterrupted_run(self):
        journal = new_journal("demo", self.dir)
        journal.append("run_started", source="demo")
        with self.assertRaises(KeyboardInterrupt):
            # Call 8 is the second proposal of the classical episode; LLM-only has already finished.
            compare(journal, reasoner=lambda: CountingReasoner(interrupt_on_call=8))
        calls_before_kill = CountingReasoner.calls - 1  # the interrupted call returned nothing
        self.assertEqual(unfinished_runs(self.dir), [journal.run_id])

        CountingReasoner.calls = 0
        DemoArcEnvironment._steps_taken = 0
        resumed = compare(open_journal("latest", self.dir))
        self.assertEqual(actions(resumed), actions(self.baseline))
        self.assertEqual([resumed.without_qahs.success, resumed.with_classical.success, resumed.with_qahs.success], [True] * 3)
        self.assertFalse(resumed.without_qahs.resumed)  # finished before the kill: rebuilt from the journal
        self.assertTrue(resumed.with_classical.resumed)
        self.assertEqual(resumed.with_classical.replayed_actions, 1)
        # Every model response from before the kill was reused.
        self.assertEqual(calls_before_kill + CountingReasoner.calls, self.baseline_calls)

    def test_replay_divergence_stops_loudly(self):
        DemoArcEnvironment.fault_at_step = 2
        environments = iter([DemoArcEnvironment(0), ShiftedDemo(0)])
        trace = run_arc_episode(
            next(environments), CountingReasoner(), "llm_only", "demo", max_steps=6,
            environment_factory=lambda: next(environments), sleep=no_sleep,
        )
        self.assertEqual(trace.stop_reason, "error")
        self.assertIn("ReplayDivergence", trace.error)

    def test_fatal_errors_are_not_retried(self):
        class Broken(DemoArcEnvironment):
            def step(self, candidate):
                raise ValueError("invalid action")

        trace = run_arc_episode(Broken(0), CountingReasoner(), "llm_only", "demo", sleep=no_sleep)
        self.assertEqual(trace.stop_reason, "error")
        self.assertEqual(trace.connection_events, [])

    def test_too_many_reconnects_end_the_episode(self):
        class AlwaysDown(DemoArcEnvironment):
            def step(self, candidate):
                raise TransientEnvironmentError("down")

        trace = run_arc_episode(
            AlwaysDown(0), CountingReasoner(), "llm_only", "demo",
            environment_factory=lambda: AlwaysDown(0), max_reconnects=2, sleep=no_sleep,
        )
        self.assertEqual(trace.stop_reason, "error")
        self.assertEqual(len(trace.connection_events), 2)


class JournalTests(unittest.TestCase):
    def test_truncated_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = RunJournal(Path(tmp) / "run" / "journal.jsonl")
            journal.append("run_started", seed=1)
            with journal.path.open("a") as handle:
                handle.write('{"event":"turn_comm')
            self.assertEqual([item["event"] for item in journal.load()], ["run_started"])
            self.assertFalse(journal.finished)

    def test_classify_failure(self):
        class HTTPError(Exception):
            def __init__(self, status):
                self.status_code = status

        self.assertEqual(classify_failure(TransientEnvironmentError("x")), "transient")
        self.assertEqual(classify_failure(TimeoutError()), "transient")
        self.assertEqual(classify_failure(HTTPError(503)), "transient")
        self.assertEqual(classify_failure(HTTPError(401)), "fatal")
        self.assertEqual(classify_failure(type("APIConnectionError", (Exception,), {})()), "transient")
        self.assertEqual(classify_failure(ValueError("bad")), "fatal")

    def test_wait_for_connectivity(self):
        answers = iter([False, False, True])
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        waited = wait_for_connectivity(lambda: next(answers), sleep=sleep, clock=lambda: now[0])
        self.assertEqual(waited, 6.0)  # backoff 2 s, then 4 s
        with self.assertRaises(ConnectivityTimeout):
            wait_for_connectivity(lambda: False, max_wait=10, sleep=sleep, clock=lambda: now[0])

    def test_open_journal_rejects_bad_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                open_journal("../etc", tmp)
            with self.assertRaises(FileNotFoundError):
                open_journal("latest", tmp)


if __name__ == "__main__":
    unittest.main()
