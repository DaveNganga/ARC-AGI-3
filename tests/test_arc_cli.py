import sys
import unittest
from unittest import mock

from qahs import arc_cli


class ArcCliTests(unittest.TestCase):
    def test_demo_runs_for_comparison_and_single_condition(self):
        for condition in ("compare", "llm_only"):
            with self.subTest(condition=condition):
                saved = []
                argv = ["qahs-arc", "--source", "demo", "--condition", condition, "--batch", "b-01", "--max-steps", "2"]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(arc_cli, "save_arc_trace", lambda trace, directory: saved.append(trace) or "t.json"), \
                        mock.patch("builtins.print"):
                    arc_cli.main()
                self.assertEqual(saved[0].batch_id, "b-01")

    def test_single_condition_is_refused_for_v10_to_v12_on_arc(self):
        for harness in ("v10", "v11", "v12"):
            with self.subTest(harness=harness):
                argv = ["qahs-arc", "--source", "arc", "--condition", "llm_only", "--harness", harness]
                with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"), \
                        self.assertRaises(SystemExit):
                    arc_cli.main()


if __name__ == "__main__":
    unittest.main()
