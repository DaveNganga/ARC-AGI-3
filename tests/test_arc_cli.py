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


if __name__ == "__main__":
    unittest.main()
