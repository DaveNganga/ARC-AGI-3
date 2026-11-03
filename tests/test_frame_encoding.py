import json
import unittest

from qahs.arc_reasoner import compact_history, frame_summary
from qahs.frame_encoding import encode_frame, find_objects, frame_diff, hex_grid, objects_summary


GRID = (
    (0, 0, 0, 0, 0, 0),
    (0, 3, 3, 0, 0, 0),
    (0, 3, 3, 0, 7, 0),
    (0, 0, 0, 0, 7, 7),
    (0, 0, 0, 0, 0, 0),
)


class FrameEncodingTests(unittest.TestCase):
    def test_hex_grid_is_lossless(self):
        lines = hex_grid(GRID).splitlines()[2:]
        decoded = tuple(tuple(int(char, 16) for char in line) for line in lines)
        self.assertEqual(decoded, GRID)

    def test_objects_are_connected_components(self):
        objects = find_objects(GRID)
        self.assertEqual(len(objects), 2)
        square, corner = objects
        self.assertEqual(square, {"c": 3, "bbox": [1, 1, 2, 2], "n": 4, "shape": "rect"})
        self.assertEqual(corner["bbox"], [4, 2, 5, 3])
        self.assertEqual(corner["shape"], "#./##")
        self.assertIn('"c":7', objects_summary(GRID))

    def test_runs_format_matches_frame_summary(self):
        self.assertEqual(encode_frame(GRID, "runs"), frame_summary(GRID))
        self.assertIn("y2: x1-2=3, x4=7\ny3: x4-5=7", frame_summary(GRID))

    def test_diff_lists_only_changed_cells(self):
        after = tuple(tuple(9 if (x, y) == (4, 2) else cell for x, cell in enumerate(row)) for y, row in enumerate(GRID))
        diff = frame_diff(GRID, after)
        self.assertIn("1 cells changed", diff)
        self.assertIn("4,2:7->9", diff)
        self.assertEqual(frame_diff(GRID, GRID), "no cells changed")
        self.assertIn("shape changed", frame_diff(GRID, GRID[:3]))

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_frame(GRID, "image")

    def test_compact_history_keeps_only_needed_fields(self):
        history = [{"turn": 1, "action_signature": "ACTION1:None:None", "changed_cells": 3, "belief_ledger": [{"id": 0}]}]
        rows = json.loads(compact_history(history, 8))
        self.assertEqual(rows[0]["changed_cells"], 3)
        self.assertNotIn("belief_ledger", rows[0])
        self.assertEqual(compact_history([], 8), "none")


if __name__ == "__main__":
    unittest.main()
