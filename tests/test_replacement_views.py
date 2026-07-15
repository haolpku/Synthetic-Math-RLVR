from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_replacement_views import RATIOS, nested_synthetic_sets, rlvr_record


class ReplacementViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "parent_seed_id": f"p{i}",
                "source": "s" if i < 8 else "t",
                "problem_type": "Algebra",
                "real_problem": f"real {i}",
                "real_answer": str(i),
                "synthetic_problem": f"synthetic {i}",
                "synthetic_answer": str(i + 1),
                "generator_model": "g",
            }
            for i in range(12)
        ]

    def test_ratio_sets_are_nested(self) -> None:
        selected = nested_synthetic_sets(self.rows, 7)
        for smaller, larger in zip(RATIOS, RATIOS[1:]):
            self.assertTrue(selected[smaller] <= selected[larger])
        self.assertEqual(selected[0.0], set())
        self.assertEqual(selected[1.0], {row["parent_seed_id"] for row in self.rows})

    def test_record_origin_controls_prompt_and_answer(self) -> None:
        real = rlvr_record(self.rows[0], False, "r000", 0.0)
        synthetic = rlvr_record(self.rows[0], True, "r100", 1.0)
        self.assertEqual(real["prompt"][0]["content"], "real 0")
        self.assertEqual(real["reward_model"]["ground_truth"], "0")
        self.assertEqual(synthetic["prompt"][0]["content"], "synthetic 0")
        self.assertEqual(synthetic["reward_model"]["ground_truth"], "1")


if __name__ == "__main__":
    unittest.main()
