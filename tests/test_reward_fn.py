from __future__ import annotations

import unittest

from training.reward_fn import compute_score


class RewardFunctionTest(unittest.TestCase):
    def test_scalar_self_match(self) -> None:
        self.assertEqual(compute_score("x", r"\\boxed{137/2}", "137/2"), 1.0)

    def test_set_self_match_falls_back_to_normalized_text(self) -> None:
        answer = r"\{24t\mid t\in\mathbb{Z}\}"
        self.assertEqual(compute_score("x", rf"\\boxed{{{answer}}}", answer), 1.0)

    def test_wrong_answer_is_zero(self) -> None:
        self.assertEqual(compute_score("x", r"\\boxed{3}", "4"), 0.0)


if __name__ == "__main__":
    unittest.main()
